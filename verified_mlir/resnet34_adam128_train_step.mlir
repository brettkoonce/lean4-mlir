module @m {
  func.func @resnet34_adam128_train_step(%x: tensor<128x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x1x1xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x1x1xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x1x1xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x3x3xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b1W1m: tensor<64x64x3x3xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b2W1m: tensor<64x64x3x3xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %d2W1m: tensor<128x64x3x3xf32>, %d2g1m: tensor<128xf32>, %d2bt1m: tensor<128xf32>, %d2W2m: tensor<128x128x3x3xf32>, %d2g2m: tensor<128xf32>, %d2bt2m: tensor<128xf32>, %d2Wpm: tensor<128x64x1x1xf32>, %d2gpm: tensor<128xf32>, %d2btpm: tensor<128xf32>, %s2b0W1m: tensor<128x128x3x3xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b1W1m: tensor<128x128x3x3xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b2W1m: tensor<128x128x3x3xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %d3W1m: tensor<256x128x3x3xf32>, %d3g1m: tensor<256xf32>, %d3bt1m: tensor<256xf32>, %d3W2m: tensor<256x256x3x3xf32>, %d3g2m: tensor<256xf32>, %d3bt2m: tensor<256xf32>, %d3Wpm: tensor<256x128x1x1xf32>, %d3gpm: tensor<256xf32>, %d3btpm: tensor<256xf32>, %s3b0W1m: tensor<256x256x3x3xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b1W1m: tensor<256x256x3x3xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b2W1m: tensor<256x256x3x3xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b3W1m: tensor<256x256x3x3xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b4W1m: tensor<256x256x3x3xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %d4W1m: tensor<512x256x3x3xf32>, %d4g1m: tensor<512xf32>, %d4bt1m: tensor<512xf32>, %d4W2m: tensor<512x512x3x3xf32>, %d4g2m: tensor<512xf32>, %d4bt2m: tensor<512xf32>, %d4Wpm: tensor<512x256x1x1xf32>, %d4gpm: tensor<512xf32>, %d4btpm: tensor<512xf32>, %s4b0W1m: tensor<512x512x3x3xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b1W1m: tensor<512x512x3x3xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %Wdm: tensor<512x10xf32>, %bdm: tensor<10xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x3x3xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b1W1v: tensor<64x64x3x3xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b2W1v: tensor<64x64x3x3xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %d2W1v: tensor<128x64x3x3xf32>, %d2g1v: tensor<128xf32>, %d2bt1v: tensor<128xf32>, %d2W2v: tensor<128x128x3x3xf32>, %d2g2v: tensor<128xf32>, %d2bt2v: tensor<128xf32>, %d2Wpv: tensor<128x64x1x1xf32>, %d2gpv: tensor<128xf32>, %d2btpv: tensor<128xf32>, %s2b0W1v: tensor<128x128x3x3xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b1W1v: tensor<128x128x3x3xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b2W1v: tensor<128x128x3x3xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %d3W1v: tensor<256x128x3x3xf32>, %d3g1v: tensor<256xf32>, %d3bt1v: tensor<256xf32>, %d3W2v: tensor<256x256x3x3xf32>, %d3g2v: tensor<256xf32>, %d3bt2v: tensor<256xf32>, %d3Wpv: tensor<256x128x1x1xf32>, %d3gpv: tensor<256xf32>, %d3btpv: tensor<256xf32>, %s3b0W1v: tensor<256x256x3x3xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b1W1v: tensor<256x256x3x3xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b2W1v: tensor<256x256x3x3xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b3W1v: tensor<256x256x3x3xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b4W1v: tensor<256x256x3x3xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %d4W1v: tensor<512x256x3x3xf32>, %d4g1v: tensor<512xf32>, %d4bt1v: tensor<512xf32>, %d4W2v: tensor<512x512x3x3xf32>, %d4g2v: tensor<512xf32>, %d4bt2v: tensor<512xf32>, %d4Wpv: tensor<512x256x1x1xf32>, %d4gpv: tensor<512xf32>, %d4btpv: tensor<512xf32>, %s4b0W1v: tensor<512x512x3x3xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b1W1v: tensor<512x512x3x3xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %Wdv: tensor<512x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %d2n1mui: tensor<128xf32>, %d2n1vari: tensor<128xf32>, %d2n2mui: tensor<128xf32>, %d2n2vari: tensor<128xf32>, %d2npmui: tensor<128xf32>, %d2npvari: tensor<128xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %d3n1mui: tensor<256xf32>, %d3n1vari: tensor<256xf32>, %d3n2mui: tensor<256xf32>, %d3n2vari: tensor<256xf32>, %d3npmui: tensor<256xf32>, %d3npvari: tensor<256xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %d4n1mui: tensor<512xf32>, %d4n1vari: tensor<512xf32>, %d4n2mui: tensor<512xf32>, %d4n2vari: tensor<512xf32>, %d4npmui: tensor<512xf32>, %d4npvari: tensor<512xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %onehot: tensor<128x10xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) {
    // ── ResNet-34 batch-BN AdamW train step: every op is pretty(verified AST node) except the constants, the input passthroughs and the marked report-only %loss ──
    // The conv biases are not in the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb64 = stablehlo.constant dense<0.0> : tensor<64xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb512 = stablehlo.constant dense<0.0> : tensor<512xf32>
    %v0 = stablehlo.reshape %x : (tensor<128x150528xf32>) -> tensor<128x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<128x64x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<128x64x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<128x64x112x112xf32>) -> tensor<128x802816xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<1605632.0> : tensor<128x64x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-05> : tensor<128x64x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<128x64x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<128x64x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<128x64x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<128x64x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<128x64x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<128x64x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<128x64x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<128x64x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<128x64x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<128x64x112x112xf32>) -> tensor<128x802816xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<128x64x112x112xf32>
    %v27 = stablehlo.maximum %v25, %v26 : tensor<128x64x112x112xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<128x64x112x112xf32>) -> tensor<128x802816xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v30 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v31 = "stablehlo.reduce_window"(%v29, %v30) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<128x64x56x56xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v34 = stablehlo.convolution(%v33, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v35 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v36 = stablehlo.add %v34, %v35 : tensor<128x64x56x56xf32>
    %v37 = stablehlo.reshape %v36 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v39 = stablehlo.constant dense<0.0> : tensor<f32>
    %v40 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v41 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v42 = stablehlo.reduce(%v38 init: %v39) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v43 = stablehlo.broadcast_in_dim %v42, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v44 = stablehlo.divide %v43, %v40 : tensor<128x64x56x56xf32>
    %v45 = stablehlo.subtract %v38, %v44 : tensor<128x64x56x56xf32>
    %v46 = stablehlo.multiply %v45, %v45 : tensor<128x64x56x56xf32>
    %v47 = stablehlo.reduce(%v46 init: %v39) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v48 = stablehlo.broadcast_in_dim %v47, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v49 = stablehlo.divide %v48, %v40 : tensor<128x64x56x56xf32>
    %v50 = stablehlo.add %v49, %v41 : tensor<128x64x56x56xf32>
    %v51 = stablehlo.rsqrt %v50 : tensor<128x64x56x56xf32>
    %v52 = stablehlo.multiply %v45, %v51 : tensor<128x64x56x56xf32>
    %v53 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v54 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v55 = stablehlo.multiply %v52, %v53 : tensor<128x64x56x56xf32>
    %v56 = stablehlo.add %v55, %v54 : tensor<128x64x56x56xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v60 = stablehlo.maximum %v58, %v59 : tensor<128x64x56x56xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v63 = stablehlo.convolution(%v62, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v64 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<128x64x56x56xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v70 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v71 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v72 = stablehlo.broadcast_in_dim %v71, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v73 = stablehlo.divide %v72, %v69 : tensor<128x64x56x56xf32>
    %v74 = stablehlo.subtract %v67, %v73 : tensor<128x64x56x56xf32>
    %v75 = stablehlo.multiply %v74, %v74 : tensor<128x64x56x56xf32>
    %v76 = stablehlo.reduce(%v75 init: %v68) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v77 = stablehlo.broadcast_in_dim %v76, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v78 = stablehlo.divide %v77, %v69 : tensor<128x64x56x56xf32>
    %v79 = stablehlo.add %v78, %v70 : tensor<128x64x56x56xf32>
    %v80 = stablehlo.rsqrt %v79 : tensor<128x64x56x56xf32>
    %v81 = stablehlo.multiply %v74, %v80 : tensor<128x64x56x56xf32>
    %v82 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v83 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v84 = stablehlo.multiply %v81, %v82 : tensor<128x64x56x56xf32>
    %v85 = stablehlo.add %v84, %v83 : tensor<128x64x56x56xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v88 = stablehlo.reshape %v32 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v89 = stablehlo.add %v87, %v88 : tensor<128x64x56x56xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v93 = stablehlo.maximum %v91, %v92 : tensor<128x64x56x56xf32>
    %v94 = stablehlo.reshape %v93 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v96 = stablehlo.convolution(%v95, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v97 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v98 = stablehlo.add %v96, %v97 : tensor<128x64x56x56xf32>
    %v99 = stablehlo.reshape %v98 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v102 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v103 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v104 = stablehlo.reduce(%v100 init: %v101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v105 = stablehlo.broadcast_in_dim %v104, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v106 = stablehlo.divide %v105, %v102 : tensor<128x64x56x56xf32>
    %v107 = stablehlo.subtract %v100, %v106 : tensor<128x64x56x56xf32>
    %v108 = stablehlo.multiply %v107, %v107 : tensor<128x64x56x56xf32>
    %v109 = stablehlo.reduce(%v108 init: %v101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v110 = stablehlo.broadcast_in_dim %v109, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v111 = stablehlo.divide %v110, %v102 : tensor<128x64x56x56xf32>
    %v112 = stablehlo.add %v111, %v103 : tensor<128x64x56x56xf32>
    %v113 = stablehlo.rsqrt %v112 : tensor<128x64x56x56xf32>
    %v114 = stablehlo.multiply %v107, %v113 : tensor<128x64x56x56xf32>
    %v115 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v116 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v117 = stablehlo.multiply %v114, %v115 : tensor<128x64x56x56xf32>
    %v118 = stablehlo.add %v117, %v116 : tensor<128x64x56x56xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v121 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v122 = stablehlo.maximum %v120, %v121 : tensor<128x64x56x56xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v125 = stablehlo.convolution(%v124, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v126 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v127 = stablehlo.add %v125, %v126 : tensor<128x64x56x56xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v131 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v132 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v133 = stablehlo.reduce(%v129 init: %v130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v134 = stablehlo.broadcast_in_dim %v133, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v135 = stablehlo.divide %v134, %v131 : tensor<128x64x56x56xf32>
    %v136 = stablehlo.subtract %v129, %v135 : tensor<128x64x56x56xf32>
    %v137 = stablehlo.multiply %v136, %v136 : tensor<128x64x56x56xf32>
    %v138 = stablehlo.reduce(%v137 init: %v130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v139 = stablehlo.broadcast_in_dim %v138, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v140 = stablehlo.divide %v139, %v131 : tensor<128x64x56x56xf32>
    %v141 = stablehlo.add %v140, %v132 : tensor<128x64x56x56xf32>
    %v142 = stablehlo.rsqrt %v141 : tensor<128x64x56x56xf32>
    %v143 = stablehlo.multiply %v136, %v142 : tensor<128x64x56x56xf32>
    %v144 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v145 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v146 = stablehlo.multiply %v143, %v144 : tensor<128x64x56x56xf32>
    %v147 = stablehlo.add %v146, %v145 : tensor<128x64x56x56xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v150 = stablehlo.reshape %v94 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v151 = stablehlo.add %v149, %v150 : tensor<128x64x56x56xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v154 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v155 = stablehlo.maximum %v153, %v154 : tensor<128x64x56x56xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v158 = stablehlo.convolution(%v157, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v159 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v160 = stablehlo.add %v158, %v159 : tensor<128x64x56x56xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v164 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v165 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v166 = stablehlo.reduce(%v162 init: %v163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v167 = stablehlo.broadcast_in_dim %v166, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v168 = stablehlo.divide %v167, %v164 : tensor<128x64x56x56xf32>
    %v169 = stablehlo.subtract %v162, %v168 : tensor<128x64x56x56xf32>
    %v170 = stablehlo.multiply %v169, %v169 : tensor<128x64x56x56xf32>
    %v171 = stablehlo.reduce(%v170 init: %v163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v172 = stablehlo.broadcast_in_dim %v171, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v173 = stablehlo.divide %v172, %v164 : tensor<128x64x56x56xf32>
    %v174 = stablehlo.add %v173, %v165 : tensor<128x64x56x56xf32>
    %v175 = stablehlo.rsqrt %v174 : tensor<128x64x56x56xf32>
    %v176 = stablehlo.multiply %v169, %v175 : tensor<128x64x56x56xf32>
    %v177 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v178 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v179 = stablehlo.multiply %v176, %v177 : tensor<128x64x56x56xf32>
    %v180 = stablehlo.add %v179, %v178 : tensor<128x64x56x56xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v183 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v184 = stablehlo.maximum %v182, %v183 : tensor<128x64x56x56xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v187 = stablehlo.convolution(%v186, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v188 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v189 = stablehlo.add %v187, %v188 : tensor<128x64x56x56xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v193 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v194 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v195 = stablehlo.reduce(%v191 init: %v192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v196 = stablehlo.broadcast_in_dim %v195, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v197 = stablehlo.divide %v196, %v193 : tensor<128x64x56x56xf32>
    %v198 = stablehlo.subtract %v191, %v197 : tensor<128x64x56x56xf32>
    %v199 = stablehlo.multiply %v198, %v198 : tensor<128x64x56x56xf32>
    %v200 = stablehlo.reduce(%v199 init: %v192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v201 = stablehlo.broadcast_in_dim %v200, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v202 = stablehlo.divide %v201, %v193 : tensor<128x64x56x56xf32>
    %v203 = stablehlo.add %v202, %v194 : tensor<128x64x56x56xf32>
    %v204 = stablehlo.rsqrt %v203 : tensor<128x64x56x56xf32>
    %v205 = stablehlo.multiply %v198, %v204 : tensor<128x64x56x56xf32>
    %v206 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v207 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v208 = stablehlo.multiply %v205, %v206 : tensor<128x64x56x56xf32>
    %v209 = stablehlo.add %v208, %v207 : tensor<128x64x56x56xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v212 = stablehlo.reshape %v156 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v213 = stablehlo.add %v211, %v212 : tensor<128x64x56x56xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v216 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v217 = stablehlo.maximum %v215, %v216 : tensor<128x64x56x56xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v220 = stablehlo.convolution(%v219, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v221 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v222 = stablehlo.add %v220, %v221 : tensor<128x128x28x28xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v226 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v227 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v228 = stablehlo.reduce(%v224 init: %v225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v229 = stablehlo.broadcast_in_dim %v228, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v230 = stablehlo.divide %v229, %v226 : tensor<128x128x28x28xf32>
    %v231 = stablehlo.subtract %v224, %v230 : tensor<128x128x28x28xf32>
    %v232 = stablehlo.multiply %v231, %v231 : tensor<128x128x28x28xf32>
    %v233 = stablehlo.reduce(%v232 init: %v225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v234 = stablehlo.broadcast_in_dim %v233, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v235 = stablehlo.divide %v234, %v226 : tensor<128x128x28x28xf32>
    %v236 = stablehlo.add %v235, %v227 : tensor<128x128x28x28xf32>
    %v237 = stablehlo.rsqrt %v236 : tensor<128x128x28x28xf32>
    %v238 = stablehlo.multiply %v231, %v237 : tensor<128x128x28x28xf32>
    %v239 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v240 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v241 = stablehlo.multiply %v238, %v239 : tensor<128x128x28x28xf32>
    %v242 = stablehlo.add %v241, %v240 : tensor<128x128x28x28xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v245 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v246 = stablehlo.maximum %v244, %v245 : tensor<128x128x28x28xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v249 = stablehlo.convolution(%v248, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v250 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v251 = stablehlo.add %v249, %v250 : tensor<128x128x28x28xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v255 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v256 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v257 = stablehlo.reduce(%v253 init: %v254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v258 = stablehlo.broadcast_in_dim %v257, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v259 = stablehlo.divide %v258, %v255 : tensor<128x128x28x28xf32>
    %v260 = stablehlo.subtract %v253, %v259 : tensor<128x128x28x28xf32>
    %v261 = stablehlo.multiply %v260, %v260 : tensor<128x128x28x28xf32>
    %v262 = stablehlo.reduce(%v261 init: %v254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v263 = stablehlo.broadcast_in_dim %v262, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v264 = stablehlo.divide %v263, %v255 : tensor<128x128x28x28xf32>
    %v265 = stablehlo.add %v264, %v256 : tensor<128x128x28x28xf32>
    %v266 = stablehlo.rsqrt %v265 : tensor<128x128x28x28xf32>
    %v267 = stablehlo.multiply %v260, %v266 : tensor<128x128x28x28xf32>
    %v268 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v269 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v270 = stablehlo.multiply %v267, %v268 : tensor<128x128x28x28xf32>
    %v271 = stablehlo.add %v270, %v269 : tensor<128x128x28x28xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v273 = stablehlo.reshape %v218 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v274 = stablehlo.convolution(%v273, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<128x64x1x1xf32>) -> tensor<128x128x28x28xf32>
    %v275 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v276 = stablehlo.add %v274, %v275 : tensor<128x128x28x28xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v280 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v281 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v282 = stablehlo.reduce(%v278 init: %v279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v283 = stablehlo.broadcast_in_dim %v282, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v284 = stablehlo.divide %v283, %v280 : tensor<128x128x28x28xf32>
    %v285 = stablehlo.subtract %v278, %v284 : tensor<128x128x28x28xf32>
    %v286 = stablehlo.multiply %v285, %v285 : tensor<128x128x28x28xf32>
    %v287 = stablehlo.reduce(%v286 init: %v279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v288 = stablehlo.broadcast_in_dim %v287, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v289 = stablehlo.divide %v288, %v280 : tensor<128x128x28x28xf32>
    %v290 = stablehlo.add %v289, %v281 : tensor<128x128x28x28xf32>
    %v291 = stablehlo.rsqrt %v290 : tensor<128x128x28x28xf32>
    %v292 = stablehlo.multiply %v285, %v291 : tensor<128x128x28x28xf32>
    %v293 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v294 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v295 = stablehlo.multiply %v292, %v293 : tensor<128x128x28x28xf32>
    %v296 = stablehlo.add %v295, %v294 : tensor<128x128x28x28xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v298 = stablehlo.reshape %v272 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v299 = stablehlo.reshape %v297 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v300 = stablehlo.add %v298, %v299 : tensor<128x128x28x28xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v303 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v304 = stablehlo.maximum %v302, %v303 : tensor<128x128x28x28xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v307 = stablehlo.convolution(%v306, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v308 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<128x128x28x28xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v313 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v314 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v315 = stablehlo.reduce(%v311 init: %v312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v316 = stablehlo.broadcast_in_dim %v315, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v317 = stablehlo.divide %v316, %v313 : tensor<128x128x28x28xf32>
    %v318 = stablehlo.subtract %v311, %v317 : tensor<128x128x28x28xf32>
    %v319 = stablehlo.multiply %v318, %v318 : tensor<128x128x28x28xf32>
    %v320 = stablehlo.reduce(%v319 init: %v312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v321 = stablehlo.broadcast_in_dim %v320, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v322 = stablehlo.divide %v321, %v313 : tensor<128x128x28x28xf32>
    %v323 = stablehlo.add %v322, %v314 : tensor<128x128x28x28xf32>
    %v324 = stablehlo.rsqrt %v323 : tensor<128x128x28x28xf32>
    %v325 = stablehlo.multiply %v318, %v324 : tensor<128x128x28x28xf32>
    %v326 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v327 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v328 = stablehlo.multiply %v325, %v326 : tensor<128x128x28x28xf32>
    %v329 = stablehlo.add %v328, %v327 : tensor<128x128x28x28xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v332 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v333 = stablehlo.maximum %v331, %v332 : tensor<128x128x28x28xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v336 = stablehlo.convolution(%v335, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v337 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v338 = stablehlo.add %v336, %v337 : tensor<128x128x28x28xf32>
    %v339 = stablehlo.reshape %v338 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v340 = stablehlo.reshape %v339 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v341 = stablehlo.constant dense<0.0> : tensor<f32>
    %v342 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v343 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v344 = stablehlo.reduce(%v340 init: %v341) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v345 = stablehlo.broadcast_in_dim %v344, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v346 = stablehlo.divide %v345, %v342 : tensor<128x128x28x28xf32>
    %v347 = stablehlo.subtract %v340, %v346 : tensor<128x128x28x28xf32>
    %v348 = stablehlo.multiply %v347, %v347 : tensor<128x128x28x28xf32>
    %v349 = stablehlo.reduce(%v348 init: %v341) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v350 = stablehlo.broadcast_in_dim %v349, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v351 = stablehlo.divide %v350, %v342 : tensor<128x128x28x28xf32>
    %v352 = stablehlo.add %v351, %v343 : tensor<128x128x28x28xf32>
    %v353 = stablehlo.rsqrt %v352 : tensor<128x128x28x28xf32>
    %v354 = stablehlo.multiply %v347, %v353 : tensor<128x128x28x28xf32>
    %v355 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v356 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v357 = stablehlo.multiply %v354, %v355 : tensor<128x128x28x28xf32>
    %v358 = stablehlo.add %v357, %v356 : tensor<128x128x28x28xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v361 = stablehlo.reshape %v305 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v362 = stablehlo.add %v360, %v361 : tensor<128x128x28x28xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v365 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v366 = stablehlo.maximum %v364, %v365 : tensor<128x128x28x28xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v369 = stablehlo.convolution(%v368, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v370 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v371 = stablehlo.add %v369, %v370 : tensor<128x128x28x28xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v374 = stablehlo.constant dense<0.0> : tensor<f32>
    %v375 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v376 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v377 = stablehlo.reduce(%v373 init: %v374) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v378 = stablehlo.broadcast_in_dim %v377, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v379 = stablehlo.divide %v378, %v375 : tensor<128x128x28x28xf32>
    %v380 = stablehlo.subtract %v373, %v379 : tensor<128x128x28x28xf32>
    %v381 = stablehlo.multiply %v380, %v380 : tensor<128x128x28x28xf32>
    %v382 = stablehlo.reduce(%v381 init: %v374) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v383 = stablehlo.broadcast_in_dim %v382, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v384 = stablehlo.divide %v383, %v375 : tensor<128x128x28x28xf32>
    %v385 = stablehlo.add %v384, %v376 : tensor<128x128x28x28xf32>
    %v386 = stablehlo.rsqrt %v385 : tensor<128x128x28x28xf32>
    %v387 = stablehlo.multiply %v380, %v386 : tensor<128x128x28x28xf32>
    %v388 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v389 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v390 = stablehlo.multiply %v387, %v388 : tensor<128x128x28x28xf32>
    %v391 = stablehlo.add %v390, %v389 : tensor<128x128x28x28xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v394 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v395 = stablehlo.maximum %v393, %v394 : tensor<128x128x28x28xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v398 = stablehlo.convolution(%v397, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v399 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v400 = stablehlo.add %v398, %v399 : tensor<128x128x28x28xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v403 = stablehlo.constant dense<0.0> : tensor<f32>
    %v404 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v405 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v406 = stablehlo.reduce(%v402 init: %v403) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v407 = stablehlo.broadcast_in_dim %v406, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v408 = stablehlo.divide %v407, %v404 : tensor<128x128x28x28xf32>
    %v409 = stablehlo.subtract %v402, %v408 : tensor<128x128x28x28xf32>
    %v410 = stablehlo.multiply %v409, %v409 : tensor<128x128x28x28xf32>
    %v411 = stablehlo.reduce(%v410 init: %v403) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v412 = stablehlo.broadcast_in_dim %v411, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v413 = stablehlo.divide %v412, %v404 : tensor<128x128x28x28xf32>
    %v414 = stablehlo.add %v413, %v405 : tensor<128x128x28x28xf32>
    %v415 = stablehlo.rsqrt %v414 : tensor<128x128x28x28xf32>
    %v416 = stablehlo.multiply %v409, %v415 : tensor<128x128x28x28xf32>
    %v417 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v418 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v419 = stablehlo.multiply %v416, %v417 : tensor<128x128x28x28xf32>
    %v420 = stablehlo.add %v419, %v418 : tensor<128x128x28x28xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v423 = stablehlo.reshape %v367 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v424 = stablehlo.add %v422, %v423 : tensor<128x128x28x28xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v426 = stablehlo.reshape %v425 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v427 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v428 = stablehlo.maximum %v426, %v427 : tensor<128x128x28x28xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v431 = stablehlo.convolution(%v430, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v432 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v433 = stablehlo.add %v431, %v432 : tensor<128x128x28x28xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v436 = stablehlo.constant dense<0.0> : tensor<f32>
    %v437 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v438 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v439 = stablehlo.reduce(%v435 init: %v436) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v440 = stablehlo.broadcast_in_dim %v439, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v441 = stablehlo.divide %v440, %v437 : tensor<128x128x28x28xf32>
    %v442 = stablehlo.subtract %v435, %v441 : tensor<128x128x28x28xf32>
    %v443 = stablehlo.multiply %v442, %v442 : tensor<128x128x28x28xf32>
    %v444 = stablehlo.reduce(%v443 init: %v436) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v445 = stablehlo.broadcast_in_dim %v444, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v446 = stablehlo.divide %v445, %v437 : tensor<128x128x28x28xf32>
    %v447 = stablehlo.add %v446, %v438 : tensor<128x128x28x28xf32>
    %v448 = stablehlo.rsqrt %v447 : tensor<128x128x28x28xf32>
    %v449 = stablehlo.multiply %v442, %v448 : tensor<128x128x28x28xf32>
    %v450 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v451 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v452 = stablehlo.multiply %v449, %v450 : tensor<128x128x28x28xf32>
    %v453 = stablehlo.add %v452, %v451 : tensor<128x128x28x28xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v456 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v457 = stablehlo.maximum %v455, %v456 : tensor<128x128x28x28xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v460 = stablehlo.convolution(%v459, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v461 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v462 = stablehlo.add %v460, %v461 : tensor<128x128x28x28xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v466 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v467 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v468 = stablehlo.reduce(%v464 init: %v465) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v469 = stablehlo.broadcast_in_dim %v468, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v470 = stablehlo.divide %v469, %v466 : tensor<128x128x28x28xf32>
    %v471 = stablehlo.subtract %v464, %v470 : tensor<128x128x28x28xf32>
    %v472 = stablehlo.multiply %v471, %v471 : tensor<128x128x28x28xf32>
    %v473 = stablehlo.reduce(%v472 init: %v465) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v474 = stablehlo.broadcast_in_dim %v473, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v475 = stablehlo.divide %v474, %v466 : tensor<128x128x28x28xf32>
    %v476 = stablehlo.add %v475, %v467 : tensor<128x128x28x28xf32>
    %v477 = stablehlo.rsqrt %v476 : tensor<128x128x28x28xf32>
    %v478 = stablehlo.multiply %v471, %v477 : tensor<128x128x28x28xf32>
    %v479 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v480 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v481 = stablehlo.multiply %v478, %v479 : tensor<128x128x28x28xf32>
    %v482 = stablehlo.add %v481, %v480 : tensor<128x128x28x28xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v485 = stablehlo.reshape %v429 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v486 = stablehlo.add %v484, %v485 : tensor<128x128x28x28xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v489 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v490 = stablehlo.maximum %v488, %v489 : tensor<128x128x28x28xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v493 = stablehlo.convolution(%v492, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v494 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v495 = stablehlo.add %v493, %v494 : tensor<128x256x14x14xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v499 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v500 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v501 = stablehlo.reduce(%v497 init: %v498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v502 = stablehlo.broadcast_in_dim %v501, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v503 = stablehlo.divide %v502, %v499 : tensor<128x256x14x14xf32>
    %v504 = stablehlo.subtract %v497, %v503 : tensor<128x256x14x14xf32>
    %v505 = stablehlo.multiply %v504, %v504 : tensor<128x256x14x14xf32>
    %v506 = stablehlo.reduce(%v505 init: %v498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v507 = stablehlo.broadcast_in_dim %v506, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v508 = stablehlo.divide %v507, %v499 : tensor<128x256x14x14xf32>
    %v509 = stablehlo.add %v508, %v500 : tensor<128x256x14x14xf32>
    %v510 = stablehlo.rsqrt %v509 : tensor<128x256x14x14xf32>
    %v511 = stablehlo.multiply %v504, %v510 : tensor<128x256x14x14xf32>
    %v512 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v513 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v514 = stablehlo.multiply %v511, %v512 : tensor<128x256x14x14xf32>
    %v515 = stablehlo.add %v514, %v513 : tensor<128x256x14x14xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v518 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v519 = stablehlo.maximum %v517, %v518 : tensor<128x256x14x14xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v522 = stablehlo.convolution(%v521, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v523 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v524 = stablehlo.add %v522, %v523 : tensor<128x256x14x14xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v528 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v529 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v530 = stablehlo.reduce(%v526 init: %v527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v531 = stablehlo.broadcast_in_dim %v530, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v532 = stablehlo.divide %v531, %v528 : tensor<128x256x14x14xf32>
    %v533 = stablehlo.subtract %v526, %v532 : tensor<128x256x14x14xf32>
    %v534 = stablehlo.multiply %v533, %v533 : tensor<128x256x14x14xf32>
    %v535 = stablehlo.reduce(%v534 init: %v527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v536 = stablehlo.broadcast_in_dim %v535, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v537 = stablehlo.divide %v536, %v528 : tensor<128x256x14x14xf32>
    %v538 = stablehlo.add %v537, %v529 : tensor<128x256x14x14xf32>
    %v539 = stablehlo.rsqrt %v538 : tensor<128x256x14x14xf32>
    %v540 = stablehlo.multiply %v533, %v539 : tensor<128x256x14x14xf32>
    %v541 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v542 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v543 = stablehlo.multiply %v540, %v541 : tensor<128x256x14x14xf32>
    %v544 = stablehlo.add %v543, %v542 : tensor<128x256x14x14xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v546 = stablehlo.reshape %v491 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v547 = stablehlo.convolution(%v546, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<256x128x1x1xf32>) -> tensor<128x256x14x14xf32>
    %v548 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v549 = stablehlo.add %v547, %v548 : tensor<128x256x14x14xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v553 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v554 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v555 = stablehlo.reduce(%v551 init: %v552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v556 = stablehlo.broadcast_in_dim %v555, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v557 = stablehlo.divide %v556, %v553 : tensor<128x256x14x14xf32>
    %v558 = stablehlo.subtract %v551, %v557 : tensor<128x256x14x14xf32>
    %v559 = stablehlo.multiply %v558, %v558 : tensor<128x256x14x14xf32>
    %v560 = stablehlo.reduce(%v559 init: %v552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v561 = stablehlo.broadcast_in_dim %v560, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v562 = stablehlo.divide %v561, %v553 : tensor<128x256x14x14xf32>
    %v563 = stablehlo.add %v562, %v554 : tensor<128x256x14x14xf32>
    %v564 = stablehlo.rsqrt %v563 : tensor<128x256x14x14xf32>
    %v565 = stablehlo.multiply %v558, %v564 : tensor<128x256x14x14xf32>
    %v566 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v567 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v568 = stablehlo.multiply %v565, %v566 : tensor<128x256x14x14xf32>
    %v569 = stablehlo.add %v568, %v567 : tensor<128x256x14x14xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v571 = stablehlo.reshape %v545 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v572 = stablehlo.reshape %v570 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v573 = stablehlo.add %v571, %v572 : tensor<128x256x14x14xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v575 = stablehlo.reshape %v574 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v576 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v577 = stablehlo.maximum %v575, %v576 : tensor<128x256x14x14xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v579 = stablehlo.reshape %v578 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v580 = stablehlo.convolution(%v579, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v581 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v582 = stablehlo.add %v580, %v581 : tensor<128x256x14x14xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v586 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v587 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v588 = stablehlo.reduce(%v584 init: %v585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v589 = stablehlo.broadcast_in_dim %v588, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v590 = stablehlo.divide %v589, %v586 : tensor<128x256x14x14xf32>
    %v591 = stablehlo.subtract %v584, %v590 : tensor<128x256x14x14xf32>
    %v592 = stablehlo.multiply %v591, %v591 : tensor<128x256x14x14xf32>
    %v593 = stablehlo.reduce(%v592 init: %v585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v594 = stablehlo.broadcast_in_dim %v593, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v595 = stablehlo.divide %v594, %v586 : tensor<128x256x14x14xf32>
    %v596 = stablehlo.add %v595, %v587 : tensor<128x256x14x14xf32>
    %v597 = stablehlo.rsqrt %v596 : tensor<128x256x14x14xf32>
    %v598 = stablehlo.multiply %v591, %v597 : tensor<128x256x14x14xf32>
    %v599 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v600 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v601 = stablehlo.multiply %v598, %v599 : tensor<128x256x14x14xf32>
    %v602 = stablehlo.add %v601, %v600 : tensor<128x256x14x14xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v605 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v606 = stablehlo.maximum %v604, %v605 : tensor<128x256x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v609 = stablehlo.convolution(%v608, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v610 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v611 = stablehlo.add %v609, %v610 : tensor<128x256x14x14xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v615 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v616 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v617 = stablehlo.reduce(%v613 init: %v614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v619 = stablehlo.divide %v618, %v615 : tensor<128x256x14x14xf32>
    %v620 = stablehlo.subtract %v613, %v619 : tensor<128x256x14x14xf32>
    %v621 = stablehlo.multiply %v620, %v620 : tensor<128x256x14x14xf32>
    %v622 = stablehlo.reduce(%v621 init: %v614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v623 = stablehlo.broadcast_in_dim %v622, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v624 = stablehlo.divide %v623, %v615 : tensor<128x256x14x14xf32>
    %v625 = stablehlo.add %v624, %v616 : tensor<128x256x14x14xf32>
    %v626 = stablehlo.rsqrt %v625 : tensor<128x256x14x14xf32>
    %v627 = stablehlo.multiply %v620, %v626 : tensor<128x256x14x14xf32>
    %v628 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v629 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v630 = stablehlo.multiply %v627, %v628 : tensor<128x256x14x14xf32>
    %v631 = stablehlo.add %v630, %v629 : tensor<128x256x14x14xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v633 = stablehlo.reshape %v632 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v634 = stablehlo.reshape %v578 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v635 = stablehlo.add %v633, %v634 : tensor<128x256x14x14xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v638 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v639 = stablehlo.maximum %v637, %v638 : tensor<128x256x14x14xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v642 = stablehlo.convolution(%v641, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v643 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v644 = stablehlo.add %v642, %v643 : tensor<128x256x14x14xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v647 = stablehlo.constant dense<0.0> : tensor<f32>
    %v648 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v649 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v650 = stablehlo.reduce(%v646 init: %v647) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v651 = stablehlo.broadcast_in_dim %v650, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v652 = stablehlo.divide %v651, %v648 : tensor<128x256x14x14xf32>
    %v653 = stablehlo.subtract %v646, %v652 : tensor<128x256x14x14xf32>
    %v654 = stablehlo.multiply %v653, %v653 : tensor<128x256x14x14xf32>
    %v655 = stablehlo.reduce(%v654 init: %v647) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v656 = stablehlo.broadcast_in_dim %v655, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v657 = stablehlo.divide %v656, %v648 : tensor<128x256x14x14xf32>
    %v658 = stablehlo.add %v657, %v649 : tensor<128x256x14x14xf32>
    %v659 = stablehlo.rsqrt %v658 : tensor<128x256x14x14xf32>
    %v660 = stablehlo.multiply %v653, %v659 : tensor<128x256x14x14xf32>
    %v661 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v663 = stablehlo.multiply %v660, %v661 : tensor<128x256x14x14xf32>
    %v664 = stablehlo.add %v663, %v662 : tensor<128x256x14x14xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v667 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v668 = stablehlo.maximum %v666, %v667 : tensor<128x256x14x14xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v671 = stablehlo.convolution(%v670, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v672 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v673 = stablehlo.add %v671, %v672 : tensor<128x256x14x14xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v675 = stablehlo.reshape %v674 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v677 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v678 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v679 = stablehlo.reduce(%v675 init: %v676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v680 = stablehlo.broadcast_in_dim %v679, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v681 = stablehlo.divide %v680, %v677 : tensor<128x256x14x14xf32>
    %v682 = stablehlo.subtract %v675, %v681 : tensor<128x256x14x14xf32>
    %v683 = stablehlo.multiply %v682, %v682 : tensor<128x256x14x14xf32>
    %v684 = stablehlo.reduce(%v683 init: %v676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v685 = stablehlo.broadcast_in_dim %v684, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v686 = stablehlo.divide %v685, %v677 : tensor<128x256x14x14xf32>
    %v687 = stablehlo.add %v686, %v678 : tensor<128x256x14x14xf32>
    %v688 = stablehlo.rsqrt %v687 : tensor<128x256x14x14xf32>
    %v689 = stablehlo.multiply %v682, %v688 : tensor<128x256x14x14xf32>
    %v690 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v691 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v692 = stablehlo.multiply %v689, %v690 : tensor<128x256x14x14xf32>
    %v693 = stablehlo.add %v692, %v691 : tensor<128x256x14x14xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v696 = stablehlo.reshape %v640 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v697 = stablehlo.add %v695, %v696 : tensor<128x256x14x14xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v700 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v701 = stablehlo.maximum %v699, %v700 : tensor<128x256x14x14xf32>
    %v702 = stablehlo.reshape %v701 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v704 = stablehlo.convolution(%v703, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v705 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v706 = stablehlo.add %v704, %v705 : tensor<128x256x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v710 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v711 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v712 = stablehlo.reduce(%v708 init: %v709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v713 = stablehlo.broadcast_in_dim %v712, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v714 = stablehlo.divide %v713, %v710 : tensor<128x256x14x14xf32>
    %v715 = stablehlo.subtract %v708, %v714 : tensor<128x256x14x14xf32>
    %v716 = stablehlo.multiply %v715, %v715 : tensor<128x256x14x14xf32>
    %v717 = stablehlo.reduce(%v716 init: %v709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v718 = stablehlo.broadcast_in_dim %v717, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v719 = stablehlo.divide %v718, %v710 : tensor<128x256x14x14xf32>
    %v720 = stablehlo.add %v719, %v711 : tensor<128x256x14x14xf32>
    %v721 = stablehlo.rsqrt %v720 : tensor<128x256x14x14xf32>
    %v722 = stablehlo.multiply %v715, %v721 : tensor<128x256x14x14xf32>
    %v723 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v724 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v725 = stablehlo.multiply %v722, %v723 : tensor<128x256x14x14xf32>
    %v726 = stablehlo.add %v725, %v724 : tensor<128x256x14x14xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v729 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v730 = stablehlo.maximum %v728, %v729 : tensor<128x256x14x14xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v733 = stablehlo.convolution(%v732, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v735 = stablehlo.add %v733, %v734 : tensor<128x256x14x14xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v738 = stablehlo.constant dense<0.0> : tensor<f32>
    %v739 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v740 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v741 = stablehlo.reduce(%v737 init: %v738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v742 = stablehlo.broadcast_in_dim %v741, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v743 = stablehlo.divide %v742, %v739 : tensor<128x256x14x14xf32>
    %v744 = stablehlo.subtract %v737, %v743 : tensor<128x256x14x14xf32>
    %v745 = stablehlo.multiply %v744, %v744 : tensor<128x256x14x14xf32>
    %v746 = stablehlo.reduce(%v745 init: %v738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v747 = stablehlo.broadcast_in_dim %v746, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v748 = stablehlo.divide %v747, %v739 : tensor<128x256x14x14xf32>
    %v749 = stablehlo.add %v748, %v740 : tensor<128x256x14x14xf32>
    %v750 = stablehlo.rsqrt %v749 : tensor<128x256x14x14xf32>
    %v751 = stablehlo.multiply %v744, %v750 : tensor<128x256x14x14xf32>
    %v752 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v753 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v754 = stablehlo.multiply %v751, %v752 : tensor<128x256x14x14xf32>
    %v755 = stablehlo.add %v754, %v753 : tensor<128x256x14x14xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v758 = stablehlo.reshape %v702 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v759 = stablehlo.add %v757, %v758 : tensor<128x256x14x14xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v762 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v763 = stablehlo.maximum %v761, %v762 : tensor<128x256x14x14xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v766 = stablehlo.convolution(%v765, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v767 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<128x256x14x14xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v772 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v773 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v774 = stablehlo.reduce(%v770 init: %v771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v775 = stablehlo.broadcast_in_dim %v774, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v776 = stablehlo.divide %v775, %v772 : tensor<128x256x14x14xf32>
    %v777 = stablehlo.subtract %v770, %v776 : tensor<128x256x14x14xf32>
    %v778 = stablehlo.multiply %v777, %v777 : tensor<128x256x14x14xf32>
    %v779 = stablehlo.reduce(%v778 init: %v771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v780 = stablehlo.broadcast_in_dim %v779, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v781 = stablehlo.divide %v780, %v772 : tensor<128x256x14x14xf32>
    %v782 = stablehlo.add %v781, %v773 : tensor<128x256x14x14xf32>
    %v783 = stablehlo.rsqrt %v782 : tensor<128x256x14x14xf32>
    %v784 = stablehlo.multiply %v777, %v783 : tensor<128x256x14x14xf32>
    %v785 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v786 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v787 = stablehlo.multiply %v784, %v785 : tensor<128x256x14x14xf32>
    %v788 = stablehlo.add %v787, %v786 : tensor<128x256x14x14xf32>
    %v789 = stablehlo.reshape %v788 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v791 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v792 = stablehlo.maximum %v790, %v791 : tensor<128x256x14x14xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v795 = stablehlo.convolution(%v794, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v796 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v797 = stablehlo.add %v795, %v796 : tensor<128x256x14x14xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v801 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v802 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v803 = stablehlo.reduce(%v799 init: %v800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v804 = stablehlo.broadcast_in_dim %v803, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v805 = stablehlo.divide %v804, %v801 : tensor<128x256x14x14xf32>
    %v806 = stablehlo.subtract %v799, %v805 : tensor<128x256x14x14xf32>
    %v807 = stablehlo.multiply %v806, %v806 : tensor<128x256x14x14xf32>
    %v808 = stablehlo.reduce(%v807 init: %v800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v809 = stablehlo.broadcast_in_dim %v808, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v810 = stablehlo.divide %v809, %v801 : tensor<128x256x14x14xf32>
    %v811 = stablehlo.add %v810, %v802 : tensor<128x256x14x14xf32>
    %v812 = stablehlo.rsqrt %v811 : tensor<128x256x14x14xf32>
    %v813 = stablehlo.multiply %v806, %v812 : tensor<128x256x14x14xf32>
    %v814 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v815 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v816 = stablehlo.multiply %v813, %v814 : tensor<128x256x14x14xf32>
    %v817 = stablehlo.add %v816, %v815 : tensor<128x256x14x14xf32>
    %v818 = stablehlo.reshape %v817 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v820 = stablehlo.reshape %v764 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v821 = stablehlo.add %v819, %v820 : tensor<128x256x14x14xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v824 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v825 = stablehlo.maximum %v823, %v824 : tensor<128x256x14x14xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v828 = stablehlo.convolution(%v827, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v829 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v830 = stablehlo.add %v828, %v829 : tensor<128x256x14x14xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v834 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v835 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v836 = stablehlo.reduce(%v832 init: %v833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v837 = stablehlo.broadcast_in_dim %v836, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v838 = stablehlo.divide %v837, %v834 : tensor<128x256x14x14xf32>
    %v839 = stablehlo.subtract %v832, %v838 : tensor<128x256x14x14xf32>
    %v840 = stablehlo.multiply %v839, %v839 : tensor<128x256x14x14xf32>
    %v841 = stablehlo.reduce(%v840 init: %v833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v842 = stablehlo.broadcast_in_dim %v841, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v843 = stablehlo.divide %v842, %v834 : tensor<128x256x14x14xf32>
    %v844 = stablehlo.add %v843, %v835 : tensor<128x256x14x14xf32>
    %v845 = stablehlo.rsqrt %v844 : tensor<128x256x14x14xf32>
    %v846 = stablehlo.multiply %v839, %v845 : tensor<128x256x14x14xf32>
    %v847 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v848 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v849 = stablehlo.multiply %v846, %v847 : tensor<128x256x14x14xf32>
    %v850 = stablehlo.add %v849, %v848 : tensor<128x256x14x14xf32>
    %v851 = stablehlo.reshape %v850 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v853 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v854 = stablehlo.maximum %v852, %v853 : tensor<128x256x14x14xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v857 = stablehlo.convolution(%v856, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v858 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v859 = stablehlo.add %v857, %v858 : tensor<128x256x14x14xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v863 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v864 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v865 = stablehlo.reduce(%v861 init: %v862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v866 = stablehlo.broadcast_in_dim %v865, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v867 = stablehlo.divide %v866, %v863 : tensor<128x256x14x14xf32>
    %v868 = stablehlo.subtract %v861, %v867 : tensor<128x256x14x14xf32>
    %v869 = stablehlo.multiply %v868, %v868 : tensor<128x256x14x14xf32>
    %v870 = stablehlo.reduce(%v869 init: %v862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v871 = stablehlo.broadcast_in_dim %v870, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v872 = stablehlo.divide %v871, %v863 : tensor<128x256x14x14xf32>
    %v873 = stablehlo.add %v872, %v864 : tensor<128x256x14x14xf32>
    %v874 = stablehlo.rsqrt %v873 : tensor<128x256x14x14xf32>
    %v875 = stablehlo.multiply %v868, %v874 : tensor<128x256x14x14xf32>
    %v876 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v877 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v878 = stablehlo.multiply %v875, %v876 : tensor<128x256x14x14xf32>
    %v879 = stablehlo.add %v878, %v877 : tensor<128x256x14x14xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v882 = stablehlo.reshape %v826 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v883 = stablehlo.add %v881, %v882 : tensor<128x256x14x14xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v886 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v887 = stablehlo.maximum %v885, %v886 : tensor<128x256x14x14xf32>
    %v888 = stablehlo.reshape %v887 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v890 = stablehlo.convolution(%v889, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v891 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v892 = stablehlo.add %v890, %v891 : tensor<128x512x7x7xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v896 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v897 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v898 = stablehlo.reduce(%v894 init: %v895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v899 = stablehlo.broadcast_in_dim %v898, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v900 = stablehlo.divide %v899, %v896 : tensor<128x512x7x7xf32>
    %v901 = stablehlo.subtract %v894, %v900 : tensor<128x512x7x7xf32>
    %v902 = stablehlo.multiply %v901, %v901 : tensor<128x512x7x7xf32>
    %v903 = stablehlo.reduce(%v902 init: %v895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v904 = stablehlo.broadcast_in_dim %v903, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v905 = stablehlo.divide %v904, %v896 : tensor<128x512x7x7xf32>
    %v906 = stablehlo.add %v905, %v897 : tensor<128x512x7x7xf32>
    %v907 = stablehlo.rsqrt %v906 : tensor<128x512x7x7xf32>
    %v908 = stablehlo.multiply %v901, %v907 : tensor<128x512x7x7xf32>
    %v909 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v910 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v911 = stablehlo.multiply %v908, %v909 : tensor<128x512x7x7xf32>
    %v912 = stablehlo.add %v911, %v910 : tensor<128x512x7x7xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v915 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v916 = stablehlo.maximum %v914, %v915 : tensor<128x512x7x7xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v919 = stablehlo.convolution(%v918, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v920 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v921 = stablehlo.add %v919, %v920 : tensor<128x512x7x7xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v925 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v926 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v927 = stablehlo.reduce(%v923 init: %v924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v928 = stablehlo.broadcast_in_dim %v927, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v929 = stablehlo.divide %v928, %v925 : tensor<128x512x7x7xf32>
    %v930 = stablehlo.subtract %v923, %v929 : tensor<128x512x7x7xf32>
    %v931 = stablehlo.multiply %v930, %v930 : tensor<128x512x7x7xf32>
    %v932 = stablehlo.reduce(%v931 init: %v924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v933 = stablehlo.broadcast_in_dim %v932, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v934 = stablehlo.divide %v933, %v925 : tensor<128x512x7x7xf32>
    %v935 = stablehlo.add %v934, %v926 : tensor<128x512x7x7xf32>
    %v936 = stablehlo.rsqrt %v935 : tensor<128x512x7x7xf32>
    %v937 = stablehlo.multiply %v930, %v936 : tensor<128x512x7x7xf32>
    %v938 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v939 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v940 = stablehlo.multiply %v937, %v938 : tensor<128x512x7x7xf32>
    %v941 = stablehlo.add %v940, %v939 : tensor<128x512x7x7xf32>
    %v942 = stablehlo.reshape %v941 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v943 = stablehlo.reshape %v888 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v944 = stablehlo.convolution(%v943, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<512x256x1x1xf32>) -> tensor<128x512x7x7xf32>
    %v945 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v946 = stablehlo.add %v944, %v945 : tensor<128x512x7x7xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v950 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v951 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v952 = stablehlo.reduce(%v948 init: %v949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v953 = stablehlo.broadcast_in_dim %v952, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v954 = stablehlo.divide %v953, %v950 : tensor<128x512x7x7xf32>
    %v955 = stablehlo.subtract %v948, %v954 : tensor<128x512x7x7xf32>
    %v956 = stablehlo.multiply %v955, %v955 : tensor<128x512x7x7xf32>
    %v957 = stablehlo.reduce(%v956 init: %v949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v958 = stablehlo.broadcast_in_dim %v957, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v959 = stablehlo.divide %v958, %v950 : tensor<128x512x7x7xf32>
    %v960 = stablehlo.add %v959, %v951 : tensor<128x512x7x7xf32>
    %v961 = stablehlo.rsqrt %v960 : tensor<128x512x7x7xf32>
    %v962 = stablehlo.multiply %v955, %v961 : tensor<128x512x7x7xf32>
    %v963 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v964 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v965 = stablehlo.multiply %v962, %v963 : tensor<128x512x7x7xf32>
    %v966 = stablehlo.add %v965, %v964 : tensor<128x512x7x7xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v968 = stablehlo.reshape %v942 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v969 = stablehlo.reshape %v967 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v970 = stablehlo.add %v968, %v969 : tensor<128x512x7x7xf32>
    %v971 = stablehlo.reshape %v970 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v973 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v974 = stablehlo.maximum %v972, %v973 : tensor<128x512x7x7xf32>
    %v975 = stablehlo.reshape %v974 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v977 = stablehlo.convolution(%v976, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v978 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v979 = stablehlo.add %v977, %v978 : tensor<128x512x7x7xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v983 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v984 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v985 = stablehlo.reduce(%v981 init: %v982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v986 = stablehlo.broadcast_in_dim %v985, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v987 = stablehlo.divide %v986, %v983 : tensor<128x512x7x7xf32>
    %v988 = stablehlo.subtract %v981, %v987 : tensor<128x512x7x7xf32>
    %v989 = stablehlo.multiply %v988, %v988 : tensor<128x512x7x7xf32>
    %v990 = stablehlo.reduce(%v989 init: %v982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v991 = stablehlo.broadcast_in_dim %v990, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v992 = stablehlo.divide %v991, %v983 : tensor<128x512x7x7xf32>
    %v993 = stablehlo.add %v992, %v984 : tensor<128x512x7x7xf32>
    %v994 = stablehlo.rsqrt %v993 : tensor<128x512x7x7xf32>
    %v995 = stablehlo.multiply %v988, %v994 : tensor<128x512x7x7xf32>
    %v996 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v997 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v998 = stablehlo.multiply %v995, %v996 : tensor<128x512x7x7xf32>
    %v999 = stablehlo.add %v998, %v997 : tensor<128x512x7x7xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1002 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1003 = stablehlo.maximum %v1001, %v1002 : tensor<128x512x7x7xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1005 = stablehlo.reshape %v1004 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1006 = stablehlo.convolution(%v1005, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1007 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1008 = stablehlo.add %v1006, %v1007 : tensor<128x512x7x7xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1012 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1013 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1014 = stablehlo.reduce(%v1010 init: %v1011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1015 = stablehlo.broadcast_in_dim %v1014, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1016 = stablehlo.divide %v1015, %v1012 : tensor<128x512x7x7xf32>
    %v1017 = stablehlo.subtract %v1010, %v1016 : tensor<128x512x7x7xf32>
    %v1018 = stablehlo.multiply %v1017, %v1017 : tensor<128x512x7x7xf32>
    %v1019 = stablehlo.reduce(%v1018 init: %v1011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1020 = stablehlo.broadcast_in_dim %v1019, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1021 = stablehlo.divide %v1020, %v1012 : tensor<128x512x7x7xf32>
    %v1022 = stablehlo.add %v1021, %v1013 : tensor<128x512x7x7xf32>
    %v1023 = stablehlo.rsqrt %v1022 : tensor<128x512x7x7xf32>
    %v1024 = stablehlo.multiply %v1017, %v1023 : tensor<128x512x7x7xf32>
    %v1025 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1026 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1027 = stablehlo.multiply %v1024, %v1025 : tensor<128x512x7x7xf32>
    %v1028 = stablehlo.add %v1027, %v1026 : tensor<128x512x7x7xf32>
    %v1029 = stablehlo.reshape %v1028 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1030 = stablehlo.reshape %v1029 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1031 = stablehlo.reshape %v975 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1032 = stablehlo.add %v1030, %v1031 : tensor<128x512x7x7xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1034 = stablehlo.reshape %v1033 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1035 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1036 = stablehlo.maximum %v1034, %v1035 : tensor<128x512x7x7xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1039 = stablehlo.convolution(%v1038, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1040 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1041 = stablehlo.add %v1039, %v1040 : tensor<128x512x7x7xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1045 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1046 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1047 = stablehlo.reduce(%v1043 init: %v1044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1048 = stablehlo.broadcast_in_dim %v1047, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1049 = stablehlo.divide %v1048, %v1045 : tensor<128x512x7x7xf32>
    %v1050 = stablehlo.subtract %v1043, %v1049 : tensor<128x512x7x7xf32>
    %v1051 = stablehlo.multiply %v1050, %v1050 : tensor<128x512x7x7xf32>
    %v1052 = stablehlo.reduce(%v1051 init: %v1044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1053 = stablehlo.broadcast_in_dim %v1052, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1054 = stablehlo.divide %v1053, %v1045 : tensor<128x512x7x7xf32>
    %v1055 = stablehlo.add %v1054, %v1046 : tensor<128x512x7x7xf32>
    %v1056 = stablehlo.rsqrt %v1055 : tensor<128x512x7x7xf32>
    %v1057 = stablehlo.multiply %v1050, %v1056 : tensor<128x512x7x7xf32>
    %v1058 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1059 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1060 = stablehlo.multiply %v1057, %v1058 : tensor<128x512x7x7xf32>
    %v1061 = stablehlo.add %v1060, %v1059 : tensor<128x512x7x7xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1064 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1065 = stablehlo.maximum %v1063, %v1064 : tensor<128x512x7x7xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1068 = stablehlo.convolution(%v1067, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1069 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1070 = stablehlo.add %v1068, %v1069 : tensor<128x512x7x7xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1074 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1075 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1076 = stablehlo.reduce(%v1072 init: %v1073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1077 = stablehlo.broadcast_in_dim %v1076, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1078 = stablehlo.divide %v1077, %v1074 : tensor<128x512x7x7xf32>
    %v1079 = stablehlo.subtract %v1072, %v1078 : tensor<128x512x7x7xf32>
    %v1080 = stablehlo.multiply %v1079, %v1079 : tensor<128x512x7x7xf32>
    %v1081 = stablehlo.reduce(%v1080 init: %v1073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1082 = stablehlo.broadcast_in_dim %v1081, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1083 = stablehlo.divide %v1082, %v1074 : tensor<128x512x7x7xf32>
    %v1084 = stablehlo.add %v1083, %v1075 : tensor<128x512x7x7xf32>
    %v1085 = stablehlo.rsqrt %v1084 : tensor<128x512x7x7xf32>
    %v1086 = stablehlo.multiply %v1079, %v1085 : tensor<128x512x7x7xf32>
    %v1087 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1088 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1089 = stablehlo.multiply %v1086, %v1087 : tensor<128x512x7x7xf32>
    %v1090 = stablehlo.add %v1089, %v1088 : tensor<128x512x7x7xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1093 = stablehlo.reshape %v1037 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1094 = stablehlo.add %v1092, %v1093 : tensor<128x512x7x7xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1097 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1098 = stablehlo.maximum %v1096, %v1097 : tensor<128x512x7x7xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1102 = stablehlo.reduce(%v1100 init: %v1101) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<128x512xf32>
    %v1103 = stablehlo.constant dense<49.0> : tensor<128x512xf32>
    %v1104 = stablehlo.divide %v1102, %v1103 : tensor<128x512xf32>
    %v1105 = stablehlo.dot_general %v1104, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v1106 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v1107 = stablehlo.add %v1105, %v1106 : tensor<128x10xf32>
    %v1108 = stablehlo.reshape %v1107 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v1109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1110 = stablehlo.exponential %v1108 : tensor<128x1x10xf32>
    %v1111 = stablehlo.reduce(%v1110 init: %v1109) applies stablehlo.add across dimensions = [2] : (tensor<128x1x10xf32>, tensor<f32>) -> tensor<128x1xf32>
    %v1112 = stablehlo.broadcast_in_dim %v1111, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x1x10xf32>
    %v1113 = stablehlo.divide %v1110, %v1112 : tensor<128x1x10xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<128x1x10xf32>) -> tensor<128x10xf32>
    %v1115 = stablehlo.subtract %v1114, %onehot : tensor<128x10xf32>
    %v1116 = stablehlo.constant dense<0.100000> : tensor<128x10xf32>
    %v1117 = stablehlo.multiply %onehot, %v1116 : tensor<128x10xf32>
    %v1118 = stablehlo.add %v1115, %v1117 : tensor<128x10xf32>
    %v1119 = stablehlo.constant dense<-0.010000> : tensor<128x10xf32>
    %v1120 = stablehlo.add %v1118, %v1119 : tensor<128x10xf32>
    %v1121 = stablehlo.constant dense<128.0> : tensor<128x10xf32>
    %v1122 = stablehlo.divide %v1120, %v1121 : tensor<128x10xf32>
    %v1123 = stablehlo.reshape %v1122 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v1124 = stablehlo.dot_general %v1123, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x10xf32>, tensor<512x10xf32>) -> tensor<128x1x512xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v1126 = stablehlo.dot_general %v1104, %v1122, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v1127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1128 = stablehlo.reduce(%v1122 init: %v1127) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1129 = stablehlo.broadcast_in_dim %v1125, dims = [0, 1] : (tensor<128x512xf32>) -> tensor<128x512x7x7xf32>
    %v1130 = stablehlo.constant dense<49.0> : tensor<128x512x7x7xf32>
    %v1131 = stablehlo.divide %v1129, %v1130 : tensor<128x512x7x7xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1134 = stablehlo.reshape %v1095 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1135 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1136 = stablehlo.compare GT, %v1134, %v1135 : (tensor<128x512x7x7xf32>, tensor<128x512x7x7xf32>) -> tensor<128x512x7x7xi1>
    %v1137 = stablehlo.select %v1136, %v1133, %v1135 : tensor<128x512x7x7xi1>, tensor<128x512x7x7xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1139 = stablehlo.reshape %v1071 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1141 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1142 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1143 = stablehlo.reduce(%v1139 init: %v1140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1144 = stablehlo.broadcast_in_dim %v1143, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1145 = stablehlo.divide %v1144, %v1141 : tensor<128x512x7x7xf32>
    %v1146 = stablehlo.subtract %v1139, %v1145 : tensor<128x512x7x7xf32>
    %v1147 = stablehlo.multiply %v1146, %v1146 : tensor<128x512x7x7xf32>
    %v1148 = stablehlo.reduce(%v1147 init: %v1140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1149 = stablehlo.broadcast_in_dim %v1148, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1150 = stablehlo.divide %v1149, %v1141 : tensor<128x512x7x7xf32>
    %v1151 = stablehlo.add %v1150, %v1142 : tensor<128x512x7x7xf32>
    %v1152 = stablehlo.rsqrt %v1151 : tensor<128x512x7x7xf32>
    %v1153 = stablehlo.multiply %v1146, %v1152 : tensor<128x512x7x7xf32>
    %v1154 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1155 = stablehlo.reshape %v1138 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1156 = stablehlo.multiply %v1154, %v1155 : tensor<128x512x7x7xf32>
    %v1157 = stablehlo.reduce(%v1156 init: %v1140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1158 = stablehlo.broadcast_in_dim %v1157, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1159 = stablehlo.multiply %v1153, %v1156 : tensor<128x512x7x7xf32>
    %v1160 = stablehlo.reduce(%v1159 init: %v1140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1161 = stablehlo.broadcast_in_dim %v1160, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1162 = stablehlo.multiply %v1156, %v1141 : tensor<128x512x7x7xf32>
    %v1163 = stablehlo.subtract %v1162, %v1158 : tensor<128x512x7x7xf32>
    %v1164 = stablehlo.multiply %v1153, %v1161 : tensor<128x512x7x7xf32>
    %v1165 = stablehlo.subtract %v1163, %v1164 : tensor<128x512x7x7xf32>
    %v1166 = stablehlo.divide %v1152, %v1141 : tensor<128x512x7x7xf32>
    %v1167 = stablehlo.multiply %v1166, %v1165 : tensor<128x512x7x7xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1170 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1171 = stablehlo.transpose %v1170, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1172 = stablehlo.convolution(%v1169, %v1171)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1175 = stablehlo.reshape %v1062 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1176 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1177 = stablehlo.compare GT, %v1175, %v1176 : (tensor<128x512x7x7xf32>, tensor<128x512x7x7xf32>) -> tensor<128x512x7x7xi1>
    %v1178 = stablehlo.select %v1177, %v1174, %v1176 : tensor<128x512x7x7xi1>, tensor<128x512x7x7xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1180 = stablehlo.reshape %v1042 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1182 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1183 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1184 = stablehlo.reduce(%v1180 init: %v1181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1185 = stablehlo.broadcast_in_dim %v1184, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1186 = stablehlo.divide %v1185, %v1182 : tensor<128x512x7x7xf32>
    %v1187 = stablehlo.subtract %v1180, %v1186 : tensor<128x512x7x7xf32>
    %v1188 = stablehlo.multiply %v1187, %v1187 : tensor<128x512x7x7xf32>
    %v1189 = stablehlo.reduce(%v1188 init: %v1181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1190 = stablehlo.broadcast_in_dim %v1189, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1191 = stablehlo.divide %v1190, %v1182 : tensor<128x512x7x7xf32>
    %v1192 = stablehlo.add %v1191, %v1183 : tensor<128x512x7x7xf32>
    %v1193 = stablehlo.rsqrt %v1192 : tensor<128x512x7x7xf32>
    %v1194 = stablehlo.multiply %v1187, %v1193 : tensor<128x512x7x7xf32>
    %v1195 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1196 = stablehlo.reshape %v1179 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1197 = stablehlo.multiply %v1195, %v1196 : tensor<128x512x7x7xf32>
    %v1198 = stablehlo.reduce(%v1197 init: %v1181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1199 = stablehlo.broadcast_in_dim %v1198, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1200 = stablehlo.multiply %v1194, %v1197 : tensor<128x512x7x7xf32>
    %v1201 = stablehlo.reduce(%v1200 init: %v1181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1202 = stablehlo.broadcast_in_dim %v1201, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1203 = stablehlo.multiply %v1197, %v1182 : tensor<128x512x7x7xf32>
    %v1204 = stablehlo.subtract %v1203, %v1199 : tensor<128x512x7x7xf32>
    %v1205 = stablehlo.multiply %v1194, %v1202 : tensor<128x512x7x7xf32>
    %v1206 = stablehlo.subtract %v1204, %v1205 : tensor<128x512x7x7xf32>
    %v1207 = stablehlo.divide %v1193, %v1182 : tensor<128x512x7x7xf32>
    %v1208 = stablehlo.multiply %v1207, %v1206 : tensor<128x512x7x7xf32>
    %v1209 = stablehlo.reshape %v1208 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1211 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1212 = stablehlo.transpose %v1211, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1213 = stablehlo.convolution(%v1210, %v1212)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1214 = stablehlo.reshape %v1213 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1215 = stablehlo.reshape %v1214 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1216 = stablehlo.reshape %v1138 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1217 = stablehlo.add %v1215, %v1216 : tensor<128x512x7x7xf32>
    %v1218 = stablehlo.reshape %v1217 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1219 = stablehlo.reshape %v1037 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1220 = stablehlo.reshape %v1209 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1221 = stablehlo.transpose %v1219, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1222 = stablehlo.transpose %v1220, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1223 = stablehlo.convolution(%v1221, %v1222)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x128x7x7xf32>, tensor<512x128x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1224 = stablehlo.transpose %v1223, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1225 = stablehlo.reshape %v1042 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1227 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1228 = stablehlo.reduce(%v1225 init: %v1226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1229 = stablehlo.broadcast_in_dim %v1228, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1230 = stablehlo.divide %v1229, %v1227 : tensor<128x512x7x7xf32>
    %v1231 = stablehlo.subtract %v1225, %v1230 : tensor<128x512x7x7xf32>
    %v1232 = stablehlo.multiply %v1231, %v1231 : tensor<128x512x7x7xf32>
    %v1233 = stablehlo.reduce(%v1232 init: %v1226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1234 = stablehlo.broadcast_in_dim %v1233, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1235 = stablehlo.divide %v1234, %v1227 : tensor<128x512x7x7xf32>
    %v1236 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1237 = stablehlo.add %v1235, %v1236 : tensor<128x512x7x7xf32>
    %v1238 = stablehlo.rsqrt %v1237 : tensor<128x512x7x7xf32>
    %v1239 = stablehlo.multiply %v1231, %v1238 : tensor<128x512x7x7xf32>
    %v1240 = stablehlo.reshape %v1179 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1241 = stablehlo.multiply %v1240, %v1239 : tensor<128x512x7x7xf32>
    %v1242 = stablehlo.reduce(%v1241 init: %v1226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1243 = stablehlo.reshape %v1179 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1245 = stablehlo.reduce(%v1243 init: %v1244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1246 = stablehlo.reshape %v1066 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1247 = stablehlo.reshape %v1168 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1248 = stablehlo.transpose %v1246, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1249 = stablehlo.transpose %v1247, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1250 = stablehlo.convolution(%v1248, %v1249)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x128x7x7xf32>, tensor<512x128x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1251 = stablehlo.transpose %v1250, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1252 = stablehlo.reshape %v1071 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1254 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1255 = stablehlo.reduce(%v1252 init: %v1253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1256 = stablehlo.broadcast_in_dim %v1255, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1257 = stablehlo.divide %v1256, %v1254 : tensor<128x512x7x7xf32>
    %v1258 = stablehlo.subtract %v1252, %v1257 : tensor<128x512x7x7xf32>
    %v1259 = stablehlo.multiply %v1258, %v1258 : tensor<128x512x7x7xf32>
    %v1260 = stablehlo.reduce(%v1259 init: %v1253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1261 = stablehlo.broadcast_in_dim %v1260, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1262 = stablehlo.divide %v1261, %v1254 : tensor<128x512x7x7xf32>
    %v1263 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1264 = stablehlo.add %v1262, %v1263 : tensor<128x512x7x7xf32>
    %v1265 = stablehlo.rsqrt %v1264 : tensor<128x512x7x7xf32>
    %v1266 = stablehlo.multiply %v1258, %v1265 : tensor<128x512x7x7xf32>
    %v1267 = stablehlo.reshape %v1138 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1268 = stablehlo.multiply %v1267, %v1266 : tensor<128x512x7x7xf32>
    %v1269 = stablehlo.reduce(%v1268 init: %v1253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1270 = stablehlo.reshape %v1138 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1272 = stablehlo.reduce(%v1270 init: %v1271) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1273 = stablehlo.reshape %v1218 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1274 = stablehlo.reshape %v1033 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1275 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1276 = stablehlo.compare GT, %v1274, %v1275 : (tensor<128x512x7x7xf32>, tensor<128x512x7x7xf32>) -> tensor<128x512x7x7xi1>
    %v1277 = stablehlo.select %v1276, %v1273, %v1275 : tensor<128x512x7x7xi1>, tensor<128x512x7x7xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1279 = stablehlo.reshape %v1009 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1281 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1282 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1283 = stablehlo.reduce(%v1279 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1284 = stablehlo.broadcast_in_dim %v1283, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1285 = stablehlo.divide %v1284, %v1281 : tensor<128x512x7x7xf32>
    %v1286 = stablehlo.subtract %v1279, %v1285 : tensor<128x512x7x7xf32>
    %v1287 = stablehlo.multiply %v1286, %v1286 : tensor<128x512x7x7xf32>
    %v1288 = stablehlo.reduce(%v1287 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1289 = stablehlo.broadcast_in_dim %v1288, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1290 = stablehlo.divide %v1289, %v1281 : tensor<128x512x7x7xf32>
    %v1291 = stablehlo.add %v1290, %v1282 : tensor<128x512x7x7xf32>
    %v1292 = stablehlo.rsqrt %v1291 : tensor<128x512x7x7xf32>
    %v1293 = stablehlo.multiply %v1286, %v1292 : tensor<128x512x7x7xf32>
    %v1294 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1295 = stablehlo.reshape %v1278 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1296 = stablehlo.multiply %v1294, %v1295 : tensor<128x512x7x7xf32>
    %v1297 = stablehlo.reduce(%v1296 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1298 = stablehlo.broadcast_in_dim %v1297, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1299 = stablehlo.multiply %v1293, %v1296 : tensor<128x512x7x7xf32>
    %v1300 = stablehlo.reduce(%v1299 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1301 = stablehlo.broadcast_in_dim %v1300, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1302 = stablehlo.multiply %v1296, %v1281 : tensor<128x512x7x7xf32>
    %v1303 = stablehlo.subtract %v1302, %v1298 : tensor<128x512x7x7xf32>
    %v1304 = stablehlo.multiply %v1293, %v1301 : tensor<128x512x7x7xf32>
    %v1305 = stablehlo.subtract %v1303, %v1304 : tensor<128x512x7x7xf32>
    %v1306 = stablehlo.divide %v1292, %v1281 : tensor<128x512x7x7xf32>
    %v1307 = stablehlo.multiply %v1306, %v1305 : tensor<128x512x7x7xf32>
    %v1308 = stablehlo.reshape %v1307 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1309 = stablehlo.reshape %v1308 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1310 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1311 = stablehlo.transpose %v1310, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1312 = stablehlo.convolution(%v1309, %v1311)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1315 = stablehlo.reshape %v1000 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1316 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1317 = stablehlo.compare GT, %v1315, %v1316 : (tensor<128x512x7x7xf32>, tensor<128x512x7x7xf32>) -> tensor<128x512x7x7xi1>
    %v1318 = stablehlo.select %v1317, %v1314, %v1316 : tensor<128x512x7x7xi1>, tensor<128x512x7x7xf32>
    %v1319 = stablehlo.reshape %v1318 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1320 = stablehlo.reshape %v980 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1322 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1323 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1324 = stablehlo.reduce(%v1320 init: %v1321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1325 = stablehlo.broadcast_in_dim %v1324, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1326 = stablehlo.divide %v1325, %v1322 : tensor<128x512x7x7xf32>
    %v1327 = stablehlo.subtract %v1320, %v1326 : tensor<128x512x7x7xf32>
    %v1328 = stablehlo.multiply %v1327, %v1327 : tensor<128x512x7x7xf32>
    %v1329 = stablehlo.reduce(%v1328 init: %v1321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1330 = stablehlo.broadcast_in_dim %v1329, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1331 = stablehlo.divide %v1330, %v1322 : tensor<128x512x7x7xf32>
    %v1332 = stablehlo.add %v1331, %v1323 : tensor<128x512x7x7xf32>
    %v1333 = stablehlo.rsqrt %v1332 : tensor<128x512x7x7xf32>
    %v1334 = stablehlo.multiply %v1327, %v1333 : tensor<128x512x7x7xf32>
    %v1335 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1336 = stablehlo.reshape %v1319 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1337 = stablehlo.multiply %v1335, %v1336 : tensor<128x512x7x7xf32>
    %v1338 = stablehlo.reduce(%v1337 init: %v1321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1339 = stablehlo.broadcast_in_dim %v1338, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1340 = stablehlo.multiply %v1334, %v1337 : tensor<128x512x7x7xf32>
    %v1341 = stablehlo.reduce(%v1340 init: %v1321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1342 = stablehlo.broadcast_in_dim %v1341, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1343 = stablehlo.multiply %v1337, %v1322 : tensor<128x512x7x7xf32>
    %v1344 = stablehlo.subtract %v1343, %v1339 : tensor<128x512x7x7xf32>
    %v1345 = stablehlo.multiply %v1334, %v1342 : tensor<128x512x7x7xf32>
    %v1346 = stablehlo.subtract %v1344, %v1345 : tensor<128x512x7x7xf32>
    %v1347 = stablehlo.divide %v1333, %v1322 : tensor<128x512x7x7xf32>
    %v1348 = stablehlo.multiply %v1347, %v1346 : tensor<128x512x7x7xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1350 = stablehlo.reshape %v1349 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1351 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1352 = stablehlo.transpose %v1351, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1353 = stablehlo.convolution(%v1350, %v1352)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1355 = stablehlo.reshape %v1354 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1356 = stablehlo.reshape %v1278 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1357 = stablehlo.add %v1355, %v1356 : tensor<128x512x7x7xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1359 = stablehlo.reshape %v975 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1360 = stablehlo.reshape %v1349 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1361 = stablehlo.transpose %v1359, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1362 = stablehlo.transpose %v1360, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1363 = stablehlo.convolution(%v1361, %v1362)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x128x7x7xf32>, tensor<512x128x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1364 = stablehlo.transpose %v1363, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1365 = stablehlo.reshape %v980 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1366 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1367 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1368 = stablehlo.reduce(%v1365 init: %v1366) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1369 = stablehlo.broadcast_in_dim %v1368, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1370 = stablehlo.divide %v1369, %v1367 : tensor<128x512x7x7xf32>
    %v1371 = stablehlo.subtract %v1365, %v1370 : tensor<128x512x7x7xf32>
    %v1372 = stablehlo.multiply %v1371, %v1371 : tensor<128x512x7x7xf32>
    %v1373 = stablehlo.reduce(%v1372 init: %v1366) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1374 = stablehlo.broadcast_in_dim %v1373, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1375 = stablehlo.divide %v1374, %v1367 : tensor<128x512x7x7xf32>
    %v1376 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1377 = stablehlo.add %v1375, %v1376 : tensor<128x512x7x7xf32>
    %v1378 = stablehlo.rsqrt %v1377 : tensor<128x512x7x7xf32>
    %v1379 = stablehlo.multiply %v1371, %v1378 : tensor<128x512x7x7xf32>
    %v1380 = stablehlo.reshape %v1319 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1381 = stablehlo.multiply %v1380, %v1379 : tensor<128x512x7x7xf32>
    %v1382 = stablehlo.reduce(%v1381 init: %v1366) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1383 = stablehlo.reshape %v1319 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1384 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1385 = stablehlo.reduce(%v1383 init: %v1384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1386 = stablehlo.reshape %v1004 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1387 = stablehlo.reshape %v1308 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1388 = stablehlo.transpose %v1386, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1389 = stablehlo.transpose %v1387, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1390 = stablehlo.convolution(%v1388, %v1389)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x128x7x7xf32>, tensor<512x128x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1391 = stablehlo.transpose %v1390, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1392 = stablehlo.reshape %v1009 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1393 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1394 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1395 = stablehlo.reduce(%v1392 init: %v1393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1396 = stablehlo.broadcast_in_dim %v1395, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1397 = stablehlo.divide %v1396, %v1394 : tensor<128x512x7x7xf32>
    %v1398 = stablehlo.subtract %v1392, %v1397 : tensor<128x512x7x7xf32>
    %v1399 = stablehlo.multiply %v1398, %v1398 : tensor<128x512x7x7xf32>
    %v1400 = stablehlo.reduce(%v1399 init: %v1393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1401 = stablehlo.broadcast_in_dim %v1400, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1402 = stablehlo.divide %v1401, %v1394 : tensor<128x512x7x7xf32>
    %v1403 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1404 = stablehlo.add %v1402, %v1403 : tensor<128x512x7x7xf32>
    %v1405 = stablehlo.rsqrt %v1404 : tensor<128x512x7x7xf32>
    %v1406 = stablehlo.multiply %v1398, %v1405 : tensor<128x512x7x7xf32>
    %v1407 = stablehlo.reshape %v1278 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1408 = stablehlo.multiply %v1407, %v1406 : tensor<128x512x7x7xf32>
    %v1409 = stablehlo.reduce(%v1408 init: %v1393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1410 = stablehlo.reshape %v1278 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1411 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1412 = stablehlo.reduce(%v1410 init: %v1411) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1413 = stablehlo.reshape %v1358 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1414 = stablehlo.reshape %v971 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1415 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1416 = stablehlo.compare GT, %v1414, %v1415 : (tensor<128x512x7x7xf32>, tensor<128x512x7x7xf32>) -> tensor<128x512x7x7xi1>
    %v1417 = stablehlo.select %v1416, %v1413, %v1415 : tensor<128x512x7x7xi1>, tensor<128x512x7x7xf32>
    %v1418 = stablehlo.reshape %v1417 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1419 = stablehlo.reshape %v922 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1421 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1422 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1423 = stablehlo.reduce(%v1419 init: %v1420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1424 = stablehlo.broadcast_in_dim %v1423, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1425 = stablehlo.divide %v1424, %v1421 : tensor<128x512x7x7xf32>
    %v1426 = stablehlo.subtract %v1419, %v1425 : tensor<128x512x7x7xf32>
    %v1427 = stablehlo.multiply %v1426, %v1426 : tensor<128x512x7x7xf32>
    %v1428 = stablehlo.reduce(%v1427 init: %v1420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1429 = stablehlo.broadcast_in_dim %v1428, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1430 = stablehlo.divide %v1429, %v1421 : tensor<128x512x7x7xf32>
    %v1431 = stablehlo.add %v1430, %v1422 : tensor<128x512x7x7xf32>
    %v1432 = stablehlo.rsqrt %v1431 : tensor<128x512x7x7xf32>
    %v1433 = stablehlo.multiply %v1426, %v1432 : tensor<128x512x7x7xf32>
    %v1434 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1435 = stablehlo.reshape %v1418 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1436 = stablehlo.multiply %v1434, %v1435 : tensor<128x512x7x7xf32>
    %v1437 = stablehlo.reduce(%v1436 init: %v1420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1438 = stablehlo.broadcast_in_dim %v1437, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1439 = stablehlo.multiply %v1433, %v1436 : tensor<128x512x7x7xf32>
    %v1440 = stablehlo.reduce(%v1439 init: %v1420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1441 = stablehlo.broadcast_in_dim %v1440, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1442 = stablehlo.multiply %v1436, %v1421 : tensor<128x512x7x7xf32>
    %v1443 = stablehlo.subtract %v1442, %v1438 : tensor<128x512x7x7xf32>
    %v1444 = stablehlo.multiply %v1433, %v1441 : tensor<128x512x7x7xf32>
    %v1445 = stablehlo.subtract %v1443, %v1444 : tensor<128x512x7x7xf32>
    %v1446 = stablehlo.divide %v1432, %v1421 : tensor<128x512x7x7xf32>
    %v1447 = stablehlo.multiply %v1446, %v1445 : tensor<128x512x7x7xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1449 = stablehlo.reshape %v1448 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1450 = stablehlo.reverse %d4W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1451 = stablehlo.transpose %v1450, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1452 = stablehlo.convolution(%v1449, %v1451)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1453 = stablehlo.reshape %v1452 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1454 = stablehlo.reshape %v1453 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1455 = stablehlo.reshape %v913 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1456 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1457 = stablehlo.compare GT, %v1455, %v1456 : (tensor<128x512x7x7xf32>, tensor<128x512x7x7xf32>) -> tensor<128x512x7x7xi1>
    %v1458 = stablehlo.select %v1457, %v1454, %v1456 : tensor<128x512x7x7xi1>, tensor<128x512x7x7xf32>
    %v1459 = stablehlo.reshape %v1458 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1460 = stablehlo.reshape %v893 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1461 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1462 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1463 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1464 = stablehlo.reduce(%v1460 init: %v1461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1465 = stablehlo.broadcast_in_dim %v1464, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1466 = stablehlo.divide %v1465, %v1462 : tensor<128x512x7x7xf32>
    %v1467 = stablehlo.subtract %v1460, %v1466 : tensor<128x512x7x7xf32>
    %v1468 = stablehlo.multiply %v1467, %v1467 : tensor<128x512x7x7xf32>
    %v1469 = stablehlo.reduce(%v1468 init: %v1461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1470 = stablehlo.broadcast_in_dim %v1469, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1471 = stablehlo.divide %v1470, %v1462 : tensor<128x512x7x7xf32>
    %v1472 = stablehlo.add %v1471, %v1463 : tensor<128x512x7x7xf32>
    %v1473 = stablehlo.rsqrt %v1472 : tensor<128x512x7x7xf32>
    %v1474 = stablehlo.multiply %v1467, %v1473 : tensor<128x512x7x7xf32>
    %v1475 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1476 = stablehlo.reshape %v1459 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1477 = stablehlo.multiply %v1475, %v1476 : tensor<128x512x7x7xf32>
    %v1478 = stablehlo.reduce(%v1477 init: %v1461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1479 = stablehlo.broadcast_in_dim %v1478, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1480 = stablehlo.multiply %v1474, %v1477 : tensor<128x512x7x7xf32>
    %v1481 = stablehlo.reduce(%v1480 init: %v1461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1482 = stablehlo.broadcast_in_dim %v1481, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1483 = stablehlo.multiply %v1477, %v1462 : tensor<128x512x7x7xf32>
    %v1484 = stablehlo.subtract %v1483, %v1479 : tensor<128x512x7x7xf32>
    %v1485 = stablehlo.multiply %v1474, %v1482 : tensor<128x512x7x7xf32>
    %v1486 = stablehlo.subtract %v1484, %v1485 : tensor<128x512x7x7xf32>
    %v1487 = stablehlo.divide %v1473, %v1462 : tensor<128x512x7x7xf32>
    %v1488 = stablehlo.multiply %v1487, %v1486 : tensor<128x512x7x7xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1491 = stablehlo.reverse %d4W1, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v1492 = stablehlo.transpose %v1491, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1493 = stablehlo.convolution(%v1490, %v1492)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<256x512x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v1494 = stablehlo.reshape %v1493 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1495 = stablehlo.reshape %v947 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1496 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1497 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1498 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1499 = stablehlo.reduce(%v1495 init: %v1496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1500 = stablehlo.broadcast_in_dim %v1499, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1501 = stablehlo.divide %v1500, %v1497 : tensor<128x512x7x7xf32>
    %v1502 = stablehlo.subtract %v1495, %v1501 : tensor<128x512x7x7xf32>
    %v1503 = stablehlo.multiply %v1502, %v1502 : tensor<128x512x7x7xf32>
    %v1504 = stablehlo.reduce(%v1503 init: %v1496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1505 = stablehlo.broadcast_in_dim %v1504, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1506 = stablehlo.divide %v1505, %v1497 : tensor<128x512x7x7xf32>
    %v1507 = stablehlo.add %v1506, %v1498 : tensor<128x512x7x7xf32>
    %v1508 = stablehlo.rsqrt %v1507 : tensor<128x512x7x7xf32>
    %v1509 = stablehlo.multiply %v1502, %v1508 : tensor<128x512x7x7xf32>
    %v1510 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1511 = stablehlo.reshape %v1418 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1512 = stablehlo.multiply %v1510, %v1511 : tensor<128x512x7x7xf32>
    %v1513 = stablehlo.reduce(%v1512 init: %v1496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1514 = stablehlo.broadcast_in_dim %v1513, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1515 = stablehlo.multiply %v1509, %v1512 : tensor<128x512x7x7xf32>
    %v1516 = stablehlo.reduce(%v1515 init: %v1496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1517 = stablehlo.broadcast_in_dim %v1516, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1518 = stablehlo.multiply %v1512, %v1497 : tensor<128x512x7x7xf32>
    %v1519 = stablehlo.subtract %v1518, %v1514 : tensor<128x512x7x7xf32>
    %v1520 = stablehlo.multiply %v1509, %v1517 : tensor<128x512x7x7xf32>
    %v1521 = stablehlo.subtract %v1519, %v1520 : tensor<128x512x7x7xf32>
    %v1522 = stablehlo.divide %v1508, %v1497 : tensor<128x512x7x7xf32>
    %v1523 = stablehlo.multiply %v1522, %v1521 : tensor<128x512x7x7xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1526 = stablehlo.reverse %d4Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v1527 = stablehlo.transpose %v1526, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v1528 = stablehlo.convolution(%v1525, %v1527)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<256x512x1x1xf32>) -> tensor<128x256x14x14xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1530 = stablehlo.reshape %v1494 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1531 = stablehlo.reshape %v1529 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1532 = stablehlo.add %v1530, %v1531 : tensor<128x256x14x14xf32>
    %v1533 = stablehlo.reshape %v1532 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1534 = stablehlo.reshape %v888 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1535 = stablehlo.reshape %v1489 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1536 = stablehlo.transpose %v1534, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v1538 = stablehlo.transpose %v1535, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1539 = stablehlo.convolution(%v1536, %v1538)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<512x128x7x7xf32>) -> tensor<256x512x3x3xf32>
    %v1540 = stablehlo.transpose %v1539, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1541 = stablehlo.reshape %v893 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1543 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1544 = stablehlo.reduce(%v1541 init: %v1542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1545 = stablehlo.broadcast_in_dim %v1544, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1546 = stablehlo.divide %v1545, %v1543 : tensor<128x512x7x7xf32>
    %v1547 = stablehlo.subtract %v1541, %v1546 : tensor<128x512x7x7xf32>
    %v1548 = stablehlo.multiply %v1547, %v1547 : tensor<128x512x7x7xf32>
    %v1549 = stablehlo.reduce(%v1548 init: %v1542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1550 = stablehlo.broadcast_in_dim %v1549, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1551 = stablehlo.divide %v1550, %v1543 : tensor<128x512x7x7xf32>
    %v1552 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1553 = stablehlo.add %v1551, %v1552 : tensor<128x512x7x7xf32>
    %v1554 = stablehlo.rsqrt %v1553 : tensor<128x512x7x7xf32>
    %v1555 = stablehlo.multiply %v1547, %v1554 : tensor<128x512x7x7xf32>
    %v1556 = stablehlo.reshape %v1459 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1557 = stablehlo.multiply %v1556, %v1555 : tensor<128x512x7x7xf32>
    %v1558 = stablehlo.reduce(%v1557 init: %v1542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1559 = stablehlo.reshape %v1459 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1560 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1561 = stablehlo.reduce(%v1559 init: %v1560) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1562 = stablehlo.reshape %v917 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1563 = stablehlo.reshape %v1448 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1564 = stablehlo.transpose %v1562, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1565 = stablehlo.transpose %v1563, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1566 = stablehlo.convolution(%v1564, %v1565)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x128x7x7xf32>, tensor<512x128x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1567 = stablehlo.transpose %v1566, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1568 = stablehlo.reshape %v922 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1570 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1571 = stablehlo.reduce(%v1568 init: %v1569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1572 = stablehlo.broadcast_in_dim %v1571, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1573 = stablehlo.divide %v1572, %v1570 : tensor<128x512x7x7xf32>
    %v1574 = stablehlo.subtract %v1568, %v1573 : tensor<128x512x7x7xf32>
    %v1575 = stablehlo.multiply %v1574, %v1574 : tensor<128x512x7x7xf32>
    %v1576 = stablehlo.reduce(%v1575 init: %v1569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1577 = stablehlo.broadcast_in_dim %v1576, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1578 = stablehlo.divide %v1577, %v1570 : tensor<128x512x7x7xf32>
    %v1579 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1580 = stablehlo.add %v1578, %v1579 : tensor<128x512x7x7xf32>
    %v1581 = stablehlo.rsqrt %v1580 : tensor<128x512x7x7xf32>
    %v1582 = stablehlo.multiply %v1574, %v1581 : tensor<128x512x7x7xf32>
    %v1583 = stablehlo.reshape %v1418 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1584 = stablehlo.multiply %v1583, %v1582 : tensor<128x512x7x7xf32>
    %v1585 = stablehlo.reduce(%v1584 init: %v1569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1586 = stablehlo.reshape %v1418 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1587 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1588 = stablehlo.reduce(%v1586 init: %v1587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1589 = stablehlo.reshape %v888 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1590 = stablehlo.reshape %v1524 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1591 = stablehlo.transpose %v1589, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v1593 = stablehlo.transpose %v1590, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1592 = stablehlo.slice %v1591 [0:256, 0:128, 0:13, 0:13] : (tensor<256x128x14x14xf32>) -> tensor<256x128x13x13xf32>
    %v1594 = stablehlo.convolution(%v1592, %v1593)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x13x13xf32>, tensor<512x128x7x7xf32>) -> tensor<256x512x1x1xf32>
    %v1595 = stablehlo.transpose %v1594, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v1596 = stablehlo.reshape %v947 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1598 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v1599 = stablehlo.reduce(%v1596 init: %v1597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1600 = stablehlo.broadcast_in_dim %v1599, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1601 = stablehlo.divide %v1600, %v1598 : tensor<128x512x7x7xf32>
    %v1602 = stablehlo.subtract %v1596, %v1601 : tensor<128x512x7x7xf32>
    %v1603 = stablehlo.multiply %v1602, %v1602 : tensor<128x512x7x7xf32>
    %v1604 = stablehlo.reduce(%v1603 init: %v1597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1605 = stablehlo.broadcast_in_dim %v1604, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1606 = stablehlo.divide %v1605, %v1598 : tensor<128x512x7x7xf32>
    %v1607 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1608 = stablehlo.add %v1606, %v1607 : tensor<128x512x7x7xf32>
    %v1609 = stablehlo.rsqrt %v1608 : tensor<128x512x7x7xf32>
    %v1610 = stablehlo.multiply %v1602, %v1609 : tensor<128x512x7x7xf32>
    %v1611 = stablehlo.reshape %v1418 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1612 = stablehlo.multiply %v1611, %v1610 : tensor<128x512x7x7xf32>
    %v1613 = stablehlo.reduce(%v1612 init: %v1597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1614 = stablehlo.reshape %v1418 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1616 = stablehlo.reduce(%v1614 init: %v1615) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1617 = stablehlo.reshape %v1533 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1618 = stablehlo.reshape %v884 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1619 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v1620 = stablehlo.compare GT, %v1618, %v1619 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v1621 = stablehlo.select %v1620, %v1617, %v1619 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v1622 = stablehlo.reshape %v1621 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1623 = stablehlo.reshape %v860 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1625 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v1626 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1627 = stablehlo.reduce(%v1623 init: %v1624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1628 = stablehlo.broadcast_in_dim %v1627, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1629 = stablehlo.divide %v1628, %v1625 : tensor<128x256x14x14xf32>
    %v1630 = stablehlo.subtract %v1623, %v1629 : tensor<128x256x14x14xf32>
    %v1631 = stablehlo.multiply %v1630, %v1630 : tensor<128x256x14x14xf32>
    %v1632 = stablehlo.reduce(%v1631 init: %v1624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1633 = stablehlo.broadcast_in_dim %v1632, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1634 = stablehlo.divide %v1633, %v1625 : tensor<128x256x14x14xf32>
    %v1635 = stablehlo.add %v1634, %v1626 : tensor<128x256x14x14xf32>
    %v1636 = stablehlo.rsqrt %v1635 : tensor<128x256x14x14xf32>
    %v1637 = stablehlo.multiply %v1630, %v1636 : tensor<128x256x14x14xf32>
    %v1638 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1639 = stablehlo.reshape %v1622 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1640 = stablehlo.multiply %v1638, %v1639 : tensor<128x256x14x14xf32>
    %v1641 = stablehlo.reduce(%v1640 init: %v1624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1642 = stablehlo.broadcast_in_dim %v1641, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1643 = stablehlo.multiply %v1637, %v1640 : tensor<128x256x14x14xf32>
    %v1644 = stablehlo.reduce(%v1643 init: %v1624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1645 = stablehlo.broadcast_in_dim %v1644, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1646 = stablehlo.multiply %v1640, %v1625 : tensor<128x256x14x14xf32>
    %v1647 = stablehlo.subtract %v1646, %v1642 : tensor<128x256x14x14xf32>
    %v1648 = stablehlo.multiply %v1637, %v1645 : tensor<128x256x14x14xf32>
    %v1649 = stablehlo.subtract %v1647, %v1648 : tensor<128x256x14x14xf32>
    %v1650 = stablehlo.divide %v1636, %v1625 : tensor<128x256x14x14xf32>
    %v1651 = stablehlo.multiply %v1650, %v1649 : tensor<128x256x14x14xf32>
    %v1652 = stablehlo.reshape %v1651 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1653 = stablehlo.reshape %v1652 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1654 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1655 = stablehlo.transpose %v1654, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1656 = stablehlo.convolution(%v1653, %v1655)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v1657 = stablehlo.reshape %v1656 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1658 = stablehlo.reshape %v1657 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1659 = stablehlo.reshape %v851 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1660 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v1661 = stablehlo.compare GT, %v1659, %v1660 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v1662 = stablehlo.select %v1661, %v1658, %v1660 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1664 = stablehlo.reshape %v831 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1666 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v1667 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1668 = stablehlo.reduce(%v1664 init: %v1665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1669 = stablehlo.broadcast_in_dim %v1668, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1670 = stablehlo.divide %v1669, %v1666 : tensor<128x256x14x14xf32>
    %v1671 = stablehlo.subtract %v1664, %v1670 : tensor<128x256x14x14xf32>
    %v1672 = stablehlo.multiply %v1671, %v1671 : tensor<128x256x14x14xf32>
    %v1673 = stablehlo.reduce(%v1672 init: %v1665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1674 = stablehlo.broadcast_in_dim %v1673, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1675 = stablehlo.divide %v1674, %v1666 : tensor<128x256x14x14xf32>
    %v1676 = stablehlo.add %v1675, %v1667 : tensor<128x256x14x14xf32>
    %v1677 = stablehlo.rsqrt %v1676 : tensor<128x256x14x14xf32>
    %v1678 = stablehlo.multiply %v1671, %v1677 : tensor<128x256x14x14xf32>
    %v1679 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1680 = stablehlo.reshape %v1663 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1681 = stablehlo.multiply %v1679, %v1680 : tensor<128x256x14x14xf32>
    %v1682 = stablehlo.reduce(%v1681 init: %v1665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1683 = stablehlo.broadcast_in_dim %v1682, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1684 = stablehlo.multiply %v1678, %v1681 : tensor<128x256x14x14xf32>
    %v1685 = stablehlo.reduce(%v1684 init: %v1665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1686 = stablehlo.broadcast_in_dim %v1685, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1687 = stablehlo.multiply %v1681, %v1666 : tensor<128x256x14x14xf32>
    %v1688 = stablehlo.subtract %v1687, %v1683 : tensor<128x256x14x14xf32>
    %v1689 = stablehlo.multiply %v1678, %v1686 : tensor<128x256x14x14xf32>
    %v1690 = stablehlo.subtract %v1688, %v1689 : tensor<128x256x14x14xf32>
    %v1691 = stablehlo.divide %v1677, %v1666 : tensor<128x256x14x14xf32>
    %v1692 = stablehlo.multiply %v1691, %v1690 : tensor<128x256x14x14xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1694 = stablehlo.reshape %v1693 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1695 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1696 = stablehlo.transpose %v1695, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1697 = stablehlo.convolution(%v1694, %v1696)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v1698 = stablehlo.reshape %v1697 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1699 = stablehlo.reshape %v1698 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1700 = stablehlo.reshape %v1622 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1701 = stablehlo.add %v1699, %v1700 : tensor<128x256x14x14xf32>
    %v1702 = stablehlo.reshape %v1701 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1703 = stablehlo.reshape %v826 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1704 = stablehlo.reshape %v1693 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1705 = stablehlo.transpose %v1703, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v1706 = stablehlo.transpose %v1704, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v1707 = stablehlo.convolution(%v1705, %v1706)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1708 = stablehlo.transpose %v1707, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1709 = stablehlo.reshape %v831 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1710 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1711 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v1712 = stablehlo.reduce(%v1709 init: %v1710) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1713 = stablehlo.broadcast_in_dim %v1712, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1714 = stablehlo.divide %v1713, %v1711 : tensor<128x256x14x14xf32>
    %v1715 = stablehlo.subtract %v1709, %v1714 : tensor<128x256x14x14xf32>
    %v1716 = stablehlo.multiply %v1715, %v1715 : tensor<128x256x14x14xf32>
    %v1717 = stablehlo.reduce(%v1716 init: %v1710) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1718 = stablehlo.broadcast_in_dim %v1717, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1719 = stablehlo.divide %v1718, %v1711 : tensor<128x256x14x14xf32>
    %v1720 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1721 = stablehlo.add %v1719, %v1720 : tensor<128x256x14x14xf32>
    %v1722 = stablehlo.rsqrt %v1721 : tensor<128x256x14x14xf32>
    %v1723 = stablehlo.multiply %v1715, %v1722 : tensor<128x256x14x14xf32>
    %v1724 = stablehlo.reshape %v1663 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1725 = stablehlo.multiply %v1724, %v1723 : tensor<128x256x14x14xf32>
    %v1726 = stablehlo.reduce(%v1725 init: %v1710) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1727 = stablehlo.reshape %v1663 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1729 = stablehlo.reduce(%v1727 init: %v1728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1730 = stablehlo.reshape %v855 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1731 = stablehlo.reshape %v1652 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1732 = stablehlo.transpose %v1730, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v1733 = stablehlo.transpose %v1731, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v1734 = stablehlo.convolution(%v1732, %v1733)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1735 = stablehlo.transpose %v1734, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1736 = stablehlo.reshape %v860 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1737 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1738 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v1739 = stablehlo.reduce(%v1736 init: %v1737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1740 = stablehlo.broadcast_in_dim %v1739, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1741 = stablehlo.divide %v1740, %v1738 : tensor<128x256x14x14xf32>
    %v1742 = stablehlo.subtract %v1736, %v1741 : tensor<128x256x14x14xf32>
    %v1743 = stablehlo.multiply %v1742, %v1742 : tensor<128x256x14x14xf32>
    %v1744 = stablehlo.reduce(%v1743 init: %v1737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1745 = stablehlo.broadcast_in_dim %v1744, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1746 = stablehlo.divide %v1745, %v1738 : tensor<128x256x14x14xf32>
    %v1747 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1748 = stablehlo.add %v1746, %v1747 : tensor<128x256x14x14xf32>
    %v1749 = stablehlo.rsqrt %v1748 : tensor<128x256x14x14xf32>
    %v1750 = stablehlo.multiply %v1742, %v1749 : tensor<128x256x14x14xf32>
    %v1751 = stablehlo.reshape %v1622 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1752 = stablehlo.multiply %v1751, %v1750 : tensor<128x256x14x14xf32>
    %v1753 = stablehlo.reduce(%v1752 init: %v1737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1754 = stablehlo.reshape %v1622 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1756 = stablehlo.reduce(%v1754 init: %v1755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1757 = stablehlo.reshape %v1702 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1758 = stablehlo.reshape %v822 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1759 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v1760 = stablehlo.compare GT, %v1758, %v1759 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v1761 = stablehlo.select %v1760, %v1757, %v1759 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v1762 = stablehlo.reshape %v1761 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1763 = stablehlo.reshape %v798 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1764 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1765 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v1766 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1767 = stablehlo.reduce(%v1763 init: %v1764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1768 = stablehlo.broadcast_in_dim %v1767, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1769 = stablehlo.divide %v1768, %v1765 : tensor<128x256x14x14xf32>
    %v1770 = stablehlo.subtract %v1763, %v1769 : tensor<128x256x14x14xf32>
    %v1771 = stablehlo.multiply %v1770, %v1770 : tensor<128x256x14x14xf32>
    %v1772 = stablehlo.reduce(%v1771 init: %v1764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1773 = stablehlo.broadcast_in_dim %v1772, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1774 = stablehlo.divide %v1773, %v1765 : tensor<128x256x14x14xf32>
    %v1775 = stablehlo.add %v1774, %v1766 : tensor<128x256x14x14xf32>
    %v1776 = stablehlo.rsqrt %v1775 : tensor<128x256x14x14xf32>
    %v1777 = stablehlo.multiply %v1770, %v1776 : tensor<128x256x14x14xf32>
    %v1778 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1779 = stablehlo.reshape %v1762 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1780 = stablehlo.multiply %v1778, %v1779 : tensor<128x256x14x14xf32>
    %v1781 = stablehlo.reduce(%v1780 init: %v1764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1782 = stablehlo.broadcast_in_dim %v1781, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1783 = stablehlo.multiply %v1777, %v1780 : tensor<128x256x14x14xf32>
    %v1784 = stablehlo.reduce(%v1783 init: %v1764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1785 = stablehlo.broadcast_in_dim %v1784, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1786 = stablehlo.multiply %v1780, %v1765 : tensor<128x256x14x14xf32>
    %v1787 = stablehlo.subtract %v1786, %v1782 : tensor<128x256x14x14xf32>
    %v1788 = stablehlo.multiply %v1777, %v1785 : tensor<128x256x14x14xf32>
    %v1789 = stablehlo.subtract %v1787, %v1788 : tensor<128x256x14x14xf32>
    %v1790 = stablehlo.divide %v1776, %v1765 : tensor<128x256x14x14xf32>
    %v1791 = stablehlo.multiply %v1790, %v1789 : tensor<128x256x14x14xf32>
    %v1792 = stablehlo.reshape %v1791 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1793 = stablehlo.reshape %v1792 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1794 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1795 = stablehlo.transpose %v1794, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1796 = stablehlo.convolution(%v1793, %v1795)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v1797 = stablehlo.reshape %v1796 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1798 = stablehlo.reshape %v1797 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1799 = stablehlo.reshape %v789 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1800 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v1801 = stablehlo.compare GT, %v1799, %v1800 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v1802 = stablehlo.select %v1801, %v1798, %v1800 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v1803 = stablehlo.reshape %v1802 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1804 = stablehlo.reshape %v769 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1805 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1806 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v1807 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1808 = stablehlo.reduce(%v1804 init: %v1805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1809 = stablehlo.broadcast_in_dim %v1808, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1810 = stablehlo.divide %v1809, %v1806 : tensor<128x256x14x14xf32>
    %v1811 = stablehlo.subtract %v1804, %v1810 : tensor<128x256x14x14xf32>
    %v1812 = stablehlo.multiply %v1811, %v1811 : tensor<128x256x14x14xf32>
    %v1813 = stablehlo.reduce(%v1812 init: %v1805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1814 = stablehlo.broadcast_in_dim %v1813, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1815 = stablehlo.divide %v1814, %v1806 : tensor<128x256x14x14xf32>
    %v1816 = stablehlo.add %v1815, %v1807 : tensor<128x256x14x14xf32>
    %v1817 = stablehlo.rsqrt %v1816 : tensor<128x256x14x14xf32>
    %v1818 = stablehlo.multiply %v1811, %v1817 : tensor<128x256x14x14xf32>
    %v1819 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1820 = stablehlo.reshape %v1803 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1821 = stablehlo.multiply %v1819, %v1820 : tensor<128x256x14x14xf32>
    %v1822 = stablehlo.reduce(%v1821 init: %v1805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1823 = stablehlo.broadcast_in_dim %v1822, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1824 = stablehlo.multiply %v1818, %v1821 : tensor<128x256x14x14xf32>
    %v1825 = stablehlo.reduce(%v1824 init: %v1805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1826 = stablehlo.broadcast_in_dim %v1825, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1827 = stablehlo.multiply %v1821, %v1806 : tensor<128x256x14x14xf32>
    %v1828 = stablehlo.subtract %v1827, %v1823 : tensor<128x256x14x14xf32>
    %v1829 = stablehlo.multiply %v1818, %v1826 : tensor<128x256x14x14xf32>
    %v1830 = stablehlo.subtract %v1828, %v1829 : tensor<128x256x14x14xf32>
    %v1831 = stablehlo.divide %v1817, %v1806 : tensor<128x256x14x14xf32>
    %v1832 = stablehlo.multiply %v1831, %v1830 : tensor<128x256x14x14xf32>
    %v1833 = stablehlo.reshape %v1832 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1834 = stablehlo.reshape %v1833 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1835 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1836 = stablehlo.transpose %v1835, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1837 = stablehlo.convolution(%v1834, %v1836)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v1838 = stablehlo.reshape %v1837 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1839 = stablehlo.reshape %v1838 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1840 = stablehlo.reshape %v1762 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1841 = stablehlo.add %v1839, %v1840 : tensor<128x256x14x14xf32>
    %v1842 = stablehlo.reshape %v1841 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1843 = stablehlo.reshape %v764 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1844 = stablehlo.reshape %v1833 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1845 = stablehlo.transpose %v1843, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v1846 = stablehlo.transpose %v1844, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v1847 = stablehlo.convolution(%v1845, %v1846)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1848 = stablehlo.transpose %v1847, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1849 = stablehlo.reshape %v769 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1851 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v1852 = stablehlo.reduce(%v1849 init: %v1850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1853 = stablehlo.broadcast_in_dim %v1852, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1854 = stablehlo.divide %v1853, %v1851 : tensor<128x256x14x14xf32>
    %v1855 = stablehlo.subtract %v1849, %v1854 : tensor<128x256x14x14xf32>
    %v1856 = stablehlo.multiply %v1855, %v1855 : tensor<128x256x14x14xf32>
    %v1857 = stablehlo.reduce(%v1856 init: %v1850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1858 = stablehlo.broadcast_in_dim %v1857, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1859 = stablehlo.divide %v1858, %v1851 : tensor<128x256x14x14xf32>
    %v1860 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1861 = stablehlo.add %v1859, %v1860 : tensor<128x256x14x14xf32>
    %v1862 = stablehlo.rsqrt %v1861 : tensor<128x256x14x14xf32>
    %v1863 = stablehlo.multiply %v1855, %v1862 : tensor<128x256x14x14xf32>
    %v1864 = stablehlo.reshape %v1803 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1865 = stablehlo.multiply %v1864, %v1863 : tensor<128x256x14x14xf32>
    %v1866 = stablehlo.reduce(%v1865 init: %v1850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1867 = stablehlo.reshape %v1803 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1868 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1869 = stablehlo.reduce(%v1867 init: %v1868) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1870 = stablehlo.reshape %v793 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1871 = stablehlo.reshape %v1792 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1872 = stablehlo.transpose %v1870, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v1873 = stablehlo.transpose %v1871, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v1874 = stablehlo.convolution(%v1872, %v1873)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1875 = stablehlo.transpose %v1874, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1876 = stablehlo.reshape %v798 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1877 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1878 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v1879 = stablehlo.reduce(%v1876 init: %v1877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1880 = stablehlo.broadcast_in_dim %v1879, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1881 = stablehlo.divide %v1880, %v1878 : tensor<128x256x14x14xf32>
    %v1882 = stablehlo.subtract %v1876, %v1881 : tensor<128x256x14x14xf32>
    %v1883 = stablehlo.multiply %v1882, %v1882 : tensor<128x256x14x14xf32>
    %v1884 = stablehlo.reduce(%v1883 init: %v1877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1885 = stablehlo.broadcast_in_dim %v1884, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1886 = stablehlo.divide %v1885, %v1878 : tensor<128x256x14x14xf32>
    %v1887 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1888 = stablehlo.add %v1886, %v1887 : tensor<128x256x14x14xf32>
    %v1889 = stablehlo.rsqrt %v1888 : tensor<128x256x14x14xf32>
    %v1890 = stablehlo.multiply %v1882, %v1889 : tensor<128x256x14x14xf32>
    %v1891 = stablehlo.reshape %v1762 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1892 = stablehlo.multiply %v1891, %v1890 : tensor<128x256x14x14xf32>
    %v1893 = stablehlo.reduce(%v1892 init: %v1877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1894 = stablehlo.reshape %v1762 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1896 = stablehlo.reduce(%v1894 init: %v1895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1897 = stablehlo.reshape %v1842 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1898 = stablehlo.reshape %v760 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1899 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v1900 = stablehlo.compare GT, %v1898, %v1899 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v1901 = stablehlo.select %v1900, %v1897, %v1899 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v1902 = stablehlo.reshape %v1901 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1903 = stablehlo.reshape %v736 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1904 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1905 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v1906 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1907 = stablehlo.reduce(%v1903 init: %v1904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1908 = stablehlo.broadcast_in_dim %v1907, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1909 = stablehlo.divide %v1908, %v1905 : tensor<128x256x14x14xf32>
    %v1910 = stablehlo.subtract %v1903, %v1909 : tensor<128x256x14x14xf32>
    %v1911 = stablehlo.multiply %v1910, %v1910 : tensor<128x256x14x14xf32>
    %v1912 = stablehlo.reduce(%v1911 init: %v1904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1913 = stablehlo.broadcast_in_dim %v1912, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1914 = stablehlo.divide %v1913, %v1905 : tensor<128x256x14x14xf32>
    %v1915 = stablehlo.add %v1914, %v1906 : tensor<128x256x14x14xf32>
    %v1916 = stablehlo.rsqrt %v1915 : tensor<128x256x14x14xf32>
    %v1917 = stablehlo.multiply %v1910, %v1916 : tensor<128x256x14x14xf32>
    %v1918 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1919 = stablehlo.reshape %v1902 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1920 = stablehlo.multiply %v1918, %v1919 : tensor<128x256x14x14xf32>
    %v1921 = stablehlo.reduce(%v1920 init: %v1904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1922 = stablehlo.broadcast_in_dim %v1921, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1923 = stablehlo.multiply %v1917, %v1920 : tensor<128x256x14x14xf32>
    %v1924 = stablehlo.reduce(%v1923 init: %v1904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1925 = stablehlo.broadcast_in_dim %v1924, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1926 = stablehlo.multiply %v1920, %v1905 : tensor<128x256x14x14xf32>
    %v1927 = stablehlo.subtract %v1926, %v1922 : tensor<128x256x14x14xf32>
    %v1928 = stablehlo.multiply %v1917, %v1925 : tensor<128x256x14x14xf32>
    %v1929 = stablehlo.subtract %v1927, %v1928 : tensor<128x256x14x14xf32>
    %v1930 = stablehlo.divide %v1916, %v1905 : tensor<128x256x14x14xf32>
    %v1931 = stablehlo.multiply %v1930, %v1929 : tensor<128x256x14x14xf32>
    %v1932 = stablehlo.reshape %v1931 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1933 = stablehlo.reshape %v1932 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1934 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1935 = stablehlo.transpose %v1934, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1936 = stablehlo.convolution(%v1933, %v1935)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v1937 = stablehlo.reshape %v1936 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1938 = stablehlo.reshape %v1937 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1939 = stablehlo.reshape %v727 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1940 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v1941 = stablehlo.compare GT, %v1939, %v1940 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v1942 = stablehlo.select %v1941, %v1938, %v1940 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v1943 = stablehlo.reshape %v1942 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1944 = stablehlo.reshape %v707 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1946 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v1947 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1948 = stablehlo.reduce(%v1944 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1949 = stablehlo.broadcast_in_dim %v1948, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1950 = stablehlo.divide %v1949, %v1946 : tensor<128x256x14x14xf32>
    %v1951 = stablehlo.subtract %v1944, %v1950 : tensor<128x256x14x14xf32>
    %v1952 = stablehlo.multiply %v1951, %v1951 : tensor<128x256x14x14xf32>
    %v1953 = stablehlo.reduce(%v1952 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1954 = stablehlo.broadcast_in_dim %v1953, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1955 = stablehlo.divide %v1954, %v1946 : tensor<128x256x14x14xf32>
    %v1956 = stablehlo.add %v1955, %v1947 : tensor<128x256x14x14xf32>
    %v1957 = stablehlo.rsqrt %v1956 : tensor<128x256x14x14xf32>
    %v1958 = stablehlo.multiply %v1951, %v1957 : tensor<128x256x14x14xf32>
    %v1959 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1960 = stablehlo.reshape %v1943 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1961 = stablehlo.multiply %v1959, %v1960 : tensor<128x256x14x14xf32>
    %v1962 = stablehlo.reduce(%v1961 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1963 = stablehlo.broadcast_in_dim %v1962, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1964 = stablehlo.multiply %v1958, %v1961 : tensor<128x256x14x14xf32>
    %v1965 = stablehlo.reduce(%v1964 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1966 = stablehlo.broadcast_in_dim %v1965, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1967 = stablehlo.multiply %v1961, %v1946 : tensor<128x256x14x14xf32>
    %v1968 = stablehlo.subtract %v1967, %v1963 : tensor<128x256x14x14xf32>
    %v1969 = stablehlo.multiply %v1958, %v1966 : tensor<128x256x14x14xf32>
    %v1970 = stablehlo.subtract %v1968, %v1969 : tensor<128x256x14x14xf32>
    %v1971 = stablehlo.divide %v1957, %v1946 : tensor<128x256x14x14xf32>
    %v1972 = stablehlo.multiply %v1971, %v1970 : tensor<128x256x14x14xf32>
    %v1973 = stablehlo.reshape %v1972 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1974 = stablehlo.reshape %v1973 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1975 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1976 = stablehlo.transpose %v1975, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1977 = stablehlo.convolution(%v1974, %v1976)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v1978 = stablehlo.reshape %v1977 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1979 = stablehlo.reshape %v1978 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1980 = stablehlo.reshape %v1902 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1981 = stablehlo.add %v1979, %v1980 : tensor<128x256x14x14xf32>
    %v1982 = stablehlo.reshape %v1981 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1983 = stablehlo.reshape %v702 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1984 = stablehlo.reshape %v1973 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1985 = stablehlo.transpose %v1983, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v1986 = stablehlo.transpose %v1984, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v1987 = stablehlo.convolution(%v1985, %v1986)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1988 = stablehlo.transpose %v1987, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1989 = stablehlo.reshape %v707 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1991 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v1992 = stablehlo.reduce(%v1989 init: %v1990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1993 = stablehlo.broadcast_in_dim %v1992, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1994 = stablehlo.divide %v1993, %v1991 : tensor<128x256x14x14xf32>
    %v1995 = stablehlo.subtract %v1989, %v1994 : tensor<128x256x14x14xf32>
    %v1996 = stablehlo.multiply %v1995, %v1995 : tensor<128x256x14x14xf32>
    %v1997 = stablehlo.reduce(%v1996 init: %v1990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1998 = stablehlo.broadcast_in_dim %v1997, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1999 = stablehlo.divide %v1998, %v1991 : tensor<128x256x14x14xf32>
    %v2000 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2001 = stablehlo.add %v1999, %v2000 : tensor<128x256x14x14xf32>
    %v2002 = stablehlo.rsqrt %v2001 : tensor<128x256x14x14xf32>
    %v2003 = stablehlo.multiply %v1995, %v2002 : tensor<128x256x14x14xf32>
    %v2004 = stablehlo.reshape %v1943 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2005 = stablehlo.multiply %v2004, %v2003 : tensor<128x256x14x14xf32>
    %v2006 = stablehlo.reduce(%v2005 init: %v1990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2007 = stablehlo.reshape %v1943 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2008 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2009 = stablehlo.reduce(%v2007 init: %v2008) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2010 = stablehlo.reshape %v731 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2011 = stablehlo.reshape %v1932 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2012 = stablehlo.transpose %v2010, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2013 = stablehlo.transpose %v2011, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2014 = stablehlo.convolution(%v2012, %v2013)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2015 = stablehlo.transpose %v2014, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2016 = stablehlo.reshape %v736 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2017 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2018 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v2019 = stablehlo.reduce(%v2016 init: %v2017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2020 = stablehlo.broadcast_in_dim %v2019, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2021 = stablehlo.divide %v2020, %v2018 : tensor<128x256x14x14xf32>
    %v2022 = stablehlo.subtract %v2016, %v2021 : tensor<128x256x14x14xf32>
    %v2023 = stablehlo.multiply %v2022, %v2022 : tensor<128x256x14x14xf32>
    %v2024 = stablehlo.reduce(%v2023 init: %v2017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2025 = stablehlo.broadcast_in_dim %v2024, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2026 = stablehlo.divide %v2025, %v2018 : tensor<128x256x14x14xf32>
    %v2027 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2028 = stablehlo.add %v2026, %v2027 : tensor<128x256x14x14xf32>
    %v2029 = stablehlo.rsqrt %v2028 : tensor<128x256x14x14xf32>
    %v2030 = stablehlo.multiply %v2022, %v2029 : tensor<128x256x14x14xf32>
    %v2031 = stablehlo.reshape %v1902 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2032 = stablehlo.multiply %v2031, %v2030 : tensor<128x256x14x14xf32>
    %v2033 = stablehlo.reduce(%v2032 init: %v2017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2034 = stablehlo.reshape %v1902 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2035 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2036 = stablehlo.reduce(%v2034 init: %v2035) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2037 = stablehlo.reshape %v1982 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2038 = stablehlo.reshape %v698 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2039 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2040 = stablehlo.compare GT, %v2038, %v2039 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2041 = stablehlo.select %v2040, %v2037, %v2039 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2042 = stablehlo.reshape %v2041 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2043 = stablehlo.reshape %v674 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2045 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v2046 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2047 = stablehlo.reduce(%v2043 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2048 = stablehlo.broadcast_in_dim %v2047, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2049 = stablehlo.divide %v2048, %v2045 : tensor<128x256x14x14xf32>
    %v2050 = stablehlo.subtract %v2043, %v2049 : tensor<128x256x14x14xf32>
    %v2051 = stablehlo.multiply %v2050, %v2050 : tensor<128x256x14x14xf32>
    %v2052 = stablehlo.reduce(%v2051 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2053 = stablehlo.broadcast_in_dim %v2052, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2054 = stablehlo.divide %v2053, %v2045 : tensor<128x256x14x14xf32>
    %v2055 = stablehlo.add %v2054, %v2046 : tensor<128x256x14x14xf32>
    %v2056 = stablehlo.rsqrt %v2055 : tensor<128x256x14x14xf32>
    %v2057 = stablehlo.multiply %v2050, %v2056 : tensor<128x256x14x14xf32>
    %v2058 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2059 = stablehlo.reshape %v2042 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2060 = stablehlo.multiply %v2058, %v2059 : tensor<128x256x14x14xf32>
    %v2061 = stablehlo.reduce(%v2060 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2062 = stablehlo.broadcast_in_dim %v2061, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2063 = stablehlo.multiply %v2057, %v2060 : tensor<128x256x14x14xf32>
    %v2064 = stablehlo.reduce(%v2063 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2065 = stablehlo.broadcast_in_dim %v2064, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2066 = stablehlo.multiply %v2060, %v2045 : tensor<128x256x14x14xf32>
    %v2067 = stablehlo.subtract %v2066, %v2062 : tensor<128x256x14x14xf32>
    %v2068 = stablehlo.multiply %v2057, %v2065 : tensor<128x256x14x14xf32>
    %v2069 = stablehlo.subtract %v2067, %v2068 : tensor<128x256x14x14xf32>
    %v2070 = stablehlo.divide %v2056, %v2045 : tensor<128x256x14x14xf32>
    %v2071 = stablehlo.multiply %v2070, %v2069 : tensor<128x256x14x14xf32>
    %v2072 = stablehlo.reshape %v2071 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2073 = stablehlo.reshape %v2072 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2074 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2075 = stablehlo.transpose %v2074, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2076 = stablehlo.convolution(%v2073, %v2075)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2077 = stablehlo.reshape %v2076 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2078 = stablehlo.reshape %v2077 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2079 = stablehlo.reshape %v665 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2080 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2081 = stablehlo.compare GT, %v2079, %v2080 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2082 = stablehlo.select %v2081, %v2078, %v2080 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2083 = stablehlo.reshape %v2082 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2084 = stablehlo.reshape %v645 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2086 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v2087 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2088 = stablehlo.reduce(%v2084 init: %v2085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2089 = stablehlo.broadcast_in_dim %v2088, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2090 = stablehlo.divide %v2089, %v2086 : tensor<128x256x14x14xf32>
    %v2091 = stablehlo.subtract %v2084, %v2090 : tensor<128x256x14x14xf32>
    %v2092 = stablehlo.multiply %v2091, %v2091 : tensor<128x256x14x14xf32>
    %v2093 = stablehlo.reduce(%v2092 init: %v2085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2094 = stablehlo.broadcast_in_dim %v2093, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2095 = stablehlo.divide %v2094, %v2086 : tensor<128x256x14x14xf32>
    %v2096 = stablehlo.add %v2095, %v2087 : tensor<128x256x14x14xf32>
    %v2097 = stablehlo.rsqrt %v2096 : tensor<128x256x14x14xf32>
    %v2098 = stablehlo.multiply %v2091, %v2097 : tensor<128x256x14x14xf32>
    %v2099 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2100 = stablehlo.reshape %v2083 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2101 = stablehlo.multiply %v2099, %v2100 : tensor<128x256x14x14xf32>
    %v2102 = stablehlo.reduce(%v2101 init: %v2085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2103 = stablehlo.broadcast_in_dim %v2102, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2104 = stablehlo.multiply %v2098, %v2101 : tensor<128x256x14x14xf32>
    %v2105 = stablehlo.reduce(%v2104 init: %v2085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2106 = stablehlo.broadcast_in_dim %v2105, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2107 = stablehlo.multiply %v2101, %v2086 : tensor<128x256x14x14xf32>
    %v2108 = stablehlo.subtract %v2107, %v2103 : tensor<128x256x14x14xf32>
    %v2109 = stablehlo.multiply %v2098, %v2106 : tensor<128x256x14x14xf32>
    %v2110 = stablehlo.subtract %v2108, %v2109 : tensor<128x256x14x14xf32>
    %v2111 = stablehlo.divide %v2097, %v2086 : tensor<128x256x14x14xf32>
    %v2112 = stablehlo.multiply %v2111, %v2110 : tensor<128x256x14x14xf32>
    %v2113 = stablehlo.reshape %v2112 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2114 = stablehlo.reshape %v2113 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2115 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2116 = stablehlo.transpose %v2115, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2117 = stablehlo.convolution(%v2114, %v2116)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2118 = stablehlo.reshape %v2117 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2120 = stablehlo.reshape %v2042 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2121 = stablehlo.add %v2119, %v2120 : tensor<128x256x14x14xf32>
    %v2122 = stablehlo.reshape %v2121 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2123 = stablehlo.reshape %v640 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2124 = stablehlo.reshape %v2113 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2125 = stablehlo.transpose %v2123, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2126 = stablehlo.transpose %v2124, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2127 = stablehlo.convolution(%v2125, %v2126)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2128 = stablehlo.transpose %v2127, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2129 = stablehlo.reshape %v645 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2131 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v2132 = stablehlo.reduce(%v2129 init: %v2130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2133 = stablehlo.broadcast_in_dim %v2132, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2134 = stablehlo.divide %v2133, %v2131 : tensor<128x256x14x14xf32>
    %v2135 = stablehlo.subtract %v2129, %v2134 : tensor<128x256x14x14xf32>
    %v2136 = stablehlo.multiply %v2135, %v2135 : tensor<128x256x14x14xf32>
    %v2137 = stablehlo.reduce(%v2136 init: %v2130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2138 = stablehlo.broadcast_in_dim %v2137, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2139 = stablehlo.divide %v2138, %v2131 : tensor<128x256x14x14xf32>
    %v2140 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2141 = stablehlo.add %v2139, %v2140 : tensor<128x256x14x14xf32>
    %v2142 = stablehlo.rsqrt %v2141 : tensor<128x256x14x14xf32>
    %v2143 = stablehlo.multiply %v2135, %v2142 : tensor<128x256x14x14xf32>
    %v2144 = stablehlo.reshape %v2083 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2145 = stablehlo.multiply %v2144, %v2143 : tensor<128x256x14x14xf32>
    %v2146 = stablehlo.reduce(%v2145 init: %v2130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2147 = stablehlo.reshape %v2083 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2148 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2149 = stablehlo.reduce(%v2147 init: %v2148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2150 = stablehlo.reshape %v669 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2151 = stablehlo.reshape %v2072 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2152 = stablehlo.transpose %v2150, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2153 = stablehlo.transpose %v2151, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2154 = stablehlo.convolution(%v2152, %v2153)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2155 = stablehlo.transpose %v2154, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2156 = stablehlo.reshape %v674 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2157 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2158 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v2159 = stablehlo.reduce(%v2156 init: %v2157) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2160 = stablehlo.broadcast_in_dim %v2159, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2161 = stablehlo.divide %v2160, %v2158 : tensor<128x256x14x14xf32>
    %v2162 = stablehlo.subtract %v2156, %v2161 : tensor<128x256x14x14xf32>
    %v2163 = stablehlo.multiply %v2162, %v2162 : tensor<128x256x14x14xf32>
    %v2164 = stablehlo.reduce(%v2163 init: %v2157) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2165 = stablehlo.broadcast_in_dim %v2164, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2166 = stablehlo.divide %v2165, %v2158 : tensor<128x256x14x14xf32>
    %v2167 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2168 = stablehlo.add %v2166, %v2167 : tensor<128x256x14x14xf32>
    %v2169 = stablehlo.rsqrt %v2168 : tensor<128x256x14x14xf32>
    %v2170 = stablehlo.multiply %v2162, %v2169 : tensor<128x256x14x14xf32>
    %v2171 = stablehlo.reshape %v2042 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2172 = stablehlo.multiply %v2171, %v2170 : tensor<128x256x14x14xf32>
    %v2173 = stablehlo.reduce(%v2172 init: %v2157) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2174 = stablehlo.reshape %v2042 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2175 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2176 = stablehlo.reduce(%v2174 init: %v2175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2177 = stablehlo.reshape %v2122 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2178 = stablehlo.reshape %v636 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2179 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2180 = stablehlo.compare GT, %v2178, %v2179 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2181 = stablehlo.select %v2180, %v2177, %v2179 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2182 = stablehlo.reshape %v2181 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2183 = stablehlo.reshape %v612 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2184 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2185 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v2186 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2187 = stablehlo.reduce(%v2183 init: %v2184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2188 = stablehlo.broadcast_in_dim %v2187, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2189 = stablehlo.divide %v2188, %v2185 : tensor<128x256x14x14xf32>
    %v2190 = stablehlo.subtract %v2183, %v2189 : tensor<128x256x14x14xf32>
    %v2191 = stablehlo.multiply %v2190, %v2190 : tensor<128x256x14x14xf32>
    %v2192 = stablehlo.reduce(%v2191 init: %v2184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2193 = stablehlo.broadcast_in_dim %v2192, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2194 = stablehlo.divide %v2193, %v2185 : tensor<128x256x14x14xf32>
    %v2195 = stablehlo.add %v2194, %v2186 : tensor<128x256x14x14xf32>
    %v2196 = stablehlo.rsqrt %v2195 : tensor<128x256x14x14xf32>
    %v2197 = stablehlo.multiply %v2190, %v2196 : tensor<128x256x14x14xf32>
    %v2198 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2199 = stablehlo.reshape %v2182 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2200 = stablehlo.multiply %v2198, %v2199 : tensor<128x256x14x14xf32>
    %v2201 = stablehlo.reduce(%v2200 init: %v2184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2202 = stablehlo.broadcast_in_dim %v2201, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2203 = stablehlo.multiply %v2197, %v2200 : tensor<128x256x14x14xf32>
    %v2204 = stablehlo.reduce(%v2203 init: %v2184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2205 = stablehlo.broadcast_in_dim %v2204, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2206 = stablehlo.multiply %v2200, %v2185 : tensor<128x256x14x14xf32>
    %v2207 = stablehlo.subtract %v2206, %v2202 : tensor<128x256x14x14xf32>
    %v2208 = stablehlo.multiply %v2197, %v2205 : tensor<128x256x14x14xf32>
    %v2209 = stablehlo.subtract %v2207, %v2208 : tensor<128x256x14x14xf32>
    %v2210 = stablehlo.divide %v2196, %v2185 : tensor<128x256x14x14xf32>
    %v2211 = stablehlo.multiply %v2210, %v2209 : tensor<128x256x14x14xf32>
    %v2212 = stablehlo.reshape %v2211 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2213 = stablehlo.reshape %v2212 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2214 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2215 = stablehlo.transpose %v2214, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2216 = stablehlo.convolution(%v2213, %v2215)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2217 = stablehlo.reshape %v2216 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2218 = stablehlo.reshape %v2217 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2219 = stablehlo.reshape %v603 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2220 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2221 = stablehlo.compare GT, %v2219, %v2220 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2222 = stablehlo.select %v2221, %v2218, %v2220 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2223 = stablehlo.reshape %v2222 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2224 = stablehlo.reshape %v583 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2226 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v2227 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2228 = stablehlo.reduce(%v2224 init: %v2225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2229 = stablehlo.broadcast_in_dim %v2228, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2230 = stablehlo.divide %v2229, %v2226 : tensor<128x256x14x14xf32>
    %v2231 = stablehlo.subtract %v2224, %v2230 : tensor<128x256x14x14xf32>
    %v2232 = stablehlo.multiply %v2231, %v2231 : tensor<128x256x14x14xf32>
    %v2233 = stablehlo.reduce(%v2232 init: %v2225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2234 = stablehlo.broadcast_in_dim %v2233, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2235 = stablehlo.divide %v2234, %v2226 : tensor<128x256x14x14xf32>
    %v2236 = stablehlo.add %v2235, %v2227 : tensor<128x256x14x14xf32>
    %v2237 = stablehlo.rsqrt %v2236 : tensor<128x256x14x14xf32>
    %v2238 = stablehlo.multiply %v2231, %v2237 : tensor<128x256x14x14xf32>
    %v2239 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2240 = stablehlo.reshape %v2223 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2241 = stablehlo.multiply %v2239, %v2240 : tensor<128x256x14x14xf32>
    %v2242 = stablehlo.reduce(%v2241 init: %v2225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2243 = stablehlo.broadcast_in_dim %v2242, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2244 = stablehlo.multiply %v2238, %v2241 : tensor<128x256x14x14xf32>
    %v2245 = stablehlo.reduce(%v2244 init: %v2225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2246 = stablehlo.broadcast_in_dim %v2245, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2247 = stablehlo.multiply %v2241, %v2226 : tensor<128x256x14x14xf32>
    %v2248 = stablehlo.subtract %v2247, %v2243 : tensor<128x256x14x14xf32>
    %v2249 = stablehlo.multiply %v2238, %v2246 : tensor<128x256x14x14xf32>
    %v2250 = stablehlo.subtract %v2248, %v2249 : tensor<128x256x14x14xf32>
    %v2251 = stablehlo.divide %v2237, %v2226 : tensor<128x256x14x14xf32>
    %v2252 = stablehlo.multiply %v2251, %v2250 : tensor<128x256x14x14xf32>
    %v2253 = stablehlo.reshape %v2252 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2254 = stablehlo.reshape %v2253 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2255 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2256 = stablehlo.transpose %v2255, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2257 = stablehlo.convolution(%v2254, %v2256)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2258 = stablehlo.reshape %v2257 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2260 = stablehlo.reshape %v2182 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2261 = stablehlo.add %v2259, %v2260 : tensor<128x256x14x14xf32>
    %v2262 = stablehlo.reshape %v2261 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2263 = stablehlo.reshape %v578 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2264 = stablehlo.reshape %v2253 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2265 = stablehlo.transpose %v2263, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2266 = stablehlo.transpose %v2264, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2267 = stablehlo.convolution(%v2265, %v2266)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2268 = stablehlo.transpose %v2267, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2269 = stablehlo.reshape %v583 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2270 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2271 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v2272 = stablehlo.reduce(%v2269 init: %v2270) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2273 = stablehlo.broadcast_in_dim %v2272, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2274 = stablehlo.divide %v2273, %v2271 : tensor<128x256x14x14xf32>
    %v2275 = stablehlo.subtract %v2269, %v2274 : tensor<128x256x14x14xf32>
    %v2276 = stablehlo.multiply %v2275, %v2275 : tensor<128x256x14x14xf32>
    %v2277 = stablehlo.reduce(%v2276 init: %v2270) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2278 = stablehlo.broadcast_in_dim %v2277, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2279 = stablehlo.divide %v2278, %v2271 : tensor<128x256x14x14xf32>
    %v2280 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2281 = stablehlo.add %v2279, %v2280 : tensor<128x256x14x14xf32>
    %v2282 = stablehlo.rsqrt %v2281 : tensor<128x256x14x14xf32>
    %v2283 = stablehlo.multiply %v2275, %v2282 : tensor<128x256x14x14xf32>
    %v2284 = stablehlo.reshape %v2223 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2285 = stablehlo.multiply %v2284, %v2283 : tensor<128x256x14x14xf32>
    %v2286 = stablehlo.reduce(%v2285 init: %v2270) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2287 = stablehlo.reshape %v2223 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2289 = stablehlo.reduce(%v2287 init: %v2288) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2290 = stablehlo.reshape %v607 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2291 = stablehlo.reshape %v2212 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2292 = stablehlo.transpose %v2290, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2293 = stablehlo.transpose %v2291, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2294 = stablehlo.convolution(%v2292, %v2293)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2295 = stablehlo.transpose %v2294, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2296 = stablehlo.reshape %v612 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2298 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v2299 = stablehlo.reduce(%v2296 init: %v2297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2300 = stablehlo.broadcast_in_dim %v2299, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2301 = stablehlo.divide %v2300, %v2298 : tensor<128x256x14x14xf32>
    %v2302 = stablehlo.subtract %v2296, %v2301 : tensor<128x256x14x14xf32>
    %v2303 = stablehlo.multiply %v2302, %v2302 : tensor<128x256x14x14xf32>
    %v2304 = stablehlo.reduce(%v2303 init: %v2297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2305 = stablehlo.broadcast_in_dim %v2304, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2306 = stablehlo.divide %v2305, %v2298 : tensor<128x256x14x14xf32>
    %v2307 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2308 = stablehlo.add %v2306, %v2307 : tensor<128x256x14x14xf32>
    %v2309 = stablehlo.rsqrt %v2308 : tensor<128x256x14x14xf32>
    %v2310 = stablehlo.multiply %v2302, %v2309 : tensor<128x256x14x14xf32>
    %v2311 = stablehlo.reshape %v2182 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2312 = stablehlo.multiply %v2311, %v2310 : tensor<128x256x14x14xf32>
    %v2313 = stablehlo.reduce(%v2312 init: %v2297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2314 = stablehlo.reshape %v2182 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2316 = stablehlo.reduce(%v2314 init: %v2315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2317 = stablehlo.reshape %v2262 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2318 = stablehlo.reshape %v574 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2319 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2320 = stablehlo.compare GT, %v2318, %v2319 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2321 = stablehlo.select %v2320, %v2317, %v2319 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2322 = stablehlo.reshape %v2321 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2323 = stablehlo.reshape %v525 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2324 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2325 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v2326 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2327 = stablehlo.reduce(%v2323 init: %v2324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2328 = stablehlo.broadcast_in_dim %v2327, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2329 = stablehlo.divide %v2328, %v2325 : tensor<128x256x14x14xf32>
    %v2330 = stablehlo.subtract %v2323, %v2329 : tensor<128x256x14x14xf32>
    %v2331 = stablehlo.multiply %v2330, %v2330 : tensor<128x256x14x14xf32>
    %v2332 = stablehlo.reduce(%v2331 init: %v2324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2333 = stablehlo.broadcast_in_dim %v2332, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2334 = stablehlo.divide %v2333, %v2325 : tensor<128x256x14x14xf32>
    %v2335 = stablehlo.add %v2334, %v2326 : tensor<128x256x14x14xf32>
    %v2336 = stablehlo.rsqrt %v2335 : tensor<128x256x14x14xf32>
    %v2337 = stablehlo.multiply %v2330, %v2336 : tensor<128x256x14x14xf32>
    %v2338 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2339 = stablehlo.reshape %v2322 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2340 = stablehlo.multiply %v2338, %v2339 : tensor<128x256x14x14xf32>
    %v2341 = stablehlo.reduce(%v2340 init: %v2324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2342 = stablehlo.broadcast_in_dim %v2341, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2343 = stablehlo.multiply %v2337, %v2340 : tensor<128x256x14x14xf32>
    %v2344 = stablehlo.reduce(%v2343 init: %v2324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2345 = stablehlo.broadcast_in_dim %v2344, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2346 = stablehlo.multiply %v2340, %v2325 : tensor<128x256x14x14xf32>
    %v2347 = stablehlo.subtract %v2346, %v2342 : tensor<128x256x14x14xf32>
    %v2348 = stablehlo.multiply %v2337, %v2345 : tensor<128x256x14x14xf32>
    %v2349 = stablehlo.subtract %v2347, %v2348 : tensor<128x256x14x14xf32>
    %v2350 = stablehlo.divide %v2336, %v2325 : tensor<128x256x14x14xf32>
    %v2351 = stablehlo.multiply %v2350, %v2349 : tensor<128x256x14x14xf32>
    %v2352 = stablehlo.reshape %v2351 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2353 = stablehlo.reshape %v2352 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2354 = stablehlo.reverse %d3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2355 = stablehlo.transpose %v2354, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2356 = stablehlo.convolution(%v2353, %v2355)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2357 = stablehlo.reshape %v2356 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2358 = stablehlo.reshape %v2357 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2359 = stablehlo.reshape %v516 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2360 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2361 = stablehlo.compare GT, %v2359, %v2360 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2362 = stablehlo.select %v2361, %v2358, %v2360 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2363 = stablehlo.reshape %v2362 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2364 = stablehlo.reshape %v496 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2366 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v2367 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2368 = stablehlo.reduce(%v2364 init: %v2365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2369 = stablehlo.broadcast_in_dim %v2368, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2370 = stablehlo.divide %v2369, %v2366 : tensor<128x256x14x14xf32>
    %v2371 = stablehlo.subtract %v2364, %v2370 : tensor<128x256x14x14xf32>
    %v2372 = stablehlo.multiply %v2371, %v2371 : tensor<128x256x14x14xf32>
    %v2373 = stablehlo.reduce(%v2372 init: %v2365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2374 = stablehlo.broadcast_in_dim %v2373, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2375 = stablehlo.divide %v2374, %v2366 : tensor<128x256x14x14xf32>
    %v2376 = stablehlo.add %v2375, %v2367 : tensor<128x256x14x14xf32>
    %v2377 = stablehlo.rsqrt %v2376 : tensor<128x256x14x14xf32>
    %v2378 = stablehlo.multiply %v2371, %v2377 : tensor<128x256x14x14xf32>
    %v2379 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2380 = stablehlo.reshape %v2363 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2381 = stablehlo.multiply %v2379, %v2380 : tensor<128x256x14x14xf32>
    %v2382 = stablehlo.reduce(%v2381 init: %v2365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2383 = stablehlo.broadcast_in_dim %v2382, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2384 = stablehlo.multiply %v2378, %v2381 : tensor<128x256x14x14xf32>
    %v2385 = stablehlo.reduce(%v2384 init: %v2365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2386 = stablehlo.broadcast_in_dim %v2385, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2387 = stablehlo.multiply %v2381, %v2366 : tensor<128x256x14x14xf32>
    %v2388 = stablehlo.subtract %v2387, %v2383 : tensor<128x256x14x14xf32>
    %v2389 = stablehlo.multiply %v2378, %v2386 : tensor<128x256x14x14xf32>
    %v2390 = stablehlo.subtract %v2388, %v2389 : tensor<128x256x14x14xf32>
    %v2391 = stablehlo.divide %v2377, %v2366 : tensor<128x256x14x14xf32>
    %v2392 = stablehlo.multiply %v2391, %v2390 : tensor<128x256x14x14xf32>
    %v2393 = stablehlo.reshape %v2392 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2394 = stablehlo.reshape %v2393 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2395 = stablehlo.reverse %d3W1, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v2396 = stablehlo.transpose %v2395, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2397 = stablehlo.convolution(%v2394, %v2396)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<128x256x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v2398 = stablehlo.reshape %v2397 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2399 = stablehlo.reshape %v550 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2401 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v2402 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2403 = stablehlo.reduce(%v2399 init: %v2400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2404 = stablehlo.broadcast_in_dim %v2403, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2405 = stablehlo.divide %v2404, %v2401 : tensor<128x256x14x14xf32>
    %v2406 = stablehlo.subtract %v2399, %v2405 : tensor<128x256x14x14xf32>
    %v2407 = stablehlo.multiply %v2406, %v2406 : tensor<128x256x14x14xf32>
    %v2408 = stablehlo.reduce(%v2407 init: %v2400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2409 = stablehlo.broadcast_in_dim %v2408, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2410 = stablehlo.divide %v2409, %v2401 : tensor<128x256x14x14xf32>
    %v2411 = stablehlo.add %v2410, %v2402 : tensor<128x256x14x14xf32>
    %v2412 = stablehlo.rsqrt %v2411 : tensor<128x256x14x14xf32>
    %v2413 = stablehlo.multiply %v2406, %v2412 : tensor<128x256x14x14xf32>
    %v2414 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2415 = stablehlo.reshape %v2322 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2416 = stablehlo.multiply %v2414, %v2415 : tensor<128x256x14x14xf32>
    %v2417 = stablehlo.reduce(%v2416 init: %v2400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2418 = stablehlo.broadcast_in_dim %v2417, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2419 = stablehlo.multiply %v2413, %v2416 : tensor<128x256x14x14xf32>
    %v2420 = stablehlo.reduce(%v2419 init: %v2400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2421 = stablehlo.broadcast_in_dim %v2420, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2422 = stablehlo.multiply %v2416, %v2401 : tensor<128x256x14x14xf32>
    %v2423 = stablehlo.subtract %v2422, %v2418 : tensor<128x256x14x14xf32>
    %v2424 = stablehlo.multiply %v2413, %v2421 : tensor<128x256x14x14xf32>
    %v2425 = stablehlo.subtract %v2423, %v2424 : tensor<128x256x14x14xf32>
    %v2426 = stablehlo.divide %v2412, %v2401 : tensor<128x256x14x14xf32>
    %v2427 = stablehlo.multiply %v2426, %v2425 : tensor<128x256x14x14xf32>
    %v2428 = stablehlo.reshape %v2427 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2429 = stablehlo.reshape %v2428 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2430 = stablehlo.reverse %d3Wp, dims = [2, 3] : tensor<256x128x1x1xf32>
    %v2431 = stablehlo.transpose %v2430, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v2432 = stablehlo.convolution(%v2429, %v2431)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<128x256x1x1xf32>) -> tensor<128x128x28x28xf32>
    %v2433 = stablehlo.reshape %v2432 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2434 = stablehlo.reshape %v2398 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2435 = stablehlo.reshape %v2433 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2436 = stablehlo.add %v2434, %v2435 : tensor<128x128x28x28xf32>
    %v2437 = stablehlo.reshape %v2436 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2438 = stablehlo.reshape %v491 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2439 = stablehlo.reshape %v2393 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2440 = stablehlo.transpose %v2438, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v2442 = stablehlo.transpose %v2439, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2443 = stablehlo.convolution(%v2440, %v2442)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<256x128x14x14xf32>) -> tensor<128x256x3x3xf32>
    %v2444 = stablehlo.transpose %v2443, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2445 = stablehlo.reshape %v496 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2446 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2447 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v2448 = stablehlo.reduce(%v2445 init: %v2446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2449 = stablehlo.broadcast_in_dim %v2448, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2450 = stablehlo.divide %v2449, %v2447 : tensor<128x256x14x14xf32>
    %v2451 = stablehlo.subtract %v2445, %v2450 : tensor<128x256x14x14xf32>
    %v2452 = stablehlo.multiply %v2451, %v2451 : tensor<128x256x14x14xf32>
    %v2453 = stablehlo.reduce(%v2452 init: %v2446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2454 = stablehlo.broadcast_in_dim %v2453, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2455 = stablehlo.divide %v2454, %v2447 : tensor<128x256x14x14xf32>
    %v2456 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2457 = stablehlo.add %v2455, %v2456 : tensor<128x256x14x14xf32>
    %v2458 = stablehlo.rsqrt %v2457 : tensor<128x256x14x14xf32>
    %v2459 = stablehlo.multiply %v2451, %v2458 : tensor<128x256x14x14xf32>
    %v2460 = stablehlo.reshape %v2363 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2461 = stablehlo.multiply %v2460, %v2459 : tensor<128x256x14x14xf32>
    %v2462 = stablehlo.reduce(%v2461 init: %v2446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2463 = stablehlo.reshape %v2363 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2464 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2465 = stablehlo.reduce(%v2463 init: %v2464) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2466 = stablehlo.reshape %v520 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2467 = stablehlo.reshape %v2352 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2468 = stablehlo.transpose %v2466, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2469 = stablehlo.transpose %v2467, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2470 = stablehlo.convolution(%v2468, %v2469)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2471 = stablehlo.transpose %v2470, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2472 = stablehlo.reshape %v525 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2474 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v2475 = stablehlo.reduce(%v2472 init: %v2473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2476 = stablehlo.broadcast_in_dim %v2475, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2477 = stablehlo.divide %v2476, %v2474 : tensor<128x256x14x14xf32>
    %v2478 = stablehlo.subtract %v2472, %v2477 : tensor<128x256x14x14xf32>
    %v2479 = stablehlo.multiply %v2478, %v2478 : tensor<128x256x14x14xf32>
    %v2480 = stablehlo.reduce(%v2479 init: %v2473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2481 = stablehlo.broadcast_in_dim %v2480, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2482 = stablehlo.divide %v2481, %v2474 : tensor<128x256x14x14xf32>
    %v2483 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2484 = stablehlo.add %v2482, %v2483 : tensor<128x256x14x14xf32>
    %v2485 = stablehlo.rsqrt %v2484 : tensor<128x256x14x14xf32>
    %v2486 = stablehlo.multiply %v2478, %v2485 : tensor<128x256x14x14xf32>
    %v2487 = stablehlo.reshape %v2322 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2488 = stablehlo.multiply %v2487, %v2486 : tensor<128x256x14x14xf32>
    %v2489 = stablehlo.reduce(%v2488 init: %v2473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2490 = stablehlo.reshape %v2322 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2492 = stablehlo.reduce(%v2490 init: %v2491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2493 = stablehlo.reshape %v491 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2494 = stablehlo.reshape %v2428 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2495 = stablehlo.transpose %v2493, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v2497 = stablehlo.transpose %v2494, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2496 = stablehlo.slice %v2495 [0:128, 0:128, 0:27, 0:27] : (tensor<128x128x28x28xf32>) -> tensor<128x128x27x27xf32>
    %v2498 = stablehlo.convolution(%v2496, %v2497)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x27x27xf32>, tensor<256x128x14x14xf32>) -> tensor<128x256x1x1xf32>
    %v2499 = stablehlo.transpose %v2498, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v2500 = stablehlo.reshape %v550 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2501 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2502 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v2503 = stablehlo.reduce(%v2500 init: %v2501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2504 = stablehlo.broadcast_in_dim %v2503, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2505 = stablehlo.divide %v2504, %v2502 : tensor<128x256x14x14xf32>
    %v2506 = stablehlo.subtract %v2500, %v2505 : tensor<128x256x14x14xf32>
    %v2507 = stablehlo.multiply %v2506, %v2506 : tensor<128x256x14x14xf32>
    %v2508 = stablehlo.reduce(%v2507 init: %v2501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2509 = stablehlo.broadcast_in_dim %v2508, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2510 = stablehlo.divide %v2509, %v2502 : tensor<128x256x14x14xf32>
    %v2511 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2512 = stablehlo.add %v2510, %v2511 : tensor<128x256x14x14xf32>
    %v2513 = stablehlo.rsqrt %v2512 : tensor<128x256x14x14xf32>
    %v2514 = stablehlo.multiply %v2506, %v2513 : tensor<128x256x14x14xf32>
    %v2515 = stablehlo.reshape %v2322 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2516 = stablehlo.multiply %v2515, %v2514 : tensor<128x256x14x14xf32>
    %v2517 = stablehlo.reduce(%v2516 init: %v2501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2518 = stablehlo.reshape %v2322 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2519 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2520 = stablehlo.reduce(%v2518 init: %v2519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2521 = stablehlo.reshape %v2437 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2522 = stablehlo.reshape %v487 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2523 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v2524 = stablehlo.compare GT, %v2522, %v2523 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v2525 = stablehlo.select %v2524, %v2521, %v2523 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v2526 = stablehlo.reshape %v2525 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2527 = stablehlo.reshape %v463 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2529 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v2530 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v2531 = stablehlo.reduce(%v2527 init: %v2528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2532 = stablehlo.broadcast_in_dim %v2531, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2533 = stablehlo.divide %v2532, %v2529 : tensor<128x128x28x28xf32>
    %v2534 = stablehlo.subtract %v2527, %v2533 : tensor<128x128x28x28xf32>
    %v2535 = stablehlo.multiply %v2534, %v2534 : tensor<128x128x28x28xf32>
    %v2536 = stablehlo.reduce(%v2535 init: %v2528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2537 = stablehlo.broadcast_in_dim %v2536, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2538 = stablehlo.divide %v2537, %v2529 : tensor<128x128x28x28xf32>
    %v2539 = stablehlo.add %v2538, %v2530 : tensor<128x128x28x28xf32>
    %v2540 = stablehlo.rsqrt %v2539 : tensor<128x128x28x28xf32>
    %v2541 = stablehlo.multiply %v2534, %v2540 : tensor<128x128x28x28xf32>
    %v2542 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2543 = stablehlo.reshape %v2526 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2544 = stablehlo.multiply %v2542, %v2543 : tensor<128x128x28x28xf32>
    %v2545 = stablehlo.reduce(%v2544 init: %v2528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2546 = stablehlo.broadcast_in_dim %v2545, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2547 = stablehlo.multiply %v2541, %v2544 : tensor<128x128x28x28xf32>
    %v2548 = stablehlo.reduce(%v2547 init: %v2528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2549 = stablehlo.broadcast_in_dim %v2548, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2550 = stablehlo.multiply %v2544, %v2529 : tensor<128x128x28x28xf32>
    %v2551 = stablehlo.subtract %v2550, %v2546 : tensor<128x128x28x28xf32>
    %v2552 = stablehlo.multiply %v2541, %v2549 : tensor<128x128x28x28xf32>
    %v2553 = stablehlo.subtract %v2551, %v2552 : tensor<128x128x28x28xf32>
    %v2554 = stablehlo.divide %v2540, %v2529 : tensor<128x128x28x28xf32>
    %v2555 = stablehlo.multiply %v2554, %v2553 : tensor<128x128x28x28xf32>
    %v2556 = stablehlo.reshape %v2555 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2557 = stablehlo.reshape %v2556 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2558 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2559 = stablehlo.transpose %v2558, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2560 = stablehlo.convolution(%v2557, %v2559)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v2561 = stablehlo.reshape %v2560 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2562 = stablehlo.reshape %v2561 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2563 = stablehlo.reshape %v454 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2564 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v2565 = stablehlo.compare GT, %v2563, %v2564 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v2566 = stablehlo.select %v2565, %v2562, %v2564 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v2567 = stablehlo.reshape %v2566 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2568 = stablehlo.reshape %v434 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2570 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v2571 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v2572 = stablehlo.reduce(%v2568 init: %v2569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2573 = stablehlo.broadcast_in_dim %v2572, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2574 = stablehlo.divide %v2573, %v2570 : tensor<128x128x28x28xf32>
    %v2575 = stablehlo.subtract %v2568, %v2574 : tensor<128x128x28x28xf32>
    %v2576 = stablehlo.multiply %v2575, %v2575 : tensor<128x128x28x28xf32>
    %v2577 = stablehlo.reduce(%v2576 init: %v2569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2578 = stablehlo.broadcast_in_dim %v2577, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2579 = stablehlo.divide %v2578, %v2570 : tensor<128x128x28x28xf32>
    %v2580 = stablehlo.add %v2579, %v2571 : tensor<128x128x28x28xf32>
    %v2581 = stablehlo.rsqrt %v2580 : tensor<128x128x28x28xf32>
    %v2582 = stablehlo.multiply %v2575, %v2581 : tensor<128x128x28x28xf32>
    %v2583 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2584 = stablehlo.reshape %v2567 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2585 = stablehlo.multiply %v2583, %v2584 : tensor<128x128x28x28xf32>
    %v2586 = stablehlo.reduce(%v2585 init: %v2569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2587 = stablehlo.broadcast_in_dim %v2586, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2588 = stablehlo.multiply %v2582, %v2585 : tensor<128x128x28x28xf32>
    %v2589 = stablehlo.reduce(%v2588 init: %v2569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2590 = stablehlo.broadcast_in_dim %v2589, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2591 = stablehlo.multiply %v2585, %v2570 : tensor<128x128x28x28xf32>
    %v2592 = stablehlo.subtract %v2591, %v2587 : tensor<128x128x28x28xf32>
    %v2593 = stablehlo.multiply %v2582, %v2590 : tensor<128x128x28x28xf32>
    %v2594 = stablehlo.subtract %v2592, %v2593 : tensor<128x128x28x28xf32>
    %v2595 = stablehlo.divide %v2581, %v2570 : tensor<128x128x28x28xf32>
    %v2596 = stablehlo.multiply %v2595, %v2594 : tensor<128x128x28x28xf32>
    %v2597 = stablehlo.reshape %v2596 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2598 = stablehlo.reshape %v2597 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2599 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2600 = stablehlo.transpose %v2599, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2601 = stablehlo.convolution(%v2598, %v2600)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v2602 = stablehlo.reshape %v2601 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2603 = stablehlo.reshape %v2602 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2604 = stablehlo.reshape %v2526 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2605 = stablehlo.add %v2603, %v2604 : tensor<128x128x28x28xf32>
    %v2606 = stablehlo.reshape %v2605 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2607 = stablehlo.reshape %v429 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2608 = stablehlo.reshape %v2597 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2609 = stablehlo.transpose %v2607, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v2610 = stablehlo.transpose %v2608, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v2611 = stablehlo.convolution(%v2609, %v2610)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2612 = stablehlo.transpose %v2611, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2613 = stablehlo.reshape %v434 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2615 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v2616 = stablehlo.reduce(%v2613 init: %v2614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2617 = stablehlo.broadcast_in_dim %v2616, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2618 = stablehlo.divide %v2617, %v2615 : tensor<128x128x28x28xf32>
    %v2619 = stablehlo.subtract %v2613, %v2618 : tensor<128x128x28x28xf32>
    %v2620 = stablehlo.multiply %v2619, %v2619 : tensor<128x128x28x28xf32>
    %v2621 = stablehlo.reduce(%v2620 init: %v2614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2622 = stablehlo.broadcast_in_dim %v2621, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2623 = stablehlo.divide %v2622, %v2615 : tensor<128x128x28x28xf32>
    %v2624 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v2625 = stablehlo.add %v2623, %v2624 : tensor<128x128x28x28xf32>
    %v2626 = stablehlo.rsqrt %v2625 : tensor<128x128x28x28xf32>
    %v2627 = stablehlo.multiply %v2619, %v2626 : tensor<128x128x28x28xf32>
    %v2628 = stablehlo.reshape %v2567 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2629 = stablehlo.multiply %v2628, %v2627 : tensor<128x128x28x28xf32>
    %v2630 = stablehlo.reduce(%v2629 init: %v2614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2631 = stablehlo.reshape %v2567 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2632 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2633 = stablehlo.reduce(%v2631 init: %v2632) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2634 = stablehlo.reshape %v458 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2635 = stablehlo.reshape %v2556 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2636 = stablehlo.transpose %v2634, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v2637 = stablehlo.transpose %v2635, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v2638 = stablehlo.convolution(%v2636, %v2637)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2639 = stablehlo.transpose %v2638, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2640 = stablehlo.reshape %v463 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2642 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v2643 = stablehlo.reduce(%v2640 init: %v2641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2644 = stablehlo.broadcast_in_dim %v2643, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2645 = stablehlo.divide %v2644, %v2642 : tensor<128x128x28x28xf32>
    %v2646 = stablehlo.subtract %v2640, %v2645 : tensor<128x128x28x28xf32>
    %v2647 = stablehlo.multiply %v2646, %v2646 : tensor<128x128x28x28xf32>
    %v2648 = stablehlo.reduce(%v2647 init: %v2641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2649 = stablehlo.broadcast_in_dim %v2648, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2650 = stablehlo.divide %v2649, %v2642 : tensor<128x128x28x28xf32>
    %v2651 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v2652 = stablehlo.add %v2650, %v2651 : tensor<128x128x28x28xf32>
    %v2653 = stablehlo.rsqrt %v2652 : tensor<128x128x28x28xf32>
    %v2654 = stablehlo.multiply %v2646, %v2653 : tensor<128x128x28x28xf32>
    %v2655 = stablehlo.reshape %v2526 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2656 = stablehlo.multiply %v2655, %v2654 : tensor<128x128x28x28xf32>
    %v2657 = stablehlo.reduce(%v2656 init: %v2641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2658 = stablehlo.reshape %v2526 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2660 = stablehlo.reduce(%v2658 init: %v2659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2661 = stablehlo.reshape %v2606 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2662 = stablehlo.reshape %v425 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2663 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v2664 = stablehlo.compare GT, %v2662, %v2663 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v2665 = stablehlo.select %v2664, %v2661, %v2663 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v2666 = stablehlo.reshape %v2665 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2667 = stablehlo.reshape %v401 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2668 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2669 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v2670 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v2671 = stablehlo.reduce(%v2667 init: %v2668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2672 = stablehlo.broadcast_in_dim %v2671, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2673 = stablehlo.divide %v2672, %v2669 : tensor<128x128x28x28xf32>
    %v2674 = stablehlo.subtract %v2667, %v2673 : tensor<128x128x28x28xf32>
    %v2675 = stablehlo.multiply %v2674, %v2674 : tensor<128x128x28x28xf32>
    %v2676 = stablehlo.reduce(%v2675 init: %v2668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2677 = stablehlo.broadcast_in_dim %v2676, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2678 = stablehlo.divide %v2677, %v2669 : tensor<128x128x28x28xf32>
    %v2679 = stablehlo.add %v2678, %v2670 : tensor<128x128x28x28xf32>
    %v2680 = stablehlo.rsqrt %v2679 : tensor<128x128x28x28xf32>
    %v2681 = stablehlo.multiply %v2674, %v2680 : tensor<128x128x28x28xf32>
    %v2682 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2683 = stablehlo.reshape %v2666 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2684 = stablehlo.multiply %v2682, %v2683 : tensor<128x128x28x28xf32>
    %v2685 = stablehlo.reduce(%v2684 init: %v2668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2686 = stablehlo.broadcast_in_dim %v2685, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2687 = stablehlo.multiply %v2681, %v2684 : tensor<128x128x28x28xf32>
    %v2688 = stablehlo.reduce(%v2687 init: %v2668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2689 = stablehlo.broadcast_in_dim %v2688, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2690 = stablehlo.multiply %v2684, %v2669 : tensor<128x128x28x28xf32>
    %v2691 = stablehlo.subtract %v2690, %v2686 : tensor<128x128x28x28xf32>
    %v2692 = stablehlo.multiply %v2681, %v2689 : tensor<128x128x28x28xf32>
    %v2693 = stablehlo.subtract %v2691, %v2692 : tensor<128x128x28x28xf32>
    %v2694 = stablehlo.divide %v2680, %v2669 : tensor<128x128x28x28xf32>
    %v2695 = stablehlo.multiply %v2694, %v2693 : tensor<128x128x28x28xf32>
    %v2696 = stablehlo.reshape %v2695 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2697 = stablehlo.reshape %v2696 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2698 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2699 = stablehlo.transpose %v2698, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2700 = stablehlo.convolution(%v2697, %v2699)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v2701 = stablehlo.reshape %v2700 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2702 = stablehlo.reshape %v2701 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2703 = stablehlo.reshape %v392 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2704 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v2705 = stablehlo.compare GT, %v2703, %v2704 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v2706 = stablehlo.select %v2705, %v2702, %v2704 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v2707 = stablehlo.reshape %v2706 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2708 = stablehlo.reshape %v372 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2710 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v2711 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v2712 = stablehlo.reduce(%v2708 init: %v2709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2713 = stablehlo.broadcast_in_dim %v2712, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2714 = stablehlo.divide %v2713, %v2710 : tensor<128x128x28x28xf32>
    %v2715 = stablehlo.subtract %v2708, %v2714 : tensor<128x128x28x28xf32>
    %v2716 = stablehlo.multiply %v2715, %v2715 : tensor<128x128x28x28xf32>
    %v2717 = stablehlo.reduce(%v2716 init: %v2709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2718 = stablehlo.broadcast_in_dim %v2717, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2719 = stablehlo.divide %v2718, %v2710 : tensor<128x128x28x28xf32>
    %v2720 = stablehlo.add %v2719, %v2711 : tensor<128x128x28x28xf32>
    %v2721 = stablehlo.rsqrt %v2720 : tensor<128x128x28x28xf32>
    %v2722 = stablehlo.multiply %v2715, %v2721 : tensor<128x128x28x28xf32>
    %v2723 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2724 = stablehlo.reshape %v2707 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2725 = stablehlo.multiply %v2723, %v2724 : tensor<128x128x28x28xf32>
    %v2726 = stablehlo.reduce(%v2725 init: %v2709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2727 = stablehlo.broadcast_in_dim %v2726, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2728 = stablehlo.multiply %v2722, %v2725 : tensor<128x128x28x28xf32>
    %v2729 = stablehlo.reduce(%v2728 init: %v2709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2730 = stablehlo.broadcast_in_dim %v2729, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2731 = stablehlo.multiply %v2725, %v2710 : tensor<128x128x28x28xf32>
    %v2732 = stablehlo.subtract %v2731, %v2727 : tensor<128x128x28x28xf32>
    %v2733 = stablehlo.multiply %v2722, %v2730 : tensor<128x128x28x28xf32>
    %v2734 = stablehlo.subtract %v2732, %v2733 : tensor<128x128x28x28xf32>
    %v2735 = stablehlo.divide %v2721, %v2710 : tensor<128x128x28x28xf32>
    %v2736 = stablehlo.multiply %v2735, %v2734 : tensor<128x128x28x28xf32>
    %v2737 = stablehlo.reshape %v2736 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2738 = stablehlo.reshape %v2737 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2739 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2740 = stablehlo.transpose %v2739, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2741 = stablehlo.convolution(%v2738, %v2740)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v2742 = stablehlo.reshape %v2741 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2743 = stablehlo.reshape %v2742 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2744 = stablehlo.reshape %v2666 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2745 = stablehlo.add %v2743, %v2744 : tensor<128x128x28x28xf32>
    %v2746 = stablehlo.reshape %v2745 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2747 = stablehlo.reshape %v367 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2748 = stablehlo.reshape %v2737 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2749 = stablehlo.transpose %v2747, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v2750 = stablehlo.transpose %v2748, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v2751 = stablehlo.convolution(%v2749, %v2750)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2752 = stablehlo.transpose %v2751, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2753 = stablehlo.reshape %v372 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2754 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2755 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v2756 = stablehlo.reduce(%v2753 init: %v2754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2757 = stablehlo.broadcast_in_dim %v2756, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2758 = stablehlo.divide %v2757, %v2755 : tensor<128x128x28x28xf32>
    %v2759 = stablehlo.subtract %v2753, %v2758 : tensor<128x128x28x28xf32>
    %v2760 = stablehlo.multiply %v2759, %v2759 : tensor<128x128x28x28xf32>
    %v2761 = stablehlo.reduce(%v2760 init: %v2754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2762 = stablehlo.broadcast_in_dim %v2761, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2763 = stablehlo.divide %v2762, %v2755 : tensor<128x128x28x28xf32>
    %v2764 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v2765 = stablehlo.add %v2763, %v2764 : tensor<128x128x28x28xf32>
    %v2766 = stablehlo.rsqrt %v2765 : tensor<128x128x28x28xf32>
    %v2767 = stablehlo.multiply %v2759, %v2766 : tensor<128x128x28x28xf32>
    %v2768 = stablehlo.reshape %v2707 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2769 = stablehlo.multiply %v2768, %v2767 : tensor<128x128x28x28xf32>
    %v2770 = stablehlo.reduce(%v2769 init: %v2754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2771 = stablehlo.reshape %v2707 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2772 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2773 = stablehlo.reduce(%v2771 init: %v2772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2774 = stablehlo.reshape %v396 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2775 = stablehlo.reshape %v2696 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2776 = stablehlo.transpose %v2774, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v2777 = stablehlo.transpose %v2775, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v2778 = stablehlo.convolution(%v2776, %v2777)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2779 = stablehlo.transpose %v2778, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2780 = stablehlo.reshape %v401 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2782 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v2783 = stablehlo.reduce(%v2780 init: %v2781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2784 = stablehlo.broadcast_in_dim %v2783, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2785 = stablehlo.divide %v2784, %v2782 : tensor<128x128x28x28xf32>
    %v2786 = stablehlo.subtract %v2780, %v2785 : tensor<128x128x28x28xf32>
    %v2787 = stablehlo.multiply %v2786, %v2786 : tensor<128x128x28x28xf32>
    %v2788 = stablehlo.reduce(%v2787 init: %v2781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2789 = stablehlo.broadcast_in_dim %v2788, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2790 = stablehlo.divide %v2789, %v2782 : tensor<128x128x28x28xf32>
    %v2791 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v2792 = stablehlo.add %v2790, %v2791 : tensor<128x128x28x28xf32>
    %v2793 = stablehlo.rsqrt %v2792 : tensor<128x128x28x28xf32>
    %v2794 = stablehlo.multiply %v2786, %v2793 : tensor<128x128x28x28xf32>
    %v2795 = stablehlo.reshape %v2666 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2796 = stablehlo.multiply %v2795, %v2794 : tensor<128x128x28x28xf32>
    %v2797 = stablehlo.reduce(%v2796 init: %v2781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2798 = stablehlo.reshape %v2666 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2800 = stablehlo.reduce(%v2798 init: %v2799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2801 = stablehlo.reshape %v2746 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2802 = stablehlo.reshape %v363 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2803 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v2804 = stablehlo.compare GT, %v2802, %v2803 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v2805 = stablehlo.select %v2804, %v2801, %v2803 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v2806 = stablehlo.reshape %v2805 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2807 = stablehlo.reshape %v339 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2808 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2809 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v2810 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v2811 = stablehlo.reduce(%v2807 init: %v2808) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2812 = stablehlo.broadcast_in_dim %v2811, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2813 = stablehlo.divide %v2812, %v2809 : tensor<128x128x28x28xf32>
    %v2814 = stablehlo.subtract %v2807, %v2813 : tensor<128x128x28x28xf32>
    %v2815 = stablehlo.multiply %v2814, %v2814 : tensor<128x128x28x28xf32>
    %v2816 = stablehlo.reduce(%v2815 init: %v2808) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2817 = stablehlo.broadcast_in_dim %v2816, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2818 = stablehlo.divide %v2817, %v2809 : tensor<128x128x28x28xf32>
    %v2819 = stablehlo.add %v2818, %v2810 : tensor<128x128x28x28xf32>
    %v2820 = stablehlo.rsqrt %v2819 : tensor<128x128x28x28xf32>
    %v2821 = stablehlo.multiply %v2814, %v2820 : tensor<128x128x28x28xf32>
    %v2822 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2823 = stablehlo.reshape %v2806 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2824 = stablehlo.multiply %v2822, %v2823 : tensor<128x128x28x28xf32>
    %v2825 = stablehlo.reduce(%v2824 init: %v2808) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2826 = stablehlo.broadcast_in_dim %v2825, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2827 = stablehlo.multiply %v2821, %v2824 : tensor<128x128x28x28xf32>
    %v2828 = stablehlo.reduce(%v2827 init: %v2808) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2829 = stablehlo.broadcast_in_dim %v2828, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2830 = stablehlo.multiply %v2824, %v2809 : tensor<128x128x28x28xf32>
    %v2831 = stablehlo.subtract %v2830, %v2826 : tensor<128x128x28x28xf32>
    %v2832 = stablehlo.multiply %v2821, %v2829 : tensor<128x128x28x28xf32>
    %v2833 = stablehlo.subtract %v2831, %v2832 : tensor<128x128x28x28xf32>
    %v2834 = stablehlo.divide %v2820, %v2809 : tensor<128x128x28x28xf32>
    %v2835 = stablehlo.multiply %v2834, %v2833 : tensor<128x128x28x28xf32>
    %v2836 = stablehlo.reshape %v2835 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2837 = stablehlo.reshape %v2836 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2838 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2839 = stablehlo.transpose %v2838, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2840 = stablehlo.convolution(%v2837, %v2839)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v2841 = stablehlo.reshape %v2840 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2842 = stablehlo.reshape %v2841 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2843 = stablehlo.reshape %v330 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2844 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v2845 = stablehlo.compare GT, %v2843, %v2844 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v2846 = stablehlo.select %v2845, %v2842, %v2844 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v2847 = stablehlo.reshape %v2846 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2848 = stablehlo.reshape %v310 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2850 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v2851 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v2852 = stablehlo.reduce(%v2848 init: %v2849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2853 = stablehlo.broadcast_in_dim %v2852, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2854 = stablehlo.divide %v2853, %v2850 : tensor<128x128x28x28xf32>
    %v2855 = stablehlo.subtract %v2848, %v2854 : tensor<128x128x28x28xf32>
    %v2856 = stablehlo.multiply %v2855, %v2855 : tensor<128x128x28x28xf32>
    %v2857 = stablehlo.reduce(%v2856 init: %v2849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2858 = stablehlo.broadcast_in_dim %v2857, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2859 = stablehlo.divide %v2858, %v2850 : tensor<128x128x28x28xf32>
    %v2860 = stablehlo.add %v2859, %v2851 : tensor<128x128x28x28xf32>
    %v2861 = stablehlo.rsqrt %v2860 : tensor<128x128x28x28xf32>
    %v2862 = stablehlo.multiply %v2855, %v2861 : tensor<128x128x28x28xf32>
    %v2863 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2864 = stablehlo.reshape %v2847 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2865 = stablehlo.multiply %v2863, %v2864 : tensor<128x128x28x28xf32>
    %v2866 = stablehlo.reduce(%v2865 init: %v2849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2867 = stablehlo.broadcast_in_dim %v2866, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2868 = stablehlo.multiply %v2862, %v2865 : tensor<128x128x28x28xf32>
    %v2869 = stablehlo.reduce(%v2868 init: %v2849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2870 = stablehlo.broadcast_in_dim %v2869, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2871 = stablehlo.multiply %v2865, %v2850 : tensor<128x128x28x28xf32>
    %v2872 = stablehlo.subtract %v2871, %v2867 : tensor<128x128x28x28xf32>
    %v2873 = stablehlo.multiply %v2862, %v2870 : tensor<128x128x28x28xf32>
    %v2874 = stablehlo.subtract %v2872, %v2873 : tensor<128x128x28x28xf32>
    %v2875 = stablehlo.divide %v2861, %v2850 : tensor<128x128x28x28xf32>
    %v2876 = stablehlo.multiply %v2875, %v2874 : tensor<128x128x28x28xf32>
    %v2877 = stablehlo.reshape %v2876 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2878 = stablehlo.reshape %v2877 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2879 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2880 = stablehlo.transpose %v2879, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2881 = stablehlo.convolution(%v2878, %v2880)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v2882 = stablehlo.reshape %v2881 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2883 = stablehlo.reshape %v2882 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2884 = stablehlo.reshape %v2806 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2885 = stablehlo.add %v2883, %v2884 : tensor<128x128x28x28xf32>
    %v2886 = stablehlo.reshape %v2885 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2887 = stablehlo.reshape %v305 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2888 = stablehlo.reshape %v2877 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2889 = stablehlo.transpose %v2887, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v2890 = stablehlo.transpose %v2888, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v2891 = stablehlo.convolution(%v2889, %v2890)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2892 = stablehlo.transpose %v2891, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2893 = stablehlo.reshape %v310 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2895 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v2896 = stablehlo.reduce(%v2893 init: %v2894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2897 = stablehlo.broadcast_in_dim %v2896, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2898 = stablehlo.divide %v2897, %v2895 : tensor<128x128x28x28xf32>
    %v2899 = stablehlo.subtract %v2893, %v2898 : tensor<128x128x28x28xf32>
    %v2900 = stablehlo.multiply %v2899, %v2899 : tensor<128x128x28x28xf32>
    %v2901 = stablehlo.reduce(%v2900 init: %v2894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2902 = stablehlo.broadcast_in_dim %v2901, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2903 = stablehlo.divide %v2902, %v2895 : tensor<128x128x28x28xf32>
    %v2904 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v2905 = stablehlo.add %v2903, %v2904 : tensor<128x128x28x28xf32>
    %v2906 = stablehlo.rsqrt %v2905 : tensor<128x128x28x28xf32>
    %v2907 = stablehlo.multiply %v2899, %v2906 : tensor<128x128x28x28xf32>
    %v2908 = stablehlo.reshape %v2847 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2909 = stablehlo.multiply %v2908, %v2907 : tensor<128x128x28x28xf32>
    %v2910 = stablehlo.reduce(%v2909 init: %v2894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2911 = stablehlo.reshape %v2847 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2912 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2913 = stablehlo.reduce(%v2911 init: %v2912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2914 = stablehlo.reshape %v334 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2915 = stablehlo.reshape %v2836 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2916 = stablehlo.transpose %v2914, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v2917 = stablehlo.transpose %v2915, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v2918 = stablehlo.convolution(%v2916, %v2917)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2919 = stablehlo.transpose %v2918, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2920 = stablehlo.reshape %v339 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2922 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v2923 = stablehlo.reduce(%v2920 init: %v2921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2924 = stablehlo.broadcast_in_dim %v2923, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2925 = stablehlo.divide %v2924, %v2922 : tensor<128x128x28x28xf32>
    %v2926 = stablehlo.subtract %v2920, %v2925 : tensor<128x128x28x28xf32>
    %v2927 = stablehlo.multiply %v2926, %v2926 : tensor<128x128x28x28xf32>
    %v2928 = stablehlo.reduce(%v2927 init: %v2921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2929 = stablehlo.broadcast_in_dim %v2928, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2930 = stablehlo.divide %v2929, %v2922 : tensor<128x128x28x28xf32>
    %v2931 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v2932 = stablehlo.add %v2930, %v2931 : tensor<128x128x28x28xf32>
    %v2933 = stablehlo.rsqrt %v2932 : tensor<128x128x28x28xf32>
    %v2934 = stablehlo.multiply %v2926, %v2933 : tensor<128x128x28x28xf32>
    %v2935 = stablehlo.reshape %v2806 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2936 = stablehlo.multiply %v2935, %v2934 : tensor<128x128x28x28xf32>
    %v2937 = stablehlo.reduce(%v2936 init: %v2921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2938 = stablehlo.reshape %v2806 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2940 = stablehlo.reduce(%v2938 init: %v2939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2941 = stablehlo.reshape %v2886 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2942 = stablehlo.reshape %v301 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2943 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v2944 = stablehlo.compare GT, %v2942, %v2943 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v2945 = stablehlo.select %v2944, %v2941, %v2943 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v2946 = stablehlo.reshape %v2945 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2947 = stablehlo.reshape %v252 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2948 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2949 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v2950 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v2951 = stablehlo.reduce(%v2947 init: %v2948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2952 = stablehlo.broadcast_in_dim %v2951, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2953 = stablehlo.divide %v2952, %v2949 : tensor<128x128x28x28xf32>
    %v2954 = stablehlo.subtract %v2947, %v2953 : tensor<128x128x28x28xf32>
    %v2955 = stablehlo.multiply %v2954, %v2954 : tensor<128x128x28x28xf32>
    %v2956 = stablehlo.reduce(%v2955 init: %v2948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2957 = stablehlo.broadcast_in_dim %v2956, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2958 = stablehlo.divide %v2957, %v2949 : tensor<128x128x28x28xf32>
    %v2959 = stablehlo.add %v2958, %v2950 : tensor<128x128x28x28xf32>
    %v2960 = stablehlo.rsqrt %v2959 : tensor<128x128x28x28xf32>
    %v2961 = stablehlo.multiply %v2954, %v2960 : tensor<128x128x28x28xf32>
    %v2962 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2963 = stablehlo.reshape %v2946 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2964 = stablehlo.multiply %v2962, %v2963 : tensor<128x128x28x28xf32>
    %v2965 = stablehlo.reduce(%v2964 init: %v2948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2966 = stablehlo.broadcast_in_dim %v2965, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2967 = stablehlo.multiply %v2961, %v2964 : tensor<128x128x28x28xf32>
    %v2968 = stablehlo.reduce(%v2967 init: %v2948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2969 = stablehlo.broadcast_in_dim %v2968, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2970 = stablehlo.multiply %v2964, %v2949 : tensor<128x128x28x28xf32>
    %v2971 = stablehlo.subtract %v2970, %v2966 : tensor<128x128x28x28xf32>
    %v2972 = stablehlo.multiply %v2961, %v2969 : tensor<128x128x28x28xf32>
    %v2973 = stablehlo.subtract %v2971, %v2972 : tensor<128x128x28x28xf32>
    %v2974 = stablehlo.divide %v2960, %v2949 : tensor<128x128x28x28xf32>
    %v2975 = stablehlo.multiply %v2974, %v2973 : tensor<128x128x28x28xf32>
    %v2976 = stablehlo.reshape %v2975 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2977 = stablehlo.reshape %v2976 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2978 = stablehlo.reverse %d2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2979 = stablehlo.transpose %v2978, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2980 = stablehlo.convolution(%v2977, %v2979)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v2981 = stablehlo.reshape %v2980 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2982 = stablehlo.reshape %v2981 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2983 = stablehlo.reshape %v243 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2984 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v2985 = stablehlo.compare GT, %v2983, %v2984 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v2986 = stablehlo.select %v2985, %v2982, %v2984 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v2987 = stablehlo.reshape %v2986 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v2988 = stablehlo.reshape %v223 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v2989 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2990 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v2991 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v2992 = stablehlo.reduce(%v2988 init: %v2989) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2993 = stablehlo.broadcast_in_dim %v2992, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2994 = stablehlo.divide %v2993, %v2990 : tensor<128x128x28x28xf32>
    %v2995 = stablehlo.subtract %v2988, %v2994 : tensor<128x128x28x28xf32>
    %v2996 = stablehlo.multiply %v2995, %v2995 : tensor<128x128x28x28xf32>
    %v2997 = stablehlo.reduce(%v2996 init: %v2989) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2998 = stablehlo.broadcast_in_dim %v2997, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v2999 = stablehlo.divide %v2998, %v2990 : tensor<128x128x28x28xf32>
    %v3000 = stablehlo.add %v2999, %v2991 : tensor<128x128x28x28xf32>
    %v3001 = stablehlo.rsqrt %v3000 : tensor<128x128x28x28xf32>
    %v3002 = stablehlo.multiply %v2995, %v3001 : tensor<128x128x28x28xf32>
    %v3003 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3004 = stablehlo.reshape %v2987 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3005 = stablehlo.multiply %v3003, %v3004 : tensor<128x128x28x28xf32>
    %v3006 = stablehlo.reduce(%v3005 init: %v2989) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3007 = stablehlo.broadcast_in_dim %v3006, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3008 = stablehlo.multiply %v3002, %v3005 : tensor<128x128x28x28xf32>
    %v3009 = stablehlo.reduce(%v3008 init: %v2989) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3010 = stablehlo.broadcast_in_dim %v3009, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3011 = stablehlo.multiply %v3005, %v2990 : tensor<128x128x28x28xf32>
    %v3012 = stablehlo.subtract %v3011, %v3007 : tensor<128x128x28x28xf32>
    %v3013 = stablehlo.multiply %v3002, %v3010 : tensor<128x128x28x28xf32>
    %v3014 = stablehlo.subtract %v3012, %v3013 : tensor<128x128x28x28xf32>
    %v3015 = stablehlo.divide %v3001, %v2990 : tensor<128x128x28x28xf32>
    %v3016 = stablehlo.multiply %v3015, %v3014 : tensor<128x128x28x28xf32>
    %v3017 = stablehlo.reshape %v3016 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3018 = stablehlo.reshape %v3017 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3019 = stablehlo.reverse %d2W1, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v3020 = stablehlo.transpose %v3019, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v3021 = stablehlo.convolution(%v3018, %v3020)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<64x128x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v3022 = stablehlo.reshape %v3021 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3023 = stablehlo.reshape %v277 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3024 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3025 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v3026 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3027 = stablehlo.reduce(%v3023 init: %v3024) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3028 = stablehlo.broadcast_in_dim %v3027, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3029 = stablehlo.divide %v3028, %v3025 : tensor<128x128x28x28xf32>
    %v3030 = stablehlo.subtract %v3023, %v3029 : tensor<128x128x28x28xf32>
    %v3031 = stablehlo.multiply %v3030, %v3030 : tensor<128x128x28x28xf32>
    %v3032 = stablehlo.reduce(%v3031 init: %v3024) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3033 = stablehlo.broadcast_in_dim %v3032, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3034 = stablehlo.divide %v3033, %v3025 : tensor<128x128x28x28xf32>
    %v3035 = stablehlo.add %v3034, %v3026 : tensor<128x128x28x28xf32>
    %v3036 = stablehlo.rsqrt %v3035 : tensor<128x128x28x28xf32>
    %v3037 = stablehlo.multiply %v3030, %v3036 : tensor<128x128x28x28xf32>
    %v3038 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3039 = stablehlo.reshape %v2946 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3040 = stablehlo.multiply %v3038, %v3039 : tensor<128x128x28x28xf32>
    %v3041 = stablehlo.reduce(%v3040 init: %v3024) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3042 = stablehlo.broadcast_in_dim %v3041, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3043 = stablehlo.multiply %v3037, %v3040 : tensor<128x128x28x28xf32>
    %v3044 = stablehlo.reduce(%v3043 init: %v3024) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3045 = stablehlo.broadcast_in_dim %v3044, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3046 = stablehlo.multiply %v3040, %v3025 : tensor<128x128x28x28xf32>
    %v3047 = stablehlo.subtract %v3046, %v3042 : tensor<128x128x28x28xf32>
    %v3048 = stablehlo.multiply %v3037, %v3045 : tensor<128x128x28x28xf32>
    %v3049 = stablehlo.subtract %v3047, %v3048 : tensor<128x128x28x28xf32>
    %v3050 = stablehlo.divide %v3036, %v3025 : tensor<128x128x28x28xf32>
    %v3051 = stablehlo.multiply %v3050, %v3049 : tensor<128x128x28x28xf32>
    %v3052 = stablehlo.reshape %v3051 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3053 = stablehlo.reshape %v3052 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3054 = stablehlo.reverse %d2Wp, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v3055 = stablehlo.transpose %v3054, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v3056 = stablehlo.convolution(%v3053, %v3055)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<64x128x1x1xf32>) -> tensor<128x64x56x56xf32>
    %v3057 = stablehlo.reshape %v3056 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3058 = stablehlo.reshape %v3022 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3059 = stablehlo.reshape %v3057 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3060 = stablehlo.add %v3058, %v3059 : tensor<128x64x56x56xf32>
    %v3061 = stablehlo.reshape %v3060 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3062 = stablehlo.reshape %v218 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3063 = stablehlo.reshape %v3017 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3064 = stablehlo.transpose %v3062, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3066 = stablehlo.transpose %v3063, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3067 = stablehlo.convolution(%v3064, %v3066)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<128x128x28x28xf32>) -> tensor<64x128x3x3xf32>
    %v3068 = stablehlo.transpose %v3067, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v3069 = stablehlo.reshape %v223 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3070 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3071 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v3072 = stablehlo.reduce(%v3069 init: %v3070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3073 = stablehlo.broadcast_in_dim %v3072, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3074 = stablehlo.divide %v3073, %v3071 : tensor<128x128x28x28xf32>
    %v3075 = stablehlo.subtract %v3069, %v3074 : tensor<128x128x28x28xf32>
    %v3076 = stablehlo.multiply %v3075, %v3075 : tensor<128x128x28x28xf32>
    %v3077 = stablehlo.reduce(%v3076 init: %v3070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3078 = stablehlo.broadcast_in_dim %v3077, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3079 = stablehlo.divide %v3078, %v3071 : tensor<128x128x28x28xf32>
    %v3080 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3081 = stablehlo.add %v3079, %v3080 : tensor<128x128x28x28xf32>
    %v3082 = stablehlo.rsqrt %v3081 : tensor<128x128x28x28xf32>
    %v3083 = stablehlo.multiply %v3075, %v3082 : tensor<128x128x28x28xf32>
    %v3084 = stablehlo.reshape %v2987 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3085 = stablehlo.multiply %v3084, %v3083 : tensor<128x128x28x28xf32>
    %v3086 = stablehlo.reduce(%v3085 init: %v3070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3087 = stablehlo.reshape %v2987 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3088 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3089 = stablehlo.reduce(%v3087 init: %v3088) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3090 = stablehlo.reshape %v247 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3091 = stablehlo.reshape %v2976 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3092 = stablehlo.transpose %v3090, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3093 = stablehlo.transpose %v3091, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3094 = stablehlo.convolution(%v3092, %v3093)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3095 = stablehlo.transpose %v3094, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3096 = stablehlo.reshape %v252 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3098 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v3099 = stablehlo.reduce(%v3096 init: %v3097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3100 = stablehlo.broadcast_in_dim %v3099, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3101 = stablehlo.divide %v3100, %v3098 : tensor<128x128x28x28xf32>
    %v3102 = stablehlo.subtract %v3096, %v3101 : tensor<128x128x28x28xf32>
    %v3103 = stablehlo.multiply %v3102, %v3102 : tensor<128x128x28x28xf32>
    %v3104 = stablehlo.reduce(%v3103 init: %v3097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3105 = stablehlo.broadcast_in_dim %v3104, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3106 = stablehlo.divide %v3105, %v3098 : tensor<128x128x28x28xf32>
    %v3107 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3108 = stablehlo.add %v3106, %v3107 : tensor<128x128x28x28xf32>
    %v3109 = stablehlo.rsqrt %v3108 : tensor<128x128x28x28xf32>
    %v3110 = stablehlo.multiply %v3102, %v3109 : tensor<128x128x28x28xf32>
    %v3111 = stablehlo.reshape %v2946 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3112 = stablehlo.multiply %v3111, %v3110 : tensor<128x128x28x28xf32>
    %v3113 = stablehlo.reduce(%v3112 init: %v3097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3114 = stablehlo.reshape %v2946 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3116 = stablehlo.reduce(%v3114 init: %v3115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3117 = stablehlo.reshape %v218 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3118 = stablehlo.reshape %v3052 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3119 = stablehlo.transpose %v3117, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3121 = stablehlo.transpose %v3118, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3120 = stablehlo.slice %v3119 [0:64, 0:128, 0:55, 0:55] : (tensor<64x128x56x56xf32>) -> tensor<64x128x55x55xf32>
    %v3122 = stablehlo.convolution(%v3120, %v3121)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x55x55xf32>, tensor<128x128x28x28xf32>) -> tensor<64x128x1x1xf32>
    %v3123 = stablehlo.transpose %v3122, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v3124 = stablehlo.reshape %v277 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3125 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3126 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v3127 = stablehlo.reduce(%v3124 init: %v3125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3128 = stablehlo.broadcast_in_dim %v3127, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3129 = stablehlo.divide %v3128, %v3126 : tensor<128x128x28x28xf32>
    %v3130 = stablehlo.subtract %v3124, %v3129 : tensor<128x128x28x28xf32>
    %v3131 = stablehlo.multiply %v3130, %v3130 : tensor<128x128x28x28xf32>
    %v3132 = stablehlo.reduce(%v3131 init: %v3125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3133 = stablehlo.broadcast_in_dim %v3132, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3134 = stablehlo.divide %v3133, %v3126 : tensor<128x128x28x28xf32>
    %v3135 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3136 = stablehlo.add %v3134, %v3135 : tensor<128x128x28x28xf32>
    %v3137 = stablehlo.rsqrt %v3136 : tensor<128x128x28x28xf32>
    %v3138 = stablehlo.multiply %v3130, %v3137 : tensor<128x128x28x28xf32>
    %v3139 = stablehlo.reshape %v2946 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3140 = stablehlo.multiply %v3139, %v3138 : tensor<128x128x28x28xf32>
    %v3141 = stablehlo.reduce(%v3140 init: %v3125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3142 = stablehlo.reshape %v2946 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3144 = stablehlo.reduce(%v3142 init: %v3143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3145 = stablehlo.reshape %v3061 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3146 = stablehlo.reshape %v214 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3147 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v3148 = stablehlo.compare GT, %v3146, %v3147 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3149 = stablehlo.select %v3148, %v3145, %v3147 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3150 = stablehlo.reshape %v3149 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3151 = stablehlo.reshape %v190 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3152 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3153 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3154 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v3155 = stablehlo.reduce(%v3151 init: %v3152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3156 = stablehlo.broadcast_in_dim %v3155, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3157 = stablehlo.divide %v3156, %v3153 : tensor<128x64x56x56xf32>
    %v3158 = stablehlo.subtract %v3151, %v3157 : tensor<128x64x56x56xf32>
    %v3159 = stablehlo.multiply %v3158, %v3158 : tensor<128x64x56x56xf32>
    %v3160 = stablehlo.reduce(%v3159 init: %v3152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3161 = stablehlo.broadcast_in_dim %v3160, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3162 = stablehlo.divide %v3161, %v3153 : tensor<128x64x56x56xf32>
    %v3163 = stablehlo.add %v3162, %v3154 : tensor<128x64x56x56xf32>
    %v3164 = stablehlo.rsqrt %v3163 : tensor<128x64x56x56xf32>
    %v3165 = stablehlo.multiply %v3158, %v3164 : tensor<128x64x56x56xf32>
    %v3166 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3167 = stablehlo.reshape %v3150 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3168 = stablehlo.multiply %v3166, %v3167 : tensor<128x64x56x56xf32>
    %v3169 = stablehlo.reduce(%v3168 init: %v3152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3170 = stablehlo.broadcast_in_dim %v3169, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3171 = stablehlo.multiply %v3165, %v3168 : tensor<128x64x56x56xf32>
    %v3172 = stablehlo.reduce(%v3171 init: %v3152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3173 = stablehlo.broadcast_in_dim %v3172, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3174 = stablehlo.multiply %v3168, %v3153 : tensor<128x64x56x56xf32>
    %v3175 = stablehlo.subtract %v3174, %v3170 : tensor<128x64x56x56xf32>
    %v3176 = stablehlo.multiply %v3165, %v3173 : tensor<128x64x56x56xf32>
    %v3177 = stablehlo.subtract %v3175, %v3176 : tensor<128x64x56x56xf32>
    %v3178 = stablehlo.divide %v3164, %v3153 : tensor<128x64x56x56xf32>
    %v3179 = stablehlo.multiply %v3178, %v3177 : tensor<128x64x56x56xf32>
    %v3180 = stablehlo.reshape %v3179 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3181 = stablehlo.reshape %v3180 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3182 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3183 = stablehlo.transpose %v3182, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3184 = stablehlo.convolution(%v3181, %v3183)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v3185 = stablehlo.reshape %v3184 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3186 = stablehlo.reshape %v3185 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3187 = stablehlo.reshape %v181 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3188 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v3189 = stablehlo.compare GT, %v3187, %v3188 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3190 = stablehlo.select %v3189, %v3186, %v3188 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3191 = stablehlo.reshape %v3190 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3192 = stablehlo.reshape %v161 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3193 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3194 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3195 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v3196 = stablehlo.reduce(%v3192 init: %v3193) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3197 = stablehlo.broadcast_in_dim %v3196, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3198 = stablehlo.divide %v3197, %v3194 : tensor<128x64x56x56xf32>
    %v3199 = stablehlo.subtract %v3192, %v3198 : tensor<128x64x56x56xf32>
    %v3200 = stablehlo.multiply %v3199, %v3199 : tensor<128x64x56x56xf32>
    %v3201 = stablehlo.reduce(%v3200 init: %v3193) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3202 = stablehlo.broadcast_in_dim %v3201, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3203 = stablehlo.divide %v3202, %v3194 : tensor<128x64x56x56xf32>
    %v3204 = stablehlo.add %v3203, %v3195 : tensor<128x64x56x56xf32>
    %v3205 = stablehlo.rsqrt %v3204 : tensor<128x64x56x56xf32>
    %v3206 = stablehlo.multiply %v3199, %v3205 : tensor<128x64x56x56xf32>
    %v3207 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3208 = stablehlo.reshape %v3191 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3209 = stablehlo.multiply %v3207, %v3208 : tensor<128x64x56x56xf32>
    %v3210 = stablehlo.reduce(%v3209 init: %v3193) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3211 = stablehlo.broadcast_in_dim %v3210, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3212 = stablehlo.multiply %v3206, %v3209 : tensor<128x64x56x56xf32>
    %v3213 = stablehlo.reduce(%v3212 init: %v3193) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3214 = stablehlo.broadcast_in_dim %v3213, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3215 = stablehlo.multiply %v3209, %v3194 : tensor<128x64x56x56xf32>
    %v3216 = stablehlo.subtract %v3215, %v3211 : tensor<128x64x56x56xf32>
    %v3217 = stablehlo.multiply %v3206, %v3214 : tensor<128x64x56x56xf32>
    %v3218 = stablehlo.subtract %v3216, %v3217 : tensor<128x64x56x56xf32>
    %v3219 = stablehlo.divide %v3205, %v3194 : tensor<128x64x56x56xf32>
    %v3220 = stablehlo.multiply %v3219, %v3218 : tensor<128x64x56x56xf32>
    %v3221 = stablehlo.reshape %v3220 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3222 = stablehlo.reshape %v3221 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3223 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3224 = stablehlo.transpose %v3223, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3225 = stablehlo.convolution(%v3222, %v3224)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v3226 = stablehlo.reshape %v3225 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3227 = stablehlo.reshape %v3226 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3228 = stablehlo.reshape %v3150 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3229 = stablehlo.add %v3227, %v3228 : tensor<128x64x56x56xf32>
    %v3230 = stablehlo.reshape %v3229 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3231 = stablehlo.reshape %v156 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3232 = stablehlo.reshape %v3221 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3233 = stablehlo.transpose %v3231, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3234 = stablehlo.transpose %v3232, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3235 = stablehlo.convolution(%v3233, %v3234)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3236 = stablehlo.transpose %v3235, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3237 = stablehlo.reshape %v161 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3238 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3239 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3240 = stablehlo.reduce(%v3237 init: %v3238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3241 = stablehlo.broadcast_in_dim %v3240, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3242 = stablehlo.divide %v3241, %v3239 : tensor<128x64x56x56xf32>
    %v3243 = stablehlo.subtract %v3237, %v3242 : tensor<128x64x56x56xf32>
    %v3244 = stablehlo.multiply %v3243, %v3243 : tensor<128x64x56x56xf32>
    %v3245 = stablehlo.reduce(%v3244 init: %v3238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3246 = stablehlo.broadcast_in_dim %v3245, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3247 = stablehlo.divide %v3246, %v3239 : tensor<128x64x56x56xf32>
    %v3248 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v3249 = stablehlo.add %v3247, %v3248 : tensor<128x64x56x56xf32>
    %v3250 = stablehlo.rsqrt %v3249 : tensor<128x64x56x56xf32>
    %v3251 = stablehlo.multiply %v3243, %v3250 : tensor<128x64x56x56xf32>
    %v3252 = stablehlo.reshape %v3191 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3253 = stablehlo.multiply %v3252, %v3251 : tensor<128x64x56x56xf32>
    %v3254 = stablehlo.reduce(%v3253 init: %v3238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3255 = stablehlo.reshape %v3191 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3257 = stablehlo.reduce(%v3255 init: %v3256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3258 = stablehlo.reshape %v185 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3259 = stablehlo.reshape %v3180 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3260 = stablehlo.transpose %v3258, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3261 = stablehlo.transpose %v3259, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3262 = stablehlo.convolution(%v3260, %v3261)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3263 = stablehlo.transpose %v3262, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3264 = stablehlo.reshape %v190 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3266 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3267 = stablehlo.reduce(%v3264 init: %v3265) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3268 = stablehlo.broadcast_in_dim %v3267, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3269 = stablehlo.divide %v3268, %v3266 : tensor<128x64x56x56xf32>
    %v3270 = stablehlo.subtract %v3264, %v3269 : tensor<128x64x56x56xf32>
    %v3271 = stablehlo.multiply %v3270, %v3270 : tensor<128x64x56x56xf32>
    %v3272 = stablehlo.reduce(%v3271 init: %v3265) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3273 = stablehlo.broadcast_in_dim %v3272, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3274 = stablehlo.divide %v3273, %v3266 : tensor<128x64x56x56xf32>
    %v3275 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v3276 = stablehlo.add %v3274, %v3275 : tensor<128x64x56x56xf32>
    %v3277 = stablehlo.rsqrt %v3276 : tensor<128x64x56x56xf32>
    %v3278 = stablehlo.multiply %v3270, %v3277 : tensor<128x64x56x56xf32>
    %v3279 = stablehlo.reshape %v3150 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3280 = stablehlo.multiply %v3279, %v3278 : tensor<128x64x56x56xf32>
    %v3281 = stablehlo.reduce(%v3280 init: %v3265) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3282 = stablehlo.reshape %v3150 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3284 = stablehlo.reduce(%v3282 init: %v3283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3285 = stablehlo.reshape %v3230 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3286 = stablehlo.reshape %v152 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3287 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v3288 = stablehlo.compare GT, %v3286, %v3287 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3289 = stablehlo.select %v3288, %v3285, %v3287 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3290 = stablehlo.reshape %v3289 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3291 = stablehlo.reshape %v128 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3292 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3293 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3294 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v3295 = stablehlo.reduce(%v3291 init: %v3292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3296 = stablehlo.broadcast_in_dim %v3295, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3297 = stablehlo.divide %v3296, %v3293 : tensor<128x64x56x56xf32>
    %v3298 = stablehlo.subtract %v3291, %v3297 : tensor<128x64x56x56xf32>
    %v3299 = stablehlo.multiply %v3298, %v3298 : tensor<128x64x56x56xf32>
    %v3300 = stablehlo.reduce(%v3299 init: %v3292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3301 = stablehlo.broadcast_in_dim %v3300, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3302 = stablehlo.divide %v3301, %v3293 : tensor<128x64x56x56xf32>
    %v3303 = stablehlo.add %v3302, %v3294 : tensor<128x64x56x56xf32>
    %v3304 = stablehlo.rsqrt %v3303 : tensor<128x64x56x56xf32>
    %v3305 = stablehlo.multiply %v3298, %v3304 : tensor<128x64x56x56xf32>
    %v3306 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3307 = stablehlo.reshape %v3290 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3308 = stablehlo.multiply %v3306, %v3307 : tensor<128x64x56x56xf32>
    %v3309 = stablehlo.reduce(%v3308 init: %v3292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3310 = stablehlo.broadcast_in_dim %v3309, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3311 = stablehlo.multiply %v3305, %v3308 : tensor<128x64x56x56xf32>
    %v3312 = stablehlo.reduce(%v3311 init: %v3292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3313 = stablehlo.broadcast_in_dim %v3312, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3314 = stablehlo.multiply %v3308, %v3293 : tensor<128x64x56x56xf32>
    %v3315 = stablehlo.subtract %v3314, %v3310 : tensor<128x64x56x56xf32>
    %v3316 = stablehlo.multiply %v3305, %v3313 : tensor<128x64x56x56xf32>
    %v3317 = stablehlo.subtract %v3315, %v3316 : tensor<128x64x56x56xf32>
    %v3318 = stablehlo.divide %v3304, %v3293 : tensor<128x64x56x56xf32>
    %v3319 = stablehlo.multiply %v3318, %v3317 : tensor<128x64x56x56xf32>
    %v3320 = stablehlo.reshape %v3319 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3321 = stablehlo.reshape %v3320 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3322 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3323 = stablehlo.transpose %v3322, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3324 = stablehlo.convolution(%v3321, %v3323)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v3325 = stablehlo.reshape %v3324 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3326 = stablehlo.reshape %v3325 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3327 = stablehlo.reshape %v119 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3328 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v3329 = stablehlo.compare GT, %v3327, %v3328 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3330 = stablehlo.select %v3329, %v3326, %v3328 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3331 = stablehlo.reshape %v3330 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3332 = stablehlo.reshape %v99 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3333 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3334 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3335 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v3336 = stablehlo.reduce(%v3332 init: %v3333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3337 = stablehlo.broadcast_in_dim %v3336, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3338 = stablehlo.divide %v3337, %v3334 : tensor<128x64x56x56xf32>
    %v3339 = stablehlo.subtract %v3332, %v3338 : tensor<128x64x56x56xf32>
    %v3340 = stablehlo.multiply %v3339, %v3339 : tensor<128x64x56x56xf32>
    %v3341 = stablehlo.reduce(%v3340 init: %v3333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3342 = stablehlo.broadcast_in_dim %v3341, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3343 = stablehlo.divide %v3342, %v3334 : tensor<128x64x56x56xf32>
    %v3344 = stablehlo.add %v3343, %v3335 : tensor<128x64x56x56xf32>
    %v3345 = stablehlo.rsqrt %v3344 : tensor<128x64x56x56xf32>
    %v3346 = stablehlo.multiply %v3339, %v3345 : tensor<128x64x56x56xf32>
    %v3347 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3348 = stablehlo.reshape %v3331 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3349 = stablehlo.multiply %v3347, %v3348 : tensor<128x64x56x56xf32>
    %v3350 = stablehlo.reduce(%v3349 init: %v3333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3351 = stablehlo.broadcast_in_dim %v3350, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3352 = stablehlo.multiply %v3346, %v3349 : tensor<128x64x56x56xf32>
    %v3353 = stablehlo.reduce(%v3352 init: %v3333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3354 = stablehlo.broadcast_in_dim %v3353, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3355 = stablehlo.multiply %v3349, %v3334 : tensor<128x64x56x56xf32>
    %v3356 = stablehlo.subtract %v3355, %v3351 : tensor<128x64x56x56xf32>
    %v3357 = stablehlo.multiply %v3346, %v3354 : tensor<128x64x56x56xf32>
    %v3358 = stablehlo.subtract %v3356, %v3357 : tensor<128x64x56x56xf32>
    %v3359 = stablehlo.divide %v3345, %v3334 : tensor<128x64x56x56xf32>
    %v3360 = stablehlo.multiply %v3359, %v3358 : tensor<128x64x56x56xf32>
    %v3361 = stablehlo.reshape %v3360 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3362 = stablehlo.reshape %v3361 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3363 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3364 = stablehlo.transpose %v3363, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3365 = stablehlo.convolution(%v3362, %v3364)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v3366 = stablehlo.reshape %v3365 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3367 = stablehlo.reshape %v3366 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3368 = stablehlo.reshape %v3290 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3369 = stablehlo.add %v3367, %v3368 : tensor<128x64x56x56xf32>
    %v3370 = stablehlo.reshape %v3369 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3371 = stablehlo.reshape %v94 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3372 = stablehlo.reshape %v3361 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3373 = stablehlo.transpose %v3371, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3374 = stablehlo.transpose %v3372, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3375 = stablehlo.convolution(%v3373, %v3374)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3376 = stablehlo.transpose %v3375, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3377 = stablehlo.reshape %v99 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3379 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3380 = stablehlo.reduce(%v3377 init: %v3378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3381 = stablehlo.broadcast_in_dim %v3380, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3382 = stablehlo.divide %v3381, %v3379 : tensor<128x64x56x56xf32>
    %v3383 = stablehlo.subtract %v3377, %v3382 : tensor<128x64x56x56xf32>
    %v3384 = stablehlo.multiply %v3383, %v3383 : tensor<128x64x56x56xf32>
    %v3385 = stablehlo.reduce(%v3384 init: %v3378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3386 = stablehlo.broadcast_in_dim %v3385, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3387 = stablehlo.divide %v3386, %v3379 : tensor<128x64x56x56xf32>
    %v3388 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v3389 = stablehlo.add %v3387, %v3388 : tensor<128x64x56x56xf32>
    %v3390 = stablehlo.rsqrt %v3389 : tensor<128x64x56x56xf32>
    %v3391 = stablehlo.multiply %v3383, %v3390 : tensor<128x64x56x56xf32>
    %v3392 = stablehlo.reshape %v3331 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3393 = stablehlo.multiply %v3392, %v3391 : tensor<128x64x56x56xf32>
    %v3394 = stablehlo.reduce(%v3393 init: %v3378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3395 = stablehlo.reshape %v3331 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3397 = stablehlo.reduce(%v3395 init: %v3396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3398 = stablehlo.reshape %v123 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3399 = stablehlo.reshape %v3320 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3400 = stablehlo.transpose %v3398, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3401 = stablehlo.transpose %v3399, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3402 = stablehlo.convolution(%v3400, %v3401)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3403 = stablehlo.transpose %v3402, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3404 = stablehlo.reshape %v128 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3405 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3406 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3407 = stablehlo.reduce(%v3404 init: %v3405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3408 = stablehlo.broadcast_in_dim %v3407, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3409 = stablehlo.divide %v3408, %v3406 : tensor<128x64x56x56xf32>
    %v3410 = stablehlo.subtract %v3404, %v3409 : tensor<128x64x56x56xf32>
    %v3411 = stablehlo.multiply %v3410, %v3410 : tensor<128x64x56x56xf32>
    %v3412 = stablehlo.reduce(%v3411 init: %v3405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3413 = stablehlo.broadcast_in_dim %v3412, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3414 = stablehlo.divide %v3413, %v3406 : tensor<128x64x56x56xf32>
    %v3415 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v3416 = stablehlo.add %v3414, %v3415 : tensor<128x64x56x56xf32>
    %v3417 = stablehlo.rsqrt %v3416 : tensor<128x64x56x56xf32>
    %v3418 = stablehlo.multiply %v3410, %v3417 : tensor<128x64x56x56xf32>
    %v3419 = stablehlo.reshape %v3290 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3420 = stablehlo.multiply %v3419, %v3418 : tensor<128x64x56x56xf32>
    %v3421 = stablehlo.reduce(%v3420 init: %v3405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3422 = stablehlo.reshape %v3290 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3424 = stablehlo.reduce(%v3422 init: %v3423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3425 = stablehlo.reshape %v3370 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3426 = stablehlo.reshape %v90 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3427 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v3428 = stablehlo.compare GT, %v3426, %v3427 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3429 = stablehlo.select %v3428, %v3425, %v3427 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3430 = stablehlo.reshape %v3429 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3431 = stablehlo.reshape %v66 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3433 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3434 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v3435 = stablehlo.reduce(%v3431 init: %v3432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3436 = stablehlo.broadcast_in_dim %v3435, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3437 = stablehlo.divide %v3436, %v3433 : tensor<128x64x56x56xf32>
    %v3438 = stablehlo.subtract %v3431, %v3437 : tensor<128x64x56x56xf32>
    %v3439 = stablehlo.multiply %v3438, %v3438 : tensor<128x64x56x56xf32>
    %v3440 = stablehlo.reduce(%v3439 init: %v3432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3441 = stablehlo.broadcast_in_dim %v3440, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3442 = stablehlo.divide %v3441, %v3433 : tensor<128x64x56x56xf32>
    %v3443 = stablehlo.add %v3442, %v3434 : tensor<128x64x56x56xf32>
    %v3444 = stablehlo.rsqrt %v3443 : tensor<128x64x56x56xf32>
    %v3445 = stablehlo.multiply %v3438, %v3444 : tensor<128x64x56x56xf32>
    %v3446 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3447 = stablehlo.reshape %v3430 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3448 = stablehlo.multiply %v3446, %v3447 : tensor<128x64x56x56xf32>
    %v3449 = stablehlo.reduce(%v3448 init: %v3432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3450 = stablehlo.broadcast_in_dim %v3449, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3451 = stablehlo.multiply %v3445, %v3448 : tensor<128x64x56x56xf32>
    %v3452 = stablehlo.reduce(%v3451 init: %v3432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3453 = stablehlo.broadcast_in_dim %v3452, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3454 = stablehlo.multiply %v3448, %v3433 : tensor<128x64x56x56xf32>
    %v3455 = stablehlo.subtract %v3454, %v3450 : tensor<128x64x56x56xf32>
    %v3456 = stablehlo.multiply %v3445, %v3453 : tensor<128x64x56x56xf32>
    %v3457 = stablehlo.subtract %v3455, %v3456 : tensor<128x64x56x56xf32>
    %v3458 = stablehlo.divide %v3444, %v3433 : tensor<128x64x56x56xf32>
    %v3459 = stablehlo.multiply %v3458, %v3457 : tensor<128x64x56x56xf32>
    %v3460 = stablehlo.reshape %v3459 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3461 = stablehlo.reshape %v3460 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3462 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3463 = stablehlo.transpose %v3462, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3464 = stablehlo.convolution(%v3461, %v3463)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v3465 = stablehlo.reshape %v3464 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3466 = stablehlo.reshape %v3465 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3467 = stablehlo.reshape %v57 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3468 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v3469 = stablehlo.compare GT, %v3467, %v3468 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3470 = stablehlo.select %v3469, %v3466, %v3468 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3471 = stablehlo.reshape %v3470 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3472 = stablehlo.reshape %v37 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3474 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3475 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v3476 = stablehlo.reduce(%v3472 init: %v3473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3477 = stablehlo.broadcast_in_dim %v3476, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3478 = stablehlo.divide %v3477, %v3474 : tensor<128x64x56x56xf32>
    %v3479 = stablehlo.subtract %v3472, %v3478 : tensor<128x64x56x56xf32>
    %v3480 = stablehlo.multiply %v3479, %v3479 : tensor<128x64x56x56xf32>
    %v3481 = stablehlo.reduce(%v3480 init: %v3473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3482 = stablehlo.broadcast_in_dim %v3481, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3483 = stablehlo.divide %v3482, %v3474 : tensor<128x64x56x56xf32>
    %v3484 = stablehlo.add %v3483, %v3475 : tensor<128x64x56x56xf32>
    %v3485 = stablehlo.rsqrt %v3484 : tensor<128x64x56x56xf32>
    %v3486 = stablehlo.multiply %v3479, %v3485 : tensor<128x64x56x56xf32>
    %v3487 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3488 = stablehlo.reshape %v3471 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3489 = stablehlo.multiply %v3487, %v3488 : tensor<128x64x56x56xf32>
    %v3490 = stablehlo.reduce(%v3489 init: %v3473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3491 = stablehlo.broadcast_in_dim %v3490, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3492 = stablehlo.multiply %v3486, %v3489 : tensor<128x64x56x56xf32>
    %v3493 = stablehlo.reduce(%v3492 init: %v3473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3494 = stablehlo.broadcast_in_dim %v3493, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3495 = stablehlo.multiply %v3489, %v3474 : tensor<128x64x56x56xf32>
    %v3496 = stablehlo.subtract %v3495, %v3491 : tensor<128x64x56x56xf32>
    %v3497 = stablehlo.multiply %v3486, %v3494 : tensor<128x64x56x56xf32>
    %v3498 = stablehlo.subtract %v3496, %v3497 : tensor<128x64x56x56xf32>
    %v3499 = stablehlo.divide %v3485, %v3474 : tensor<128x64x56x56xf32>
    %v3500 = stablehlo.multiply %v3499, %v3498 : tensor<128x64x56x56xf32>
    %v3501 = stablehlo.reshape %v3500 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3502 = stablehlo.reshape %v3501 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3503 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3504 = stablehlo.transpose %v3503, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3505 = stablehlo.convolution(%v3502, %v3504)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v3506 = stablehlo.reshape %v3505 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3507 = stablehlo.reshape %v3506 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3508 = stablehlo.reshape %v3430 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3509 = stablehlo.add %v3507, %v3508 : tensor<128x64x56x56xf32>
    %v3510 = stablehlo.reshape %v3509 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3511 = stablehlo.reshape %v32 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3512 = stablehlo.reshape %v3501 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3513 = stablehlo.transpose %v3511, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3514 = stablehlo.transpose %v3512, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3515 = stablehlo.convolution(%v3513, %v3514)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3516 = stablehlo.transpose %v3515, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3517 = stablehlo.reshape %v37 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3519 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3520 = stablehlo.reduce(%v3517 init: %v3518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3521 = stablehlo.broadcast_in_dim %v3520, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3522 = stablehlo.divide %v3521, %v3519 : tensor<128x64x56x56xf32>
    %v3523 = stablehlo.subtract %v3517, %v3522 : tensor<128x64x56x56xf32>
    %v3524 = stablehlo.multiply %v3523, %v3523 : tensor<128x64x56x56xf32>
    %v3525 = stablehlo.reduce(%v3524 init: %v3518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3526 = stablehlo.broadcast_in_dim %v3525, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3527 = stablehlo.divide %v3526, %v3519 : tensor<128x64x56x56xf32>
    %v3528 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v3529 = stablehlo.add %v3527, %v3528 : tensor<128x64x56x56xf32>
    %v3530 = stablehlo.rsqrt %v3529 : tensor<128x64x56x56xf32>
    %v3531 = stablehlo.multiply %v3523, %v3530 : tensor<128x64x56x56xf32>
    %v3532 = stablehlo.reshape %v3471 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3533 = stablehlo.multiply %v3532, %v3531 : tensor<128x64x56x56xf32>
    %v3534 = stablehlo.reduce(%v3533 init: %v3518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3535 = stablehlo.reshape %v3471 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3537 = stablehlo.reduce(%v3535 init: %v3536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3538 = stablehlo.reshape %v61 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3539 = stablehlo.reshape %v3460 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3540 = stablehlo.transpose %v3538, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3541 = stablehlo.transpose %v3539, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3542 = stablehlo.convolution(%v3540, %v3541)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3543 = stablehlo.transpose %v3542, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3544 = stablehlo.reshape %v66 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3546 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3547 = stablehlo.reduce(%v3544 init: %v3545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3548 = stablehlo.broadcast_in_dim %v3547, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3549 = stablehlo.divide %v3548, %v3546 : tensor<128x64x56x56xf32>
    %v3550 = stablehlo.subtract %v3544, %v3549 : tensor<128x64x56x56xf32>
    %v3551 = stablehlo.multiply %v3550, %v3550 : tensor<128x64x56x56xf32>
    %v3552 = stablehlo.reduce(%v3551 init: %v3545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3553 = stablehlo.broadcast_in_dim %v3552, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3554 = stablehlo.divide %v3553, %v3546 : tensor<128x64x56x56xf32>
    %v3555 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v3556 = stablehlo.add %v3554, %v3555 : tensor<128x64x56x56xf32>
    %v3557 = stablehlo.rsqrt %v3556 : tensor<128x64x56x56xf32>
    %v3558 = stablehlo.multiply %v3550, %v3557 : tensor<128x64x56x56xf32>
    %v3559 = stablehlo.reshape %v3430 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3560 = stablehlo.multiply %v3559, %v3558 : tensor<128x64x56x56xf32>
    %v3561 = stablehlo.reduce(%v3560 init: %v3545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3562 = stablehlo.reshape %v3430 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3564 = stablehlo.reduce(%v3562 init: %v3563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3565 = stablehlo.reshape %v28 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v3566 = stablehlo.reshape %v3510 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3567 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3568 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3569 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v3570 = stablehlo.pad %v3565, %v3567, low = [0, 0, 1, 1], high = [0, 0, 1, 1], interior = [0, 0, 0, 0] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<128x64x114x114xf32>
    %v3571 = stablehlo.slice %v3570 [0:128, 0:64, 0:112:2, 0:112:2] : (tensor<128x64x114x114xf32>) -> tensor<128x64x56x56xf32>
    %v3572 = stablehlo.slice %v3570 [0:128, 0:64, 0:112:2, 1:113:2] : (tensor<128x64x114x114xf32>) -> tensor<128x64x56x56xf32>
    %v3573 = stablehlo.slice %v3570 [0:128, 0:64, 0:112:2, 2:114:2] : (tensor<128x64x114x114xf32>) -> tensor<128x64x56x56xf32>
    %v3574 = stablehlo.slice %v3570 [0:128, 0:64, 1:113:2, 0:112:2] : (tensor<128x64x114x114xf32>) -> tensor<128x64x56x56xf32>
    %v3575 = stablehlo.slice %v3570 [0:128, 0:64, 1:113:2, 1:113:2] : (tensor<128x64x114x114xf32>) -> tensor<128x64x56x56xf32>
    %v3576 = stablehlo.slice %v3570 [0:128, 0:64, 1:113:2, 2:114:2] : (tensor<128x64x114x114xf32>) -> tensor<128x64x56x56xf32>
    %v3577 = stablehlo.slice %v3570 [0:128, 0:64, 2:114:2, 0:112:2] : (tensor<128x64x114x114xf32>) -> tensor<128x64x56x56xf32>
    %v3578 = stablehlo.slice %v3570 [0:128, 0:64, 2:114:2, 1:113:2] : (tensor<128x64x114x114xf32>) -> tensor<128x64x56x56xf32>
    %v3579 = stablehlo.slice %v3570 [0:128, 0:64, 2:114:2, 2:114:2] : (tensor<128x64x114x114xf32>) -> tensor<128x64x56x56xf32>
    %v3580 = stablehlo.maximum %v3571, %v3572 : tensor<128x64x56x56xf32>
    %v3581 = stablehlo.maximum %v3580, %v3573 : tensor<128x64x56x56xf32>
    %v3582 = stablehlo.maximum %v3581, %v3574 : tensor<128x64x56x56xf32>
    %v3583 = stablehlo.maximum %v3582, %v3575 : tensor<128x64x56x56xf32>
    %v3584 = stablehlo.maximum %v3583, %v3576 : tensor<128x64x56x56xf32>
    %v3585 = stablehlo.maximum %v3584, %v3577 : tensor<128x64x56x56xf32>
    %v3586 = stablehlo.maximum %v3585, %v3578 : tensor<128x64x56x56xf32>
    %v3587 = stablehlo.maximum %v3586, %v3579 : tensor<128x64x56x56xf32>
    %v3588 = stablehlo.compare EQ, %v3571, %v3587 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3589 = stablehlo.select %v3588, %v3566, %v3569 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3590 = stablehlo.pad %v3589, %v3568, low = [0, 0, 0, 0], high = [0, 0, 3, 3], interior = [0, 0, 1, 1] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<128x64x114x114xf32>
    %v3591 = stablehlo.compare EQ, %v3572, %v3587 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3592 = stablehlo.not %v3588 : tensor<128x64x56x56xi1>
    %v3593 = stablehlo.and %v3591, %v3592 : tensor<128x64x56x56xi1>
    %v3594 = stablehlo.or %v3588, %v3591 : tensor<128x64x56x56xi1>
    %v3595 = stablehlo.select %v3593, %v3566, %v3569 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3596 = stablehlo.pad %v3595, %v3568, low = [0, 0, 0, 1], high = [0, 0, 3, 2], interior = [0, 0, 1, 1] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<128x64x114x114xf32>
    %v3597 = stablehlo.add %v3590, %v3596 : tensor<128x64x114x114xf32>
    %v3598 = stablehlo.compare EQ, %v3573, %v3587 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3599 = stablehlo.not %v3594 : tensor<128x64x56x56xi1>
    %v3600 = stablehlo.and %v3598, %v3599 : tensor<128x64x56x56xi1>
    %v3601 = stablehlo.or %v3594, %v3598 : tensor<128x64x56x56xi1>
    %v3602 = stablehlo.select %v3600, %v3566, %v3569 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3603 = stablehlo.pad %v3602, %v3568, low = [0, 0, 0, 2], high = [0, 0, 3, 1], interior = [0, 0, 1, 1] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<128x64x114x114xf32>
    %v3604 = stablehlo.add %v3597, %v3603 : tensor<128x64x114x114xf32>
    %v3605 = stablehlo.compare EQ, %v3574, %v3587 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3606 = stablehlo.not %v3601 : tensor<128x64x56x56xi1>
    %v3607 = stablehlo.and %v3605, %v3606 : tensor<128x64x56x56xi1>
    %v3608 = stablehlo.or %v3601, %v3605 : tensor<128x64x56x56xi1>
    %v3609 = stablehlo.select %v3607, %v3566, %v3569 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3610 = stablehlo.pad %v3609, %v3568, low = [0, 0, 1, 0], high = [0, 0, 2, 3], interior = [0, 0, 1, 1] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<128x64x114x114xf32>
    %v3611 = stablehlo.add %v3604, %v3610 : tensor<128x64x114x114xf32>
    %v3612 = stablehlo.compare EQ, %v3575, %v3587 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3613 = stablehlo.not %v3608 : tensor<128x64x56x56xi1>
    %v3614 = stablehlo.and %v3612, %v3613 : tensor<128x64x56x56xi1>
    %v3615 = stablehlo.or %v3608, %v3612 : tensor<128x64x56x56xi1>
    %v3616 = stablehlo.select %v3614, %v3566, %v3569 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3617 = stablehlo.pad %v3616, %v3568, low = [0, 0, 1, 1], high = [0, 0, 2, 2], interior = [0, 0, 1, 1] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<128x64x114x114xf32>
    %v3618 = stablehlo.add %v3611, %v3617 : tensor<128x64x114x114xf32>
    %v3619 = stablehlo.compare EQ, %v3576, %v3587 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3620 = stablehlo.not %v3615 : tensor<128x64x56x56xi1>
    %v3621 = stablehlo.and %v3619, %v3620 : tensor<128x64x56x56xi1>
    %v3622 = stablehlo.or %v3615, %v3619 : tensor<128x64x56x56xi1>
    %v3623 = stablehlo.select %v3621, %v3566, %v3569 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3624 = stablehlo.pad %v3623, %v3568, low = [0, 0, 1, 2], high = [0, 0, 2, 1], interior = [0, 0, 1, 1] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<128x64x114x114xf32>
    %v3625 = stablehlo.add %v3618, %v3624 : tensor<128x64x114x114xf32>
    %v3626 = stablehlo.compare EQ, %v3577, %v3587 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3627 = stablehlo.not %v3622 : tensor<128x64x56x56xi1>
    %v3628 = stablehlo.and %v3626, %v3627 : tensor<128x64x56x56xi1>
    %v3629 = stablehlo.or %v3622, %v3626 : tensor<128x64x56x56xi1>
    %v3630 = stablehlo.select %v3628, %v3566, %v3569 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3631 = stablehlo.pad %v3630, %v3568, low = [0, 0, 2, 0], high = [0, 0, 1, 3], interior = [0, 0, 1, 1] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<128x64x114x114xf32>
    %v3632 = stablehlo.add %v3625, %v3631 : tensor<128x64x114x114xf32>
    %v3633 = stablehlo.compare EQ, %v3578, %v3587 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3634 = stablehlo.not %v3629 : tensor<128x64x56x56xi1>
    %v3635 = stablehlo.and %v3633, %v3634 : tensor<128x64x56x56xi1>
    %v3636 = stablehlo.or %v3629, %v3633 : tensor<128x64x56x56xi1>
    %v3637 = stablehlo.select %v3635, %v3566, %v3569 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3638 = stablehlo.pad %v3637, %v3568, low = [0, 0, 2, 1], high = [0, 0, 1, 2], interior = [0, 0, 1, 1] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<128x64x114x114xf32>
    %v3639 = stablehlo.add %v3632, %v3638 : tensor<128x64x114x114xf32>
    %v3640 = stablehlo.compare EQ, %v3579, %v3587 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3641 = stablehlo.not %v3636 : tensor<128x64x56x56xi1>
    %v3642 = stablehlo.and %v3640, %v3641 : tensor<128x64x56x56xi1>
    %v3643 = stablehlo.or %v3636, %v3640 : tensor<128x64x56x56xi1>
    %v3644 = stablehlo.select %v3642, %v3566, %v3569 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3645 = stablehlo.pad %v3644, %v3568, low = [0, 0, 2, 2], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<128x64x114x114xf32>
    %v3646 = stablehlo.add %v3639, %v3645 : tensor<128x64x114x114xf32>
    %v3647 = stablehlo.slice %v3646 [0:128, 0:64, 1:113, 1:113] : (tensor<128x64x114x114xf32>) -> tensor<128x64x112x112xf32>
    %v3648 = stablehlo.reshape %v3647 : (tensor<128x64x112x112xf32>) -> tensor<128x802816xf32>
    %v3649 = stablehlo.reshape %v3648 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v3650 = stablehlo.reshape %v24 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v3651 = stablehlo.constant dense<0.0> : tensor<128x64x112x112xf32>
    %v3652 = stablehlo.compare GT, %v3650, %v3651 : (tensor<128x64x112x112xf32>, tensor<128x64x112x112xf32>) -> tensor<128x64x112x112xi1>
    %v3653 = stablehlo.select %v3652, %v3649, %v3651 : tensor<128x64x112x112xi1>, tensor<128x64x112x112xf32>
    %v3654 = stablehlo.reshape %v3653 : (tensor<128x64x112x112xf32>) -> tensor<128x802816xf32>
    %v3655 = stablehlo.reshape %v4 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v3656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3657 = stablehlo.constant dense<1605632.0> : tensor<128x64x112x112xf32>
    %v3658 = stablehlo.constant dense<1.0e-05> : tensor<128x64x112x112xf32>
    %v3659 = stablehlo.reduce(%v3655 init: %v3656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3660 = stablehlo.broadcast_in_dim %v3659, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v3661 = stablehlo.divide %v3660, %v3657 : tensor<128x64x112x112xf32>
    %v3662 = stablehlo.subtract %v3655, %v3661 : tensor<128x64x112x112xf32>
    %v3663 = stablehlo.multiply %v3662, %v3662 : tensor<128x64x112x112xf32>
    %v3664 = stablehlo.reduce(%v3663 init: %v3656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3665 = stablehlo.broadcast_in_dim %v3664, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v3666 = stablehlo.divide %v3665, %v3657 : tensor<128x64x112x112xf32>
    %v3667 = stablehlo.add %v3666, %v3658 : tensor<128x64x112x112xf32>
    %v3668 = stablehlo.rsqrt %v3667 : tensor<128x64x112x112xf32>
    %v3669 = stablehlo.multiply %v3662, %v3668 : tensor<128x64x112x112xf32>
    %v3670 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v3671 = stablehlo.reshape %v3654 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v3672 = stablehlo.multiply %v3670, %v3671 : tensor<128x64x112x112xf32>
    %v3673 = stablehlo.reduce(%v3672 init: %v3656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3674 = stablehlo.broadcast_in_dim %v3673, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v3675 = stablehlo.multiply %v3669, %v3672 : tensor<128x64x112x112xf32>
    %v3676 = stablehlo.reduce(%v3675 init: %v3656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3677 = stablehlo.broadcast_in_dim %v3676, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v3678 = stablehlo.multiply %v3672, %v3657 : tensor<128x64x112x112xf32>
    %v3679 = stablehlo.subtract %v3678, %v3674 : tensor<128x64x112x112xf32>
    %v3680 = stablehlo.multiply %v3669, %v3677 : tensor<128x64x112x112xf32>
    %v3681 = stablehlo.subtract %v3679, %v3680 : tensor<128x64x112x112xf32>
    %v3682 = stablehlo.divide %v3668, %v3657 : tensor<128x64x112x112xf32>
    %v3683 = stablehlo.multiply %v3682, %v3681 : tensor<128x64x112x112xf32>
    %v3684 = stablehlo.reshape %v3683 : (tensor<128x64x112x112xf32>) -> tensor<128x802816xf32>
    %v3685 = stablehlo.reshape %x : (tensor<128x150528xf32>) -> tensor<128x3x224x224xf32>
    %v3686 = stablehlo.reshape %v3684 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v3687 = stablehlo.transpose %v3685, dims = [1, 0, 2, 3] : (tensor<128x3x224x224xf32>) -> tensor<3x128x224x224xf32>
    %v3689 = stablehlo.transpose %v3686, dims = [1, 0, 2, 3] : (tensor<128x64x112x112xf32>) -> tensor<64x128x112x112xf32>
    %v3690 = stablehlo.convolution(%v3687, %v3689)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x224x224xf32>, tensor<64x128x112x112xf32>) -> tensor<3x64x7x7xf32>
    %v3691 = stablehlo.transpose %v3690, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v3692 = stablehlo.reshape %v4 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v3693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3694 = stablehlo.constant dense<1605632.0> : tensor<128x64x112x112xf32>
    %v3695 = stablehlo.reduce(%v3692 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3696 = stablehlo.broadcast_in_dim %v3695, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v3697 = stablehlo.divide %v3696, %v3694 : tensor<128x64x112x112xf32>
    %v3698 = stablehlo.subtract %v3692, %v3697 : tensor<128x64x112x112xf32>
    %v3699 = stablehlo.multiply %v3698, %v3698 : tensor<128x64x112x112xf32>
    %v3700 = stablehlo.reduce(%v3699 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3701 = stablehlo.broadcast_in_dim %v3700, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v3702 = stablehlo.divide %v3701, %v3694 : tensor<128x64x112x112xf32>
    %v3703 = stablehlo.constant dense<1.0e-05> : tensor<128x64x112x112xf32>
    %v3704 = stablehlo.add %v3702, %v3703 : tensor<128x64x112x112xf32>
    %v3705 = stablehlo.rsqrt %v3704 : tensor<128x64x112x112xf32>
    %v3706 = stablehlo.multiply %v3698, %v3705 : tensor<128x64x112x112xf32>
    %v3707 = stablehlo.reshape %v3654 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v3708 = stablehlo.multiply %v3707, %v3706 : tensor<128x64x112x112xf32>
    %v3709 = stablehlo.reduce(%v3708 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3710 = stablehlo.reshape %v3654 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v3711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3712 = stablehlo.reduce(%v3710 init: %v3711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3713 = stablehlo.reshape %v4 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v3714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3715 = stablehlo.constant dense<1605632.0> : tensor<64xf32>
    %v3716 = stablehlo.reduce(%v3713 init: %v3714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3717 = stablehlo.divide %v3716, %v3715 : tensor<64xf32>
    %v3718 = stablehlo.reshape %v4 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v3719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3720 = stablehlo.constant dense<1605632.0> : tensor<128x64x112x112xf32>
    %v3721 = stablehlo.reduce(%v3718 init: %v3719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3722 = stablehlo.broadcast_in_dim %v3721, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v3723 = stablehlo.divide %v3722, %v3720 : tensor<128x64x112x112xf32>
    %v3724 = stablehlo.subtract %v3718, %v3723 : tensor<128x64x112x112xf32>
    %v3725 = stablehlo.multiply %v3724, %v3724 : tensor<128x64x112x112xf32>
    %v3726 = stablehlo.reduce(%v3725 init: %v3719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3727 = stablehlo.constant dense<1605632.0> : tensor<64xf32>
    %v3728 = stablehlo.divide %v3726, %v3727 : tensor<64xf32>
    %v3729 = stablehlo.reshape %v37 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3731 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3732 = stablehlo.reduce(%v3729 init: %v3730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3733 = stablehlo.divide %v3732, %v3731 : tensor<64xf32>
    %v3734 = stablehlo.reshape %v37 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3736 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3737 = stablehlo.reduce(%v3734 init: %v3735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3738 = stablehlo.broadcast_in_dim %v3737, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3739 = stablehlo.divide %v3738, %v3736 : tensor<128x64x56x56xf32>
    %v3740 = stablehlo.subtract %v3734, %v3739 : tensor<128x64x56x56xf32>
    %v3741 = stablehlo.multiply %v3740, %v3740 : tensor<128x64x56x56xf32>
    %v3742 = stablehlo.reduce(%v3741 init: %v3735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3743 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3744 = stablehlo.divide %v3742, %v3743 : tensor<64xf32>
    %v3745 = stablehlo.reshape %v66 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3747 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3748 = stablehlo.reduce(%v3745 init: %v3746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3749 = stablehlo.divide %v3748, %v3747 : tensor<64xf32>
    %v3750 = stablehlo.reshape %v66 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3751 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3752 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3753 = stablehlo.reduce(%v3750 init: %v3751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3754 = stablehlo.broadcast_in_dim %v3753, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3755 = stablehlo.divide %v3754, %v3752 : tensor<128x64x56x56xf32>
    %v3756 = stablehlo.subtract %v3750, %v3755 : tensor<128x64x56x56xf32>
    %v3757 = stablehlo.multiply %v3756, %v3756 : tensor<128x64x56x56xf32>
    %v3758 = stablehlo.reduce(%v3757 init: %v3751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3759 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3760 = stablehlo.divide %v3758, %v3759 : tensor<64xf32>
    %v3761 = stablehlo.reshape %v99 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3763 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3764 = stablehlo.reduce(%v3761 init: %v3762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3765 = stablehlo.divide %v3764, %v3763 : tensor<64xf32>
    %v3766 = stablehlo.reshape %v99 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3768 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3769 = stablehlo.reduce(%v3766 init: %v3767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3770 = stablehlo.broadcast_in_dim %v3769, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3771 = stablehlo.divide %v3770, %v3768 : tensor<128x64x56x56xf32>
    %v3772 = stablehlo.subtract %v3766, %v3771 : tensor<128x64x56x56xf32>
    %v3773 = stablehlo.multiply %v3772, %v3772 : tensor<128x64x56x56xf32>
    %v3774 = stablehlo.reduce(%v3773 init: %v3767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3775 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3776 = stablehlo.divide %v3774, %v3775 : tensor<64xf32>
    %v3777 = stablehlo.reshape %v128 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3779 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3780 = stablehlo.reduce(%v3777 init: %v3778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3781 = stablehlo.divide %v3780, %v3779 : tensor<64xf32>
    %v3782 = stablehlo.reshape %v128 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3784 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3785 = stablehlo.reduce(%v3782 init: %v3783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3786 = stablehlo.broadcast_in_dim %v3785, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3787 = stablehlo.divide %v3786, %v3784 : tensor<128x64x56x56xf32>
    %v3788 = stablehlo.subtract %v3782, %v3787 : tensor<128x64x56x56xf32>
    %v3789 = stablehlo.multiply %v3788, %v3788 : tensor<128x64x56x56xf32>
    %v3790 = stablehlo.reduce(%v3789 init: %v3783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3791 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3792 = stablehlo.divide %v3790, %v3791 : tensor<64xf32>
    %v3793 = stablehlo.reshape %v161 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3794 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3795 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3796 = stablehlo.reduce(%v3793 init: %v3794) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3797 = stablehlo.divide %v3796, %v3795 : tensor<64xf32>
    %v3798 = stablehlo.reshape %v161 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3800 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3801 = stablehlo.reduce(%v3798 init: %v3799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3802 = stablehlo.broadcast_in_dim %v3801, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3803 = stablehlo.divide %v3802, %v3800 : tensor<128x64x56x56xf32>
    %v3804 = stablehlo.subtract %v3798, %v3803 : tensor<128x64x56x56xf32>
    %v3805 = stablehlo.multiply %v3804, %v3804 : tensor<128x64x56x56xf32>
    %v3806 = stablehlo.reduce(%v3805 init: %v3799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3807 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3808 = stablehlo.divide %v3806, %v3807 : tensor<64xf32>
    %v3809 = stablehlo.reshape %v190 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3811 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3812 = stablehlo.reduce(%v3809 init: %v3810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3813 = stablehlo.divide %v3812, %v3811 : tensor<64xf32>
    %v3814 = stablehlo.reshape %v190 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3815 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3816 = stablehlo.constant dense<401408.0> : tensor<128x64x56x56xf32>
    %v3817 = stablehlo.reduce(%v3814 init: %v3815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3818 = stablehlo.broadcast_in_dim %v3817, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3819 = stablehlo.divide %v3818, %v3816 : tensor<128x64x56x56xf32>
    %v3820 = stablehlo.subtract %v3814, %v3819 : tensor<128x64x56x56xf32>
    %v3821 = stablehlo.multiply %v3820, %v3820 : tensor<128x64x56x56xf32>
    %v3822 = stablehlo.reduce(%v3821 init: %v3815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3823 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3824 = stablehlo.divide %v3822, %v3823 : tensor<64xf32>
    %v3825 = stablehlo.reshape %v223 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3827 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3828 = stablehlo.reduce(%v3825 init: %v3826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3829 = stablehlo.divide %v3828, %v3827 : tensor<128xf32>
    %v3830 = stablehlo.reshape %v223 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3831 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3832 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v3833 = stablehlo.reduce(%v3830 init: %v3831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3834 = stablehlo.broadcast_in_dim %v3833, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3835 = stablehlo.divide %v3834, %v3832 : tensor<128x128x28x28xf32>
    %v3836 = stablehlo.subtract %v3830, %v3835 : tensor<128x128x28x28xf32>
    %v3837 = stablehlo.multiply %v3836, %v3836 : tensor<128x128x28x28xf32>
    %v3838 = stablehlo.reduce(%v3837 init: %v3831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3839 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3840 = stablehlo.divide %v3838, %v3839 : tensor<128xf32>
    %v3841 = stablehlo.reshape %v252 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3843 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3844 = stablehlo.reduce(%v3841 init: %v3842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3845 = stablehlo.divide %v3844, %v3843 : tensor<128xf32>
    %v3846 = stablehlo.reshape %v252 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3847 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3848 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v3849 = stablehlo.reduce(%v3846 init: %v3847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3850 = stablehlo.broadcast_in_dim %v3849, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3851 = stablehlo.divide %v3850, %v3848 : tensor<128x128x28x28xf32>
    %v3852 = stablehlo.subtract %v3846, %v3851 : tensor<128x128x28x28xf32>
    %v3853 = stablehlo.multiply %v3852, %v3852 : tensor<128x128x28x28xf32>
    %v3854 = stablehlo.reduce(%v3853 init: %v3847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3855 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3856 = stablehlo.divide %v3854, %v3855 : tensor<128xf32>
    %v3857 = stablehlo.reshape %v277 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3858 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3859 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3860 = stablehlo.reduce(%v3857 init: %v3858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3861 = stablehlo.divide %v3860, %v3859 : tensor<128xf32>
    %v3862 = stablehlo.reshape %v277 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3864 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v3865 = stablehlo.reduce(%v3862 init: %v3863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3866 = stablehlo.broadcast_in_dim %v3865, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3867 = stablehlo.divide %v3866, %v3864 : tensor<128x128x28x28xf32>
    %v3868 = stablehlo.subtract %v3862, %v3867 : tensor<128x128x28x28xf32>
    %v3869 = stablehlo.multiply %v3868, %v3868 : tensor<128x128x28x28xf32>
    %v3870 = stablehlo.reduce(%v3869 init: %v3863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3871 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3872 = stablehlo.divide %v3870, %v3871 : tensor<128xf32>
    %v3873 = stablehlo.reshape %v310 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3875 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3876 = stablehlo.reduce(%v3873 init: %v3874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3877 = stablehlo.divide %v3876, %v3875 : tensor<128xf32>
    %v3878 = stablehlo.reshape %v310 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3879 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3880 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v3881 = stablehlo.reduce(%v3878 init: %v3879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3882 = stablehlo.broadcast_in_dim %v3881, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3883 = stablehlo.divide %v3882, %v3880 : tensor<128x128x28x28xf32>
    %v3884 = stablehlo.subtract %v3878, %v3883 : tensor<128x128x28x28xf32>
    %v3885 = stablehlo.multiply %v3884, %v3884 : tensor<128x128x28x28xf32>
    %v3886 = stablehlo.reduce(%v3885 init: %v3879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3887 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3888 = stablehlo.divide %v3886, %v3887 : tensor<128xf32>
    %v3889 = stablehlo.reshape %v339 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3891 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3892 = stablehlo.reduce(%v3889 init: %v3890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3893 = stablehlo.divide %v3892, %v3891 : tensor<128xf32>
    %v3894 = stablehlo.reshape %v339 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3896 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v3897 = stablehlo.reduce(%v3894 init: %v3895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3898 = stablehlo.broadcast_in_dim %v3897, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3899 = stablehlo.divide %v3898, %v3896 : tensor<128x128x28x28xf32>
    %v3900 = stablehlo.subtract %v3894, %v3899 : tensor<128x128x28x28xf32>
    %v3901 = stablehlo.multiply %v3900, %v3900 : tensor<128x128x28x28xf32>
    %v3902 = stablehlo.reduce(%v3901 init: %v3895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3903 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3904 = stablehlo.divide %v3902, %v3903 : tensor<128xf32>
    %v3905 = stablehlo.reshape %v372 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3907 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3908 = stablehlo.reduce(%v3905 init: %v3906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3909 = stablehlo.divide %v3908, %v3907 : tensor<128xf32>
    %v3910 = stablehlo.reshape %v372 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3912 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v3913 = stablehlo.reduce(%v3910 init: %v3911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3914 = stablehlo.broadcast_in_dim %v3913, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3915 = stablehlo.divide %v3914, %v3912 : tensor<128x128x28x28xf32>
    %v3916 = stablehlo.subtract %v3910, %v3915 : tensor<128x128x28x28xf32>
    %v3917 = stablehlo.multiply %v3916, %v3916 : tensor<128x128x28x28xf32>
    %v3918 = stablehlo.reduce(%v3917 init: %v3911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3919 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3920 = stablehlo.divide %v3918, %v3919 : tensor<128xf32>
    %v3921 = stablehlo.reshape %v401 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3922 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3923 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3924 = stablehlo.reduce(%v3921 init: %v3922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3925 = stablehlo.divide %v3924, %v3923 : tensor<128xf32>
    %v3926 = stablehlo.reshape %v401 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3928 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v3929 = stablehlo.reduce(%v3926 init: %v3927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3930 = stablehlo.broadcast_in_dim %v3929, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3931 = stablehlo.divide %v3930, %v3928 : tensor<128x128x28x28xf32>
    %v3932 = stablehlo.subtract %v3926, %v3931 : tensor<128x128x28x28xf32>
    %v3933 = stablehlo.multiply %v3932, %v3932 : tensor<128x128x28x28xf32>
    %v3934 = stablehlo.reduce(%v3933 init: %v3927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3935 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3936 = stablehlo.divide %v3934, %v3935 : tensor<128xf32>
    %v3937 = stablehlo.reshape %v434 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3939 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3940 = stablehlo.reduce(%v3937 init: %v3938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3941 = stablehlo.divide %v3940, %v3939 : tensor<128xf32>
    %v3942 = stablehlo.reshape %v434 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3943 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3944 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v3945 = stablehlo.reduce(%v3942 init: %v3943) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3946 = stablehlo.broadcast_in_dim %v3945, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3947 = stablehlo.divide %v3946, %v3944 : tensor<128x128x28x28xf32>
    %v3948 = stablehlo.subtract %v3942, %v3947 : tensor<128x128x28x28xf32>
    %v3949 = stablehlo.multiply %v3948, %v3948 : tensor<128x128x28x28xf32>
    %v3950 = stablehlo.reduce(%v3949 init: %v3943) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3951 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3952 = stablehlo.divide %v3950, %v3951 : tensor<128xf32>
    %v3953 = stablehlo.reshape %v463 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3955 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3956 = stablehlo.reduce(%v3953 init: %v3954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3957 = stablehlo.divide %v3956, %v3955 : tensor<128xf32>
    %v3958 = stablehlo.reshape %v463 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3960 = stablehlo.constant dense<100352.0> : tensor<128x128x28x28xf32>
    %v3961 = stablehlo.reduce(%v3958 init: %v3959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3962 = stablehlo.broadcast_in_dim %v3961, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3963 = stablehlo.divide %v3962, %v3960 : tensor<128x128x28x28xf32>
    %v3964 = stablehlo.subtract %v3958, %v3963 : tensor<128x128x28x28xf32>
    %v3965 = stablehlo.multiply %v3964, %v3964 : tensor<128x128x28x28xf32>
    %v3966 = stablehlo.reduce(%v3965 init: %v3959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3967 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3968 = stablehlo.divide %v3966, %v3967 : tensor<128xf32>
    %v3969 = stablehlo.reshape %v496 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3971 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v3972 = stablehlo.reduce(%v3969 init: %v3970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3973 = stablehlo.divide %v3972, %v3971 : tensor<256xf32>
    %v3974 = stablehlo.reshape %v496 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3976 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v3977 = stablehlo.reduce(%v3974 init: %v3975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3978 = stablehlo.broadcast_in_dim %v3977, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3979 = stablehlo.divide %v3978, %v3976 : tensor<128x256x14x14xf32>
    %v3980 = stablehlo.subtract %v3974, %v3979 : tensor<128x256x14x14xf32>
    %v3981 = stablehlo.multiply %v3980, %v3980 : tensor<128x256x14x14xf32>
    %v3982 = stablehlo.reduce(%v3981 init: %v3975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3983 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v3984 = stablehlo.divide %v3982, %v3983 : tensor<256xf32>
    %v3985 = stablehlo.reshape %v525 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3987 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v3988 = stablehlo.reduce(%v3985 init: %v3986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3989 = stablehlo.divide %v3988, %v3987 : tensor<256xf32>
    %v3990 = stablehlo.reshape %v525 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3992 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v3993 = stablehlo.reduce(%v3990 init: %v3991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3994 = stablehlo.broadcast_in_dim %v3993, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3995 = stablehlo.divide %v3994, %v3992 : tensor<128x256x14x14xf32>
    %v3996 = stablehlo.subtract %v3990, %v3995 : tensor<128x256x14x14xf32>
    %v3997 = stablehlo.multiply %v3996, %v3996 : tensor<128x256x14x14xf32>
    %v3998 = stablehlo.reduce(%v3997 init: %v3991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3999 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4000 = stablehlo.divide %v3998, %v3999 : tensor<256xf32>
    %v4001 = stablehlo.reshape %v550 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4003 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4004 = stablehlo.reduce(%v4001 init: %v4002) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4005 = stablehlo.divide %v4004, %v4003 : tensor<256xf32>
    %v4006 = stablehlo.reshape %v550 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4007 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4008 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v4009 = stablehlo.reduce(%v4006 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4010 = stablehlo.broadcast_in_dim %v4009, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v4011 = stablehlo.divide %v4010, %v4008 : tensor<128x256x14x14xf32>
    %v4012 = stablehlo.subtract %v4006, %v4011 : tensor<128x256x14x14xf32>
    %v4013 = stablehlo.multiply %v4012, %v4012 : tensor<128x256x14x14xf32>
    %v4014 = stablehlo.reduce(%v4013 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4015 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4016 = stablehlo.divide %v4014, %v4015 : tensor<256xf32>
    %v4017 = stablehlo.reshape %v583 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4019 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4020 = stablehlo.reduce(%v4017 init: %v4018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4021 = stablehlo.divide %v4020, %v4019 : tensor<256xf32>
    %v4022 = stablehlo.reshape %v583 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4024 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v4025 = stablehlo.reduce(%v4022 init: %v4023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4026 = stablehlo.broadcast_in_dim %v4025, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v4027 = stablehlo.divide %v4026, %v4024 : tensor<128x256x14x14xf32>
    %v4028 = stablehlo.subtract %v4022, %v4027 : tensor<128x256x14x14xf32>
    %v4029 = stablehlo.multiply %v4028, %v4028 : tensor<128x256x14x14xf32>
    %v4030 = stablehlo.reduce(%v4029 init: %v4023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4031 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4032 = stablehlo.divide %v4030, %v4031 : tensor<256xf32>
    %v4033 = stablehlo.reshape %v612 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4034 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4035 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4036 = stablehlo.reduce(%v4033 init: %v4034) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4037 = stablehlo.divide %v4036, %v4035 : tensor<256xf32>
    %v4038 = stablehlo.reshape %v612 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4040 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v4041 = stablehlo.reduce(%v4038 init: %v4039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4042 = stablehlo.broadcast_in_dim %v4041, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v4043 = stablehlo.divide %v4042, %v4040 : tensor<128x256x14x14xf32>
    %v4044 = stablehlo.subtract %v4038, %v4043 : tensor<128x256x14x14xf32>
    %v4045 = stablehlo.multiply %v4044, %v4044 : tensor<128x256x14x14xf32>
    %v4046 = stablehlo.reduce(%v4045 init: %v4039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4047 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4048 = stablehlo.divide %v4046, %v4047 : tensor<256xf32>
    %v4049 = stablehlo.reshape %v645 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4051 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4052 = stablehlo.reduce(%v4049 init: %v4050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4053 = stablehlo.divide %v4052, %v4051 : tensor<256xf32>
    %v4054 = stablehlo.reshape %v645 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4056 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v4057 = stablehlo.reduce(%v4054 init: %v4055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4058 = stablehlo.broadcast_in_dim %v4057, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v4059 = stablehlo.divide %v4058, %v4056 : tensor<128x256x14x14xf32>
    %v4060 = stablehlo.subtract %v4054, %v4059 : tensor<128x256x14x14xf32>
    %v4061 = stablehlo.multiply %v4060, %v4060 : tensor<128x256x14x14xf32>
    %v4062 = stablehlo.reduce(%v4061 init: %v4055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4063 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4064 = stablehlo.divide %v4062, %v4063 : tensor<256xf32>
    %v4065 = stablehlo.reshape %v674 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4066 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4067 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4068 = stablehlo.reduce(%v4065 init: %v4066) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4069 = stablehlo.divide %v4068, %v4067 : tensor<256xf32>
    %v4070 = stablehlo.reshape %v674 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4071 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4072 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v4073 = stablehlo.reduce(%v4070 init: %v4071) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4074 = stablehlo.broadcast_in_dim %v4073, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v4075 = stablehlo.divide %v4074, %v4072 : tensor<128x256x14x14xf32>
    %v4076 = stablehlo.subtract %v4070, %v4075 : tensor<128x256x14x14xf32>
    %v4077 = stablehlo.multiply %v4076, %v4076 : tensor<128x256x14x14xf32>
    %v4078 = stablehlo.reduce(%v4077 init: %v4071) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4079 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4080 = stablehlo.divide %v4078, %v4079 : tensor<256xf32>
    %v4081 = stablehlo.reshape %v707 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4082 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4083 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4084 = stablehlo.reduce(%v4081 init: %v4082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4085 = stablehlo.divide %v4084, %v4083 : tensor<256xf32>
    %v4086 = stablehlo.reshape %v707 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4088 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v4089 = stablehlo.reduce(%v4086 init: %v4087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4090 = stablehlo.broadcast_in_dim %v4089, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v4091 = stablehlo.divide %v4090, %v4088 : tensor<128x256x14x14xf32>
    %v4092 = stablehlo.subtract %v4086, %v4091 : tensor<128x256x14x14xf32>
    %v4093 = stablehlo.multiply %v4092, %v4092 : tensor<128x256x14x14xf32>
    %v4094 = stablehlo.reduce(%v4093 init: %v4087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4095 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4096 = stablehlo.divide %v4094, %v4095 : tensor<256xf32>
    %v4097 = stablehlo.reshape %v736 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4098 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4099 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4100 = stablehlo.reduce(%v4097 init: %v4098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4101 = stablehlo.divide %v4100, %v4099 : tensor<256xf32>
    %v4102 = stablehlo.reshape %v736 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4104 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v4105 = stablehlo.reduce(%v4102 init: %v4103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4106 = stablehlo.broadcast_in_dim %v4105, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v4107 = stablehlo.divide %v4106, %v4104 : tensor<128x256x14x14xf32>
    %v4108 = stablehlo.subtract %v4102, %v4107 : tensor<128x256x14x14xf32>
    %v4109 = stablehlo.multiply %v4108, %v4108 : tensor<128x256x14x14xf32>
    %v4110 = stablehlo.reduce(%v4109 init: %v4103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4111 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4112 = stablehlo.divide %v4110, %v4111 : tensor<256xf32>
    %v4113 = stablehlo.reshape %v769 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4115 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4116 = stablehlo.reduce(%v4113 init: %v4114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4117 = stablehlo.divide %v4116, %v4115 : tensor<256xf32>
    %v4118 = stablehlo.reshape %v769 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4120 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v4121 = stablehlo.reduce(%v4118 init: %v4119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4122 = stablehlo.broadcast_in_dim %v4121, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v4123 = stablehlo.divide %v4122, %v4120 : tensor<128x256x14x14xf32>
    %v4124 = stablehlo.subtract %v4118, %v4123 : tensor<128x256x14x14xf32>
    %v4125 = stablehlo.multiply %v4124, %v4124 : tensor<128x256x14x14xf32>
    %v4126 = stablehlo.reduce(%v4125 init: %v4119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4127 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4128 = stablehlo.divide %v4126, %v4127 : tensor<256xf32>
    %v4129 = stablehlo.reshape %v798 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4131 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4132 = stablehlo.reduce(%v4129 init: %v4130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4133 = stablehlo.divide %v4132, %v4131 : tensor<256xf32>
    %v4134 = stablehlo.reshape %v798 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4135 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4136 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v4137 = stablehlo.reduce(%v4134 init: %v4135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4138 = stablehlo.broadcast_in_dim %v4137, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v4139 = stablehlo.divide %v4138, %v4136 : tensor<128x256x14x14xf32>
    %v4140 = stablehlo.subtract %v4134, %v4139 : tensor<128x256x14x14xf32>
    %v4141 = stablehlo.multiply %v4140, %v4140 : tensor<128x256x14x14xf32>
    %v4142 = stablehlo.reduce(%v4141 init: %v4135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4143 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4144 = stablehlo.divide %v4142, %v4143 : tensor<256xf32>
    %v4145 = stablehlo.reshape %v831 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4147 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4148 = stablehlo.reduce(%v4145 init: %v4146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4149 = stablehlo.divide %v4148, %v4147 : tensor<256xf32>
    %v4150 = stablehlo.reshape %v831 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4152 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v4153 = stablehlo.reduce(%v4150 init: %v4151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4154 = stablehlo.broadcast_in_dim %v4153, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v4155 = stablehlo.divide %v4154, %v4152 : tensor<128x256x14x14xf32>
    %v4156 = stablehlo.subtract %v4150, %v4155 : tensor<128x256x14x14xf32>
    %v4157 = stablehlo.multiply %v4156, %v4156 : tensor<128x256x14x14xf32>
    %v4158 = stablehlo.reduce(%v4157 init: %v4151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4159 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4160 = stablehlo.divide %v4158, %v4159 : tensor<256xf32>
    %v4161 = stablehlo.reshape %v860 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4163 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4164 = stablehlo.reduce(%v4161 init: %v4162) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4165 = stablehlo.divide %v4164, %v4163 : tensor<256xf32>
    %v4166 = stablehlo.reshape %v860 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v4167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4168 = stablehlo.constant dense<25088.0> : tensor<128x256x14x14xf32>
    %v4169 = stablehlo.reduce(%v4166 init: %v4167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4170 = stablehlo.broadcast_in_dim %v4169, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v4171 = stablehlo.divide %v4170, %v4168 : tensor<128x256x14x14xf32>
    %v4172 = stablehlo.subtract %v4166, %v4171 : tensor<128x256x14x14xf32>
    %v4173 = stablehlo.multiply %v4172, %v4172 : tensor<128x256x14x14xf32>
    %v4174 = stablehlo.reduce(%v4173 init: %v4167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4175 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v4176 = stablehlo.divide %v4174, %v4175 : tensor<256xf32>
    %v4177 = stablehlo.reshape %v893 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v4178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4179 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v4180 = stablehlo.reduce(%v4177 init: %v4178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4181 = stablehlo.divide %v4180, %v4179 : tensor<512xf32>
    %v4182 = stablehlo.reshape %v893 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v4183 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4184 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v4185 = stablehlo.reduce(%v4182 init: %v4183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4186 = stablehlo.broadcast_in_dim %v4185, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v4187 = stablehlo.divide %v4186, %v4184 : tensor<128x512x7x7xf32>
    %v4188 = stablehlo.subtract %v4182, %v4187 : tensor<128x512x7x7xf32>
    %v4189 = stablehlo.multiply %v4188, %v4188 : tensor<128x512x7x7xf32>
    %v4190 = stablehlo.reduce(%v4189 init: %v4183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4191 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v4192 = stablehlo.divide %v4190, %v4191 : tensor<512xf32>
    %v4193 = stablehlo.reshape %v922 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v4194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4195 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v4196 = stablehlo.reduce(%v4193 init: %v4194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4197 = stablehlo.divide %v4196, %v4195 : tensor<512xf32>
    %v4198 = stablehlo.reshape %v922 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v4199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4200 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v4201 = stablehlo.reduce(%v4198 init: %v4199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4202 = stablehlo.broadcast_in_dim %v4201, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v4203 = stablehlo.divide %v4202, %v4200 : tensor<128x512x7x7xf32>
    %v4204 = stablehlo.subtract %v4198, %v4203 : tensor<128x512x7x7xf32>
    %v4205 = stablehlo.multiply %v4204, %v4204 : tensor<128x512x7x7xf32>
    %v4206 = stablehlo.reduce(%v4205 init: %v4199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4207 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v4208 = stablehlo.divide %v4206, %v4207 : tensor<512xf32>
    %v4209 = stablehlo.reshape %v947 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v4210 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4211 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v4212 = stablehlo.reduce(%v4209 init: %v4210) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4213 = stablehlo.divide %v4212, %v4211 : tensor<512xf32>
    %v4214 = stablehlo.reshape %v947 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v4215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4216 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v4217 = stablehlo.reduce(%v4214 init: %v4215) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4218 = stablehlo.broadcast_in_dim %v4217, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v4219 = stablehlo.divide %v4218, %v4216 : tensor<128x512x7x7xf32>
    %v4220 = stablehlo.subtract %v4214, %v4219 : tensor<128x512x7x7xf32>
    %v4221 = stablehlo.multiply %v4220, %v4220 : tensor<128x512x7x7xf32>
    %v4222 = stablehlo.reduce(%v4221 init: %v4215) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4223 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v4224 = stablehlo.divide %v4222, %v4223 : tensor<512xf32>
    %v4225 = stablehlo.reshape %v980 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v4226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4227 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v4228 = stablehlo.reduce(%v4225 init: %v4226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4229 = stablehlo.divide %v4228, %v4227 : tensor<512xf32>
    %v4230 = stablehlo.reshape %v980 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v4231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4232 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v4233 = stablehlo.reduce(%v4230 init: %v4231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4234 = stablehlo.broadcast_in_dim %v4233, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v4235 = stablehlo.divide %v4234, %v4232 : tensor<128x512x7x7xf32>
    %v4236 = stablehlo.subtract %v4230, %v4235 : tensor<128x512x7x7xf32>
    %v4237 = stablehlo.multiply %v4236, %v4236 : tensor<128x512x7x7xf32>
    %v4238 = stablehlo.reduce(%v4237 init: %v4231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4239 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v4240 = stablehlo.divide %v4238, %v4239 : tensor<512xf32>
    %v4241 = stablehlo.reshape %v1009 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v4242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4243 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v4244 = stablehlo.reduce(%v4241 init: %v4242) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4245 = stablehlo.divide %v4244, %v4243 : tensor<512xf32>
    %v4246 = stablehlo.reshape %v1009 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v4247 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4248 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v4249 = stablehlo.reduce(%v4246 init: %v4247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4250 = stablehlo.broadcast_in_dim %v4249, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v4251 = stablehlo.divide %v4250, %v4248 : tensor<128x512x7x7xf32>
    %v4252 = stablehlo.subtract %v4246, %v4251 : tensor<128x512x7x7xf32>
    %v4253 = stablehlo.multiply %v4252, %v4252 : tensor<128x512x7x7xf32>
    %v4254 = stablehlo.reduce(%v4253 init: %v4247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4255 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v4256 = stablehlo.divide %v4254, %v4255 : tensor<512xf32>
    %v4257 = stablehlo.reshape %v1042 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v4258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4259 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v4260 = stablehlo.reduce(%v4257 init: %v4258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4261 = stablehlo.divide %v4260, %v4259 : tensor<512xf32>
    %v4262 = stablehlo.reshape %v1042 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v4263 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4264 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v4265 = stablehlo.reduce(%v4262 init: %v4263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4266 = stablehlo.broadcast_in_dim %v4265, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v4267 = stablehlo.divide %v4266, %v4264 : tensor<128x512x7x7xf32>
    %v4268 = stablehlo.subtract %v4262, %v4267 : tensor<128x512x7x7xf32>
    %v4269 = stablehlo.multiply %v4268, %v4268 : tensor<128x512x7x7xf32>
    %v4270 = stablehlo.reduce(%v4269 init: %v4263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4271 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v4272 = stablehlo.divide %v4270, %v4271 : tensor<512xf32>
    %v4273 = stablehlo.reshape %v1071 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v4274 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4275 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v4276 = stablehlo.reduce(%v4273 init: %v4274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4277 = stablehlo.divide %v4276, %v4275 : tensor<512xf32>
    %v4278 = stablehlo.reshape %v1071 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v4279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4280 = stablehlo.constant dense<6272.0> : tensor<128x512x7x7xf32>
    %v4281 = stablehlo.reduce(%v4278 init: %v4279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4282 = stablehlo.broadcast_in_dim %v4281, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v4283 = stablehlo.divide %v4282, %v4280 : tensor<128x512x7x7xf32>
    %v4284 = stablehlo.subtract %v4278, %v4283 : tensor<128x512x7x7xf32>
    %v4285 = stablehlo.multiply %v4284, %v4284 : tensor<128x512x7x7xf32>
    %v4286 = stablehlo.reduce(%v4285 init: %v4279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4287 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v4288 = stablehlo.divide %v4286, %v4287 : tensor<512xf32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %v4289 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4290 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4291 = stablehlo.multiply %v4289, %sWm : tensor<64x3x7x7xf32>
    %v4292 = stablehlo.multiply %v4290, %v3691 : tensor<64x3x7x7xf32>
    %v4293 = stablehlo.add %v4291, %v4292 : tensor<64x3x7x7xf32>
    %v4294 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4295 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4296 = stablehlo.multiply %v4294, %sWv : tensor<64x3x7x7xf32>
    %v4297 = stablehlo.multiply %v3691, %v3691 : tensor<64x3x7x7xf32>
    %v4298 = stablehlo.multiply %v4295, %v4297 : tensor<64x3x7x7xf32>
    %v4299 = stablehlo.add %v4296, %v4298 : tensor<64x3x7x7xf32>
    %v4300 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4301 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4302 = stablehlo.multiply %v4300, %sWm : tensor<64x3x7x7xf32>
    %v4303 = stablehlo.multiply %v4301, %v3691 : tensor<64x3x7x7xf32>
    %v4304 = stablehlo.add %v4302, %v4303 : tensor<64x3x7x7xf32>
    %v4305 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4306 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4307 = stablehlo.multiply %v4305, %sWv : tensor<64x3x7x7xf32>
    %v4308 = stablehlo.multiply %v3691, %v3691 : tensor<64x3x7x7xf32>
    %v4309 = stablehlo.multiply %v4306, %v4308 : tensor<64x3x7x7xf32>
    %v4310 = stablehlo.add %v4307, %v4309 : tensor<64x3x7x7xf32>
    %v4311 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4312 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4313 = stablehlo.divide %v4304, %v4311 : tensor<64x3x7x7xf32>
    %v4314 = stablehlo.divide %v4310, %v4312 : tensor<64x3x7x7xf32>
    %v4315 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4316 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4317 = stablehlo.sqrt %v4314 : tensor<64x3x7x7xf32>
    %v4318 = stablehlo.add %v4317, %v4316 : tensor<64x3x7x7xf32>
    %v4319 = stablehlo.divide %v4313, %v4318 : tensor<64x3x7x7xf32>
    %v4320 = stablehlo.multiply %v4315, %v4319 : tensor<64x3x7x7xf32>
    %v4321 = stablehlo.subtract %sW, %v4320 : tensor<64x3x7x7xf32>
    %v4322 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4323 = stablehlo.multiply %v4322, %v4315 : tensor<64x3x7x7xf32>
    %v4324 = stablehlo.multiply %v4323, %sW : tensor<64x3x7x7xf32>
    %v4325 = stablehlo.subtract %v4321, %v4324 : tensor<64x3x7x7xf32>
    %v4326 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4327 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4328 = stablehlo.multiply %v4326, %sgm : tensor<64xf32>
    %v4329 = stablehlo.multiply %v4327, %v3709 : tensor<64xf32>
    %v4330 = stablehlo.add %v4328, %v4329 : tensor<64xf32>
    %v4331 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4332 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4333 = stablehlo.multiply %v4331, %sgv : tensor<64xf32>
    %v4334 = stablehlo.multiply %v3709, %v3709 : tensor<64xf32>
    %v4335 = stablehlo.multiply %v4332, %v4334 : tensor<64xf32>
    %v4336 = stablehlo.add %v4333, %v4335 : tensor<64xf32>
    %v4337 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4338 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4339 = stablehlo.multiply %v4337, %sgm : tensor<64xf32>
    %v4340 = stablehlo.multiply %v4338, %v3709 : tensor<64xf32>
    %v4341 = stablehlo.add %v4339, %v4340 : tensor<64xf32>
    %v4342 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4343 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4344 = stablehlo.multiply %v4342, %sgv : tensor<64xf32>
    %v4345 = stablehlo.multiply %v3709, %v3709 : tensor<64xf32>
    %v4346 = stablehlo.multiply %v4343, %v4345 : tensor<64xf32>
    %v4347 = stablehlo.add %v4344, %v4346 : tensor<64xf32>
    %v4348 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4349 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4350 = stablehlo.divide %v4341, %v4348 : tensor<64xf32>
    %v4351 = stablehlo.divide %v4347, %v4349 : tensor<64xf32>
    %v4352 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4353 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4354 = stablehlo.sqrt %v4351 : tensor<64xf32>
    %v4355 = stablehlo.add %v4354, %v4353 : tensor<64xf32>
    %v4356 = stablehlo.divide %v4350, %v4355 : tensor<64xf32>
    %v4357 = stablehlo.multiply %v4352, %v4356 : tensor<64xf32>
    %v4358 = stablehlo.subtract %sg, %v4357 : tensor<64xf32>
    %v4359 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4360 = stablehlo.multiply %v4359, %v4352 : tensor<64xf32>
    %v4361 = stablehlo.multiply %v4360, %sg : tensor<64xf32>
    %v4362 = stablehlo.subtract %v4358, %v4361 : tensor<64xf32>
    %v4363 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4364 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4365 = stablehlo.multiply %v4363, %sbtm : tensor<64xf32>
    %v4366 = stablehlo.multiply %v4364, %v3712 : tensor<64xf32>
    %v4367 = stablehlo.add %v4365, %v4366 : tensor<64xf32>
    %v4368 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4369 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4370 = stablehlo.multiply %v4368, %sbtv : tensor<64xf32>
    %v4371 = stablehlo.multiply %v3712, %v3712 : tensor<64xf32>
    %v4372 = stablehlo.multiply %v4369, %v4371 : tensor<64xf32>
    %v4373 = stablehlo.add %v4370, %v4372 : tensor<64xf32>
    %v4374 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4375 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4376 = stablehlo.multiply %v4374, %sbtm : tensor<64xf32>
    %v4377 = stablehlo.multiply %v4375, %v3712 : tensor<64xf32>
    %v4378 = stablehlo.add %v4376, %v4377 : tensor<64xf32>
    %v4379 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4380 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4381 = stablehlo.multiply %v4379, %sbtv : tensor<64xf32>
    %v4382 = stablehlo.multiply %v3712, %v3712 : tensor<64xf32>
    %v4383 = stablehlo.multiply %v4380, %v4382 : tensor<64xf32>
    %v4384 = stablehlo.add %v4381, %v4383 : tensor<64xf32>
    %v4385 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4386 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4387 = stablehlo.divide %v4378, %v4385 : tensor<64xf32>
    %v4388 = stablehlo.divide %v4384, %v4386 : tensor<64xf32>
    %v4389 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4390 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4391 = stablehlo.sqrt %v4388 : tensor<64xf32>
    %v4392 = stablehlo.add %v4391, %v4390 : tensor<64xf32>
    %v4393 = stablehlo.divide %v4387, %v4392 : tensor<64xf32>
    %v4394 = stablehlo.multiply %v4389, %v4393 : tensor<64xf32>
    %v4395 = stablehlo.subtract %sbt, %v4394 : tensor<64xf32>
    %v4396 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4397 = stablehlo.multiply %v4396, %v4389 : tensor<64xf32>
    %v4398 = stablehlo.multiply %v4397, %sbt : tensor<64xf32>
    %v4399 = stablehlo.subtract %v4395, %v4398 : tensor<64xf32>
    %v4400 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4401 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4402 = stablehlo.multiply %v4400, %s1b0W1m : tensor<64x64x3x3xf32>
    %v4403 = stablehlo.multiply %v4401, %v3516 : tensor<64x64x3x3xf32>
    %v4404 = stablehlo.add %v4402, %v4403 : tensor<64x64x3x3xf32>
    %v4405 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4406 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4407 = stablehlo.multiply %v4405, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4408 = stablehlo.multiply %v3516, %v3516 : tensor<64x64x3x3xf32>
    %v4409 = stablehlo.multiply %v4406, %v4408 : tensor<64x64x3x3xf32>
    %v4410 = stablehlo.add %v4407, %v4409 : tensor<64x64x3x3xf32>
    %v4411 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4412 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4413 = stablehlo.multiply %v4411, %s1b0W1m : tensor<64x64x3x3xf32>
    %v4414 = stablehlo.multiply %v4412, %v3516 : tensor<64x64x3x3xf32>
    %v4415 = stablehlo.add %v4413, %v4414 : tensor<64x64x3x3xf32>
    %v4416 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4417 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4418 = stablehlo.multiply %v4416, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4419 = stablehlo.multiply %v3516, %v3516 : tensor<64x64x3x3xf32>
    %v4420 = stablehlo.multiply %v4417, %v4419 : tensor<64x64x3x3xf32>
    %v4421 = stablehlo.add %v4418, %v4420 : tensor<64x64x3x3xf32>
    %v4422 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4423 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4424 = stablehlo.divide %v4415, %v4422 : tensor<64x64x3x3xf32>
    %v4425 = stablehlo.divide %v4421, %v4423 : tensor<64x64x3x3xf32>
    %v4426 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4427 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4428 = stablehlo.sqrt %v4425 : tensor<64x64x3x3xf32>
    %v4429 = stablehlo.add %v4428, %v4427 : tensor<64x64x3x3xf32>
    %v4430 = stablehlo.divide %v4424, %v4429 : tensor<64x64x3x3xf32>
    %v4431 = stablehlo.multiply %v4426, %v4430 : tensor<64x64x3x3xf32>
    %v4432 = stablehlo.subtract %s1b0W1, %v4431 : tensor<64x64x3x3xf32>
    %v4433 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4434 = stablehlo.multiply %v4433, %v4426 : tensor<64x64x3x3xf32>
    %v4435 = stablehlo.multiply %v4434, %s1b0W1 : tensor<64x64x3x3xf32>
    %v4436 = stablehlo.subtract %v4432, %v4435 : tensor<64x64x3x3xf32>
    %v4437 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4438 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4439 = stablehlo.multiply %v4437, %s1b0g1m : tensor<64xf32>
    %v4440 = stablehlo.multiply %v4438, %v3534 : tensor<64xf32>
    %v4441 = stablehlo.add %v4439, %v4440 : tensor<64xf32>
    %v4442 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4443 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4444 = stablehlo.multiply %v4442, %s1b0g1v : tensor<64xf32>
    %v4445 = stablehlo.multiply %v3534, %v3534 : tensor<64xf32>
    %v4446 = stablehlo.multiply %v4443, %v4445 : tensor<64xf32>
    %v4447 = stablehlo.add %v4444, %v4446 : tensor<64xf32>
    %v4448 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4449 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4450 = stablehlo.multiply %v4448, %s1b0g1m : tensor<64xf32>
    %v4451 = stablehlo.multiply %v4449, %v3534 : tensor<64xf32>
    %v4452 = stablehlo.add %v4450, %v4451 : tensor<64xf32>
    %v4453 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4454 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4455 = stablehlo.multiply %v4453, %s1b0g1v : tensor<64xf32>
    %v4456 = stablehlo.multiply %v3534, %v3534 : tensor<64xf32>
    %v4457 = stablehlo.multiply %v4454, %v4456 : tensor<64xf32>
    %v4458 = stablehlo.add %v4455, %v4457 : tensor<64xf32>
    %v4459 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4460 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4461 = stablehlo.divide %v4452, %v4459 : tensor<64xf32>
    %v4462 = stablehlo.divide %v4458, %v4460 : tensor<64xf32>
    %v4463 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4464 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4465 = stablehlo.sqrt %v4462 : tensor<64xf32>
    %v4466 = stablehlo.add %v4465, %v4464 : tensor<64xf32>
    %v4467 = stablehlo.divide %v4461, %v4466 : tensor<64xf32>
    %v4468 = stablehlo.multiply %v4463, %v4467 : tensor<64xf32>
    %v4469 = stablehlo.subtract %s1b0g1, %v4468 : tensor<64xf32>
    %v4470 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4471 = stablehlo.multiply %v4470, %v4463 : tensor<64xf32>
    %v4472 = stablehlo.multiply %v4471, %s1b0g1 : tensor<64xf32>
    %v4473 = stablehlo.subtract %v4469, %v4472 : tensor<64xf32>
    %v4474 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4475 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4476 = stablehlo.multiply %v4474, %s1b0bt1m : tensor<64xf32>
    %v4477 = stablehlo.multiply %v4475, %v3537 : tensor<64xf32>
    %v4478 = stablehlo.add %v4476, %v4477 : tensor<64xf32>
    %v4479 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4480 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4481 = stablehlo.multiply %v4479, %s1b0bt1v : tensor<64xf32>
    %v4482 = stablehlo.multiply %v3537, %v3537 : tensor<64xf32>
    %v4483 = stablehlo.multiply %v4480, %v4482 : tensor<64xf32>
    %v4484 = stablehlo.add %v4481, %v4483 : tensor<64xf32>
    %v4485 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4486 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4487 = stablehlo.multiply %v4485, %s1b0bt1m : tensor<64xf32>
    %v4488 = stablehlo.multiply %v4486, %v3537 : tensor<64xf32>
    %v4489 = stablehlo.add %v4487, %v4488 : tensor<64xf32>
    %v4490 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4491 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4492 = stablehlo.multiply %v4490, %s1b0bt1v : tensor<64xf32>
    %v4493 = stablehlo.multiply %v3537, %v3537 : tensor<64xf32>
    %v4494 = stablehlo.multiply %v4491, %v4493 : tensor<64xf32>
    %v4495 = stablehlo.add %v4492, %v4494 : tensor<64xf32>
    %v4496 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4497 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4498 = stablehlo.divide %v4489, %v4496 : tensor<64xf32>
    %v4499 = stablehlo.divide %v4495, %v4497 : tensor<64xf32>
    %v4500 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4501 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4502 = stablehlo.sqrt %v4499 : tensor<64xf32>
    %v4503 = stablehlo.add %v4502, %v4501 : tensor<64xf32>
    %v4504 = stablehlo.divide %v4498, %v4503 : tensor<64xf32>
    %v4505 = stablehlo.multiply %v4500, %v4504 : tensor<64xf32>
    %v4506 = stablehlo.subtract %s1b0bt1, %v4505 : tensor<64xf32>
    %v4507 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4508 = stablehlo.multiply %v4507, %v4500 : tensor<64xf32>
    %v4509 = stablehlo.multiply %v4508, %s1b0bt1 : tensor<64xf32>
    %v4510 = stablehlo.subtract %v4506, %v4509 : tensor<64xf32>
    %v4511 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4512 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4513 = stablehlo.multiply %v4511, %s1b0W2m : tensor<64x64x3x3xf32>
    %v4514 = stablehlo.multiply %v4512, %v3543 : tensor<64x64x3x3xf32>
    %v4515 = stablehlo.add %v4513, %v4514 : tensor<64x64x3x3xf32>
    %v4516 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4517 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4518 = stablehlo.multiply %v4516, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4519 = stablehlo.multiply %v3543, %v3543 : tensor<64x64x3x3xf32>
    %v4520 = stablehlo.multiply %v4517, %v4519 : tensor<64x64x3x3xf32>
    %v4521 = stablehlo.add %v4518, %v4520 : tensor<64x64x3x3xf32>
    %v4522 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4523 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4524 = stablehlo.multiply %v4522, %s1b0W2m : tensor<64x64x3x3xf32>
    %v4525 = stablehlo.multiply %v4523, %v3543 : tensor<64x64x3x3xf32>
    %v4526 = stablehlo.add %v4524, %v4525 : tensor<64x64x3x3xf32>
    %v4527 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4528 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4529 = stablehlo.multiply %v4527, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4530 = stablehlo.multiply %v3543, %v3543 : tensor<64x64x3x3xf32>
    %v4531 = stablehlo.multiply %v4528, %v4530 : tensor<64x64x3x3xf32>
    %v4532 = stablehlo.add %v4529, %v4531 : tensor<64x64x3x3xf32>
    %v4533 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4534 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4535 = stablehlo.divide %v4526, %v4533 : tensor<64x64x3x3xf32>
    %v4536 = stablehlo.divide %v4532, %v4534 : tensor<64x64x3x3xf32>
    %v4537 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4538 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4539 = stablehlo.sqrt %v4536 : tensor<64x64x3x3xf32>
    %v4540 = stablehlo.add %v4539, %v4538 : tensor<64x64x3x3xf32>
    %v4541 = stablehlo.divide %v4535, %v4540 : tensor<64x64x3x3xf32>
    %v4542 = stablehlo.multiply %v4537, %v4541 : tensor<64x64x3x3xf32>
    %v4543 = stablehlo.subtract %s1b0W2, %v4542 : tensor<64x64x3x3xf32>
    %v4544 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4545 = stablehlo.multiply %v4544, %v4537 : tensor<64x64x3x3xf32>
    %v4546 = stablehlo.multiply %v4545, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4547 = stablehlo.subtract %v4543, %v4546 : tensor<64x64x3x3xf32>
    %v4548 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4549 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4550 = stablehlo.multiply %v4548, %s1b0g2m : tensor<64xf32>
    %v4551 = stablehlo.multiply %v4549, %v3561 : tensor<64xf32>
    %v4552 = stablehlo.add %v4550, %v4551 : tensor<64xf32>
    %v4553 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4554 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4555 = stablehlo.multiply %v4553, %s1b0g2v : tensor<64xf32>
    %v4556 = stablehlo.multiply %v3561, %v3561 : tensor<64xf32>
    %v4557 = stablehlo.multiply %v4554, %v4556 : tensor<64xf32>
    %v4558 = stablehlo.add %v4555, %v4557 : tensor<64xf32>
    %v4559 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4560 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4561 = stablehlo.multiply %v4559, %s1b0g2m : tensor<64xf32>
    %v4562 = stablehlo.multiply %v4560, %v3561 : tensor<64xf32>
    %v4563 = stablehlo.add %v4561, %v4562 : tensor<64xf32>
    %v4564 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4565 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4566 = stablehlo.multiply %v4564, %s1b0g2v : tensor<64xf32>
    %v4567 = stablehlo.multiply %v3561, %v3561 : tensor<64xf32>
    %v4568 = stablehlo.multiply %v4565, %v4567 : tensor<64xf32>
    %v4569 = stablehlo.add %v4566, %v4568 : tensor<64xf32>
    %v4570 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4571 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4572 = stablehlo.divide %v4563, %v4570 : tensor<64xf32>
    %v4573 = stablehlo.divide %v4569, %v4571 : tensor<64xf32>
    %v4574 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4575 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4576 = stablehlo.sqrt %v4573 : tensor<64xf32>
    %v4577 = stablehlo.add %v4576, %v4575 : tensor<64xf32>
    %v4578 = stablehlo.divide %v4572, %v4577 : tensor<64xf32>
    %v4579 = stablehlo.multiply %v4574, %v4578 : tensor<64xf32>
    %v4580 = stablehlo.subtract %s1b0g2, %v4579 : tensor<64xf32>
    %v4581 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4582 = stablehlo.multiply %v4581, %v4574 : tensor<64xf32>
    %v4583 = stablehlo.multiply %v4582, %s1b0g2 : tensor<64xf32>
    %v4584 = stablehlo.subtract %v4580, %v4583 : tensor<64xf32>
    %v4585 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4586 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4587 = stablehlo.multiply %v4585, %s1b0bt2m : tensor<64xf32>
    %v4588 = stablehlo.multiply %v4586, %v3564 : tensor<64xf32>
    %v4589 = stablehlo.add %v4587, %v4588 : tensor<64xf32>
    %v4590 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4591 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4592 = stablehlo.multiply %v4590, %s1b0bt2v : tensor<64xf32>
    %v4593 = stablehlo.multiply %v3564, %v3564 : tensor<64xf32>
    %v4594 = stablehlo.multiply %v4591, %v4593 : tensor<64xf32>
    %v4595 = stablehlo.add %v4592, %v4594 : tensor<64xf32>
    %v4596 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4597 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4598 = stablehlo.multiply %v4596, %s1b0bt2m : tensor<64xf32>
    %v4599 = stablehlo.multiply %v4597, %v3564 : tensor<64xf32>
    %v4600 = stablehlo.add %v4598, %v4599 : tensor<64xf32>
    %v4601 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4602 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4603 = stablehlo.multiply %v4601, %s1b0bt2v : tensor<64xf32>
    %v4604 = stablehlo.multiply %v3564, %v3564 : tensor<64xf32>
    %v4605 = stablehlo.multiply %v4602, %v4604 : tensor<64xf32>
    %v4606 = stablehlo.add %v4603, %v4605 : tensor<64xf32>
    %v4607 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4608 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4609 = stablehlo.divide %v4600, %v4607 : tensor<64xf32>
    %v4610 = stablehlo.divide %v4606, %v4608 : tensor<64xf32>
    %v4611 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4612 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4613 = stablehlo.sqrt %v4610 : tensor<64xf32>
    %v4614 = stablehlo.add %v4613, %v4612 : tensor<64xf32>
    %v4615 = stablehlo.divide %v4609, %v4614 : tensor<64xf32>
    %v4616 = stablehlo.multiply %v4611, %v4615 : tensor<64xf32>
    %v4617 = stablehlo.subtract %s1b0bt2, %v4616 : tensor<64xf32>
    %v4618 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4619 = stablehlo.multiply %v4618, %v4611 : tensor<64xf32>
    %v4620 = stablehlo.multiply %v4619, %s1b0bt2 : tensor<64xf32>
    %v4621 = stablehlo.subtract %v4617, %v4620 : tensor<64xf32>
    %v4622 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4623 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4624 = stablehlo.multiply %v4622, %s1b1W1m : tensor<64x64x3x3xf32>
    %v4625 = stablehlo.multiply %v4623, %v3376 : tensor<64x64x3x3xf32>
    %v4626 = stablehlo.add %v4624, %v4625 : tensor<64x64x3x3xf32>
    %v4627 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4628 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4629 = stablehlo.multiply %v4627, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4630 = stablehlo.multiply %v3376, %v3376 : tensor<64x64x3x3xf32>
    %v4631 = stablehlo.multiply %v4628, %v4630 : tensor<64x64x3x3xf32>
    %v4632 = stablehlo.add %v4629, %v4631 : tensor<64x64x3x3xf32>
    %v4633 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4634 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4635 = stablehlo.multiply %v4633, %s1b1W1m : tensor<64x64x3x3xf32>
    %v4636 = stablehlo.multiply %v4634, %v3376 : tensor<64x64x3x3xf32>
    %v4637 = stablehlo.add %v4635, %v4636 : tensor<64x64x3x3xf32>
    %v4638 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4639 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4640 = stablehlo.multiply %v4638, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4641 = stablehlo.multiply %v3376, %v3376 : tensor<64x64x3x3xf32>
    %v4642 = stablehlo.multiply %v4639, %v4641 : tensor<64x64x3x3xf32>
    %v4643 = stablehlo.add %v4640, %v4642 : tensor<64x64x3x3xf32>
    %v4644 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4645 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4646 = stablehlo.divide %v4637, %v4644 : tensor<64x64x3x3xf32>
    %v4647 = stablehlo.divide %v4643, %v4645 : tensor<64x64x3x3xf32>
    %v4648 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4649 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4650 = stablehlo.sqrt %v4647 : tensor<64x64x3x3xf32>
    %v4651 = stablehlo.add %v4650, %v4649 : tensor<64x64x3x3xf32>
    %v4652 = stablehlo.divide %v4646, %v4651 : tensor<64x64x3x3xf32>
    %v4653 = stablehlo.multiply %v4648, %v4652 : tensor<64x64x3x3xf32>
    %v4654 = stablehlo.subtract %s1b1W1, %v4653 : tensor<64x64x3x3xf32>
    %v4655 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4656 = stablehlo.multiply %v4655, %v4648 : tensor<64x64x3x3xf32>
    %v4657 = stablehlo.multiply %v4656, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4658 = stablehlo.subtract %v4654, %v4657 : tensor<64x64x3x3xf32>
    %v4659 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4660 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4661 = stablehlo.multiply %v4659, %s1b1g1m : tensor<64xf32>
    %v4662 = stablehlo.multiply %v4660, %v3394 : tensor<64xf32>
    %v4663 = stablehlo.add %v4661, %v4662 : tensor<64xf32>
    %v4664 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4665 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4666 = stablehlo.multiply %v4664, %s1b1g1v : tensor<64xf32>
    %v4667 = stablehlo.multiply %v3394, %v3394 : tensor<64xf32>
    %v4668 = stablehlo.multiply %v4665, %v4667 : tensor<64xf32>
    %v4669 = stablehlo.add %v4666, %v4668 : tensor<64xf32>
    %v4670 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4671 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4672 = stablehlo.multiply %v4670, %s1b1g1m : tensor<64xf32>
    %v4673 = stablehlo.multiply %v4671, %v3394 : tensor<64xf32>
    %v4674 = stablehlo.add %v4672, %v4673 : tensor<64xf32>
    %v4675 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4676 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4677 = stablehlo.multiply %v4675, %s1b1g1v : tensor<64xf32>
    %v4678 = stablehlo.multiply %v3394, %v3394 : tensor<64xf32>
    %v4679 = stablehlo.multiply %v4676, %v4678 : tensor<64xf32>
    %v4680 = stablehlo.add %v4677, %v4679 : tensor<64xf32>
    %v4681 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4682 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4683 = stablehlo.divide %v4674, %v4681 : tensor<64xf32>
    %v4684 = stablehlo.divide %v4680, %v4682 : tensor<64xf32>
    %v4685 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4686 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4687 = stablehlo.sqrt %v4684 : tensor<64xf32>
    %v4688 = stablehlo.add %v4687, %v4686 : tensor<64xf32>
    %v4689 = stablehlo.divide %v4683, %v4688 : tensor<64xf32>
    %v4690 = stablehlo.multiply %v4685, %v4689 : tensor<64xf32>
    %v4691 = stablehlo.subtract %s1b1g1, %v4690 : tensor<64xf32>
    %v4692 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4693 = stablehlo.multiply %v4692, %v4685 : tensor<64xf32>
    %v4694 = stablehlo.multiply %v4693, %s1b1g1 : tensor<64xf32>
    %v4695 = stablehlo.subtract %v4691, %v4694 : tensor<64xf32>
    %v4696 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4697 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4698 = stablehlo.multiply %v4696, %s1b1bt1m : tensor<64xf32>
    %v4699 = stablehlo.multiply %v4697, %v3397 : tensor<64xf32>
    %v4700 = stablehlo.add %v4698, %v4699 : tensor<64xf32>
    %v4701 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4702 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4703 = stablehlo.multiply %v4701, %s1b1bt1v : tensor<64xf32>
    %v4704 = stablehlo.multiply %v3397, %v3397 : tensor<64xf32>
    %v4705 = stablehlo.multiply %v4702, %v4704 : tensor<64xf32>
    %v4706 = stablehlo.add %v4703, %v4705 : tensor<64xf32>
    %v4707 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4708 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4709 = stablehlo.multiply %v4707, %s1b1bt1m : tensor<64xf32>
    %v4710 = stablehlo.multiply %v4708, %v3397 : tensor<64xf32>
    %v4711 = stablehlo.add %v4709, %v4710 : tensor<64xf32>
    %v4712 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4713 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4714 = stablehlo.multiply %v4712, %s1b1bt1v : tensor<64xf32>
    %v4715 = stablehlo.multiply %v3397, %v3397 : tensor<64xf32>
    %v4716 = stablehlo.multiply %v4713, %v4715 : tensor<64xf32>
    %v4717 = stablehlo.add %v4714, %v4716 : tensor<64xf32>
    %v4718 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4719 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4720 = stablehlo.divide %v4711, %v4718 : tensor<64xf32>
    %v4721 = stablehlo.divide %v4717, %v4719 : tensor<64xf32>
    %v4722 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4723 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4724 = stablehlo.sqrt %v4721 : tensor<64xf32>
    %v4725 = stablehlo.add %v4724, %v4723 : tensor<64xf32>
    %v4726 = stablehlo.divide %v4720, %v4725 : tensor<64xf32>
    %v4727 = stablehlo.multiply %v4722, %v4726 : tensor<64xf32>
    %v4728 = stablehlo.subtract %s1b1bt1, %v4727 : tensor<64xf32>
    %v4729 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4730 = stablehlo.multiply %v4729, %v4722 : tensor<64xf32>
    %v4731 = stablehlo.multiply %v4730, %s1b1bt1 : tensor<64xf32>
    %v4732 = stablehlo.subtract %v4728, %v4731 : tensor<64xf32>
    %v4733 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4734 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4735 = stablehlo.multiply %v4733, %s1b1W2m : tensor<64x64x3x3xf32>
    %v4736 = stablehlo.multiply %v4734, %v3403 : tensor<64x64x3x3xf32>
    %v4737 = stablehlo.add %v4735, %v4736 : tensor<64x64x3x3xf32>
    %v4738 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4739 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4740 = stablehlo.multiply %v4738, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4741 = stablehlo.multiply %v3403, %v3403 : tensor<64x64x3x3xf32>
    %v4742 = stablehlo.multiply %v4739, %v4741 : tensor<64x64x3x3xf32>
    %v4743 = stablehlo.add %v4740, %v4742 : tensor<64x64x3x3xf32>
    %v4744 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4745 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4746 = stablehlo.multiply %v4744, %s1b1W2m : tensor<64x64x3x3xf32>
    %v4747 = stablehlo.multiply %v4745, %v3403 : tensor<64x64x3x3xf32>
    %v4748 = stablehlo.add %v4746, %v4747 : tensor<64x64x3x3xf32>
    %v4749 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4750 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4751 = stablehlo.multiply %v4749, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4752 = stablehlo.multiply %v3403, %v3403 : tensor<64x64x3x3xf32>
    %v4753 = stablehlo.multiply %v4750, %v4752 : tensor<64x64x3x3xf32>
    %v4754 = stablehlo.add %v4751, %v4753 : tensor<64x64x3x3xf32>
    %v4755 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4756 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4757 = stablehlo.divide %v4748, %v4755 : tensor<64x64x3x3xf32>
    %v4758 = stablehlo.divide %v4754, %v4756 : tensor<64x64x3x3xf32>
    %v4759 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4760 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4761 = stablehlo.sqrt %v4758 : tensor<64x64x3x3xf32>
    %v4762 = stablehlo.add %v4761, %v4760 : tensor<64x64x3x3xf32>
    %v4763 = stablehlo.divide %v4757, %v4762 : tensor<64x64x3x3xf32>
    %v4764 = stablehlo.multiply %v4759, %v4763 : tensor<64x64x3x3xf32>
    %v4765 = stablehlo.subtract %s1b1W2, %v4764 : tensor<64x64x3x3xf32>
    %v4766 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4767 = stablehlo.multiply %v4766, %v4759 : tensor<64x64x3x3xf32>
    %v4768 = stablehlo.multiply %v4767, %s1b1W2 : tensor<64x64x3x3xf32>
    %v4769 = stablehlo.subtract %v4765, %v4768 : tensor<64x64x3x3xf32>
    %v4770 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4771 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4772 = stablehlo.multiply %v4770, %s1b1g2m : tensor<64xf32>
    %v4773 = stablehlo.multiply %v4771, %v3421 : tensor<64xf32>
    %v4774 = stablehlo.add %v4772, %v4773 : tensor<64xf32>
    %v4775 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4776 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4777 = stablehlo.multiply %v4775, %s1b1g2v : tensor<64xf32>
    %v4778 = stablehlo.multiply %v3421, %v3421 : tensor<64xf32>
    %v4779 = stablehlo.multiply %v4776, %v4778 : tensor<64xf32>
    %v4780 = stablehlo.add %v4777, %v4779 : tensor<64xf32>
    %v4781 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4782 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4783 = stablehlo.multiply %v4781, %s1b1g2m : tensor<64xf32>
    %v4784 = stablehlo.multiply %v4782, %v3421 : tensor<64xf32>
    %v4785 = stablehlo.add %v4783, %v4784 : tensor<64xf32>
    %v4786 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4787 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4788 = stablehlo.multiply %v4786, %s1b1g2v : tensor<64xf32>
    %v4789 = stablehlo.multiply %v3421, %v3421 : tensor<64xf32>
    %v4790 = stablehlo.multiply %v4787, %v4789 : tensor<64xf32>
    %v4791 = stablehlo.add %v4788, %v4790 : tensor<64xf32>
    %v4792 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4793 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4794 = stablehlo.divide %v4785, %v4792 : tensor<64xf32>
    %v4795 = stablehlo.divide %v4791, %v4793 : tensor<64xf32>
    %v4796 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4797 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4798 = stablehlo.sqrt %v4795 : tensor<64xf32>
    %v4799 = stablehlo.add %v4798, %v4797 : tensor<64xf32>
    %v4800 = stablehlo.divide %v4794, %v4799 : tensor<64xf32>
    %v4801 = stablehlo.multiply %v4796, %v4800 : tensor<64xf32>
    %v4802 = stablehlo.subtract %s1b1g2, %v4801 : tensor<64xf32>
    %v4803 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4804 = stablehlo.multiply %v4803, %v4796 : tensor<64xf32>
    %v4805 = stablehlo.multiply %v4804, %s1b1g2 : tensor<64xf32>
    %v4806 = stablehlo.subtract %v4802, %v4805 : tensor<64xf32>
    %v4807 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4808 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4809 = stablehlo.multiply %v4807, %s1b1bt2m : tensor<64xf32>
    %v4810 = stablehlo.multiply %v4808, %v3424 : tensor<64xf32>
    %v4811 = stablehlo.add %v4809, %v4810 : tensor<64xf32>
    %v4812 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4813 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4814 = stablehlo.multiply %v4812, %s1b1bt2v : tensor<64xf32>
    %v4815 = stablehlo.multiply %v3424, %v3424 : tensor<64xf32>
    %v4816 = stablehlo.multiply %v4813, %v4815 : tensor<64xf32>
    %v4817 = stablehlo.add %v4814, %v4816 : tensor<64xf32>
    %v4818 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4819 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4820 = stablehlo.multiply %v4818, %s1b1bt2m : tensor<64xf32>
    %v4821 = stablehlo.multiply %v4819, %v3424 : tensor<64xf32>
    %v4822 = stablehlo.add %v4820, %v4821 : tensor<64xf32>
    %v4823 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4824 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4825 = stablehlo.multiply %v4823, %s1b1bt2v : tensor<64xf32>
    %v4826 = stablehlo.multiply %v3424, %v3424 : tensor<64xf32>
    %v4827 = stablehlo.multiply %v4824, %v4826 : tensor<64xf32>
    %v4828 = stablehlo.add %v4825, %v4827 : tensor<64xf32>
    %v4829 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4830 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4831 = stablehlo.divide %v4822, %v4829 : tensor<64xf32>
    %v4832 = stablehlo.divide %v4828, %v4830 : tensor<64xf32>
    %v4833 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4834 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4835 = stablehlo.sqrt %v4832 : tensor<64xf32>
    %v4836 = stablehlo.add %v4835, %v4834 : tensor<64xf32>
    %v4837 = stablehlo.divide %v4831, %v4836 : tensor<64xf32>
    %v4838 = stablehlo.multiply %v4833, %v4837 : tensor<64xf32>
    %v4839 = stablehlo.subtract %s1b1bt2, %v4838 : tensor<64xf32>
    %v4840 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4841 = stablehlo.multiply %v4840, %v4833 : tensor<64xf32>
    %v4842 = stablehlo.multiply %v4841, %s1b1bt2 : tensor<64xf32>
    %v4843 = stablehlo.subtract %v4839, %v4842 : tensor<64xf32>
    %v4844 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4845 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4846 = stablehlo.multiply %v4844, %s1b2W1m : tensor<64x64x3x3xf32>
    %v4847 = stablehlo.multiply %v4845, %v3236 : tensor<64x64x3x3xf32>
    %v4848 = stablehlo.add %v4846, %v4847 : tensor<64x64x3x3xf32>
    %v4849 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4850 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4851 = stablehlo.multiply %v4849, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4852 = stablehlo.multiply %v3236, %v3236 : tensor<64x64x3x3xf32>
    %v4853 = stablehlo.multiply %v4850, %v4852 : tensor<64x64x3x3xf32>
    %v4854 = stablehlo.add %v4851, %v4853 : tensor<64x64x3x3xf32>
    %v4855 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4856 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4857 = stablehlo.multiply %v4855, %s1b2W1m : tensor<64x64x3x3xf32>
    %v4858 = stablehlo.multiply %v4856, %v3236 : tensor<64x64x3x3xf32>
    %v4859 = stablehlo.add %v4857, %v4858 : tensor<64x64x3x3xf32>
    %v4860 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4861 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4862 = stablehlo.multiply %v4860, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4863 = stablehlo.multiply %v3236, %v3236 : tensor<64x64x3x3xf32>
    %v4864 = stablehlo.multiply %v4861, %v4863 : tensor<64x64x3x3xf32>
    %v4865 = stablehlo.add %v4862, %v4864 : tensor<64x64x3x3xf32>
    %v4866 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4867 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4868 = stablehlo.divide %v4859, %v4866 : tensor<64x64x3x3xf32>
    %v4869 = stablehlo.divide %v4865, %v4867 : tensor<64x64x3x3xf32>
    %v4870 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4871 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4872 = stablehlo.sqrt %v4869 : tensor<64x64x3x3xf32>
    %v4873 = stablehlo.add %v4872, %v4871 : tensor<64x64x3x3xf32>
    %v4874 = stablehlo.divide %v4868, %v4873 : tensor<64x64x3x3xf32>
    %v4875 = stablehlo.multiply %v4870, %v4874 : tensor<64x64x3x3xf32>
    %v4876 = stablehlo.subtract %s1b2W1, %v4875 : tensor<64x64x3x3xf32>
    %v4877 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4878 = stablehlo.multiply %v4877, %v4870 : tensor<64x64x3x3xf32>
    %v4879 = stablehlo.multiply %v4878, %s1b2W1 : tensor<64x64x3x3xf32>
    %v4880 = stablehlo.subtract %v4876, %v4879 : tensor<64x64x3x3xf32>
    %v4881 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4882 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4883 = stablehlo.multiply %v4881, %s1b2g1m : tensor<64xf32>
    %v4884 = stablehlo.multiply %v4882, %v3254 : tensor<64xf32>
    %v4885 = stablehlo.add %v4883, %v4884 : tensor<64xf32>
    %v4886 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4887 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4888 = stablehlo.multiply %v4886, %s1b2g1v : tensor<64xf32>
    %v4889 = stablehlo.multiply %v3254, %v3254 : tensor<64xf32>
    %v4890 = stablehlo.multiply %v4887, %v4889 : tensor<64xf32>
    %v4891 = stablehlo.add %v4888, %v4890 : tensor<64xf32>
    %v4892 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4893 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4894 = stablehlo.multiply %v4892, %s1b2g1m : tensor<64xf32>
    %v4895 = stablehlo.multiply %v4893, %v3254 : tensor<64xf32>
    %v4896 = stablehlo.add %v4894, %v4895 : tensor<64xf32>
    %v4897 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4898 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4899 = stablehlo.multiply %v4897, %s1b2g1v : tensor<64xf32>
    %v4900 = stablehlo.multiply %v3254, %v3254 : tensor<64xf32>
    %v4901 = stablehlo.multiply %v4898, %v4900 : tensor<64xf32>
    %v4902 = stablehlo.add %v4899, %v4901 : tensor<64xf32>
    %v4903 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4904 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4905 = stablehlo.divide %v4896, %v4903 : tensor<64xf32>
    %v4906 = stablehlo.divide %v4902, %v4904 : tensor<64xf32>
    %v4907 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4908 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4909 = stablehlo.sqrt %v4906 : tensor<64xf32>
    %v4910 = stablehlo.add %v4909, %v4908 : tensor<64xf32>
    %v4911 = stablehlo.divide %v4905, %v4910 : tensor<64xf32>
    %v4912 = stablehlo.multiply %v4907, %v4911 : tensor<64xf32>
    %v4913 = stablehlo.subtract %s1b2g1, %v4912 : tensor<64xf32>
    %v4914 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4915 = stablehlo.multiply %v4914, %v4907 : tensor<64xf32>
    %v4916 = stablehlo.multiply %v4915, %s1b2g1 : tensor<64xf32>
    %v4917 = stablehlo.subtract %v4913, %v4916 : tensor<64xf32>
    %v4918 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4919 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4920 = stablehlo.multiply %v4918, %s1b2bt1m : tensor<64xf32>
    %v4921 = stablehlo.multiply %v4919, %v3257 : tensor<64xf32>
    %v4922 = stablehlo.add %v4920, %v4921 : tensor<64xf32>
    %v4923 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4924 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4925 = stablehlo.multiply %v4923, %s1b2bt1v : tensor<64xf32>
    %v4926 = stablehlo.multiply %v3257, %v3257 : tensor<64xf32>
    %v4927 = stablehlo.multiply %v4924, %v4926 : tensor<64xf32>
    %v4928 = stablehlo.add %v4925, %v4927 : tensor<64xf32>
    %v4929 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4930 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4931 = stablehlo.multiply %v4929, %s1b2bt1m : tensor<64xf32>
    %v4932 = stablehlo.multiply %v4930, %v3257 : tensor<64xf32>
    %v4933 = stablehlo.add %v4931, %v4932 : tensor<64xf32>
    %v4934 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4935 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4936 = stablehlo.multiply %v4934, %s1b2bt1v : tensor<64xf32>
    %v4937 = stablehlo.multiply %v3257, %v3257 : tensor<64xf32>
    %v4938 = stablehlo.multiply %v4935, %v4937 : tensor<64xf32>
    %v4939 = stablehlo.add %v4936, %v4938 : tensor<64xf32>
    %v4940 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4941 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4942 = stablehlo.divide %v4933, %v4940 : tensor<64xf32>
    %v4943 = stablehlo.divide %v4939, %v4941 : tensor<64xf32>
    %v4944 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4945 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4946 = stablehlo.sqrt %v4943 : tensor<64xf32>
    %v4947 = stablehlo.add %v4946, %v4945 : tensor<64xf32>
    %v4948 = stablehlo.divide %v4942, %v4947 : tensor<64xf32>
    %v4949 = stablehlo.multiply %v4944, %v4948 : tensor<64xf32>
    %v4950 = stablehlo.subtract %s1b2bt1, %v4949 : tensor<64xf32>
    %v4951 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4952 = stablehlo.multiply %v4951, %v4944 : tensor<64xf32>
    %v4953 = stablehlo.multiply %v4952, %s1b2bt1 : tensor<64xf32>
    %v4954 = stablehlo.subtract %v4950, %v4953 : tensor<64xf32>
    %v4955 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4956 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4957 = stablehlo.multiply %v4955, %s1b2W2m : tensor<64x64x3x3xf32>
    %v4958 = stablehlo.multiply %v4956, %v3263 : tensor<64x64x3x3xf32>
    %v4959 = stablehlo.add %v4957, %v4958 : tensor<64x64x3x3xf32>
    %v4960 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4961 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4962 = stablehlo.multiply %v4960, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4963 = stablehlo.multiply %v3263, %v3263 : tensor<64x64x3x3xf32>
    %v4964 = stablehlo.multiply %v4961, %v4963 : tensor<64x64x3x3xf32>
    %v4965 = stablehlo.add %v4962, %v4964 : tensor<64x64x3x3xf32>
    %v4966 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4967 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4968 = stablehlo.multiply %v4966, %s1b2W2m : tensor<64x64x3x3xf32>
    %v4969 = stablehlo.multiply %v4967, %v3263 : tensor<64x64x3x3xf32>
    %v4970 = stablehlo.add %v4968, %v4969 : tensor<64x64x3x3xf32>
    %v4971 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4972 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4973 = stablehlo.multiply %v4971, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4974 = stablehlo.multiply %v3263, %v3263 : tensor<64x64x3x3xf32>
    %v4975 = stablehlo.multiply %v4972, %v4974 : tensor<64x64x3x3xf32>
    %v4976 = stablehlo.add %v4973, %v4975 : tensor<64x64x3x3xf32>
    %v4977 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4978 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4979 = stablehlo.divide %v4970, %v4977 : tensor<64x64x3x3xf32>
    %v4980 = stablehlo.divide %v4976, %v4978 : tensor<64x64x3x3xf32>
    %v4981 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4982 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4983 = stablehlo.sqrt %v4980 : tensor<64x64x3x3xf32>
    %v4984 = stablehlo.add %v4983, %v4982 : tensor<64x64x3x3xf32>
    %v4985 = stablehlo.divide %v4979, %v4984 : tensor<64x64x3x3xf32>
    %v4986 = stablehlo.multiply %v4981, %v4985 : tensor<64x64x3x3xf32>
    %v4987 = stablehlo.subtract %s1b2W2, %v4986 : tensor<64x64x3x3xf32>
    %v4988 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4989 = stablehlo.multiply %v4988, %v4981 : tensor<64x64x3x3xf32>
    %v4990 = stablehlo.multiply %v4989, %s1b2W2 : tensor<64x64x3x3xf32>
    %v4991 = stablehlo.subtract %v4987, %v4990 : tensor<64x64x3x3xf32>
    %v4992 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4993 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4994 = stablehlo.multiply %v4992, %s1b2g2m : tensor<64xf32>
    %v4995 = stablehlo.multiply %v4993, %v3281 : tensor<64xf32>
    %v4996 = stablehlo.add %v4994, %v4995 : tensor<64xf32>
    %v4997 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4998 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4999 = stablehlo.multiply %v4997, %s1b2g2v : tensor<64xf32>
    %v5000 = stablehlo.multiply %v3281, %v3281 : tensor<64xf32>
    %v5001 = stablehlo.multiply %v4998, %v5000 : tensor<64xf32>
    %v5002 = stablehlo.add %v4999, %v5001 : tensor<64xf32>
    %v5003 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5004 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5005 = stablehlo.multiply %v5003, %s1b2g2m : tensor<64xf32>
    %v5006 = stablehlo.multiply %v5004, %v3281 : tensor<64xf32>
    %v5007 = stablehlo.add %v5005, %v5006 : tensor<64xf32>
    %v5008 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5009 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5010 = stablehlo.multiply %v5008, %s1b2g2v : tensor<64xf32>
    %v5011 = stablehlo.multiply %v3281, %v3281 : tensor<64xf32>
    %v5012 = stablehlo.multiply %v5009, %v5011 : tensor<64xf32>
    %v5013 = stablehlo.add %v5010, %v5012 : tensor<64xf32>
    %v5014 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5015 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5016 = stablehlo.divide %v5007, %v5014 : tensor<64xf32>
    %v5017 = stablehlo.divide %v5013, %v5015 : tensor<64xf32>
    %v5018 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5019 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5020 = stablehlo.sqrt %v5017 : tensor<64xf32>
    %v5021 = stablehlo.add %v5020, %v5019 : tensor<64xf32>
    %v5022 = stablehlo.divide %v5016, %v5021 : tensor<64xf32>
    %v5023 = stablehlo.multiply %v5018, %v5022 : tensor<64xf32>
    %v5024 = stablehlo.subtract %s1b2g2, %v5023 : tensor<64xf32>
    %v5025 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5026 = stablehlo.multiply %v5025, %v5018 : tensor<64xf32>
    %v5027 = stablehlo.multiply %v5026, %s1b2g2 : tensor<64xf32>
    %v5028 = stablehlo.subtract %v5024, %v5027 : tensor<64xf32>
    %v5029 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5030 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5031 = stablehlo.multiply %v5029, %s1b2bt2m : tensor<64xf32>
    %v5032 = stablehlo.multiply %v5030, %v3284 : tensor<64xf32>
    %v5033 = stablehlo.add %v5031, %v5032 : tensor<64xf32>
    %v5034 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5035 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5036 = stablehlo.multiply %v5034, %s1b2bt2v : tensor<64xf32>
    %v5037 = stablehlo.multiply %v3284, %v3284 : tensor<64xf32>
    %v5038 = stablehlo.multiply %v5035, %v5037 : tensor<64xf32>
    %v5039 = stablehlo.add %v5036, %v5038 : tensor<64xf32>
    %v5040 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5041 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5042 = stablehlo.multiply %v5040, %s1b2bt2m : tensor<64xf32>
    %v5043 = stablehlo.multiply %v5041, %v3284 : tensor<64xf32>
    %v5044 = stablehlo.add %v5042, %v5043 : tensor<64xf32>
    %v5045 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5046 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5047 = stablehlo.multiply %v5045, %s1b2bt2v : tensor<64xf32>
    %v5048 = stablehlo.multiply %v3284, %v3284 : tensor<64xf32>
    %v5049 = stablehlo.multiply %v5046, %v5048 : tensor<64xf32>
    %v5050 = stablehlo.add %v5047, %v5049 : tensor<64xf32>
    %v5051 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5052 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5053 = stablehlo.divide %v5044, %v5051 : tensor<64xf32>
    %v5054 = stablehlo.divide %v5050, %v5052 : tensor<64xf32>
    %v5055 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5056 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5057 = stablehlo.sqrt %v5054 : tensor<64xf32>
    %v5058 = stablehlo.add %v5057, %v5056 : tensor<64xf32>
    %v5059 = stablehlo.divide %v5053, %v5058 : tensor<64xf32>
    %v5060 = stablehlo.multiply %v5055, %v5059 : tensor<64xf32>
    %v5061 = stablehlo.subtract %s1b2bt2, %v5060 : tensor<64xf32>
    %v5062 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5063 = stablehlo.multiply %v5062, %v5055 : tensor<64xf32>
    %v5064 = stablehlo.multiply %v5063, %s1b2bt2 : tensor<64xf32>
    %v5065 = stablehlo.subtract %v5061, %v5064 : tensor<64xf32>
    %v5066 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5067 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5068 = stablehlo.multiply %v5066, %d2W1m : tensor<128x64x3x3xf32>
    %v5069 = stablehlo.multiply %v5067, %v3068 : tensor<128x64x3x3xf32>
    %v5070 = stablehlo.add %v5068, %v5069 : tensor<128x64x3x3xf32>
    %v5071 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5072 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5073 = stablehlo.multiply %v5071, %d2W1v : tensor<128x64x3x3xf32>
    %v5074 = stablehlo.multiply %v3068, %v3068 : tensor<128x64x3x3xf32>
    %v5075 = stablehlo.multiply %v5072, %v5074 : tensor<128x64x3x3xf32>
    %v5076 = stablehlo.add %v5073, %v5075 : tensor<128x64x3x3xf32>
    %v5077 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5078 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5079 = stablehlo.multiply %v5077, %d2W1m : tensor<128x64x3x3xf32>
    %v5080 = stablehlo.multiply %v5078, %v3068 : tensor<128x64x3x3xf32>
    %v5081 = stablehlo.add %v5079, %v5080 : tensor<128x64x3x3xf32>
    %v5082 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5083 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5084 = stablehlo.multiply %v5082, %d2W1v : tensor<128x64x3x3xf32>
    %v5085 = stablehlo.multiply %v3068, %v3068 : tensor<128x64x3x3xf32>
    %v5086 = stablehlo.multiply %v5083, %v5085 : tensor<128x64x3x3xf32>
    %v5087 = stablehlo.add %v5084, %v5086 : tensor<128x64x3x3xf32>
    %v5088 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5089 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5090 = stablehlo.divide %v5081, %v5088 : tensor<128x64x3x3xf32>
    %v5091 = stablehlo.divide %v5087, %v5089 : tensor<128x64x3x3xf32>
    %v5092 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5093 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5094 = stablehlo.sqrt %v5091 : tensor<128x64x3x3xf32>
    %v5095 = stablehlo.add %v5094, %v5093 : tensor<128x64x3x3xf32>
    %v5096 = stablehlo.divide %v5090, %v5095 : tensor<128x64x3x3xf32>
    %v5097 = stablehlo.multiply %v5092, %v5096 : tensor<128x64x3x3xf32>
    %v5098 = stablehlo.subtract %d2W1, %v5097 : tensor<128x64x3x3xf32>
    %v5099 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5100 = stablehlo.multiply %v5099, %v5092 : tensor<128x64x3x3xf32>
    %v5101 = stablehlo.multiply %v5100, %d2W1 : tensor<128x64x3x3xf32>
    %v5102 = stablehlo.subtract %v5098, %v5101 : tensor<128x64x3x3xf32>
    %v5103 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5104 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5105 = stablehlo.multiply %v5103, %d2g1m : tensor<128xf32>
    %v5106 = stablehlo.multiply %v5104, %v3086 : tensor<128xf32>
    %v5107 = stablehlo.add %v5105, %v5106 : tensor<128xf32>
    %v5108 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5109 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5110 = stablehlo.multiply %v5108, %d2g1v : tensor<128xf32>
    %v5111 = stablehlo.multiply %v3086, %v3086 : tensor<128xf32>
    %v5112 = stablehlo.multiply %v5109, %v5111 : tensor<128xf32>
    %v5113 = stablehlo.add %v5110, %v5112 : tensor<128xf32>
    %v5114 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5115 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5116 = stablehlo.multiply %v5114, %d2g1m : tensor<128xf32>
    %v5117 = stablehlo.multiply %v5115, %v3086 : tensor<128xf32>
    %v5118 = stablehlo.add %v5116, %v5117 : tensor<128xf32>
    %v5119 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5120 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5121 = stablehlo.multiply %v5119, %d2g1v : tensor<128xf32>
    %v5122 = stablehlo.multiply %v3086, %v3086 : tensor<128xf32>
    %v5123 = stablehlo.multiply %v5120, %v5122 : tensor<128xf32>
    %v5124 = stablehlo.add %v5121, %v5123 : tensor<128xf32>
    %v5125 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5126 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5127 = stablehlo.divide %v5118, %v5125 : tensor<128xf32>
    %v5128 = stablehlo.divide %v5124, %v5126 : tensor<128xf32>
    %v5129 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5130 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5131 = stablehlo.sqrt %v5128 : tensor<128xf32>
    %v5132 = stablehlo.add %v5131, %v5130 : tensor<128xf32>
    %v5133 = stablehlo.divide %v5127, %v5132 : tensor<128xf32>
    %v5134 = stablehlo.multiply %v5129, %v5133 : tensor<128xf32>
    %v5135 = stablehlo.subtract %d2g1, %v5134 : tensor<128xf32>
    %v5136 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5137 = stablehlo.multiply %v5136, %v5129 : tensor<128xf32>
    %v5138 = stablehlo.multiply %v5137, %d2g1 : tensor<128xf32>
    %v5139 = stablehlo.subtract %v5135, %v5138 : tensor<128xf32>
    %v5140 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5141 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5142 = stablehlo.multiply %v5140, %d2bt1m : tensor<128xf32>
    %v5143 = stablehlo.multiply %v5141, %v3089 : tensor<128xf32>
    %v5144 = stablehlo.add %v5142, %v5143 : tensor<128xf32>
    %v5145 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5146 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5147 = stablehlo.multiply %v5145, %d2bt1v : tensor<128xf32>
    %v5148 = stablehlo.multiply %v3089, %v3089 : tensor<128xf32>
    %v5149 = stablehlo.multiply %v5146, %v5148 : tensor<128xf32>
    %v5150 = stablehlo.add %v5147, %v5149 : tensor<128xf32>
    %v5151 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5152 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5153 = stablehlo.multiply %v5151, %d2bt1m : tensor<128xf32>
    %v5154 = stablehlo.multiply %v5152, %v3089 : tensor<128xf32>
    %v5155 = stablehlo.add %v5153, %v5154 : tensor<128xf32>
    %v5156 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5157 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5158 = stablehlo.multiply %v5156, %d2bt1v : tensor<128xf32>
    %v5159 = stablehlo.multiply %v3089, %v3089 : tensor<128xf32>
    %v5160 = stablehlo.multiply %v5157, %v5159 : tensor<128xf32>
    %v5161 = stablehlo.add %v5158, %v5160 : tensor<128xf32>
    %v5162 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5163 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5164 = stablehlo.divide %v5155, %v5162 : tensor<128xf32>
    %v5165 = stablehlo.divide %v5161, %v5163 : tensor<128xf32>
    %v5166 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5167 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5168 = stablehlo.sqrt %v5165 : tensor<128xf32>
    %v5169 = stablehlo.add %v5168, %v5167 : tensor<128xf32>
    %v5170 = stablehlo.divide %v5164, %v5169 : tensor<128xf32>
    %v5171 = stablehlo.multiply %v5166, %v5170 : tensor<128xf32>
    %v5172 = stablehlo.subtract %d2bt1, %v5171 : tensor<128xf32>
    %v5173 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5174 = stablehlo.multiply %v5173, %v5166 : tensor<128xf32>
    %v5175 = stablehlo.multiply %v5174, %d2bt1 : tensor<128xf32>
    %v5176 = stablehlo.subtract %v5172, %v5175 : tensor<128xf32>
    %v5177 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5178 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5179 = stablehlo.multiply %v5177, %d2W2m : tensor<128x128x3x3xf32>
    %v5180 = stablehlo.multiply %v5178, %v3095 : tensor<128x128x3x3xf32>
    %v5181 = stablehlo.add %v5179, %v5180 : tensor<128x128x3x3xf32>
    %v5182 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5183 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5184 = stablehlo.multiply %v5182, %d2W2v : tensor<128x128x3x3xf32>
    %v5185 = stablehlo.multiply %v3095, %v3095 : tensor<128x128x3x3xf32>
    %v5186 = stablehlo.multiply %v5183, %v5185 : tensor<128x128x3x3xf32>
    %v5187 = stablehlo.add %v5184, %v5186 : tensor<128x128x3x3xf32>
    %v5188 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5189 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5190 = stablehlo.multiply %v5188, %d2W2m : tensor<128x128x3x3xf32>
    %v5191 = stablehlo.multiply %v5189, %v3095 : tensor<128x128x3x3xf32>
    %v5192 = stablehlo.add %v5190, %v5191 : tensor<128x128x3x3xf32>
    %v5193 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5194 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5195 = stablehlo.multiply %v5193, %d2W2v : tensor<128x128x3x3xf32>
    %v5196 = stablehlo.multiply %v3095, %v3095 : tensor<128x128x3x3xf32>
    %v5197 = stablehlo.multiply %v5194, %v5196 : tensor<128x128x3x3xf32>
    %v5198 = stablehlo.add %v5195, %v5197 : tensor<128x128x3x3xf32>
    %v5199 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5200 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5201 = stablehlo.divide %v5192, %v5199 : tensor<128x128x3x3xf32>
    %v5202 = stablehlo.divide %v5198, %v5200 : tensor<128x128x3x3xf32>
    %v5203 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5204 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5205 = stablehlo.sqrt %v5202 : tensor<128x128x3x3xf32>
    %v5206 = stablehlo.add %v5205, %v5204 : tensor<128x128x3x3xf32>
    %v5207 = stablehlo.divide %v5201, %v5206 : tensor<128x128x3x3xf32>
    %v5208 = stablehlo.multiply %v5203, %v5207 : tensor<128x128x3x3xf32>
    %v5209 = stablehlo.subtract %d2W2, %v5208 : tensor<128x128x3x3xf32>
    %v5210 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5211 = stablehlo.multiply %v5210, %v5203 : tensor<128x128x3x3xf32>
    %v5212 = stablehlo.multiply %v5211, %d2W2 : tensor<128x128x3x3xf32>
    %v5213 = stablehlo.subtract %v5209, %v5212 : tensor<128x128x3x3xf32>
    %v5214 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5215 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5216 = stablehlo.multiply %v5214, %d2g2m : tensor<128xf32>
    %v5217 = stablehlo.multiply %v5215, %v3113 : tensor<128xf32>
    %v5218 = stablehlo.add %v5216, %v5217 : tensor<128xf32>
    %v5219 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5220 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5221 = stablehlo.multiply %v5219, %d2g2v : tensor<128xf32>
    %v5222 = stablehlo.multiply %v3113, %v3113 : tensor<128xf32>
    %v5223 = stablehlo.multiply %v5220, %v5222 : tensor<128xf32>
    %v5224 = stablehlo.add %v5221, %v5223 : tensor<128xf32>
    %v5225 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5226 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5227 = stablehlo.multiply %v5225, %d2g2m : tensor<128xf32>
    %v5228 = stablehlo.multiply %v5226, %v3113 : tensor<128xf32>
    %v5229 = stablehlo.add %v5227, %v5228 : tensor<128xf32>
    %v5230 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5231 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5232 = stablehlo.multiply %v5230, %d2g2v : tensor<128xf32>
    %v5233 = stablehlo.multiply %v3113, %v3113 : tensor<128xf32>
    %v5234 = stablehlo.multiply %v5231, %v5233 : tensor<128xf32>
    %v5235 = stablehlo.add %v5232, %v5234 : tensor<128xf32>
    %v5236 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5237 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5238 = stablehlo.divide %v5229, %v5236 : tensor<128xf32>
    %v5239 = stablehlo.divide %v5235, %v5237 : tensor<128xf32>
    %v5240 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5241 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5242 = stablehlo.sqrt %v5239 : tensor<128xf32>
    %v5243 = stablehlo.add %v5242, %v5241 : tensor<128xf32>
    %v5244 = stablehlo.divide %v5238, %v5243 : tensor<128xf32>
    %v5245 = stablehlo.multiply %v5240, %v5244 : tensor<128xf32>
    %v5246 = stablehlo.subtract %d2g2, %v5245 : tensor<128xf32>
    %v5247 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5248 = stablehlo.multiply %v5247, %v5240 : tensor<128xf32>
    %v5249 = stablehlo.multiply %v5248, %d2g2 : tensor<128xf32>
    %v5250 = stablehlo.subtract %v5246, %v5249 : tensor<128xf32>
    %v5251 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5252 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5253 = stablehlo.multiply %v5251, %d2bt2m : tensor<128xf32>
    %v5254 = stablehlo.multiply %v5252, %v3116 : tensor<128xf32>
    %v5255 = stablehlo.add %v5253, %v5254 : tensor<128xf32>
    %v5256 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5257 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5258 = stablehlo.multiply %v5256, %d2bt2v : tensor<128xf32>
    %v5259 = stablehlo.multiply %v3116, %v3116 : tensor<128xf32>
    %v5260 = stablehlo.multiply %v5257, %v5259 : tensor<128xf32>
    %v5261 = stablehlo.add %v5258, %v5260 : tensor<128xf32>
    %v5262 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5263 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5264 = stablehlo.multiply %v5262, %d2bt2m : tensor<128xf32>
    %v5265 = stablehlo.multiply %v5263, %v3116 : tensor<128xf32>
    %v5266 = stablehlo.add %v5264, %v5265 : tensor<128xf32>
    %v5267 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5268 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5269 = stablehlo.multiply %v5267, %d2bt2v : tensor<128xf32>
    %v5270 = stablehlo.multiply %v3116, %v3116 : tensor<128xf32>
    %v5271 = stablehlo.multiply %v5268, %v5270 : tensor<128xf32>
    %v5272 = stablehlo.add %v5269, %v5271 : tensor<128xf32>
    %v5273 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5274 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5275 = stablehlo.divide %v5266, %v5273 : tensor<128xf32>
    %v5276 = stablehlo.divide %v5272, %v5274 : tensor<128xf32>
    %v5277 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5278 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5279 = stablehlo.sqrt %v5276 : tensor<128xf32>
    %v5280 = stablehlo.add %v5279, %v5278 : tensor<128xf32>
    %v5281 = stablehlo.divide %v5275, %v5280 : tensor<128xf32>
    %v5282 = stablehlo.multiply %v5277, %v5281 : tensor<128xf32>
    %v5283 = stablehlo.subtract %d2bt2, %v5282 : tensor<128xf32>
    %v5284 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5285 = stablehlo.multiply %v5284, %v5277 : tensor<128xf32>
    %v5286 = stablehlo.multiply %v5285, %d2bt2 : tensor<128xf32>
    %v5287 = stablehlo.subtract %v5283, %v5286 : tensor<128xf32>
    %v5288 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5289 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5290 = stablehlo.multiply %v5288, %d2Wpm : tensor<128x64x1x1xf32>
    %v5291 = stablehlo.multiply %v5289, %v3123 : tensor<128x64x1x1xf32>
    %v5292 = stablehlo.add %v5290, %v5291 : tensor<128x64x1x1xf32>
    %v5293 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5294 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5295 = stablehlo.multiply %v5293, %d2Wpv : tensor<128x64x1x1xf32>
    %v5296 = stablehlo.multiply %v3123, %v3123 : tensor<128x64x1x1xf32>
    %v5297 = stablehlo.multiply %v5294, %v5296 : tensor<128x64x1x1xf32>
    %v5298 = stablehlo.add %v5295, %v5297 : tensor<128x64x1x1xf32>
    %v5299 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5300 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5301 = stablehlo.multiply %v5299, %d2Wpm : tensor<128x64x1x1xf32>
    %v5302 = stablehlo.multiply %v5300, %v3123 : tensor<128x64x1x1xf32>
    %v5303 = stablehlo.add %v5301, %v5302 : tensor<128x64x1x1xf32>
    %v5304 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5305 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5306 = stablehlo.multiply %v5304, %d2Wpv : tensor<128x64x1x1xf32>
    %v5307 = stablehlo.multiply %v3123, %v3123 : tensor<128x64x1x1xf32>
    %v5308 = stablehlo.multiply %v5305, %v5307 : tensor<128x64x1x1xf32>
    %v5309 = stablehlo.add %v5306, %v5308 : tensor<128x64x1x1xf32>
    %v5310 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5311 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5312 = stablehlo.divide %v5303, %v5310 : tensor<128x64x1x1xf32>
    %v5313 = stablehlo.divide %v5309, %v5311 : tensor<128x64x1x1xf32>
    %v5314 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5315 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5316 = stablehlo.sqrt %v5313 : tensor<128x64x1x1xf32>
    %v5317 = stablehlo.add %v5316, %v5315 : tensor<128x64x1x1xf32>
    %v5318 = stablehlo.divide %v5312, %v5317 : tensor<128x64x1x1xf32>
    %v5319 = stablehlo.multiply %v5314, %v5318 : tensor<128x64x1x1xf32>
    %v5320 = stablehlo.subtract %d2Wp, %v5319 : tensor<128x64x1x1xf32>
    %v5321 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5322 = stablehlo.multiply %v5321, %v5314 : tensor<128x64x1x1xf32>
    %v5323 = stablehlo.multiply %v5322, %d2Wp : tensor<128x64x1x1xf32>
    %v5324 = stablehlo.subtract %v5320, %v5323 : tensor<128x64x1x1xf32>
    %v5325 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5326 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5327 = stablehlo.multiply %v5325, %d2gpm : tensor<128xf32>
    %v5328 = stablehlo.multiply %v5326, %v3141 : tensor<128xf32>
    %v5329 = stablehlo.add %v5327, %v5328 : tensor<128xf32>
    %v5330 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5331 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5332 = stablehlo.multiply %v5330, %d2gpv : tensor<128xf32>
    %v5333 = stablehlo.multiply %v3141, %v3141 : tensor<128xf32>
    %v5334 = stablehlo.multiply %v5331, %v5333 : tensor<128xf32>
    %v5335 = stablehlo.add %v5332, %v5334 : tensor<128xf32>
    %v5336 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5337 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5338 = stablehlo.multiply %v5336, %d2gpm : tensor<128xf32>
    %v5339 = stablehlo.multiply %v5337, %v3141 : tensor<128xf32>
    %v5340 = stablehlo.add %v5338, %v5339 : tensor<128xf32>
    %v5341 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5342 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5343 = stablehlo.multiply %v5341, %d2gpv : tensor<128xf32>
    %v5344 = stablehlo.multiply %v3141, %v3141 : tensor<128xf32>
    %v5345 = stablehlo.multiply %v5342, %v5344 : tensor<128xf32>
    %v5346 = stablehlo.add %v5343, %v5345 : tensor<128xf32>
    %v5347 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5348 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5349 = stablehlo.divide %v5340, %v5347 : tensor<128xf32>
    %v5350 = stablehlo.divide %v5346, %v5348 : tensor<128xf32>
    %v5351 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5352 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5353 = stablehlo.sqrt %v5350 : tensor<128xf32>
    %v5354 = stablehlo.add %v5353, %v5352 : tensor<128xf32>
    %v5355 = stablehlo.divide %v5349, %v5354 : tensor<128xf32>
    %v5356 = stablehlo.multiply %v5351, %v5355 : tensor<128xf32>
    %v5357 = stablehlo.subtract %d2gp, %v5356 : tensor<128xf32>
    %v5358 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5359 = stablehlo.multiply %v5358, %v5351 : tensor<128xf32>
    %v5360 = stablehlo.multiply %v5359, %d2gp : tensor<128xf32>
    %v5361 = stablehlo.subtract %v5357, %v5360 : tensor<128xf32>
    %v5362 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5363 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5364 = stablehlo.multiply %v5362, %d2btpm : tensor<128xf32>
    %v5365 = stablehlo.multiply %v5363, %v3144 : tensor<128xf32>
    %v5366 = stablehlo.add %v5364, %v5365 : tensor<128xf32>
    %v5367 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5368 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5369 = stablehlo.multiply %v5367, %d2btpv : tensor<128xf32>
    %v5370 = stablehlo.multiply %v3144, %v3144 : tensor<128xf32>
    %v5371 = stablehlo.multiply %v5368, %v5370 : tensor<128xf32>
    %v5372 = stablehlo.add %v5369, %v5371 : tensor<128xf32>
    %v5373 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5374 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5375 = stablehlo.multiply %v5373, %d2btpm : tensor<128xf32>
    %v5376 = stablehlo.multiply %v5374, %v3144 : tensor<128xf32>
    %v5377 = stablehlo.add %v5375, %v5376 : tensor<128xf32>
    %v5378 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5379 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5380 = stablehlo.multiply %v5378, %d2btpv : tensor<128xf32>
    %v5381 = stablehlo.multiply %v3144, %v3144 : tensor<128xf32>
    %v5382 = stablehlo.multiply %v5379, %v5381 : tensor<128xf32>
    %v5383 = stablehlo.add %v5380, %v5382 : tensor<128xf32>
    %v5384 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5385 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5386 = stablehlo.divide %v5377, %v5384 : tensor<128xf32>
    %v5387 = stablehlo.divide %v5383, %v5385 : tensor<128xf32>
    %v5388 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5389 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5390 = stablehlo.sqrt %v5387 : tensor<128xf32>
    %v5391 = stablehlo.add %v5390, %v5389 : tensor<128xf32>
    %v5392 = stablehlo.divide %v5386, %v5391 : tensor<128xf32>
    %v5393 = stablehlo.multiply %v5388, %v5392 : tensor<128xf32>
    %v5394 = stablehlo.subtract %d2btp, %v5393 : tensor<128xf32>
    %v5395 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5396 = stablehlo.multiply %v5395, %v5388 : tensor<128xf32>
    %v5397 = stablehlo.multiply %v5396, %d2btp : tensor<128xf32>
    %v5398 = stablehlo.subtract %v5394, %v5397 : tensor<128xf32>
    %v5399 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5400 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5401 = stablehlo.multiply %v5399, %s2b0W1m : tensor<128x128x3x3xf32>
    %v5402 = stablehlo.multiply %v5400, %v2892 : tensor<128x128x3x3xf32>
    %v5403 = stablehlo.add %v5401, %v5402 : tensor<128x128x3x3xf32>
    %v5404 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5405 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5406 = stablehlo.multiply %v5404, %s2b0W1v : tensor<128x128x3x3xf32>
    %v5407 = stablehlo.multiply %v2892, %v2892 : tensor<128x128x3x3xf32>
    %v5408 = stablehlo.multiply %v5405, %v5407 : tensor<128x128x3x3xf32>
    %v5409 = stablehlo.add %v5406, %v5408 : tensor<128x128x3x3xf32>
    %v5410 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5411 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5412 = stablehlo.multiply %v5410, %s2b0W1m : tensor<128x128x3x3xf32>
    %v5413 = stablehlo.multiply %v5411, %v2892 : tensor<128x128x3x3xf32>
    %v5414 = stablehlo.add %v5412, %v5413 : tensor<128x128x3x3xf32>
    %v5415 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5416 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5417 = stablehlo.multiply %v5415, %s2b0W1v : tensor<128x128x3x3xf32>
    %v5418 = stablehlo.multiply %v2892, %v2892 : tensor<128x128x3x3xf32>
    %v5419 = stablehlo.multiply %v5416, %v5418 : tensor<128x128x3x3xf32>
    %v5420 = stablehlo.add %v5417, %v5419 : tensor<128x128x3x3xf32>
    %v5421 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5422 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5423 = stablehlo.divide %v5414, %v5421 : tensor<128x128x3x3xf32>
    %v5424 = stablehlo.divide %v5420, %v5422 : tensor<128x128x3x3xf32>
    %v5425 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5426 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5427 = stablehlo.sqrt %v5424 : tensor<128x128x3x3xf32>
    %v5428 = stablehlo.add %v5427, %v5426 : tensor<128x128x3x3xf32>
    %v5429 = stablehlo.divide %v5423, %v5428 : tensor<128x128x3x3xf32>
    %v5430 = stablehlo.multiply %v5425, %v5429 : tensor<128x128x3x3xf32>
    %v5431 = stablehlo.subtract %s2b0W1, %v5430 : tensor<128x128x3x3xf32>
    %v5432 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5433 = stablehlo.multiply %v5432, %v5425 : tensor<128x128x3x3xf32>
    %v5434 = stablehlo.multiply %v5433, %s2b0W1 : tensor<128x128x3x3xf32>
    %v5435 = stablehlo.subtract %v5431, %v5434 : tensor<128x128x3x3xf32>
    %v5436 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5437 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5438 = stablehlo.multiply %v5436, %s2b0g1m : tensor<128xf32>
    %v5439 = stablehlo.multiply %v5437, %v2910 : tensor<128xf32>
    %v5440 = stablehlo.add %v5438, %v5439 : tensor<128xf32>
    %v5441 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5442 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5443 = stablehlo.multiply %v5441, %s2b0g1v : tensor<128xf32>
    %v5444 = stablehlo.multiply %v2910, %v2910 : tensor<128xf32>
    %v5445 = stablehlo.multiply %v5442, %v5444 : tensor<128xf32>
    %v5446 = stablehlo.add %v5443, %v5445 : tensor<128xf32>
    %v5447 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5448 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5449 = stablehlo.multiply %v5447, %s2b0g1m : tensor<128xf32>
    %v5450 = stablehlo.multiply %v5448, %v2910 : tensor<128xf32>
    %v5451 = stablehlo.add %v5449, %v5450 : tensor<128xf32>
    %v5452 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5453 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5454 = stablehlo.multiply %v5452, %s2b0g1v : tensor<128xf32>
    %v5455 = stablehlo.multiply %v2910, %v2910 : tensor<128xf32>
    %v5456 = stablehlo.multiply %v5453, %v5455 : tensor<128xf32>
    %v5457 = stablehlo.add %v5454, %v5456 : tensor<128xf32>
    %v5458 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5459 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5460 = stablehlo.divide %v5451, %v5458 : tensor<128xf32>
    %v5461 = stablehlo.divide %v5457, %v5459 : tensor<128xf32>
    %v5462 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5463 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5464 = stablehlo.sqrt %v5461 : tensor<128xf32>
    %v5465 = stablehlo.add %v5464, %v5463 : tensor<128xf32>
    %v5466 = stablehlo.divide %v5460, %v5465 : tensor<128xf32>
    %v5467 = stablehlo.multiply %v5462, %v5466 : tensor<128xf32>
    %v5468 = stablehlo.subtract %s2b0g1, %v5467 : tensor<128xf32>
    %v5469 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5470 = stablehlo.multiply %v5469, %v5462 : tensor<128xf32>
    %v5471 = stablehlo.multiply %v5470, %s2b0g1 : tensor<128xf32>
    %v5472 = stablehlo.subtract %v5468, %v5471 : tensor<128xf32>
    %v5473 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5474 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5475 = stablehlo.multiply %v5473, %s2b0bt1m : tensor<128xf32>
    %v5476 = stablehlo.multiply %v5474, %v2913 : tensor<128xf32>
    %v5477 = stablehlo.add %v5475, %v5476 : tensor<128xf32>
    %v5478 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5479 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5480 = stablehlo.multiply %v5478, %s2b0bt1v : tensor<128xf32>
    %v5481 = stablehlo.multiply %v2913, %v2913 : tensor<128xf32>
    %v5482 = stablehlo.multiply %v5479, %v5481 : tensor<128xf32>
    %v5483 = stablehlo.add %v5480, %v5482 : tensor<128xf32>
    %v5484 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5485 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5486 = stablehlo.multiply %v5484, %s2b0bt1m : tensor<128xf32>
    %v5487 = stablehlo.multiply %v5485, %v2913 : tensor<128xf32>
    %v5488 = stablehlo.add %v5486, %v5487 : tensor<128xf32>
    %v5489 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5490 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5491 = stablehlo.multiply %v5489, %s2b0bt1v : tensor<128xf32>
    %v5492 = stablehlo.multiply %v2913, %v2913 : tensor<128xf32>
    %v5493 = stablehlo.multiply %v5490, %v5492 : tensor<128xf32>
    %v5494 = stablehlo.add %v5491, %v5493 : tensor<128xf32>
    %v5495 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5496 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5497 = stablehlo.divide %v5488, %v5495 : tensor<128xf32>
    %v5498 = stablehlo.divide %v5494, %v5496 : tensor<128xf32>
    %v5499 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5500 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5501 = stablehlo.sqrt %v5498 : tensor<128xf32>
    %v5502 = stablehlo.add %v5501, %v5500 : tensor<128xf32>
    %v5503 = stablehlo.divide %v5497, %v5502 : tensor<128xf32>
    %v5504 = stablehlo.multiply %v5499, %v5503 : tensor<128xf32>
    %v5505 = stablehlo.subtract %s2b0bt1, %v5504 : tensor<128xf32>
    %v5506 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5507 = stablehlo.multiply %v5506, %v5499 : tensor<128xf32>
    %v5508 = stablehlo.multiply %v5507, %s2b0bt1 : tensor<128xf32>
    %v5509 = stablehlo.subtract %v5505, %v5508 : tensor<128xf32>
    %v5510 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5511 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5512 = stablehlo.multiply %v5510, %s2b0W2m : tensor<128x128x3x3xf32>
    %v5513 = stablehlo.multiply %v5511, %v2919 : tensor<128x128x3x3xf32>
    %v5514 = stablehlo.add %v5512, %v5513 : tensor<128x128x3x3xf32>
    %v5515 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5516 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5517 = stablehlo.multiply %v5515, %s2b0W2v : tensor<128x128x3x3xf32>
    %v5518 = stablehlo.multiply %v2919, %v2919 : tensor<128x128x3x3xf32>
    %v5519 = stablehlo.multiply %v5516, %v5518 : tensor<128x128x3x3xf32>
    %v5520 = stablehlo.add %v5517, %v5519 : tensor<128x128x3x3xf32>
    %v5521 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5522 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5523 = stablehlo.multiply %v5521, %s2b0W2m : tensor<128x128x3x3xf32>
    %v5524 = stablehlo.multiply %v5522, %v2919 : tensor<128x128x3x3xf32>
    %v5525 = stablehlo.add %v5523, %v5524 : tensor<128x128x3x3xf32>
    %v5526 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5527 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5528 = stablehlo.multiply %v5526, %s2b0W2v : tensor<128x128x3x3xf32>
    %v5529 = stablehlo.multiply %v2919, %v2919 : tensor<128x128x3x3xf32>
    %v5530 = stablehlo.multiply %v5527, %v5529 : tensor<128x128x3x3xf32>
    %v5531 = stablehlo.add %v5528, %v5530 : tensor<128x128x3x3xf32>
    %v5532 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5533 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5534 = stablehlo.divide %v5525, %v5532 : tensor<128x128x3x3xf32>
    %v5535 = stablehlo.divide %v5531, %v5533 : tensor<128x128x3x3xf32>
    %v5536 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5537 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5538 = stablehlo.sqrt %v5535 : tensor<128x128x3x3xf32>
    %v5539 = stablehlo.add %v5538, %v5537 : tensor<128x128x3x3xf32>
    %v5540 = stablehlo.divide %v5534, %v5539 : tensor<128x128x3x3xf32>
    %v5541 = stablehlo.multiply %v5536, %v5540 : tensor<128x128x3x3xf32>
    %v5542 = stablehlo.subtract %s2b0W2, %v5541 : tensor<128x128x3x3xf32>
    %v5543 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5544 = stablehlo.multiply %v5543, %v5536 : tensor<128x128x3x3xf32>
    %v5545 = stablehlo.multiply %v5544, %s2b0W2 : tensor<128x128x3x3xf32>
    %v5546 = stablehlo.subtract %v5542, %v5545 : tensor<128x128x3x3xf32>
    %v5547 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5548 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5549 = stablehlo.multiply %v5547, %s2b0g2m : tensor<128xf32>
    %v5550 = stablehlo.multiply %v5548, %v2937 : tensor<128xf32>
    %v5551 = stablehlo.add %v5549, %v5550 : tensor<128xf32>
    %v5552 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5553 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5554 = stablehlo.multiply %v5552, %s2b0g2v : tensor<128xf32>
    %v5555 = stablehlo.multiply %v2937, %v2937 : tensor<128xf32>
    %v5556 = stablehlo.multiply %v5553, %v5555 : tensor<128xf32>
    %v5557 = stablehlo.add %v5554, %v5556 : tensor<128xf32>
    %v5558 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5559 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5560 = stablehlo.multiply %v5558, %s2b0g2m : tensor<128xf32>
    %v5561 = stablehlo.multiply %v5559, %v2937 : tensor<128xf32>
    %v5562 = stablehlo.add %v5560, %v5561 : tensor<128xf32>
    %v5563 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5564 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5565 = stablehlo.multiply %v5563, %s2b0g2v : tensor<128xf32>
    %v5566 = stablehlo.multiply %v2937, %v2937 : tensor<128xf32>
    %v5567 = stablehlo.multiply %v5564, %v5566 : tensor<128xf32>
    %v5568 = stablehlo.add %v5565, %v5567 : tensor<128xf32>
    %v5569 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5570 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5571 = stablehlo.divide %v5562, %v5569 : tensor<128xf32>
    %v5572 = stablehlo.divide %v5568, %v5570 : tensor<128xf32>
    %v5573 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5574 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5575 = stablehlo.sqrt %v5572 : tensor<128xf32>
    %v5576 = stablehlo.add %v5575, %v5574 : tensor<128xf32>
    %v5577 = stablehlo.divide %v5571, %v5576 : tensor<128xf32>
    %v5578 = stablehlo.multiply %v5573, %v5577 : tensor<128xf32>
    %v5579 = stablehlo.subtract %s2b0g2, %v5578 : tensor<128xf32>
    %v5580 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5581 = stablehlo.multiply %v5580, %v5573 : tensor<128xf32>
    %v5582 = stablehlo.multiply %v5581, %s2b0g2 : tensor<128xf32>
    %v5583 = stablehlo.subtract %v5579, %v5582 : tensor<128xf32>
    %v5584 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5585 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5586 = stablehlo.multiply %v5584, %s2b0bt2m : tensor<128xf32>
    %v5587 = stablehlo.multiply %v5585, %v2940 : tensor<128xf32>
    %v5588 = stablehlo.add %v5586, %v5587 : tensor<128xf32>
    %v5589 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5590 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5591 = stablehlo.multiply %v5589, %s2b0bt2v : tensor<128xf32>
    %v5592 = stablehlo.multiply %v2940, %v2940 : tensor<128xf32>
    %v5593 = stablehlo.multiply %v5590, %v5592 : tensor<128xf32>
    %v5594 = stablehlo.add %v5591, %v5593 : tensor<128xf32>
    %v5595 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5596 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5597 = stablehlo.multiply %v5595, %s2b0bt2m : tensor<128xf32>
    %v5598 = stablehlo.multiply %v5596, %v2940 : tensor<128xf32>
    %v5599 = stablehlo.add %v5597, %v5598 : tensor<128xf32>
    %v5600 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5601 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5602 = stablehlo.multiply %v5600, %s2b0bt2v : tensor<128xf32>
    %v5603 = stablehlo.multiply %v2940, %v2940 : tensor<128xf32>
    %v5604 = stablehlo.multiply %v5601, %v5603 : tensor<128xf32>
    %v5605 = stablehlo.add %v5602, %v5604 : tensor<128xf32>
    %v5606 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5607 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5608 = stablehlo.divide %v5599, %v5606 : tensor<128xf32>
    %v5609 = stablehlo.divide %v5605, %v5607 : tensor<128xf32>
    %v5610 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5611 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5612 = stablehlo.sqrt %v5609 : tensor<128xf32>
    %v5613 = stablehlo.add %v5612, %v5611 : tensor<128xf32>
    %v5614 = stablehlo.divide %v5608, %v5613 : tensor<128xf32>
    %v5615 = stablehlo.multiply %v5610, %v5614 : tensor<128xf32>
    %v5616 = stablehlo.subtract %s2b0bt2, %v5615 : tensor<128xf32>
    %v5617 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5618 = stablehlo.multiply %v5617, %v5610 : tensor<128xf32>
    %v5619 = stablehlo.multiply %v5618, %s2b0bt2 : tensor<128xf32>
    %v5620 = stablehlo.subtract %v5616, %v5619 : tensor<128xf32>
    %v5621 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5622 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5623 = stablehlo.multiply %v5621, %s2b1W1m : tensor<128x128x3x3xf32>
    %v5624 = stablehlo.multiply %v5622, %v2752 : tensor<128x128x3x3xf32>
    %v5625 = stablehlo.add %v5623, %v5624 : tensor<128x128x3x3xf32>
    %v5626 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5627 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5628 = stablehlo.multiply %v5626, %s2b1W1v : tensor<128x128x3x3xf32>
    %v5629 = stablehlo.multiply %v2752, %v2752 : tensor<128x128x3x3xf32>
    %v5630 = stablehlo.multiply %v5627, %v5629 : tensor<128x128x3x3xf32>
    %v5631 = stablehlo.add %v5628, %v5630 : tensor<128x128x3x3xf32>
    %v5632 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5633 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5634 = stablehlo.multiply %v5632, %s2b1W1m : tensor<128x128x3x3xf32>
    %v5635 = stablehlo.multiply %v5633, %v2752 : tensor<128x128x3x3xf32>
    %v5636 = stablehlo.add %v5634, %v5635 : tensor<128x128x3x3xf32>
    %v5637 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5638 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5639 = stablehlo.multiply %v5637, %s2b1W1v : tensor<128x128x3x3xf32>
    %v5640 = stablehlo.multiply %v2752, %v2752 : tensor<128x128x3x3xf32>
    %v5641 = stablehlo.multiply %v5638, %v5640 : tensor<128x128x3x3xf32>
    %v5642 = stablehlo.add %v5639, %v5641 : tensor<128x128x3x3xf32>
    %v5643 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5644 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5645 = stablehlo.divide %v5636, %v5643 : tensor<128x128x3x3xf32>
    %v5646 = stablehlo.divide %v5642, %v5644 : tensor<128x128x3x3xf32>
    %v5647 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5648 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5649 = stablehlo.sqrt %v5646 : tensor<128x128x3x3xf32>
    %v5650 = stablehlo.add %v5649, %v5648 : tensor<128x128x3x3xf32>
    %v5651 = stablehlo.divide %v5645, %v5650 : tensor<128x128x3x3xf32>
    %v5652 = stablehlo.multiply %v5647, %v5651 : tensor<128x128x3x3xf32>
    %v5653 = stablehlo.subtract %s2b1W1, %v5652 : tensor<128x128x3x3xf32>
    %v5654 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5655 = stablehlo.multiply %v5654, %v5647 : tensor<128x128x3x3xf32>
    %v5656 = stablehlo.multiply %v5655, %s2b1W1 : tensor<128x128x3x3xf32>
    %v5657 = stablehlo.subtract %v5653, %v5656 : tensor<128x128x3x3xf32>
    %v5658 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5659 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5660 = stablehlo.multiply %v5658, %s2b1g1m : tensor<128xf32>
    %v5661 = stablehlo.multiply %v5659, %v2770 : tensor<128xf32>
    %v5662 = stablehlo.add %v5660, %v5661 : tensor<128xf32>
    %v5663 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5664 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5665 = stablehlo.multiply %v5663, %s2b1g1v : tensor<128xf32>
    %v5666 = stablehlo.multiply %v2770, %v2770 : tensor<128xf32>
    %v5667 = stablehlo.multiply %v5664, %v5666 : tensor<128xf32>
    %v5668 = stablehlo.add %v5665, %v5667 : tensor<128xf32>
    %v5669 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5670 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5671 = stablehlo.multiply %v5669, %s2b1g1m : tensor<128xf32>
    %v5672 = stablehlo.multiply %v5670, %v2770 : tensor<128xf32>
    %v5673 = stablehlo.add %v5671, %v5672 : tensor<128xf32>
    %v5674 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5675 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5676 = stablehlo.multiply %v5674, %s2b1g1v : tensor<128xf32>
    %v5677 = stablehlo.multiply %v2770, %v2770 : tensor<128xf32>
    %v5678 = stablehlo.multiply %v5675, %v5677 : tensor<128xf32>
    %v5679 = stablehlo.add %v5676, %v5678 : tensor<128xf32>
    %v5680 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5681 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5682 = stablehlo.divide %v5673, %v5680 : tensor<128xf32>
    %v5683 = stablehlo.divide %v5679, %v5681 : tensor<128xf32>
    %v5684 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5685 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5686 = stablehlo.sqrt %v5683 : tensor<128xf32>
    %v5687 = stablehlo.add %v5686, %v5685 : tensor<128xf32>
    %v5688 = stablehlo.divide %v5682, %v5687 : tensor<128xf32>
    %v5689 = stablehlo.multiply %v5684, %v5688 : tensor<128xf32>
    %v5690 = stablehlo.subtract %s2b1g1, %v5689 : tensor<128xf32>
    %v5691 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5692 = stablehlo.multiply %v5691, %v5684 : tensor<128xf32>
    %v5693 = stablehlo.multiply %v5692, %s2b1g1 : tensor<128xf32>
    %v5694 = stablehlo.subtract %v5690, %v5693 : tensor<128xf32>
    %v5695 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5696 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5697 = stablehlo.multiply %v5695, %s2b1bt1m : tensor<128xf32>
    %v5698 = stablehlo.multiply %v5696, %v2773 : tensor<128xf32>
    %v5699 = stablehlo.add %v5697, %v5698 : tensor<128xf32>
    %v5700 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5701 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5702 = stablehlo.multiply %v5700, %s2b1bt1v : tensor<128xf32>
    %v5703 = stablehlo.multiply %v2773, %v2773 : tensor<128xf32>
    %v5704 = stablehlo.multiply %v5701, %v5703 : tensor<128xf32>
    %v5705 = stablehlo.add %v5702, %v5704 : tensor<128xf32>
    %v5706 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5707 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5708 = stablehlo.multiply %v5706, %s2b1bt1m : tensor<128xf32>
    %v5709 = stablehlo.multiply %v5707, %v2773 : tensor<128xf32>
    %v5710 = stablehlo.add %v5708, %v5709 : tensor<128xf32>
    %v5711 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5712 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5713 = stablehlo.multiply %v5711, %s2b1bt1v : tensor<128xf32>
    %v5714 = stablehlo.multiply %v2773, %v2773 : tensor<128xf32>
    %v5715 = stablehlo.multiply %v5712, %v5714 : tensor<128xf32>
    %v5716 = stablehlo.add %v5713, %v5715 : tensor<128xf32>
    %v5717 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5718 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5719 = stablehlo.divide %v5710, %v5717 : tensor<128xf32>
    %v5720 = stablehlo.divide %v5716, %v5718 : tensor<128xf32>
    %v5721 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5722 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5723 = stablehlo.sqrt %v5720 : tensor<128xf32>
    %v5724 = stablehlo.add %v5723, %v5722 : tensor<128xf32>
    %v5725 = stablehlo.divide %v5719, %v5724 : tensor<128xf32>
    %v5726 = stablehlo.multiply %v5721, %v5725 : tensor<128xf32>
    %v5727 = stablehlo.subtract %s2b1bt1, %v5726 : tensor<128xf32>
    %v5728 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5729 = stablehlo.multiply %v5728, %v5721 : tensor<128xf32>
    %v5730 = stablehlo.multiply %v5729, %s2b1bt1 : tensor<128xf32>
    %v5731 = stablehlo.subtract %v5727, %v5730 : tensor<128xf32>
    %v5732 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5733 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5734 = stablehlo.multiply %v5732, %s2b1W2m : tensor<128x128x3x3xf32>
    %v5735 = stablehlo.multiply %v5733, %v2779 : tensor<128x128x3x3xf32>
    %v5736 = stablehlo.add %v5734, %v5735 : tensor<128x128x3x3xf32>
    %v5737 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5738 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5739 = stablehlo.multiply %v5737, %s2b1W2v : tensor<128x128x3x3xf32>
    %v5740 = stablehlo.multiply %v2779, %v2779 : tensor<128x128x3x3xf32>
    %v5741 = stablehlo.multiply %v5738, %v5740 : tensor<128x128x3x3xf32>
    %v5742 = stablehlo.add %v5739, %v5741 : tensor<128x128x3x3xf32>
    %v5743 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5744 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5745 = stablehlo.multiply %v5743, %s2b1W2m : tensor<128x128x3x3xf32>
    %v5746 = stablehlo.multiply %v5744, %v2779 : tensor<128x128x3x3xf32>
    %v5747 = stablehlo.add %v5745, %v5746 : tensor<128x128x3x3xf32>
    %v5748 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5749 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5750 = stablehlo.multiply %v5748, %s2b1W2v : tensor<128x128x3x3xf32>
    %v5751 = stablehlo.multiply %v2779, %v2779 : tensor<128x128x3x3xf32>
    %v5752 = stablehlo.multiply %v5749, %v5751 : tensor<128x128x3x3xf32>
    %v5753 = stablehlo.add %v5750, %v5752 : tensor<128x128x3x3xf32>
    %v5754 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5755 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5756 = stablehlo.divide %v5747, %v5754 : tensor<128x128x3x3xf32>
    %v5757 = stablehlo.divide %v5753, %v5755 : tensor<128x128x3x3xf32>
    %v5758 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5759 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5760 = stablehlo.sqrt %v5757 : tensor<128x128x3x3xf32>
    %v5761 = stablehlo.add %v5760, %v5759 : tensor<128x128x3x3xf32>
    %v5762 = stablehlo.divide %v5756, %v5761 : tensor<128x128x3x3xf32>
    %v5763 = stablehlo.multiply %v5758, %v5762 : tensor<128x128x3x3xf32>
    %v5764 = stablehlo.subtract %s2b1W2, %v5763 : tensor<128x128x3x3xf32>
    %v5765 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5766 = stablehlo.multiply %v5765, %v5758 : tensor<128x128x3x3xf32>
    %v5767 = stablehlo.multiply %v5766, %s2b1W2 : tensor<128x128x3x3xf32>
    %v5768 = stablehlo.subtract %v5764, %v5767 : tensor<128x128x3x3xf32>
    %v5769 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5770 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5771 = stablehlo.multiply %v5769, %s2b1g2m : tensor<128xf32>
    %v5772 = stablehlo.multiply %v5770, %v2797 : tensor<128xf32>
    %v5773 = stablehlo.add %v5771, %v5772 : tensor<128xf32>
    %v5774 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5775 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5776 = stablehlo.multiply %v5774, %s2b1g2v : tensor<128xf32>
    %v5777 = stablehlo.multiply %v2797, %v2797 : tensor<128xf32>
    %v5778 = stablehlo.multiply %v5775, %v5777 : tensor<128xf32>
    %v5779 = stablehlo.add %v5776, %v5778 : tensor<128xf32>
    %v5780 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5781 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5782 = stablehlo.multiply %v5780, %s2b1g2m : tensor<128xf32>
    %v5783 = stablehlo.multiply %v5781, %v2797 : tensor<128xf32>
    %v5784 = stablehlo.add %v5782, %v5783 : tensor<128xf32>
    %v5785 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5786 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5787 = stablehlo.multiply %v5785, %s2b1g2v : tensor<128xf32>
    %v5788 = stablehlo.multiply %v2797, %v2797 : tensor<128xf32>
    %v5789 = stablehlo.multiply %v5786, %v5788 : tensor<128xf32>
    %v5790 = stablehlo.add %v5787, %v5789 : tensor<128xf32>
    %v5791 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5792 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5793 = stablehlo.divide %v5784, %v5791 : tensor<128xf32>
    %v5794 = stablehlo.divide %v5790, %v5792 : tensor<128xf32>
    %v5795 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5796 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5797 = stablehlo.sqrt %v5794 : tensor<128xf32>
    %v5798 = stablehlo.add %v5797, %v5796 : tensor<128xf32>
    %v5799 = stablehlo.divide %v5793, %v5798 : tensor<128xf32>
    %v5800 = stablehlo.multiply %v5795, %v5799 : tensor<128xf32>
    %v5801 = stablehlo.subtract %s2b1g2, %v5800 : tensor<128xf32>
    %v5802 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5803 = stablehlo.multiply %v5802, %v5795 : tensor<128xf32>
    %v5804 = stablehlo.multiply %v5803, %s2b1g2 : tensor<128xf32>
    %v5805 = stablehlo.subtract %v5801, %v5804 : tensor<128xf32>
    %v5806 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5807 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5808 = stablehlo.multiply %v5806, %s2b1bt2m : tensor<128xf32>
    %v5809 = stablehlo.multiply %v5807, %v2800 : tensor<128xf32>
    %v5810 = stablehlo.add %v5808, %v5809 : tensor<128xf32>
    %v5811 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5812 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5813 = stablehlo.multiply %v5811, %s2b1bt2v : tensor<128xf32>
    %v5814 = stablehlo.multiply %v2800, %v2800 : tensor<128xf32>
    %v5815 = stablehlo.multiply %v5812, %v5814 : tensor<128xf32>
    %v5816 = stablehlo.add %v5813, %v5815 : tensor<128xf32>
    %v5817 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5818 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5819 = stablehlo.multiply %v5817, %s2b1bt2m : tensor<128xf32>
    %v5820 = stablehlo.multiply %v5818, %v2800 : tensor<128xf32>
    %v5821 = stablehlo.add %v5819, %v5820 : tensor<128xf32>
    %v5822 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5823 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5824 = stablehlo.multiply %v5822, %s2b1bt2v : tensor<128xf32>
    %v5825 = stablehlo.multiply %v2800, %v2800 : tensor<128xf32>
    %v5826 = stablehlo.multiply %v5823, %v5825 : tensor<128xf32>
    %v5827 = stablehlo.add %v5824, %v5826 : tensor<128xf32>
    %v5828 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5829 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5830 = stablehlo.divide %v5821, %v5828 : tensor<128xf32>
    %v5831 = stablehlo.divide %v5827, %v5829 : tensor<128xf32>
    %v5832 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5833 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5834 = stablehlo.sqrt %v5831 : tensor<128xf32>
    %v5835 = stablehlo.add %v5834, %v5833 : tensor<128xf32>
    %v5836 = stablehlo.divide %v5830, %v5835 : tensor<128xf32>
    %v5837 = stablehlo.multiply %v5832, %v5836 : tensor<128xf32>
    %v5838 = stablehlo.subtract %s2b1bt2, %v5837 : tensor<128xf32>
    %v5839 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5840 = stablehlo.multiply %v5839, %v5832 : tensor<128xf32>
    %v5841 = stablehlo.multiply %v5840, %s2b1bt2 : tensor<128xf32>
    %v5842 = stablehlo.subtract %v5838, %v5841 : tensor<128xf32>
    %v5843 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5844 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5845 = stablehlo.multiply %v5843, %s2b2W1m : tensor<128x128x3x3xf32>
    %v5846 = stablehlo.multiply %v5844, %v2612 : tensor<128x128x3x3xf32>
    %v5847 = stablehlo.add %v5845, %v5846 : tensor<128x128x3x3xf32>
    %v5848 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5849 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5850 = stablehlo.multiply %v5848, %s2b2W1v : tensor<128x128x3x3xf32>
    %v5851 = stablehlo.multiply %v2612, %v2612 : tensor<128x128x3x3xf32>
    %v5852 = stablehlo.multiply %v5849, %v5851 : tensor<128x128x3x3xf32>
    %v5853 = stablehlo.add %v5850, %v5852 : tensor<128x128x3x3xf32>
    %v5854 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5855 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5856 = stablehlo.multiply %v5854, %s2b2W1m : tensor<128x128x3x3xf32>
    %v5857 = stablehlo.multiply %v5855, %v2612 : tensor<128x128x3x3xf32>
    %v5858 = stablehlo.add %v5856, %v5857 : tensor<128x128x3x3xf32>
    %v5859 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5860 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5861 = stablehlo.multiply %v5859, %s2b2W1v : tensor<128x128x3x3xf32>
    %v5862 = stablehlo.multiply %v2612, %v2612 : tensor<128x128x3x3xf32>
    %v5863 = stablehlo.multiply %v5860, %v5862 : tensor<128x128x3x3xf32>
    %v5864 = stablehlo.add %v5861, %v5863 : tensor<128x128x3x3xf32>
    %v5865 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5866 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5867 = stablehlo.divide %v5858, %v5865 : tensor<128x128x3x3xf32>
    %v5868 = stablehlo.divide %v5864, %v5866 : tensor<128x128x3x3xf32>
    %v5869 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5870 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5871 = stablehlo.sqrt %v5868 : tensor<128x128x3x3xf32>
    %v5872 = stablehlo.add %v5871, %v5870 : tensor<128x128x3x3xf32>
    %v5873 = stablehlo.divide %v5867, %v5872 : tensor<128x128x3x3xf32>
    %v5874 = stablehlo.multiply %v5869, %v5873 : tensor<128x128x3x3xf32>
    %v5875 = stablehlo.subtract %s2b2W1, %v5874 : tensor<128x128x3x3xf32>
    %v5876 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5877 = stablehlo.multiply %v5876, %v5869 : tensor<128x128x3x3xf32>
    %v5878 = stablehlo.multiply %v5877, %s2b2W1 : tensor<128x128x3x3xf32>
    %v5879 = stablehlo.subtract %v5875, %v5878 : tensor<128x128x3x3xf32>
    %v5880 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5881 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5882 = stablehlo.multiply %v5880, %s2b2g1m : tensor<128xf32>
    %v5883 = stablehlo.multiply %v5881, %v2630 : tensor<128xf32>
    %v5884 = stablehlo.add %v5882, %v5883 : tensor<128xf32>
    %v5885 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5886 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5887 = stablehlo.multiply %v5885, %s2b2g1v : tensor<128xf32>
    %v5888 = stablehlo.multiply %v2630, %v2630 : tensor<128xf32>
    %v5889 = stablehlo.multiply %v5886, %v5888 : tensor<128xf32>
    %v5890 = stablehlo.add %v5887, %v5889 : tensor<128xf32>
    %v5891 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5892 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5893 = stablehlo.multiply %v5891, %s2b2g1m : tensor<128xf32>
    %v5894 = stablehlo.multiply %v5892, %v2630 : tensor<128xf32>
    %v5895 = stablehlo.add %v5893, %v5894 : tensor<128xf32>
    %v5896 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5897 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5898 = stablehlo.multiply %v5896, %s2b2g1v : tensor<128xf32>
    %v5899 = stablehlo.multiply %v2630, %v2630 : tensor<128xf32>
    %v5900 = stablehlo.multiply %v5897, %v5899 : tensor<128xf32>
    %v5901 = stablehlo.add %v5898, %v5900 : tensor<128xf32>
    %v5902 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5903 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5904 = stablehlo.divide %v5895, %v5902 : tensor<128xf32>
    %v5905 = stablehlo.divide %v5901, %v5903 : tensor<128xf32>
    %v5906 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5907 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5908 = stablehlo.sqrt %v5905 : tensor<128xf32>
    %v5909 = stablehlo.add %v5908, %v5907 : tensor<128xf32>
    %v5910 = stablehlo.divide %v5904, %v5909 : tensor<128xf32>
    %v5911 = stablehlo.multiply %v5906, %v5910 : tensor<128xf32>
    %v5912 = stablehlo.subtract %s2b2g1, %v5911 : tensor<128xf32>
    %v5913 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5914 = stablehlo.multiply %v5913, %v5906 : tensor<128xf32>
    %v5915 = stablehlo.multiply %v5914, %s2b2g1 : tensor<128xf32>
    %v5916 = stablehlo.subtract %v5912, %v5915 : tensor<128xf32>
    %v5917 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5918 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5919 = stablehlo.multiply %v5917, %s2b2bt1m : tensor<128xf32>
    %v5920 = stablehlo.multiply %v5918, %v2633 : tensor<128xf32>
    %v5921 = stablehlo.add %v5919, %v5920 : tensor<128xf32>
    %v5922 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5923 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5924 = stablehlo.multiply %v5922, %s2b2bt1v : tensor<128xf32>
    %v5925 = stablehlo.multiply %v2633, %v2633 : tensor<128xf32>
    %v5926 = stablehlo.multiply %v5923, %v5925 : tensor<128xf32>
    %v5927 = stablehlo.add %v5924, %v5926 : tensor<128xf32>
    %v5928 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5929 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5930 = stablehlo.multiply %v5928, %s2b2bt1m : tensor<128xf32>
    %v5931 = stablehlo.multiply %v5929, %v2633 : tensor<128xf32>
    %v5932 = stablehlo.add %v5930, %v5931 : tensor<128xf32>
    %v5933 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5934 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5935 = stablehlo.multiply %v5933, %s2b2bt1v : tensor<128xf32>
    %v5936 = stablehlo.multiply %v2633, %v2633 : tensor<128xf32>
    %v5937 = stablehlo.multiply %v5934, %v5936 : tensor<128xf32>
    %v5938 = stablehlo.add %v5935, %v5937 : tensor<128xf32>
    %v5939 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5940 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5941 = stablehlo.divide %v5932, %v5939 : tensor<128xf32>
    %v5942 = stablehlo.divide %v5938, %v5940 : tensor<128xf32>
    %v5943 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5944 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5945 = stablehlo.sqrt %v5942 : tensor<128xf32>
    %v5946 = stablehlo.add %v5945, %v5944 : tensor<128xf32>
    %v5947 = stablehlo.divide %v5941, %v5946 : tensor<128xf32>
    %v5948 = stablehlo.multiply %v5943, %v5947 : tensor<128xf32>
    %v5949 = stablehlo.subtract %s2b2bt1, %v5948 : tensor<128xf32>
    %v5950 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5951 = stablehlo.multiply %v5950, %v5943 : tensor<128xf32>
    %v5952 = stablehlo.multiply %v5951, %s2b2bt1 : tensor<128xf32>
    %v5953 = stablehlo.subtract %v5949, %v5952 : tensor<128xf32>
    %v5954 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5955 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5956 = stablehlo.multiply %v5954, %s2b2W2m : tensor<128x128x3x3xf32>
    %v5957 = stablehlo.multiply %v5955, %v2639 : tensor<128x128x3x3xf32>
    %v5958 = stablehlo.add %v5956, %v5957 : tensor<128x128x3x3xf32>
    %v5959 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5960 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5961 = stablehlo.multiply %v5959, %s2b2W2v : tensor<128x128x3x3xf32>
    %v5962 = stablehlo.multiply %v2639, %v2639 : tensor<128x128x3x3xf32>
    %v5963 = stablehlo.multiply %v5960, %v5962 : tensor<128x128x3x3xf32>
    %v5964 = stablehlo.add %v5961, %v5963 : tensor<128x128x3x3xf32>
    %v5965 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5966 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5967 = stablehlo.multiply %v5965, %s2b2W2m : tensor<128x128x3x3xf32>
    %v5968 = stablehlo.multiply %v5966, %v2639 : tensor<128x128x3x3xf32>
    %v5969 = stablehlo.add %v5967, %v5968 : tensor<128x128x3x3xf32>
    %v5970 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5971 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5972 = stablehlo.multiply %v5970, %s2b2W2v : tensor<128x128x3x3xf32>
    %v5973 = stablehlo.multiply %v2639, %v2639 : tensor<128x128x3x3xf32>
    %v5974 = stablehlo.multiply %v5971, %v5973 : tensor<128x128x3x3xf32>
    %v5975 = stablehlo.add %v5972, %v5974 : tensor<128x128x3x3xf32>
    %v5976 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5977 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5978 = stablehlo.divide %v5969, %v5976 : tensor<128x128x3x3xf32>
    %v5979 = stablehlo.divide %v5975, %v5977 : tensor<128x128x3x3xf32>
    %v5980 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5981 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5982 = stablehlo.sqrt %v5979 : tensor<128x128x3x3xf32>
    %v5983 = stablehlo.add %v5982, %v5981 : tensor<128x128x3x3xf32>
    %v5984 = stablehlo.divide %v5978, %v5983 : tensor<128x128x3x3xf32>
    %v5985 = stablehlo.multiply %v5980, %v5984 : tensor<128x128x3x3xf32>
    %v5986 = stablehlo.subtract %s2b2W2, %v5985 : tensor<128x128x3x3xf32>
    %v5987 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5988 = stablehlo.multiply %v5987, %v5980 : tensor<128x128x3x3xf32>
    %v5989 = stablehlo.multiply %v5988, %s2b2W2 : tensor<128x128x3x3xf32>
    %v5990 = stablehlo.subtract %v5986, %v5989 : tensor<128x128x3x3xf32>
    %v5991 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5992 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5993 = stablehlo.multiply %v5991, %s2b2g2m : tensor<128xf32>
    %v5994 = stablehlo.multiply %v5992, %v2657 : tensor<128xf32>
    %v5995 = stablehlo.add %v5993, %v5994 : tensor<128xf32>
    %v5996 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5997 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5998 = stablehlo.multiply %v5996, %s2b2g2v : tensor<128xf32>
    %v5999 = stablehlo.multiply %v2657, %v2657 : tensor<128xf32>
    %v6000 = stablehlo.multiply %v5997, %v5999 : tensor<128xf32>
    %v6001 = stablehlo.add %v5998, %v6000 : tensor<128xf32>
    %v6002 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6003 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6004 = stablehlo.multiply %v6002, %s2b2g2m : tensor<128xf32>
    %v6005 = stablehlo.multiply %v6003, %v2657 : tensor<128xf32>
    %v6006 = stablehlo.add %v6004, %v6005 : tensor<128xf32>
    %v6007 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6008 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6009 = stablehlo.multiply %v6007, %s2b2g2v : tensor<128xf32>
    %v6010 = stablehlo.multiply %v2657, %v2657 : tensor<128xf32>
    %v6011 = stablehlo.multiply %v6008, %v6010 : tensor<128xf32>
    %v6012 = stablehlo.add %v6009, %v6011 : tensor<128xf32>
    %v6013 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6014 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6015 = stablehlo.divide %v6006, %v6013 : tensor<128xf32>
    %v6016 = stablehlo.divide %v6012, %v6014 : tensor<128xf32>
    %v6017 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6018 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6019 = stablehlo.sqrt %v6016 : tensor<128xf32>
    %v6020 = stablehlo.add %v6019, %v6018 : tensor<128xf32>
    %v6021 = stablehlo.divide %v6015, %v6020 : tensor<128xf32>
    %v6022 = stablehlo.multiply %v6017, %v6021 : tensor<128xf32>
    %v6023 = stablehlo.subtract %s2b2g2, %v6022 : tensor<128xf32>
    %v6024 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6025 = stablehlo.multiply %v6024, %v6017 : tensor<128xf32>
    %v6026 = stablehlo.multiply %v6025, %s2b2g2 : tensor<128xf32>
    %v6027 = stablehlo.subtract %v6023, %v6026 : tensor<128xf32>
    %v6028 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6029 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6030 = stablehlo.multiply %v6028, %s2b2bt2m : tensor<128xf32>
    %v6031 = stablehlo.multiply %v6029, %v2660 : tensor<128xf32>
    %v6032 = stablehlo.add %v6030, %v6031 : tensor<128xf32>
    %v6033 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6034 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6035 = stablehlo.multiply %v6033, %s2b2bt2v : tensor<128xf32>
    %v6036 = stablehlo.multiply %v2660, %v2660 : tensor<128xf32>
    %v6037 = stablehlo.multiply %v6034, %v6036 : tensor<128xf32>
    %v6038 = stablehlo.add %v6035, %v6037 : tensor<128xf32>
    %v6039 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6040 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6041 = stablehlo.multiply %v6039, %s2b2bt2m : tensor<128xf32>
    %v6042 = stablehlo.multiply %v6040, %v2660 : tensor<128xf32>
    %v6043 = stablehlo.add %v6041, %v6042 : tensor<128xf32>
    %v6044 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6045 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6046 = stablehlo.multiply %v6044, %s2b2bt2v : tensor<128xf32>
    %v6047 = stablehlo.multiply %v2660, %v2660 : tensor<128xf32>
    %v6048 = stablehlo.multiply %v6045, %v6047 : tensor<128xf32>
    %v6049 = stablehlo.add %v6046, %v6048 : tensor<128xf32>
    %v6050 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6051 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6052 = stablehlo.divide %v6043, %v6050 : tensor<128xf32>
    %v6053 = stablehlo.divide %v6049, %v6051 : tensor<128xf32>
    %v6054 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6055 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6056 = stablehlo.sqrt %v6053 : tensor<128xf32>
    %v6057 = stablehlo.add %v6056, %v6055 : tensor<128xf32>
    %v6058 = stablehlo.divide %v6052, %v6057 : tensor<128xf32>
    %v6059 = stablehlo.multiply %v6054, %v6058 : tensor<128xf32>
    %v6060 = stablehlo.subtract %s2b2bt2, %v6059 : tensor<128xf32>
    %v6061 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6062 = stablehlo.multiply %v6061, %v6054 : tensor<128xf32>
    %v6063 = stablehlo.multiply %v6062, %s2b2bt2 : tensor<128xf32>
    %v6064 = stablehlo.subtract %v6060, %v6063 : tensor<128xf32>
    %v6065 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6066 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6067 = stablehlo.multiply %v6065, %d3W1m : tensor<256x128x3x3xf32>
    %v6068 = stablehlo.multiply %v6066, %v2444 : tensor<256x128x3x3xf32>
    %v6069 = stablehlo.add %v6067, %v6068 : tensor<256x128x3x3xf32>
    %v6070 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6071 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6072 = stablehlo.multiply %v6070, %d3W1v : tensor<256x128x3x3xf32>
    %v6073 = stablehlo.multiply %v2444, %v2444 : tensor<256x128x3x3xf32>
    %v6074 = stablehlo.multiply %v6071, %v6073 : tensor<256x128x3x3xf32>
    %v6075 = stablehlo.add %v6072, %v6074 : tensor<256x128x3x3xf32>
    %v6076 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6077 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6078 = stablehlo.multiply %v6076, %d3W1m : tensor<256x128x3x3xf32>
    %v6079 = stablehlo.multiply %v6077, %v2444 : tensor<256x128x3x3xf32>
    %v6080 = stablehlo.add %v6078, %v6079 : tensor<256x128x3x3xf32>
    %v6081 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6082 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6083 = stablehlo.multiply %v6081, %d3W1v : tensor<256x128x3x3xf32>
    %v6084 = stablehlo.multiply %v2444, %v2444 : tensor<256x128x3x3xf32>
    %v6085 = stablehlo.multiply %v6082, %v6084 : tensor<256x128x3x3xf32>
    %v6086 = stablehlo.add %v6083, %v6085 : tensor<256x128x3x3xf32>
    %v6087 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6088 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6089 = stablehlo.divide %v6080, %v6087 : tensor<256x128x3x3xf32>
    %v6090 = stablehlo.divide %v6086, %v6088 : tensor<256x128x3x3xf32>
    %v6091 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6092 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6093 = stablehlo.sqrt %v6090 : tensor<256x128x3x3xf32>
    %v6094 = stablehlo.add %v6093, %v6092 : tensor<256x128x3x3xf32>
    %v6095 = stablehlo.divide %v6089, %v6094 : tensor<256x128x3x3xf32>
    %v6096 = stablehlo.multiply %v6091, %v6095 : tensor<256x128x3x3xf32>
    %v6097 = stablehlo.subtract %d3W1, %v6096 : tensor<256x128x3x3xf32>
    %v6098 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6099 = stablehlo.multiply %v6098, %v6091 : tensor<256x128x3x3xf32>
    %v6100 = stablehlo.multiply %v6099, %d3W1 : tensor<256x128x3x3xf32>
    %v6101 = stablehlo.subtract %v6097, %v6100 : tensor<256x128x3x3xf32>
    %v6102 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6103 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6104 = stablehlo.multiply %v6102, %d3g1m : tensor<256xf32>
    %v6105 = stablehlo.multiply %v6103, %v2462 : tensor<256xf32>
    %v6106 = stablehlo.add %v6104, %v6105 : tensor<256xf32>
    %v6107 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6108 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6109 = stablehlo.multiply %v6107, %d3g1v : tensor<256xf32>
    %v6110 = stablehlo.multiply %v2462, %v2462 : tensor<256xf32>
    %v6111 = stablehlo.multiply %v6108, %v6110 : tensor<256xf32>
    %v6112 = stablehlo.add %v6109, %v6111 : tensor<256xf32>
    %v6113 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6114 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6115 = stablehlo.multiply %v6113, %d3g1m : tensor<256xf32>
    %v6116 = stablehlo.multiply %v6114, %v2462 : tensor<256xf32>
    %v6117 = stablehlo.add %v6115, %v6116 : tensor<256xf32>
    %v6118 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6119 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6120 = stablehlo.multiply %v6118, %d3g1v : tensor<256xf32>
    %v6121 = stablehlo.multiply %v2462, %v2462 : tensor<256xf32>
    %v6122 = stablehlo.multiply %v6119, %v6121 : tensor<256xf32>
    %v6123 = stablehlo.add %v6120, %v6122 : tensor<256xf32>
    %v6124 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6125 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6126 = stablehlo.divide %v6117, %v6124 : tensor<256xf32>
    %v6127 = stablehlo.divide %v6123, %v6125 : tensor<256xf32>
    %v6128 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6129 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6130 = stablehlo.sqrt %v6127 : tensor<256xf32>
    %v6131 = stablehlo.add %v6130, %v6129 : tensor<256xf32>
    %v6132 = stablehlo.divide %v6126, %v6131 : tensor<256xf32>
    %v6133 = stablehlo.multiply %v6128, %v6132 : tensor<256xf32>
    %v6134 = stablehlo.subtract %d3g1, %v6133 : tensor<256xf32>
    %v6135 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6136 = stablehlo.multiply %v6135, %v6128 : tensor<256xf32>
    %v6137 = stablehlo.multiply %v6136, %d3g1 : tensor<256xf32>
    %v6138 = stablehlo.subtract %v6134, %v6137 : tensor<256xf32>
    %v6139 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6140 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6141 = stablehlo.multiply %v6139, %d3bt1m : tensor<256xf32>
    %v6142 = stablehlo.multiply %v6140, %v2465 : tensor<256xf32>
    %v6143 = stablehlo.add %v6141, %v6142 : tensor<256xf32>
    %v6144 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6145 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6146 = stablehlo.multiply %v6144, %d3bt1v : tensor<256xf32>
    %v6147 = stablehlo.multiply %v2465, %v2465 : tensor<256xf32>
    %v6148 = stablehlo.multiply %v6145, %v6147 : tensor<256xf32>
    %v6149 = stablehlo.add %v6146, %v6148 : tensor<256xf32>
    %v6150 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6151 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6152 = stablehlo.multiply %v6150, %d3bt1m : tensor<256xf32>
    %v6153 = stablehlo.multiply %v6151, %v2465 : tensor<256xf32>
    %v6154 = stablehlo.add %v6152, %v6153 : tensor<256xf32>
    %v6155 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6156 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6157 = stablehlo.multiply %v6155, %d3bt1v : tensor<256xf32>
    %v6158 = stablehlo.multiply %v2465, %v2465 : tensor<256xf32>
    %v6159 = stablehlo.multiply %v6156, %v6158 : tensor<256xf32>
    %v6160 = stablehlo.add %v6157, %v6159 : tensor<256xf32>
    %v6161 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6162 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6163 = stablehlo.divide %v6154, %v6161 : tensor<256xf32>
    %v6164 = stablehlo.divide %v6160, %v6162 : tensor<256xf32>
    %v6165 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6166 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6167 = stablehlo.sqrt %v6164 : tensor<256xf32>
    %v6168 = stablehlo.add %v6167, %v6166 : tensor<256xf32>
    %v6169 = stablehlo.divide %v6163, %v6168 : tensor<256xf32>
    %v6170 = stablehlo.multiply %v6165, %v6169 : tensor<256xf32>
    %v6171 = stablehlo.subtract %d3bt1, %v6170 : tensor<256xf32>
    %v6172 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6173 = stablehlo.multiply %v6172, %v6165 : tensor<256xf32>
    %v6174 = stablehlo.multiply %v6173, %d3bt1 : tensor<256xf32>
    %v6175 = stablehlo.subtract %v6171, %v6174 : tensor<256xf32>
    %v6176 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6177 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6178 = stablehlo.multiply %v6176, %d3W2m : tensor<256x256x3x3xf32>
    %v6179 = stablehlo.multiply %v6177, %v2471 : tensor<256x256x3x3xf32>
    %v6180 = stablehlo.add %v6178, %v6179 : tensor<256x256x3x3xf32>
    %v6181 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6182 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6183 = stablehlo.multiply %v6181, %d3W2v : tensor<256x256x3x3xf32>
    %v6184 = stablehlo.multiply %v2471, %v2471 : tensor<256x256x3x3xf32>
    %v6185 = stablehlo.multiply %v6182, %v6184 : tensor<256x256x3x3xf32>
    %v6186 = stablehlo.add %v6183, %v6185 : tensor<256x256x3x3xf32>
    %v6187 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6188 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6189 = stablehlo.multiply %v6187, %d3W2m : tensor<256x256x3x3xf32>
    %v6190 = stablehlo.multiply %v6188, %v2471 : tensor<256x256x3x3xf32>
    %v6191 = stablehlo.add %v6189, %v6190 : tensor<256x256x3x3xf32>
    %v6192 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6193 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6194 = stablehlo.multiply %v6192, %d3W2v : tensor<256x256x3x3xf32>
    %v6195 = stablehlo.multiply %v2471, %v2471 : tensor<256x256x3x3xf32>
    %v6196 = stablehlo.multiply %v6193, %v6195 : tensor<256x256x3x3xf32>
    %v6197 = stablehlo.add %v6194, %v6196 : tensor<256x256x3x3xf32>
    %v6198 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6199 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6200 = stablehlo.divide %v6191, %v6198 : tensor<256x256x3x3xf32>
    %v6201 = stablehlo.divide %v6197, %v6199 : tensor<256x256x3x3xf32>
    %v6202 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6203 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6204 = stablehlo.sqrt %v6201 : tensor<256x256x3x3xf32>
    %v6205 = stablehlo.add %v6204, %v6203 : tensor<256x256x3x3xf32>
    %v6206 = stablehlo.divide %v6200, %v6205 : tensor<256x256x3x3xf32>
    %v6207 = stablehlo.multiply %v6202, %v6206 : tensor<256x256x3x3xf32>
    %v6208 = stablehlo.subtract %d3W2, %v6207 : tensor<256x256x3x3xf32>
    %v6209 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6210 = stablehlo.multiply %v6209, %v6202 : tensor<256x256x3x3xf32>
    %v6211 = stablehlo.multiply %v6210, %d3W2 : tensor<256x256x3x3xf32>
    %v6212 = stablehlo.subtract %v6208, %v6211 : tensor<256x256x3x3xf32>
    %v6213 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6214 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6215 = stablehlo.multiply %v6213, %d3g2m : tensor<256xf32>
    %v6216 = stablehlo.multiply %v6214, %v2489 : tensor<256xf32>
    %v6217 = stablehlo.add %v6215, %v6216 : tensor<256xf32>
    %v6218 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6219 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6220 = stablehlo.multiply %v6218, %d3g2v : tensor<256xf32>
    %v6221 = stablehlo.multiply %v2489, %v2489 : tensor<256xf32>
    %v6222 = stablehlo.multiply %v6219, %v6221 : tensor<256xf32>
    %v6223 = stablehlo.add %v6220, %v6222 : tensor<256xf32>
    %v6224 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6225 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6226 = stablehlo.multiply %v6224, %d3g2m : tensor<256xf32>
    %v6227 = stablehlo.multiply %v6225, %v2489 : tensor<256xf32>
    %v6228 = stablehlo.add %v6226, %v6227 : tensor<256xf32>
    %v6229 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6230 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6231 = stablehlo.multiply %v6229, %d3g2v : tensor<256xf32>
    %v6232 = stablehlo.multiply %v2489, %v2489 : tensor<256xf32>
    %v6233 = stablehlo.multiply %v6230, %v6232 : tensor<256xf32>
    %v6234 = stablehlo.add %v6231, %v6233 : tensor<256xf32>
    %v6235 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6236 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6237 = stablehlo.divide %v6228, %v6235 : tensor<256xf32>
    %v6238 = stablehlo.divide %v6234, %v6236 : tensor<256xf32>
    %v6239 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6240 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6241 = stablehlo.sqrt %v6238 : tensor<256xf32>
    %v6242 = stablehlo.add %v6241, %v6240 : tensor<256xf32>
    %v6243 = stablehlo.divide %v6237, %v6242 : tensor<256xf32>
    %v6244 = stablehlo.multiply %v6239, %v6243 : tensor<256xf32>
    %v6245 = stablehlo.subtract %d3g2, %v6244 : tensor<256xf32>
    %v6246 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6247 = stablehlo.multiply %v6246, %v6239 : tensor<256xf32>
    %v6248 = stablehlo.multiply %v6247, %d3g2 : tensor<256xf32>
    %v6249 = stablehlo.subtract %v6245, %v6248 : tensor<256xf32>
    %v6250 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6251 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6252 = stablehlo.multiply %v6250, %d3bt2m : tensor<256xf32>
    %v6253 = stablehlo.multiply %v6251, %v2492 : tensor<256xf32>
    %v6254 = stablehlo.add %v6252, %v6253 : tensor<256xf32>
    %v6255 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6256 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6257 = stablehlo.multiply %v6255, %d3bt2v : tensor<256xf32>
    %v6258 = stablehlo.multiply %v2492, %v2492 : tensor<256xf32>
    %v6259 = stablehlo.multiply %v6256, %v6258 : tensor<256xf32>
    %v6260 = stablehlo.add %v6257, %v6259 : tensor<256xf32>
    %v6261 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6262 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6263 = stablehlo.multiply %v6261, %d3bt2m : tensor<256xf32>
    %v6264 = stablehlo.multiply %v6262, %v2492 : tensor<256xf32>
    %v6265 = stablehlo.add %v6263, %v6264 : tensor<256xf32>
    %v6266 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6267 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6268 = stablehlo.multiply %v6266, %d3bt2v : tensor<256xf32>
    %v6269 = stablehlo.multiply %v2492, %v2492 : tensor<256xf32>
    %v6270 = stablehlo.multiply %v6267, %v6269 : tensor<256xf32>
    %v6271 = stablehlo.add %v6268, %v6270 : tensor<256xf32>
    %v6272 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6273 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6274 = stablehlo.divide %v6265, %v6272 : tensor<256xf32>
    %v6275 = stablehlo.divide %v6271, %v6273 : tensor<256xf32>
    %v6276 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6277 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6278 = stablehlo.sqrt %v6275 : tensor<256xf32>
    %v6279 = stablehlo.add %v6278, %v6277 : tensor<256xf32>
    %v6280 = stablehlo.divide %v6274, %v6279 : tensor<256xf32>
    %v6281 = stablehlo.multiply %v6276, %v6280 : tensor<256xf32>
    %v6282 = stablehlo.subtract %d3bt2, %v6281 : tensor<256xf32>
    %v6283 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6284 = stablehlo.multiply %v6283, %v6276 : tensor<256xf32>
    %v6285 = stablehlo.multiply %v6284, %d3bt2 : tensor<256xf32>
    %v6286 = stablehlo.subtract %v6282, %v6285 : tensor<256xf32>
    %v6287 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6288 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6289 = stablehlo.multiply %v6287, %d3Wpm : tensor<256x128x1x1xf32>
    %v6290 = stablehlo.multiply %v6288, %v2499 : tensor<256x128x1x1xf32>
    %v6291 = stablehlo.add %v6289, %v6290 : tensor<256x128x1x1xf32>
    %v6292 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6293 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6294 = stablehlo.multiply %v6292, %d3Wpv : tensor<256x128x1x1xf32>
    %v6295 = stablehlo.multiply %v2499, %v2499 : tensor<256x128x1x1xf32>
    %v6296 = stablehlo.multiply %v6293, %v6295 : tensor<256x128x1x1xf32>
    %v6297 = stablehlo.add %v6294, %v6296 : tensor<256x128x1x1xf32>
    %v6298 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6299 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6300 = stablehlo.multiply %v6298, %d3Wpm : tensor<256x128x1x1xf32>
    %v6301 = stablehlo.multiply %v6299, %v2499 : tensor<256x128x1x1xf32>
    %v6302 = stablehlo.add %v6300, %v6301 : tensor<256x128x1x1xf32>
    %v6303 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6304 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6305 = stablehlo.multiply %v6303, %d3Wpv : tensor<256x128x1x1xf32>
    %v6306 = stablehlo.multiply %v2499, %v2499 : tensor<256x128x1x1xf32>
    %v6307 = stablehlo.multiply %v6304, %v6306 : tensor<256x128x1x1xf32>
    %v6308 = stablehlo.add %v6305, %v6307 : tensor<256x128x1x1xf32>
    %v6309 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6310 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6311 = stablehlo.divide %v6302, %v6309 : tensor<256x128x1x1xf32>
    %v6312 = stablehlo.divide %v6308, %v6310 : tensor<256x128x1x1xf32>
    %v6313 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6314 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6315 = stablehlo.sqrt %v6312 : tensor<256x128x1x1xf32>
    %v6316 = stablehlo.add %v6315, %v6314 : tensor<256x128x1x1xf32>
    %v6317 = stablehlo.divide %v6311, %v6316 : tensor<256x128x1x1xf32>
    %v6318 = stablehlo.multiply %v6313, %v6317 : tensor<256x128x1x1xf32>
    %v6319 = stablehlo.subtract %d3Wp, %v6318 : tensor<256x128x1x1xf32>
    %v6320 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6321 = stablehlo.multiply %v6320, %v6313 : tensor<256x128x1x1xf32>
    %v6322 = stablehlo.multiply %v6321, %d3Wp : tensor<256x128x1x1xf32>
    %v6323 = stablehlo.subtract %v6319, %v6322 : tensor<256x128x1x1xf32>
    %v6324 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6325 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6326 = stablehlo.multiply %v6324, %d3gpm : tensor<256xf32>
    %v6327 = stablehlo.multiply %v6325, %v2517 : tensor<256xf32>
    %v6328 = stablehlo.add %v6326, %v6327 : tensor<256xf32>
    %v6329 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6330 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6331 = stablehlo.multiply %v6329, %d3gpv : tensor<256xf32>
    %v6332 = stablehlo.multiply %v2517, %v2517 : tensor<256xf32>
    %v6333 = stablehlo.multiply %v6330, %v6332 : tensor<256xf32>
    %v6334 = stablehlo.add %v6331, %v6333 : tensor<256xf32>
    %v6335 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6336 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6337 = stablehlo.multiply %v6335, %d3gpm : tensor<256xf32>
    %v6338 = stablehlo.multiply %v6336, %v2517 : tensor<256xf32>
    %v6339 = stablehlo.add %v6337, %v6338 : tensor<256xf32>
    %v6340 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6341 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6342 = stablehlo.multiply %v6340, %d3gpv : tensor<256xf32>
    %v6343 = stablehlo.multiply %v2517, %v2517 : tensor<256xf32>
    %v6344 = stablehlo.multiply %v6341, %v6343 : tensor<256xf32>
    %v6345 = stablehlo.add %v6342, %v6344 : tensor<256xf32>
    %v6346 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6347 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6348 = stablehlo.divide %v6339, %v6346 : tensor<256xf32>
    %v6349 = stablehlo.divide %v6345, %v6347 : tensor<256xf32>
    %v6350 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6351 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6352 = stablehlo.sqrt %v6349 : tensor<256xf32>
    %v6353 = stablehlo.add %v6352, %v6351 : tensor<256xf32>
    %v6354 = stablehlo.divide %v6348, %v6353 : tensor<256xf32>
    %v6355 = stablehlo.multiply %v6350, %v6354 : tensor<256xf32>
    %v6356 = stablehlo.subtract %d3gp, %v6355 : tensor<256xf32>
    %v6357 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6358 = stablehlo.multiply %v6357, %v6350 : tensor<256xf32>
    %v6359 = stablehlo.multiply %v6358, %d3gp : tensor<256xf32>
    %v6360 = stablehlo.subtract %v6356, %v6359 : tensor<256xf32>
    %v6361 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6362 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6363 = stablehlo.multiply %v6361, %d3btpm : tensor<256xf32>
    %v6364 = stablehlo.multiply %v6362, %v2520 : tensor<256xf32>
    %v6365 = stablehlo.add %v6363, %v6364 : tensor<256xf32>
    %v6366 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6367 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6368 = stablehlo.multiply %v6366, %d3btpv : tensor<256xf32>
    %v6369 = stablehlo.multiply %v2520, %v2520 : tensor<256xf32>
    %v6370 = stablehlo.multiply %v6367, %v6369 : tensor<256xf32>
    %v6371 = stablehlo.add %v6368, %v6370 : tensor<256xf32>
    %v6372 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6373 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6374 = stablehlo.multiply %v6372, %d3btpm : tensor<256xf32>
    %v6375 = stablehlo.multiply %v6373, %v2520 : tensor<256xf32>
    %v6376 = stablehlo.add %v6374, %v6375 : tensor<256xf32>
    %v6377 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6378 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6379 = stablehlo.multiply %v6377, %d3btpv : tensor<256xf32>
    %v6380 = stablehlo.multiply %v2520, %v2520 : tensor<256xf32>
    %v6381 = stablehlo.multiply %v6378, %v6380 : tensor<256xf32>
    %v6382 = stablehlo.add %v6379, %v6381 : tensor<256xf32>
    %v6383 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6384 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6385 = stablehlo.divide %v6376, %v6383 : tensor<256xf32>
    %v6386 = stablehlo.divide %v6382, %v6384 : tensor<256xf32>
    %v6387 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6388 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6389 = stablehlo.sqrt %v6386 : tensor<256xf32>
    %v6390 = stablehlo.add %v6389, %v6388 : tensor<256xf32>
    %v6391 = stablehlo.divide %v6385, %v6390 : tensor<256xf32>
    %v6392 = stablehlo.multiply %v6387, %v6391 : tensor<256xf32>
    %v6393 = stablehlo.subtract %d3btp, %v6392 : tensor<256xf32>
    %v6394 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6395 = stablehlo.multiply %v6394, %v6387 : tensor<256xf32>
    %v6396 = stablehlo.multiply %v6395, %d3btp : tensor<256xf32>
    %v6397 = stablehlo.subtract %v6393, %v6396 : tensor<256xf32>
    %v6398 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6399 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6400 = stablehlo.multiply %v6398, %s3b0W1m : tensor<256x256x3x3xf32>
    %v6401 = stablehlo.multiply %v6399, %v2268 : tensor<256x256x3x3xf32>
    %v6402 = stablehlo.add %v6400, %v6401 : tensor<256x256x3x3xf32>
    %v6403 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6404 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6405 = stablehlo.multiply %v6403, %s3b0W1v : tensor<256x256x3x3xf32>
    %v6406 = stablehlo.multiply %v2268, %v2268 : tensor<256x256x3x3xf32>
    %v6407 = stablehlo.multiply %v6404, %v6406 : tensor<256x256x3x3xf32>
    %v6408 = stablehlo.add %v6405, %v6407 : tensor<256x256x3x3xf32>
    %v6409 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6410 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6411 = stablehlo.multiply %v6409, %s3b0W1m : tensor<256x256x3x3xf32>
    %v6412 = stablehlo.multiply %v6410, %v2268 : tensor<256x256x3x3xf32>
    %v6413 = stablehlo.add %v6411, %v6412 : tensor<256x256x3x3xf32>
    %v6414 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6415 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6416 = stablehlo.multiply %v6414, %s3b0W1v : tensor<256x256x3x3xf32>
    %v6417 = stablehlo.multiply %v2268, %v2268 : tensor<256x256x3x3xf32>
    %v6418 = stablehlo.multiply %v6415, %v6417 : tensor<256x256x3x3xf32>
    %v6419 = stablehlo.add %v6416, %v6418 : tensor<256x256x3x3xf32>
    %v6420 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6421 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6422 = stablehlo.divide %v6413, %v6420 : tensor<256x256x3x3xf32>
    %v6423 = stablehlo.divide %v6419, %v6421 : tensor<256x256x3x3xf32>
    %v6424 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6425 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6426 = stablehlo.sqrt %v6423 : tensor<256x256x3x3xf32>
    %v6427 = stablehlo.add %v6426, %v6425 : tensor<256x256x3x3xf32>
    %v6428 = stablehlo.divide %v6422, %v6427 : tensor<256x256x3x3xf32>
    %v6429 = stablehlo.multiply %v6424, %v6428 : tensor<256x256x3x3xf32>
    %v6430 = stablehlo.subtract %s3b0W1, %v6429 : tensor<256x256x3x3xf32>
    %v6431 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6432 = stablehlo.multiply %v6431, %v6424 : tensor<256x256x3x3xf32>
    %v6433 = stablehlo.multiply %v6432, %s3b0W1 : tensor<256x256x3x3xf32>
    %v6434 = stablehlo.subtract %v6430, %v6433 : tensor<256x256x3x3xf32>
    %v6435 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6436 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6437 = stablehlo.multiply %v6435, %s3b0g1m : tensor<256xf32>
    %v6438 = stablehlo.multiply %v6436, %v2286 : tensor<256xf32>
    %v6439 = stablehlo.add %v6437, %v6438 : tensor<256xf32>
    %v6440 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6441 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6442 = stablehlo.multiply %v6440, %s3b0g1v : tensor<256xf32>
    %v6443 = stablehlo.multiply %v2286, %v2286 : tensor<256xf32>
    %v6444 = stablehlo.multiply %v6441, %v6443 : tensor<256xf32>
    %v6445 = stablehlo.add %v6442, %v6444 : tensor<256xf32>
    %v6446 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6447 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6448 = stablehlo.multiply %v6446, %s3b0g1m : tensor<256xf32>
    %v6449 = stablehlo.multiply %v6447, %v2286 : tensor<256xf32>
    %v6450 = stablehlo.add %v6448, %v6449 : tensor<256xf32>
    %v6451 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6452 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6453 = stablehlo.multiply %v6451, %s3b0g1v : tensor<256xf32>
    %v6454 = stablehlo.multiply %v2286, %v2286 : tensor<256xf32>
    %v6455 = stablehlo.multiply %v6452, %v6454 : tensor<256xf32>
    %v6456 = stablehlo.add %v6453, %v6455 : tensor<256xf32>
    %v6457 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6458 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6459 = stablehlo.divide %v6450, %v6457 : tensor<256xf32>
    %v6460 = stablehlo.divide %v6456, %v6458 : tensor<256xf32>
    %v6461 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6462 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6463 = stablehlo.sqrt %v6460 : tensor<256xf32>
    %v6464 = stablehlo.add %v6463, %v6462 : tensor<256xf32>
    %v6465 = stablehlo.divide %v6459, %v6464 : tensor<256xf32>
    %v6466 = stablehlo.multiply %v6461, %v6465 : tensor<256xf32>
    %v6467 = stablehlo.subtract %s3b0g1, %v6466 : tensor<256xf32>
    %v6468 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6469 = stablehlo.multiply %v6468, %v6461 : tensor<256xf32>
    %v6470 = stablehlo.multiply %v6469, %s3b0g1 : tensor<256xf32>
    %v6471 = stablehlo.subtract %v6467, %v6470 : tensor<256xf32>
    %v6472 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6473 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6474 = stablehlo.multiply %v6472, %s3b0bt1m : tensor<256xf32>
    %v6475 = stablehlo.multiply %v6473, %v2289 : tensor<256xf32>
    %v6476 = stablehlo.add %v6474, %v6475 : tensor<256xf32>
    %v6477 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6478 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6479 = stablehlo.multiply %v6477, %s3b0bt1v : tensor<256xf32>
    %v6480 = stablehlo.multiply %v2289, %v2289 : tensor<256xf32>
    %v6481 = stablehlo.multiply %v6478, %v6480 : tensor<256xf32>
    %v6482 = stablehlo.add %v6479, %v6481 : tensor<256xf32>
    %v6483 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6484 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6485 = stablehlo.multiply %v6483, %s3b0bt1m : tensor<256xf32>
    %v6486 = stablehlo.multiply %v6484, %v2289 : tensor<256xf32>
    %v6487 = stablehlo.add %v6485, %v6486 : tensor<256xf32>
    %v6488 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6489 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6490 = stablehlo.multiply %v6488, %s3b0bt1v : tensor<256xf32>
    %v6491 = stablehlo.multiply %v2289, %v2289 : tensor<256xf32>
    %v6492 = stablehlo.multiply %v6489, %v6491 : tensor<256xf32>
    %v6493 = stablehlo.add %v6490, %v6492 : tensor<256xf32>
    %v6494 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6495 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6496 = stablehlo.divide %v6487, %v6494 : tensor<256xf32>
    %v6497 = stablehlo.divide %v6493, %v6495 : tensor<256xf32>
    %v6498 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6499 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6500 = stablehlo.sqrt %v6497 : tensor<256xf32>
    %v6501 = stablehlo.add %v6500, %v6499 : tensor<256xf32>
    %v6502 = stablehlo.divide %v6496, %v6501 : tensor<256xf32>
    %v6503 = stablehlo.multiply %v6498, %v6502 : tensor<256xf32>
    %v6504 = stablehlo.subtract %s3b0bt1, %v6503 : tensor<256xf32>
    %v6505 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6506 = stablehlo.multiply %v6505, %v6498 : tensor<256xf32>
    %v6507 = stablehlo.multiply %v6506, %s3b0bt1 : tensor<256xf32>
    %v6508 = stablehlo.subtract %v6504, %v6507 : tensor<256xf32>
    %v6509 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6510 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6511 = stablehlo.multiply %v6509, %s3b0W2m : tensor<256x256x3x3xf32>
    %v6512 = stablehlo.multiply %v6510, %v2295 : tensor<256x256x3x3xf32>
    %v6513 = stablehlo.add %v6511, %v6512 : tensor<256x256x3x3xf32>
    %v6514 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6515 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6516 = stablehlo.multiply %v6514, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6517 = stablehlo.multiply %v2295, %v2295 : tensor<256x256x3x3xf32>
    %v6518 = stablehlo.multiply %v6515, %v6517 : tensor<256x256x3x3xf32>
    %v6519 = stablehlo.add %v6516, %v6518 : tensor<256x256x3x3xf32>
    %v6520 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6521 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6522 = stablehlo.multiply %v6520, %s3b0W2m : tensor<256x256x3x3xf32>
    %v6523 = stablehlo.multiply %v6521, %v2295 : tensor<256x256x3x3xf32>
    %v6524 = stablehlo.add %v6522, %v6523 : tensor<256x256x3x3xf32>
    %v6525 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6526 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6527 = stablehlo.multiply %v6525, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6528 = stablehlo.multiply %v2295, %v2295 : tensor<256x256x3x3xf32>
    %v6529 = stablehlo.multiply %v6526, %v6528 : tensor<256x256x3x3xf32>
    %v6530 = stablehlo.add %v6527, %v6529 : tensor<256x256x3x3xf32>
    %v6531 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6532 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6533 = stablehlo.divide %v6524, %v6531 : tensor<256x256x3x3xf32>
    %v6534 = stablehlo.divide %v6530, %v6532 : tensor<256x256x3x3xf32>
    %v6535 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6536 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6537 = stablehlo.sqrt %v6534 : tensor<256x256x3x3xf32>
    %v6538 = stablehlo.add %v6537, %v6536 : tensor<256x256x3x3xf32>
    %v6539 = stablehlo.divide %v6533, %v6538 : tensor<256x256x3x3xf32>
    %v6540 = stablehlo.multiply %v6535, %v6539 : tensor<256x256x3x3xf32>
    %v6541 = stablehlo.subtract %s3b0W2, %v6540 : tensor<256x256x3x3xf32>
    %v6542 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6543 = stablehlo.multiply %v6542, %v6535 : tensor<256x256x3x3xf32>
    %v6544 = stablehlo.multiply %v6543, %s3b0W2 : tensor<256x256x3x3xf32>
    %v6545 = stablehlo.subtract %v6541, %v6544 : tensor<256x256x3x3xf32>
    %v6546 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6547 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6548 = stablehlo.multiply %v6546, %s3b0g2m : tensor<256xf32>
    %v6549 = stablehlo.multiply %v6547, %v2313 : tensor<256xf32>
    %v6550 = stablehlo.add %v6548, %v6549 : tensor<256xf32>
    %v6551 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6552 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6553 = stablehlo.multiply %v6551, %s3b0g2v : tensor<256xf32>
    %v6554 = stablehlo.multiply %v2313, %v2313 : tensor<256xf32>
    %v6555 = stablehlo.multiply %v6552, %v6554 : tensor<256xf32>
    %v6556 = stablehlo.add %v6553, %v6555 : tensor<256xf32>
    %v6557 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6558 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6559 = stablehlo.multiply %v6557, %s3b0g2m : tensor<256xf32>
    %v6560 = stablehlo.multiply %v6558, %v2313 : tensor<256xf32>
    %v6561 = stablehlo.add %v6559, %v6560 : tensor<256xf32>
    %v6562 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6563 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6564 = stablehlo.multiply %v6562, %s3b0g2v : tensor<256xf32>
    %v6565 = stablehlo.multiply %v2313, %v2313 : tensor<256xf32>
    %v6566 = stablehlo.multiply %v6563, %v6565 : tensor<256xf32>
    %v6567 = stablehlo.add %v6564, %v6566 : tensor<256xf32>
    %v6568 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6569 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6570 = stablehlo.divide %v6561, %v6568 : tensor<256xf32>
    %v6571 = stablehlo.divide %v6567, %v6569 : tensor<256xf32>
    %v6572 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6573 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6574 = stablehlo.sqrt %v6571 : tensor<256xf32>
    %v6575 = stablehlo.add %v6574, %v6573 : tensor<256xf32>
    %v6576 = stablehlo.divide %v6570, %v6575 : tensor<256xf32>
    %v6577 = stablehlo.multiply %v6572, %v6576 : tensor<256xf32>
    %v6578 = stablehlo.subtract %s3b0g2, %v6577 : tensor<256xf32>
    %v6579 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6580 = stablehlo.multiply %v6579, %v6572 : tensor<256xf32>
    %v6581 = stablehlo.multiply %v6580, %s3b0g2 : tensor<256xf32>
    %v6582 = stablehlo.subtract %v6578, %v6581 : tensor<256xf32>
    %v6583 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6584 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6585 = stablehlo.multiply %v6583, %s3b0bt2m : tensor<256xf32>
    %v6586 = stablehlo.multiply %v6584, %v2316 : tensor<256xf32>
    %v6587 = stablehlo.add %v6585, %v6586 : tensor<256xf32>
    %v6588 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6589 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6590 = stablehlo.multiply %v6588, %s3b0bt2v : tensor<256xf32>
    %v6591 = stablehlo.multiply %v2316, %v2316 : tensor<256xf32>
    %v6592 = stablehlo.multiply %v6589, %v6591 : tensor<256xf32>
    %v6593 = stablehlo.add %v6590, %v6592 : tensor<256xf32>
    %v6594 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6595 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6596 = stablehlo.multiply %v6594, %s3b0bt2m : tensor<256xf32>
    %v6597 = stablehlo.multiply %v6595, %v2316 : tensor<256xf32>
    %v6598 = stablehlo.add %v6596, %v6597 : tensor<256xf32>
    %v6599 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6600 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6601 = stablehlo.multiply %v6599, %s3b0bt2v : tensor<256xf32>
    %v6602 = stablehlo.multiply %v2316, %v2316 : tensor<256xf32>
    %v6603 = stablehlo.multiply %v6600, %v6602 : tensor<256xf32>
    %v6604 = stablehlo.add %v6601, %v6603 : tensor<256xf32>
    %v6605 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6606 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6607 = stablehlo.divide %v6598, %v6605 : tensor<256xf32>
    %v6608 = stablehlo.divide %v6604, %v6606 : tensor<256xf32>
    %v6609 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6610 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6611 = stablehlo.sqrt %v6608 : tensor<256xf32>
    %v6612 = stablehlo.add %v6611, %v6610 : tensor<256xf32>
    %v6613 = stablehlo.divide %v6607, %v6612 : tensor<256xf32>
    %v6614 = stablehlo.multiply %v6609, %v6613 : tensor<256xf32>
    %v6615 = stablehlo.subtract %s3b0bt2, %v6614 : tensor<256xf32>
    %v6616 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6617 = stablehlo.multiply %v6616, %v6609 : tensor<256xf32>
    %v6618 = stablehlo.multiply %v6617, %s3b0bt2 : tensor<256xf32>
    %v6619 = stablehlo.subtract %v6615, %v6618 : tensor<256xf32>
    %v6620 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6621 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6622 = stablehlo.multiply %v6620, %s3b1W1m : tensor<256x256x3x3xf32>
    %v6623 = stablehlo.multiply %v6621, %v2128 : tensor<256x256x3x3xf32>
    %v6624 = stablehlo.add %v6622, %v6623 : tensor<256x256x3x3xf32>
    %v6625 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6626 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6627 = stablehlo.multiply %v6625, %s3b1W1v : tensor<256x256x3x3xf32>
    %v6628 = stablehlo.multiply %v2128, %v2128 : tensor<256x256x3x3xf32>
    %v6629 = stablehlo.multiply %v6626, %v6628 : tensor<256x256x3x3xf32>
    %v6630 = stablehlo.add %v6627, %v6629 : tensor<256x256x3x3xf32>
    %v6631 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6632 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6633 = stablehlo.multiply %v6631, %s3b1W1m : tensor<256x256x3x3xf32>
    %v6634 = stablehlo.multiply %v6632, %v2128 : tensor<256x256x3x3xf32>
    %v6635 = stablehlo.add %v6633, %v6634 : tensor<256x256x3x3xf32>
    %v6636 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6637 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6638 = stablehlo.multiply %v6636, %s3b1W1v : tensor<256x256x3x3xf32>
    %v6639 = stablehlo.multiply %v2128, %v2128 : tensor<256x256x3x3xf32>
    %v6640 = stablehlo.multiply %v6637, %v6639 : tensor<256x256x3x3xf32>
    %v6641 = stablehlo.add %v6638, %v6640 : tensor<256x256x3x3xf32>
    %v6642 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6643 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6644 = stablehlo.divide %v6635, %v6642 : tensor<256x256x3x3xf32>
    %v6645 = stablehlo.divide %v6641, %v6643 : tensor<256x256x3x3xf32>
    %v6646 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6647 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6648 = stablehlo.sqrt %v6645 : tensor<256x256x3x3xf32>
    %v6649 = stablehlo.add %v6648, %v6647 : tensor<256x256x3x3xf32>
    %v6650 = stablehlo.divide %v6644, %v6649 : tensor<256x256x3x3xf32>
    %v6651 = stablehlo.multiply %v6646, %v6650 : tensor<256x256x3x3xf32>
    %v6652 = stablehlo.subtract %s3b1W1, %v6651 : tensor<256x256x3x3xf32>
    %v6653 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6654 = stablehlo.multiply %v6653, %v6646 : tensor<256x256x3x3xf32>
    %v6655 = stablehlo.multiply %v6654, %s3b1W1 : tensor<256x256x3x3xf32>
    %v6656 = stablehlo.subtract %v6652, %v6655 : tensor<256x256x3x3xf32>
    %v6657 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6658 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6659 = stablehlo.multiply %v6657, %s3b1g1m : tensor<256xf32>
    %v6660 = stablehlo.multiply %v6658, %v2146 : tensor<256xf32>
    %v6661 = stablehlo.add %v6659, %v6660 : tensor<256xf32>
    %v6662 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6663 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6664 = stablehlo.multiply %v6662, %s3b1g1v : tensor<256xf32>
    %v6665 = stablehlo.multiply %v2146, %v2146 : tensor<256xf32>
    %v6666 = stablehlo.multiply %v6663, %v6665 : tensor<256xf32>
    %v6667 = stablehlo.add %v6664, %v6666 : tensor<256xf32>
    %v6668 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6669 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6670 = stablehlo.multiply %v6668, %s3b1g1m : tensor<256xf32>
    %v6671 = stablehlo.multiply %v6669, %v2146 : tensor<256xf32>
    %v6672 = stablehlo.add %v6670, %v6671 : tensor<256xf32>
    %v6673 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6674 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6675 = stablehlo.multiply %v6673, %s3b1g1v : tensor<256xf32>
    %v6676 = stablehlo.multiply %v2146, %v2146 : tensor<256xf32>
    %v6677 = stablehlo.multiply %v6674, %v6676 : tensor<256xf32>
    %v6678 = stablehlo.add %v6675, %v6677 : tensor<256xf32>
    %v6679 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6680 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6681 = stablehlo.divide %v6672, %v6679 : tensor<256xf32>
    %v6682 = stablehlo.divide %v6678, %v6680 : tensor<256xf32>
    %v6683 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6684 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6685 = stablehlo.sqrt %v6682 : tensor<256xf32>
    %v6686 = stablehlo.add %v6685, %v6684 : tensor<256xf32>
    %v6687 = stablehlo.divide %v6681, %v6686 : tensor<256xf32>
    %v6688 = stablehlo.multiply %v6683, %v6687 : tensor<256xf32>
    %v6689 = stablehlo.subtract %s3b1g1, %v6688 : tensor<256xf32>
    %v6690 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6691 = stablehlo.multiply %v6690, %v6683 : tensor<256xf32>
    %v6692 = stablehlo.multiply %v6691, %s3b1g1 : tensor<256xf32>
    %v6693 = stablehlo.subtract %v6689, %v6692 : tensor<256xf32>
    %v6694 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6695 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6696 = stablehlo.multiply %v6694, %s3b1bt1m : tensor<256xf32>
    %v6697 = stablehlo.multiply %v6695, %v2149 : tensor<256xf32>
    %v6698 = stablehlo.add %v6696, %v6697 : tensor<256xf32>
    %v6699 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6700 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6701 = stablehlo.multiply %v6699, %s3b1bt1v : tensor<256xf32>
    %v6702 = stablehlo.multiply %v2149, %v2149 : tensor<256xf32>
    %v6703 = stablehlo.multiply %v6700, %v6702 : tensor<256xf32>
    %v6704 = stablehlo.add %v6701, %v6703 : tensor<256xf32>
    %v6705 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6706 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6707 = stablehlo.multiply %v6705, %s3b1bt1m : tensor<256xf32>
    %v6708 = stablehlo.multiply %v6706, %v2149 : tensor<256xf32>
    %v6709 = stablehlo.add %v6707, %v6708 : tensor<256xf32>
    %v6710 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6711 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6712 = stablehlo.multiply %v6710, %s3b1bt1v : tensor<256xf32>
    %v6713 = stablehlo.multiply %v2149, %v2149 : tensor<256xf32>
    %v6714 = stablehlo.multiply %v6711, %v6713 : tensor<256xf32>
    %v6715 = stablehlo.add %v6712, %v6714 : tensor<256xf32>
    %v6716 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6717 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6718 = stablehlo.divide %v6709, %v6716 : tensor<256xf32>
    %v6719 = stablehlo.divide %v6715, %v6717 : tensor<256xf32>
    %v6720 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6721 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6722 = stablehlo.sqrt %v6719 : tensor<256xf32>
    %v6723 = stablehlo.add %v6722, %v6721 : tensor<256xf32>
    %v6724 = stablehlo.divide %v6718, %v6723 : tensor<256xf32>
    %v6725 = stablehlo.multiply %v6720, %v6724 : tensor<256xf32>
    %v6726 = stablehlo.subtract %s3b1bt1, %v6725 : tensor<256xf32>
    %v6727 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6728 = stablehlo.multiply %v6727, %v6720 : tensor<256xf32>
    %v6729 = stablehlo.multiply %v6728, %s3b1bt1 : tensor<256xf32>
    %v6730 = stablehlo.subtract %v6726, %v6729 : tensor<256xf32>
    %v6731 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6732 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6733 = stablehlo.multiply %v6731, %s3b1W2m : tensor<256x256x3x3xf32>
    %v6734 = stablehlo.multiply %v6732, %v2155 : tensor<256x256x3x3xf32>
    %v6735 = stablehlo.add %v6733, %v6734 : tensor<256x256x3x3xf32>
    %v6736 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6737 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6738 = stablehlo.multiply %v6736, %s3b1W2v : tensor<256x256x3x3xf32>
    %v6739 = stablehlo.multiply %v2155, %v2155 : tensor<256x256x3x3xf32>
    %v6740 = stablehlo.multiply %v6737, %v6739 : tensor<256x256x3x3xf32>
    %v6741 = stablehlo.add %v6738, %v6740 : tensor<256x256x3x3xf32>
    %v6742 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6743 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6744 = stablehlo.multiply %v6742, %s3b1W2m : tensor<256x256x3x3xf32>
    %v6745 = stablehlo.multiply %v6743, %v2155 : tensor<256x256x3x3xf32>
    %v6746 = stablehlo.add %v6744, %v6745 : tensor<256x256x3x3xf32>
    %v6747 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6748 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6749 = stablehlo.multiply %v6747, %s3b1W2v : tensor<256x256x3x3xf32>
    %v6750 = stablehlo.multiply %v2155, %v2155 : tensor<256x256x3x3xf32>
    %v6751 = stablehlo.multiply %v6748, %v6750 : tensor<256x256x3x3xf32>
    %v6752 = stablehlo.add %v6749, %v6751 : tensor<256x256x3x3xf32>
    %v6753 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6754 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6755 = stablehlo.divide %v6746, %v6753 : tensor<256x256x3x3xf32>
    %v6756 = stablehlo.divide %v6752, %v6754 : tensor<256x256x3x3xf32>
    %v6757 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6758 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6759 = stablehlo.sqrt %v6756 : tensor<256x256x3x3xf32>
    %v6760 = stablehlo.add %v6759, %v6758 : tensor<256x256x3x3xf32>
    %v6761 = stablehlo.divide %v6755, %v6760 : tensor<256x256x3x3xf32>
    %v6762 = stablehlo.multiply %v6757, %v6761 : tensor<256x256x3x3xf32>
    %v6763 = stablehlo.subtract %s3b1W2, %v6762 : tensor<256x256x3x3xf32>
    %v6764 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6765 = stablehlo.multiply %v6764, %v6757 : tensor<256x256x3x3xf32>
    %v6766 = stablehlo.multiply %v6765, %s3b1W2 : tensor<256x256x3x3xf32>
    %v6767 = stablehlo.subtract %v6763, %v6766 : tensor<256x256x3x3xf32>
    %v6768 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6769 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6770 = stablehlo.multiply %v6768, %s3b1g2m : tensor<256xf32>
    %v6771 = stablehlo.multiply %v6769, %v2173 : tensor<256xf32>
    %v6772 = stablehlo.add %v6770, %v6771 : tensor<256xf32>
    %v6773 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6774 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6775 = stablehlo.multiply %v6773, %s3b1g2v : tensor<256xf32>
    %v6776 = stablehlo.multiply %v2173, %v2173 : tensor<256xf32>
    %v6777 = stablehlo.multiply %v6774, %v6776 : tensor<256xf32>
    %v6778 = stablehlo.add %v6775, %v6777 : tensor<256xf32>
    %v6779 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6780 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6781 = stablehlo.multiply %v6779, %s3b1g2m : tensor<256xf32>
    %v6782 = stablehlo.multiply %v6780, %v2173 : tensor<256xf32>
    %v6783 = stablehlo.add %v6781, %v6782 : tensor<256xf32>
    %v6784 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6785 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6786 = stablehlo.multiply %v6784, %s3b1g2v : tensor<256xf32>
    %v6787 = stablehlo.multiply %v2173, %v2173 : tensor<256xf32>
    %v6788 = stablehlo.multiply %v6785, %v6787 : tensor<256xf32>
    %v6789 = stablehlo.add %v6786, %v6788 : tensor<256xf32>
    %v6790 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6791 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6792 = stablehlo.divide %v6783, %v6790 : tensor<256xf32>
    %v6793 = stablehlo.divide %v6789, %v6791 : tensor<256xf32>
    %v6794 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6795 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6796 = stablehlo.sqrt %v6793 : tensor<256xf32>
    %v6797 = stablehlo.add %v6796, %v6795 : tensor<256xf32>
    %v6798 = stablehlo.divide %v6792, %v6797 : tensor<256xf32>
    %v6799 = stablehlo.multiply %v6794, %v6798 : tensor<256xf32>
    %v6800 = stablehlo.subtract %s3b1g2, %v6799 : tensor<256xf32>
    %v6801 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6802 = stablehlo.multiply %v6801, %v6794 : tensor<256xf32>
    %v6803 = stablehlo.multiply %v6802, %s3b1g2 : tensor<256xf32>
    %v6804 = stablehlo.subtract %v6800, %v6803 : tensor<256xf32>
    %v6805 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6806 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6807 = stablehlo.multiply %v6805, %s3b1bt2m : tensor<256xf32>
    %v6808 = stablehlo.multiply %v6806, %v2176 : tensor<256xf32>
    %v6809 = stablehlo.add %v6807, %v6808 : tensor<256xf32>
    %v6810 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6811 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6812 = stablehlo.multiply %v6810, %s3b1bt2v : tensor<256xf32>
    %v6813 = stablehlo.multiply %v2176, %v2176 : tensor<256xf32>
    %v6814 = stablehlo.multiply %v6811, %v6813 : tensor<256xf32>
    %v6815 = stablehlo.add %v6812, %v6814 : tensor<256xf32>
    %v6816 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6817 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6818 = stablehlo.multiply %v6816, %s3b1bt2m : tensor<256xf32>
    %v6819 = stablehlo.multiply %v6817, %v2176 : tensor<256xf32>
    %v6820 = stablehlo.add %v6818, %v6819 : tensor<256xf32>
    %v6821 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6822 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6823 = stablehlo.multiply %v6821, %s3b1bt2v : tensor<256xf32>
    %v6824 = stablehlo.multiply %v2176, %v2176 : tensor<256xf32>
    %v6825 = stablehlo.multiply %v6822, %v6824 : tensor<256xf32>
    %v6826 = stablehlo.add %v6823, %v6825 : tensor<256xf32>
    %v6827 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6828 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6829 = stablehlo.divide %v6820, %v6827 : tensor<256xf32>
    %v6830 = stablehlo.divide %v6826, %v6828 : tensor<256xf32>
    %v6831 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6832 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6833 = stablehlo.sqrt %v6830 : tensor<256xf32>
    %v6834 = stablehlo.add %v6833, %v6832 : tensor<256xf32>
    %v6835 = stablehlo.divide %v6829, %v6834 : tensor<256xf32>
    %v6836 = stablehlo.multiply %v6831, %v6835 : tensor<256xf32>
    %v6837 = stablehlo.subtract %s3b1bt2, %v6836 : tensor<256xf32>
    %v6838 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6839 = stablehlo.multiply %v6838, %v6831 : tensor<256xf32>
    %v6840 = stablehlo.multiply %v6839, %s3b1bt2 : tensor<256xf32>
    %v6841 = stablehlo.subtract %v6837, %v6840 : tensor<256xf32>
    %v6842 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6843 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6844 = stablehlo.multiply %v6842, %s3b2W1m : tensor<256x256x3x3xf32>
    %v6845 = stablehlo.multiply %v6843, %v1988 : tensor<256x256x3x3xf32>
    %v6846 = stablehlo.add %v6844, %v6845 : tensor<256x256x3x3xf32>
    %v6847 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6848 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6849 = stablehlo.multiply %v6847, %s3b2W1v : tensor<256x256x3x3xf32>
    %v6850 = stablehlo.multiply %v1988, %v1988 : tensor<256x256x3x3xf32>
    %v6851 = stablehlo.multiply %v6848, %v6850 : tensor<256x256x3x3xf32>
    %v6852 = stablehlo.add %v6849, %v6851 : tensor<256x256x3x3xf32>
    %v6853 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6854 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6855 = stablehlo.multiply %v6853, %s3b2W1m : tensor<256x256x3x3xf32>
    %v6856 = stablehlo.multiply %v6854, %v1988 : tensor<256x256x3x3xf32>
    %v6857 = stablehlo.add %v6855, %v6856 : tensor<256x256x3x3xf32>
    %v6858 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6859 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6860 = stablehlo.multiply %v6858, %s3b2W1v : tensor<256x256x3x3xf32>
    %v6861 = stablehlo.multiply %v1988, %v1988 : tensor<256x256x3x3xf32>
    %v6862 = stablehlo.multiply %v6859, %v6861 : tensor<256x256x3x3xf32>
    %v6863 = stablehlo.add %v6860, %v6862 : tensor<256x256x3x3xf32>
    %v6864 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6865 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6866 = stablehlo.divide %v6857, %v6864 : tensor<256x256x3x3xf32>
    %v6867 = stablehlo.divide %v6863, %v6865 : tensor<256x256x3x3xf32>
    %v6868 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6869 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6870 = stablehlo.sqrt %v6867 : tensor<256x256x3x3xf32>
    %v6871 = stablehlo.add %v6870, %v6869 : tensor<256x256x3x3xf32>
    %v6872 = stablehlo.divide %v6866, %v6871 : tensor<256x256x3x3xf32>
    %v6873 = stablehlo.multiply %v6868, %v6872 : tensor<256x256x3x3xf32>
    %v6874 = stablehlo.subtract %s3b2W1, %v6873 : tensor<256x256x3x3xf32>
    %v6875 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6876 = stablehlo.multiply %v6875, %v6868 : tensor<256x256x3x3xf32>
    %v6877 = stablehlo.multiply %v6876, %s3b2W1 : tensor<256x256x3x3xf32>
    %v6878 = stablehlo.subtract %v6874, %v6877 : tensor<256x256x3x3xf32>
    %v6879 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6880 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6881 = stablehlo.multiply %v6879, %s3b2g1m : tensor<256xf32>
    %v6882 = stablehlo.multiply %v6880, %v2006 : tensor<256xf32>
    %v6883 = stablehlo.add %v6881, %v6882 : tensor<256xf32>
    %v6884 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6885 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6886 = stablehlo.multiply %v6884, %s3b2g1v : tensor<256xf32>
    %v6887 = stablehlo.multiply %v2006, %v2006 : tensor<256xf32>
    %v6888 = stablehlo.multiply %v6885, %v6887 : tensor<256xf32>
    %v6889 = stablehlo.add %v6886, %v6888 : tensor<256xf32>
    %v6890 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6891 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6892 = stablehlo.multiply %v6890, %s3b2g1m : tensor<256xf32>
    %v6893 = stablehlo.multiply %v6891, %v2006 : tensor<256xf32>
    %v6894 = stablehlo.add %v6892, %v6893 : tensor<256xf32>
    %v6895 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6896 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6897 = stablehlo.multiply %v6895, %s3b2g1v : tensor<256xf32>
    %v6898 = stablehlo.multiply %v2006, %v2006 : tensor<256xf32>
    %v6899 = stablehlo.multiply %v6896, %v6898 : tensor<256xf32>
    %v6900 = stablehlo.add %v6897, %v6899 : tensor<256xf32>
    %v6901 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6902 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6903 = stablehlo.divide %v6894, %v6901 : tensor<256xf32>
    %v6904 = stablehlo.divide %v6900, %v6902 : tensor<256xf32>
    %v6905 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6906 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6907 = stablehlo.sqrt %v6904 : tensor<256xf32>
    %v6908 = stablehlo.add %v6907, %v6906 : tensor<256xf32>
    %v6909 = stablehlo.divide %v6903, %v6908 : tensor<256xf32>
    %v6910 = stablehlo.multiply %v6905, %v6909 : tensor<256xf32>
    %v6911 = stablehlo.subtract %s3b2g1, %v6910 : tensor<256xf32>
    %v6912 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6913 = stablehlo.multiply %v6912, %v6905 : tensor<256xf32>
    %v6914 = stablehlo.multiply %v6913, %s3b2g1 : tensor<256xf32>
    %v6915 = stablehlo.subtract %v6911, %v6914 : tensor<256xf32>
    %v6916 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6917 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6918 = stablehlo.multiply %v6916, %s3b2bt1m : tensor<256xf32>
    %v6919 = stablehlo.multiply %v6917, %v2009 : tensor<256xf32>
    %v6920 = stablehlo.add %v6918, %v6919 : tensor<256xf32>
    %v6921 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6922 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6923 = stablehlo.multiply %v6921, %s3b2bt1v : tensor<256xf32>
    %v6924 = stablehlo.multiply %v2009, %v2009 : tensor<256xf32>
    %v6925 = stablehlo.multiply %v6922, %v6924 : tensor<256xf32>
    %v6926 = stablehlo.add %v6923, %v6925 : tensor<256xf32>
    %v6927 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6928 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6929 = stablehlo.multiply %v6927, %s3b2bt1m : tensor<256xf32>
    %v6930 = stablehlo.multiply %v6928, %v2009 : tensor<256xf32>
    %v6931 = stablehlo.add %v6929, %v6930 : tensor<256xf32>
    %v6932 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6933 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6934 = stablehlo.multiply %v6932, %s3b2bt1v : tensor<256xf32>
    %v6935 = stablehlo.multiply %v2009, %v2009 : tensor<256xf32>
    %v6936 = stablehlo.multiply %v6933, %v6935 : tensor<256xf32>
    %v6937 = stablehlo.add %v6934, %v6936 : tensor<256xf32>
    %v6938 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6939 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6940 = stablehlo.divide %v6931, %v6938 : tensor<256xf32>
    %v6941 = stablehlo.divide %v6937, %v6939 : tensor<256xf32>
    %v6942 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6943 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6944 = stablehlo.sqrt %v6941 : tensor<256xf32>
    %v6945 = stablehlo.add %v6944, %v6943 : tensor<256xf32>
    %v6946 = stablehlo.divide %v6940, %v6945 : tensor<256xf32>
    %v6947 = stablehlo.multiply %v6942, %v6946 : tensor<256xf32>
    %v6948 = stablehlo.subtract %s3b2bt1, %v6947 : tensor<256xf32>
    %v6949 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6950 = stablehlo.multiply %v6949, %v6942 : tensor<256xf32>
    %v6951 = stablehlo.multiply %v6950, %s3b2bt1 : tensor<256xf32>
    %v6952 = stablehlo.subtract %v6948, %v6951 : tensor<256xf32>
    %v6953 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6954 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6955 = stablehlo.multiply %v6953, %s3b2W2m : tensor<256x256x3x3xf32>
    %v6956 = stablehlo.multiply %v6954, %v2015 : tensor<256x256x3x3xf32>
    %v6957 = stablehlo.add %v6955, %v6956 : tensor<256x256x3x3xf32>
    %v6958 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6959 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6960 = stablehlo.multiply %v6958, %s3b2W2v : tensor<256x256x3x3xf32>
    %v6961 = stablehlo.multiply %v2015, %v2015 : tensor<256x256x3x3xf32>
    %v6962 = stablehlo.multiply %v6959, %v6961 : tensor<256x256x3x3xf32>
    %v6963 = stablehlo.add %v6960, %v6962 : tensor<256x256x3x3xf32>
    %v6964 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6965 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6966 = stablehlo.multiply %v6964, %s3b2W2m : tensor<256x256x3x3xf32>
    %v6967 = stablehlo.multiply %v6965, %v2015 : tensor<256x256x3x3xf32>
    %v6968 = stablehlo.add %v6966, %v6967 : tensor<256x256x3x3xf32>
    %v6969 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6970 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6971 = stablehlo.multiply %v6969, %s3b2W2v : tensor<256x256x3x3xf32>
    %v6972 = stablehlo.multiply %v2015, %v2015 : tensor<256x256x3x3xf32>
    %v6973 = stablehlo.multiply %v6970, %v6972 : tensor<256x256x3x3xf32>
    %v6974 = stablehlo.add %v6971, %v6973 : tensor<256x256x3x3xf32>
    %v6975 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6976 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6977 = stablehlo.divide %v6968, %v6975 : tensor<256x256x3x3xf32>
    %v6978 = stablehlo.divide %v6974, %v6976 : tensor<256x256x3x3xf32>
    %v6979 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6980 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6981 = stablehlo.sqrt %v6978 : tensor<256x256x3x3xf32>
    %v6982 = stablehlo.add %v6981, %v6980 : tensor<256x256x3x3xf32>
    %v6983 = stablehlo.divide %v6977, %v6982 : tensor<256x256x3x3xf32>
    %v6984 = stablehlo.multiply %v6979, %v6983 : tensor<256x256x3x3xf32>
    %v6985 = stablehlo.subtract %s3b2W2, %v6984 : tensor<256x256x3x3xf32>
    %v6986 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6987 = stablehlo.multiply %v6986, %v6979 : tensor<256x256x3x3xf32>
    %v6988 = stablehlo.multiply %v6987, %s3b2W2 : tensor<256x256x3x3xf32>
    %v6989 = stablehlo.subtract %v6985, %v6988 : tensor<256x256x3x3xf32>
    %v6990 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6991 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6992 = stablehlo.multiply %v6990, %s3b2g2m : tensor<256xf32>
    %v6993 = stablehlo.multiply %v6991, %v2033 : tensor<256xf32>
    %v6994 = stablehlo.add %v6992, %v6993 : tensor<256xf32>
    %v6995 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6996 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6997 = stablehlo.multiply %v6995, %s3b2g2v : tensor<256xf32>
    %v6998 = stablehlo.multiply %v2033, %v2033 : tensor<256xf32>
    %v6999 = stablehlo.multiply %v6996, %v6998 : tensor<256xf32>
    %v7000 = stablehlo.add %v6997, %v6999 : tensor<256xf32>
    %v7001 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7002 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7003 = stablehlo.multiply %v7001, %s3b2g2m : tensor<256xf32>
    %v7004 = stablehlo.multiply %v7002, %v2033 : tensor<256xf32>
    %v7005 = stablehlo.add %v7003, %v7004 : tensor<256xf32>
    %v7006 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7007 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7008 = stablehlo.multiply %v7006, %s3b2g2v : tensor<256xf32>
    %v7009 = stablehlo.multiply %v2033, %v2033 : tensor<256xf32>
    %v7010 = stablehlo.multiply %v7007, %v7009 : tensor<256xf32>
    %v7011 = stablehlo.add %v7008, %v7010 : tensor<256xf32>
    %v7012 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7013 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7014 = stablehlo.divide %v7005, %v7012 : tensor<256xf32>
    %v7015 = stablehlo.divide %v7011, %v7013 : tensor<256xf32>
    %v7016 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7017 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7018 = stablehlo.sqrt %v7015 : tensor<256xf32>
    %v7019 = stablehlo.add %v7018, %v7017 : tensor<256xf32>
    %v7020 = stablehlo.divide %v7014, %v7019 : tensor<256xf32>
    %v7021 = stablehlo.multiply %v7016, %v7020 : tensor<256xf32>
    %v7022 = stablehlo.subtract %s3b2g2, %v7021 : tensor<256xf32>
    %v7023 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7024 = stablehlo.multiply %v7023, %v7016 : tensor<256xf32>
    %v7025 = stablehlo.multiply %v7024, %s3b2g2 : tensor<256xf32>
    %v7026 = stablehlo.subtract %v7022, %v7025 : tensor<256xf32>
    %v7027 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7028 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7029 = stablehlo.multiply %v7027, %s3b2bt2m : tensor<256xf32>
    %v7030 = stablehlo.multiply %v7028, %v2036 : tensor<256xf32>
    %v7031 = stablehlo.add %v7029, %v7030 : tensor<256xf32>
    %v7032 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7033 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7034 = stablehlo.multiply %v7032, %s3b2bt2v : tensor<256xf32>
    %v7035 = stablehlo.multiply %v2036, %v2036 : tensor<256xf32>
    %v7036 = stablehlo.multiply %v7033, %v7035 : tensor<256xf32>
    %v7037 = stablehlo.add %v7034, %v7036 : tensor<256xf32>
    %v7038 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7039 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7040 = stablehlo.multiply %v7038, %s3b2bt2m : tensor<256xf32>
    %v7041 = stablehlo.multiply %v7039, %v2036 : tensor<256xf32>
    %v7042 = stablehlo.add %v7040, %v7041 : tensor<256xf32>
    %v7043 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7044 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7045 = stablehlo.multiply %v7043, %s3b2bt2v : tensor<256xf32>
    %v7046 = stablehlo.multiply %v2036, %v2036 : tensor<256xf32>
    %v7047 = stablehlo.multiply %v7044, %v7046 : tensor<256xf32>
    %v7048 = stablehlo.add %v7045, %v7047 : tensor<256xf32>
    %v7049 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7050 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7051 = stablehlo.divide %v7042, %v7049 : tensor<256xf32>
    %v7052 = stablehlo.divide %v7048, %v7050 : tensor<256xf32>
    %v7053 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7054 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7055 = stablehlo.sqrt %v7052 : tensor<256xf32>
    %v7056 = stablehlo.add %v7055, %v7054 : tensor<256xf32>
    %v7057 = stablehlo.divide %v7051, %v7056 : tensor<256xf32>
    %v7058 = stablehlo.multiply %v7053, %v7057 : tensor<256xf32>
    %v7059 = stablehlo.subtract %s3b2bt2, %v7058 : tensor<256xf32>
    %v7060 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7061 = stablehlo.multiply %v7060, %v7053 : tensor<256xf32>
    %v7062 = stablehlo.multiply %v7061, %s3b2bt2 : tensor<256xf32>
    %v7063 = stablehlo.subtract %v7059, %v7062 : tensor<256xf32>
    %v7064 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7065 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7066 = stablehlo.multiply %v7064, %s3b3W1m : tensor<256x256x3x3xf32>
    %v7067 = stablehlo.multiply %v7065, %v1848 : tensor<256x256x3x3xf32>
    %v7068 = stablehlo.add %v7066, %v7067 : tensor<256x256x3x3xf32>
    %v7069 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7070 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7071 = stablehlo.multiply %v7069, %s3b3W1v : tensor<256x256x3x3xf32>
    %v7072 = stablehlo.multiply %v1848, %v1848 : tensor<256x256x3x3xf32>
    %v7073 = stablehlo.multiply %v7070, %v7072 : tensor<256x256x3x3xf32>
    %v7074 = stablehlo.add %v7071, %v7073 : tensor<256x256x3x3xf32>
    %v7075 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7076 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7077 = stablehlo.multiply %v7075, %s3b3W1m : tensor<256x256x3x3xf32>
    %v7078 = stablehlo.multiply %v7076, %v1848 : tensor<256x256x3x3xf32>
    %v7079 = stablehlo.add %v7077, %v7078 : tensor<256x256x3x3xf32>
    %v7080 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7081 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7082 = stablehlo.multiply %v7080, %s3b3W1v : tensor<256x256x3x3xf32>
    %v7083 = stablehlo.multiply %v1848, %v1848 : tensor<256x256x3x3xf32>
    %v7084 = stablehlo.multiply %v7081, %v7083 : tensor<256x256x3x3xf32>
    %v7085 = stablehlo.add %v7082, %v7084 : tensor<256x256x3x3xf32>
    %v7086 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7087 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7088 = stablehlo.divide %v7079, %v7086 : tensor<256x256x3x3xf32>
    %v7089 = stablehlo.divide %v7085, %v7087 : tensor<256x256x3x3xf32>
    %v7090 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7091 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7092 = stablehlo.sqrt %v7089 : tensor<256x256x3x3xf32>
    %v7093 = stablehlo.add %v7092, %v7091 : tensor<256x256x3x3xf32>
    %v7094 = stablehlo.divide %v7088, %v7093 : tensor<256x256x3x3xf32>
    %v7095 = stablehlo.multiply %v7090, %v7094 : tensor<256x256x3x3xf32>
    %v7096 = stablehlo.subtract %s3b3W1, %v7095 : tensor<256x256x3x3xf32>
    %v7097 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7098 = stablehlo.multiply %v7097, %v7090 : tensor<256x256x3x3xf32>
    %v7099 = stablehlo.multiply %v7098, %s3b3W1 : tensor<256x256x3x3xf32>
    %v7100 = stablehlo.subtract %v7096, %v7099 : tensor<256x256x3x3xf32>
    %v7101 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7102 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7103 = stablehlo.multiply %v7101, %s3b3g1m : tensor<256xf32>
    %v7104 = stablehlo.multiply %v7102, %v1866 : tensor<256xf32>
    %v7105 = stablehlo.add %v7103, %v7104 : tensor<256xf32>
    %v7106 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7107 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7108 = stablehlo.multiply %v7106, %s3b3g1v : tensor<256xf32>
    %v7109 = stablehlo.multiply %v1866, %v1866 : tensor<256xf32>
    %v7110 = stablehlo.multiply %v7107, %v7109 : tensor<256xf32>
    %v7111 = stablehlo.add %v7108, %v7110 : tensor<256xf32>
    %v7112 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7113 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7114 = stablehlo.multiply %v7112, %s3b3g1m : tensor<256xf32>
    %v7115 = stablehlo.multiply %v7113, %v1866 : tensor<256xf32>
    %v7116 = stablehlo.add %v7114, %v7115 : tensor<256xf32>
    %v7117 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7118 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7119 = stablehlo.multiply %v7117, %s3b3g1v : tensor<256xf32>
    %v7120 = stablehlo.multiply %v1866, %v1866 : tensor<256xf32>
    %v7121 = stablehlo.multiply %v7118, %v7120 : tensor<256xf32>
    %v7122 = stablehlo.add %v7119, %v7121 : tensor<256xf32>
    %v7123 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7124 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7125 = stablehlo.divide %v7116, %v7123 : tensor<256xf32>
    %v7126 = stablehlo.divide %v7122, %v7124 : tensor<256xf32>
    %v7127 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7128 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7129 = stablehlo.sqrt %v7126 : tensor<256xf32>
    %v7130 = stablehlo.add %v7129, %v7128 : tensor<256xf32>
    %v7131 = stablehlo.divide %v7125, %v7130 : tensor<256xf32>
    %v7132 = stablehlo.multiply %v7127, %v7131 : tensor<256xf32>
    %v7133 = stablehlo.subtract %s3b3g1, %v7132 : tensor<256xf32>
    %v7134 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7135 = stablehlo.multiply %v7134, %v7127 : tensor<256xf32>
    %v7136 = stablehlo.multiply %v7135, %s3b3g1 : tensor<256xf32>
    %v7137 = stablehlo.subtract %v7133, %v7136 : tensor<256xf32>
    %v7138 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7139 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7140 = stablehlo.multiply %v7138, %s3b3bt1m : tensor<256xf32>
    %v7141 = stablehlo.multiply %v7139, %v1869 : tensor<256xf32>
    %v7142 = stablehlo.add %v7140, %v7141 : tensor<256xf32>
    %v7143 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7144 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7145 = stablehlo.multiply %v7143, %s3b3bt1v : tensor<256xf32>
    %v7146 = stablehlo.multiply %v1869, %v1869 : tensor<256xf32>
    %v7147 = stablehlo.multiply %v7144, %v7146 : tensor<256xf32>
    %v7148 = stablehlo.add %v7145, %v7147 : tensor<256xf32>
    %v7149 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7150 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7151 = stablehlo.multiply %v7149, %s3b3bt1m : tensor<256xf32>
    %v7152 = stablehlo.multiply %v7150, %v1869 : tensor<256xf32>
    %v7153 = stablehlo.add %v7151, %v7152 : tensor<256xf32>
    %v7154 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7155 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7156 = stablehlo.multiply %v7154, %s3b3bt1v : tensor<256xf32>
    %v7157 = stablehlo.multiply %v1869, %v1869 : tensor<256xf32>
    %v7158 = stablehlo.multiply %v7155, %v7157 : tensor<256xf32>
    %v7159 = stablehlo.add %v7156, %v7158 : tensor<256xf32>
    %v7160 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7161 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7162 = stablehlo.divide %v7153, %v7160 : tensor<256xf32>
    %v7163 = stablehlo.divide %v7159, %v7161 : tensor<256xf32>
    %v7164 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7165 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7166 = stablehlo.sqrt %v7163 : tensor<256xf32>
    %v7167 = stablehlo.add %v7166, %v7165 : tensor<256xf32>
    %v7168 = stablehlo.divide %v7162, %v7167 : tensor<256xf32>
    %v7169 = stablehlo.multiply %v7164, %v7168 : tensor<256xf32>
    %v7170 = stablehlo.subtract %s3b3bt1, %v7169 : tensor<256xf32>
    %v7171 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7172 = stablehlo.multiply %v7171, %v7164 : tensor<256xf32>
    %v7173 = stablehlo.multiply %v7172, %s3b3bt1 : tensor<256xf32>
    %v7174 = stablehlo.subtract %v7170, %v7173 : tensor<256xf32>
    %v7175 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7176 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7177 = stablehlo.multiply %v7175, %s3b3W2m : tensor<256x256x3x3xf32>
    %v7178 = stablehlo.multiply %v7176, %v1875 : tensor<256x256x3x3xf32>
    %v7179 = stablehlo.add %v7177, %v7178 : tensor<256x256x3x3xf32>
    %v7180 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7181 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7182 = stablehlo.multiply %v7180, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7183 = stablehlo.multiply %v1875, %v1875 : tensor<256x256x3x3xf32>
    %v7184 = stablehlo.multiply %v7181, %v7183 : tensor<256x256x3x3xf32>
    %v7185 = stablehlo.add %v7182, %v7184 : tensor<256x256x3x3xf32>
    %v7186 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7187 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7188 = stablehlo.multiply %v7186, %s3b3W2m : tensor<256x256x3x3xf32>
    %v7189 = stablehlo.multiply %v7187, %v1875 : tensor<256x256x3x3xf32>
    %v7190 = stablehlo.add %v7188, %v7189 : tensor<256x256x3x3xf32>
    %v7191 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7192 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7193 = stablehlo.multiply %v7191, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7194 = stablehlo.multiply %v1875, %v1875 : tensor<256x256x3x3xf32>
    %v7195 = stablehlo.multiply %v7192, %v7194 : tensor<256x256x3x3xf32>
    %v7196 = stablehlo.add %v7193, %v7195 : tensor<256x256x3x3xf32>
    %v7197 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7198 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7199 = stablehlo.divide %v7190, %v7197 : tensor<256x256x3x3xf32>
    %v7200 = stablehlo.divide %v7196, %v7198 : tensor<256x256x3x3xf32>
    %v7201 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7202 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7203 = stablehlo.sqrt %v7200 : tensor<256x256x3x3xf32>
    %v7204 = stablehlo.add %v7203, %v7202 : tensor<256x256x3x3xf32>
    %v7205 = stablehlo.divide %v7199, %v7204 : tensor<256x256x3x3xf32>
    %v7206 = stablehlo.multiply %v7201, %v7205 : tensor<256x256x3x3xf32>
    %v7207 = stablehlo.subtract %s3b3W2, %v7206 : tensor<256x256x3x3xf32>
    %v7208 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7209 = stablehlo.multiply %v7208, %v7201 : tensor<256x256x3x3xf32>
    %v7210 = stablehlo.multiply %v7209, %s3b3W2 : tensor<256x256x3x3xf32>
    %v7211 = stablehlo.subtract %v7207, %v7210 : tensor<256x256x3x3xf32>
    %v7212 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7213 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7214 = stablehlo.multiply %v7212, %s3b3g2m : tensor<256xf32>
    %v7215 = stablehlo.multiply %v7213, %v1893 : tensor<256xf32>
    %v7216 = stablehlo.add %v7214, %v7215 : tensor<256xf32>
    %v7217 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7218 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7219 = stablehlo.multiply %v7217, %s3b3g2v : tensor<256xf32>
    %v7220 = stablehlo.multiply %v1893, %v1893 : tensor<256xf32>
    %v7221 = stablehlo.multiply %v7218, %v7220 : tensor<256xf32>
    %v7222 = stablehlo.add %v7219, %v7221 : tensor<256xf32>
    %v7223 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7224 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7225 = stablehlo.multiply %v7223, %s3b3g2m : tensor<256xf32>
    %v7226 = stablehlo.multiply %v7224, %v1893 : tensor<256xf32>
    %v7227 = stablehlo.add %v7225, %v7226 : tensor<256xf32>
    %v7228 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7229 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7230 = stablehlo.multiply %v7228, %s3b3g2v : tensor<256xf32>
    %v7231 = stablehlo.multiply %v1893, %v1893 : tensor<256xf32>
    %v7232 = stablehlo.multiply %v7229, %v7231 : tensor<256xf32>
    %v7233 = stablehlo.add %v7230, %v7232 : tensor<256xf32>
    %v7234 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7235 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7236 = stablehlo.divide %v7227, %v7234 : tensor<256xf32>
    %v7237 = stablehlo.divide %v7233, %v7235 : tensor<256xf32>
    %v7238 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7239 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7240 = stablehlo.sqrt %v7237 : tensor<256xf32>
    %v7241 = stablehlo.add %v7240, %v7239 : tensor<256xf32>
    %v7242 = stablehlo.divide %v7236, %v7241 : tensor<256xf32>
    %v7243 = stablehlo.multiply %v7238, %v7242 : tensor<256xf32>
    %v7244 = stablehlo.subtract %s3b3g2, %v7243 : tensor<256xf32>
    %v7245 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7246 = stablehlo.multiply %v7245, %v7238 : tensor<256xf32>
    %v7247 = stablehlo.multiply %v7246, %s3b3g2 : tensor<256xf32>
    %v7248 = stablehlo.subtract %v7244, %v7247 : tensor<256xf32>
    %v7249 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7250 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7251 = stablehlo.multiply %v7249, %s3b3bt2m : tensor<256xf32>
    %v7252 = stablehlo.multiply %v7250, %v1896 : tensor<256xf32>
    %v7253 = stablehlo.add %v7251, %v7252 : tensor<256xf32>
    %v7254 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7255 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7256 = stablehlo.multiply %v7254, %s3b3bt2v : tensor<256xf32>
    %v7257 = stablehlo.multiply %v1896, %v1896 : tensor<256xf32>
    %v7258 = stablehlo.multiply %v7255, %v7257 : tensor<256xf32>
    %v7259 = stablehlo.add %v7256, %v7258 : tensor<256xf32>
    %v7260 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7261 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7262 = stablehlo.multiply %v7260, %s3b3bt2m : tensor<256xf32>
    %v7263 = stablehlo.multiply %v7261, %v1896 : tensor<256xf32>
    %v7264 = stablehlo.add %v7262, %v7263 : tensor<256xf32>
    %v7265 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7266 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7267 = stablehlo.multiply %v7265, %s3b3bt2v : tensor<256xf32>
    %v7268 = stablehlo.multiply %v1896, %v1896 : tensor<256xf32>
    %v7269 = stablehlo.multiply %v7266, %v7268 : tensor<256xf32>
    %v7270 = stablehlo.add %v7267, %v7269 : tensor<256xf32>
    %v7271 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7272 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7273 = stablehlo.divide %v7264, %v7271 : tensor<256xf32>
    %v7274 = stablehlo.divide %v7270, %v7272 : tensor<256xf32>
    %v7275 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7276 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7277 = stablehlo.sqrt %v7274 : tensor<256xf32>
    %v7278 = stablehlo.add %v7277, %v7276 : tensor<256xf32>
    %v7279 = stablehlo.divide %v7273, %v7278 : tensor<256xf32>
    %v7280 = stablehlo.multiply %v7275, %v7279 : tensor<256xf32>
    %v7281 = stablehlo.subtract %s3b3bt2, %v7280 : tensor<256xf32>
    %v7282 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7283 = stablehlo.multiply %v7282, %v7275 : tensor<256xf32>
    %v7284 = stablehlo.multiply %v7283, %s3b3bt2 : tensor<256xf32>
    %v7285 = stablehlo.subtract %v7281, %v7284 : tensor<256xf32>
    %v7286 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7287 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7288 = stablehlo.multiply %v7286, %s3b4W1m : tensor<256x256x3x3xf32>
    %v7289 = stablehlo.multiply %v7287, %v1708 : tensor<256x256x3x3xf32>
    %v7290 = stablehlo.add %v7288, %v7289 : tensor<256x256x3x3xf32>
    %v7291 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7292 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7293 = stablehlo.multiply %v7291, %s3b4W1v : tensor<256x256x3x3xf32>
    %v7294 = stablehlo.multiply %v1708, %v1708 : tensor<256x256x3x3xf32>
    %v7295 = stablehlo.multiply %v7292, %v7294 : tensor<256x256x3x3xf32>
    %v7296 = stablehlo.add %v7293, %v7295 : tensor<256x256x3x3xf32>
    %v7297 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7298 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7299 = stablehlo.multiply %v7297, %s3b4W1m : tensor<256x256x3x3xf32>
    %v7300 = stablehlo.multiply %v7298, %v1708 : tensor<256x256x3x3xf32>
    %v7301 = stablehlo.add %v7299, %v7300 : tensor<256x256x3x3xf32>
    %v7302 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7303 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7304 = stablehlo.multiply %v7302, %s3b4W1v : tensor<256x256x3x3xf32>
    %v7305 = stablehlo.multiply %v1708, %v1708 : tensor<256x256x3x3xf32>
    %v7306 = stablehlo.multiply %v7303, %v7305 : tensor<256x256x3x3xf32>
    %v7307 = stablehlo.add %v7304, %v7306 : tensor<256x256x3x3xf32>
    %v7308 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7309 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7310 = stablehlo.divide %v7301, %v7308 : tensor<256x256x3x3xf32>
    %v7311 = stablehlo.divide %v7307, %v7309 : tensor<256x256x3x3xf32>
    %v7312 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7313 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7314 = stablehlo.sqrt %v7311 : tensor<256x256x3x3xf32>
    %v7315 = stablehlo.add %v7314, %v7313 : tensor<256x256x3x3xf32>
    %v7316 = stablehlo.divide %v7310, %v7315 : tensor<256x256x3x3xf32>
    %v7317 = stablehlo.multiply %v7312, %v7316 : tensor<256x256x3x3xf32>
    %v7318 = stablehlo.subtract %s3b4W1, %v7317 : tensor<256x256x3x3xf32>
    %v7319 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7320 = stablehlo.multiply %v7319, %v7312 : tensor<256x256x3x3xf32>
    %v7321 = stablehlo.multiply %v7320, %s3b4W1 : tensor<256x256x3x3xf32>
    %v7322 = stablehlo.subtract %v7318, %v7321 : tensor<256x256x3x3xf32>
    %v7323 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7324 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7325 = stablehlo.multiply %v7323, %s3b4g1m : tensor<256xf32>
    %v7326 = stablehlo.multiply %v7324, %v1726 : tensor<256xf32>
    %v7327 = stablehlo.add %v7325, %v7326 : tensor<256xf32>
    %v7328 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7329 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7330 = stablehlo.multiply %v7328, %s3b4g1v : tensor<256xf32>
    %v7331 = stablehlo.multiply %v1726, %v1726 : tensor<256xf32>
    %v7332 = stablehlo.multiply %v7329, %v7331 : tensor<256xf32>
    %v7333 = stablehlo.add %v7330, %v7332 : tensor<256xf32>
    %v7334 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7335 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7336 = stablehlo.multiply %v7334, %s3b4g1m : tensor<256xf32>
    %v7337 = stablehlo.multiply %v7335, %v1726 : tensor<256xf32>
    %v7338 = stablehlo.add %v7336, %v7337 : tensor<256xf32>
    %v7339 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7340 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7341 = stablehlo.multiply %v7339, %s3b4g1v : tensor<256xf32>
    %v7342 = stablehlo.multiply %v1726, %v1726 : tensor<256xf32>
    %v7343 = stablehlo.multiply %v7340, %v7342 : tensor<256xf32>
    %v7344 = stablehlo.add %v7341, %v7343 : tensor<256xf32>
    %v7345 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7346 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7347 = stablehlo.divide %v7338, %v7345 : tensor<256xf32>
    %v7348 = stablehlo.divide %v7344, %v7346 : tensor<256xf32>
    %v7349 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7350 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7351 = stablehlo.sqrt %v7348 : tensor<256xf32>
    %v7352 = stablehlo.add %v7351, %v7350 : tensor<256xf32>
    %v7353 = stablehlo.divide %v7347, %v7352 : tensor<256xf32>
    %v7354 = stablehlo.multiply %v7349, %v7353 : tensor<256xf32>
    %v7355 = stablehlo.subtract %s3b4g1, %v7354 : tensor<256xf32>
    %v7356 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7357 = stablehlo.multiply %v7356, %v7349 : tensor<256xf32>
    %v7358 = stablehlo.multiply %v7357, %s3b4g1 : tensor<256xf32>
    %v7359 = stablehlo.subtract %v7355, %v7358 : tensor<256xf32>
    %v7360 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7361 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7362 = stablehlo.multiply %v7360, %s3b4bt1m : tensor<256xf32>
    %v7363 = stablehlo.multiply %v7361, %v1729 : tensor<256xf32>
    %v7364 = stablehlo.add %v7362, %v7363 : tensor<256xf32>
    %v7365 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7366 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7367 = stablehlo.multiply %v7365, %s3b4bt1v : tensor<256xf32>
    %v7368 = stablehlo.multiply %v1729, %v1729 : tensor<256xf32>
    %v7369 = stablehlo.multiply %v7366, %v7368 : tensor<256xf32>
    %v7370 = stablehlo.add %v7367, %v7369 : tensor<256xf32>
    %v7371 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7372 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7373 = stablehlo.multiply %v7371, %s3b4bt1m : tensor<256xf32>
    %v7374 = stablehlo.multiply %v7372, %v1729 : tensor<256xf32>
    %v7375 = stablehlo.add %v7373, %v7374 : tensor<256xf32>
    %v7376 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7377 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7378 = stablehlo.multiply %v7376, %s3b4bt1v : tensor<256xf32>
    %v7379 = stablehlo.multiply %v1729, %v1729 : tensor<256xf32>
    %v7380 = stablehlo.multiply %v7377, %v7379 : tensor<256xf32>
    %v7381 = stablehlo.add %v7378, %v7380 : tensor<256xf32>
    %v7382 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7383 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7384 = stablehlo.divide %v7375, %v7382 : tensor<256xf32>
    %v7385 = stablehlo.divide %v7381, %v7383 : tensor<256xf32>
    %v7386 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7387 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7388 = stablehlo.sqrt %v7385 : tensor<256xf32>
    %v7389 = stablehlo.add %v7388, %v7387 : tensor<256xf32>
    %v7390 = stablehlo.divide %v7384, %v7389 : tensor<256xf32>
    %v7391 = stablehlo.multiply %v7386, %v7390 : tensor<256xf32>
    %v7392 = stablehlo.subtract %s3b4bt1, %v7391 : tensor<256xf32>
    %v7393 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7394 = stablehlo.multiply %v7393, %v7386 : tensor<256xf32>
    %v7395 = stablehlo.multiply %v7394, %s3b4bt1 : tensor<256xf32>
    %v7396 = stablehlo.subtract %v7392, %v7395 : tensor<256xf32>
    %v7397 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7398 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7399 = stablehlo.multiply %v7397, %s3b4W2m : tensor<256x256x3x3xf32>
    %v7400 = stablehlo.multiply %v7398, %v1735 : tensor<256x256x3x3xf32>
    %v7401 = stablehlo.add %v7399, %v7400 : tensor<256x256x3x3xf32>
    %v7402 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7403 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7404 = stablehlo.multiply %v7402, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7405 = stablehlo.multiply %v1735, %v1735 : tensor<256x256x3x3xf32>
    %v7406 = stablehlo.multiply %v7403, %v7405 : tensor<256x256x3x3xf32>
    %v7407 = stablehlo.add %v7404, %v7406 : tensor<256x256x3x3xf32>
    %v7408 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7409 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7410 = stablehlo.multiply %v7408, %s3b4W2m : tensor<256x256x3x3xf32>
    %v7411 = stablehlo.multiply %v7409, %v1735 : tensor<256x256x3x3xf32>
    %v7412 = stablehlo.add %v7410, %v7411 : tensor<256x256x3x3xf32>
    %v7413 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7414 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7415 = stablehlo.multiply %v7413, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7416 = stablehlo.multiply %v1735, %v1735 : tensor<256x256x3x3xf32>
    %v7417 = stablehlo.multiply %v7414, %v7416 : tensor<256x256x3x3xf32>
    %v7418 = stablehlo.add %v7415, %v7417 : tensor<256x256x3x3xf32>
    %v7419 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7420 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7421 = stablehlo.divide %v7412, %v7419 : tensor<256x256x3x3xf32>
    %v7422 = stablehlo.divide %v7418, %v7420 : tensor<256x256x3x3xf32>
    %v7423 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7424 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7425 = stablehlo.sqrt %v7422 : tensor<256x256x3x3xf32>
    %v7426 = stablehlo.add %v7425, %v7424 : tensor<256x256x3x3xf32>
    %v7427 = stablehlo.divide %v7421, %v7426 : tensor<256x256x3x3xf32>
    %v7428 = stablehlo.multiply %v7423, %v7427 : tensor<256x256x3x3xf32>
    %v7429 = stablehlo.subtract %s3b4W2, %v7428 : tensor<256x256x3x3xf32>
    %v7430 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7431 = stablehlo.multiply %v7430, %v7423 : tensor<256x256x3x3xf32>
    %v7432 = stablehlo.multiply %v7431, %s3b4W2 : tensor<256x256x3x3xf32>
    %v7433 = stablehlo.subtract %v7429, %v7432 : tensor<256x256x3x3xf32>
    %v7434 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7435 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7436 = stablehlo.multiply %v7434, %s3b4g2m : tensor<256xf32>
    %v7437 = stablehlo.multiply %v7435, %v1753 : tensor<256xf32>
    %v7438 = stablehlo.add %v7436, %v7437 : tensor<256xf32>
    %v7439 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7440 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7441 = stablehlo.multiply %v7439, %s3b4g2v : tensor<256xf32>
    %v7442 = stablehlo.multiply %v1753, %v1753 : tensor<256xf32>
    %v7443 = stablehlo.multiply %v7440, %v7442 : tensor<256xf32>
    %v7444 = stablehlo.add %v7441, %v7443 : tensor<256xf32>
    %v7445 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7446 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7447 = stablehlo.multiply %v7445, %s3b4g2m : tensor<256xf32>
    %v7448 = stablehlo.multiply %v7446, %v1753 : tensor<256xf32>
    %v7449 = stablehlo.add %v7447, %v7448 : tensor<256xf32>
    %v7450 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7451 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7452 = stablehlo.multiply %v7450, %s3b4g2v : tensor<256xf32>
    %v7453 = stablehlo.multiply %v1753, %v1753 : tensor<256xf32>
    %v7454 = stablehlo.multiply %v7451, %v7453 : tensor<256xf32>
    %v7455 = stablehlo.add %v7452, %v7454 : tensor<256xf32>
    %v7456 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7457 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7458 = stablehlo.divide %v7449, %v7456 : tensor<256xf32>
    %v7459 = stablehlo.divide %v7455, %v7457 : tensor<256xf32>
    %v7460 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7461 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7462 = stablehlo.sqrt %v7459 : tensor<256xf32>
    %v7463 = stablehlo.add %v7462, %v7461 : tensor<256xf32>
    %v7464 = stablehlo.divide %v7458, %v7463 : tensor<256xf32>
    %v7465 = stablehlo.multiply %v7460, %v7464 : tensor<256xf32>
    %v7466 = stablehlo.subtract %s3b4g2, %v7465 : tensor<256xf32>
    %v7467 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7468 = stablehlo.multiply %v7467, %v7460 : tensor<256xf32>
    %v7469 = stablehlo.multiply %v7468, %s3b4g2 : tensor<256xf32>
    %v7470 = stablehlo.subtract %v7466, %v7469 : tensor<256xf32>
    %v7471 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7472 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7473 = stablehlo.multiply %v7471, %s3b4bt2m : tensor<256xf32>
    %v7474 = stablehlo.multiply %v7472, %v1756 : tensor<256xf32>
    %v7475 = stablehlo.add %v7473, %v7474 : tensor<256xf32>
    %v7476 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7477 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7478 = stablehlo.multiply %v7476, %s3b4bt2v : tensor<256xf32>
    %v7479 = stablehlo.multiply %v1756, %v1756 : tensor<256xf32>
    %v7480 = stablehlo.multiply %v7477, %v7479 : tensor<256xf32>
    %v7481 = stablehlo.add %v7478, %v7480 : tensor<256xf32>
    %v7482 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7483 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7484 = stablehlo.multiply %v7482, %s3b4bt2m : tensor<256xf32>
    %v7485 = stablehlo.multiply %v7483, %v1756 : tensor<256xf32>
    %v7486 = stablehlo.add %v7484, %v7485 : tensor<256xf32>
    %v7487 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7488 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7489 = stablehlo.multiply %v7487, %s3b4bt2v : tensor<256xf32>
    %v7490 = stablehlo.multiply %v1756, %v1756 : tensor<256xf32>
    %v7491 = stablehlo.multiply %v7488, %v7490 : tensor<256xf32>
    %v7492 = stablehlo.add %v7489, %v7491 : tensor<256xf32>
    %v7493 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7494 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7495 = stablehlo.divide %v7486, %v7493 : tensor<256xf32>
    %v7496 = stablehlo.divide %v7492, %v7494 : tensor<256xf32>
    %v7497 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7498 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7499 = stablehlo.sqrt %v7496 : tensor<256xf32>
    %v7500 = stablehlo.add %v7499, %v7498 : tensor<256xf32>
    %v7501 = stablehlo.divide %v7495, %v7500 : tensor<256xf32>
    %v7502 = stablehlo.multiply %v7497, %v7501 : tensor<256xf32>
    %v7503 = stablehlo.subtract %s3b4bt2, %v7502 : tensor<256xf32>
    %v7504 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7505 = stablehlo.multiply %v7504, %v7497 : tensor<256xf32>
    %v7506 = stablehlo.multiply %v7505, %s3b4bt2 : tensor<256xf32>
    %v7507 = stablehlo.subtract %v7503, %v7506 : tensor<256xf32>
    %v7508 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7509 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7510 = stablehlo.multiply %v7508, %d4W1m : tensor<512x256x3x3xf32>
    %v7511 = stablehlo.multiply %v7509, %v1540 : tensor<512x256x3x3xf32>
    %v7512 = stablehlo.add %v7510, %v7511 : tensor<512x256x3x3xf32>
    %v7513 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7514 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7515 = stablehlo.multiply %v7513, %d4W1v : tensor<512x256x3x3xf32>
    %v7516 = stablehlo.multiply %v1540, %v1540 : tensor<512x256x3x3xf32>
    %v7517 = stablehlo.multiply %v7514, %v7516 : tensor<512x256x3x3xf32>
    %v7518 = stablehlo.add %v7515, %v7517 : tensor<512x256x3x3xf32>
    %v7519 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7520 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7521 = stablehlo.multiply %v7519, %d4W1m : tensor<512x256x3x3xf32>
    %v7522 = stablehlo.multiply %v7520, %v1540 : tensor<512x256x3x3xf32>
    %v7523 = stablehlo.add %v7521, %v7522 : tensor<512x256x3x3xf32>
    %v7524 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7525 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7526 = stablehlo.multiply %v7524, %d4W1v : tensor<512x256x3x3xf32>
    %v7527 = stablehlo.multiply %v1540, %v1540 : tensor<512x256x3x3xf32>
    %v7528 = stablehlo.multiply %v7525, %v7527 : tensor<512x256x3x3xf32>
    %v7529 = stablehlo.add %v7526, %v7528 : tensor<512x256x3x3xf32>
    %v7530 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7531 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7532 = stablehlo.divide %v7523, %v7530 : tensor<512x256x3x3xf32>
    %v7533 = stablehlo.divide %v7529, %v7531 : tensor<512x256x3x3xf32>
    %v7534 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7535 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7536 = stablehlo.sqrt %v7533 : tensor<512x256x3x3xf32>
    %v7537 = stablehlo.add %v7536, %v7535 : tensor<512x256x3x3xf32>
    %v7538 = stablehlo.divide %v7532, %v7537 : tensor<512x256x3x3xf32>
    %v7539 = stablehlo.multiply %v7534, %v7538 : tensor<512x256x3x3xf32>
    %v7540 = stablehlo.subtract %d4W1, %v7539 : tensor<512x256x3x3xf32>
    %v7541 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7542 = stablehlo.multiply %v7541, %v7534 : tensor<512x256x3x3xf32>
    %v7543 = stablehlo.multiply %v7542, %d4W1 : tensor<512x256x3x3xf32>
    %v7544 = stablehlo.subtract %v7540, %v7543 : tensor<512x256x3x3xf32>
    %v7545 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7546 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7547 = stablehlo.multiply %v7545, %d4g1m : tensor<512xf32>
    %v7548 = stablehlo.multiply %v7546, %v1558 : tensor<512xf32>
    %v7549 = stablehlo.add %v7547, %v7548 : tensor<512xf32>
    %v7550 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7551 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7552 = stablehlo.multiply %v7550, %d4g1v : tensor<512xf32>
    %v7553 = stablehlo.multiply %v1558, %v1558 : tensor<512xf32>
    %v7554 = stablehlo.multiply %v7551, %v7553 : tensor<512xf32>
    %v7555 = stablehlo.add %v7552, %v7554 : tensor<512xf32>
    %v7556 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7557 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7558 = stablehlo.multiply %v7556, %d4g1m : tensor<512xf32>
    %v7559 = stablehlo.multiply %v7557, %v1558 : tensor<512xf32>
    %v7560 = stablehlo.add %v7558, %v7559 : tensor<512xf32>
    %v7561 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7562 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7563 = stablehlo.multiply %v7561, %d4g1v : tensor<512xf32>
    %v7564 = stablehlo.multiply %v1558, %v1558 : tensor<512xf32>
    %v7565 = stablehlo.multiply %v7562, %v7564 : tensor<512xf32>
    %v7566 = stablehlo.add %v7563, %v7565 : tensor<512xf32>
    %v7567 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7568 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7569 = stablehlo.divide %v7560, %v7567 : tensor<512xf32>
    %v7570 = stablehlo.divide %v7566, %v7568 : tensor<512xf32>
    %v7571 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7572 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7573 = stablehlo.sqrt %v7570 : tensor<512xf32>
    %v7574 = stablehlo.add %v7573, %v7572 : tensor<512xf32>
    %v7575 = stablehlo.divide %v7569, %v7574 : tensor<512xf32>
    %v7576 = stablehlo.multiply %v7571, %v7575 : tensor<512xf32>
    %v7577 = stablehlo.subtract %d4g1, %v7576 : tensor<512xf32>
    %v7578 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7579 = stablehlo.multiply %v7578, %v7571 : tensor<512xf32>
    %v7580 = stablehlo.multiply %v7579, %d4g1 : tensor<512xf32>
    %v7581 = stablehlo.subtract %v7577, %v7580 : tensor<512xf32>
    %v7582 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7583 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7584 = stablehlo.multiply %v7582, %d4bt1m : tensor<512xf32>
    %v7585 = stablehlo.multiply %v7583, %v1561 : tensor<512xf32>
    %v7586 = stablehlo.add %v7584, %v7585 : tensor<512xf32>
    %v7587 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7588 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7589 = stablehlo.multiply %v7587, %d4bt1v : tensor<512xf32>
    %v7590 = stablehlo.multiply %v1561, %v1561 : tensor<512xf32>
    %v7591 = stablehlo.multiply %v7588, %v7590 : tensor<512xf32>
    %v7592 = stablehlo.add %v7589, %v7591 : tensor<512xf32>
    %v7593 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7594 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7595 = stablehlo.multiply %v7593, %d4bt1m : tensor<512xf32>
    %v7596 = stablehlo.multiply %v7594, %v1561 : tensor<512xf32>
    %v7597 = stablehlo.add %v7595, %v7596 : tensor<512xf32>
    %v7598 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7599 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7600 = stablehlo.multiply %v7598, %d4bt1v : tensor<512xf32>
    %v7601 = stablehlo.multiply %v1561, %v1561 : tensor<512xf32>
    %v7602 = stablehlo.multiply %v7599, %v7601 : tensor<512xf32>
    %v7603 = stablehlo.add %v7600, %v7602 : tensor<512xf32>
    %v7604 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7605 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7606 = stablehlo.divide %v7597, %v7604 : tensor<512xf32>
    %v7607 = stablehlo.divide %v7603, %v7605 : tensor<512xf32>
    %v7608 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7609 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7610 = stablehlo.sqrt %v7607 : tensor<512xf32>
    %v7611 = stablehlo.add %v7610, %v7609 : tensor<512xf32>
    %v7612 = stablehlo.divide %v7606, %v7611 : tensor<512xf32>
    %v7613 = stablehlo.multiply %v7608, %v7612 : tensor<512xf32>
    %v7614 = stablehlo.subtract %d4bt1, %v7613 : tensor<512xf32>
    %v7615 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7616 = stablehlo.multiply %v7615, %v7608 : tensor<512xf32>
    %v7617 = stablehlo.multiply %v7616, %d4bt1 : tensor<512xf32>
    %v7618 = stablehlo.subtract %v7614, %v7617 : tensor<512xf32>
    %v7619 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7620 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7621 = stablehlo.multiply %v7619, %d4W2m : tensor<512x512x3x3xf32>
    %v7622 = stablehlo.multiply %v7620, %v1567 : tensor<512x512x3x3xf32>
    %v7623 = stablehlo.add %v7621, %v7622 : tensor<512x512x3x3xf32>
    %v7624 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7625 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7626 = stablehlo.multiply %v7624, %d4W2v : tensor<512x512x3x3xf32>
    %v7627 = stablehlo.multiply %v1567, %v1567 : tensor<512x512x3x3xf32>
    %v7628 = stablehlo.multiply %v7625, %v7627 : tensor<512x512x3x3xf32>
    %v7629 = stablehlo.add %v7626, %v7628 : tensor<512x512x3x3xf32>
    %v7630 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7631 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7632 = stablehlo.multiply %v7630, %d4W2m : tensor<512x512x3x3xf32>
    %v7633 = stablehlo.multiply %v7631, %v1567 : tensor<512x512x3x3xf32>
    %v7634 = stablehlo.add %v7632, %v7633 : tensor<512x512x3x3xf32>
    %v7635 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7636 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7637 = stablehlo.multiply %v7635, %d4W2v : tensor<512x512x3x3xf32>
    %v7638 = stablehlo.multiply %v1567, %v1567 : tensor<512x512x3x3xf32>
    %v7639 = stablehlo.multiply %v7636, %v7638 : tensor<512x512x3x3xf32>
    %v7640 = stablehlo.add %v7637, %v7639 : tensor<512x512x3x3xf32>
    %v7641 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7642 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7643 = stablehlo.divide %v7634, %v7641 : tensor<512x512x3x3xf32>
    %v7644 = stablehlo.divide %v7640, %v7642 : tensor<512x512x3x3xf32>
    %v7645 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7646 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7647 = stablehlo.sqrt %v7644 : tensor<512x512x3x3xf32>
    %v7648 = stablehlo.add %v7647, %v7646 : tensor<512x512x3x3xf32>
    %v7649 = stablehlo.divide %v7643, %v7648 : tensor<512x512x3x3xf32>
    %v7650 = stablehlo.multiply %v7645, %v7649 : tensor<512x512x3x3xf32>
    %v7651 = stablehlo.subtract %d4W2, %v7650 : tensor<512x512x3x3xf32>
    %v7652 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7653 = stablehlo.multiply %v7652, %v7645 : tensor<512x512x3x3xf32>
    %v7654 = stablehlo.multiply %v7653, %d4W2 : tensor<512x512x3x3xf32>
    %v7655 = stablehlo.subtract %v7651, %v7654 : tensor<512x512x3x3xf32>
    %v7656 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7657 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7658 = stablehlo.multiply %v7656, %d4g2m : tensor<512xf32>
    %v7659 = stablehlo.multiply %v7657, %v1585 : tensor<512xf32>
    %v7660 = stablehlo.add %v7658, %v7659 : tensor<512xf32>
    %v7661 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7662 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7663 = stablehlo.multiply %v7661, %d4g2v : tensor<512xf32>
    %v7664 = stablehlo.multiply %v1585, %v1585 : tensor<512xf32>
    %v7665 = stablehlo.multiply %v7662, %v7664 : tensor<512xf32>
    %v7666 = stablehlo.add %v7663, %v7665 : tensor<512xf32>
    %v7667 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7668 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7669 = stablehlo.multiply %v7667, %d4g2m : tensor<512xf32>
    %v7670 = stablehlo.multiply %v7668, %v1585 : tensor<512xf32>
    %v7671 = stablehlo.add %v7669, %v7670 : tensor<512xf32>
    %v7672 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7673 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7674 = stablehlo.multiply %v7672, %d4g2v : tensor<512xf32>
    %v7675 = stablehlo.multiply %v1585, %v1585 : tensor<512xf32>
    %v7676 = stablehlo.multiply %v7673, %v7675 : tensor<512xf32>
    %v7677 = stablehlo.add %v7674, %v7676 : tensor<512xf32>
    %v7678 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7679 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7680 = stablehlo.divide %v7671, %v7678 : tensor<512xf32>
    %v7681 = stablehlo.divide %v7677, %v7679 : tensor<512xf32>
    %v7682 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7683 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7684 = stablehlo.sqrt %v7681 : tensor<512xf32>
    %v7685 = stablehlo.add %v7684, %v7683 : tensor<512xf32>
    %v7686 = stablehlo.divide %v7680, %v7685 : tensor<512xf32>
    %v7687 = stablehlo.multiply %v7682, %v7686 : tensor<512xf32>
    %v7688 = stablehlo.subtract %d4g2, %v7687 : tensor<512xf32>
    %v7689 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7690 = stablehlo.multiply %v7689, %v7682 : tensor<512xf32>
    %v7691 = stablehlo.multiply %v7690, %d4g2 : tensor<512xf32>
    %v7692 = stablehlo.subtract %v7688, %v7691 : tensor<512xf32>
    %v7693 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7694 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7695 = stablehlo.multiply %v7693, %d4bt2m : tensor<512xf32>
    %v7696 = stablehlo.multiply %v7694, %v1588 : tensor<512xf32>
    %v7697 = stablehlo.add %v7695, %v7696 : tensor<512xf32>
    %v7698 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7699 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7700 = stablehlo.multiply %v7698, %d4bt2v : tensor<512xf32>
    %v7701 = stablehlo.multiply %v1588, %v1588 : tensor<512xf32>
    %v7702 = stablehlo.multiply %v7699, %v7701 : tensor<512xf32>
    %v7703 = stablehlo.add %v7700, %v7702 : tensor<512xf32>
    %v7704 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7705 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7706 = stablehlo.multiply %v7704, %d4bt2m : tensor<512xf32>
    %v7707 = stablehlo.multiply %v7705, %v1588 : tensor<512xf32>
    %v7708 = stablehlo.add %v7706, %v7707 : tensor<512xf32>
    %v7709 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7710 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7711 = stablehlo.multiply %v7709, %d4bt2v : tensor<512xf32>
    %v7712 = stablehlo.multiply %v1588, %v1588 : tensor<512xf32>
    %v7713 = stablehlo.multiply %v7710, %v7712 : tensor<512xf32>
    %v7714 = stablehlo.add %v7711, %v7713 : tensor<512xf32>
    %v7715 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7716 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7717 = stablehlo.divide %v7708, %v7715 : tensor<512xf32>
    %v7718 = stablehlo.divide %v7714, %v7716 : tensor<512xf32>
    %v7719 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7720 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7721 = stablehlo.sqrt %v7718 : tensor<512xf32>
    %v7722 = stablehlo.add %v7721, %v7720 : tensor<512xf32>
    %v7723 = stablehlo.divide %v7717, %v7722 : tensor<512xf32>
    %v7724 = stablehlo.multiply %v7719, %v7723 : tensor<512xf32>
    %v7725 = stablehlo.subtract %d4bt2, %v7724 : tensor<512xf32>
    %v7726 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7727 = stablehlo.multiply %v7726, %v7719 : tensor<512xf32>
    %v7728 = stablehlo.multiply %v7727, %d4bt2 : tensor<512xf32>
    %v7729 = stablehlo.subtract %v7725, %v7728 : tensor<512xf32>
    %v7730 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7731 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7732 = stablehlo.multiply %v7730, %d4Wpm : tensor<512x256x1x1xf32>
    %v7733 = stablehlo.multiply %v7731, %v1595 : tensor<512x256x1x1xf32>
    %v7734 = stablehlo.add %v7732, %v7733 : tensor<512x256x1x1xf32>
    %v7735 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7736 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7737 = stablehlo.multiply %v7735, %d4Wpv : tensor<512x256x1x1xf32>
    %v7738 = stablehlo.multiply %v1595, %v1595 : tensor<512x256x1x1xf32>
    %v7739 = stablehlo.multiply %v7736, %v7738 : tensor<512x256x1x1xf32>
    %v7740 = stablehlo.add %v7737, %v7739 : tensor<512x256x1x1xf32>
    %v7741 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7742 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7743 = stablehlo.multiply %v7741, %d4Wpm : tensor<512x256x1x1xf32>
    %v7744 = stablehlo.multiply %v7742, %v1595 : tensor<512x256x1x1xf32>
    %v7745 = stablehlo.add %v7743, %v7744 : tensor<512x256x1x1xf32>
    %v7746 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7747 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7748 = stablehlo.multiply %v7746, %d4Wpv : tensor<512x256x1x1xf32>
    %v7749 = stablehlo.multiply %v1595, %v1595 : tensor<512x256x1x1xf32>
    %v7750 = stablehlo.multiply %v7747, %v7749 : tensor<512x256x1x1xf32>
    %v7751 = stablehlo.add %v7748, %v7750 : tensor<512x256x1x1xf32>
    %v7752 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7753 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7754 = stablehlo.divide %v7745, %v7752 : tensor<512x256x1x1xf32>
    %v7755 = stablehlo.divide %v7751, %v7753 : tensor<512x256x1x1xf32>
    %v7756 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7757 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7758 = stablehlo.sqrt %v7755 : tensor<512x256x1x1xf32>
    %v7759 = stablehlo.add %v7758, %v7757 : tensor<512x256x1x1xf32>
    %v7760 = stablehlo.divide %v7754, %v7759 : tensor<512x256x1x1xf32>
    %v7761 = stablehlo.multiply %v7756, %v7760 : tensor<512x256x1x1xf32>
    %v7762 = stablehlo.subtract %d4Wp, %v7761 : tensor<512x256x1x1xf32>
    %v7763 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7764 = stablehlo.multiply %v7763, %v7756 : tensor<512x256x1x1xf32>
    %v7765 = stablehlo.multiply %v7764, %d4Wp : tensor<512x256x1x1xf32>
    %v7766 = stablehlo.subtract %v7762, %v7765 : tensor<512x256x1x1xf32>
    %v7767 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7768 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7769 = stablehlo.multiply %v7767, %d4gpm : tensor<512xf32>
    %v7770 = stablehlo.multiply %v7768, %v1613 : tensor<512xf32>
    %v7771 = stablehlo.add %v7769, %v7770 : tensor<512xf32>
    %v7772 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7773 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7774 = stablehlo.multiply %v7772, %d4gpv : tensor<512xf32>
    %v7775 = stablehlo.multiply %v1613, %v1613 : tensor<512xf32>
    %v7776 = stablehlo.multiply %v7773, %v7775 : tensor<512xf32>
    %v7777 = stablehlo.add %v7774, %v7776 : tensor<512xf32>
    %v7778 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7779 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7780 = stablehlo.multiply %v7778, %d4gpm : tensor<512xf32>
    %v7781 = stablehlo.multiply %v7779, %v1613 : tensor<512xf32>
    %v7782 = stablehlo.add %v7780, %v7781 : tensor<512xf32>
    %v7783 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7784 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7785 = stablehlo.multiply %v7783, %d4gpv : tensor<512xf32>
    %v7786 = stablehlo.multiply %v1613, %v1613 : tensor<512xf32>
    %v7787 = stablehlo.multiply %v7784, %v7786 : tensor<512xf32>
    %v7788 = stablehlo.add %v7785, %v7787 : tensor<512xf32>
    %v7789 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7790 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7791 = stablehlo.divide %v7782, %v7789 : tensor<512xf32>
    %v7792 = stablehlo.divide %v7788, %v7790 : tensor<512xf32>
    %v7793 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7794 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7795 = stablehlo.sqrt %v7792 : tensor<512xf32>
    %v7796 = stablehlo.add %v7795, %v7794 : tensor<512xf32>
    %v7797 = stablehlo.divide %v7791, %v7796 : tensor<512xf32>
    %v7798 = stablehlo.multiply %v7793, %v7797 : tensor<512xf32>
    %v7799 = stablehlo.subtract %d4gp, %v7798 : tensor<512xf32>
    %v7800 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7801 = stablehlo.multiply %v7800, %v7793 : tensor<512xf32>
    %v7802 = stablehlo.multiply %v7801, %d4gp : tensor<512xf32>
    %v7803 = stablehlo.subtract %v7799, %v7802 : tensor<512xf32>
    %v7804 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7805 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7806 = stablehlo.multiply %v7804, %d4btpm : tensor<512xf32>
    %v7807 = stablehlo.multiply %v7805, %v1616 : tensor<512xf32>
    %v7808 = stablehlo.add %v7806, %v7807 : tensor<512xf32>
    %v7809 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7810 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7811 = stablehlo.multiply %v7809, %d4btpv : tensor<512xf32>
    %v7812 = stablehlo.multiply %v1616, %v1616 : tensor<512xf32>
    %v7813 = stablehlo.multiply %v7810, %v7812 : tensor<512xf32>
    %v7814 = stablehlo.add %v7811, %v7813 : tensor<512xf32>
    %v7815 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7816 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7817 = stablehlo.multiply %v7815, %d4btpm : tensor<512xf32>
    %v7818 = stablehlo.multiply %v7816, %v1616 : tensor<512xf32>
    %v7819 = stablehlo.add %v7817, %v7818 : tensor<512xf32>
    %v7820 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7821 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7822 = stablehlo.multiply %v7820, %d4btpv : tensor<512xf32>
    %v7823 = stablehlo.multiply %v1616, %v1616 : tensor<512xf32>
    %v7824 = stablehlo.multiply %v7821, %v7823 : tensor<512xf32>
    %v7825 = stablehlo.add %v7822, %v7824 : tensor<512xf32>
    %v7826 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7827 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7828 = stablehlo.divide %v7819, %v7826 : tensor<512xf32>
    %v7829 = stablehlo.divide %v7825, %v7827 : tensor<512xf32>
    %v7830 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7831 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7832 = stablehlo.sqrt %v7829 : tensor<512xf32>
    %v7833 = stablehlo.add %v7832, %v7831 : tensor<512xf32>
    %v7834 = stablehlo.divide %v7828, %v7833 : tensor<512xf32>
    %v7835 = stablehlo.multiply %v7830, %v7834 : tensor<512xf32>
    %v7836 = stablehlo.subtract %d4btp, %v7835 : tensor<512xf32>
    %v7837 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7838 = stablehlo.multiply %v7837, %v7830 : tensor<512xf32>
    %v7839 = stablehlo.multiply %v7838, %d4btp : tensor<512xf32>
    %v7840 = stablehlo.subtract %v7836, %v7839 : tensor<512xf32>
    %v7841 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7842 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7843 = stablehlo.multiply %v7841, %s4b0W1m : tensor<512x512x3x3xf32>
    %v7844 = stablehlo.multiply %v7842, %v1364 : tensor<512x512x3x3xf32>
    %v7845 = stablehlo.add %v7843, %v7844 : tensor<512x512x3x3xf32>
    %v7846 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7847 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7848 = stablehlo.multiply %v7846, %s4b0W1v : tensor<512x512x3x3xf32>
    %v7849 = stablehlo.multiply %v1364, %v1364 : tensor<512x512x3x3xf32>
    %v7850 = stablehlo.multiply %v7847, %v7849 : tensor<512x512x3x3xf32>
    %v7851 = stablehlo.add %v7848, %v7850 : tensor<512x512x3x3xf32>
    %v7852 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7853 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7854 = stablehlo.multiply %v7852, %s4b0W1m : tensor<512x512x3x3xf32>
    %v7855 = stablehlo.multiply %v7853, %v1364 : tensor<512x512x3x3xf32>
    %v7856 = stablehlo.add %v7854, %v7855 : tensor<512x512x3x3xf32>
    %v7857 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7858 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7859 = stablehlo.multiply %v7857, %s4b0W1v : tensor<512x512x3x3xf32>
    %v7860 = stablehlo.multiply %v1364, %v1364 : tensor<512x512x3x3xf32>
    %v7861 = stablehlo.multiply %v7858, %v7860 : tensor<512x512x3x3xf32>
    %v7862 = stablehlo.add %v7859, %v7861 : tensor<512x512x3x3xf32>
    %v7863 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7864 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7865 = stablehlo.divide %v7856, %v7863 : tensor<512x512x3x3xf32>
    %v7866 = stablehlo.divide %v7862, %v7864 : tensor<512x512x3x3xf32>
    %v7867 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7868 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7869 = stablehlo.sqrt %v7866 : tensor<512x512x3x3xf32>
    %v7870 = stablehlo.add %v7869, %v7868 : tensor<512x512x3x3xf32>
    %v7871 = stablehlo.divide %v7865, %v7870 : tensor<512x512x3x3xf32>
    %v7872 = stablehlo.multiply %v7867, %v7871 : tensor<512x512x3x3xf32>
    %v7873 = stablehlo.subtract %s4b0W1, %v7872 : tensor<512x512x3x3xf32>
    %v7874 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7875 = stablehlo.multiply %v7874, %v7867 : tensor<512x512x3x3xf32>
    %v7876 = stablehlo.multiply %v7875, %s4b0W1 : tensor<512x512x3x3xf32>
    %v7877 = stablehlo.subtract %v7873, %v7876 : tensor<512x512x3x3xf32>
    %v7878 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7879 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7880 = stablehlo.multiply %v7878, %s4b0g1m : tensor<512xf32>
    %v7881 = stablehlo.multiply %v7879, %v1382 : tensor<512xf32>
    %v7882 = stablehlo.add %v7880, %v7881 : tensor<512xf32>
    %v7883 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7884 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7885 = stablehlo.multiply %v7883, %s4b0g1v : tensor<512xf32>
    %v7886 = stablehlo.multiply %v1382, %v1382 : tensor<512xf32>
    %v7887 = stablehlo.multiply %v7884, %v7886 : tensor<512xf32>
    %v7888 = stablehlo.add %v7885, %v7887 : tensor<512xf32>
    %v7889 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7890 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7891 = stablehlo.multiply %v7889, %s4b0g1m : tensor<512xf32>
    %v7892 = stablehlo.multiply %v7890, %v1382 : tensor<512xf32>
    %v7893 = stablehlo.add %v7891, %v7892 : tensor<512xf32>
    %v7894 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7895 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7896 = stablehlo.multiply %v7894, %s4b0g1v : tensor<512xf32>
    %v7897 = stablehlo.multiply %v1382, %v1382 : tensor<512xf32>
    %v7898 = stablehlo.multiply %v7895, %v7897 : tensor<512xf32>
    %v7899 = stablehlo.add %v7896, %v7898 : tensor<512xf32>
    %v7900 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7901 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7902 = stablehlo.divide %v7893, %v7900 : tensor<512xf32>
    %v7903 = stablehlo.divide %v7899, %v7901 : tensor<512xf32>
    %v7904 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7905 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7906 = stablehlo.sqrt %v7903 : tensor<512xf32>
    %v7907 = stablehlo.add %v7906, %v7905 : tensor<512xf32>
    %v7908 = stablehlo.divide %v7902, %v7907 : tensor<512xf32>
    %v7909 = stablehlo.multiply %v7904, %v7908 : tensor<512xf32>
    %v7910 = stablehlo.subtract %s4b0g1, %v7909 : tensor<512xf32>
    %v7911 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7912 = stablehlo.multiply %v7911, %v7904 : tensor<512xf32>
    %v7913 = stablehlo.multiply %v7912, %s4b0g1 : tensor<512xf32>
    %v7914 = stablehlo.subtract %v7910, %v7913 : tensor<512xf32>
    %v7915 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7916 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7917 = stablehlo.multiply %v7915, %s4b0bt1m : tensor<512xf32>
    %v7918 = stablehlo.multiply %v7916, %v1385 : tensor<512xf32>
    %v7919 = stablehlo.add %v7917, %v7918 : tensor<512xf32>
    %v7920 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7921 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7922 = stablehlo.multiply %v7920, %s4b0bt1v : tensor<512xf32>
    %v7923 = stablehlo.multiply %v1385, %v1385 : tensor<512xf32>
    %v7924 = stablehlo.multiply %v7921, %v7923 : tensor<512xf32>
    %v7925 = stablehlo.add %v7922, %v7924 : tensor<512xf32>
    %v7926 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7927 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7928 = stablehlo.multiply %v7926, %s4b0bt1m : tensor<512xf32>
    %v7929 = stablehlo.multiply %v7927, %v1385 : tensor<512xf32>
    %v7930 = stablehlo.add %v7928, %v7929 : tensor<512xf32>
    %v7931 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7932 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7933 = stablehlo.multiply %v7931, %s4b0bt1v : tensor<512xf32>
    %v7934 = stablehlo.multiply %v1385, %v1385 : tensor<512xf32>
    %v7935 = stablehlo.multiply %v7932, %v7934 : tensor<512xf32>
    %v7936 = stablehlo.add %v7933, %v7935 : tensor<512xf32>
    %v7937 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7938 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7939 = stablehlo.divide %v7930, %v7937 : tensor<512xf32>
    %v7940 = stablehlo.divide %v7936, %v7938 : tensor<512xf32>
    %v7941 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7942 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7943 = stablehlo.sqrt %v7940 : tensor<512xf32>
    %v7944 = stablehlo.add %v7943, %v7942 : tensor<512xf32>
    %v7945 = stablehlo.divide %v7939, %v7944 : tensor<512xf32>
    %v7946 = stablehlo.multiply %v7941, %v7945 : tensor<512xf32>
    %v7947 = stablehlo.subtract %s4b0bt1, %v7946 : tensor<512xf32>
    %v7948 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7949 = stablehlo.multiply %v7948, %v7941 : tensor<512xf32>
    %v7950 = stablehlo.multiply %v7949, %s4b0bt1 : tensor<512xf32>
    %v7951 = stablehlo.subtract %v7947, %v7950 : tensor<512xf32>
    %v7952 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7953 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7954 = stablehlo.multiply %v7952, %s4b0W2m : tensor<512x512x3x3xf32>
    %v7955 = stablehlo.multiply %v7953, %v1391 : tensor<512x512x3x3xf32>
    %v7956 = stablehlo.add %v7954, %v7955 : tensor<512x512x3x3xf32>
    %v7957 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7958 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7959 = stablehlo.multiply %v7957, %s4b0W2v : tensor<512x512x3x3xf32>
    %v7960 = stablehlo.multiply %v1391, %v1391 : tensor<512x512x3x3xf32>
    %v7961 = stablehlo.multiply %v7958, %v7960 : tensor<512x512x3x3xf32>
    %v7962 = stablehlo.add %v7959, %v7961 : tensor<512x512x3x3xf32>
    %v7963 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7964 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7965 = stablehlo.multiply %v7963, %s4b0W2m : tensor<512x512x3x3xf32>
    %v7966 = stablehlo.multiply %v7964, %v1391 : tensor<512x512x3x3xf32>
    %v7967 = stablehlo.add %v7965, %v7966 : tensor<512x512x3x3xf32>
    %v7968 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7969 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7970 = stablehlo.multiply %v7968, %s4b0W2v : tensor<512x512x3x3xf32>
    %v7971 = stablehlo.multiply %v1391, %v1391 : tensor<512x512x3x3xf32>
    %v7972 = stablehlo.multiply %v7969, %v7971 : tensor<512x512x3x3xf32>
    %v7973 = stablehlo.add %v7970, %v7972 : tensor<512x512x3x3xf32>
    %v7974 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7975 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7976 = stablehlo.divide %v7967, %v7974 : tensor<512x512x3x3xf32>
    %v7977 = stablehlo.divide %v7973, %v7975 : tensor<512x512x3x3xf32>
    %v7978 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7979 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7980 = stablehlo.sqrt %v7977 : tensor<512x512x3x3xf32>
    %v7981 = stablehlo.add %v7980, %v7979 : tensor<512x512x3x3xf32>
    %v7982 = stablehlo.divide %v7976, %v7981 : tensor<512x512x3x3xf32>
    %v7983 = stablehlo.multiply %v7978, %v7982 : tensor<512x512x3x3xf32>
    %v7984 = stablehlo.subtract %s4b0W2, %v7983 : tensor<512x512x3x3xf32>
    %v7985 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7986 = stablehlo.multiply %v7985, %v7978 : tensor<512x512x3x3xf32>
    %v7987 = stablehlo.multiply %v7986, %s4b0W2 : tensor<512x512x3x3xf32>
    %v7988 = stablehlo.subtract %v7984, %v7987 : tensor<512x512x3x3xf32>
    %v7989 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7990 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7991 = stablehlo.multiply %v7989, %s4b0g2m : tensor<512xf32>
    %v7992 = stablehlo.multiply %v7990, %v1409 : tensor<512xf32>
    %v7993 = stablehlo.add %v7991, %v7992 : tensor<512xf32>
    %v7994 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7995 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7996 = stablehlo.multiply %v7994, %s4b0g2v : tensor<512xf32>
    %v7997 = stablehlo.multiply %v1409, %v1409 : tensor<512xf32>
    %v7998 = stablehlo.multiply %v7995, %v7997 : tensor<512xf32>
    %v7999 = stablehlo.add %v7996, %v7998 : tensor<512xf32>
    %v8000 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8001 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8002 = stablehlo.multiply %v8000, %s4b0g2m : tensor<512xf32>
    %v8003 = stablehlo.multiply %v8001, %v1409 : tensor<512xf32>
    %v8004 = stablehlo.add %v8002, %v8003 : tensor<512xf32>
    %v8005 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8006 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8007 = stablehlo.multiply %v8005, %s4b0g2v : tensor<512xf32>
    %v8008 = stablehlo.multiply %v1409, %v1409 : tensor<512xf32>
    %v8009 = stablehlo.multiply %v8006, %v8008 : tensor<512xf32>
    %v8010 = stablehlo.add %v8007, %v8009 : tensor<512xf32>
    %v8011 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8012 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8013 = stablehlo.divide %v8004, %v8011 : tensor<512xf32>
    %v8014 = stablehlo.divide %v8010, %v8012 : tensor<512xf32>
    %v8015 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8016 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8017 = stablehlo.sqrt %v8014 : tensor<512xf32>
    %v8018 = stablehlo.add %v8017, %v8016 : tensor<512xf32>
    %v8019 = stablehlo.divide %v8013, %v8018 : tensor<512xf32>
    %v8020 = stablehlo.multiply %v8015, %v8019 : tensor<512xf32>
    %v8021 = stablehlo.subtract %s4b0g2, %v8020 : tensor<512xf32>
    %v8022 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8023 = stablehlo.multiply %v8022, %v8015 : tensor<512xf32>
    %v8024 = stablehlo.multiply %v8023, %s4b0g2 : tensor<512xf32>
    %v8025 = stablehlo.subtract %v8021, %v8024 : tensor<512xf32>
    %v8026 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8027 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8028 = stablehlo.multiply %v8026, %s4b0bt2m : tensor<512xf32>
    %v8029 = stablehlo.multiply %v8027, %v1412 : tensor<512xf32>
    %v8030 = stablehlo.add %v8028, %v8029 : tensor<512xf32>
    %v8031 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8032 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8033 = stablehlo.multiply %v8031, %s4b0bt2v : tensor<512xf32>
    %v8034 = stablehlo.multiply %v1412, %v1412 : tensor<512xf32>
    %v8035 = stablehlo.multiply %v8032, %v8034 : tensor<512xf32>
    %v8036 = stablehlo.add %v8033, %v8035 : tensor<512xf32>
    %v8037 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8038 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8039 = stablehlo.multiply %v8037, %s4b0bt2m : tensor<512xf32>
    %v8040 = stablehlo.multiply %v8038, %v1412 : tensor<512xf32>
    %v8041 = stablehlo.add %v8039, %v8040 : tensor<512xf32>
    %v8042 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8043 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8044 = stablehlo.multiply %v8042, %s4b0bt2v : tensor<512xf32>
    %v8045 = stablehlo.multiply %v1412, %v1412 : tensor<512xf32>
    %v8046 = stablehlo.multiply %v8043, %v8045 : tensor<512xf32>
    %v8047 = stablehlo.add %v8044, %v8046 : tensor<512xf32>
    %v8048 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8049 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8050 = stablehlo.divide %v8041, %v8048 : tensor<512xf32>
    %v8051 = stablehlo.divide %v8047, %v8049 : tensor<512xf32>
    %v8052 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8053 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8054 = stablehlo.sqrt %v8051 : tensor<512xf32>
    %v8055 = stablehlo.add %v8054, %v8053 : tensor<512xf32>
    %v8056 = stablehlo.divide %v8050, %v8055 : tensor<512xf32>
    %v8057 = stablehlo.multiply %v8052, %v8056 : tensor<512xf32>
    %v8058 = stablehlo.subtract %s4b0bt2, %v8057 : tensor<512xf32>
    %v8059 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8060 = stablehlo.multiply %v8059, %v8052 : tensor<512xf32>
    %v8061 = stablehlo.multiply %v8060, %s4b0bt2 : tensor<512xf32>
    %v8062 = stablehlo.subtract %v8058, %v8061 : tensor<512xf32>
    %v8063 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8064 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8065 = stablehlo.multiply %v8063, %s4b1W1m : tensor<512x512x3x3xf32>
    %v8066 = stablehlo.multiply %v8064, %v1224 : tensor<512x512x3x3xf32>
    %v8067 = stablehlo.add %v8065, %v8066 : tensor<512x512x3x3xf32>
    %v8068 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8069 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8070 = stablehlo.multiply %v8068, %s4b1W1v : tensor<512x512x3x3xf32>
    %v8071 = stablehlo.multiply %v1224, %v1224 : tensor<512x512x3x3xf32>
    %v8072 = stablehlo.multiply %v8069, %v8071 : tensor<512x512x3x3xf32>
    %v8073 = stablehlo.add %v8070, %v8072 : tensor<512x512x3x3xf32>
    %v8074 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8075 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8076 = stablehlo.multiply %v8074, %s4b1W1m : tensor<512x512x3x3xf32>
    %v8077 = stablehlo.multiply %v8075, %v1224 : tensor<512x512x3x3xf32>
    %v8078 = stablehlo.add %v8076, %v8077 : tensor<512x512x3x3xf32>
    %v8079 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8080 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8081 = stablehlo.multiply %v8079, %s4b1W1v : tensor<512x512x3x3xf32>
    %v8082 = stablehlo.multiply %v1224, %v1224 : tensor<512x512x3x3xf32>
    %v8083 = stablehlo.multiply %v8080, %v8082 : tensor<512x512x3x3xf32>
    %v8084 = stablehlo.add %v8081, %v8083 : tensor<512x512x3x3xf32>
    %v8085 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8086 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8087 = stablehlo.divide %v8078, %v8085 : tensor<512x512x3x3xf32>
    %v8088 = stablehlo.divide %v8084, %v8086 : tensor<512x512x3x3xf32>
    %v8089 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8090 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8091 = stablehlo.sqrt %v8088 : tensor<512x512x3x3xf32>
    %v8092 = stablehlo.add %v8091, %v8090 : tensor<512x512x3x3xf32>
    %v8093 = stablehlo.divide %v8087, %v8092 : tensor<512x512x3x3xf32>
    %v8094 = stablehlo.multiply %v8089, %v8093 : tensor<512x512x3x3xf32>
    %v8095 = stablehlo.subtract %s4b1W1, %v8094 : tensor<512x512x3x3xf32>
    %v8096 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8097 = stablehlo.multiply %v8096, %v8089 : tensor<512x512x3x3xf32>
    %v8098 = stablehlo.multiply %v8097, %s4b1W1 : tensor<512x512x3x3xf32>
    %v8099 = stablehlo.subtract %v8095, %v8098 : tensor<512x512x3x3xf32>
    %v8100 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8101 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8102 = stablehlo.multiply %v8100, %s4b1g1m : tensor<512xf32>
    %v8103 = stablehlo.multiply %v8101, %v1242 : tensor<512xf32>
    %v8104 = stablehlo.add %v8102, %v8103 : tensor<512xf32>
    %v8105 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8106 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8107 = stablehlo.multiply %v8105, %s4b1g1v : tensor<512xf32>
    %v8108 = stablehlo.multiply %v1242, %v1242 : tensor<512xf32>
    %v8109 = stablehlo.multiply %v8106, %v8108 : tensor<512xf32>
    %v8110 = stablehlo.add %v8107, %v8109 : tensor<512xf32>
    %v8111 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8112 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8113 = stablehlo.multiply %v8111, %s4b1g1m : tensor<512xf32>
    %v8114 = stablehlo.multiply %v8112, %v1242 : tensor<512xf32>
    %v8115 = stablehlo.add %v8113, %v8114 : tensor<512xf32>
    %v8116 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8117 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8118 = stablehlo.multiply %v8116, %s4b1g1v : tensor<512xf32>
    %v8119 = stablehlo.multiply %v1242, %v1242 : tensor<512xf32>
    %v8120 = stablehlo.multiply %v8117, %v8119 : tensor<512xf32>
    %v8121 = stablehlo.add %v8118, %v8120 : tensor<512xf32>
    %v8122 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8123 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8124 = stablehlo.divide %v8115, %v8122 : tensor<512xf32>
    %v8125 = stablehlo.divide %v8121, %v8123 : tensor<512xf32>
    %v8126 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8127 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8128 = stablehlo.sqrt %v8125 : tensor<512xf32>
    %v8129 = stablehlo.add %v8128, %v8127 : tensor<512xf32>
    %v8130 = stablehlo.divide %v8124, %v8129 : tensor<512xf32>
    %v8131 = stablehlo.multiply %v8126, %v8130 : tensor<512xf32>
    %v8132 = stablehlo.subtract %s4b1g1, %v8131 : tensor<512xf32>
    %v8133 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8134 = stablehlo.multiply %v8133, %v8126 : tensor<512xf32>
    %v8135 = stablehlo.multiply %v8134, %s4b1g1 : tensor<512xf32>
    %v8136 = stablehlo.subtract %v8132, %v8135 : tensor<512xf32>
    %v8137 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8138 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8139 = stablehlo.multiply %v8137, %s4b1bt1m : tensor<512xf32>
    %v8140 = stablehlo.multiply %v8138, %v1245 : tensor<512xf32>
    %v8141 = stablehlo.add %v8139, %v8140 : tensor<512xf32>
    %v8142 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8143 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8144 = stablehlo.multiply %v8142, %s4b1bt1v : tensor<512xf32>
    %v8145 = stablehlo.multiply %v1245, %v1245 : tensor<512xf32>
    %v8146 = stablehlo.multiply %v8143, %v8145 : tensor<512xf32>
    %v8147 = stablehlo.add %v8144, %v8146 : tensor<512xf32>
    %v8148 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8149 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8150 = stablehlo.multiply %v8148, %s4b1bt1m : tensor<512xf32>
    %v8151 = stablehlo.multiply %v8149, %v1245 : tensor<512xf32>
    %v8152 = stablehlo.add %v8150, %v8151 : tensor<512xf32>
    %v8153 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8154 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8155 = stablehlo.multiply %v8153, %s4b1bt1v : tensor<512xf32>
    %v8156 = stablehlo.multiply %v1245, %v1245 : tensor<512xf32>
    %v8157 = stablehlo.multiply %v8154, %v8156 : tensor<512xf32>
    %v8158 = stablehlo.add %v8155, %v8157 : tensor<512xf32>
    %v8159 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8160 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8161 = stablehlo.divide %v8152, %v8159 : tensor<512xf32>
    %v8162 = stablehlo.divide %v8158, %v8160 : tensor<512xf32>
    %v8163 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8164 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8165 = stablehlo.sqrt %v8162 : tensor<512xf32>
    %v8166 = stablehlo.add %v8165, %v8164 : tensor<512xf32>
    %v8167 = stablehlo.divide %v8161, %v8166 : tensor<512xf32>
    %v8168 = stablehlo.multiply %v8163, %v8167 : tensor<512xf32>
    %v8169 = stablehlo.subtract %s4b1bt1, %v8168 : tensor<512xf32>
    %v8170 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8171 = stablehlo.multiply %v8170, %v8163 : tensor<512xf32>
    %v8172 = stablehlo.multiply %v8171, %s4b1bt1 : tensor<512xf32>
    %v8173 = stablehlo.subtract %v8169, %v8172 : tensor<512xf32>
    %v8174 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8175 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8176 = stablehlo.multiply %v8174, %s4b1W2m : tensor<512x512x3x3xf32>
    %v8177 = stablehlo.multiply %v8175, %v1251 : tensor<512x512x3x3xf32>
    %v8178 = stablehlo.add %v8176, %v8177 : tensor<512x512x3x3xf32>
    %v8179 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8180 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8181 = stablehlo.multiply %v8179, %s4b1W2v : tensor<512x512x3x3xf32>
    %v8182 = stablehlo.multiply %v1251, %v1251 : tensor<512x512x3x3xf32>
    %v8183 = stablehlo.multiply %v8180, %v8182 : tensor<512x512x3x3xf32>
    %v8184 = stablehlo.add %v8181, %v8183 : tensor<512x512x3x3xf32>
    %v8185 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8186 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8187 = stablehlo.multiply %v8185, %s4b1W2m : tensor<512x512x3x3xf32>
    %v8188 = stablehlo.multiply %v8186, %v1251 : tensor<512x512x3x3xf32>
    %v8189 = stablehlo.add %v8187, %v8188 : tensor<512x512x3x3xf32>
    %v8190 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8191 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8192 = stablehlo.multiply %v8190, %s4b1W2v : tensor<512x512x3x3xf32>
    %v8193 = stablehlo.multiply %v1251, %v1251 : tensor<512x512x3x3xf32>
    %v8194 = stablehlo.multiply %v8191, %v8193 : tensor<512x512x3x3xf32>
    %v8195 = stablehlo.add %v8192, %v8194 : tensor<512x512x3x3xf32>
    %v8196 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8197 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8198 = stablehlo.divide %v8189, %v8196 : tensor<512x512x3x3xf32>
    %v8199 = stablehlo.divide %v8195, %v8197 : tensor<512x512x3x3xf32>
    %v8200 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8201 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8202 = stablehlo.sqrt %v8199 : tensor<512x512x3x3xf32>
    %v8203 = stablehlo.add %v8202, %v8201 : tensor<512x512x3x3xf32>
    %v8204 = stablehlo.divide %v8198, %v8203 : tensor<512x512x3x3xf32>
    %v8205 = stablehlo.multiply %v8200, %v8204 : tensor<512x512x3x3xf32>
    %v8206 = stablehlo.subtract %s4b1W2, %v8205 : tensor<512x512x3x3xf32>
    %v8207 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8208 = stablehlo.multiply %v8207, %v8200 : tensor<512x512x3x3xf32>
    %v8209 = stablehlo.multiply %v8208, %s4b1W2 : tensor<512x512x3x3xf32>
    %v8210 = stablehlo.subtract %v8206, %v8209 : tensor<512x512x3x3xf32>
    %v8211 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8212 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8213 = stablehlo.multiply %v8211, %s4b1g2m : tensor<512xf32>
    %v8214 = stablehlo.multiply %v8212, %v1269 : tensor<512xf32>
    %v8215 = stablehlo.add %v8213, %v8214 : tensor<512xf32>
    %v8216 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8217 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8218 = stablehlo.multiply %v8216, %s4b1g2v : tensor<512xf32>
    %v8219 = stablehlo.multiply %v1269, %v1269 : tensor<512xf32>
    %v8220 = stablehlo.multiply %v8217, %v8219 : tensor<512xf32>
    %v8221 = stablehlo.add %v8218, %v8220 : tensor<512xf32>
    %v8222 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8223 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8224 = stablehlo.multiply %v8222, %s4b1g2m : tensor<512xf32>
    %v8225 = stablehlo.multiply %v8223, %v1269 : tensor<512xf32>
    %v8226 = stablehlo.add %v8224, %v8225 : tensor<512xf32>
    %v8227 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8228 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8229 = stablehlo.multiply %v8227, %s4b1g2v : tensor<512xf32>
    %v8230 = stablehlo.multiply %v1269, %v1269 : tensor<512xf32>
    %v8231 = stablehlo.multiply %v8228, %v8230 : tensor<512xf32>
    %v8232 = stablehlo.add %v8229, %v8231 : tensor<512xf32>
    %v8233 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8234 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8235 = stablehlo.divide %v8226, %v8233 : tensor<512xf32>
    %v8236 = stablehlo.divide %v8232, %v8234 : tensor<512xf32>
    %v8237 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8238 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8239 = stablehlo.sqrt %v8236 : tensor<512xf32>
    %v8240 = stablehlo.add %v8239, %v8238 : tensor<512xf32>
    %v8241 = stablehlo.divide %v8235, %v8240 : tensor<512xf32>
    %v8242 = stablehlo.multiply %v8237, %v8241 : tensor<512xf32>
    %v8243 = stablehlo.subtract %s4b1g2, %v8242 : tensor<512xf32>
    %v8244 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8245 = stablehlo.multiply %v8244, %v8237 : tensor<512xf32>
    %v8246 = stablehlo.multiply %v8245, %s4b1g2 : tensor<512xf32>
    %v8247 = stablehlo.subtract %v8243, %v8246 : tensor<512xf32>
    %v8248 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8249 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8250 = stablehlo.multiply %v8248, %s4b1bt2m : tensor<512xf32>
    %v8251 = stablehlo.multiply %v8249, %v1272 : tensor<512xf32>
    %v8252 = stablehlo.add %v8250, %v8251 : tensor<512xf32>
    %v8253 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8254 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8255 = stablehlo.multiply %v8253, %s4b1bt2v : tensor<512xf32>
    %v8256 = stablehlo.multiply %v1272, %v1272 : tensor<512xf32>
    %v8257 = stablehlo.multiply %v8254, %v8256 : tensor<512xf32>
    %v8258 = stablehlo.add %v8255, %v8257 : tensor<512xf32>
    %v8259 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8260 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8261 = stablehlo.multiply %v8259, %s4b1bt2m : tensor<512xf32>
    %v8262 = stablehlo.multiply %v8260, %v1272 : tensor<512xf32>
    %v8263 = stablehlo.add %v8261, %v8262 : tensor<512xf32>
    %v8264 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8265 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8266 = stablehlo.multiply %v8264, %s4b1bt2v : tensor<512xf32>
    %v8267 = stablehlo.multiply %v1272, %v1272 : tensor<512xf32>
    %v8268 = stablehlo.multiply %v8265, %v8267 : tensor<512xf32>
    %v8269 = stablehlo.add %v8266, %v8268 : tensor<512xf32>
    %v8270 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8271 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8272 = stablehlo.divide %v8263, %v8270 : tensor<512xf32>
    %v8273 = stablehlo.divide %v8269, %v8271 : tensor<512xf32>
    %v8274 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8275 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8276 = stablehlo.sqrt %v8273 : tensor<512xf32>
    %v8277 = stablehlo.add %v8276, %v8275 : tensor<512xf32>
    %v8278 = stablehlo.divide %v8272, %v8277 : tensor<512xf32>
    %v8279 = stablehlo.multiply %v8274, %v8278 : tensor<512xf32>
    %v8280 = stablehlo.subtract %s4b1bt2, %v8279 : tensor<512xf32>
    %v8281 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8282 = stablehlo.multiply %v8281, %v8274 : tensor<512xf32>
    %v8283 = stablehlo.multiply %v8282, %s4b1bt2 : tensor<512xf32>
    %v8284 = stablehlo.subtract %v8280, %v8283 : tensor<512xf32>
    %v8285 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8286 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8287 = stablehlo.multiply %v8285, %Wdm : tensor<512x10xf32>
    %v8288 = stablehlo.multiply %v8286, %v1126 : tensor<512x10xf32>
    %v8289 = stablehlo.add %v8287, %v8288 : tensor<512x10xf32>
    %v8290 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8291 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8292 = stablehlo.multiply %v8290, %Wdv : tensor<512x10xf32>
    %v8293 = stablehlo.multiply %v1126, %v1126 : tensor<512x10xf32>
    %v8294 = stablehlo.multiply %v8291, %v8293 : tensor<512x10xf32>
    %v8295 = stablehlo.add %v8292, %v8294 : tensor<512x10xf32>
    %v8296 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8297 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8298 = stablehlo.multiply %v8296, %Wdm : tensor<512x10xf32>
    %v8299 = stablehlo.multiply %v8297, %v1126 : tensor<512x10xf32>
    %v8300 = stablehlo.add %v8298, %v8299 : tensor<512x10xf32>
    %v8301 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8302 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8303 = stablehlo.multiply %v8301, %Wdv : tensor<512x10xf32>
    %v8304 = stablehlo.multiply %v1126, %v1126 : tensor<512x10xf32>
    %v8305 = stablehlo.multiply %v8302, %v8304 : tensor<512x10xf32>
    %v8306 = stablehlo.add %v8303, %v8305 : tensor<512x10xf32>
    %v8307 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8308 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8309 = stablehlo.divide %v8300, %v8307 : tensor<512x10xf32>
    %v8310 = stablehlo.divide %v8306, %v8308 : tensor<512x10xf32>
    %v8311 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8312 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8313 = stablehlo.sqrt %v8310 : tensor<512x10xf32>
    %v8314 = stablehlo.add %v8313, %v8312 : tensor<512x10xf32>
    %v8315 = stablehlo.divide %v8309, %v8314 : tensor<512x10xf32>
    %v8316 = stablehlo.multiply %v8311, %v8315 : tensor<512x10xf32>
    %v8317 = stablehlo.subtract %Wd, %v8316 : tensor<512x10xf32>
    %v8318 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8319 = stablehlo.multiply %v8318, %v8311 : tensor<512x10xf32>
    %v8320 = stablehlo.multiply %v8319, %Wd : tensor<512x10xf32>
    %v8321 = stablehlo.subtract %v8317, %v8320 : tensor<512x10xf32>
    %v8322 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8323 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8324 = stablehlo.multiply %v8322, %bdm : tensor<10xf32>
    %v8325 = stablehlo.multiply %v8323, %v1128 : tensor<10xf32>
    %v8326 = stablehlo.add %v8324, %v8325 : tensor<10xf32>
    %v8327 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8328 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8329 = stablehlo.multiply %v8327, %bdv : tensor<10xf32>
    %v8330 = stablehlo.multiply %v1128, %v1128 : tensor<10xf32>
    %v8331 = stablehlo.multiply %v8328, %v8330 : tensor<10xf32>
    %v8332 = stablehlo.add %v8329, %v8331 : tensor<10xf32>
    %v8333 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8334 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8335 = stablehlo.multiply %v8333, %bdm : tensor<10xf32>
    %v8336 = stablehlo.multiply %v8334, %v1128 : tensor<10xf32>
    %v8337 = stablehlo.add %v8335, %v8336 : tensor<10xf32>
    %v8338 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8339 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8340 = stablehlo.multiply %v8338, %bdv : tensor<10xf32>
    %v8341 = stablehlo.multiply %v1128, %v1128 : tensor<10xf32>
    %v8342 = stablehlo.multiply %v8339, %v8341 : tensor<10xf32>
    %v8343 = stablehlo.add %v8340, %v8342 : tensor<10xf32>
    %v8344 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8345 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8346 = stablehlo.divide %v8337, %v8344 : tensor<10xf32>
    %v8347 = stablehlo.divide %v8343, %v8345 : tensor<10xf32>
    %v8348 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8349 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8350 = stablehlo.sqrt %v8347 : tensor<10xf32>
    %v8351 = stablehlo.add %v8350, %v8349 : tensor<10xf32>
    %v8352 = stablehlo.divide %v8346, %v8351 : tensor<10xf32>
    %v8353 = stablehlo.multiply %v8348, %v8352 : tensor<10xf32>
    %v8354 = stablehlo.subtract %bd, %v8353 : tensor<10xf32>
    %v8355 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8356 = stablehlo.multiply %v8355, %v8348 : tensor<10xf32>
    %v8357 = stablehlo.multiply %v8356, %bd : tensor<10xf32>
    %v8358 = stablehlo.subtract %v8354, %v8357 : tensor<10xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1114 : tensor<128x10xf32>
    %lohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %lt1s = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %llsr = stablehlo.reduce(%llog init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %lomac = stablehlo.constant dense<0.900000> : tensor<128xf32>
    %laKc = stablehlo.constant dense<0.010000> : tensor<128xf32>
    %llt1 = stablehlo.multiply %lomac, %lt1s : tensor<128xf32>
    %llt2 = stablehlo.multiply %laKc, %llsr : tensor<128xf32>
    %llpe = stablehlo.add %llt1, %llt2 : tensor<128xf32>
    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    %lbfc = stablehlo.constant dense<128.0> : tensor<f32>
    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>
    %loss = stablehlo.negate %lossm : tensor<f32>
    return %v4325, %v4362, %v4399, %v4436, %v4473, %v4510, %v4547, %v4584, %v4621, %v4658, %v4695, %v4732, %v4769, %v4806, %v4843, %v4880, %v4917, %v4954, %v4991, %v5028, %v5065, %v5102, %v5139, %v5176, %v5213, %v5250, %v5287, %v5324, %v5361, %v5398, %v5435, %v5472, %v5509, %v5546, %v5583, %v5620, %v5657, %v5694, %v5731, %v5768, %v5805, %v5842, %v5879, %v5916, %v5953, %v5990, %v6027, %v6064, %v6101, %v6138, %v6175, %v6212, %v6249, %v6286, %v6323, %v6360, %v6397, %v6434, %v6471, %v6508, %v6545, %v6582, %v6619, %v6656, %v6693, %v6730, %v6767, %v6804, %v6841, %v6878, %v6915, %v6952, %v6989, %v7026, %v7063, %v7100, %v7137, %v7174, %v7211, %v7248, %v7285, %v7322, %v7359, %v7396, %v7433, %v7470, %v7507, %v7544, %v7581, %v7618, %v7655, %v7692, %v7729, %v7766, %v7803, %v7840, %v7877, %v7914, %v7951, %v7988, %v8025, %v8062, %v8099, %v8136, %v8173, %v8210, %v8247, %v8284, %v8321, %v8358, %v4293, %v4330, %v4367, %v4404, %v4441, %v4478, %v4515, %v4552, %v4589, %v4626, %v4663, %v4700, %v4737, %v4774, %v4811, %v4848, %v4885, %v4922, %v4959, %v4996, %v5033, %v5070, %v5107, %v5144, %v5181, %v5218, %v5255, %v5292, %v5329, %v5366, %v5403, %v5440, %v5477, %v5514, %v5551, %v5588, %v5625, %v5662, %v5699, %v5736, %v5773, %v5810, %v5847, %v5884, %v5921, %v5958, %v5995, %v6032, %v6069, %v6106, %v6143, %v6180, %v6217, %v6254, %v6291, %v6328, %v6365, %v6402, %v6439, %v6476, %v6513, %v6550, %v6587, %v6624, %v6661, %v6698, %v6735, %v6772, %v6809, %v6846, %v6883, %v6920, %v6957, %v6994, %v7031, %v7068, %v7105, %v7142, %v7179, %v7216, %v7253, %v7290, %v7327, %v7364, %v7401, %v7438, %v7475, %v7512, %v7549, %v7586, %v7623, %v7660, %v7697, %v7734, %v7771, %v7808, %v7845, %v7882, %v7919, %v7956, %v7993, %v8030, %v8067, %v8104, %v8141, %v8178, %v8215, %v8252, %v8289, %v8326, %v4299, %v4336, %v4373, %v4410, %v4447, %v4484, %v4521, %v4558, %v4595, %v4632, %v4669, %v4706, %v4743, %v4780, %v4817, %v4854, %v4891, %v4928, %v4965, %v5002, %v5039, %v5076, %v5113, %v5150, %v5187, %v5224, %v5261, %v5298, %v5335, %v5372, %v5409, %v5446, %v5483, %v5520, %v5557, %v5594, %v5631, %v5668, %v5705, %v5742, %v5779, %v5816, %v5853, %v5890, %v5927, %v5964, %v6001, %v6038, %v6075, %v6112, %v6149, %v6186, %v6223, %v6260, %v6297, %v6334, %v6371, %v6408, %v6445, %v6482, %v6519, %v6556, %v6593, %v6630, %v6667, %v6704, %v6741, %v6778, %v6815, %v6852, %v6889, %v6926, %v6963, %v7000, %v7037, %v7074, %v7111, %v7148, %v7185, %v7222, %v7259, %v7296, %v7333, %v7370, %v7407, %v7444, %v7481, %v7518, %v7555, %v7592, %v7629, %v7666, %v7703, %v7740, %v7777, %v7814, %v7851, %v7888, %v7925, %v7962, %v7999, %v8036, %v8073, %v8110, %v8147, %v8184, %v8221, %v8258, %v8295, %v8332, %loss, %bc1, %bc2, %v3717, %v3728, %v3733, %v3744, %v3749, %v3760, %v3765, %v3776, %v3781, %v3792, %v3797, %v3808, %v3813, %v3824, %v3829, %v3840, %v3845, %v3856, %v3861, %v3872, %v3877, %v3888, %v3893, %v3904, %v3909, %v3920, %v3925, %v3936, %v3941, %v3952, %v3957, %v3968, %v3973, %v3984, %v3989, %v4000, %v4005, %v4016, %v4021, %v4032, %v4037, %v4048, %v4053, %v4064, %v4069, %v4080, %v4085, %v4096, %v4101, %v4112, %v4117, %v4128, %v4133, %v4144, %v4149, %v4160, %v4165, %v4176, %v4181, %v4192, %v4197, %v4208, %v4213, %v4224, %v4229, %v4240, %v4245, %v4256, %v4261, %v4272, %v4277, %v4288 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}

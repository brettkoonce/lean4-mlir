module @m {
  func.func @resnet34_adamdp128_train_step(%x: tensor<128x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x1x1xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x1x1xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x1x1xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x3x3xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b1W1m: tensor<64x64x3x3xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b2W1m: tensor<64x64x3x3xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %d2W1m: tensor<128x64x3x3xf32>, %d2g1m: tensor<128xf32>, %d2bt1m: tensor<128xf32>, %d2W2m: tensor<128x128x3x3xf32>, %d2g2m: tensor<128xf32>, %d2bt2m: tensor<128xf32>, %d2Wpm: tensor<128x64x1x1xf32>, %d2gpm: tensor<128xf32>, %d2btpm: tensor<128xf32>, %s2b0W1m: tensor<128x128x3x3xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b1W1m: tensor<128x128x3x3xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b2W1m: tensor<128x128x3x3xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %d3W1m: tensor<256x128x3x3xf32>, %d3g1m: tensor<256xf32>, %d3bt1m: tensor<256xf32>, %d3W2m: tensor<256x256x3x3xf32>, %d3g2m: tensor<256xf32>, %d3bt2m: tensor<256xf32>, %d3Wpm: tensor<256x128x1x1xf32>, %d3gpm: tensor<256xf32>, %d3btpm: tensor<256xf32>, %s3b0W1m: tensor<256x256x3x3xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b1W1m: tensor<256x256x3x3xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b2W1m: tensor<256x256x3x3xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b3W1m: tensor<256x256x3x3xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b4W1m: tensor<256x256x3x3xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %d4W1m: tensor<512x256x3x3xf32>, %d4g1m: tensor<512xf32>, %d4bt1m: tensor<512xf32>, %d4W2m: tensor<512x512x3x3xf32>, %d4g2m: tensor<512xf32>, %d4bt2m: tensor<512xf32>, %d4Wpm: tensor<512x256x1x1xf32>, %d4gpm: tensor<512xf32>, %d4btpm: tensor<512xf32>, %s4b0W1m: tensor<512x512x3x3xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b1W1m: tensor<512x512x3x3xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %Wdm: tensor<512x10xf32>, %bdm: tensor<10xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x3x3xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b1W1v: tensor<64x64x3x3xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b2W1v: tensor<64x64x3x3xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %d2W1v: tensor<128x64x3x3xf32>, %d2g1v: tensor<128xf32>, %d2bt1v: tensor<128xf32>, %d2W2v: tensor<128x128x3x3xf32>, %d2g2v: tensor<128xf32>, %d2bt2v: tensor<128xf32>, %d2Wpv: tensor<128x64x1x1xf32>, %d2gpv: tensor<128xf32>, %d2btpv: tensor<128xf32>, %s2b0W1v: tensor<128x128x3x3xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b1W1v: tensor<128x128x3x3xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b2W1v: tensor<128x128x3x3xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %d3W1v: tensor<256x128x3x3xf32>, %d3g1v: tensor<256xf32>, %d3bt1v: tensor<256xf32>, %d3W2v: tensor<256x256x3x3xf32>, %d3g2v: tensor<256xf32>, %d3bt2v: tensor<256xf32>, %d3Wpv: tensor<256x128x1x1xf32>, %d3gpv: tensor<256xf32>, %d3btpv: tensor<256xf32>, %s3b0W1v: tensor<256x256x3x3xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b1W1v: tensor<256x256x3x3xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b2W1v: tensor<256x256x3x3xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b3W1v: tensor<256x256x3x3xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b4W1v: tensor<256x256x3x3xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %d4W1v: tensor<512x256x3x3xf32>, %d4g1v: tensor<512xf32>, %d4bt1v: tensor<512xf32>, %d4W2v: tensor<512x512x3x3xf32>, %d4g2v: tensor<512xf32>, %d4bt2v: tensor<512xf32>, %d4Wpv: tensor<512x256x1x1xf32>, %d4gpv: tensor<512xf32>, %d4btpv: tensor<512xf32>, %s4b0W1v: tensor<512x512x3x3xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b1W1v: tensor<512x512x3x3xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %Wdv: tensor<512x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %d2n1mui: tensor<128xf32>, %d2n1vari: tensor<128xf32>, %d2n2mui: tensor<128xf32>, %d2n2vari: tensor<128xf32>, %d2npmui: tensor<128xf32>, %d2npvari: tensor<128xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %d3n1mui: tensor<256xf32>, %d3n1vari: tensor<256xf32>, %d3n2mui: tensor<256xf32>, %d3n2vari: tensor<256xf32>, %d3npmui: tensor<256xf32>, %d3npvari: tensor<256xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %d4n1mui: tensor<512xf32>, %d4n1vari: tensor<512xf32>, %d4n2mui: tensor<512xf32>, %d4n2vari: tensor<512xf32>, %d4npmui: tensor<512xf32>, %d4npvari: tensor<512xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %onehot: tensor<128x10xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) {
    // ── ResNet-34 batch-BN AdamW train step, DATA-PARALLEL over 2 replicas ──
    // Every line is pretty(verified AST node), the per-parameter `%arsum*` all_reduce /
    // `%armean*` blocks included: pretty(allReduceMeanF), whose den is the replica MEAN of
    // the per-replica gradient nodes (4d piece 2). BatchNorm is SYNCHRONISED: every BN
    // layer all-reduces its mu, then var_r + (mu_r - mu)^2 (bnBatchVarAtB, Chan's parallel
    // variance), before normalising with the global [mu | var] (bnSyncF); its
    // backward all-reduces the two dy-reductions (bnSyncDyStatsB -> bnSyncBack), and the gamma
    // gradient reads the same global x-hat (bnSyncGammaGradB). Each replica therefore computes
    // its shard of the GLOBAL-batch function, and this step IS the single-device step at the
    // global batch N x b: proved as ResNet34SyncTieB.r34_net_syncTiedB (every all-reduced
    // gradient) and StableHLO.resnet34FwdGraphSyncFull_shard (the forward), both in
    // LeanMlir/Proofs/Nets/ResNet/ (planning/global_bn_verified.md).
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
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
    %v7 = stablehlo.constant dense<1605632.0> : tensor<64xf32>
    %v8 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v9 = stablehlo.divide %v8, %v7 : tensor<64xf32>
    %arsumsgmu = "stablehlo.all_reduce"(%v9) ({
    ^bb0(%arasgmu: tensor<f32>, %arbsgmu: tensor<f32>):
      %araddsgmu = stablehlo.add %arasgmu, %arbsgmu : tensor<f32>
      stablehlo.return %araddsgmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsgmu = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeansgmu = stablehlo.divide %arsumsgmu, %arnsgmu : tensor<64xf32>
    %v10 = stablehlo.reshape %v4 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v11 = stablehlo.constant dense<0.0> : tensor<f32>
    %v12 = stablehlo.constant dense<1605632.0> : tensor<64xf32>
    %v13 = stablehlo.reduce(%v10 init: %v11) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v14 = stablehlo.divide %v13, %v12 : tensor<64xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v16 = stablehlo.subtract %v10, %v15 : tensor<128x64x112x112xf32>
    %v17 = stablehlo.multiply %v16, %v16 : tensor<128x64x112x112xf32>
    %v18 = stablehlo.reduce(%v17 init: %v11) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
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
    %v24 = stablehlo.reshape %v4 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v25 = stablehlo.slice %v23 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v26 = stablehlo.slice %v23 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v27 = stablehlo.broadcast_in_dim %v25, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v28 = stablehlo.broadcast_in_dim %v26, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v29 = stablehlo.constant dense<1.0e-05> : tensor<128x64x112x112xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<128x64x112x112xf32>
    %v31 = stablehlo.rsqrt %v30 : tensor<128x64x112x112xf32>
    %v32 = stablehlo.subtract %v24, %v27 : tensor<128x64x112x112xf32>
    %v33 = stablehlo.multiply %v32, %v31 : tensor<128x64x112x112xf32>
    %v34 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v35 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v36 = stablehlo.multiply %v33, %v34 : tensor<128x64x112x112xf32>
    %v37 = stablehlo.add %v36, %v35 : tensor<128x64x112x112xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<128x64x112x112xf32>) -> tensor<128x802816xf32>
    %v39 = stablehlo.constant dense<0.0> : tensor<128x802816xf32>
    %v40 = stablehlo.maximum %v38, %v39 : tensor<128x802816xf32>
    %v41 = stablehlo.reshape %v40 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v42 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v43 = "stablehlo.reduce_window"(%v41, %v42) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<128x64x56x56xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v45 = stablehlo.reshape %v44 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v46 = stablehlo.convolution(%v45, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v47 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v48 = stablehlo.add %v46, %v47 : tensor<128x64x56x56xf32>
    %v49 = stablehlo.reshape %v48 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v50 = stablehlo.reshape %v49 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v51 = stablehlo.constant dense<0.0> : tensor<f32>
    %v52 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v53 = stablehlo.reduce(%v50 init: %v51) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v54 = stablehlo.divide %v53, %v52 : tensor<64xf32>
    %arsums1b0g1mu = "stablehlo.all_reduce"(%v54) ({
    ^bb0(%aras1b0g1mu: tensor<f32>, %arbs1b0g1mu: tensor<f32>):
      %aradds1b0g1mu = stablehlo.add %aras1b0g1mu, %arbs1b0g1mu : tensor<f32>
      stablehlo.return %aradds1b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1mu = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g1mu = stablehlo.divide %arsums1b0g1mu, %arns1b0g1mu : tensor<64xf32>
    %v55 = stablehlo.reshape %v49 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v56 = stablehlo.constant dense<0.0> : tensor<f32>
    %v57 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v58 = stablehlo.reduce(%v55 init: %v56) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v59 = stablehlo.divide %v58, %v57 : tensor<64xf32>
    %v60 = stablehlo.broadcast_in_dim %v59, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v61 = stablehlo.subtract %v55, %v60 : tensor<128x64x56x56xf32>
    %v62 = stablehlo.multiply %v61, %v61 : tensor<128x64x56x56xf32>
    %v63 = stablehlo.reduce(%v62 init: %v56) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
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
    %v69 = stablehlo.reshape %v49 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v70 = stablehlo.slice %v68 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v71 = stablehlo.slice %v68 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v72 = stablehlo.broadcast_in_dim %v70, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v73 = stablehlo.broadcast_in_dim %v71, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v74 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v75 = stablehlo.add %v73, %v74 : tensor<128x64x56x56xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<128x64x56x56xf32>
    %v77 = stablehlo.subtract %v69, %v72 : tensor<128x64x56x56xf32>
    %v78 = stablehlo.multiply %v77, %v76 : tensor<128x64x56x56xf32>
    %v79 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v80 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v81 = stablehlo.multiply %v78, %v79 : tensor<128x64x56x56xf32>
    %v82 = stablehlo.add %v81, %v80 : tensor<128x64x56x56xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v84 = stablehlo.constant dense<0.0> : tensor<128x200704xf32>
    %v85 = stablehlo.maximum %v83, %v84 : tensor<128x200704xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v87 = stablehlo.convolution(%v86, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v88 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v89 = stablehlo.add %v87, %v88 : tensor<128x64x56x56xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<f32>
    %v93 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v94 = stablehlo.reduce(%v91 init: %v92) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v95 = stablehlo.divide %v94, %v93 : tensor<64xf32>
    %arsums1b0g2mu = "stablehlo.all_reduce"(%v95) ({
    ^bb0(%aras1b0g2mu: tensor<f32>, %arbs1b0g2mu: tensor<f32>):
      %aradds1b0g2mu = stablehlo.add %aras1b0g2mu, %arbs1b0g2mu : tensor<f32>
      stablehlo.return %aradds1b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2mu = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g2mu = stablehlo.divide %arsums1b0g2mu, %arns1b0g2mu : tensor<64xf32>
    %v96 = stablehlo.reshape %v90 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v97 = stablehlo.constant dense<0.0> : tensor<f32>
    %v98 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v99 = stablehlo.reduce(%v96 init: %v97) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v100 = stablehlo.divide %v99, %v98 : tensor<64xf32>
    %v101 = stablehlo.broadcast_in_dim %v100, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v102 = stablehlo.subtract %v96, %v101 : tensor<128x64x56x56xf32>
    %v103 = stablehlo.multiply %v102, %v102 : tensor<128x64x56x56xf32>
    %v104 = stablehlo.reduce(%v103 init: %v97) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
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
    %v110 = stablehlo.reshape %v90 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v111 = stablehlo.slice %v109 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v112 = stablehlo.slice %v109 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v113 = stablehlo.broadcast_in_dim %v111, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v114 = stablehlo.broadcast_in_dim %v112, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v115 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v116 = stablehlo.add %v114, %v115 : tensor<128x64x56x56xf32>
    %v117 = stablehlo.rsqrt %v116 : tensor<128x64x56x56xf32>
    %v118 = stablehlo.subtract %v110, %v113 : tensor<128x64x56x56xf32>
    %v119 = stablehlo.multiply %v118, %v117 : tensor<128x64x56x56xf32>
    %v120 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v121 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v122 = stablehlo.multiply %v119, %v120 : tensor<128x64x56x56xf32>
    %v123 = stablehlo.add %v122, %v121 : tensor<128x64x56x56xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v125 = stablehlo.reshape %v124 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v126 = stablehlo.reshape %v44 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v127 = stablehlo.add %v125, %v126 : tensor<128x64x56x56xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v130 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v131 = stablehlo.maximum %v129, %v130 : tensor<128x64x56x56xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v134 = stablehlo.convolution(%v133, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v135 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v136 = stablehlo.add %v134, %v135 : tensor<128x64x56x56xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v139 = stablehlo.constant dense<0.0> : tensor<f32>
    %v140 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v141 = stablehlo.reduce(%v138 init: %v139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v142 = stablehlo.divide %v141, %v140 : tensor<64xf32>
    %arsums1b1g1mu = "stablehlo.all_reduce"(%v142) ({
    ^bb0(%aras1b1g1mu: tensor<f32>, %arbs1b1g1mu: tensor<f32>):
      %aradds1b1g1mu = stablehlo.add %aras1b1g1mu, %arbs1b1g1mu : tensor<f32>
      stablehlo.return %aradds1b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1mu = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g1mu = stablehlo.divide %arsums1b1g1mu, %arns1b1g1mu : tensor<64xf32>
    %v143 = stablehlo.reshape %v137 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v145 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v146 = stablehlo.reduce(%v143 init: %v144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v147 = stablehlo.divide %v146, %v145 : tensor<64xf32>
    %v148 = stablehlo.broadcast_in_dim %v147, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v149 = stablehlo.subtract %v143, %v148 : tensor<128x64x56x56xf32>
    %v150 = stablehlo.multiply %v149, %v149 : tensor<128x64x56x56xf32>
    %v151 = stablehlo.reduce(%v150 init: %v144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
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
    %v157 = stablehlo.reshape %v137 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v158 = stablehlo.slice %v156 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v159 = stablehlo.slice %v156 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v160 = stablehlo.broadcast_in_dim %v158, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v161 = stablehlo.broadcast_in_dim %v159, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v162 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v163 = stablehlo.add %v161, %v162 : tensor<128x64x56x56xf32>
    %v164 = stablehlo.rsqrt %v163 : tensor<128x64x56x56xf32>
    %v165 = stablehlo.subtract %v157, %v160 : tensor<128x64x56x56xf32>
    %v166 = stablehlo.multiply %v165, %v164 : tensor<128x64x56x56xf32>
    %v167 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v168 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v169 = stablehlo.multiply %v166, %v167 : tensor<128x64x56x56xf32>
    %v170 = stablehlo.add %v169, %v168 : tensor<128x64x56x56xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v172 = stablehlo.constant dense<0.0> : tensor<128x200704xf32>
    %v173 = stablehlo.maximum %v171, %v172 : tensor<128x200704xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v175 = stablehlo.convolution(%v174, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v176 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v177 = stablehlo.add %v175, %v176 : tensor<128x64x56x56xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v181 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v182 = stablehlo.reduce(%v179 init: %v180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v183 = stablehlo.divide %v182, %v181 : tensor<64xf32>
    %arsums1b1g2mu = "stablehlo.all_reduce"(%v183) ({
    ^bb0(%aras1b1g2mu: tensor<f32>, %arbs1b1g2mu: tensor<f32>):
      %aradds1b1g2mu = stablehlo.add %aras1b1g2mu, %arbs1b1g2mu : tensor<f32>
      stablehlo.return %aradds1b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2mu = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g2mu = stablehlo.divide %arsums1b1g2mu, %arns1b1g2mu : tensor<64xf32>
    %v184 = stablehlo.reshape %v178 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v186 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v187 = stablehlo.reduce(%v184 init: %v185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v188 = stablehlo.divide %v187, %v186 : tensor<64xf32>
    %v189 = stablehlo.broadcast_in_dim %v188, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v190 = stablehlo.subtract %v184, %v189 : tensor<128x64x56x56xf32>
    %v191 = stablehlo.multiply %v190, %v190 : tensor<128x64x56x56xf32>
    %v192 = stablehlo.reduce(%v191 init: %v185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
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
    %v198 = stablehlo.reshape %v178 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v199 = stablehlo.slice %v197 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v200 = stablehlo.slice %v197 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v201 = stablehlo.broadcast_in_dim %v199, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v202 = stablehlo.broadcast_in_dim %v200, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v203 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v204 = stablehlo.add %v202, %v203 : tensor<128x64x56x56xf32>
    %v205 = stablehlo.rsqrt %v204 : tensor<128x64x56x56xf32>
    %v206 = stablehlo.subtract %v198, %v201 : tensor<128x64x56x56xf32>
    %v207 = stablehlo.multiply %v206, %v205 : tensor<128x64x56x56xf32>
    %v208 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v209 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v210 = stablehlo.multiply %v207, %v208 : tensor<128x64x56x56xf32>
    %v211 = stablehlo.add %v210, %v209 : tensor<128x64x56x56xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v214 = stablehlo.reshape %v132 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v215 = stablehlo.add %v213, %v214 : tensor<128x64x56x56xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v218 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v219 = stablehlo.maximum %v217, %v218 : tensor<128x64x56x56xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v221 = stablehlo.reshape %v220 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v222 = stablehlo.convolution(%v221, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v223 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v224 = stablehlo.add %v222, %v223 : tensor<128x64x56x56xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v227 = stablehlo.constant dense<0.0> : tensor<f32>
    %v228 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v229 = stablehlo.reduce(%v226 init: %v227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v230 = stablehlo.divide %v229, %v228 : tensor<64xf32>
    %arsums1b2g1mu = "stablehlo.all_reduce"(%v230) ({
    ^bb0(%aras1b2g1mu: tensor<f32>, %arbs1b2g1mu: tensor<f32>):
      %aradds1b2g1mu = stablehlo.add %aras1b2g1mu, %arbs1b2g1mu : tensor<f32>
      stablehlo.return %aradds1b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1mu = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g1mu = stablehlo.divide %arsums1b2g1mu, %arns1b2g1mu : tensor<64xf32>
    %v231 = stablehlo.reshape %v225 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v233 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v234 = stablehlo.reduce(%v231 init: %v232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v235 = stablehlo.divide %v234, %v233 : tensor<64xf32>
    %v236 = stablehlo.broadcast_in_dim %v235, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v237 = stablehlo.subtract %v231, %v236 : tensor<128x64x56x56xf32>
    %v238 = stablehlo.multiply %v237, %v237 : tensor<128x64x56x56xf32>
    %v239 = stablehlo.reduce(%v238 init: %v232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
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
    %v245 = stablehlo.reshape %v225 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v246 = stablehlo.slice %v244 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v247 = stablehlo.slice %v244 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v248 = stablehlo.broadcast_in_dim %v246, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v249 = stablehlo.broadcast_in_dim %v247, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v250 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v251 = stablehlo.add %v249, %v250 : tensor<128x64x56x56xf32>
    %v252 = stablehlo.rsqrt %v251 : tensor<128x64x56x56xf32>
    %v253 = stablehlo.subtract %v245, %v248 : tensor<128x64x56x56xf32>
    %v254 = stablehlo.multiply %v253, %v252 : tensor<128x64x56x56xf32>
    %v255 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v256 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v257 = stablehlo.multiply %v254, %v255 : tensor<128x64x56x56xf32>
    %v258 = stablehlo.add %v257, %v256 : tensor<128x64x56x56xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v260 = stablehlo.constant dense<0.0> : tensor<128x200704xf32>
    %v261 = stablehlo.maximum %v259, %v260 : tensor<128x200704xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v263 = stablehlo.convolution(%v262, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v264 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v265 = stablehlo.add %v263, %v264 : tensor<128x64x56x56xf32>
    %v266 = stablehlo.reshape %v265 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v269 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v270 = stablehlo.reduce(%v267 init: %v268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v271 = stablehlo.divide %v270, %v269 : tensor<64xf32>
    %arsums1b2g2mu = "stablehlo.all_reduce"(%v271) ({
    ^bb0(%aras1b2g2mu: tensor<f32>, %arbs1b2g2mu: tensor<f32>):
      %aradds1b2g2mu = stablehlo.add %aras1b2g2mu, %arbs1b2g2mu : tensor<f32>
      stablehlo.return %aradds1b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2mu = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g2mu = stablehlo.divide %arsums1b2g2mu, %arns1b2g2mu : tensor<64xf32>
    %v272 = stablehlo.reshape %v266 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v274 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v275 = stablehlo.reduce(%v272 init: %v273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v276 = stablehlo.divide %v275, %v274 : tensor<64xf32>
    %v277 = stablehlo.broadcast_in_dim %v276, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v278 = stablehlo.subtract %v272, %v277 : tensor<128x64x56x56xf32>
    %v279 = stablehlo.multiply %v278, %v278 : tensor<128x64x56x56xf32>
    %v280 = stablehlo.reduce(%v279 init: %v273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
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
    %v286 = stablehlo.reshape %v266 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v287 = stablehlo.slice %v285 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v288 = stablehlo.slice %v285 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v289 = stablehlo.broadcast_in_dim %v287, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v290 = stablehlo.broadcast_in_dim %v288, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v291 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v292 = stablehlo.add %v290, %v291 : tensor<128x64x56x56xf32>
    %v293 = stablehlo.rsqrt %v292 : tensor<128x64x56x56xf32>
    %v294 = stablehlo.subtract %v286, %v289 : tensor<128x64x56x56xf32>
    %v295 = stablehlo.multiply %v294, %v293 : tensor<128x64x56x56xf32>
    %v296 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v297 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v298 = stablehlo.multiply %v295, %v296 : tensor<128x64x56x56xf32>
    %v299 = stablehlo.add %v298, %v297 : tensor<128x64x56x56xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v302 = stablehlo.reshape %v220 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v303 = stablehlo.add %v301, %v302 : tensor<128x64x56x56xf32>
    %v304 = stablehlo.reshape %v303 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v306 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v307 = stablehlo.maximum %v305, %v306 : tensor<128x64x56x56xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v309 = stablehlo.reshape %v308 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v310 = stablehlo.convolution(%v309, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v311 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v312 = stablehlo.add %v310, %v311 : tensor<128x128x28x28xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v316 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v317 = stablehlo.reduce(%v314 init: %v315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v318 = stablehlo.divide %v317, %v316 : tensor<128xf32>
    %arsumd2g1mu = "stablehlo.all_reduce"(%v318) ({
    ^bb0(%arad2g1mu: tensor<f32>, %arbd2g1mu: tensor<f32>):
      %araddd2g1mu = stablehlo.add %arad2g1mu, %arbd2g1mu : tensor<f32>
      stablehlo.return %araddd2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g1mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g1mu = stablehlo.divide %arsumd2g1mu, %arnd2g1mu : tensor<128xf32>
    %v319 = stablehlo.reshape %v313 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v320 = stablehlo.constant dense<0.0> : tensor<f32>
    %v321 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v322 = stablehlo.reduce(%v319 init: %v320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v323 = stablehlo.divide %v322, %v321 : tensor<128xf32>
    %v324 = stablehlo.broadcast_in_dim %v323, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v325 = stablehlo.subtract %v319, %v324 : tensor<128x128x28x28xf32>
    %v326 = stablehlo.multiply %v325, %v325 : tensor<128x128x28x28xf32>
    %v327 = stablehlo.reduce(%v326 init: %v320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
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
    %v333 = stablehlo.reshape %v313 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v334 = stablehlo.slice %v332 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v335 = stablehlo.slice %v332 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v336 = stablehlo.broadcast_in_dim %v334, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v337 = stablehlo.broadcast_in_dim %v335, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v338 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v339 = stablehlo.add %v337, %v338 : tensor<128x128x28x28xf32>
    %v340 = stablehlo.rsqrt %v339 : tensor<128x128x28x28xf32>
    %v341 = stablehlo.subtract %v333, %v336 : tensor<128x128x28x28xf32>
    %v342 = stablehlo.multiply %v341, %v340 : tensor<128x128x28x28xf32>
    %v343 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v344 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v345 = stablehlo.multiply %v342, %v343 : tensor<128x128x28x28xf32>
    %v346 = stablehlo.add %v345, %v344 : tensor<128x128x28x28xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v348 = stablehlo.constant dense<0.0> : tensor<128x100352xf32>
    %v349 = stablehlo.maximum %v347, %v348 : tensor<128x100352xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v351 = stablehlo.convolution(%v350, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v352 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v353 = stablehlo.add %v351, %v352 : tensor<128x128x28x28xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v357 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v358 = stablehlo.reduce(%v355 init: %v356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v359 = stablehlo.divide %v358, %v357 : tensor<128xf32>
    %arsumd2g2mu = "stablehlo.all_reduce"(%v359) ({
    ^bb0(%arad2g2mu: tensor<f32>, %arbd2g2mu: tensor<f32>):
      %araddd2g2mu = stablehlo.add %arad2g2mu, %arbd2g2mu : tensor<f32>
      stablehlo.return %araddd2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g2mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g2mu = stablehlo.divide %arsumd2g2mu, %arnd2g2mu : tensor<128xf32>
    %v360 = stablehlo.reshape %v354 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v362 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v363 = stablehlo.reduce(%v360 init: %v361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v364 = stablehlo.divide %v363, %v362 : tensor<128xf32>
    %v365 = stablehlo.broadcast_in_dim %v364, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v366 = stablehlo.subtract %v360, %v365 : tensor<128x128x28x28xf32>
    %v367 = stablehlo.multiply %v366, %v366 : tensor<128x128x28x28xf32>
    %v368 = stablehlo.reduce(%v367 init: %v361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
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
    %v374 = stablehlo.reshape %v354 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v375 = stablehlo.slice %v373 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v376 = stablehlo.slice %v373 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v377 = stablehlo.broadcast_in_dim %v375, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v378 = stablehlo.broadcast_in_dim %v376, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v379 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v380 = stablehlo.add %v378, %v379 : tensor<128x128x28x28xf32>
    %v381 = stablehlo.rsqrt %v380 : tensor<128x128x28x28xf32>
    %v382 = stablehlo.subtract %v374, %v377 : tensor<128x128x28x28xf32>
    %v383 = stablehlo.multiply %v382, %v381 : tensor<128x128x28x28xf32>
    %v384 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v385 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v386 = stablehlo.multiply %v383, %v384 : tensor<128x128x28x28xf32>
    %v387 = stablehlo.add %v386, %v385 : tensor<128x128x28x28xf32>
    %v388 = stablehlo.reshape %v387 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v389 = stablehlo.reshape %v308 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v390 = stablehlo.convolution(%v389, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<128x64x1x1xf32>) -> tensor<128x128x28x28xf32>
    %v391 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v392 = stablehlo.add %v390, %v391 : tensor<128x128x28x28xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v395 = stablehlo.constant dense<0.0> : tensor<f32>
    %v396 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v397 = stablehlo.reduce(%v394 init: %v395) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v398 = stablehlo.divide %v397, %v396 : tensor<128xf32>
    %arsumd2gpmu = "stablehlo.all_reduce"(%v398) ({
    ^bb0(%arad2gpmu: tensor<f32>, %arbd2gpmu: tensor<f32>):
      %araddd2gpmu = stablehlo.add %arad2gpmu, %arbd2gpmu : tensor<f32>
      stablehlo.return %araddd2gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2gpmu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2gpmu = stablehlo.divide %arsumd2gpmu, %arnd2gpmu : tensor<128xf32>
    %v399 = stablehlo.reshape %v393 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v401 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v402 = stablehlo.reduce(%v399 init: %v400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v403 = stablehlo.divide %v402, %v401 : tensor<128xf32>
    %v404 = stablehlo.broadcast_in_dim %v403, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v405 = stablehlo.subtract %v399, %v404 : tensor<128x128x28x28xf32>
    %v406 = stablehlo.multiply %v405, %v405 : tensor<128x128x28x28xf32>
    %v407 = stablehlo.reduce(%v406 init: %v400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
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
    %v413 = stablehlo.reshape %v393 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v414 = stablehlo.slice %v412 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v415 = stablehlo.slice %v412 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v416 = stablehlo.broadcast_in_dim %v414, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v417 = stablehlo.broadcast_in_dim %v415, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v418 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v419 = stablehlo.add %v417, %v418 : tensor<128x128x28x28xf32>
    %v420 = stablehlo.rsqrt %v419 : tensor<128x128x28x28xf32>
    %v421 = stablehlo.subtract %v413, %v416 : tensor<128x128x28x28xf32>
    %v422 = stablehlo.multiply %v421, %v420 : tensor<128x128x28x28xf32>
    %v423 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v424 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v425 = stablehlo.multiply %v422, %v423 : tensor<128x128x28x28xf32>
    %v426 = stablehlo.add %v425, %v424 : tensor<128x128x28x28xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v428 = stablehlo.add %v388, %v427 : tensor<128x100352xf32>
    %v429 = stablehlo.constant dense<0.0> : tensor<128x100352xf32>
    %v430 = stablehlo.maximum %v428, %v429 : tensor<128x100352xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v432 = stablehlo.convolution(%v431, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v433 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v434 = stablehlo.add %v432, %v433 : tensor<128x128x28x28xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v438 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v439 = stablehlo.reduce(%v436 init: %v437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v440 = stablehlo.divide %v439, %v438 : tensor<128xf32>
    %arsums2b0g1mu = "stablehlo.all_reduce"(%v440) ({
    ^bb0(%aras2b0g1mu: tensor<f32>, %arbs2b0g1mu: tensor<f32>):
      %aradds2b0g1mu = stablehlo.add %aras2b0g1mu, %arbs2b0g1mu : tensor<f32>
      stablehlo.return %aradds2b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g1mu = stablehlo.divide %arsums2b0g1mu, %arns2b0g1mu : tensor<128xf32>
    %v441 = stablehlo.reshape %v435 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v443 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v444 = stablehlo.reduce(%v441 init: %v442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v445 = stablehlo.divide %v444, %v443 : tensor<128xf32>
    %v446 = stablehlo.broadcast_in_dim %v445, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v447 = stablehlo.subtract %v441, %v446 : tensor<128x128x28x28xf32>
    %v448 = stablehlo.multiply %v447, %v447 : tensor<128x128x28x28xf32>
    %v449 = stablehlo.reduce(%v448 init: %v442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
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
    %v455 = stablehlo.reshape %v435 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v456 = stablehlo.slice %v454 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v457 = stablehlo.slice %v454 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v458 = stablehlo.broadcast_in_dim %v456, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v459 = stablehlo.broadcast_in_dim %v457, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v460 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v461 = stablehlo.add %v459, %v460 : tensor<128x128x28x28xf32>
    %v462 = stablehlo.rsqrt %v461 : tensor<128x128x28x28xf32>
    %v463 = stablehlo.subtract %v455, %v458 : tensor<128x128x28x28xf32>
    %v464 = stablehlo.multiply %v463, %v462 : tensor<128x128x28x28xf32>
    %v465 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v466 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v467 = stablehlo.multiply %v464, %v465 : tensor<128x128x28x28xf32>
    %v468 = stablehlo.add %v467, %v466 : tensor<128x128x28x28xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v470 = stablehlo.constant dense<0.0> : tensor<128x100352xf32>
    %v471 = stablehlo.maximum %v469, %v470 : tensor<128x100352xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v473 = stablehlo.convolution(%v472, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v474 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v475 = stablehlo.add %v473, %v474 : tensor<128x128x28x28xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v477 = stablehlo.reshape %v476 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v479 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v480 = stablehlo.reduce(%v477 init: %v478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v481 = stablehlo.divide %v480, %v479 : tensor<128xf32>
    %arsums2b0g2mu = "stablehlo.all_reduce"(%v481) ({
    ^bb0(%aras2b0g2mu: tensor<f32>, %arbs2b0g2mu: tensor<f32>):
      %aradds2b0g2mu = stablehlo.add %aras2b0g2mu, %arbs2b0g2mu : tensor<f32>
      stablehlo.return %aradds2b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g2mu = stablehlo.divide %arsums2b0g2mu, %arns2b0g2mu : tensor<128xf32>
    %v482 = stablehlo.reshape %v476 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v483 = stablehlo.constant dense<0.0> : tensor<f32>
    %v484 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v485 = stablehlo.reduce(%v482 init: %v483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v486 = stablehlo.divide %v485, %v484 : tensor<128xf32>
    %v487 = stablehlo.broadcast_in_dim %v486, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v488 = stablehlo.subtract %v482, %v487 : tensor<128x128x28x28xf32>
    %v489 = stablehlo.multiply %v488, %v488 : tensor<128x128x28x28xf32>
    %v490 = stablehlo.reduce(%v489 init: %v483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
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
    %v496 = stablehlo.reshape %v476 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v497 = stablehlo.slice %v495 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v498 = stablehlo.slice %v495 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v499 = stablehlo.broadcast_in_dim %v497, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v500 = stablehlo.broadcast_in_dim %v498, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v501 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v502 = stablehlo.add %v500, %v501 : tensor<128x128x28x28xf32>
    %v503 = stablehlo.rsqrt %v502 : tensor<128x128x28x28xf32>
    %v504 = stablehlo.subtract %v496, %v499 : tensor<128x128x28x28xf32>
    %v505 = stablehlo.multiply %v504, %v503 : tensor<128x128x28x28xf32>
    %v506 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v507 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v508 = stablehlo.multiply %v505, %v506 : tensor<128x128x28x28xf32>
    %v509 = stablehlo.add %v508, %v507 : tensor<128x128x28x28xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v512 = stablehlo.reshape %v430 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v513 = stablehlo.add %v511, %v512 : tensor<128x128x28x28xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v516 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v517 = stablehlo.maximum %v515, %v516 : tensor<128x128x28x28xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v520 = stablehlo.convolution(%v519, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v521 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v522 = stablehlo.add %v520, %v521 : tensor<128x128x28x28xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v524 = stablehlo.reshape %v523 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v526 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v527 = stablehlo.reduce(%v524 init: %v525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v528 = stablehlo.divide %v527, %v526 : tensor<128xf32>
    %arsums2b1g1mu = "stablehlo.all_reduce"(%v528) ({
    ^bb0(%aras2b1g1mu: tensor<f32>, %arbs2b1g1mu: tensor<f32>):
      %aradds2b1g1mu = stablehlo.add %aras2b1g1mu, %arbs2b1g1mu : tensor<f32>
      stablehlo.return %aradds2b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g1mu = stablehlo.divide %arsums2b1g1mu, %arns2b1g1mu : tensor<128xf32>
    %v529 = stablehlo.reshape %v523 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v531 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v532 = stablehlo.reduce(%v529 init: %v530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v533 = stablehlo.divide %v532, %v531 : tensor<128xf32>
    %v534 = stablehlo.broadcast_in_dim %v533, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v535 = stablehlo.subtract %v529, %v534 : tensor<128x128x28x28xf32>
    %v536 = stablehlo.multiply %v535, %v535 : tensor<128x128x28x28xf32>
    %v537 = stablehlo.reduce(%v536 init: %v530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
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
    %v543 = stablehlo.reshape %v523 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v544 = stablehlo.slice %v542 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v545 = stablehlo.slice %v542 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v546 = stablehlo.broadcast_in_dim %v544, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v547 = stablehlo.broadcast_in_dim %v545, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v548 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v549 = stablehlo.add %v547, %v548 : tensor<128x128x28x28xf32>
    %v550 = stablehlo.rsqrt %v549 : tensor<128x128x28x28xf32>
    %v551 = stablehlo.subtract %v543, %v546 : tensor<128x128x28x28xf32>
    %v552 = stablehlo.multiply %v551, %v550 : tensor<128x128x28x28xf32>
    %v553 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v554 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v555 = stablehlo.multiply %v552, %v553 : tensor<128x128x28x28xf32>
    %v556 = stablehlo.add %v555, %v554 : tensor<128x128x28x28xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v558 = stablehlo.constant dense<0.0> : tensor<128x100352xf32>
    %v559 = stablehlo.maximum %v557, %v558 : tensor<128x100352xf32>
    %v560 = stablehlo.reshape %v559 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v561 = stablehlo.convolution(%v560, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v562 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v563 = stablehlo.add %v561, %v562 : tensor<128x128x28x28xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v567 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v568 = stablehlo.reduce(%v565 init: %v566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v569 = stablehlo.divide %v568, %v567 : tensor<128xf32>
    %arsums2b1g2mu = "stablehlo.all_reduce"(%v569) ({
    ^bb0(%aras2b1g2mu: tensor<f32>, %arbs2b1g2mu: tensor<f32>):
      %aradds2b1g2mu = stablehlo.add %aras2b1g2mu, %arbs2b1g2mu : tensor<f32>
      stablehlo.return %aradds2b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g2mu = stablehlo.divide %arsums2b1g2mu, %arns2b1g2mu : tensor<128xf32>
    %v570 = stablehlo.reshape %v564 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v571 = stablehlo.constant dense<0.0> : tensor<f32>
    %v572 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v573 = stablehlo.reduce(%v570 init: %v571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v574 = stablehlo.divide %v573, %v572 : tensor<128xf32>
    %v575 = stablehlo.broadcast_in_dim %v574, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v576 = stablehlo.subtract %v570, %v575 : tensor<128x128x28x28xf32>
    %v577 = stablehlo.multiply %v576, %v576 : tensor<128x128x28x28xf32>
    %v578 = stablehlo.reduce(%v577 init: %v571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
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
    %v584 = stablehlo.reshape %v564 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v585 = stablehlo.slice %v583 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v586 = stablehlo.slice %v583 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v587 = stablehlo.broadcast_in_dim %v585, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v588 = stablehlo.broadcast_in_dim %v586, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v589 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v590 = stablehlo.add %v588, %v589 : tensor<128x128x28x28xf32>
    %v591 = stablehlo.rsqrt %v590 : tensor<128x128x28x28xf32>
    %v592 = stablehlo.subtract %v584, %v587 : tensor<128x128x28x28xf32>
    %v593 = stablehlo.multiply %v592, %v591 : tensor<128x128x28x28xf32>
    %v594 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v595 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v596 = stablehlo.multiply %v593, %v594 : tensor<128x128x28x28xf32>
    %v597 = stablehlo.add %v596, %v595 : tensor<128x128x28x28xf32>
    %v598 = stablehlo.reshape %v597 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v600 = stablehlo.reshape %v518 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v601 = stablehlo.add %v599, %v600 : tensor<128x128x28x28xf32>
    %v602 = stablehlo.reshape %v601 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v604 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v605 = stablehlo.maximum %v603, %v604 : tensor<128x128x28x28xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v608 = stablehlo.convolution(%v607, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v609 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v610 = stablehlo.add %v608, %v609 : tensor<128x128x28x28xf32>
    %v611 = stablehlo.reshape %v610 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v614 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v615 = stablehlo.reduce(%v612 init: %v613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v616 = stablehlo.divide %v615, %v614 : tensor<128xf32>
    %arsums2b2g1mu = "stablehlo.all_reduce"(%v616) ({
    ^bb0(%aras2b2g1mu: tensor<f32>, %arbs2b2g1mu: tensor<f32>):
      %aradds2b2g1mu = stablehlo.add %aras2b2g1mu, %arbs2b2g1mu : tensor<f32>
      stablehlo.return %aradds2b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g1mu = stablehlo.divide %arsums2b2g1mu, %arns2b2g1mu : tensor<128xf32>
    %v617 = stablehlo.reshape %v611 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v619 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v620 = stablehlo.reduce(%v617 init: %v618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v621 = stablehlo.divide %v620, %v619 : tensor<128xf32>
    %v622 = stablehlo.broadcast_in_dim %v621, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v623 = stablehlo.subtract %v617, %v622 : tensor<128x128x28x28xf32>
    %v624 = stablehlo.multiply %v623, %v623 : tensor<128x128x28x28xf32>
    %v625 = stablehlo.reduce(%v624 init: %v618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
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
    %v631 = stablehlo.reshape %v611 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v632 = stablehlo.slice %v630 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v633 = stablehlo.slice %v630 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v634 = stablehlo.broadcast_in_dim %v632, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v635 = stablehlo.broadcast_in_dim %v633, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v636 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v637 = stablehlo.add %v635, %v636 : tensor<128x128x28x28xf32>
    %v638 = stablehlo.rsqrt %v637 : tensor<128x128x28x28xf32>
    %v639 = stablehlo.subtract %v631, %v634 : tensor<128x128x28x28xf32>
    %v640 = stablehlo.multiply %v639, %v638 : tensor<128x128x28x28xf32>
    %v641 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v642 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v643 = stablehlo.multiply %v640, %v641 : tensor<128x128x28x28xf32>
    %v644 = stablehlo.add %v643, %v642 : tensor<128x128x28x28xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v646 = stablehlo.constant dense<0.0> : tensor<128x100352xf32>
    %v647 = stablehlo.maximum %v645, %v646 : tensor<128x100352xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v649 = stablehlo.convolution(%v648, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v650 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v651 = stablehlo.add %v649, %v650 : tensor<128x128x28x28xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v655 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v656 = stablehlo.reduce(%v653 init: %v654) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v657 = stablehlo.divide %v656, %v655 : tensor<128xf32>
    %arsums2b2g2mu = "stablehlo.all_reduce"(%v657) ({
    ^bb0(%aras2b2g2mu: tensor<f32>, %arbs2b2g2mu: tensor<f32>):
      %aradds2b2g2mu = stablehlo.add %aras2b2g2mu, %arbs2b2g2mu : tensor<f32>
      stablehlo.return %aradds2b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2mu = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g2mu = stablehlo.divide %arsums2b2g2mu, %arns2b2g2mu : tensor<128xf32>
    %v658 = stablehlo.reshape %v652 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v660 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v661 = stablehlo.reduce(%v658 init: %v659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v662 = stablehlo.divide %v661, %v660 : tensor<128xf32>
    %v663 = stablehlo.broadcast_in_dim %v662, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v664 = stablehlo.subtract %v658, %v663 : tensor<128x128x28x28xf32>
    %v665 = stablehlo.multiply %v664, %v664 : tensor<128x128x28x28xf32>
    %v666 = stablehlo.reduce(%v665 init: %v659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
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
    %v672 = stablehlo.reshape %v652 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v673 = stablehlo.slice %v671 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v674 = stablehlo.slice %v671 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v675 = stablehlo.broadcast_in_dim %v673, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v676 = stablehlo.broadcast_in_dim %v674, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v677 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v678 = stablehlo.add %v676, %v677 : tensor<128x128x28x28xf32>
    %v679 = stablehlo.rsqrt %v678 : tensor<128x128x28x28xf32>
    %v680 = stablehlo.subtract %v672, %v675 : tensor<128x128x28x28xf32>
    %v681 = stablehlo.multiply %v680, %v679 : tensor<128x128x28x28xf32>
    %v682 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v683 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v684 = stablehlo.multiply %v681, %v682 : tensor<128x128x28x28xf32>
    %v685 = stablehlo.add %v684, %v683 : tensor<128x128x28x28xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v688 = stablehlo.reshape %v606 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v689 = stablehlo.add %v687, %v688 : tensor<128x128x28x28xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v692 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v693 = stablehlo.maximum %v691, %v692 : tensor<128x128x28x28xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v696 = stablehlo.convolution(%v695, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v697 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v698 = stablehlo.add %v696, %v697 : tensor<128x256x14x14xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v702 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v703 = stablehlo.reduce(%v700 init: %v701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v704 = stablehlo.divide %v703, %v702 : tensor<256xf32>
    %arsumd3g1mu = "stablehlo.all_reduce"(%v704) ({
    ^bb0(%arad3g1mu: tensor<f32>, %arbd3g1mu: tensor<f32>):
      %araddd3g1mu = stablehlo.add %arad3g1mu, %arbd3g1mu : tensor<f32>
      stablehlo.return %araddd3g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g1mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g1mu = stablehlo.divide %arsumd3g1mu, %arnd3g1mu : tensor<256xf32>
    %v705 = stablehlo.reshape %v699 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v706 = stablehlo.constant dense<0.0> : tensor<f32>
    %v707 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v708 = stablehlo.reduce(%v705 init: %v706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v709 = stablehlo.divide %v708, %v707 : tensor<256xf32>
    %v710 = stablehlo.broadcast_in_dim %v709, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v711 = stablehlo.subtract %v705, %v710 : tensor<128x256x14x14xf32>
    %v712 = stablehlo.multiply %v711, %v711 : tensor<128x256x14x14xf32>
    %v713 = stablehlo.reduce(%v712 init: %v706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
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
    %v719 = stablehlo.reshape %v699 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v720 = stablehlo.slice %v718 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v721 = stablehlo.slice %v718 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v722 = stablehlo.broadcast_in_dim %v720, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v723 = stablehlo.broadcast_in_dim %v721, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v724 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v725 = stablehlo.add %v723, %v724 : tensor<128x256x14x14xf32>
    %v726 = stablehlo.rsqrt %v725 : tensor<128x256x14x14xf32>
    %v727 = stablehlo.subtract %v719, %v722 : tensor<128x256x14x14xf32>
    %v728 = stablehlo.multiply %v727, %v726 : tensor<128x256x14x14xf32>
    %v729 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v730 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v731 = stablehlo.multiply %v728, %v729 : tensor<128x256x14x14xf32>
    %v732 = stablehlo.add %v731, %v730 : tensor<128x256x14x14xf32>
    %v733 = stablehlo.reshape %v732 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v734 = stablehlo.constant dense<0.0> : tensor<128x50176xf32>
    %v735 = stablehlo.maximum %v733, %v734 : tensor<128x50176xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v737 = stablehlo.convolution(%v736, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v738 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v739 = stablehlo.add %v737, %v738 : tensor<128x256x14x14xf32>
    %v740 = stablehlo.reshape %v739 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v743 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v744 = stablehlo.reduce(%v741 init: %v742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v745 = stablehlo.divide %v744, %v743 : tensor<256xf32>
    %arsumd3g2mu = "stablehlo.all_reduce"(%v745) ({
    ^bb0(%arad3g2mu: tensor<f32>, %arbd3g2mu: tensor<f32>):
      %araddd3g2mu = stablehlo.add %arad3g2mu, %arbd3g2mu : tensor<f32>
      stablehlo.return %araddd3g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g2mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g2mu = stablehlo.divide %arsumd3g2mu, %arnd3g2mu : tensor<256xf32>
    %v746 = stablehlo.reshape %v740 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v748 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v749 = stablehlo.reduce(%v746 init: %v747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v750 = stablehlo.divide %v749, %v748 : tensor<256xf32>
    %v751 = stablehlo.broadcast_in_dim %v750, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v752 = stablehlo.subtract %v746, %v751 : tensor<128x256x14x14xf32>
    %v753 = stablehlo.multiply %v752, %v752 : tensor<128x256x14x14xf32>
    %v754 = stablehlo.reduce(%v753 init: %v747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
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
    %v760 = stablehlo.reshape %v740 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v761 = stablehlo.slice %v759 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v762 = stablehlo.slice %v759 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v763 = stablehlo.broadcast_in_dim %v761, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v764 = stablehlo.broadcast_in_dim %v762, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v765 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v766 = stablehlo.add %v764, %v765 : tensor<128x256x14x14xf32>
    %v767 = stablehlo.rsqrt %v766 : tensor<128x256x14x14xf32>
    %v768 = stablehlo.subtract %v760, %v763 : tensor<128x256x14x14xf32>
    %v769 = stablehlo.multiply %v768, %v767 : tensor<128x256x14x14xf32>
    %v770 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v771 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v772 = stablehlo.multiply %v769, %v770 : tensor<128x256x14x14xf32>
    %v773 = stablehlo.add %v772, %v771 : tensor<128x256x14x14xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v775 = stablehlo.reshape %v694 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v776 = stablehlo.convolution(%v775, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<256x128x1x1xf32>) -> tensor<128x256x14x14xf32>
    %v777 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v778 = stablehlo.add %v776, %v777 : tensor<128x256x14x14xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v780 = stablehlo.reshape %v779 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v782 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v783 = stablehlo.reduce(%v780 init: %v781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v784 = stablehlo.divide %v783, %v782 : tensor<256xf32>
    %arsumd3gpmu = "stablehlo.all_reduce"(%v784) ({
    ^bb0(%arad3gpmu: tensor<f32>, %arbd3gpmu: tensor<f32>):
      %araddd3gpmu = stablehlo.add %arad3gpmu, %arbd3gpmu : tensor<f32>
      stablehlo.return %araddd3gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3gpmu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3gpmu = stablehlo.divide %arsumd3gpmu, %arnd3gpmu : tensor<256xf32>
    %v785 = stablehlo.reshape %v779 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v787 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v788 = stablehlo.reduce(%v785 init: %v786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v789 = stablehlo.divide %v788, %v787 : tensor<256xf32>
    %v790 = stablehlo.broadcast_in_dim %v789, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v791 = stablehlo.subtract %v785, %v790 : tensor<128x256x14x14xf32>
    %v792 = stablehlo.multiply %v791, %v791 : tensor<128x256x14x14xf32>
    %v793 = stablehlo.reduce(%v792 init: %v786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
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
    %v799 = stablehlo.reshape %v779 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v800 = stablehlo.slice %v798 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v801 = stablehlo.slice %v798 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v802 = stablehlo.broadcast_in_dim %v800, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v803 = stablehlo.broadcast_in_dim %v801, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v804 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v805 = stablehlo.add %v803, %v804 : tensor<128x256x14x14xf32>
    %v806 = stablehlo.rsqrt %v805 : tensor<128x256x14x14xf32>
    %v807 = stablehlo.subtract %v799, %v802 : tensor<128x256x14x14xf32>
    %v808 = stablehlo.multiply %v807, %v806 : tensor<128x256x14x14xf32>
    %v809 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v810 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v811 = stablehlo.multiply %v808, %v809 : tensor<128x256x14x14xf32>
    %v812 = stablehlo.add %v811, %v810 : tensor<128x256x14x14xf32>
    %v813 = stablehlo.reshape %v812 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v814 = stablehlo.add %v774, %v813 : tensor<128x50176xf32>
    %v815 = stablehlo.constant dense<0.0> : tensor<128x50176xf32>
    %v816 = stablehlo.maximum %v814, %v815 : tensor<128x50176xf32>
    %v817 = stablehlo.reshape %v816 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v818 = stablehlo.convolution(%v817, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v819 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v820 = stablehlo.add %v818, %v819 : tensor<128x256x14x14xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v823 = stablehlo.constant dense<0.0> : tensor<f32>
    %v824 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v825 = stablehlo.reduce(%v822 init: %v823) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v826 = stablehlo.divide %v825, %v824 : tensor<256xf32>
    %arsums3b0g1mu = "stablehlo.all_reduce"(%v826) ({
    ^bb0(%aras3b0g1mu: tensor<f32>, %arbs3b0g1mu: tensor<f32>):
      %aradds3b0g1mu = stablehlo.add %aras3b0g1mu, %arbs3b0g1mu : tensor<f32>
      stablehlo.return %aradds3b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g1mu = stablehlo.divide %arsums3b0g1mu, %arns3b0g1mu : tensor<256xf32>
    %v827 = stablehlo.reshape %v821 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v828 = stablehlo.constant dense<0.0> : tensor<f32>
    %v829 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v830 = stablehlo.reduce(%v827 init: %v828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v831 = stablehlo.divide %v830, %v829 : tensor<256xf32>
    %v832 = stablehlo.broadcast_in_dim %v831, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v833 = stablehlo.subtract %v827, %v832 : tensor<128x256x14x14xf32>
    %v834 = stablehlo.multiply %v833, %v833 : tensor<128x256x14x14xf32>
    %v835 = stablehlo.reduce(%v834 init: %v828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
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
    %v841 = stablehlo.reshape %v821 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v842 = stablehlo.slice %v840 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v843 = stablehlo.slice %v840 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v844 = stablehlo.broadcast_in_dim %v842, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v845 = stablehlo.broadcast_in_dim %v843, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v846 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v847 = stablehlo.add %v845, %v846 : tensor<128x256x14x14xf32>
    %v848 = stablehlo.rsqrt %v847 : tensor<128x256x14x14xf32>
    %v849 = stablehlo.subtract %v841, %v844 : tensor<128x256x14x14xf32>
    %v850 = stablehlo.multiply %v849, %v848 : tensor<128x256x14x14xf32>
    %v851 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v852 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v853 = stablehlo.multiply %v850, %v851 : tensor<128x256x14x14xf32>
    %v854 = stablehlo.add %v853, %v852 : tensor<128x256x14x14xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v856 = stablehlo.constant dense<0.0> : tensor<128x50176xf32>
    %v857 = stablehlo.maximum %v855, %v856 : tensor<128x50176xf32>
    %v858 = stablehlo.reshape %v857 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v859 = stablehlo.convolution(%v858, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v860 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v861 = stablehlo.add %v859, %v860 : tensor<128x256x14x14xf32>
    %v862 = stablehlo.reshape %v861 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v863 = stablehlo.reshape %v862 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v865 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v866 = stablehlo.reduce(%v863 init: %v864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v867 = stablehlo.divide %v866, %v865 : tensor<256xf32>
    %arsums3b0g2mu = "stablehlo.all_reduce"(%v867) ({
    ^bb0(%aras3b0g2mu: tensor<f32>, %arbs3b0g2mu: tensor<f32>):
      %aradds3b0g2mu = stablehlo.add %aras3b0g2mu, %arbs3b0g2mu : tensor<f32>
      stablehlo.return %aradds3b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g2mu = stablehlo.divide %arsums3b0g2mu, %arns3b0g2mu : tensor<256xf32>
    %v868 = stablehlo.reshape %v862 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v870 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v871 = stablehlo.reduce(%v868 init: %v869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v872 = stablehlo.divide %v871, %v870 : tensor<256xf32>
    %v873 = stablehlo.broadcast_in_dim %v872, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v874 = stablehlo.subtract %v868, %v873 : tensor<128x256x14x14xf32>
    %v875 = stablehlo.multiply %v874, %v874 : tensor<128x256x14x14xf32>
    %v876 = stablehlo.reduce(%v875 init: %v869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
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
    %v882 = stablehlo.reshape %v862 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v883 = stablehlo.slice %v881 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v884 = stablehlo.slice %v881 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v885 = stablehlo.broadcast_in_dim %v883, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v886 = stablehlo.broadcast_in_dim %v884, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v887 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v888 = stablehlo.add %v886, %v887 : tensor<128x256x14x14xf32>
    %v889 = stablehlo.rsqrt %v888 : tensor<128x256x14x14xf32>
    %v890 = stablehlo.subtract %v882, %v885 : tensor<128x256x14x14xf32>
    %v891 = stablehlo.multiply %v890, %v889 : tensor<128x256x14x14xf32>
    %v892 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v893 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v894 = stablehlo.multiply %v891, %v892 : tensor<128x256x14x14xf32>
    %v895 = stablehlo.add %v894, %v893 : tensor<128x256x14x14xf32>
    %v896 = stablehlo.reshape %v895 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v897 = stablehlo.reshape %v896 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v898 = stablehlo.reshape %v816 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v899 = stablehlo.add %v897, %v898 : tensor<128x256x14x14xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v901 = stablehlo.reshape %v900 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v902 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v903 = stablehlo.maximum %v901, %v902 : tensor<128x256x14x14xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v906 = stablehlo.convolution(%v905, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v907 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v908 = stablehlo.add %v906, %v907 : tensor<128x256x14x14xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v910 = stablehlo.reshape %v909 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v912 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v913 = stablehlo.reduce(%v910 init: %v911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v914 = stablehlo.divide %v913, %v912 : tensor<256xf32>
    %arsums3b1g1mu = "stablehlo.all_reduce"(%v914) ({
    ^bb0(%aras3b1g1mu: tensor<f32>, %arbs3b1g1mu: tensor<f32>):
      %aradds3b1g1mu = stablehlo.add %aras3b1g1mu, %arbs3b1g1mu : tensor<f32>
      stablehlo.return %aradds3b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g1mu = stablehlo.divide %arsums3b1g1mu, %arns3b1g1mu : tensor<256xf32>
    %v915 = stablehlo.reshape %v909 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v916 = stablehlo.constant dense<0.0> : tensor<f32>
    %v917 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v918 = stablehlo.reduce(%v915 init: %v916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v919 = stablehlo.divide %v918, %v917 : tensor<256xf32>
    %v920 = stablehlo.broadcast_in_dim %v919, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v921 = stablehlo.subtract %v915, %v920 : tensor<128x256x14x14xf32>
    %v922 = stablehlo.multiply %v921, %v921 : tensor<128x256x14x14xf32>
    %v923 = stablehlo.reduce(%v922 init: %v916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
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
    %v929 = stablehlo.reshape %v909 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v930 = stablehlo.slice %v928 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v931 = stablehlo.slice %v928 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v932 = stablehlo.broadcast_in_dim %v930, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v933 = stablehlo.broadcast_in_dim %v931, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v934 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v935 = stablehlo.add %v933, %v934 : tensor<128x256x14x14xf32>
    %v936 = stablehlo.rsqrt %v935 : tensor<128x256x14x14xf32>
    %v937 = stablehlo.subtract %v929, %v932 : tensor<128x256x14x14xf32>
    %v938 = stablehlo.multiply %v937, %v936 : tensor<128x256x14x14xf32>
    %v939 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v940 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v941 = stablehlo.multiply %v938, %v939 : tensor<128x256x14x14xf32>
    %v942 = stablehlo.add %v941, %v940 : tensor<128x256x14x14xf32>
    %v943 = stablehlo.reshape %v942 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v944 = stablehlo.constant dense<0.0> : tensor<128x50176xf32>
    %v945 = stablehlo.maximum %v943, %v944 : tensor<128x50176xf32>
    %v946 = stablehlo.reshape %v945 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v947 = stablehlo.convolution(%v946, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v948 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v949 = stablehlo.add %v947, %v948 : tensor<128x256x14x14xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v951 = stablehlo.reshape %v950 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v952 = stablehlo.constant dense<0.0> : tensor<f32>
    %v953 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v954 = stablehlo.reduce(%v951 init: %v952) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v955 = stablehlo.divide %v954, %v953 : tensor<256xf32>
    %arsums3b1g2mu = "stablehlo.all_reduce"(%v955) ({
    ^bb0(%aras3b1g2mu: tensor<f32>, %arbs3b1g2mu: tensor<f32>):
      %aradds3b1g2mu = stablehlo.add %aras3b1g2mu, %arbs3b1g2mu : tensor<f32>
      stablehlo.return %aradds3b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g2mu = stablehlo.divide %arsums3b1g2mu, %arns3b1g2mu : tensor<256xf32>
    %v956 = stablehlo.reshape %v950 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v957 = stablehlo.constant dense<0.0> : tensor<f32>
    %v958 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v959 = stablehlo.reduce(%v956 init: %v957) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v960 = stablehlo.divide %v959, %v958 : tensor<256xf32>
    %v961 = stablehlo.broadcast_in_dim %v960, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v962 = stablehlo.subtract %v956, %v961 : tensor<128x256x14x14xf32>
    %v963 = stablehlo.multiply %v962, %v962 : tensor<128x256x14x14xf32>
    %v964 = stablehlo.reduce(%v963 init: %v957) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
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
    %v970 = stablehlo.reshape %v950 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v971 = stablehlo.slice %v969 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v972 = stablehlo.slice %v969 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v973 = stablehlo.broadcast_in_dim %v971, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v974 = stablehlo.broadcast_in_dim %v972, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v975 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v976 = stablehlo.add %v974, %v975 : tensor<128x256x14x14xf32>
    %v977 = stablehlo.rsqrt %v976 : tensor<128x256x14x14xf32>
    %v978 = stablehlo.subtract %v970, %v973 : tensor<128x256x14x14xf32>
    %v979 = stablehlo.multiply %v978, %v977 : tensor<128x256x14x14xf32>
    %v980 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v981 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v982 = stablehlo.multiply %v979, %v980 : tensor<128x256x14x14xf32>
    %v983 = stablehlo.add %v982, %v981 : tensor<128x256x14x14xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v986 = stablehlo.reshape %v904 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v987 = stablehlo.add %v985, %v986 : tensor<128x256x14x14xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v990 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v991 = stablehlo.maximum %v989, %v990 : tensor<128x256x14x14xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v993 = stablehlo.reshape %v992 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v994 = stablehlo.convolution(%v993, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v995 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v996 = stablehlo.add %v994, %v995 : tensor<128x256x14x14xf32>
    %v997 = stablehlo.reshape %v996 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v998 = stablehlo.reshape %v997 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v999 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1000 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v1001 = stablehlo.reduce(%v998 init: %v999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1002 = stablehlo.divide %v1001, %v1000 : tensor<256xf32>
    %arsums3b2g1mu = "stablehlo.all_reduce"(%v1002) ({
    ^bb0(%aras3b2g1mu: tensor<f32>, %arbs3b2g1mu: tensor<f32>):
      %aradds3b2g1mu = stablehlo.add %aras3b2g1mu, %arbs3b2g1mu : tensor<f32>
      stablehlo.return %aradds3b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g1mu = stablehlo.divide %arsums3b2g1mu, %arns3b2g1mu : tensor<256xf32>
    %v1003 = stablehlo.reshape %v997 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1004 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1005 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v1006 = stablehlo.reduce(%v1003 init: %v1004) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1007 = stablehlo.divide %v1006, %v1005 : tensor<256xf32>
    %v1008 = stablehlo.broadcast_in_dim %v1007, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1009 = stablehlo.subtract %v1003, %v1008 : tensor<128x256x14x14xf32>
    %v1010 = stablehlo.multiply %v1009, %v1009 : tensor<128x256x14x14xf32>
    %v1011 = stablehlo.reduce(%v1010 init: %v1004) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
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
    %v1017 = stablehlo.reshape %v997 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1018 = stablehlo.slice %v1016 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1019 = stablehlo.slice %v1016 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1020 = stablehlo.broadcast_in_dim %v1018, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1021 = stablehlo.broadcast_in_dim %v1019, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1022 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1023 = stablehlo.add %v1021, %v1022 : tensor<128x256x14x14xf32>
    %v1024 = stablehlo.rsqrt %v1023 : tensor<128x256x14x14xf32>
    %v1025 = stablehlo.subtract %v1017, %v1020 : tensor<128x256x14x14xf32>
    %v1026 = stablehlo.multiply %v1025, %v1024 : tensor<128x256x14x14xf32>
    %v1027 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1028 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1029 = stablehlo.multiply %v1026, %v1027 : tensor<128x256x14x14xf32>
    %v1030 = stablehlo.add %v1029, %v1028 : tensor<128x256x14x14xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1032 = stablehlo.constant dense<0.0> : tensor<128x50176xf32>
    %v1033 = stablehlo.maximum %v1031, %v1032 : tensor<128x50176xf32>
    %v1034 = stablehlo.reshape %v1033 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1035 = stablehlo.convolution(%v1034, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v1036 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1037 = stablehlo.add %v1035, %v1036 : tensor<128x256x14x14xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1041 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v1042 = stablehlo.reduce(%v1039 init: %v1040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1043 = stablehlo.divide %v1042, %v1041 : tensor<256xf32>
    %arsums3b2g2mu = "stablehlo.all_reduce"(%v1043) ({
    ^bb0(%aras3b2g2mu: tensor<f32>, %arbs3b2g2mu: tensor<f32>):
      %aradds3b2g2mu = stablehlo.add %aras3b2g2mu, %arbs3b2g2mu : tensor<f32>
      stablehlo.return %aradds3b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g2mu = stablehlo.divide %arsums3b2g2mu, %arns3b2g2mu : tensor<256xf32>
    %v1044 = stablehlo.reshape %v1038 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1046 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v1047 = stablehlo.reduce(%v1044 init: %v1045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1048 = stablehlo.divide %v1047, %v1046 : tensor<256xf32>
    %v1049 = stablehlo.broadcast_in_dim %v1048, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1050 = stablehlo.subtract %v1044, %v1049 : tensor<128x256x14x14xf32>
    %v1051 = stablehlo.multiply %v1050, %v1050 : tensor<128x256x14x14xf32>
    %v1052 = stablehlo.reduce(%v1051 init: %v1045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
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
    %v1058 = stablehlo.reshape %v1038 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1059 = stablehlo.slice %v1057 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1060 = stablehlo.slice %v1057 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1061 = stablehlo.broadcast_in_dim %v1059, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1062 = stablehlo.broadcast_in_dim %v1060, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1063 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1064 = stablehlo.add %v1062, %v1063 : tensor<128x256x14x14xf32>
    %v1065 = stablehlo.rsqrt %v1064 : tensor<128x256x14x14xf32>
    %v1066 = stablehlo.subtract %v1058, %v1061 : tensor<128x256x14x14xf32>
    %v1067 = stablehlo.multiply %v1066, %v1065 : tensor<128x256x14x14xf32>
    %v1068 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1069 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1070 = stablehlo.multiply %v1067, %v1068 : tensor<128x256x14x14xf32>
    %v1071 = stablehlo.add %v1070, %v1069 : tensor<128x256x14x14xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1074 = stablehlo.reshape %v992 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1075 = stablehlo.add %v1073, %v1074 : tensor<128x256x14x14xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1077 = stablehlo.reshape %v1076 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1078 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v1079 = stablehlo.maximum %v1077, %v1078 : tensor<128x256x14x14xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1082 = stablehlo.convolution(%v1081, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v1083 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1084 = stablehlo.add %v1082, %v1083 : tensor<128x256x14x14xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1088 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v1089 = stablehlo.reduce(%v1086 init: %v1087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1090 = stablehlo.divide %v1089, %v1088 : tensor<256xf32>
    %arsums3b3g1mu = "stablehlo.all_reduce"(%v1090) ({
    ^bb0(%aras3b3g1mu: tensor<f32>, %arbs3b3g1mu: tensor<f32>):
      %aradds3b3g1mu = stablehlo.add %aras3b3g1mu, %arbs3b3g1mu : tensor<f32>
      stablehlo.return %aradds3b3g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g1mu = stablehlo.divide %arsums3b3g1mu, %arns3b3g1mu : tensor<256xf32>
    %v1091 = stablehlo.reshape %v1085 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1093 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v1094 = stablehlo.reduce(%v1091 init: %v1092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1095 = stablehlo.divide %v1094, %v1093 : tensor<256xf32>
    %v1096 = stablehlo.broadcast_in_dim %v1095, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1097 = stablehlo.subtract %v1091, %v1096 : tensor<128x256x14x14xf32>
    %v1098 = stablehlo.multiply %v1097, %v1097 : tensor<128x256x14x14xf32>
    %v1099 = stablehlo.reduce(%v1098 init: %v1092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
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
    %v1105 = stablehlo.reshape %v1085 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1106 = stablehlo.slice %v1104 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1107 = stablehlo.slice %v1104 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1108 = stablehlo.broadcast_in_dim %v1106, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1109 = stablehlo.broadcast_in_dim %v1107, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1110 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1111 = stablehlo.add %v1109, %v1110 : tensor<128x256x14x14xf32>
    %v1112 = stablehlo.rsqrt %v1111 : tensor<128x256x14x14xf32>
    %v1113 = stablehlo.subtract %v1105, %v1108 : tensor<128x256x14x14xf32>
    %v1114 = stablehlo.multiply %v1113, %v1112 : tensor<128x256x14x14xf32>
    %v1115 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1116 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1117 = stablehlo.multiply %v1114, %v1115 : tensor<128x256x14x14xf32>
    %v1118 = stablehlo.add %v1117, %v1116 : tensor<128x256x14x14xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1120 = stablehlo.constant dense<0.0> : tensor<128x50176xf32>
    %v1121 = stablehlo.maximum %v1119, %v1120 : tensor<128x50176xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1123 = stablehlo.convolution(%v1122, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v1124 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1125 = stablehlo.add %v1123, %v1124 : tensor<128x256x14x14xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1129 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v1130 = stablehlo.reduce(%v1127 init: %v1128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1131 = stablehlo.divide %v1130, %v1129 : tensor<256xf32>
    %arsums3b3g2mu = "stablehlo.all_reduce"(%v1131) ({
    ^bb0(%aras3b3g2mu: tensor<f32>, %arbs3b3g2mu: tensor<f32>):
      %aradds3b3g2mu = stablehlo.add %aras3b3g2mu, %arbs3b3g2mu : tensor<f32>
      stablehlo.return %aradds3b3g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g2mu = stablehlo.divide %arsums3b3g2mu, %arns3b3g2mu : tensor<256xf32>
    %v1132 = stablehlo.reshape %v1126 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1134 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v1135 = stablehlo.reduce(%v1132 init: %v1133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1136 = stablehlo.divide %v1135, %v1134 : tensor<256xf32>
    %v1137 = stablehlo.broadcast_in_dim %v1136, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1138 = stablehlo.subtract %v1132, %v1137 : tensor<128x256x14x14xf32>
    %v1139 = stablehlo.multiply %v1138, %v1138 : tensor<128x256x14x14xf32>
    %v1140 = stablehlo.reduce(%v1139 init: %v1133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
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
    %v1146 = stablehlo.reshape %v1126 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1147 = stablehlo.slice %v1145 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1148 = stablehlo.slice %v1145 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1149 = stablehlo.broadcast_in_dim %v1147, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1150 = stablehlo.broadcast_in_dim %v1148, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1151 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1152 = stablehlo.add %v1150, %v1151 : tensor<128x256x14x14xf32>
    %v1153 = stablehlo.rsqrt %v1152 : tensor<128x256x14x14xf32>
    %v1154 = stablehlo.subtract %v1146, %v1149 : tensor<128x256x14x14xf32>
    %v1155 = stablehlo.multiply %v1154, %v1153 : tensor<128x256x14x14xf32>
    %v1156 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1157 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1158 = stablehlo.multiply %v1155, %v1156 : tensor<128x256x14x14xf32>
    %v1159 = stablehlo.add %v1158, %v1157 : tensor<128x256x14x14xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1162 = stablehlo.reshape %v1080 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1163 = stablehlo.add %v1161, %v1162 : tensor<128x256x14x14xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1166 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v1167 = stablehlo.maximum %v1165, %v1166 : tensor<128x256x14x14xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1170 = stablehlo.convolution(%v1169, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v1171 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1172 = stablehlo.add %v1170, %v1171 : tensor<128x256x14x14xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1175 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1176 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v1177 = stablehlo.reduce(%v1174 init: %v1175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1178 = stablehlo.divide %v1177, %v1176 : tensor<256xf32>
    %arsums3b4g1mu = "stablehlo.all_reduce"(%v1178) ({
    ^bb0(%aras3b4g1mu: tensor<f32>, %arbs3b4g1mu: tensor<f32>):
      %aradds3b4g1mu = stablehlo.add %aras3b4g1mu, %arbs3b4g1mu : tensor<f32>
      stablehlo.return %aradds3b4g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g1mu = stablehlo.divide %arsums3b4g1mu, %arns3b4g1mu : tensor<256xf32>
    %v1179 = stablehlo.reshape %v1173 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1181 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v1182 = stablehlo.reduce(%v1179 init: %v1180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1183 = stablehlo.divide %v1182, %v1181 : tensor<256xf32>
    %v1184 = stablehlo.broadcast_in_dim %v1183, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1185 = stablehlo.subtract %v1179, %v1184 : tensor<128x256x14x14xf32>
    %v1186 = stablehlo.multiply %v1185, %v1185 : tensor<128x256x14x14xf32>
    %v1187 = stablehlo.reduce(%v1186 init: %v1180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
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
    %v1193 = stablehlo.reshape %v1173 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1194 = stablehlo.slice %v1192 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1195 = stablehlo.slice %v1192 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1196 = stablehlo.broadcast_in_dim %v1194, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1197 = stablehlo.broadcast_in_dim %v1195, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1198 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1199 = stablehlo.add %v1197, %v1198 : tensor<128x256x14x14xf32>
    %v1200 = stablehlo.rsqrt %v1199 : tensor<128x256x14x14xf32>
    %v1201 = stablehlo.subtract %v1193, %v1196 : tensor<128x256x14x14xf32>
    %v1202 = stablehlo.multiply %v1201, %v1200 : tensor<128x256x14x14xf32>
    %v1203 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1204 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1205 = stablehlo.multiply %v1202, %v1203 : tensor<128x256x14x14xf32>
    %v1206 = stablehlo.add %v1205, %v1204 : tensor<128x256x14x14xf32>
    %v1207 = stablehlo.reshape %v1206 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1208 = stablehlo.constant dense<0.0> : tensor<128x50176xf32>
    %v1209 = stablehlo.maximum %v1207, %v1208 : tensor<128x50176xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1211 = stablehlo.convolution(%v1210, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v1212 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1213 = stablehlo.add %v1211, %v1212 : tensor<128x256x14x14xf32>
    %v1214 = stablehlo.reshape %v1213 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1215 = stablehlo.reshape %v1214 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1216 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1217 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v1218 = stablehlo.reduce(%v1215 init: %v1216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1219 = stablehlo.divide %v1218, %v1217 : tensor<256xf32>
    %arsums3b4g2mu = "stablehlo.all_reduce"(%v1219) ({
    ^bb0(%aras3b4g2mu: tensor<f32>, %arbs3b4g2mu: tensor<f32>):
      %aradds3b4g2mu = stablehlo.add %aras3b4g2mu, %arbs3b4g2mu : tensor<f32>
      stablehlo.return %aradds3b4g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2mu = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g2mu = stablehlo.divide %arsums3b4g2mu, %arns3b4g2mu : tensor<256xf32>
    %v1220 = stablehlo.reshape %v1214 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1221 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1222 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v1223 = stablehlo.reduce(%v1220 init: %v1221) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1224 = stablehlo.divide %v1223, %v1222 : tensor<256xf32>
    %v1225 = stablehlo.broadcast_in_dim %v1224, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1226 = stablehlo.subtract %v1220, %v1225 : tensor<128x256x14x14xf32>
    %v1227 = stablehlo.multiply %v1226, %v1226 : tensor<128x256x14x14xf32>
    %v1228 = stablehlo.reduce(%v1227 init: %v1221) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
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
    %v1234 = stablehlo.reshape %v1214 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1235 = stablehlo.slice %v1233 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1236 = stablehlo.slice %v1233 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1237 = stablehlo.broadcast_in_dim %v1235, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1238 = stablehlo.broadcast_in_dim %v1236, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1239 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v1240 = stablehlo.add %v1238, %v1239 : tensor<128x256x14x14xf32>
    %v1241 = stablehlo.rsqrt %v1240 : tensor<128x256x14x14xf32>
    %v1242 = stablehlo.subtract %v1234, %v1237 : tensor<128x256x14x14xf32>
    %v1243 = stablehlo.multiply %v1242, %v1241 : tensor<128x256x14x14xf32>
    %v1244 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1245 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v1246 = stablehlo.multiply %v1243, %v1244 : tensor<128x256x14x14xf32>
    %v1247 = stablehlo.add %v1246, %v1245 : tensor<128x256x14x14xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1249 = stablehlo.reshape %v1248 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1250 = stablehlo.reshape %v1168 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1251 = stablehlo.add %v1249, %v1250 : tensor<128x256x14x14xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1253 = stablehlo.reshape %v1252 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1254 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v1255 = stablehlo.maximum %v1253, %v1254 : tensor<128x256x14x14xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1258 = stablehlo.convolution(%v1257, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1259 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1260 = stablehlo.add %v1258, %v1259 : tensor<128x512x7x7xf32>
    %v1261 = stablehlo.reshape %v1260 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1262 = stablehlo.reshape %v1261 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1263 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1264 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1265 = stablehlo.reduce(%v1262 init: %v1263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1266 = stablehlo.divide %v1265, %v1264 : tensor<512xf32>
    %arsumd4g1mu = "stablehlo.all_reduce"(%v1266) ({
    ^bb0(%arad4g1mu: tensor<f32>, %arbd4g1mu: tensor<f32>):
      %araddd4g1mu = stablehlo.add %arad4g1mu, %arbd4g1mu : tensor<f32>
      stablehlo.return %araddd4g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g1mu = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g1mu = stablehlo.divide %arsumd4g1mu, %arnd4g1mu : tensor<512xf32>
    %v1267 = stablehlo.reshape %v1261 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1269 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1270 = stablehlo.reduce(%v1267 init: %v1268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1271 = stablehlo.divide %v1270, %v1269 : tensor<512xf32>
    %v1272 = stablehlo.broadcast_in_dim %v1271, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1273 = stablehlo.subtract %v1267, %v1272 : tensor<128x512x7x7xf32>
    %v1274 = stablehlo.multiply %v1273, %v1273 : tensor<128x512x7x7xf32>
    %v1275 = stablehlo.reduce(%v1274 init: %v1268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
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
    %v1281 = stablehlo.reshape %v1261 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1282 = stablehlo.slice %v1280 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1283 = stablehlo.slice %v1280 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1284 = stablehlo.broadcast_in_dim %v1282, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1285 = stablehlo.broadcast_in_dim %v1283, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1286 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1287 = stablehlo.add %v1285, %v1286 : tensor<128x512x7x7xf32>
    %v1288 = stablehlo.rsqrt %v1287 : tensor<128x512x7x7xf32>
    %v1289 = stablehlo.subtract %v1281, %v1284 : tensor<128x512x7x7xf32>
    %v1290 = stablehlo.multiply %v1289, %v1288 : tensor<128x512x7x7xf32>
    %v1291 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1292 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1293 = stablehlo.multiply %v1290, %v1291 : tensor<128x512x7x7xf32>
    %v1294 = stablehlo.add %v1293, %v1292 : tensor<128x512x7x7xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1296 = stablehlo.constant dense<0.0> : tensor<128x25088xf32>
    %v1297 = stablehlo.maximum %v1295, %v1296 : tensor<128x25088xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1299 = stablehlo.convolution(%v1298, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1300 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1301 = stablehlo.add %v1299, %v1300 : tensor<128x512x7x7xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1304 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1305 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1306 = stablehlo.reduce(%v1303 init: %v1304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1307 = stablehlo.divide %v1306, %v1305 : tensor<512xf32>
    %arsumd4g2mu = "stablehlo.all_reduce"(%v1307) ({
    ^bb0(%arad4g2mu: tensor<f32>, %arbd4g2mu: tensor<f32>):
      %araddd4g2mu = stablehlo.add %arad4g2mu, %arbd4g2mu : tensor<f32>
      stablehlo.return %araddd4g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g2mu = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g2mu = stablehlo.divide %arsumd4g2mu, %arnd4g2mu : tensor<512xf32>
    %v1308 = stablehlo.reshape %v1302 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1310 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1311 = stablehlo.reduce(%v1308 init: %v1309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1312 = stablehlo.divide %v1311, %v1310 : tensor<512xf32>
    %v1313 = stablehlo.broadcast_in_dim %v1312, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1314 = stablehlo.subtract %v1308, %v1313 : tensor<128x512x7x7xf32>
    %v1315 = stablehlo.multiply %v1314, %v1314 : tensor<128x512x7x7xf32>
    %v1316 = stablehlo.reduce(%v1315 init: %v1309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
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
    %v1322 = stablehlo.reshape %v1302 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1323 = stablehlo.slice %v1321 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1324 = stablehlo.slice %v1321 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1325 = stablehlo.broadcast_in_dim %v1323, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1326 = stablehlo.broadcast_in_dim %v1324, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1327 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1328 = stablehlo.add %v1326, %v1327 : tensor<128x512x7x7xf32>
    %v1329 = stablehlo.rsqrt %v1328 : tensor<128x512x7x7xf32>
    %v1330 = stablehlo.subtract %v1322, %v1325 : tensor<128x512x7x7xf32>
    %v1331 = stablehlo.multiply %v1330, %v1329 : tensor<128x512x7x7xf32>
    %v1332 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1333 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1334 = stablehlo.multiply %v1331, %v1332 : tensor<128x512x7x7xf32>
    %v1335 = stablehlo.add %v1334, %v1333 : tensor<128x512x7x7xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1337 = stablehlo.reshape %v1256 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v1338 = stablehlo.convolution(%v1337, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<512x256x1x1xf32>) -> tensor<128x512x7x7xf32>
    %v1339 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1340 = stablehlo.add %v1338, %v1339 : tensor<128x512x7x7xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1342 = stablehlo.reshape %v1341 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1344 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1345 = stablehlo.reduce(%v1342 init: %v1343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1346 = stablehlo.divide %v1345, %v1344 : tensor<512xf32>
    %arsumd4gpmu = "stablehlo.all_reduce"(%v1346) ({
    ^bb0(%arad4gpmu: tensor<f32>, %arbd4gpmu: tensor<f32>):
      %araddd4gpmu = stablehlo.add %arad4gpmu, %arbd4gpmu : tensor<f32>
      stablehlo.return %araddd4gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4gpmu = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4gpmu = stablehlo.divide %arsumd4gpmu, %arnd4gpmu : tensor<512xf32>
    %v1347 = stablehlo.reshape %v1341 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1348 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1349 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1350 = stablehlo.reduce(%v1347 init: %v1348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1351 = stablehlo.divide %v1350, %v1349 : tensor<512xf32>
    %v1352 = stablehlo.broadcast_in_dim %v1351, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1353 = stablehlo.subtract %v1347, %v1352 : tensor<128x512x7x7xf32>
    %v1354 = stablehlo.multiply %v1353, %v1353 : tensor<128x512x7x7xf32>
    %v1355 = stablehlo.reduce(%v1354 init: %v1348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
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
    %v1361 = stablehlo.reshape %v1341 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1362 = stablehlo.slice %v1360 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1363 = stablehlo.slice %v1360 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1364 = stablehlo.broadcast_in_dim %v1362, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1365 = stablehlo.broadcast_in_dim %v1363, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1366 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1367 = stablehlo.add %v1365, %v1366 : tensor<128x512x7x7xf32>
    %v1368 = stablehlo.rsqrt %v1367 : tensor<128x512x7x7xf32>
    %v1369 = stablehlo.subtract %v1361, %v1364 : tensor<128x512x7x7xf32>
    %v1370 = stablehlo.multiply %v1369, %v1368 : tensor<128x512x7x7xf32>
    %v1371 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1372 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1373 = stablehlo.multiply %v1370, %v1371 : tensor<128x512x7x7xf32>
    %v1374 = stablehlo.add %v1373, %v1372 : tensor<128x512x7x7xf32>
    %v1375 = stablehlo.reshape %v1374 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1376 = stablehlo.add %v1336, %v1375 : tensor<128x25088xf32>
    %v1377 = stablehlo.constant dense<0.0> : tensor<128x25088xf32>
    %v1378 = stablehlo.maximum %v1376, %v1377 : tensor<128x25088xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1380 = stablehlo.convolution(%v1379, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1381 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1382 = stablehlo.add %v1380, %v1381 : tensor<128x512x7x7xf32>
    %v1383 = stablehlo.reshape %v1382 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1384 = stablehlo.reshape %v1383 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1386 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1387 = stablehlo.reduce(%v1384 init: %v1385) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1388 = stablehlo.divide %v1387, %v1386 : tensor<512xf32>
    %arsums4b0g1mu = "stablehlo.all_reduce"(%v1388) ({
    ^bb0(%aras4b0g1mu: tensor<f32>, %arbs4b0g1mu: tensor<f32>):
      %aradds4b0g1mu = stablehlo.add %aras4b0g1mu, %arbs4b0g1mu : tensor<f32>
      stablehlo.return %aradds4b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1mu = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g1mu = stablehlo.divide %arsums4b0g1mu, %arns4b0g1mu : tensor<512xf32>
    %v1389 = stablehlo.reshape %v1383 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1391 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1392 = stablehlo.reduce(%v1389 init: %v1390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1393 = stablehlo.divide %v1392, %v1391 : tensor<512xf32>
    %v1394 = stablehlo.broadcast_in_dim %v1393, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1395 = stablehlo.subtract %v1389, %v1394 : tensor<128x512x7x7xf32>
    %v1396 = stablehlo.multiply %v1395, %v1395 : tensor<128x512x7x7xf32>
    %v1397 = stablehlo.reduce(%v1396 init: %v1390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
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
    %v1403 = stablehlo.reshape %v1383 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1404 = stablehlo.slice %v1402 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1405 = stablehlo.slice %v1402 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1406 = stablehlo.broadcast_in_dim %v1404, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1407 = stablehlo.broadcast_in_dim %v1405, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1408 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1409 = stablehlo.add %v1407, %v1408 : tensor<128x512x7x7xf32>
    %v1410 = stablehlo.rsqrt %v1409 : tensor<128x512x7x7xf32>
    %v1411 = stablehlo.subtract %v1403, %v1406 : tensor<128x512x7x7xf32>
    %v1412 = stablehlo.multiply %v1411, %v1410 : tensor<128x512x7x7xf32>
    %v1413 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1414 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1415 = stablehlo.multiply %v1412, %v1413 : tensor<128x512x7x7xf32>
    %v1416 = stablehlo.add %v1415, %v1414 : tensor<128x512x7x7xf32>
    %v1417 = stablehlo.reshape %v1416 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1418 = stablehlo.constant dense<0.0> : tensor<128x25088xf32>
    %v1419 = stablehlo.maximum %v1417, %v1418 : tensor<128x25088xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1421 = stablehlo.convolution(%v1420, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1422 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1423 = stablehlo.add %v1421, %v1422 : tensor<128x512x7x7xf32>
    %v1424 = stablehlo.reshape %v1423 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1425 = stablehlo.reshape %v1424 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1426 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1427 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1428 = stablehlo.reduce(%v1425 init: %v1426) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1429 = stablehlo.divide %v1428, %v1427 : tensor<512xf32>
    %arsums4b0g2mu = "stablehlo.all_reduce"(%v1429) ({
    ^bb0(%aras4b0g2mu: tensor<f32>, %arbs4b0g2mu: tensor<f32>):
      %aradds4b0g2mu = stablehlo.add %aras4b0g2mu, %arbs4b0g2mu : tensor<f32>
      stablehlo.return %aradds4b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2mu = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g2mu = stablehlo.divide %arsums4b0g2mu, %arns4b0g2mu : tensor<512xf32>
    %v1430 = stablehlo.reshape %v1424 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1432 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1433 = stablehlo.reduce(%v1430 init: %v1431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1434 = stablehlo.divide %v1433, %v1432 : tensor<512xf32>
    %v1435 = stablehlo.broadcast_in_dim %v1434, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1436 = stablehlo.subtract %v1430, %v1435 : tensor<128x512x7x7xf32>
    %v1437 = stablehlo.multiply %v1436, %v1436 : tensor<128x512x7x7xf32>
    %v1438 = stablehlo.reduce(%v1437 init: %v1431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
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
    %v1444 = stablehlo.reshape %v1424 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1445 = stablehlo.slice %v1443 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1446 = stablehlo.slice %v1443 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1447 = stablehlo.broadcast_in_dim %v1445, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1448 = stablehlo.broadcast_in_dim %v1446, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1449 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1450 = stablehlo.add %v1448, %v1449 : tensor<128x512x7x7xf32>
    %v1451 = stablehlo.rsqrt %v1450 : tensor<128x512x7x7xf32>
    %v1452 = stablehlo.subtract %v1444, %v1447 : tensor<128x512x7x7xf32>
    %v1453 = stablehlo.multiply %v1452, %v1451 : tensor<128x512x7x7xf32>
    %v1454 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1455 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1456 = stablehlo.multiply %v1453, %v1454 : tensor<128x512x7x7xf32>
    %v1457 = stablehlo.add %v1456, %v1455 : tensor<128x512x7x7xf32>
    %v1458 = stablehlo.reshape %v1457 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1459 = stablehlo.reshape %v1458 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1460 = stablehlo.reshape %v1378 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1461 = stablehlo.add %v1459, %v1460 : tensor<128x512x7x7xf32>
    %v1462 = stablehlo.reshape %v1461 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1463 = stablehlo.reshape %v1462 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1464 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1465 = stablehlo.maximum %v1463, %v1464 : tensor<128x512x7x7xf32>
    %v1466 = stablehlo.reshape %v1465 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1467 = stablehlo.reshape %v1466 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1468 = stablehlo.convolution(%v1467, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1469 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1470 = stablehlo.add %v1468, %v1469 : tensor<128x512x7x7xf32>
    %v1471 = stablehlo.reshape %v1470 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1472 = stablehlo.reshape %v1471 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1474 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1475 = stablehlo.reduce(%v1472 init: %v1473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1476 = stablehlo.divide %v1475, %v1474 : tensor<512xf32>
    %arsums4b1g1mu = "stablehlo.all_reduce"(%v1476) ({
    ^bb0(%aras4b1g1mu: tensor<f32>, %arbs4b1g1mu: tensor<f32>):
      %aradds4b1g1mu = stablehlo.add %aras4b1g1mu, %arbs4b1g1mu : tensor<f32>
      stablehlo.return %aradds4b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1mu = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g1mu = stablehlo.divide %arsums4b1g1mu, %arns4b1g1mu : tensor<512xf32>
    %v1477 = stablehlo.reshape %v1471 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1479 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1480 = stablehlo.reduce(%v1477 init: %v1478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1481 = stablehlo.divide %v1480, %v1479 : tensor<512xf32>
    %v1482 = stablehlo.broadcast_in_dim %v1481, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1483 = stablehlo.subtract %v1477, %v1482 : tensor<128x512x7x7xf32>
    %v1484 = stablehlo.multiply %v1483, %v1483 : tensor<128x512x7x7xf32>
    %v1485 = stablehlo.reduce(%v1484 init: %v1478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
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
    %v1491 = stablehlo.reshape %v1471 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1492 = stablehlo.slice %v1490 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1493 = stablehlo.slice %v1490 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1494 = stablehlo.broadcast_in_dim %v1492, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1495 = stablehlo.broadcast_in_dim %v1493, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1496 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1497 = stablehlo.add %v1495, %v1496 : tensor<128x512x7x7xf32>
    %v1498 = stablehlo.rsqrt %v1497 : tensor<128x512x7x7xf32>
    %v1499 = stablehlo.subtract %v1491, %v1494 : tensor<128x512x7x7xf32>
    %v1500 = stablehlo.multiply %v1499, %v1498 : tensor<128x512x7x7xf32>
    %v1501 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1502 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1503 = stablehlo.multiply %v1500, %v1501 : tensor<128x512x7x7xf32>
    %v1504 = stablehlo.add %v1503, %v1502 : tensor<128x512x7x7xf32>
    %v1505 = stablehlo.reshape %v1504 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1506 = stablehlo.constant dense<0.0> : tensor<128x25088xf32>
    %v1507 = stablehlo.maximum %v1505, %v1506 : tensor<128x25088xf32>
    %v1508 = stablehlo.reshape %v1507 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1509 = stablehlo.convolution(%v1508, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1510 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1511 = stablehlo.add %v1509, %v1510 : tensor<128x512x7x7xf32>
    %v1512 = stablehlo.reshape %v1511 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1513 = stablehlo.reshape %v1512 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1515 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1516 = stablehlo.reduce(%v1513 init: %v1514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1517 = stablehlo.divide %v1516, %v1515 : tensor<512xf32>
    %arsums4b1g2mu = "stablehlo.all_reduce"(%v1517) ({
    ^bb0(%aras4b1g2mu: tensor<f32>, %arbs4b1g2mu: tensor<f32>):
      %aradds4b1g2mu = stablehlo.add %aras4b1g2mu, %arbs4b1g2mu : tensor<f32>
      stablehlo.return %aradds4b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2mu = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g2mu = stablehlo.divide %arsums4b1g2mu, %arns4b1g2mu : tensor<512xf32>
    %v1518 = stablehlo.reshape %v1512 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1519 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1520 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1521 = stablehlo.reduce(%v1518 init: %v1519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1522 = stablehlo.divide %v1521, %v1520 : tensor<512xf32>
    %v1523 = stablehlo.broadcast_in_dim %v1522, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1524 = stablehlo.subtract %v1518, %v1523 : tensor<128x512x7x7xf32>
    %v1525 = stablehlo.multiply %v1524, %v1524 : tensor<128x512x7x7xf32>
    %v1526 = stablehlo.reduce(%v1525 init: %v1519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
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
    %v1532 = stablehlo.reshape %v1512 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1533 = stablehlo.slice %v1531 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1534 = stablehlo.slice %v1531 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1535 = stablehlo.broadcast_in_dim %v1533, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1536 = stablehlo.broadcast_in_dim %v1534, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1537 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1538 = stablehlo.add %v1536, %v1537 : tensor<128x512x7x7xf32>
    %v1539 = stablehlo.rsqrt %v1538 : tensor<128x512x7x7xf32>
    %v1540 = stablehlo.subtract %v1532, %v1535 : tensor<128x512x7x7xf32>
    %v1541 = stablehlo.multiply %v1540, %v1539 : tensor<128x512x7x7xf32>
    %v1542 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1543 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1544 = stablehlo.multiply %v1541, %v1542 : tensor<128x512x7x7xf32>
    %v1545 = stablehlo.add %v1544, %v1543 : tensor<128x512x7x7xf32>
    %v1546 = stablehlo.reshape %v1545 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1547 = stablehlo.reshape %v1546 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1548 = stablehlo.reshape %v1466 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1549 = stablehlo.add %v1547, %v1548 : tensor<128x512x7x7xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1551 = stablehlo.reshape %v1550 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1552 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1553 = stablehlo.maximum %v1551, %v1552 : tensor<128x512x7x7xf32>
    %v1554 = stablehlo.reshape %v1553 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1557 = stablehlo.reduce(%v1555 init: %v1556) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<128x512xf32>
    %v1558 = stablehlo.constant dense<49.0> : tensor<128x512xf32>
    %v1559 = stablehlo.divide %v1557, %v1558 : tensor<128x512xf32>
    %v1560 = stablehlo.dot_general %v1559, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v1561 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v1562 = stablehlo.add %v1560, %v1561 : tensor<128x10xf32>
    %v1563 = stablehlo.reshape %v1562 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v1564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1565 = stablehlo.exponential %v1563 : tensor<128x1x10xf32>
    %v1566 = stablehlo.reduce(%v1565 init: %v1564) applies stablehlo.add across dimensions = [2] : (tensor<128x1x10xf32>, tensor<f32>) -> tensor<128x1xf32>
    %v1567 = stablehlo.broadcast_in_dim %v1566, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x1x10xf32>
    %v1568 = stablehlo.divide %v1565, %v1567 : tensor<128x1x10xf32>
    %v1569 = stablehlo.reshape %v1568 : (tensor<128x1x10xf32>) -> tensor<128x10xf32>
    %v1570 = stablehlo.subtract %v1569, %onehot : tensor<128x10xf32>
    %v1571 = stablehlo.constant dense<0.100000> : tensor<128x10xf32>
    %v1572 = stablehlo.multiply %onehot, %v1571 : tensor<128x10xf32>
    %v1573 = stablehlo.add %v1570, %v1572 : tensor<128x10xf32>
    %v1574 = stablehlo.constant dense<-0.010000> : tensor<128x10xf32>
    %v1575 = stablehlo.add %v1573, %v1574 : tensor<128x10xf32>
    %v1576 = stablehlo.constant dense<128.0> : tensor<128x10xf32>
    %v1577 = stablehlo.divide %v1575, %v1576 : tensor<128x10xf32>
    %v1578 = stablehlo.reshape %v1577 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v1579 = stablehlo.dot_general %v1578, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x10xf32>, tensor<512x10xf32>) -> tensor<128x1x512xf32>
    %v1580 = stablehlo.reshape %v1579 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v1581 = stablehlo.dot_general %v1559, %v1577, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v1582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1583 = stablehlo.reduce(%v1577 init: %v1582) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1584 = stablehlo.broadcast_in_dim %v1580, dims = [0, 1] : (tensor<128x512xf32>) -> tensor<128x512x7x7xf32>
    %v1585 = stablehlo.constant dense<49.0> : tensor<128x512x7x7xf32>
    %v1586 = stablehlo.divide %v1584, %v1585 : tensor<128x512x7x7xf32>
    %v1587 = stablehlo.reshape %v1586 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1588 = stablehlo.reshape %v1587 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1589 = stablehlo.reshape %v1550 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1590 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1591 = stablehlo.compare GT, %v1589, %v1590 : (tensor<128x512x7x7xf32>, tensor<128x512x7x7xf32>) -> tensor<128x512x7x7xi1>
    %v1592 = stablehlo.select %v1591, %v1588, %v1590 : tensor<128x512x7x7xi1>, tensor<128x512x7x7xf32>
    %v1593 = stablehlo.reshape %v1592 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1594 = stablehlo.reshape %v1512 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1595 = stablehlo.slice %v1531 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1596 = stablehlo.slice %v1531 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1597 = stablehlo.broadcast_in_dim %v1595, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1598 = stablehlo.broadcast_in_dim %v1596, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1599 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1600 = stablehlo.add %v1598, %v1599 : tensor<128x512x7x7xf32>
    %v1601 = stablehlo.rsqrt %v1600 : tensor<128x512x7x7xf32>
    %v1602 = stablehlo.subtract %v1594, %v1597 : tensor<128x512x7x7xf32>
    %v1603 = stablehlo.multiply %v1602, %v1601 : tensor<128x512x7x7xf32>
    %v1604 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1605 = stablehlo.reshape %v1593 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1606 = stablehlo.multiply %v1604, %v1605 : tensor<128x512x7x7xf32>
    %v1607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1608 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1609 = stablehlo.reduce(%v1606 init: %v1607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1610 = stablehlo.divide %v1609, %v1608 : tensor<512xf32>
    %v1611 = stablehlo.multiply %v1603, %v1606 : tensor<128x512x7x7xf32>
    %v1612 = stablehlo.reduce(%v1611 init: %v1607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
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
    %v1616 = stablehlo.reshape %v1512 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1617 = stablehlo.slice %armeans4b1g2dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1618 = stablehlo.slice %armeans4b1g2dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1619 = stablehlo.slice %armeans4b1g2dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1620 = stablehlo.slice %armeans4b1g2dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1621 = stablehlo.broadcast_in_dim %v1617, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1622 = stablehlo.broadcast_in_dim %v1618, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1623 = stablehlo.broadcast_in_dim %v1619, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1624 = stablehlo.broadcast_in_dim %v1620, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1625 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1626 = stablehlo.add %v1622, %v1625 : tensor<128x512x7x7xf32>
    %v1627 = stablehlo.rsqrt %v1626 : tensor<128x512x7x7xf32>
    %v1628 = stablehlo.subtract %v1616, %v1621 : tensor<128x512x7x7xf32>
    %v1629 = stablehlo.multiply %v1628, %v1627 : tensor<128x512x7x7xf32>
    %v1630 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1631 = stablehlo.reshape %v1593 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1632 = stablehlo.multiply %v1630, %v1631 : tensor<128x512x7x7xf32>
    %v1633 = stablehlo.subtract %v1632, %v1623 : tensor<128x512x7x7xf32>
    %v1634 = stablehlo.multiply %v1629, %v1624 : tensor<128x512x7x7xf32>
    %v1635 = stablehlo.subtract %v1633, %v1634 : tensor<128x512x7x7xf32>
    %v1636 = stablehlo.multiply %v1627, %v1635 : tensor<128x512x7x7xf32>
    %v1637 = stablehlo.reshape %v1636 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1638 = stablehlo.reshape %v1637 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1639 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1640 = stablehlo.transpose %v1639, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1641 = stablehlo.convolution(%v1638, %v1640)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1642 = stablehlo.reshape %v1641 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1644 = stablehlo.reshape %v1505 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1645 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1646 = stablehlo.compare GT, %v1644, %v1645 : (tensor<128x512x7x7xf32>, tensor<128x512x7x7xf32>) -> tensor<128x512x7x7xi1>
    %v1647 = stablehlo.select %v1646, %v1643, %v1645 : tensor<128x512x7x7xi1>, tensor<128x512x7x7xf32>
    %v1648 = stablehlo.reshape %v1647 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1649 = stablehlo.reshape %v1471 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1650 = stablehlo.slice %v1490 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1651 = stablehlo.slice %v1490 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1652 = stablehlo.broadcast_in_dim %v1650, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1653 = stablehlo.broadcast_in_dim %v1651, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1654 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1655 = stablehlo.add %v1653, %v1654 : tensor<128x512x7x7xf32>
    %v1656 = stablehlo.rsqrt %v1655 : tensor<128x512x7x7xf32>
    %v1657 = stablehlo.subtract %v1649, %v1652 : tensor<128x512x7x7xf32>
    %v1658 = stablehlo.multiply %v1657, %v1656 : tensor<128x512x7x7xf32>
    %v1659 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1660 = stablehlo.reshape %v1648 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1661 = stablehlo.multiply %v1659, %v1660 : tensor<128x512x7x7xf32>
    %v1662 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1663 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1664 = stablehlo.reduce(%v1661 init: %v1662) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1665 = stablehlo.divide %v1664, %v1663 : tensor<512xf32>
    %v1666 = stablehlo.multiply %v1658, %v1661 : tensor<128x512x7x7xf32>
    %v1667 = stablehlo.reduce(%v1666 init: %v1662) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
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
    %v1671 = stablehlo.reshape %v1471 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1672 = stablehlo.slice %armeans4b1g1dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1673 = stablehlo.slice %armeans4b1g1dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1674 = stablehlo.slice %armeans4b1g1dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1675 = stablehlo.slice %armeans4b1g1dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1676 = stablehlo.broadcast_in_dim %v1672, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1677 = stablehlo.broadcast_in_dim %v1673, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1678 = stablehlo.broadcast_in_dim %v1674, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1679 = stablehlo.broadcast_in_dim %v1675, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1680 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1681 = stablehlo.add %v1677, %v1680 : tensor<128x512x7x7xf32>
    %v1682 = stablehlo.rsqrt %v1681 : tensor<128x512x7x7xf32>
    %v1683 = stablehlo.subtract %v1671, %v1676 : tensor<128x512x7x7xf32>
    %v1684 = stablehlo.multiply %v1683, %v1682 : tensor<128x512x7x7xf32>
    %v1685 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1686 = stablehlo.reshape %v1648 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1687 = stablehlo.multiply %v1685, %v1686 : tensor<128x512x7x7xf32>
    %v1688 = stablehlo.subtract %v1687, %v1678 : tensor<128x512x7x7xf32>
    %v1689 = stablehlo.multiply %v1684, %v1679 : tensor<128x512x7x7xf32>
    %v1690 = stablehlo.subtract %v1688, %v1689 : tensor<128x512x7x7xf32>
    %v1691 = stablehlo.multiply %v1682, %v1690 : tensor<128x512x7x7xf32>
    %v1692 = stablehlo.reshape %v1691 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1694 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1695 = stablehlo.transpose %v1694, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1696 = stablehlo.convolution(%v1693, %v1695)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1697 = stablehlo.reshape %v1696 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1698 = stablehlo.reshape %v1697 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1699 = stablehlo.reshape %v1593 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1700 = stablehlo.add %v1698, %v1699 : tensor<128x512x7x7xf32>
    %v1701 = stablehlo.reshape %v1700 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1702 = stablehlo.reshape %v1466 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1703 = stablehlo.reshape %v1692 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1704 = stablehlo.transpose %v1702, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1705 = stablehlo.transpose %v1703, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1706 = stablehlo.convolution(%v1704, %v1705)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x128x7x7xf32>, tensor<512x128x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1707 = stablehlo.transpose %v1706, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1708 = stablehlo.reshape %v1471 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1709 = stablehlo.slice %v1490 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1710 = stablehlo.slice %v1490 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1711 = stablehlo.broadcast_in_dim %v1709, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1712 = stablehlo.broadcast_in_dim %v1710, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1713 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1714 = stablehlo.add %v1712, %v1713 : tensor<128x512x7x7xf32>
    %v1715 = stablehlo.rsqrt %v1714 : tensor<128x512x7x7xf32>
    %v1716 = stablehlo.subtract %v1708, %v1711 : tensor<128x512x7x7xf32>
    %v1717 = stablehlo.multiply %v1716, %v1715 : tensor<128x512x7x7xf32>
    %v1718 = stablehlo.reshape %v1648 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1719 = stablehlo.multiply %v1718, %v1717 : tensor<128x512x7x7xf32>
    %v1720 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1721 = stablehlo.reduce(%v1719 init: %v1720) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1722 = stablehlo.reshape %v1648 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1723 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1724 = stablehlo.reduce(%v1722 init: %v1723) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1725 = stablehlo.reshape %v1507 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1726 = stablehlo.reshape %v1637 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1727 = stablehlo.transpose %v1725, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1728 = stablehlo.transpose %v1726, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1729 = stablehlo.convolution(%v1727, %v1728)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x128x7x7xf32>, tensor<512x128x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1730 = stablehlo.transpose %v1729, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1731 = stablehlo.reshape %v1512 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1732 = stablehlo.slice %v1531 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1733 = stablehlo.slice %v1531 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1734 = stablehlo.broadcast_in_dim %v1732, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1735 = stablehlo.broadcast_in_dim %v1733, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1736 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1737 = stablehlo.add %v1735, %v1736 : tensor<128x512x7x7xf32>
    %v1738 = stablehlo.rsqrt %v1737 : tensor<128x512x7x7xf32>
    %v1739 = stablehlo.subtract %v1731, %v1734 : tensor<128x512x7x7xf32>
    %v1740 = stablehlo.multiply %v1739, %v1738 : tensor<128x512x7x7xf32>
    %v1741 = stablehlo.reshape %v1593 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1742 = stablehlo.multiply %v1741, %v1740 : tensor<128x512x7x7xf32>
    %v1743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1744 = stablehlo.reduce(%v1742 init: %v1743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1745 = stablehlo.reshape %v1593 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1747 = stablehlo.reduce(%v1745 init: %v1746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1748 = stablehlo.reshape %v1701 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1749 = stablehlo.reshape %v1462 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1750 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1751 = stablehlo.compare GT, %v1749, %v1750 : (tensor<128x512x7x7xf32>, tensor<128x512x7x7xf32>) -> tensor<128x512x7x7xi1>
    %v1752 = stablehlo.select %v1751, %v1748, %v1750 : tensor<128x512x7x7xi1>, tensor<128x512x7x7xf32>
    %v1753 = stablehlo.reshape %v1752 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1754 = stablehlo.reshape %v1424 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1755 = stablehlo.slice %v1443 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1756 = stablehlo.slice %v1443 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1757 = stablehlo.broadcast_in_dim %v1755, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1758 = stablehlo.broadcast_in_dim %v1756, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1759 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1760 = stablehlo.add %v1758, %v1759 : tensor<128x512x7x7xf32>
    %v1761 = stablehlo.rsqrt %v1760 : tensor<128x512x7x7xf32>
    %v1762 = stablehlo.subtract %v1754, %v1757 : tensor<128x512x7x7xf32>
    %v1763 = stablehlo.multiply %v1762, %v1761 : tensor<128x512x7x7xf32>
    %v1764 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1765 = stablehlo.reshape %v1753 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1766 = stablehlo.multiply %v1764, %v1765 : tensor<128x512x7x7xf32>
    %v1767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1768 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1769 = stablehlo.reduce(%v1766 init: %v1767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1770 = stablehlo.divide %v1769, %v1768 : tensor<512xf32>
    %v1771 = stablehlo.multiply %v1763, %v1766 : tensor<128x512x7x7xf32>
    %v1772 = stablehlo.reduce(%v1771 init: %v1767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
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
    %v1776 = stablehlo.reshape %v1424 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1777 = stablehlo.slice %armeans4b0g2dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1778 = stablehlo.slice %armeans4b0g2dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1779 = stablehlo.slice %armeans4b0g2dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1780 = stablehlo.slice %armeans4b0g2dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1781 = stablehlo.broadcast_in_dim %v1777, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1782 = stablehlo.broadcast_in_dim %v1778, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1783 = stablehlo.broadcast_in_dim %v1779, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1784 = stablehlo.broadcast_in_dim %v1780, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1785 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1786 = stablehlo.add %v1782, %v1785 : tensor<128x512x7x7xf32>
    %v1787 = stablehlo.rsqrt %v1786 : tensor<128x512x7x7xf32>
    %v1788 = stablehlo.subtract %v1776, %v1781 : tensor<128x512x7x7xf32>
    %v1789 = stablehlo.multiply %v1788, %v1787 : tensor<128x512x7x7xf32>
    %v1790 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1791 = stablehlo.reshape %v1753 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1792 = stablehlo.multiply %v1790, %v1791 : tensor<128x512x7x7xf32>
    %v1793 = stablehlo.subtract %v1792, %v1783 : tensor<128x512x7x7xf32>
    %v1794 = stablehlo.multiply %v1789, %v1784 : tensor<128x512x7x7xf32>
    %v1795 = stablehlo.subtract %v1793, %v1794 : tensor<128x512x7x7xf32>
    %v1796 = stablehlo.multiply %v1787, %v1795 : tensor<128x512x7x7xf32>
    %v1797 = stablehlo.reshape %v1796 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1798 = stablehlo.reshape %v1797 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1799 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1800 = stablehlo.transpose %v1799, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1801 = stablehlo.convolution(%v1798, %v1800)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1802 = stablehlo.reshape %v1801 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1803 = stablehlo.reshape %v1802 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1804 = stablehlo.reshape %v1417 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1805 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1806 = stablehlo.compare GT, %v1804, %v1805 : (tensor<128x512x7x7xf32>, tensor<128x512x7x7xf32>) -> tensor<128x512x7x7xi1>
    %v1807 = stablehlo.select %v1806, %v1803, %v1805 : tensor<128x512x7x7xi1>, tensor<128x512x7x7xf32>
    %v1808 = stablehlo.reshape %v1807 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1809 = stablehlo.reshape %v1383 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1810 = stablehlo.slice %v1402 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1811 = stablehlo.slice %v1402 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1812 = stablehlo.broadcast_in_dim %v1810, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1813 = stablehlo.broadcast_in_dim %v1811, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1814 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1815 = stablehlo.add %v1813, %v1814 : tensor<128x512x7x7xf32>
    %v1816 = stablehlo.rsqrt %v1815 : tensor<128x512x7x7xf32>
    %v1817 = stablehlo.subtract %v1809, %v1812 : tensor<128x512x7x7xf32>
    %v1818 = stablehlo.multiply %v1817, %v1816 : tensor<128x512x7x7xf32>
    %v1819 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1820 = stablehlo.reshape %v1808 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1821 = stablehlo.multiply %v1819, %v1820 : tensor<128x512x7x7xf32>
    %v1822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1823 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1824 = stablehlo.reduce(%v1821 init: %v1822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1825 = stablehlo.divide %v1824, %v1823 : tensor<512xf32>
    %v1826 = stablehlo.multiply %v1818, %v1821 : tensor<128x512x7x7xf32>
    %v1827 = stablehlo.reduce(%v1826 init: %v1822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
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
    %v1831 = stablehlo.reshape %v1383 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1832 = stablehlo.slice %armeans4b0g1dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1833 = stablehlo.slice %armeans4b0g1dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1834 = stablehlo.slice %armeans4b0g1dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1835 = stablehlo.slice %armeans4b0g1dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1836 = stablehlo.broadcast_in_dim %v1832, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1837 = stablehlo.broadcast_in_dim %v1833, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1838 = stablehlo.broadcast_in_dim %v1834, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1839 = stablehlo.broadcast_in_dim %v1835, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1840 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1841 = stablehlo.add %v1837, %v1840 : tensor<128x512x7x7xf32>
    %v1842 = stablehlo.rsqrt %v1841 : tensor<128x512x7x7xf32>
    %v1843 = stablehlo.subtract %v1831, %v1836 : tensor<128x512x7x7xf32>
    %v1844 = stablehlo.multiply %v1843, %v1842 : tensor<128x512x7x7xf32>
    %v1845 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1846 = stablehlo.reshape %v1808 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1847 = stablehlo.multiply %v1845, %v1846 : tensor<128x512x7x7xf32>
    %v1848 = stablehlo.subtract %v1847, %v1838 : tensor<128x512x7x7xf32>
    %v1849 = stablehlo.multiply %v1844, %v1839 : tensor<128x512x7x7xf32>
    %v1850 = stablehlo.subtract %v1848, %v1849 : tensor<128x512x7x7xf32>
    %v1851 = stablehlo.multiply %v1842, %v1850 : tensor<128x512x7x7xf32>
    %v1852 = stablehlo.reshape %v1851 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1853 = stablehlo.reshape %v1852 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1854 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1855 = stablehlo.transpose %v1854, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1856 = stablehlo.convolution(%v1853, %v1855)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1857 = stablehlo.reshape %v1856 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1858 = stablehlo.reshape %v1857 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1859 = stablehlo.reshape %v1753 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1860 = stablehlo.add %v1858, %v1859 : tensor<128x512x7x7xf32>
    %v1861 = stablehlo.reshape %v1860 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1862 = stablehlo.reshape %v1378 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1863 = stablehlo.reshape %v1852 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1864 = stablehlo.transpose %v1862, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1865 = stablehlo.transpose %v1863, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1866 = stablehlo.convolution(%v1864, %v1865)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x128x7x7xf32>, tensor<512x128x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1867 = stablehlo.transpose %v1866, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1868 = stablehlo.reshape %v1383 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1869 = stablehlo.slice %v1402 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1870 = stablehlo.slice %v1402 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1871 = stablehlo.broadcast_in_dim %v1869, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1872 = stablehlo.broadcast_in_dim %v1870, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1873 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1874 = stablehlo.add %v1872, %v1873 : tensor<128x512x7x7xf32>
    %v1875 = stablehlo.rsqrt %v1874 : tensor<128x512x7x7xf32>
    %v1876 = stablehlo.subtract %v1868, %v1871 : tensor<128x512x7x7xf32>
    %v1877 = stablehlo.multiply %v1876, %v1875 : tensor<128x512x7x7xf32>
    %v1878 = stablehlo.reshape %v1808 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1879 = stablehlo.multiply %v1878, %v1877 : tensor<128x512x7x7xf32>
    %v1880 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1881 = stablehlo.reduce(%v1879 init: %v1880) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1882 = stablehlo.reshape %v1808 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1884 = stablehlo.reduce(%v1882 init: %v1883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1885 = stablehlo.reshape %v1419 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1886 = stablehlo.reshape %v1797 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1887 = stablehlo.transpose %v1885, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1888 = stablehlo.transpose %v1886, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v1889 = stablehlo.convolution(%v1887, %v1888)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x128x7x7xf32>, tensor<512x128x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1890 = stablehlo.transpose %v1889, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1891 = stablehlo.reshape %v1424 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1892 = stablehlo.slice %v1443 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1893 = stablehlo.slice %v1443 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1894 = stablehlo.broadcast_in_dim %v1892, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1895 = stablehlo.broadcast_in_dim %v1893, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1896 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1897 = stablehlo.add %v1895, %v1896 : tensor<128x512x7x7xf32>
    %v1898 = stablehlo.rsqrt %v1897 : tensor<128x512x7x7xf32>
    %v1899 = stablehlo.subtract %v1891, %v1894 : tensor<128x512x7x7xf32>
    %v1900 = stablehlo.multiply %v1899, %v1898 : tensor<128x512x7x7xf32>
    %v1901 = stablehlo.reshape %v1753 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1902 = stablehlo.multiply %v1901, %v1900 : tensor<128x512x7x7xf32>
    %v1903 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1904 = stablehlo.reduce(%v1902 init: %v1903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1905 = stablehlo.reshape %v1753 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1907 = stablehlo.reduce(%v1905 init: %v1906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1908 = stablehlo.reshape %v1861 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1909 = stablehlo.reshape %v1376 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1910 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1911 = stablehlo.compare GT, %v1909, %v1910 : (tensor<128x512x7x7xf32>, tensor<128x512x7x7xf32>) -> tensor<128x512x7x7xi1>
    %v1912 = stablehlo.select %v1911, %v1908, %v1910 : tensor<128x512x7x7xi1>, tensor<128x512x7x7xf32>
    %v1913 = stablehlo.reshape %v1912 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1914 = stablehlo.reshape %v1302 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1915 = stablehlo.slice %v1321 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1916 = stablehlo.slice %v1321 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1917 = stablehlo.broadcast_in_dim %v1915, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1918 = stablehlo.broadcast_in_dim %v1916, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1919 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1920 = stablehlo.add %v1918, %v1919 : tensor<128x512x7x7xf32>
    %v1921 = stablehlo.rsqrt %v1920 : tensor<128x512x7x7xf32>
    %v1922 = stablehlo.subtract %v1914, %v1917 : tensor<128x512x7x7xf32>
    %v1923 = stablehlo.multiply %v1922, %v1921 : tensor<128x512x7x7xf32>
    %v1924 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1925 = stablehlo.reshape %v1913 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1926 = stablehlo.multiply %v1924, %v1925 : tensor<128x512x7x7xf32>
    %v1927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1928 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1929 = stablehlo.reduce(%v1926 init: %v1927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1930 = stablehlo.divide %v1929, %v1928 : tensor<512xf32>
    %v1931 = stablehlo.multiply %v1923, %v1926 : tensor<128x512x7x7xf32>
    %v1932 = stablehlo.reduce(%v1931 init: %v1927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
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
    %v1936 = stablehlo.reshape %v1302 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1937 = stablehlo.slice %armeand4g2dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1938 = stablehlo.slice %armeand4g2dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1939 = stablehlo.slice %armeand4g2dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1940 = stablehlo.slice %armeand4g2dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1941 = stablehlo.broadcast_in_dim %v1937, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1942 = stablehlo.broadcast_in_dim %v1938, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1943 = stablehlo.broadcast_in_dim %v1939, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1944 = stablehlo.broadcast_in_dim %v1940, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1945 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1946 = stablehlo.add %v1942, %v1945 : tensor<128x512x7x7xf32>
    %v1947 = stablehlo.rsqrt %v1946 : tensor<128x512x7x7xf32>
    %v1948 = stablehlo.subtract %v1936, %v1941 : tensor<128x512x7x7xf32>
    %v1949 = stablehlo.multiply %v1948, %v1947 : tensor<128x512x7x7xf32>
    %v1950 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1951 = stablehlo.reshape %v1913 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1952 = stablehlo.multiply %v1950, %v1951 : tensor<128x512x7x7xf32>
    %v1953 = stablehlo.subtract %v1952, %v1943 : tensor<128x512x7x7xf32>
    %v1954 = stablehlo.multiply %v1949, %v1944 : tensor<128x512x7x7xf32>
    %v1955 = stablehlo.subtract %v1953, %v1954 : tensor<128x512x7x7xf32>
    %v1956 = stablehlo.multiply %v1947, %v1955 : tensor<128x512x7x7xf32>
    %v1957 = stablehlo.reshape %v1956 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1958 = stablehlo.reshape %v1957 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1959 = stablehlo.reverse %d4W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1960 = stablehlo.transpose %v1959, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1961 = stablehlo.convolution(%v1958, %v1960)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<128x512x7x7xf32>
    %v1962 = stablehlo.reshape %v1961 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1963 = stablehlo.reshape %v1962 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1964 = stablehlo.reshape %v1295 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1965 = stablehlo.constant dense<0.0> : tensor<128x512x7x7xf32>
    %v1966 = stablehlo.compare GT, %v1964, %v1965 : (tensor<128x512x7x7xf32>, tensor<128x512x7x7xf32>) -> tensor<128x512x7x7xi1>
    %v1967 = stablehlo.select %v1966, %v1963, %v1965 : tensor<128x512x7x7xi1>, tensor<128x512x7x7xf32>
    %v1968 = stablehlo.reshape %v1967 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v1969 = stablehlo.reshape %v1261 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1970 = stablehlo.slice %v1280 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1971 = stablehlo.slice %v1280 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1972 = stablehlo.broadcast_in_dim %v1970, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1973 = stablehlo.broadcast_in_dim %v1971, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1974 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v1975 = stablehlo.add %v1973, %v1974 : tensor<128x512x7x7xf32>
    %v1976 = stablehlo.rsqrt %v1975 : tensor<128x512x7x7xf32>
    %v1977 = stablehlo.subtract %v1969, %v1972 : tensor<128x512x7x7xf32>
    %v1978 = stablehlo.multiply %v1977, %v1976 : tensor<128x512x7x7xf32>
    %v1979 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1980 = stablehlo.reshape %v1968 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1981 = stablehlo.multiply %v1979, %v1980 : tensor<128x512x7x7xf32>
    %v1982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1983 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v1984 = stablehlo.reduce(%v1981 init: %v1982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1985 = stablehlo.divide %v1984, %v1983 : tensor<512xf32>
    %v1986 = stablehlo.multiply %v1978, %v1981 : tensor<128x512x7x7xf32>
    %v1987 = stablehlo.reduce(%v1986 init: %v1982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
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
    %v1991 = stablehlo.reshape %v1261 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v1992 = stablehlo.slice %armeand4g1dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1993 = stablehlo.slice %armeand4g1dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1994 = stablehlo.slice %armeand4g1dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1995 = stablehlo.slice %armeand4g1dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1996 = stablehlo.broadcast_in_dim %v1992, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1997 = stablehlo.broadcast_in_dim %v1993, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1998 = stablehlo.broadcast_in_dim %v1994, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v1999 = stablehlo.broadcast_in_dim %v1995, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2000 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v2001 = stablehlo.add %v1997, %v2000 : tensor<128x512x7x7xf32>
    %v2002 = stablehlo.rsqrt %v2001 : tensor<128x512x7x7xf32>
    %v2003 = stablehlo.subtract %v1991, %v1996 : tensor<128x512x7x7xf32>
    %v2004 = stablehlo.multiply %v2003, %v2002 : tensor<128x512x7x7xf32>
    %v2005 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2006 = stablehlo.reshape %v1968 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2007 = stablehlo.multiply %v2005, %v2006 : tensor<128x512x7x7xf32>
    %v2008 = stablehlo.subtract %v2007, %v1998 : tensor<128x512x7x7xf32>
    %v2009 = stablehlo.multiply %v2004, %v1999 : tensor<128x512x7x7xf32>
    %v2010 = stablehlo.subtract %v2008, %v2009 : tensor<128x512x7x7xf32>
    %v2011 = stablehlo.multiply %v2002, %v2010 : tensor<128x512x7x7xf32>
    %v2012 = stablehlo.reshape %v2011 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v2013 = stablehlo.reshape %v2012 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2015 = stablehlo.pad %v2013, %v2014, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<128x512x14x14xf32>
    %v2016 = stablehlo.reverse %d4W1, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v2017 = stablehlo.transpose %v2016, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v2018 = stablehlo.convolution(%v2015, %v2017)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2019 = stablehlo.reshape %v2018 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2020 = stablehlo.reshape %v1341 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2021 = stablehlo.slice %v1360 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2022 = stablehlo.slice %v1360 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2023 = stablehlo.broadcast_in_dim %v2021, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2024 = stablehlo.broadcast_in_dim %v2022, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2025 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v2026 = stablehlo.add %v2024, %v2025 : tensor<128x512x7x7xf32>
    %v2027 = stablehlo.rsqrt %v2026 : tensor<128x512x7x7xf32>
    %v2028 = stablehlo.subtract %v2020, %v2023 : tensor<128x512x7x7xf32>
    %v2029 = stablehlo.multiply %v2028, %v2027 : tensor<128x512x7x7xf32>
    %v2030 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2031 = stablehlo.reshape %v1913 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2032 = stablehlo.multiply %v2030, %v2031 : tensor<128x512x7x7xf32>
    %v2033 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2034 = stablehlo.constant dense<6272.0> : tensor<512xf32>
    %v2035 = stablehlo.reduce(%v2032 init: %v2033) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2036 = stablehlo.divide %v2035, %v2034 : tensor<512xf32>
    %v2037 = stablehlo.multiply %v2029, %v2032 : tensor<128x512x7x7xf32>
    %v2038 = stablehlo.reduce(%v2037 init: %v2033) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2039 = stablehlo.divide %v2038, %v2034 : tensor<512xf32>
    %v2040 = stablehlo.concatenate %v2036, %v2039, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2041 = stablehlo.concatenate %v1360, %v2040, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsumd4gpdst = "stablehlo.all_reduce"(%v2041) ({
    ^bb0(%arad4gpdst: tensor<f32>, %arbd4gpdst: tensor<f32>):
      %araddd4gpdst = stablehlo.add %arad4gpdst, %arbd4gpdst : tensor<f32>
      stablehlo.return %araddd4gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arnd4gpdst = stablehlo.constant dense<2.0> : tensor<2048xf32>
    %armeand4gpdst = stablehlo.divide %arsumd4gpdst, %arnd4gpdst : tensor<2048xf32>
    %v2042 = stablehlo.reshape %v1341 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2043 = stablehlo.slice %armeand4gpdst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2044 = stablehlo.slice %armeand4gpdst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2045 = stablehlo.slice %armeand4gpdst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2046 = stablehlo.slice %armeand4gpdst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2047 = stablehlo.broadcast_in_dim %v2043, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2048 = stablehlo.broadcast_in_dim %v2044, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2049 = stablehlo.broadcast_in_dim %v2045, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2050 = stablehlo.broadcast_in_dim %v2046, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2051 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v2052 = stablehlo.add %v2048, %v2051 : tensor<128x512x7x7xf32>
    %v2053 = stablehlo.rsqrt %v2052 : tensor<128x512x7x7xf32>
    %v2054 = stablehlo.subtract %v2042, %v2047 : tensor<128x512x7x7xf32>
    %v2055 = stablehlo.multiply %v2054, %v2053 : tensor<128x512x7x7xf32>
    %v2056 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2057 = stablehlo.reshape %v1913 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2058 = stablehlo.multiply %v2056, %v2057 : tensor<128x512x7x7xf32>
    %v2059 = stablehlo.subtract %v2058, %v2049 : tensor<128x512x7x7xf32>
    %v2060 = stablehlo.multiply %v2055, %v2050 : tensor<128x512x7x7xf32>
    %v2061 = stablehlo.subtract %v2059, %v2060 : tensor<128x512x7x7xf32>
    %v2062 = stablehlo.multiply %v2053, %v2061 : tensor<128x512x7x7xf32>
    %v2063 = stablehlo.reshape %v2062 : (tensor<128x512x7x7xf32>) -> tensor<128x25088xf32>
    %v2064 = stablehlo.reshape %v2063 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2065 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2066 = stablehlo.pad %v2064, %v2065, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<128x512x14x14xf32>
    %v2067 = stablehlo.reverse %d4Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v2068 = stablehlo.transpose %v2067, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v2069 = stablehlo.convolution(%v2066, %v2068)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x512x14x14xf32>, tensor<256x512x1x1xf32>) -> tensor<128x256x14x14xf32>
    %v2070 = stablehlo.reshape %v2069 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2071 = stablehlo.reshape %v2019 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2072 = stablehlo.reshape %v2070 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2073 = stablehlo.add %v2071, %v2072 : tensor<128x256x14x14xf32>
    %v2074 = stablehlo.reshape %v2073 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2075 = stablehlo.reshape %v1256 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2076 = stablehlo.reshape %v2012 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2077 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2078 = stablehlo.pad %v2076, %v2077, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<128x512x14x14xf32>
    %v2079 = stablehlo.transpose %v2075, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2080 = stablehlo.transpose %v2078, dims = [1, 0, 2, 3] : (tensor<128x512x14x14xf32>) -> tensor<512x128x14x14xf32>
    %v2081 = stablehlo.convolution(%v2079, %v2080)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<512x128x14x14xf32>) -> tensor<256x512x3x3xf32>
    %v2082 = stablehlo.transpose %v2081, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v2083 = stablehlo.reshape %v1261 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2084 = stablehlo.slice %v1280 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2085 = stablehlo.slice %v1280 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2086 = stablehlo.broadcast_in_dim %v2084, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2087 = stablehlo.broadcast_in_dim %v2085, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2088 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v2089 = stablehlo.add %v2087, %v2088 : tensor<128x512x7x7xf32>
    %v2090 = stablehlo.rsqrt %v2089 : tensor<128x512x7x7xf32>
    %v2091 = stablehlo.subtract %v2083, %v2086 : tensor<128x512x7x7xf32>
    %v2092 = stablehlo.multiply %v2091, %v2090 : tensor<128x512x7x7xf32>
    %v2093 = stablehlo.reshape %v1968 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2094 = stablehlo.multiply %v2093, %v2092 : tensor<128x512x7x7xf32>
    %v2095 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2096 = stablehlo.reduce(%v2094 init: %v2095) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2097 = stablehlo.reshape %v1968 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2098 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2099 = stablehlo.reduce(%v2097 init: %v2098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2100 = stablehlo.reshape %v1297 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2101 = stablehlo.reshape %v1957 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2102 = stablehlo.transpose %v2100, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v2103 = stablehlo.transpose %v2101, dims = [1, 0, 2, 3] : (tensor<128x512x7x7xf32>) -> tensor<512x128x7x7xf32>
    %v2104 = stablehlo.convolution(%v2102, %v2103)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x128x7x7xf32>, tensor<512x128x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v2105 = stablehlo.transpose %v2104, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2106 = stablehlo.reshape %v1302 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2107 = stablehlo.slice %v1321 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2108 = stablehlo.slice %v1321 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2109 = stablehlo.broadcast_in_dim %v2107, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2110 = stablehlo.broadcast_in_dim %v2108, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2111 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v2112 = stablehlo.add %v2110, %v2111 : tensor<128x512x7x7xf32>
    %v2113 = stablehlo.rsqrt %v2112 : tensor<128x512x7x7xf32>
    %v2114 = stablehlo.subtract %v2106, %v2109 : tensor<128x512x7x7xf32>
    %v2115 = stablehlo.multiply %v2114, %v2113 : tensor<128x512x7x7xf32>
    %v2116 = stablehlo.reshape %v1913 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2117 = stablehlo.multiply %v2116, %v2115 : tensor<128x512x7x7xf32>
    %v2118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2119 = stablehlo.reduce(%v2117 init: %v2118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2120 = stablehlo.reshape %v1913 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2122 = stablehlo.reduce(%v2120 init: %v2121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2123 = stablehlo.reshape %v1256 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2124 = stablehlo.reshape %v2063 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2125 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2126 = stablehlo.pad %v2124, %v2125, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<128x512x14x14xf32>
    %v2127 = stablehlo.transpose %v2123, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2128 = stablehlo.transpose %v2126, dims = [1, 0, 2, 3] : (tensor<128x512x14x14xf32>) -> tensor<512x128x14x14xf32>
    %v2129 = stablehlo.convolution(%v2127, %v2128)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<512x128x14x14xf32>) -> tensor<256x512x1x1xf32>
    %v2130 = stablehlo.transpose %v2129, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v2131 = stablehlo.reshape %v1341 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2132 = stablehlo.slice %v1360 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2133 = stablehlo.slice %v1360 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2134 = stablehlo.broadcast_in_dim %v2132, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2135 = stablehlo.broadcast_in_dim %v2133, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %v2136 = stablehlo.constant dense<1.0e-05> : tensor<128x512x7x7xf32>
    %v2137 = stablehlo.add %v2135, %v2136 : tensor<128x512x7x7xf32>
    %v2138 = stablehlo.rsqrt %v2137 : tensor<128x512x7x7xf32>
    %v2139 = stablehlo.subtract %v2131, %v2134 : tensor<128x512x7x7xf32>
    %v2140 = stablehlo.multiply %v2139, %v2138 : tensor<128x512x7x7xf32>
    %v2141 = stablehlo.reshape %v1913 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2142 = stablehlo.multiply %v2141, %v2140 : tensor<128x512x7x7xf32>
    %v2143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2144 = stablehlo.reduce(%v2142 init: %v2143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2145 = stablehlo.reshape %v1913 : (tensor<128x25088xf32>) -> tensor<128x512x7x7xf32>
    %v2146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2147 = stablehlo.reduce(%v2145 init: %v2146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2148 = stablehlo.reshape %v2074 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2149 = stablehlo.reshape %v1252 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2150 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2151 = stablehlo.compare GT, %v2149, %v2150 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2152 = stablehlo.select %v2151, %v2148, %v2150 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2153 = stablehlo.reshape %v2152 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2154 = stablehlo.reshape %v1214 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2155 = stablehlo.slice %v1233 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2156 = stablehlo.slice %v1233 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2157 = stablehlo.broadcast_in_dim %v2155, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2158 = stablehlo.broadcast_in_dim %v2156, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2159 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2160 = stablehlo.add %v2158, %v2159 : tensor<128x256x14x14xf32>
    %v2161 = stablehlo.rsqrt %v2160 : tensor<128x256x14x14xf32>
    %v2162 = stablehlo.subtract %v2154, %v2157 : tensor<128x256x14x14xf32>
    %v2163 = stablehlo.multiply %v2162, %v2161 : tensor<128x256x14x14xf32>
    %v2164 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2165 = stablehlo.reshape %v2153 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2166 = stablehlo.multiply %v2164, %v2165 : tensor<128x256x14x14xf32>
    %v2167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2168 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v2169 = stablehlo.reduce(%v2166 init: %v2167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2170 = stablehlo.divide %v2169, %v2168 : tensor<256xf32>
    %v2171 = stablehlo.multiply %v2163, %v2166 : tensor<128x256x14x14xf32>
    %v2172 = stablehlo.reduce(%v2171 init: %v2167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2173 = stablehlo.divide %v2172, %v2168 : tensor<256xf32>
    %v2174 = stablehlo.concatenate %v2170, %v2173, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2175 = stablehlo.concatenate %v1233, %v2174, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b4g2dst = "stablehlo.all_reduce"(%v2175) ({
    ^bb0(%aras3b4g2dst: tensor<f32>, %arbs3b4g2dst: tensor<f32>):
      %aradds3b4g2dst = stablehlo.add %aras3b4g2dst, %arbs3b4g2dst : tensor<f32>
      stablehlo.return %aradds3b4g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g2dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b4g2dst = stablehlo.divide %arsums3b4g2dst, %arns3b4g2dst : tensor<1024xf32>
    %v2176 = stablehlo.reshape %v1214 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2177 = stablehlo.slice %armeans3b4g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2178 = stablehlo.slice %armeans3b4g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2179 = stablehlo.slice %armeans3b4g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2180 = stablehlo.slice %armeans3b4g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2181 = stablehlo.broadcast_in_dim %v2177, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2182 = stablehlo.broadcast_in_dim %v2178, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2183 = stablehlo.broadcast_in_dim %v2179, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2184 = stablehlo.broadcast_in_dim %v2180, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2185 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2186 = stablehlo.add %v2182, %v2185 : tensor<128x256x14x14xf32>
    %v2187 = stablehlo.rsqrt %v2186 : tensor<128x256x14x14xf32>
    %v2188 = stablehlo.subtract %v2176, %v2181 : tensor<128x256x14x14xf32>
    %v2189 = stablehlo.multiply %v2188, %v2187 : tensor<128x256x14x14xf32>
    %v2190 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2191 = stablehlo.reshape %v2153 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2192 = stablehlo.multiply %v2190, %v2191 : tensor<128x256x14x14xf32>
    %v2193 = stablehlo.subtract %v2192, %v2183 : tensor<128x256x14x14xf32>
    %v2194 = stablehlo.multiply %v2189, %v2184 : tensor<128x256x14x14xf32>
    %v2195 = stablehlo.subtract %v2193, %v2194 : tensor<128x256x14x14xf32>
    %v2196 = stablehlo.multiply %v2187, %v2195 : tensor<128x256x14x14xf32>
    %v2197 = stablehlo.reshape %v2196 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2198 = stablehlo.reshape %v2197 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2199 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2200 = stablehlo.transpose %v2199, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2201 = stablehlo.convolution(%v2198, %v2200)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2202 = stablehlo.reshape %v2201 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2203 = stablehlo.reshape %v2202 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2204 = stablehlo.reshape %v1207 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2205 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2206 = stablehlo.compare GT, %v2204, %v2205 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2207 = stablehlo.select %v2206, %v2203, %v2205 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2208 = stablehlo.reshape %v2207 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2209 = stablehlo.reshape %v1173 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2210 = stablehlo.slice %v1192 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2211 = stablehlo.slice %v1192 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2212 = stablehlo.broadcast_in_dim %v2210, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2213 = stablehlo.broadcast_in_dim %v2211, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2214 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2215 = stablehlo.add %v2213, %v2214 : tensor<128x256x14x14xf32>
    %v2216 = stablehlo.rsqrt %v2215 : tensor<128x256x14x14xf32>
    %v2217 = stablehlo.subtract %v2209, %v2212 : tensor<128x256x14x14xf32>
    %v2218 = stablehlo.multiply %v2217, %v2216 : tensor<128x256x14x14xf32>
    %v2219 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2220 = stablehlo.reshape %v2208 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2221 = stablehlo.multiply %v2219, %v2220 : tensor<128x256x14x14xf32>
    %v2222 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2223 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v2224 = stablehlo.reduce(%v2221 init: %v2222) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2225 = stablehlo.divide %v2224, %v2223 : tensor<256xf32>
    %v2226 = stablehlo.multiply %v2218, %v2221 : tensor<128x256x14x14xf32>
    %v2227 = stablehlo.reduce(%v2226 init: %v2222) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2228 = stablehlo.divide %v2227, %v2223 : tensor<256xf32>
    %v2229 = stablehlo.concatenate %v2225, %v2228, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2230 = stablehlo.concatenate %v1192, %v2229, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b4g1dst = "stablehlo.all_reduce"(%v2230) ({
    ^bb0(%aras3b4g1dst: tensor<f32>, %arbs3b4g1dst: tensor<f32>):
      %aradds3b4g1dst = stablehlo.add %aras3b4g1dst, %arbs3b4g1dst : tensor<f32>
      stablehlo.return %aradds3b4g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g1dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b4g1dst = stablehlo.divide %arsums3b4g1dst, %arns3b4g1dst : tensor<1024xf32>
    %v2231 = stablehlo.reshape %v1173 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2232 = stablehlo.slice %armeans3b4g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2233 = stablehlo.slice %armeans3b4g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2234 = stablehlo.slice %armeans3b4g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2235 = stablehlo.slice %armeans3b4g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2236 = stablehlo.broadcast_in_dim %v2232, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2237 = stablehlo.broadcast_in_dim %v2233, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2238 = stablehlo.broadcast_in_dim %v2234, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2239 = stablehlo.broadcast_in_dim %v2235, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2240 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2241 = stablehlo.add %v2237, %v2240 : tensor<128x256x14x14xf32>
    %v2242 = stablehlo.rsqrt %v2241 : tensor<128x256x14x14xf32>
    %v2243 = stablehlo.subtract %v2231, %v2236 : tensor<128x256x14x14xf32>
    %v2244 = stablehlo.multiply %v2243, %v2242 : tensor<128x256x14x14xf32>
    %v2245 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2246 = stablehlo.reshape %v2208 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2247 = stablehlo.multiply %v2245, %v2246 : tensor<128x256x14x14xf32>
    %v2248 = stablehlo.subtract %v2247, %v2238 : tensor<128x256x14x14xf32>
    %v2249 = stablehlo.multiply %v2244, %v2239 : tensor<128x256x14x14xf32>
    %v2250 = stablehlo.subtract %v2248, %v2249 : tensor<128x256x14x14xf32>
    %v2251 = stablehlo.multiply %v2242, %v2250 : tensor<128x256x14x14xf32>
    %v2252 = stablehlo.reshape %v2251 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2253 = stablehlo.reshape %v2252 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2254 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2255 = stablehlo.transpose %v2254, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2256 = stablehlo.convolution(%v2253, %v2255)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2257 = stablehlo.reshape %v2256 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2258 = stablehlo.reshape %v2257 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2259 = stablehlo.reshape %v2153 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2260 = stablehlo.add %v2258, %v2259 : tensor<128x256x14x14xf32>
    %v2261 = stablehlo.reshape %v2260 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2262 = stablehlo.reshape %v1168 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2263 = stablehlo.reshape %v2252 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2264 = stablehlo.transpose %v2262, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2265 = stablehlo.transpose %v2263, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2266 = stablehlo.convolution(%v2264, %v2265)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2267 = stablehlo.transpose %v2266, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2268 = stablehlo.reshape %v1173 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2269 = stablehlo.slice %v1192 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2270 = stablehlo.slice %v1192 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2271 = stablehlo.broadcast_in_dim %v2269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2272 = stablehlo.broadcast_in_dim %v2270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2273 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2274 = stablehlo.add %v2272, %v2273 : tensor<128x256x14x14xf32>
    %v2275 = stablehlo.rsqrt %v2274 : tensor<128x256x14x14xf32>
    %v2276 = stablehlo.subtract %v2268, %v2271 : tensor<128x256x14x14xf32>
    %v2277 = stablehlo.multiply %v2276, %v2275 : tensor<128x256x14x14xf32>
    %v2278 = stablehlo.reshape %v2208 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2279 = stablehlo.multiply %v2278, %v2277 : tensor<128x256x14x14xf32>
    %v2280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2281 = stablehlo.reduce(%v2279 init: %v2280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2282 = stablehlo.reshape %v2208 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2284 = stablehlo.reduce(%v2282 init: %v2283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2285 = stablehlo.reshape %v1209 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2286 = stablehlo.reshape %v2197 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2287 = stablehlo.transpose %v2285, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2288 = stablehlo.transpose %v2286, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2289 = stablehlo.convolution(%v2287, %v2288)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2290 = stablehlo.transpose %v2289, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2291 = stablehlo.reshape %v1214 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2292 = stablehlo.slice %v1233 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2293 = stablehlo.slice %v1233 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2294 = stablehlo.broadcast_in_dim %v2292, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2295 = stablehlo.broadcast_in_dim %v2293, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2296 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2297 = stablehlo.add %v2295, %v2296 : tensor<128x256x14x14xf32>
    %v2298 = stablehlo.rsqrt %v2297 : tensor<128x256x14x14xf32>
    %v2299 = stablehlo.subtract %v2291, %v2294 : tensor<128x256x14x14xf32>
    %v2300 = stablehlo.multiply %v2299, %v2298 : tensor<128x256x14x14xf32>
    %v2301 = stablehlo.reshape %v2153 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2302 = stablehlo.multiply %v2301, %v2300 : tensor<128x256x14x14xf32>
    %v2303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2304 = stablehlo.reduce(%v2302 init: %v2303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2305 = stablehlo.reshape %v2153 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2307 = stablehlo.reduce(%v2305 init: %v2306) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2308 = stablehlo.reshape %v2261 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2309 = stablehlo.reshape %v1164 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2310 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2311 = stablehlo.compare GT, %v2309, %v2310 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2312 = stablehlo.select %v2311, %v2308, %v2310 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2313 = stablehlo.reshape %v2312 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2314 = stablehlo.reshape %v1126 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2315 = stablehlo.slice %v1145 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2316 = stablehlo.slice %v1145 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2317 = stablehlo.broadcast_in_dim %v2315, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2318 = stablehlo.broadcast_in_dim %v2316, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2319 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2320 = stablehlo.add %v2318, %v2319 : tensor<128x256x14x14xf32>
    %v2321 = stablehlo.rsqrt %v2320 : tensor<128x256x14x14xf32>
    %v2322 = stablehlo.subtract %v2314, %v2317 : tensor<128x256x14x14xf32>
    %v2323 = stablehlo.multiply %v2322, %v2321 : tensor<128x256x14x14xf32>
    %v2324 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2325 = stablehlo.reshape %v2313 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2326 = stablehlo.multiply %v2324, %v2325 : tensor<128x256x14x14xf32>
    %v2327 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2328 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v2329 = stablehlo.reduce(%v2326 init: %v2327) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2330 = stablehlo.divide %v2329, %v2328 : tensor<256xf32>
    %v2331 = stablehlo.multiply %v2323, %v2326 : tensor<128x256x14x14xf32>
    %v2332 = stablehlo.reduce(%v2331 init: %v2327) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2333 = stablehlo.divide %v2332, %v2328 : tensor<256xf32>
    %v2334 = stablehlo.concatenate %v2330, %v2333, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2335 = stablehlo.concatenate %v1145, %v2334, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b3g2dst = "stablehlo.all_reduce"(%v2335) ({
    ^bb0(%aras3b3g2dst: tensor<f32>, %arbs3b3g2dst: tensor<f32>):
      %aradds3b3g2dst = stablehlo.add %aras3b3g2dst, %arbs3b3g2dst : tensor<f32>
      stablehlo.return %aradds3b3g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g2dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b3g2dst = stablehlo.divide %arsums3b3g2dst, %arns3b3g2dst : tensor<1024xf32>
    %v2336 = stablehlo.reshape %v1126 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2337 = stablehlo.slice %armeans3b3g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2338 = stablehlo.slice %armeans3b3g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2339 = stablehlo.slice %armeans3b3g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2340 = stablehlo.slice %armeans3b3g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2341 = stablehlo.broadcast_in_dim %v2337, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2342 = stablehlo.broadcast_in_dim %v2338, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2343 = stablehlo.broadcast_in_dim %v2339, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2344 = stablehlo.broadcast_in_dim %v2340, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2345 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2346 = stablehlo.add %v2342, %v2345 : tensor<128x256x14x14xf32>
    %v2347 = stablehlo.rsqrt %v2346 : tensor<128x256x14x14xf32>
    %v2348 = stablehlo.subtract %v2336, %v2341 : tensor<128x256x14x14xf32>
    %v2349 = stablehlo.multiply %v2348, %v2347 : tensor<128x256x14x14xf32>
    %v2350 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2351 = stablehlo.reshape %v2313 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2352 = stablehlo.multiply %v2350, %v2351 : tensor<128x256x14x14xf32>
    %v2353 = stablehlo.subtract %v2352, %v2343 : tensor<128x256x14x14xf32>
    %v2354 = stablehlo.multiply %v2349, %v2344 : tensor<128x256x14x14xf32>
    %v2355 = stablehlo.subtract %v2353, %v2354 : tensor<128x256x14x14xf32>
    %v2356 = stablehlo.multiply %v2347, %v2355 : tensor<128x256x14x14xf32>
    %v2357 = stablehlo.reshape %v2356 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2358 = stablehlo.reshape %v2357 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2359 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2360 = stablehlo.transpose %v2359, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2361 = stablehlo.convolution(%v2358, %v2360)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2362 = stablehlo.reshape %v2361 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2363 = stablehlo.reshape %v2362 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2364 = stablehlo.reshape %v1119 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2365 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2366 = stablehlo.compare GT, %v2364, %v2365 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2367 = stablehlo.select %v2366, %v2363, %v2365 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2368 = stablehlo.reshape %v2367 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2369 = stablehlo.reshape %v1085 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2370 = stablehlo.slice %v1104 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2371 = stablehlo.slice %v1104 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2372 = stablehlo.broadcast_in_dim %v2370, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2373 = stablehlo.broadcast_in_dim %v2371, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2374 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2375 = stablehlo.add %v2373, %v2374 : tensor<128x256x14x14xf32>
    %v2376 = stablehlo.rsqrt %v2375 : tensor<128x256x14x14xf32>
    %v2377 = stablehlo.subtract %v2369, %v2372 : tensor<128x256x14x14xf32>
    %v2378 = stablehlo.multiply %v2377, %v2376 : tensor<128x256x14x14xf32>
    %v2379 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2380 = stablehlo.reshape %v2368 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2381 = stablehlo.multiply %v2379, %v2380 : tensor<128x256x14x14xf32>
    %v2382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2383 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v2384 = stablehlo.reduce(%v2381 init: %v2382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2385 = stablehlo.divide %v2384, %v2383 : tensor<256xf32>
    %v2386 = stablehlo.multiply %v2378, %v2381 : tensor<128x256x14x14xf32>
    %v2387 = stablehlo.reduce(%v2386 init: %v2382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2388 = stablehlo.divide %v2387, %v2383 : tensor<256xf32>
    %v2389 = stablehlo.concatenate %v2385, %v2388, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2390 = stablehlo.concatenate %v1104, %v2389, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b3g1dst = "stablehlo.all_reduce"(%v2390) ({
    ^bb0(%aras3b3g1dst: tensor<f32>, %arbs3b3g1dst: tensor<f32>):
      %aradds3b3g1dst = stablehlo.add %aras3b3g1dst, %arbs3b3g1dst : tensor<f32>
      stablehlo.return %aradds3b3g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g1dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b3g1dst = stablehlo.divide %arsums3b3g1dst, %arns3b3g1dst : tensor<1024xf32>
    %v2391 = stablehlo.reshape %v1085 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2392 = stablehlo.slice %armeans3b3g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2393 = stablehlo.slice %armeans3b3g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2394 = stablehlo.slice %armeans3b3g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2395 = stablehlo.slice %armeans3b3g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2396 = stablehlo.broadcast_in_dim %v2392, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2397 = stablehlo.broadcast_in_dim %v2393, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2398 = stablehlo.broadcast_in_dim %v2394, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2399 = stablehlo.broadcast_in_dim %v2395, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2400 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2401 = stablehlo.add %v2397, %v2400 : tensor<128x256x14x14xf32>
    %v2402 = stablehlo.rsqrt %v2401 : tensor<128x256x14x14xf32>
    %v2403 = stablehlo.subtract %v2391, %v2396 : tensor<128x256x14x14xf32>
    %v2404 = stablehlo.multiply %v2403, %v2402 : tensor<128x256x14x14xf32>
    %v2405 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2406 = stablehlo.reshape %v2368 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2407 = stablehlo.multiply %v2405, %v2406 : tensor<128x256x14x14xf32>
    %v2408 = stablehlo.subtract %v2407, %v2398 : tensor<128x256x14x14xf32>
    %v2409 = stablehlo.multiply %v2404, %v2399 : tensor<128x256x14x14xf32>
    %v2410 = stablehlo.subtract %v2408, %v2409 : tensor<128x256x14x14xf32>
    %v2411 = stablehlo.multiply %v2402, %v2410 : tensor<128x256x14x14xf32>
    %v2412 = stablehlo.reshape %v2411 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2413 = stablehlo.reshape %v2412 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2414 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2415 = stablehlo.transpose %v2414, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2416 = stablehlo.convolution(%v2413, %v2415)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2417 = stablehlo.reshape %v2416 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2418 = stablehlo.reshape %v2417 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2419 = stablehlo.reshape %v2313 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2420 = stablehlo.add %v2418, %v2419 : tensor<128x256x14x14xf32>
    %v2421 = stablehlo.reshape %v2420 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2422 = stablehlo.reshape %v1080 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2423 = stablehlo.reshape %v2412 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2424 = stablehlo.transpose %v2422, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2425 = stablehlo.transpose %v2423, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2426 = stablehlo.convolution(%v2424, %v2425)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2427 = stablehlo.transpose %v2426, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2428 = stablehlo.reshape %v1085 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2429 = stablehlo.slice %v1104 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2430 = stablehlo.slice %v1104 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2431 = stablehlo.broadcast_in_dim %v2429, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2432 = stablehlo.broadcast_in_dim %v2430, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2433 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2434 = stablehlo.add %v2432, %v2433 : tensor<128x256x14x14xf32>
    %v2435 = stablehlo.rsqrt %v2434 : tensor<128x256x14x14xf32>
    %v2436 = stablehlo.subtract %v2428, %v2431 : tensor<128x256x14x14xf32>
    %v2437 = stablehlo.multiply %v2436, %v2435 : tensor<128x256x14x14xf32>
    %v2438 = stablehlo.reshape %v2368 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2439 = stablehlo.multiply %v2438, %v2437 : tensor<128x256x14x14xf32>
    %v2440 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2441 = stablehlo.reduce(%v2439 init: %v2440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2442 = stablehlo.reshape %v2368 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2444 = stablehlo.reduce(%v2442 init: %v2443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2445 = stablehlo.reshape %v1121 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2446 = stablehlo.reshape %v2357 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2447 = stablehlo.transpose %v2445, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2448 = stablehlo.transpose %v2446, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2449 = stablehlo.convolution(%v2447, %v2448)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2450 = stablehlo.transpose %v2449, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2451 = stablehlo.reshape %v1126 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2452 = stablehlo.slice %v1145 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2453 = stablehlo.slice %v1145 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2454 = stablehlo.broadcast_in_dim %v2452, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2455 = stablehlo.broadcast_in_dim %v2453, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2456 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2457 = stablehlo.add %v2455, %v2456 : tensor<128x256x14x14xf32>
    %v2458 = stablehlo.rsqrt %v2457 : tensor<128x256x14x14xf32>
    %v2459 = stablehlo.subtract %v2451, %v2454 : tensor<128x256x14x14xf32>
    %v2460 = stablehlo.multiply %v2459, %v2458 : tensor<128x256x14x14xf32>
    %v2461 = stablehlo.reshape %v2313 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2462 = stablehlo.multiply %v2461, %v2460 : tensor<128x256x14x14xf32>
    %v2463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2464 = stablehlo.reduce(%v2462 init: %v2463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2465 = stablehlo.reshape %v2313 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2466 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2467 = stablehlo.reduce(%v2465 init: %v2466) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2468 = stablehlo.reshape %v2421 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2469 = stablehlo.reshape %v1076 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2470 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2471 = stablehlo.compare GT, %v2469, %v2470 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2472 = stablehlo.select %v2471, %v2468, %v2470 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2473 = stablehlo.reshape %v2472 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2474 = stablehlo.reshape %v1038 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2475 = stablehlo.slice %v1057 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2476 = stablehlo.slice %v1057 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2477 = stablehlo.broadcast_in_dim %v2475, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2478 = stablehlo.broadcast_in_dim %v2476, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2479 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2480 = stablehlo.add %v2478, %v2479 : tensor<128x256x14x14xf32>
    %v2481 = stablehlo.rsqrt %v2480 : tensor<128x256x14x14xf32>
    %v2482 = stablehlo.subtract %v2474, %v2477 : tensor<128x256x14x14xf32>
    %v2483 = stablehlo.multiply %v2482, %v2481 : tensor<128x256x14x14xf32>
    %v2484 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2485 = stablehlo.reshape %v2473 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2486 = stablehlo.multiply %v2484, %v2485 : tensor<128x256x14x14xf32>
    %v2487 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2488 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v2489 = stablehlo.reduce(%v2486 init: %v2487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2490 = stablehlo.divide %v2489, %v2488 : tensor<256xf32>
    %v2491 = stablehlo.multiply %v2483, %v2486 : tensor<128x256x14x14xf32>
    %v2492 = stablehlo.reduce(%v2491 init: %v2487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2493 = stablehlo.divide %v2492, %v2488 : tensor<256xf32>
    %v2494 = stablehlo.concatenate %v2490, %v2493, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2495 = stablehlo.concatenate %v1057, %v2494, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b2g2dst = "stablehlo.all_reduce"(%v2495) ({
    ^bb0(%aras3b2g2dst: tensor<f32>, %arbs3b2g2dst: tensor<f32>):
      %aradds3b2g2dst = stablehlo.add %aras3b2g2dst, %arbs3b2g2dst : tensor<f32>
      stablehlo.return %aradds3b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g2dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b2g2dst = stablehlo.divide %arsums3b2g2dst, %arns3b2g2dst : tensor<1024xf32>
    %v2496 = stablehlo.reshape %v1038 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2497 = stablehlo.slice %armeans3b2g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2498 = stablehlo.slice %armeans3b2g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2499 = stablehlo.slice %armeans3b2g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2500 = stablehlo.slice %armeans3b2g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2501 = stablehlo.broadcast_in_dim %v2497, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2502 = stablehlo.broadcast_in_dim %v2498, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2503 = stablehlo.broadcast_in_dim %v2499, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2504 = stablehlo.broadcast_in_dim %v2500, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2505 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2506 = stablehlo.add %v2502, %v2505 : tensor<128x256x14x14xf32>
    %v2507 = stablehlo.rsqrt %v2506 : tensor<128x256x14x14xf32>
    %v2508 = stablehlo.subtract %v2496, %v2501 : tensor<128x256x14x14xf32>
    %v2509 = stablehlo.multiply %v2508, %v2507 : tensor<128x256x14x14xf32>
    %v2510 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2511 = stablehlo.reshape %v2473 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2512 = stablehlo.multiply %v2510, %v2511 : tensor<128x256x14x14xf32>
    %v2513 = stablehlo.subtract %v2512, %v2503 : tensor<128x256x14x14xf32>
    %v2514 = stablehlo.multiply %v2509, %v2504 : tensor<128x256x14x14xf32>
    %v2515 = stablehlo.subtract %v2513, %v2514 : tensor<128x256x14x14xf32>
    %v2516 = stablehlo.multiply %v2507, %v2515 : tensor<128x256x14x14xf32>
    %v2517 = stablehlo.reshape %v2516 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2518 = stablehlo.reshape %v2517 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2519 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2520 = stablehlo.transpose %v2519, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2521 = stablehlo.convolution(%v2518, %v2520)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2522 = stablehlo.reshape %v2521 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2523 = stablehlo.reshape %v2522 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2524 = stablehlo.reshape %v1031 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2525 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2526 = stablehlo.compare GT, %v2524, %v2525 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2527 = stablehlo.select %v2526, %v2523, %v2525 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2528 = stablehlo.reshape %v2527 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2529 = stablehlo.reshape %v997 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2530 = stablehlo.slice %v1016 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2531 = stablehlo.slice %v1016 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2532 = stablehlo.broadcast_in_dim %v2530, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2533 = stablehlo.broadcast_in_dim %v2531, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2534 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2535 = stablehlo.add %v2533, %v2534 : tensor<128x256x14x14xf32>
    %v2536 = stablehlo.rsqrt %v2535 : tensor<128x256x14x14xf32>
    %v2537 = stablehlo.subtract %v2529, %v2532 : tensor<128x256x14x14xf32>
    %v2538 = stablehlo.multiply %v2537, %v2536 : tensor<128x256x14x14xf32>
    %v2539 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2540 = stablehlo.reshape %v2528 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2541 = stablehlo.multiply %v2539, %v2540 : tensor<128x256x14x14xf32>
    %v2542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2543 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v2544 = stablehlo.reduce(%v2541 init: %v2542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2545 = stablehlo.divide %v2544, %v2543 : tensor<256xf32>
    %v2546 = stablehlo.multiply %v2538, %v2541 : tensor<128x256x14x14xf32>
    %v2547 = stablehlo.reduce(%v2546 init: %v2542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2548 = stablehlo.divide %v2547, %v2543 : tensor<256xf32>
    %v2549 = stablehlo.concatenate %v2545, %v2548, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2550 = stablehlo.concatenate %v1016, %v2549, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b2g1dst = "stablehlo.all_reduce"(%v2550) ({
    ^bb0(%aras3b2g1dst: tensor<f32>, %arbs3b2g1dst: tensor<f32>):
      %aradds3b2g1dst = stablehlo.add %aras3b2g1dst, %arbs3b2g1dst : tensor<f32>
      stablehlo.return %aradds3b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g1dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b2g1dst = stablehlo.divide %arsums3b2g1dst, %arns3b2g1dst : tensor<1024xf32>
    %v2551 = stablehlo.reshape %v997 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2552 = stablehlo.slice %armeans3b2g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2553 = stablehlo.slice %armeans3b2g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2554 = stablehlo.slice %armeans3b2g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2555 = stablehlo.slice %armeans3b2g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2556 = stablehlo.broadcast_in_dim %v2552, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2557 = stablehlo.broadcast_in_dim %v2553, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2558 = stablehlo.broadcast_in_dim %v2554, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2559 = stablehlo.broadcast_in_dim %v2555, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2560 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2561 = stablehlo.add %v2557, %v2560 : tensor<128x256x14x14xf32>
    %v2562 = stablehlo.rsqrt %v2561 : tensor<128x256x14x14xf32>
    %v2563 = stablehlo.subtract %v2551, %v2556 : tensor<128x256x14x14xf32>
    %v2564 = stablehlo.multiply %v2563, %v2562 : tensor<128x256x14x14xf32>
    %v2565 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2566 = stablehlo.reshape %v2528 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2567 = stablehlo.multiply %v2565, %v2566 : tensor<128x256x14x14xf32>
    %v2568 = stablehlo.subtract %v2567, %v2558 : tensor<128x256x14x14xf32>
    %v2569 = stablehlo.multiply %v2564, %v2559 : tensor<128x256x14x14xf32>
    %v2570 = stablehlo.subtract %v2568, %v2569 : tensor<128x256x14x14xf32>
    %v2571 = stablehlo.multiply %v2562, %v2570 : tensor<128x256x14x14xf32>
    %v2572 = stablehlo.reshape %v2571 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2573 = stablehlo.reshape %v2572 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2574 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2575 = stablehlo.transpose %v2574, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2576 = stablehlo.convolution(%v2573, %v2575)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2577 = stablehlo.reshape %v2576 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2578 = stablehlo.reshape %v2577 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2579 = stablehlo.reshape %v2473 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2580 = stablehlo.add %v2578, %v2579 : tensor<128x256x14x14xf32>
    %v2581 = stablehlo.reshape %v2580 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2582 = stablehlo.reshape %v992 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2583 = stablehlo.reshape %v2572 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2584 = stablehlo.transpose %v2582, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2585 = stablehlo.transpose %v2583, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2586 = stablehlo.convolution(%v2584, %v2585)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2587 = stablehlo.transpose %v2586, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2588 = stablehlo.reshape %v997 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2589 = stablehlo.slice %v1016 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2590 = stablehlo.slice %v1016 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2591 = stablehlo.broadcast_in_dim %v2589, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2592 = stablehlo.broadcast_in_dim %v2590, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2593 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2594 = stablehlo.add %v2592, %v2593 : tensor<128x256x14x14xf32>
    %v2595 = stablehlo.rsqrt %v2594 : tensor<128x256x14x14xf32>
    %v2596 = stablehlo.subtract %v2588, %v2591 : tensor<128x256x14x14xf32>
    %v2597 = stablehlo.multiply %v2596, %v2595 : tensor<128x256x14x14xf32>
    %v2598 = stablehlo.reshape %v2528 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2599 = stablehlo.multiply %v2598, %v2597 : tensor<128x256x14x14xf32>
    %v2600 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2601 = stablehlo.reduce(%v2599 init: %v2600) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2602 = stablehlo.reshape %v2528 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2603 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2604 = stablehlo.reduce(%v2602 init: %v2603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2605 = stablehlo.reshape %v1033 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2606 = stablehlo.reshape %v2517 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2607 = stablehlo.transpose %v2605, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2608 = stablehlo.transpose %v2606, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2609 = stablehlo.convolution(%v2607, %v2608)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2610 = stablehlo.transpose %v2609, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2611 = stablehlo.reshape %v1038 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2612 = stablehlo.slice %v1057 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2613 = stablehlo.slice %v1057 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2614 = stablehlo.broadcast_in_dim %v2612, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2615 = stablehlo.broadcast_in_dim %v2613, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2616 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2617 = stablehlo.add %v2615, %v2616 : tensor<128x256x14x14xf32>
    %v2618 = stablehlo.rsqrt %v2617 : tensor<128x256x14x14xf32>
    %v2619 = stablehlo.subtract %v2611, %v2614 : tensor<128x256x14x14xf32>
    %v2620 = stablehlo.multiply %v2619, %v2618 : tensor<128x256x14x14xf32>
    %v2621 = stablehlo.reshape %v2473 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2622 = stablehlo.multiply %v2621, %v2620 : tensor<128x256x14x14xf32>
    %v2623 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2624 = stablehlo.reduce(%v2622 init: %v2623) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2625 = stablehlo.reshape %v2473 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2627 = stablehlo.reduce(%v2625 init: %v2626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2628 = stablehlo.reshape %v2581 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2629 = stablehlo.reshape %v988 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2630 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2631 = stablehlo.compare GT, %v2629, %v2630 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2632 = stablehlo.select %v2631, %v2628, %v2630 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2633 = stablehlo.reshape %v2632 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2634 = stablehlo.reshape %v950 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2635 = stablehlo.slice %v969 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2636 = stablehlo.slice %v969 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2637 = stablehlo.broadcast_in_dim %v2635, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2638 = stablehlo.broadcast_in_dim %v2636, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2639 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2640 = stablehlo.add %v2638, %v2639 : tensor<128x256x14x14xf32>
    %v2641 = stablehlo.rsqrt %v2640 : tensor<128x256x14x14xf32>
    %v2642 = stablehlo.subtract %v2634, %v2637 : tensor<128x256x14x14xf32>
    %v2643 = stablehlo.multiply %v2642, %v2641 : tensor<128x256x14x14xf32>
    %v2644 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2645 = stablehlo.reshape %v2633 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2646 = stablehlo.multiply %v2644, %v2645 : tensor<128x256x14x14xf32>
    %v2647 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2648 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v2649 = stablehlo.reduce(%v2646 init: %v2647) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2650 = stablehlo.divide %v2649, %v2648 : tensor<256xf32>
    %v2651 = stablehlo.multiply %v2643, %v2646 : tensor<128x256x14x14xf32>
    %v2652 = stablehlo.reduce(%v2651 init: %v2647) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2653 = stablehlo.divide %v2652, %v2648 : tensor<256xf32>
    %v2654 = stablehlo.concatenate %v2650, %v2653, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2655 = stablehlo.concatenate %v969, %v2654, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b1g2dst = "stablehlo.all_reduce"(%v2655) ({
    ^bb0(%aras3b1g2dst: tensor<f32>, %arbs3b1g2dst: tensor<f32>):
      %aradds3b1g2dst = stablehlo.add %aras3b1g2dst, %arbs3b1g2dst : tensor<f32>
      stablehlo.return %aradds3b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g2dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b1g2dst = stablehlo.divide %arsums3b1g2dst, %arns3b1g2dst : tensor<1024xf32>
    %v2656 = stablehlo.reshape %v950 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2657 = stablehlo.slice %armeans3b1g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2658 = stablehlo.slice %armeans3b1g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2659 = stablehlo.slice %armeans3b1g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2660 = stablehlo.slice %armeans3b1g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2661 = stablehlo.broadcast_in_dim %v2657, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2662 = stablehlo.broadcast_in_dim %v2658, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2663 = stablehlo.broadcast_in_dim %v2659, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2664 = stablehlo.broadcast_in_dim %v2660, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2665 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2666 = stablehlo.add %v2662, %v2665 : tensor<128x256x14x14xf32>
    %v2667 = stablehlo.rsqrt %v2666 : tensor<128x256x14x14xf32>
    %v2668 = stablehlo.subtract %v2656, %v2661 : tensor<128x256x14x14xf32>
    %v2669 = stablehlo.multiply %v2668, %v2667 : tensor<128x256x14x14xf32>
    %v2670 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2671 = stablehlo.reshape %v2633 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2672 = stablehlo.multiply %v2670, %v2671 : tensor<128x256x14x14xf32>
    %v2673 = stablehlo.subtract %v2672, %v2663 : tensor<128x256x14x14xf32>
    %v2674 = stablehlo.multiply %v2669, %v2664 : tensor<128x256x14x14xf32>
    %v2675 = stablehlo.subtract %v2673, %v2674 : tensor<128x256x14x14xf32>
    %v2676 = stablehlo.multiply %v2667, %v2675 : tensor<128x256x14x14xf32>
    %v2677 = stablehlo.reshape %v2676 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2678 = stablehlo.reshape %v2677 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2679 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2680 = stablehlo.transpose %v2679, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2681 = stablehlo.convolution(%v2678, %v2680)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2682 = stablehlo.reshape %v2681 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2683 = stablehlo.reshape %v2682 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2684 = stablehlo.reshape %v943 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2685 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2686 = stablehlo.compare GT, %v2684, %v2685 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2687 = stablehlo.select %v2686, %v2683, %v2685 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2688 = stablehlo.reshape %v2687 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2689 = stablehlo.reshape %v909 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2690 = stablehlo.slice %v928 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2691 = stablehlo.slice %v928 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2692 = stablehlo.broadcast_in_dim %v2690, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2693 = stablehlo.broadcast_in_dim %v2691, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2694 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2695 = stablehlo.add %v2693, %v2694 : tensor<128x256x14x14xf32>
    %v2696 = stablehlo.rsqrt %v2695 : tensor<128x256x14x14xf32>
    %v2697 = stablehlo.subtract %v2689, %v2692 : tensor<128x256x14x14xf32>
    %v2698 = stablehlo.multiply %v2697, %v2696 : tensor<128x256x14x14xf32>
    %v2699 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2700 = stablehlo.reshape %v2688 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2701 = stablehlo.multiply %v2699, %v2700 : tensor<128x256x14x14xf32>
    %v2702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2703 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v2704 = stablehlo.reduce(%v2701 init: %v2702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2705 = stablehlo.divide %v2704, %v2703 : tensor<256xf32>
    %v2706 = stablehlo.multiply %v2698, %v2701 : tensor<128x256x14x14xf32>
    %v2707 = stablehlo.reduce(%v2706 init: %v2702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2708 = stablehlo.divide %v2707, %v2703 : tensor<256xf32>
    %v2709 = stablehlo.concatenate %v2705, %v2708, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2710 = stablehlo.concatenate %v928, %v2709, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b1g1dst = "stablehlo.all_reduce"(%v2710) ({
    ^bb0(%aras3b1g1dst: tensor<f32>, %arbs3b1g1dst: tensor<f32>):
      %aradds3b1g1dst = stablehlo.add %aras3b1g1dst, %arbs3b1g1dst : tensor<f32>
      stablehlo.return %aradds3b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g1dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b1g1dst = stablehlo.divide %arsums3b1g1dst, %arns3b1g1dst : tensor<1024xf32>
    %v2711 = stablehlo.reshape %v909 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2712 = stablehlo.slice %armeans3b1g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2713 = stablehlo.slice %armeans3b1g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2714 = stablehlo.slice %armeans3b1g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2715 = stablehlo.slice %armeans3b1g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2716 = stablehlo.broadcast_in_dim %v2712, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2717 = stablehlo.broadcast_in_dim %v2713, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2718 = stablehlo.broadcast_in_dim %v2714, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2719 = stablehlo.broadcast_in_dim %v2715, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2720 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2721 = stablehlo.add %v2717, %v2720 : tensor<128x256x14x14xf32>
    %v2722 = stablehlo.rsqrt %v2721 : tensor<128x256x14x14xf32>
    %v2723 = stablehlo.subtract %v2711, %v2716 : tensor<128x256x14x14xf32>
    %v2724 = stablehlo.multiply %v2723, %v2722 : tensor<128x256x14x14xf32>
    %v2725 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2726 = stablehlo.reshape %v2688 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2727 = stablehlo.multiply %v2725, %v2726 : tensor<128x256x14x14xf32>
    %v2728 = stablehlo.subtract %v2727, %v2718 : tensor<128x256x14x14xf32>
    %v2729 = stablehlo.multiply %v2724, %v2719 : tensor<128x256x14x14xf32>
    %v2730 = stablehlo.subtract %v2728, %v2729 : tensor<128x256x14x14xf32>
    %v2731 = stablehlo.multiply %v2722, %v2730 : tensor<128x256x14x14xf32>
    %v2732 = stablehlo.reshape %v2731 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2733 = stablehlo.reshape %v2732 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2734 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2735 = stablehlo.transpose %v2734, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2736 = stablehlo.convolution(%v2733, %v2735)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2737 = stablehlo.reshape %v2736 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2738 = stablehlo.reshape %v2737 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2739 = stablehlo.reshape %v2633 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2740 = stablehlo.add %v2738, %v2739 : tensor<128x256x14x14xf32>
    %v2741 = stablehlo.reshape %v2740 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2742 = stablehlo.reshape %v904 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2743 = stablehlo.reshape %v2732 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2744 = stablehlo.transpose %v2742, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2745 = stablehlo.transpose %v2743, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2746 = stablehlo.convolution(%v2744, %v2745)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2747 = stablehlo.transpose %v2746, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2748 = stablehlo.reshape %v909 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2749 = stablehlo.slice %v928 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2750 = stablehlo.slice %v928 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2751 = stablehlo.broadcast_in_dim %v2749, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2752 = stablehlo.broadcast_in_dim %v2750, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2753 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2754 = stablehlo.add %v2752, %v2753 : tensor<128x256x14x14xf32>
    %v2755 = stablehlo.rsqrt %v2754 : tensor<128x256x14x14xf32>
    %v2756 = stablehlo.subtract %v2748, %v2751 : tensor<128x256x14x14xf32>
    %v2757 = stablehlo.multiply %v2756, %v2755 : tensor<128x256x14x14xf32>
    %v2758 = stablehlo.reshape %v2688 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2759 = stablehlo.multiply %v2758, %v2757 : tensor<128x256x14x14xf32>
    %v2760 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2761 = stablehlo.reduce(%v2759 init: %v2760) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2762 = stablehlo.reshape %v2688 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2763 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2764 = stablehlo.reduce(%v2762 init: %v2763) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2765 = stablehlo.reshape %v945 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2766 = stablehlo.reshape %v2677 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2767 = stablehlo.transpose %v2765, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2768 = stablehlo.transpose %v2766, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2769 = stablehlo.convolution(%v2767, %v2768)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2770 = stablehlo.transpose %v2769, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2771 = stablehlo.reshape %v950 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2772 = stablehlo.slice %v969 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2773 = stablehlo.slice %v969 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2774 = stablehlo.broadcast_in_dim %v2772, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2775 = stablehlo.broadcast_in_dim %v2773, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2776 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2777 = stablehlo.add %v2775, %v2776 : tensor<128x256x14x14xf32>
    %v2778 = stablehlo.rsqrt %v2777 : tensor<128x256x14x14xf32>
    %v2779 = stablehlo.subtract %v2771, %v2774 : tensor<128x256x14x14xf32>
    %v2780 = stablehlo.multiply %v2779, %v2778 : tensor<128x256x14x14xf32>
    %v2781 = stablehlo.reshape %v2633 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2782 = stablehlo.multiply %v2781, %v2780 : tensor<128x256x14x14xf32>
    %v2783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2784 = stablehlo.reduce(%v2782 init: %v2783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2785 = stablehlo.reshape %v2633 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2787 = stablehlo.reduce(%v2785 init: %v2786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2788 = stablehlo.reshape %v2741 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2789 = stablehlo.reshape %v900 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2790 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2791 = stablehlo.compare GT, %v2789, %v2790 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2792 = stablehlo.select %v2791, %v2788, %v2790 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2793 = stablehlo.reshape %v2792 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2794 = stablehlo.reshape %v862 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2795 = stablehlo.slice %v881 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2796 = stablehlo.slice %v881 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2797 = stablehlo.broadcast_in_dim %v2795, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2798 = stablehlo.broadcast_in_dim %v2796, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2799 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2800 = stablehlo.add %v2798, %v2799 : tensor<128x256x14x14xf32>
    %v2801 = stablehlo.rsqrt %v2800 : tensor<128x256x14x14xf32>
    %v2802 = stablehlo.subtract %v2794, %v2797 : tensor<128x256x14x14xf32>
    %v2803 = stablehlo.multiply %v2802, %v2801 : tensor<128x256x14x14xf32>
    %v2804 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2805 = stablehlo.reshape %v2793 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2806 = stablehlo.multiply %v2804, %v2805 : tensor<128x256x14x14xf32>
    %v2807 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2808 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v2809 = stablehlo.reduce(%v2806 init: %v2807) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2810 = stablehlo.divide %v2809, %v2808 : tensor<256xf32>
    %v2811 = stablehlo.multiply %v2803, %v2806 : tensor<128x256x14x14xf32>
    %v2812 = stablehlo.reduce(%v2811 init: %v2807) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2813 = stablehlo.divide %v2812, %v2808 : tensor<256xf32>
    %v2814 = stablehlo.concatenate %v2810, %v2813, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2815 = stablehlo.concatenate %v881, %v2814, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b0g2dst = "stablehlo.all_reduce"(%v2815) ({
    ^bb0(%aras3b0g2dst: tensor<f32>, %arbs3b0g2dst: tensor<f32>):
      %aradds3b0g2dst = stablehlo.add %aras3b0g2dst, %arbs3b0g2dst : tensor<f32>
      stablehlo.return %aradds3b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g2dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b0g2dst = stablehlo.divide %arsums3b0g2dst, %arns3b0g2dst : tensor<1024xf32>
    %v2816 = stablehlo.reshape %v862 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2817 = stablehlo.slice %armeans3b0g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2818 = stablehlo.slice %armeans3b0g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2819 = stablehlo.slice %armeans3b0g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2820 = stablehlo.slice %armeans3b0g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2821 = stablehlo.broadcast_in_dim %v2817, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2822 = stablehlo.broadcast_in_dim %v2818, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2823 = stablehlo.broadcast_in_dim %v2819, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2824 = stablehlo.broadcast_in_dim %v2820, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2825 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2826 = stablehlo.add %v2822, %v2825 : tensor<128x256x14x14xf32>
    %v2827 = stablehlo.rsqrt %v2826 : tensor<128x256x14x14xf32>
    %v2828 = stablehlo.subtract %v2816, %v2821 : tensor<128x256x14x14xf32>
    %v2829 = stablehlo.multiply %v2828, %v2827 : tensor<128x256x14x14xf32>
    %v2830 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2831 = stablehlo.reshape %v2793 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2832 = stablehlo.multiply %v2830, %v2831 : tensor<128x256x14x14xf32>
    %v2833 = stablehlo.subtract %v2832, %v2823 : tensor<128x256x14x14xf32>
    %v2834 = stablehlo.multiply %v2829, %v2824 : tensor<128x256x14x14xf32>
    %v2835 = stablehlo.subtract %v2833, %v2834 : tensor<128x256x14x14xf32>
    %v2836 = stablehlo.multiply %v2827, %v2835 : tensor<128x256x14x14xf32>
    %v2837 = stablehlo.reshape %v2836 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2838 = stablehlo.reshape %v2837 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2839 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2840 = stablehlo.transpose %v2839, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2841 = stablehlo.convolution(%v2838, %v2840)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2842 = stablehlo.reshape %v2841 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2843 = stablehlo.reshape %v2842 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2844 = stablehlo.reshape %v855 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2845 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2846 = stablehlo.compare GT, %v2844, %v2845 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2847 = stablehlo.select %v2846, %v2843, %v2845 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2848 = stablehlo.reshape %v2847 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2849 = stablehlo.reshape %v821 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2850 = stablehlo.slice %v840 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2851 = stablehlo.slice %v840 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2852 = stablehlo.broadcast_in_dim %v2850, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2853 = stablehlo.broadcast_in_dim %v2851, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2854 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2855 = stablehlo.add %v2853, %v2854 : tensor<128x256x14x14xf32>
    %v2856 = stablehlo.rsqrt %v2855 : tensor<128x256x14x14xf32>
    %v2857 = stablehlo.subtract %v2849, %v2852 : tensor<128x256x14x14xf32>
    %v2858 = stablehlo.multiply %v2857, %v2856 : tensor<128x256x14x14xf32>
    %v2859 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2860 = stablehlo.reshape %v2848 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2861 = stablehlo.multiply %v2859, %v2860 : tensor<128x256x14x14xf32>
    %v2862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2863 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v2864 = stablehlo.reduce(%v2861 init: %v2862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2865 = stablehlo.divide %v2864, %v2863 : tensor<256xf32>
    %v2866 = stablehlo.multiply %v2858, %v2861 : tensor<128x256x14x14xf32>
    %v2867 = stablehlo.reduce(%v2866 init: %v2862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2868 = stablehlo.divide %v2867, %v2863 : tensor<256xf32>
    %v2869 = stablehlo.concatenate %v2865, %v2868, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2870 = stablehlo.concatenate %v840, %v2869, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b0g1dst = "stablehlo.all_reduce"(%v2870) ({
    ^bb0(%aras3b0g1dst: tensor<f32>, %arbs3b0g1dst: tensor<f32>):
      %aradds3b0g1dst = stablehlo.add %aras3b0g1dst, %arbs3b0g1dst : tensor<f32>
      stablehlo.return %aradds3b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g1dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeans3b0g1dst = stablehlo.divide %arsums3b0g1dst, %arns3b0g1dst : tensor<1024xf32>
    %v2871 = stablehlo.reshape %v821 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2872 = stablehlo.slice %armeans3b0g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2873 = stablehlo.slice %armeans3b0g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2874 = stablehlo.slice %armeans3b0g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2875 = stablehlo.slice %armeans3b0g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2876 = stablehlo.broadcast_in_dim %v2872, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2877 = stablehlo.broadcast_in_dim %v2873, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2878 = stablehlo.broadcast_in_dim %v2874, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2879 = stablehlo.broadcast_in_dim %v2875, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2880 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2881 = stablehlo.add %v2877, %v2880 : tensor<128x256x14x14xf32>
    %v2882 = stablehlo.rsqrt %v2881 : tensor<128x256x14x14xf32>
    %v2883 = stablehlo.subtract %v2871, %v2876 : tensor<128x256x14x14xf32>
    %v2884 = stablehlo.multiply %v2883, %v2882 : tensor<128x256x14x14xf32>
    %v2885 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2886 = stablehlo.reshape %v2848 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2887 = stablehlo.multiply %v2885, %v2886 : tensor<128x256x14x14xf32>
    %v2888 = stablehlo.subtract %v2887, %v2878 : tensor<128x256x14x14xf32>
    %v2889 = stablehlo.multiply %v2884, %v2879 : tensor<128x256x14x14xf32>
    %v2890 = stablehlo.subtract %v2888, %v2889 : tensor<128x256x14x14xf32>
    %v2891 = stablehlo.multiply %v2882, %v2890 : tensor<128x256x14x14xf32>
    %v2892 = stablehlo.reshape %v2891 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2893 = stablehlo.reshape %v2892 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2894 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2895 = stablehlo.transpose %v2894, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2896 = stablehlo.convolution(%v2893, %v2895)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v2897 = stablehlo.reshape %v2896 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2898 = stablehlo.reshape %v2897 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2899 = stablehlo.reshape %v2793 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2900 = stablehlo.add %v2898, %v2899 : tensor<128x256x14x14xf32>
    %v2901 = stablehlo.reshape %v2900 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2902 = stablehlo.reshape %v816 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2903 = stablehlo.reshape %v2892 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2904 = stablehlo.transpose %v2902, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2905 = stablehlo.transpose %v2903, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2906 = stablehlo.convolution(%v2904, %v2905)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2907 = stablehlo.transpose %v2906, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2908 = stablehlo.reshape %v821 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2909 = stablehlo.slice %v840 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2910 = stablehlo.slice %v840 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2911 = stablehlo.broadcast_in_dim %v2909, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2912 = stablehlo.broadcast_in_dim %v2910, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2913 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2914 = stablehlo.add %v2912, %v2913 : tensor<128x256x14x14xf32>
    %v2915 = stablehlo.rsqrt %v2914 : tensor<128x256x14x14xf32>
    %v2916 = stablehlo.subtract %v2908, %v2911 : tensor<128x256x14x14xf32>
    %v2917 = stablehlo.multiply %v2916, %v2915 : tensor<128x256x14x14xf32>
    %v2918 = stablehlo.reshape %v2848 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2919 = stablehlo.multiply %v2918, %v2917 : tensor<128x256x14x14xf32>
    %v2920 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2921 = stablehlo.reduce(%v2919 init: %v2920) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2922 = stablehlo.reshape %v2848 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2923 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2924 = stablehlo.reduce(%v2922 init: %v2923) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2925 = stablehlo.reshape %v857 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2926 = stablehlo.reshape %v2837 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2927 = stablehlo.transpose %v2925, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2928 = stablehlo.transpose %v2926, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v2929 = stablehlo.convolution(%v2927, %v2928)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2930 = stablehlo.transpose %v2929, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2931 = stablehlo.reshape %v862 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2932 = stablehlo.slice %v881 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2933 = stablehlo.slice %v881 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2934 = stablehlo.broadcast_in_dim %v2932, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2935 = stablehlo.broadcast_in_dim %v2933, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2936 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2937 = stablehlo.add %v2935, %v2936 : tensor<128x256x14x14xf32>
    %v2938 = stablehlo.rsqrt %v2937 : tensor<128x256x14x14xf32>
    %v2939 = stablehlo.subtract %v2931, %v2934 : tensor<128x256x14x14xf32>
    %v2940 = stablehlo.multiply %v2939, %v2938 : tensor<128x256x14x14xf32>
    %v2941 = stablehlo.reshape %v2793 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2942 = stablehlo.multiply %v2941, %v2940 : tensor<128x256x14x14xf32>
    %v2943 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2944 = stablehlo.reduce(%v2942 init: %v2943) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2945 = stablehlo.reshape %v2793 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2946 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2947 = stablehlo.reduce(%v2945 init: %v2946) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2948 = stablehlo.reshape %v2901 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2949 = stablehlo.reshape %v814 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2950 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v2951 = stablehlo.compare GT, %v2949, %v2950 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v2952 = stablehlo.select %v2951, %v2948, %v2950 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v2953 = stablehlo.reshape %v2952 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2954 = stablehlo.reshape %v740 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2955 = stablehlo.slice %v759 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2956 = stablehlo.slice %v759 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2957 = stablehlo.broadcast_in_dim %v2955, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2958 = stablehlo.broadcast_in_dim %v2956, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2959 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2960 = stablehlo.add %v2958, %v2959 : tensor<128x256x14x14xf32>
    %v2961 = stablehlo.rsqrt %v2960 : tensor<128x256x14x14xf32>
    %v2962 = stablehlo.subtract %v2954, %v2957 : tensor<128x256x14x14xf32>
    %v2963 = stablehlo.multiply %v2962, %v2961 : tensor<128x256x14x14xf32>
    %v2964 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2965 = stablehlo.reshape %v2953 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2966 = stablehlo.multiply %v2964, %v2965 : tensor<128x256x14x14xf32>
    %v2967 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2968 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v2969 = stablehlo.reduce(%v2966 init: %v2967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2970 = stablehlo.divide %v2969, %v2968 : tensor<256xf32>
    %v2971 = stablehlo.multiply %v2963, %v2966 : tensor<128x256x14x14xf32>
    %v2972 = stablehlo.reduce(%v2971 init: %v2967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2973 = stablehlo.divide %v2972, %v2968 : tensor<256xf32>
    %v2974 = stablehlo.concatenate %v2970, %v2973, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2975 = stablehlo.concatenate %v759, %v2974, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsumd3g2dst = "stablehlo.all_reduce"(%v2975) ({
    ^bb0(%arad3g2dst: tensor<f32>, %arbd3g2dst: tensor<f32>):
      %araddd3g2dst = stablehlo.add %arad3g2dst, %arbd3g2dst : tensor<f32>
      stablehlo.return %araddd3g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arnd3g2dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeand3g2dst = stablehlo.divide %arsumd3g2dst, %arnd3g2dst : tensor<1024xf32>
    %v2976 = stablehlo.reshape %v740 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2977 = stablehlo.slice %armeand3g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2978 = stablehlo.slice %armeand3g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2979 = stablehlo.slice %armeand3g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2980 = stablehlo.slice %armeand3g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2981 = stablehlo.broadcast_in_dim %v2977, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2982 = stablehlo.broadcast_in_dim %v2978, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2983 = stablehlo.broadcast_in_dim %v2979, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2984 = stablehlo.broadcast_in_dim %v2980, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2985 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v2986 = stablehlo.add %v2982, %v2985 : tensor<128x256x14x14xf32>
    %v2987 = stablehlo.rsqrt %v2986 : tensor<128x256x14x14xf32>
    %v2988 = stablehlo.subtract %v2976, %v2981 : tensor<128x256x14x14xf32>
    %v2989 = stablehlo.multiply %v2988, %v2987 : tensor<128x256x14x14xf32>
    %v2990 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v2991 = stablehlo.reshape %v2953 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2992 = stablehlo.multiply %v2990, %v2991 : tensor<128x256x14x14xf32>
    %v2993 = stablehlo.subtract %v2992, %v2983 : tensor<128x256x14x14xf32>
    %v2994 = stablehlo.multiply %v2989, %v2984 : tensor<128x256x14x14xf32>
    %v2995 = stablehlo.subtract %v2993, %v2994 : tensor<128x256x14x14xf32>
    %v2996 = stablehlo.multiply %v2987, %v2995 : tensor<128x256x14x14xf32>
    %v2997 = stablehlo.reshape %v2996 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v2998 = stablehlo.reshape %v2997 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v2999 = stablehlo.reverse %d3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3000 = stablehlo.transpose %v2999, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3001 = stablehlo.convolution(%v2998, %v3000)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<128x256x14x14xf32>
    %v3002 = stablehlo.reshape %v3001 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v3003 = stablehlo.reshape %v3002 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3004 = stablehlo.reshape %v733 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3005 = stablehlo.constant dense<0.0> : tensor<128x256x14x14xf32>
    %v3006 = stablehlo.compare GT, %v3004, %v3005 : (tensor<128x256x14x14xf32>, tensor<128x256x14x14xf32>) -> tensor<128x256x14x14xi1>
    %v3007 = stablehlo.select %v3006, %v3003, %v3005 : tensor<128x256x14x14xi1>, tensor<128x256x14x14xf32>
    %v3008 = stablehlo.reshape %v3007 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v3009 = stablehlo.reshape %v699 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3010 = stablehlo.slice %v718 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3011 = stablehlo.slice %v718 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3012 = stablehlo.broadcast_in_dim %v3010, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3013 = stablehlo.broadcast_in_dim %v3011, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3014 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v3015 = stablehlo.add %v3013, %v3014 : tensor<128x256x14x14xf32>
    %v3016 = stablehlo.rsqrt %v3015 : tensor<128x256x14x14xf32>
    %v3017 = stablehlo.subtract %v3009, %v3012 : tensor<128x256x14x14xf32>
    %v3018 = stablehlo.multiply %v3017, %v3016 : tensor<128x256x14x14xf32>
    %v3019 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3020 = stablehlo.reshape %v3008 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3021 = stablehlo.multiply %v3019, %v3020 : tensor<128x256x14x14xf32>
    %v3022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3023 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v3024 = stablehlo.reduce(%v3021 init: %v3022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3025 = stablehlo.divide %v3024, %v3023 : tensor<256xf32>
    %v3026 = stablehlo.multiply %v3018, %v3021 : tensor<128x256x14x14xf32>
    %v3027 = stablehlo.reduce(%v3026 init: %v3022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3028 = stablehlo.divide %v3027, %v3023 : tensor<256xf32>
    %v3029 = stablehlo.concatenate %v3025, %v3028, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3030 = stablehlo.concatenate %v718, %v3029, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsumd3g1dst = "stablehlo.all_reduce"(%v3030) ({
    ^bb0(%arad3g1dst: tensor<f32>, %arbd3g1dst: tensor<f32>):
      %araddd3g1dst = stablehlo.add %arad3g1dst, %arbd3g1dst : tensor<f32>
      stablehlo.return %araddd3g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arnd3g1dst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeand3g1dst = stablehlo.divide %arsumd3g1dst, %arnd3g1dst : tensor<1024xf32>
    %v3031 = stablehlo.reshape %v699 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3032 = stablehlo.slice %armeand3g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3033 = stablehlo.slice %armeand3g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3034 = stablehlo.slice %armeand3g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3035 = stablehlo.slice %armeand3g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3036 = stablehlo.broadcast_in_dim %v3032, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3037 = stablehlo.broadcast_in_dim %v3033, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3038 = stablehlo.broadcast_in_dim %v3034, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3039 = stablehlo.broadcast_in_dim %v3035, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3040 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v3041 = stablehlo.add %v3037, %v3040 : tensor<128x256x14x14xf32>
    %v3042 = stablehlo.rsqrt %v3041 : tensor<128x256x14x14xf32>
    %v3043 = stablehlo.subtract %v3031, %v3036 : tensor<128x256x14x14xf32>
    %v3044 = stablehlo.multiply %v3043, %v3042 : tensor<128x256x14x14xf32>
    %v3045 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3046 = stablehlo.reshape %v3008 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3047 = stablehlo.multiply %v3045, %v3046 : tensor<128x256x14x14xf32>
    %v3048 = stablehlo.subtract %v3047, %v3038 : tensor<128x256x14x14xf32>
    %v3049 = stablehlo.multiply %v3044, %v3039 : tensor<128x256x14x14xf32>
    %v3050 = stablehlo.subtract %v3048, %v3049 : tensor<128x256x14x14xf32>
    %v3051 = stablehlo.multiply %v3042, %v3050 : tensor<128x256x14x14xf32>
    %v3052 = stablehlo.reshape %v3051 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v3053 = stablehlo.reshape %v3052 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3054 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3055 = stablehlo.pad %v3053, %v3054, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<128x256x28x28xf32>
    %v3056 = stablehlo.reverse %d3W1, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v3057 = stablehlo.transpose %v3056, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v3058 = stablehlo.convolution(%v3055, %v3057)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v3059 = stablehlo.reshape %v3058 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3060 = stablehlo.reshape %v779 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3061 = stablehlo.slice %v798 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3062 = stablehlo.slice %v798 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3063 = stablehlo.broadcast_in_dim %v3061, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3064 = stablehlo.broadcast_in_dim %v3062, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3065 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v3066 = stablehlo.add %v3064, %v3065 : tensor<128x256x14x14xf32>
    %v3067 = stablehlo.rsqrt %v3066 : tensor<128x256x14x14xf32>
    %v3068 = stablehlo.subtract %v3060, %v3063 : tensor<128x256x14x14xf32>
    %v3069 = stablehlo.multiply %v3068, %v3067 : tensor<128x256x14x14xf32>
    %v3070 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3071 = stablehlo.reshape %v2953 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3072 = stablehlo.multiply %v3070, %v3071 : tensor<128x256x14x14xf32>
    %v3073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3074 = stablehlo.constant dense<25088.0> : tensor<256xf32>
    %v3075 = stablehlo.reduce(%v3072 init: %v3073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3076 = stablehlo.divide %v3075, %v3074 : tensor<256xf32>
    %v3077 = stablehlo.multiply %v3069, %v3072 : tensor<128x256x14x14xf32>
    %v3078 = stablehlo.reduce(%v3077 init: %v3073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3079 = stablehlo.divide %v3078, %v3074 : tensor<256xf32>
    %v3080 = stablehlo.concatenate %v3076, %v3079, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3081 = stablehlo.concatenate %v798, %v3080, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsumd3gpdst = "stablehlo.all_reduce"(%v3081) ({
    ^bb0(%arad3gpdst: tensor<f32>, %arbd3gpdst: tensor<f32>):
      %araddd3gpdst = stablehlo.add %arad3gpdst, %arbd3gpdst : tensor<f32>
      stablehlo.return %araddd3gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arnd3gpdst = stablehlo.constant dense<2.0> : tensor<1024xf32>
    %armeand3gpdst = stablehlo.divide %arsumd3gpdst, %arnd3gpdst : tensor<1024xf32>
    %v3082 = stablehlo.reshape %v779 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3083 = stablehlo.slice %armeand3gpdst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3084 = stablehlo.slice %armeand3gpdst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3085 = stablehlo.slice %armeand3gpdst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3086 = stablehlo.slice %armeand3gpdst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3087 = stablehlo.broadcast_in_dim %v3083, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3088 = stablehlo.broadcast_in_dim %v3084, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3089 = stablehlo.broadcast_in_dim %v3085, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3090 = stablehlo.broadcast_in_dim %v3086, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3091 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v3092 = stablehlo.add %v3088, %v3091 : tensor<128x256x14x14xf32>
    %v3093 = stablehlo.rsqrt %v3092 : tensor<128x256x14x14xf32>
    %v3094 = stablehlo.subtract %v3082, %v3087 : tensor<128x256x14x14xf32>
    %v3095 = stablehlo.multiply %v3094, %v3093 : tensor<128x256x14x14xf32>
    %v3096 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3097 = stablehlo.reshape %v2953 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3098 = stablehlo.multiply %v3096, %v3097 : tensor<128x256x14x14xf32>
    %v3099 = stablehlo.subtract %v3098, %v3089 : tensor<128x256x14x14xf32>
    %v3100 = stablehlo.multiply %v3095, %v3090 : tensor<128x256x14x14xf32>
    %v3101 = stablehlo.subtract %v3099, %v3100 : tensor<128x256x14x14xf32>
    %v3102 = stablehlo.multiply %v3093, %v3101 : tensor<128x256x14x14xf32>
    %v3103 = stablehlo.reshape %v3102 : (tensor<128x256x14x14xf32>) -> tensor<128x50176xf32>
    %v3104 = stablehlo.reshape %v3103 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3105 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3106 = stablehlo.pad %v3104, %v3105, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<128x256x28x28xf32>
    %v3107 = stablehlo.reverse %d3Wp, dims = [2, 3] : tensor<256x128x1x1xf32>
    %v3108 = stablehlo.transpose %v3107, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v3109 = stablehlo.convolution(%v3106, %v3108)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x1x1xf32>) -> tensor<128x128x28x28xf32>
    %v3110 = stablehlo.reshape %v3109 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3111 = stablehlo.reshape %v3059 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3112 = stablehlo.reshape %v3110 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3113 = stablehlo.add %v3111, %v3112 : tensor<128x128x28x28xf32>
    %v3114 = stablehlo.reshape %v3113 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3115 = stablehlo.reshape %v694 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3116 = stablehlo.reshape %v3052 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3118 = stablehlo.pad %v3116, %v3117, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<128x256x28x28xf32>
    %v3119 = stablehlo.transpose %v3115, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3120 = stablehlo.transpose %v3118, dims = [1, 0, 2, 3] : (tensor<128x256x28x28xf32>) -> tensor<256x128x28x28xf32>
    %v3121 = stablehlo.convolution(%v3119, %v3120)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<128x256x3x3xf32>
    %v3122 = stablehlo.transpose %v3121, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v3123 = stablehlo.reshape %v699 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3124 = stablehlo.slice %v718 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3125 = stablehlo.slice %v718 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3126 = stablehlo.broadcast_in_dim %v3124, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3127 = stablehlo.broadcast_in_dim %v3125, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3128 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v3129 = stablehlo.add %v3127, %v3128 : tensor<128x256x14x14xf32>
    %v3130 = stablehlo.rsqrt %v3129 : tensor<128x256x14x14xf32>
    %v3131 = stablehlo.subtract %v3123, %v3126 : tensor<128x256x14x14xf32>
    %v3132 = stablehlo.multiply %v3131, %v3130 : tensor<128x256x14x14xf32>
    %v3133 = stablehlo.reshape %v3008 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3134 = stablehlo.multiply %v3133, %v3132 : tensor<128x256x14x14xf32>
    %v3135 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3136 = stablehlo.reduce(%v3134 init: %v3135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3137 = stablehlo.reshape %v3008 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3139 = stablehlo.reduce(%v3137 init: %v3138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3140 = stablehlo.reshape %v735 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3141 = stablehlo.reshape %v2997 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3142 = stablehlo.transpose %v3140, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v3143 = stablehlo.transpose %v3141, dims = [1, 0, 2, 3] : (tensor<128x256x14x14xf32>) -> tensor<256x128x14x14xf32>
    %v3144 = stablehlo.convolution(%v3142, %v3143)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x14x14xf32>, tensor<256x128x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v3145 = stablehlo.transpose %v3144, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3146 = stablehlo.reshape %v740 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3147 = stablehlo.slice %v759 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3148 = stablehlo.slice %v759 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3149 = stablehlo.broadcast_in_dim %v3147, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3150 = stablehlo.broadcast_in_dim %v3148, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3151 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v3152 = stablehlo.add %v3150, %v3151 : tensor<128x256x14x14xf32>
    %v3153 = stablehlo.rsqrt %v3152 : tensor<128x256x14x14xf32>
    %v3154 = stablehlo.subtract %v3146, %v3149 : tensor<128x256x14x14xf32>
    %v3155 = stablehlo.multiply %v3154, %v3153 : tensor<128x256x14x14xf32>
    %v3156 = stablehlo.reshape %v2953 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3157 = stablehlo.multiply %v3156, %v3155 : tensor<128x256x14x14xf32>
    %v3158 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3159 = stablehlo.reduce(%v3157 init: %v3158) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3160 = stablehlo.reshape %v2953 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3161 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3162 = stablehlo.reduce(%v3160 init: %v3161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3163 = stablehlo.reshape %v694 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3164 = stablehlo.reshape %v3103 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3166 = stablehlo.pad %v3164, %v3165, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<128x256x28x28xf32>
    %v3167 = stablehlo.transpose %v3163, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3168 = stablehlo.transpose %v3166, dims = [1, 0, 2, 3] : (tensor<128x256x28x28xf32>) -> tensor<256x128x28x28xf32>
    %v3169 = stablehlo.convolution(%v3167, %v3168)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<128x256x1x1xf32>
    %v3170 = stablehlo.transpose %v3169, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v3171 = stablehlo.reshape %v779 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3172 = stablehlo.slice %v798 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3173 = stablehlo.slice %v798 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3174 = stablehlo.broadcast_in_dim %v3172, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3175 = stablehlo.broadcast_in_dim %v3173, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %v3176 = stablehlo.constant dense<1.0e-05> : tensor<128x256x14x14xf32>
    %v3177 = stablehlo.add %v3175, %v3176 : tensor<128x256x14x14xf32>
    %v3178 = stablehlo.rsqrt %v3177 : tensor<128x256x14x14xf32>
    %v3179 = stablehlo.subtract %v3171, %v3174 : tensor<128x256x14x14xf32>
    %v3180 = stablehlo.multiply %v3179, %v3178 : tensor<128x256x14x14xf32>
    %v3181 = stablehlo.reshape %v2953 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3182 = stablehlo.multiply %v3181, %v3180 : tensor<128x256x14x14xf32>
    %v3183 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3184 = stablehlo.reduce(%v3182 init: %v3183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3185 = stablehlo.reshape %v2953 : (tensor<128x50176xf32>) -> tensor<128x256x14x14xf32>
    %v3186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3187 = stablehlo.reduce(%v3185 init: %v3186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3188 = stablehlo.reshape %v3114 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3189 = stablehlo.reshape %v690 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3190 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v3191 = stablehlo.compare GT, %v3189, %v3190 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v3192 = stablehlo.select %v3191, %v3188, %v3190 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v3193 = stablehlo.reshape %v3192 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3194 = stablehlo.reshape %v652 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3195 = stablehlo.slice %v671 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3196 = stablehlo.slice %v671 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3197 = stablehlo.broadcast_in_dim %v3195, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3198 = stablehlo.broadcast_in_dim %v3196, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3199 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3200 = stablehlo.add %v3198, %v3199 : tensor<128x128x28x28xf32>
    %v3201 = stablehlo.rsqrt %v3200 : tensor<128x128x28x28xf32>
    %v3202 = stablehlo.subtract %v3194, %v3197 : tensor<128x128x28x28xf32>
    %v3203 = stablehlo.multiply %v3202, %v3201 : tensor<128x128x28x28xf32>
    %v3204 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3205 = stablehlo.reshape %v3193 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3206 = stablehlo.multiply %v3204, %v3205 : tensor<128x128x28x28xf32>
    %v3207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3208 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3209 = stablehlo.reduce(%v3206 init: %v3207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3210 = stablehlo.divide %v3209, %v3208 : tensor<128xf32>
    %v3211 = stablehlo.multiply %v3203, %v3206 : tensor<128x128x28x28xf32>
    %v3212 = stablehlo.reduce(%v3211 init: %v3207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3213 = stablehlo.divide %v3212, %v3208 : tensor<128xf32>
    %v3214 = stablehlo.concatenate %v3210, %v3213, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3215 = stablehlo.concatenate %v671, %v3214, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b2g2dst = "stablehlo.all_reduce"(%v3215) ({
    ^bb0(%aras2b2g2dst: tensor<f32>, %arbs2b2g2dst: tensor<f32>):
      %aradds2b2g2dst = stablehlo.add %aras2b2g2dst, %arbs2b2g2dst : tensor<f32>
      stablehlo.return %aradds2b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g2dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans2b2g2dst = stablehlo.divide %arsums2b2g2dst, %arns2b2g2dst : tensor<512xf32>
    %v3216 = stablehlo.reshape %v652 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3217 = stablehlo.slice %armeans2b2g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3218 = stablehlo.slice %armeans2b2g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3219 = stablehlo.slice %armeans2b2g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3220 = stablehlo.slice %armeans2b2g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3221 = stablehlo.broadcast_in_dim %v3217, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3222 = stablehlo.broadcast_in_dim %v3218, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3223 = stablehlo.broadcast_in_dim %v3219, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3224 = stablehlo.broadcast_in_dim %v3220, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3225 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3226 = stablehlo.add %v3222, %v3225 : tensor<128x128x28x28xf32>
    %v3227 = stablehlo.rsqrt %v3226 : tensor<128x128x28x28xf32>
    %v3228 = stablehlo.subtract %v3216, %v3221 : tensor<128x128x28x28xf32>
    %v3229 = stablehlo.multiply %v3228, %v3227 : tensor<128x128x28x28xf32>
    %v3230 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3231 = stablehlo.reshape %v3193 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3232 = stablehlo.multiply %v3230, %v3231 : tensor<128x128x28x28xf32>
    %v3233 = stablehlo.subtract %v3232, %v3223 : tensor<128x128x28x28xf32>
    %v3234 = stablehlo.multiply %v3229, %v3224 : tensor<128x128x28x28xf32>
    %v3235 = stablehlo.subtract %v3233, %v3234 : tensor<128x128x28x28xf32>
    %v3236 = stablehlo.multiply %v3227, %v3235 : tensor<128x128x28x28xf32>
    %v3237 = stablehlo.reshape %v3236 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3238 = stablehlo.reshape %v3237 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3239 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3240 = stablehlo.transpose %v3239, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3241 = stablehlo.convolution(%v3238, %v3240)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v3242 = stablehlo.reshape %v3241 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3243 = stablehlo.reshape %v3242 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3244 = stablehlo.reshape %v645 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3245 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v3246 = stablehlo.compare GT, %v3244, %v3245 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v3247 = stablehlo.select %v3246, %v3243, %v3245 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v3248 = stablehlo.reshape %v3247 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3249 = stablehlo.reshape %v611 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3250 = stablehlo.slice %v630 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3251 = stablehlo.slice %v630 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3252 = stablehlo.broadcast_in_dim %v3250, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3253 = stablehlo.broadcast_in_dim %v3251, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3254 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3255 = stablehlo.add %v3253, %v3254 : tensor<128x128x28x28xf32>
    %v3256 = stablehlo.rsqrt %v3255 : tensor<128x128x28x28xf32>
    %v3257 = stablehlo.subtract %v3249, %v3252 : tensor<128x128x28x28xf32>
    %v3258 = stablehlo.multiply %v3257, %v3256 : tensor<128x128x28x28xf32>
    %v3259 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3260 = stablehlo.reshape %v3248 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3261 = stablehlo.multiply %v3259, %v3260 : tensor<128x128x28x28xf32>
    %v3262 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3263 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3264 = stablehlo.reduce(%v3261 init: %v3262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3265 = stablehlo.divide %v3264, %v3263 : tensor<128xf32>
    %v3266 = stablehlo.multiply %v3258, %v3261 : tensor<128x128x28x28xf32>
    %v3267 = stablehlo.reduce(%v3266 init: %v3262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3268 = stablehlo.divide %v3267, %v3263 : tensor<128xf32>
    %v3269 = stablehlo.concatenate %v3265, %v3268, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3270 = stablehlo.concatenate %v630, %v3269, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b2g1dst = "stablehlo.all_reduce"(%v3270) ({
    ^bb0(%aras2b2g1dst: tensor<f32>, %arbs2b2g1dst: tensor<f32>):
      %aradds2b2g1dst = stablehlo.add %aras2b2g1dst, %arbs2b2g1dst : tensor<f32>
      stablehlo.return %aradds2b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g1dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans2b2g1dst = stablehlo.divide %arsums2b2g1dst, %arns2b2g1dst : tensor<512xf32>
    %v3271 = stablehlo.reshape %v611 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3272 = stablehlo.slice %armeans2b2g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3273 = stablehlo.slice %armeans2b2g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3274 = stablehlo.slice %armeans2b2g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3275 = stablehlo.slice %armeans2b2g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3276 = stablehlo.broadcast_in_dim %v3272, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3277 = stablehlo.broadcast_in_dim %v3273, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3278 = stablehlo.broadcast_in_dim %v3274, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3279 = stablehlo.broadcast_in_dim %v3275, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3280 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3281 = stablehlo.add %v3277, %v3280 : tensor<128x128x28x28xf32>
    %v3282 = stablehlo.rsqrt %v3281 : tensor<128x128x28x28xf32>
    %v3283 = stablehlo.subtract %v3271, %v3276 : tensor<128x128x28x28xf32>
    %v3284 = stablehlo.multiply %v3283, %v3282 : tensor<128x128x28x28xf32>
    %v3285 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3286 = stablehlo.reshape %v3248 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3287 = stablehlo.multiply %v3285, %v3286 : tensor<128x128x28x28xf32>
    %v3288 = stablehlo.subtract %v3287, %v3278 : tensor<128x128x28x28xf32>
    %v3289 = stablehlo.multiply %v3284, %v3279 : tensor<128x128x28x28xf32>
    %v3290 = stablehlo.subtract %v3288, %v3289 : tensor<128x128x28x28xf32>
    %v3291 = stablehlo.multiply %v3282, %v3290 : tensor<128x128x28x28xf32>
    %v3292 = stablehlo.reshape %v3291 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3293 = stablehlo.reshape %v3292 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3294 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3295 = stablehlo.transpose %v3294, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3296 = stablehlo.convolution(%v3293, %v3295)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v3297 = stablehlo.reshape %v3296 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3298 = stablehlo.reshape %v3297 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3299 = stablehlo.reshape %v3193 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3300 = stablehlo.add %v3298, %v3299 : tensor<128x128x28x28xf32>
    %v3301 = stablehlo.reshape %v3300 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3302 = stablehlo.reshape %v606 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3303 = stablehlo.reshape %v3292 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3304 = stablehlo.transpose %v3302, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3305 = stablehlo.transpose %v3303, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3306 = stablehlo.convolution(%v3304, %v3305)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3307 = stablehlo.transpose %v3306, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3308 = stablehlo.reshape %v611 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3309 = stablehlo.slice %v630 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3310 = stablehlo.slice %v630 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3311 = stablehlo.broadcast_in_dim %v3309, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3312 = stablehlo.broadcast_in_dim %v3310, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3313 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3314 = stablehlo.add %v3312, %v3313 : tensor<128x128x28x28xf32>
    %v3315 = stablehlo.rsqrt %v3314 : tensor<128x128x28x28xf32>
    %v3316 = stablehlo.subtract %v3308, %v3311 : tensor<128x128x28x28xf32>
    %v3317 = stablehlo.multiply %v3316, %v3315 : tensor<128x128x28x28xf32>
    %v3318 = stablehlo.reshape %v3248 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3319 = stablehlo.multiply %v3318, %v3317 : tensor<128x128x28x28xf32>
    %v3320 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3321 = stablehlo.reduce(%v3319 init: %v3320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3322 = stablehlo.reshape %v3248 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3324 = stablehlo.reduce(%v3322 init: %v3323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3325 = stablehlo.reshape %v647 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3326 = stablehlo.reshape %v3237 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3327 = stablehlo.transpose %v3325, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3328 = stablehlo.transpose %v3326, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3329 = stablehlo.convolution(%v3327, %v3328)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3330 = stablehlo.transpose %v3329, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3331 = stablehlo.reshape %v652 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3332 = stablehlo.slice %v671 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3333 = stablehlo.slice %v671 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3334 = stablehlo.broadcast_in_dim %v3332, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3335 = stablehlo.broadcast_in_dim %v3333, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3336 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3337 = stablehlo.add %v3335, %v3336 : tensor<128x128x28x28xf32>
    %v3338 = stablehlo.rsqrt %v3337 : tensor<128x128x28x28xf32>
    %v3339 = stablehlo.subtract %v3331, %v3334 : tensor<128x128x28x28xf32>
    %v3340 = stablehlo.multiply %v3339, %v3338 : tensor<128x128x28x28xf32>
    %v3341 = stablehlo.reshape %v3193 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3342 = stablehlo.multiply %v3341, %v3340 : tensor<128x128x28x28xf32>
    %v3343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3344 = stablehlo.reduce(%v3342 init: %v3343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3345 = stablehlo.reshape %v3193 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3346 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3347 = stablehlo.reduce(%v3345 init: %v3346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3348 = stablehlo.reshape %v3301 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3349 = stablehlo.reshape %v602 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3350 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v3351 = stablehlo.compare GT, %v3349, %v3350 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v3352 = stablehlo.select %v3351, %v3348, %v3350 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v3353 = stablehlo.reshape %v3352 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3354 = stablehlo.reshape %v564 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3355 = stablehlo.slice %v583 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3356 = stablehlo.slice %v583 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3357 = stablehlo.broadcast_in_dim %v3355, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3358 = stablehlo.broadcast_in_dim %v3356, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3359 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3360 = stablehlo.add %v3358, %v3359 : tensor<128x128x28x28xf32>
    %v3361 = stablehlo.rsqrt %v3360 : tensor<128x128x28x28xf32>
    %v3362 = stablehlo.subtract %v3354, %v3357 : tensor<128x128x28x28xf32>
    %v3363 = stablehlo.multiply %v3362, %v3361 : tensor<128x128x28x28xf32>
    %v3364 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3365 = stablehlo.reshape %v3353 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3366 = stablehlo.multiply %v3364, %v3365 : tensor<128x128x28x28xf32>
    %v3367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3368 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3369 = stablehlo.reduce(%v3366 init: %v3367) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3370 = stablehlo.divide %v3369, %v3368 : tensor<128xf32>
    %v3371 = stablehlo.multiply %v3363, %v3366 : tensor<128x128x28x28xf32>
    %v3372 = stablehlo.reduce(%v3371 init: %v3367) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3373 = stablehlo.divide %v3372, %v3368 : tensor<128xf32>
    %v3374 = stablehlo.concatenate %v3370, %v3373, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3375 = stablehlo.concatenate %v583, %v3374, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b1g2dst = "stablehlo.all_reduce"(%v3375) ({
    ^bb0(%aras2b1g2dst: tensor<f32>, %arbs2b1g2dst: tensor<f32>):
      %aradds2b1g2dst = stablehlo.add %aras2b1g2dst, %arbs2b1g2dst : tensor<f32>
      stablehlo.return %aradds2b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g2dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans2b1g2dst = stablehlo.divide %arsums2b1g2dst, %arns2b1g2dst : tensor<512xf32>
    %v3376 = stablehlo.reshape %v564 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3377 = stablehlo.slice %armeans2b1g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3378 = stablehlo.slice %armeans2b1g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3379 = stablehlo.slice %armeans2b1g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3380 = stablehlo.slice %armeans2b1g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3381 = stablehlo.broadcast_in_dim %v3377, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3382 = stablehlo.broadcast_in_dim %v3378, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3383 = stablehlo.broadcast_in_dim %v3379, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3384 = stablehlo.broadcast_in_dim %v3380, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3385 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3386 = stablehlo.add %v3382, %v3385 : tensor<128x128x28x28xf32>
    %v3387 = stablehlo.rsqrt %v3386 : tensor<128x128x28x28xf32>
    %v3388 = stablehlo.subtract %v3376, %v3381 : tensor<128x128x28x28xf32>
    %v3389 = stablehlo.multiply %v3388, %v3387 : tensor<128x128x28x28xf32>
    %v3390 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3391 = stablehlo.reshape %v3353 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3392 = stablehlo.multiply %v3390, %v3391 : tensor<128x128x28x28xf32>
    %v3393 = stablehlo.subtract %v3392, %v3383 : tensor<128x128x28x28xf32>
    %v3394 = stablehlo.multiply %v3389, %v3384 : tensor<128x128x28x28xf32>
    %v3395 = stablehlo.subtract %v3393, %v3394 : tensor<128x128x28x28xf32>
    %v3396 = stablehlo.multiply %v3387, %v3395 : tensor<128x128x28x28xf32>
    %v3397 = stablehlo.reshape %v3396 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3398 = stablehlo.reshape %v3397 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3399 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3400 = stablehlo.transpose %v3399, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3401 = stablehlo.convolution(%v3398, %v3400)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v3402 = stablehlo.reshape %v3401 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3403 = stablehlo.reshape %v3402 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3404 = stablehlo.reshape %v557 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3405 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v3406 = stablehlo.compare GT, %v3404, %v3405 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v3407 = stablehlo.select %v3406, %v3403, %v3405 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v3408 = stablehlo.reshape %v3407 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3409 = stablehlo.reshape %v523 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3410 = stablehlo.slice %v542 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3411 = stablehlo.slice %v542 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3412 = stablehlo.broadcast_in_dim %v3410, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3413 = stablehlo.broadcast_in_dim %v3411, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3414 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3415 = stablehlo.add %v3413, %v3414 : tensor<128x128x28x28xf32>
    %v3416 = stablehlo.rsqrt %v3415 : tensor<128x128x28x28xf32>
    %v3417 = stablehlo.subtract %v3409, %v3412 : tensor<128x128x28x28xf32>
    %v3418 = stablehlo.multiply %v3417, %v3416 : tensor<128x128x28x28xf32>
    %v3419 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3420 = stablehlo.reshape %v3408 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3421 = stablehlo.multiply %v3419, %v3420 : tensor<128x128x28x28xf32>
    %v3422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3423 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3424 = stablehlo.reduce(%v3421 init: %v3422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3425 = stablehlo.divide %v3424, %v3423 : tensor<128xf32>
    %v3426 = stablehlo.multiply %v3418, %v3421 : tensor<128x128x28x28xf32>
    %v3427 = stablehlo.reduce(%v3426 init: %v3422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3428 = stablehlo.divide %v3427, %v3423 : tensor<128xf32>
    %v3429 = stablehlo.concatenate %v3425, %v3428, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3430 = stablehlo.concatenate %v542, %v3429, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b1g1dst = "stablehlo.all_reduce"(%v3430) ({
    ^bb0(%aras2b1g1dst: tensor<f32>, %arbs2b1g1dst: tensor<f32>):
      %aradds2b1g1dst = stablehlo.add %aras2b1g1dst, %arbs2b1g1dst : tensor<f32>
      stablehlo.return %aradds2b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g1dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans2b1g1dst = stablehlo.divide %arsums2b1g1dst, %arns2b1g1dst : tensor<512xf32>
    %v3431 = stablehlo.reshape %v523 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3432 = stablehlo.slice %armeans2b1g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3433 = stablehlo.slice %armeans2b1g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3434 = stablehlo.slice %armeans2b1g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3435 = stablehlo.slice %armeans2b1g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3436 = stablehlo.broadcast_in_dim %v3432, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3437 = stablehlo.broadcast_in_dim %v3433, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3438 = stablehlo.broadcast_in_dim %v3434, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3439 = stablehlo.broadcast_in_dim %v3435, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3440 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3441 = stablehlo.add %v3437, %v3440 : tensor<128x128x28x28xf32>
    %v3442 = stablehlo.rsqrt %v3441 : tensor<128x128x28x28xf32>
    %v3443 = stablehlo.subtract %v3431, %v3436 : tensor<128x128x28x28xf32>
    %v3444 = stablehlo.multiply %v3443, %v3442 : tensor<128x128x28x28xf32>
    %v3445 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3446 = stablehlo.reshape %v3408 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3447 = stablehlo.multiply %v3445, %v3446 : tensor<128x128x28x28xf32>
    %v3448 = stablehlo.subtract %v3447, %v3438 : tensor<128x128x28x28xf32>
    %v3449 = stablehlo.multiply %v3444, %v3439 : tensor<128x128x28x28xf32>
    %v3450 = stablehlo.subtract %v3448, %v3449 : tensor<128x128x28x28xf32>
    %v3451 = stablehlo.multiply %v3442, %v3450 : tensor<128x128x28x28xf32>
    %v3452 = stablehlo.reshape %v3451 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3453 = stablehlo.reshape %v3452 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3454 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3455 = stablehlo.transpose %v3454, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3456 = stablehlo.convolution(%v3453, %v3455)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v3457 = stablehlo.reshape %v3456 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3458 = stablehlo.reshape %v3457 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3459 = stablehlo.reshape %v3353 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3460 = stablehlo.add %v3458, %v3459 : tensor<128x128x28x28xf32>
    %v3461 = stablehlo.reshape %v3460 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3462 = stablehlo.reshape %v518 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3463 = stablehlo.reshape %v3452 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3464 = stablehlo.transpose %v3462, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3465 = stablehlo.transpose %v3463, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3466 = stablehlo.convolution(%v3464, %v3465)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3467 = stablehlo.transpose %v3466, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3468 = stablehlo.reshape %v523 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3469 = stablehlo.slice %v542 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3470 = stablehlo.slice %v542 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3471 = stablehlo.broadcast_in_dim %v3469, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3472 = stablehlo.broadcast_in_dim %v3470, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3473 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3474 = stablehlo.add %v3472, %v3473 : tensor<128x128x28x28xf32>
    %v3475 = stablehlo.rsqrt %v3474 : tensor<128x128x28x28xf32>
    %v3476 = stablehlo.subtract %v3468, %v3471 : tensor<128x128x28x28xf32>
    %v3477 = stablehlo.multiply %v3476, %v3475 : tensor<128x128x28x28xf32>
    %v3478 = stablehlo.reshape %v3408 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3479 = stablehlo.multiply %v3478, %v3477 : tensor<128x128x28x28xf32>
    %v3480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3481 = stablehlo.reduce(%v3479 init: %v3480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3482 = stablehlo.reshape %v3408 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3483 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3484 = stablehlo.reduce(%v3482 init: %v3483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3485 = stablehlo.reshape %v559 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3486 = stablehlo.reshape %v3397 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3487 = stablehlo.transpose %v3485, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3488 = stablehlo.transpose %v3486, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3489 = stablehlo.convolution(%v3487, %v3488)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3490 = stablehlo.transpose %v3489, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3491 = stablehlo.reshape %v564 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3492 = stablehlo.slice %v583 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3493 = stablehlo.slice %v583 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3494 = stablehlo.broadcast_in_dim %v3492, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3495 = stablehlo.broadcast_in_dim %v3493, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3496 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3497 = stablehlo.add %v3495, %v3496 : tensor<128x128x28x28xf32>
    %v3498 = stablehlo.rsqrt %v3497 : tensor<128x128x28x28xf32>
    %v3499 = stablehlo.subtract %v3491, %v3494 : tensor<128x128x28x28xf32>
    %v3500 = stablehlo.multiply %v3499, %v3498 : tensor<128x128x28x28xf32>
    %v3501 = stablehlo.reshape %v3353 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3502 = stablehlo.multiply %v3501, %v3500 : tensor<128x128x28x28xf32>
    %v3503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3504 = stablehlo.reduce(%v3502 init: %v3503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3505 = stablehlo.reshape %v3353 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3507 = stablehlo.reduce(%v3505 init: %v3506) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3508 = stablehlo.reshape %v3461 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3509 = stablehlo.reshape %v514 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3510 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v3511 = stablehlo.compare GT, %v3509, %v3510 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v3512 = stablehlo.select %v3511, %v3508, %v3510 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v3513 = stablehlo.reshape %v3512 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3514 = stablehlo.reshape %v476 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3515 = stablehlo.slice %v495 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3516 = stablehlo.slice %v495 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3517 = stablehlo.broadcast_in_dim %v3515, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3518 = stablehlo.broadcast_in_dim %v3516, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3519 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3520 = stablehlo.add %v3518, %v3519 : tensor<128x128x28x28xf32>
    %v3521 = stablehlo.rsqrt %v3520 : tensor<128x128x28x28xf32>
    %v3522 = stablehlo.subtract %v3514, %v3517 : tensor<128x128x28x28xf32>
    %v3523 = stablehlo.multiply %v3522, %v3521 : tensor<128x128x28x28xf32>
    %v3524 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3525 = stablehlo.reshape %v3513 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3526 = stablehlo.multiply %v3524, %v3525 : tensor<128x128x28x28xf32>
    %v3527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3528 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3529 = stablehlo.reduce(%v3526 init: %v3527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3530 = stablehlo.divide %v3529, %v3528 : tensor<128xf32>
    %v3531 = stablehlo.multiply %v3523, %v3526 : tensor<128x128x28x28xf32>
    %v3532 = stablehlo.reduce(%v3531 init: %v3527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3533 = stablehlo.divide %v3532, %v3528 : tensor<128xf32>
    %v3534 = stablehlo.concatenate %v3530, %v3533, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3535 = stablehlo.concatenate %v495, %v3534, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b0g2dst = "stablehlo.all_reduce"(%v3535) ({
    ^bb0(%aras2b0g2dst: tensor<f32>, %arbs2b0g2dst: tensor<f32>):
      %aradds2b0g2dst = stablehlo.add %aras2b0g2dst, %arbs2b0g2dst : tensor<f32>
      stablehlo.return %aradds2b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g2dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans2b0g2dst = stablehlo.divide %arsums2b0g2dst, %arns2b0g2dst : tensor<512xf32>
    %v3536 = stablehlo.reshape %v476 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3537 = stablehlo.slice %armeans2b0g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3538 = stablehlo.slice %armeans2b0g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3539 = stablehlo.slice %armeans2b0g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3540 = stablehlo.slice %armeans2b0g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3541 = stablehlo.broadcast_in_dim %v3537, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3542 = stablehlo.broadcast_in_dim %v3538, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3543 = stablehlo.broadcast_in_dim %v3539, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3544 = stablehlo.broadcast_in_dim %v3540, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3545 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3546 = stablehlo.add %v3542, %v3545 : tensor<128x128x28x28xf32>
    %v3547 = stablehlo.rsqrt %v3546 : tensor<128x128x28x28xf32>
    %v3548 = stablehlo.subtract %v3536, %v3541 : tensor<128x128x28x28xf32>
    %v3549 = stablehlo.multiply %v3548, %v3547 : tensor<128x128x28x28xf32>
    %v3550 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3551 = stablehlo.reshape %v3513 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3552 = stablehlo.multiply %v3550, %v3551 : tensor<128x128x28x28xf32>
    %v3553 = stablehlo.subtract %v3552, %v3543 : tensor<128x128x28x28xf32>
    %v3554 = stablehlo.multiply %v3549, %v3544 : tensor<128x128x28x28xf32>
    %v3555 = stablehlo.subtract %v3553, %v3554 : tensor<128x128x28x28xf32>
    %v3556 = stablehlo.multiply %v3547, %v3555 : tensor<128x128x28x28xf32>
    %v3557 = stablehlo.reshape %v3556 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3558 = stablehlo.reshape %v3557 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3559 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3560 = stablehlo.transpose %v3559, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3561 = stablehlo.convolution(%v3558, %v3560)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v3562 = stablehlo.reshape %v3561 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3563 = stablehlo.reshape %v3562 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3564 = stablehlo.reshape %v469 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3565 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v3566 = stablehlo.compare GT, %v3564, %v3565 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v3567 = stablehlo.select %v3566, %v3563, %v3565 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v3568 = stablehlo.reshape %v3567 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3569 = stablehlo.reshape %v435 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3570 = stablehlo.slice %v454 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3571 = stablehlo.slice %v454 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3572 = stablehlo.broadcast_in_dim %v3570, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3573 = stablehlo.broadcast_in_dim %v3571, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3574 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3575 = stablehlo.add %v3573, %v3574 : tensor<128x128x28x28xf32>
    %v3576 = stablehlo.rsqrt %v3575 : tensor<128x128x28x28xf32>
    %v3577 = stablehlo.subtract %v3569, %v3572 : tensor<128x128x28x28xf32>
    %v3578 = stablehlo.multiply %v3577, %v3576 : tensor<128x128x28x28xf32>
    %v3579 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3580 = stablehlo.reshape %v3568 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3581 = stablehlo.multiply %v3579, %v3580 : tensor<128x128x28x28xf32>
    %v3582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3583 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3584 = stablehlo.reduce(%v3581 init: %v3582) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3585 = stablehlo.divide %v3584, %v3583 : tensor<128xf32>
    %v3586 = stablehlo.multiply %v3578, %v3581 : tensor<128x128x28x28xf32>
    %v3587 = stablehlo.reduce(%v3586 init: %v3582) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3588 = stablehlo.divide %v3587, %v3583 : tensor<128xf32>
    %v3589 = stablehlo.concatenate %v3585, %v3588, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3590 = stablehlo.concatenate %v454, %v3589, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b0g1dst = "stablehlo.all_reduce"(%v3590) ({
    ^bb0(%aras2b0g1dst: tensor<f32>, %arbs2b0g1dst: tensor<f32>):
      %aradds2b0g1dst = stablehlo.add %aras2b0g1dst, %arbs2b0g1dst : tensor<f32>
      stablehlo.return %aradds2b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g1dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans2b0g1dst = stablehlo.divide %arsums2b0g1dst, %arns2b0g1dst : tensor<512xf32>
    %v3591 = stablehlo.reshape %v435 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3592 = stablehlo.slice %armeans2b0g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3593 = stablehlo.slice %armeans2b0g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3594 = stablehlo.slice %armeans2b0g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3595 = stablehlo.slice %armeans2b0g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3596 = stablehlo.broadcast_in_dim %v3592, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3597 = stablehlo.broadcast_in_dim %v3593, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3598 = stablehlo.broadcast_in_dim %v3594, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3599 = stablehlo.broadcast_in_dim %v3595, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3600 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3601 = stablehlo.add %v3597, %v3600 : tensor<128x128x28x28xf32>
    %v3602 = stablehlo.rsqrt %v3601 : tensor<128x128x28x28xf32>
    %v3603 = stablehlo.subtract %v3591, %v3596 : tensor<128x128x28x28xf32>
    %v3604 = stablehlo.multiply %v3603, %v3602 : tensor<128x128x28x28xf32>
    %v3605 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3606 = stablehlo.reshape %v3568 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3607 = stablehlo.multiply %v3605, %v3606 : tensor<128x128x28x28xf32>
    %v3608 = stablehlo.subtract %v3607, %v3598 : tensor<128x128x28x28xf32>
    %v3609 = stablehlo.multiply %v3604, %v3599 : tensor<128x128x28x28xf32>
    %v3610 = stablehlo.subtract %v3608, %v3609 : tensor<128x128x28x28xf32>
    %v3611 = stablehlo.multiply %v3602, %v3610 : tensor<128x128x28x28xf32>
    %v3612 = stablehlo.reshape %v3611 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3613 = stablehlo.reshape %v3612 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3614 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3615 = stablehlo.transpose %v3614, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3616 = stablehlo.convolution(%v3613, %v3615)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v3617 = stablehlo.reshape %v3616 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3618 = stablehlo.reshape %v3617 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3619 = stablehlo.reshape %v3513 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3620 = stablehlo.add %v3618, %v3619 : tensor<128x128x28x28xf32>
    %v3621 = stablehlo.reshape %v3620 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3622 = stablehlo.reshape %v430 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3623 = stablehlo.reshape %v3612 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3624 = stablehlo.transpose %v3622, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3625 = stablehlo.transpose %v3623, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3626 = stablehlo.convolution(%v3624, %v3625)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3627 = stablehlo.transpose %v3626, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3628 = stablehlo.reshape %v435 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3629 = stablehlo.slice %v454 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3630 = stablehlo.slice %v454 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3631 = stablehlo.broadcast_in_dim %v3629, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3632 = stablehlo.broadcast_in_dim %v3630, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3633 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3634 = stablehlo.add %v3632, %v3633 : tensor<128x128x28x28xf32>
    %v3635 = stablehlo.rsqrt %v3634 : tensor<128x128x28x28xf32>
    %v3636 = stablehlo.subtract %v3628, %v3631 : tensor<128x128x28x28xf32>
    %v3637 = stablehlo.multiply %v3636, %v3635 : tensor<128x128x28x28xf32>
    %v3638 = stablehlo.reshape %v3568 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3639 = stablehlo.multiply %v3638, %v3637 : tensor<128x128x28x28xf32>
    %v3640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3641 = stablehlo.reduce(%v3639 init: %v3640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3642 = stablehlo.reshape %v3568 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3643 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3644 = stablehlo.reduce(%v3642 init: %v3643) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3645 = stablehlo.reshape %v471 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3646 = stablehlo.reshape %v3557 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3647 = stablehlo.transpose %v3645, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3648 = stablehlo.transpose %v3646, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3649 = stablehlo.convolution(%v3647, %v3648)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3650 = stablehlo.transpose %v3649, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3651 = stablehlo.reshape %v476 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3652 = stablehlo.slice %v495 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3653 = stablehlo.slice %v495 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3654 = stablehlo.broadcast_in_dim %v3652, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3655 = stablehlo.broadcast_in_dim %v3653, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3656 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3657 = stablehlo.add %v3655, %v3656 : tensor<128x128x28x28xf32>
    %v3658 = stablehlo.rsqrt %v3657 : tensor<128x128x28x28xf32>
    %v3659 = stablehlo.subtract %v3651, %v3654 : tensor<128x128x28x28xf32>
    %v3660 = stablehlo.multiply %v3659, %v3658 : tensor<128x128x28x28xf32>
    %v3661 = stablehlo.reshape %v3513 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3662 = stablehlo.multiply %v3661, %v3660 : tensor<128x128x28x28xf32>
    %v3663 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3664 = stablehlo.reduce(%v3662 init: %v3663) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3665 = stablehlo.reshape %v3513 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3666 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3667 = stablehlo.reduce(%v3665 init: %v3666) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3668 = stablehlo.reshape %v3621 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3669 = stablehlo.reshape %v428 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3670 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v3671 = stablehlo.compare GT, %v3669, %v3670 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v3672 = stablehlo.select %v3671, %v3668, %v3670 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v3673 = stablehlo.reshape %v3672 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3674 = stablehlo.reshape %v354 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3675 = stablehlo.slice %v373 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3676 = stablehlo.slice %v373 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3677 = stablehlo.broadcast_in_dim %v3675, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3678 = stablehlo.broadcast_in_dim %v3676, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3679 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3680 = stablehlo.add %v3678, %v3679 : tensor<128x128x28x28xf32>
    %v3681 = stablehlo.rsqrt %v3680 : tensor<128x128x28x28xf32>
    %v3682 = stablehlo.subtract %v3674, %v3677 : tensor<128x128x28x28xf32>
    %v3683 = stablehlo.multiply %v3682, %v3681 : tensor<128x128x28x28xf32>
    %v3684 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3685 = stablehlo.reshape %v3673 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3686 = stablehlo.multiply %v3684, %v3685 : tensor<128x128x28x28xf32>
    %v3687 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3688 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3689 = stablehlo.reduce(%v3686 init: %v3687) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3690 = stablehlo.divide %v3689, %v3688 : tensor<128xf32>
    %v3691 = stablehlo.multiply %v3683, %v3686 : tensor<128x128x28x28xf32>
    %v3692 = stablehlo.reduce(%v3691 init: %v3687) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3693 = stablehlo.divide %v3692, %v3688 : tensor<128xf32>
    %v3694 = stablehlo.concatenate %v3690, %v3693, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3695 = stablehlo.concatenate %v373, %v3694, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsumd2g2dst = "stablehlo.all_reduce"(%v3695) ({
    ^bb0(%arad2g2dst: tensor<f32>, %arbd2g2dst: tensor<f32>):
      %araddd2g2dst = stablehlo.add %arad2g2dst, %arbd2g2dst : tensor<f32>
      stablehlo.return %araddd2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd2g2dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand2g2dst = stablehlo.divide %arsumd2g2dst, %arnd2g2dst : tensor<512xf32>
    %v3696 = stablehlo.reshape %v354 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3697 = stablehlo.slice %armeand2g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3698 = stablehlo.slice %armeand2g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3699 = stablehlo.slice %armeand2g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3700 = stablehlo.slice %armeand2g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3701 = stablehlo.broadcast_in_dim %v3697, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3702 = stablehlo.broadcast_in_dim %v3698, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3703 = stablehlo.broadcast_in_dim %v3699, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3704 = stablehlo.broadcast_in_dim %v3700, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3705 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3706 = stablehlo.add %v3702, %v3705 : tensor<128x128x28x28xf32>
    %v3707 = stablehlo.rsqrt %v3706 : tensor<128x128x28x28xf32>
    %v3708 = stablehlo.subtract %v3696, %v3701 : tensor<128x128x28x28xf32>
    %v3709 = stablehlo.multiply %v3708, %v3707 : tensor<128x128x28x28xf32>
    %v3710 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3711 = stablehlo.reshape %v3673 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3712 = stablehlo.multiply %v3710, %v3711 : tensor<128x128x28x28xf32>
    %v3713 = stablehlo.subtract %v3712, %v3703 : tensor<128x128x28x28xf32>
    %v3714 = stablehlo.multiply %v3709, %v3704 : tensor<128x128x28x28xf32>
    %v3715 = stablehlo.subtract %v3713, %v3714 : tensor<128x128x28x28xf32>
    %v3716 = stablehlo.multiply %v3707, %v3715 : tensor<128x128x28x28xf32>
    %v3717 = stablehlo.reshape %v3716 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3718 = stablehlo.reshape %v3717 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3719 = stablehlo.reverse %d2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3720 = stablehlo.transpose %v3719, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3721 = stablehlo.convolution(%v3718, %v3720)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<128x128x28x28xf32>
    %v3722 = stablehlo.reshape %v3721 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3723 = stablehlo.reshape %v3722 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3724 = stablehlo.reshape %v347 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3725 = stablehlo.constant dense<0.0> : tensor<128x128x28x28xf32>
    %v3726 = stablehlo.compare GT, %v3724, %v3725 : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xi1>
    %v3727 = stablehlo.select %v3726, %v3723, %v3725 : tensor<128x128x28x28xi1>, tensor<128x128x28x28xf32>
    %v3728 = stablehlo.reshape %v3727 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3729 = stablehlo.reshape %v313 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3730 = stablehlo.slice %v332 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3731 = stablehlo.slice %v332 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3732 = stablehlo.broadcast_in_dim %v3730, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3733 = stablehlo.broadcast_in_dim %v3731, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3734 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3735 = stablehlo.add %v3733, %v3734 : tensor<128x128x28x28xf32>
    %v3736 = stablehlo.rsqrt %v3735 : tensor<128x128x28x28xf32>
    %v3737 = stablehlo.subtract %v3729, %v3732 : tensor<128x128x28x28xf32>
    %v3738 = stablehlo.multiply %v3737, %v3736 : tensor<128x128x28x28xf32>
    %v3739 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3740 = stablehlo.reshape %v3728 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3741 = stablehlo.multiply %v3739, %v3740 : tensor<128x128x28x28xf32>
    %v3742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3743 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3744 = stablehlo.reduce(%v3741 init: %v3742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3745 = stablehlo.divide %v3744, %v3743 : tensor<128xf32>
    %v3746 = stablehlo.multiply %v3738, %v3741 : tensor<128x128x28x28xf32>
    %v3747 = stablehlo.reduce(%v3746 init: %v3742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3748 = stablehlo.divide %v3747, %v3743 : tensor<128xf32>
    %v3749 = stablehlo.concatenate %v3745, %v3748, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3750 = stablehlo.concatenate %v332, %v3749, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsumd2g1dst = "stablehlo.all_reduce"(%v3750) ({
    ^bb0(%arad2g1dst: tensor<f32>, %arbd2g1dst: tensor<f32>):
      %araddd2g1dst = stablehlo.add %arad2g1dst, %arbd2g1dst : tensor<f32>
      stablehlo.return %araddd2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd2g1dst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand2g1dst = stablehlo.divide %arsumd2g1dst, %arnd2g1dst : tensor<512xf32>
    %v3751 = stablehlo.reshape %v313 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3752 = stablehlo.slice %armeand2g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3753 = stablehlo.slice %armeand2g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3754 = stablehlo.slice %armeand2g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3755 = stablehlo.slice %armeand2g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3756 = stablehlo.broadcast_in_dim %v3752, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3757 = stablehlo.broadcast_in_dim %v3753, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3758 = stablehlo.broadcast_in_dim %v3754, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3759 = stablehlo.broadcast_in_dim %v3755, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3760 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3761 = stablehlo.add %v3757, %v3760 : tensor<128x128x28x28xf32>
    %v3762 = stablehlo.rsqrt %v3761 : tensor<128x128x28x28xf32>
    %v3763 = stablehlo.subtract %v3751, %v3756 : tensor<128x128x28x28xf32>
    %v3764 = stablehlo.multiply %v3763, %v3762 : tensor<128x128x28x28xf32>
    %v3765 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3766 = stablehlo.reshape %v3728 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3767 = stablehlo.multiply %v3765, %v3766 : tensor<128x128x28x28xf32>
    %v3768 = stablehlo.subtract %v3767, %v3758 : tensor<128x128x28x28xf32>
    %v3769 = stablehlo.multiply %v3764, %v3759 : tensor<128x128x28x28xf32>
    %v3770 = stablehlo.subtract %v3768, %v3769 : tensor<128x128x28x28xf32>
    %v3771 = stablehlo.multiply %v3762, %v3770 : tensor<128x128x28x28xf32>
    %v3772 = stablehlo.reshape %v3771 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3773 = stablehlo.reshape %v3772 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3775 = stablehlo.pad %v3773, %v3774, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128x128x56x56xf32>
    %v3776 = stablehlo.reverse %d2W1, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v3777 = stablehlo.transpose %v3776, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v3778 = stablehlo.convolution(%v3775, %v3777)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v3779 = stablehlo.reshape %v3778 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3780 = stablehlo.reshape %v393 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3781 = stablehlo.slice %v412 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3782 = stablehlo.slice %v412 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3783 = stablehlo.broadcast_in_dim %v3781, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3784 = stablehlo.broadcast_in_dim %v3782, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3785 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3786 = stablehlo.add %v3784, %v3785 : tensor<128x128x28x28xf32>
    %v3787 = stablehlo.rsqrt %v3786 : tensor<128x128x28x28xf32>
    %v3788 = stablehlo.subtract %v3780, %v3783 : tensor<128x128x28x28xf32>
    %v3789 = stablehlo.multiply %v3788, %v3787 : tensor<128x128x28x28xf32>
    %v3790 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3791 = stablehlo.reshape %v3673 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3792 = stablehlo.multiply %v3790, %v3791 : tensor<128x128x28x28xf32>
    %v3793 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3794 = stablehlo.constant dense<100352.0> : tensor<128xf32>
    %v3795 = stablehlo.reduce(%v3792 init: %v3793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3796 = stablehlo.divide %v3795, %v3794 : tensor<128xf32>
    %v3797 = stablehlo.multiply %v3789, %v3792 : tensor<128x128x28x28xf32>
    %v3798 = stablehlo.reduce(%v3797 init: %v3793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3799 = stablehlo.divide %v3798, %v3794 : tensor<128xf32>
    %v3800 = stablehlo.concatenate %v3796, %v3799, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3801 = stablehlo.concatenate %v412, %v3800, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsumd2gpdst = "stablehlo.all_reduce"(%v3801) ({
    ^bb0(%arad2gpdst: tensor<f32>, %arbd2gpdst: tensor<f32>):
      %araddd2gpdst = stablehlo.add %arad2gpdst, %arbd2gpdst : tensor<f32>
      stablehlo.return %araddd2gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd2gpdst = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand2gpdst = stablehlo.divide %arsumd2gpdst, %arnd2gpdst : tensor<512xf32>
    %v3802 = stablehlo.reshape %v393 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3803 = stablehlo.slice %armeand2gpdst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3804 = stablehlo.slice %armeand2gpdst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3805 = stablehlo.slice %armeand2gpdst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3806 = stablehlo.slice %armeand2gpdst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3807 = stablehlo.broadcast_in_dim %v3803, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3808 = stablehlo.broadcast_in_dim %v3804, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3809 = stablehlo.broadcast_in_dim %v3805, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3810 = stablehlo.broadcast_in_dim %v3806, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3811 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3812 = stablehlo.add %v3808, %v3811 : tensor<128x128x28x28xf32>
    %v3813 = stablehlo.rsqrt %v3812 : tensor<128x128x28x28xf32>
    %v3814 = stablehlo.subtract %v3802, %v3807 : tensor<128x128x28x28xf32>
    %v3815 = stablehlo.multiply %v3814, %v3813 : tensor<128x128x28x28xf32>
    %v3816 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3817 = stablehlo.reshape %v3673 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3818 = stablehlo.multiply %v3816, %v3817 : tensor<128x128x28x28xf32>
    %v3819 = stablehlo.subtract %v3818, %v3809 : tensor<128x128x28x28xf32>
    %v3820 = stablehlo.multiply %v3815, %v3810 : tensor<128x128x28x28xf32>
    %v3821 = stablehlo.subtract %v3819, %v3820 : tensor<128x128x28x28xf32>
    %v3822 = stablehlo.multiply %v3813, %v3821 : tensor<128x128x28x28xf32>
    %v3823 = stablehlo.reshape %v3822 : (tensor<128x128x28x28xf32>) -> tensor<128x100352xf32>
    %v3824 = stablehlo.reshape %v3823 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3825 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3826 = stablehlo.pad %v3824, %v3825, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128x128x56x56xf32>
    %v3827 = stablehlo.reverse %d2Wp, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v3828 = stablehlo.transpose %v3827, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v3829 = stablehlo.convolution(%v3826, %v3828)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x56x56xf32>, tensor<64x128x1x1xf32>) -> tensor<128x64x56x56xf32>
    %v3830 = stablehlo.reshape %v3829 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3831 = stablehlo.reshape %v3779 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3832 = stablehlo.reshape %v3830 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3833 = stablehlo.add %v3831, %v3832 : tensor<128x64x56x56xf32>
    %v3834 = stablehlo.reshape %v3833 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3835 = stablehlo.reshape %v308 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3836 = stablehlo.reshape %v3772 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3838 = stablehlo.pad %v3836, %v3837, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128x128x56x56xf32>
    %v3839 = stablehlo.transpose %v3835, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3840 = stablehlo.transpose %v3838, dims = [1, 0, 2, 3] : (tensor<128x128x56x56xf32>) -> tensor<128x128x56x56xf32>
    %v3841 = stablehlo.convolution(%v3839, %v3840)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<128x128x56x56xf32>) -> tensor<64x128x3x3xf32>
    %v3842 = stablehlo.transpose %v3841, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v3843 = stablehlo.reshape %v313 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3844 = stablehlo.slice %v332 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3845 = stablehlo.slice %v332 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3846 = stablehlo.broadcast_in_dim %v3844, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3847 = stablehlo.broadcast_in_dim %v3845, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3848 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3849 = stablehlo.add %v3847, %v3848 : tensor<128x128x28x28xf32>
    %v3850 = stablehlo.rsqrt %v3849 : tensor<128x128x28x28xf32>
    %v3851 = stablehlo.subtract %v3843, %v3846 : tensor<128x128x28x28xf32>
    %v3852 = stablehlo.multiply %v3851, %v3850 : tensor<128x128x28x28xf32>
    %v3853 = stablehlo.reshape %v3728 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3854 = stablehlo.multiply %v3853, %v3852 : tensor<128x128x28x28xf32>
    %v3855 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3856 = stablehlo.reduce(%v3854 init: %v3855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3857 = stablehlo.reshape %v3728 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3858 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3859 = stablehlo.reduce(%v3857 init: %v3858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3860 = stablehlo.reshape %v349 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3861 = stablehlo.reshape %v3717 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3862 = stablehlo.transpose %v3860, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3863 = stablehlo.transpose %v3861, dims = [1, 0, 2, 3] : (tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32>
    %v3864 = stablehlo.convolution(%v3862, %v3863)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x28x28xf32>, tensor<128x128x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3865 = stablehlo.transpose %v3864, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3866 = stablehlo.reshape %v354 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3867 = stablehlo.slice %v373 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3868 = stablehlo.slice %v373 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3869 = stablehlo.broadcast_in_dim %v3867, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3870 = stablehlo.broadcast_in_dim %v3868, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3871 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3872 = stablehlo.add %v3870, %v3871 : tensor<128x128x28x28xf32>
    %v3873 = stablehlo.rsqrt %v3872 : tensor<128x128x28x28xf32>
    %v3874 = stablehlo.subtract %v3866, %v3869 : tensor<128x128x28x28xf32>
    %v3875 = stablehlo.multiply %v3874, %v3873 : tensor<128x128x28x28xf32>
    %v3876 = stablehlo.reshape %v3673 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3877 = stablehlo.multiply %v3876, %v3875 : tensor<128x128x28x28xf32>
    %v3878 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3879 = stablehlo.reduce(%v3877 init: %v3878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3880 = stablehlo.reshape %v3673 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3882 = stablehlo.reduce(%v3880 init: %v3881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3883 = stablehlo.reshape %v308 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3884 = stablehlo.reshape %v3823 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3886 = stablehlo.pad %v3884, %v3885, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128x128x56x56xf32>
    %v3887 = stablehlo.transpose %v3883, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v3888 = stablehlo.transpose %v3886, dims = [1, 0, 2, 3] : (tensor<128x128x56x56xf32>) -> tensor<128x128x56x56xf32>
    %v3889 = stablehlo.convolution(%v3887, %v3888)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<128x128x56x56xf32>) -> tensor<64x128x1x1xf32>
    %v3890 = stablehlo.transpose %v3889, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v3891 = stablehlo.reshape %v393 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3892 = stablehlo.slice %v412 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3893 = stablehlo.slice %v412 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3894 = stablehlo.broadcast_in_dim %v3892, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3895 = stablehlo.broadcast_in_dim %v3893, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %v3896 = stablehlo.constant dense<1.0e-05> : tensor<128x128x28x28xf32>
    %v3897 = stablehlo.add %v3895, %v3896 : tensor<128x128x28x28xf32>
    %v3898 = stablehlo.rsqrt %v3897 : tensor<128x128x28x28xf32>
    %v3899 = stablehlo.subtract %v3891, %v3894 : tensor<128x128x28x28xf32>
    %v3900 = stablehlo.multiply %v3899, %v3898 : tensor<128x128x28x28xf32>
    %v3901 = stablehlo.reshape %v3673 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3902 = stablehlo.multiply %v3901, %v3900 : tensor<128x128x28x28xf32>
    %v3903 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3904 = stablehlo.reduce(%v3902 init: %v3903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3905 = stablehlo.reshape %v3673 : (tensor<128x100352xf32>) -> tensor<128x128x28x28xf32>
    %v3906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3907 = stablehlo.reduce(%v3905 init: %v3906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3908 = stablehlo.reshape %v3834 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3909 = stablehlo.reshape %v304 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3910 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v3911 = stablehlo.compare GT, %v3909, %v3910 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3912 = stablehlo.select %v3911, %v3908, %v3910 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3913 = stablehlo.reshape %v3912 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3914 = stablehlo.reshape %v266 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3915 = stablehlo.slice %v285 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v3916 = stablehlo.slice %v285 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v3917 = stablehlo.broadcast_in_dim %v3915, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3918 = stablehlo.broadcast_in_dim %v3916, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3919 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v3920 = stablehlo.add %v3918, %v3919 : tensor<128x64x56x56xf32>
    %v3921 = stablehlo.rsqrt %v3920 : tensor<128x64x56x56xf32>
    %v3922 = stablehlo.subtract %v3914, %v3917 : tensor<128x64x56x56xf32>
    %v3923 = stablehlo.multiply %v3922, %v3921 : tensor<128x64x56x56xf32>
    %v3924 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3925 = stablehlo.reshape %v3913 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3926 = stablehlo.multiply %v3924, %v3925 : tensor<128x64x56x56xf32>
    %v3927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3928 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3929 = stablehlo.reduce(%v3926 init: %v3927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3930 = stablehlo.divide %v3929, %v3928 : tensor<64xf32>
    %v3931 = stablehlo.multiply %v3923, %v3926 : tensor<128x64x56x56xf32>
    %v3932 = stablehlo.reduce(%v3931 init: %v3927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3933 = stablehlo.divide %v3932, %v3928 : tensor<64xf32>
    %v3934 = stablehlo.concatenate %v3930, %v3933, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v3935 = stablehlo.concatenate %v285, %v3934, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b2g2dst = "stablehlo.all_reduce"(%v3935) ({
    ^bb0(%aras1b2g2dst: tensor<f32>, %arbs1b2g2dst: tensor<f32>):
      %aradds1b2g2dst = stablehlo.add %aras1b2g2dst, %arbs1b2g2dst : tensor<f32>
      stablehlo.return %aradds1b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g2dst = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans1b2g2dst = stablehlo.divide %arsums1b2g2dst, %arns1b2g2dst : tensor<256xf32>
    %v3936 = stablehlo.reshape %v266 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3937 = stablehlo.slice %armeans1b2g2dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v3938 = stablehlo.slice %armeans1b2g2dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v3939 = stablehlo.slice %armeans1b2g2dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v3940 = stablehlo.slice %armeans1b2g2dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v3941 = stablehlo.broadcast_in_dim %v3937, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3942 = stablehlo.broadcast_in_dim %v3938, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3943 = stablehlo.broadcast_in_dim %v3939, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3944 = stablehlo.broadcast_in_dim %v3940, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3945 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v3946 = stablehlo.add %v3942, %v3945 : tensor<128x64x56x56xf32>
    %v3947 = stablehlo.rsqrt %v3946 : tensor<128x64x56x56xf32>
    %v3948 = stablehlo.subtract %v3936, %v3941 : tensor<128x64x56x56xf32>
    %v3949 = stablehlo.multiply %v3948, %v3947 : tensor<128x64x56x56xf32>
    %v3950 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3951 = stablehlo.reshape %v3913 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3952 = stablehlo.multiply %v3950, %v3951 : tensor<128x64x56x56xf32>
    %v3953 = stablehlo.subtract %v3952, %v3943 : tensor<128x64x56x56xf32>
    %v3954 = stablehlo.multiply %v3949, %v3944 : tensor<128x64x56x56xf32>
    %v3955 = stablehlo.subtract %v3953, %v3954 : tensor<128x64x56x56xf32>
    %v3956 = stablehlo.multiply %v3947, %v3955 : tensor<128x64x56x56xf32>
    %v3957 = stablehlo.reshape %v3956 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3958 = stablehlo.reshape %v3957 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3959 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3960 = stablehlo.transpose %v3959, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3961 = stablehlo.convolution(%v3958, %v3960)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v3962 = stablehlo.reshape %v3961 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3963 = stablehlo.reshape %v3962 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3964 = stablehlo.reshape %v259 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3965 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v3966 = stablehlo.compare GT, %v3964, %v3965 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v3967 = stablehlo.select %v3966, %v3963, %v3965 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v3968 = stablehlo.reshape %v3967 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v3969 = stablehlo.reshape %v225 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3970 = stablehlo.slice %v244 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v3971 = stablehlo.slice %v244 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v3972 = stablehlo.broadcast_in_dim %v3970, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3973 = stablehlo.broadcast_in_dim %v3971, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3974 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v3975 = stablehlo.add %v3973, %v3974 : tensor<128x64x56x56xf32>
    %v3976 = stablehlo.rsqrt %v3975 : tensor<128x64x56x56xf32>
    %v3977 = stablehlo.subtract %v3969, %v3972 : tensor<128x64x56x56xf32>
    %v3978 = stablehlo.multiply %v3977, %v3976 : tensor<128x64x56x56xf32>
    %v3979 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3980 = stablehlo.reshape %v3968 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3981 = stablehlo.multiply %v3979, %v3980 : tensor<128x64x56x56xf32>
    %v3982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3983 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3984 = stablehlo.reduce(%v3981 init: %v3982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3985 = stablehlo.divide %v3984, %v3983 : tensor<64xf32>
    %v3986 = stablehlo.multiply %v3978, %v3981 : tensor<128x64x56x56xf32>
    %v3987 = stablehlo.reduce(%v3986 init: %v3982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3988 = stablehlo.divide %v3987, %v3983 : tensor<64xf32>
    %v3989 = stablehlo.concatenate %v3985, %v3988, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v3990 = stablehlo.concatenate %v244, %v3989, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b2g1dst = "stablehlo.all_reduce"(%v3990) ({
    ^bb0(%aras1b2g1dst: tensor<f32>, %arbs1b2g1dst: tensor<f32>):
      %aradds1b2g1dst = stablehlo.add %aras1b2g1dst, %arbs1b2g1dst : tensor<f32>
      stablehlo.return %aradds1b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g1dst = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans1b2g1dst = stablehlo.divide %arsums1b2g1dst, %arns1b2g1dst : tensor<256xf32>
    %v3991 = stablehlo.reshape %v225 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v3992 = stablehlo.slice %armeans1b2g1dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v3993 = stablehlo.slice %armeans1b2g1dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v3994 = stablehlo.slice %armeans1b2g1dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v3995 = stablehlo.slice %armeans1b2g1dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v3996 = stablehlo.broadcast_in_dim %v3992, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3997 = stablehlo.broadcast_in_dim %v3993, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3998 = stablehlo.broadcast_in_dim %v3994, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v3999 = stablehlo.broadcast_in_dim %v3995, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4000 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v4001 = stablehlo.add %v3997, %v4000 : tensor<128x64x56x56xf32>
    %v4002 = stablehlo.rsqrt %v4001 : tensor<128x64x56x56xf32>
    %v4003 = stablehlo.subtract %v3991, %v3996 : tensor<128x64x56x56xf32>
    %v4004 = stablehlo.multiply %v4003, %v4002 : tensor<128x64x56x56xf32>
    %v4005 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4006 = stablehlo.reshape %v3968 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4007 = stablehlo.multiply %v4005, %v4006 : tensor<128x64x56x56xf32>
    %v4008 = stablehlo.subtract %v4007, %v3998 : tensor<128x64x56x56xf32>
    %v4009 = stablehlo.multiply %v4004, %v3999 : tensor<128x64x56x56xf32>
    %v4010 = stablehlo.subtract %v4008, %v4009 : tensor<128x64x56x56xf32>
    %v4011 = stablehlo.multiply %v4002, %v4010 : tensor<128x64x56x56xf32>
    %v4012 = stablehlo.reshape %v4011 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4013 = stablehlo.reshape %v4012 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4014 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4015 = stablehlo.transpose %v4014, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4016 = stablehlo.convolution(%v4013, %v4015)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v4017 = stablehlo.reshape %v4016 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4018 = stablehlo.reshape %v4017 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4019 = stablehlo.reshape %v3913 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4020 = stablehlo.add %v4018, %v4019 : tensor<128x64x56x56xf32>
    %v4021 = stablehlo.reshape %v4020 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4022 = stablehlo.reshape %v220 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4023 = stablehlo.reshape %v4012 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4024 = stablehlo.transpose %v4022, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v4025 = stablehlo.transpose %v4023, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v4026 = stablehlo.convolution(%v4024, %v4025)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4027 = stablehlo.transpose %v4026, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4028 = stablehlo.reshape %v225 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4029 = stablehlo.slice %v244 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4030 = stablehlo.slice %v244 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4031 = stablehlo.broadcast_in_dim %v4029, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4032 = stablehlo.broadcast_in_dim %v4030, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4033 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v4034 = stablehlo.add %v4032, %v4033 : tensor<128x64x56x56xf32>
    %v4035 = stablehlo.rsqrt %v4034 : tensor<128x64x56x56xf32>
    %v4036 = stablehlo.subtract %v4028, %v4031 : tensor<128x64x56x56xf32>
    %v4037 = stablehlo.multiply %v4036, %v4035 : tensor<128x64x56x56xf32>
    %v4038 = stablehlo.reshape %v3968 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4039 = stablehlo.multiply %v4038, %v4037 : tensor<128x64x56x56xf32>
    %v4040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4041 = stablehlo.reduce(%v4039 init: %v4040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4042 = stablehlo.reshape %v3968 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4044 = stablehlo.reduce(%v4042 init: %v4043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4045 = stablehlo.reshape %v261 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4046 = stablehlo.reshape %v3957 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4047 = stablehlo.transpose %v4045, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v4048 = stablehlo.transpose %v4046, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v4049 = stablehlo.convolution(%v4047, %v4048)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4050 = stablehlo.transpose %v4049, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4051 = stablehlo.reshape %v266 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4052 = stablehlo.slice %v285 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4053 = stablehlo.slice %v285 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4054 = stablehlo.broadcast_in_dim %v4052, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4055 = stablehlo.broadcast_in_dim %v4053, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4056 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v4057 = stablehlo.add %v4055, %v4056 : tensor<128x64x56x56xf32>
    %v4058 = stablehlo.rsqrt %v4057 : tensor<128x64x56x56xf32>
    %v4059 = stablehlo.subtract %v4051, %v4054 : tensor<128x64x56x56xf32>
    %v4060 = stablehlo.multiply %v4059, %v4058 : tensor<128x64x56x56xf32>
    %v4061 = stablehlo.reshape %v3913 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4062 = stablehlo.multiply %v4061, %v4060 : tensor<128x64x56x56xf32>
    %v4063 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4064 = stablehlo.reduce(%v4062 init: %v4063) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4065 = stablehlo.reshape %v3913 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4066 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4067 = stablehlo.reduce(%v4065 init: %v4066) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4068 = stablehlo.reshape %v4021 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4069 = stablehlo.reshape %v216 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4070 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v4071 = stablehlo.compare GT, %v4069, %v4070 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v4072 = stablehlo.select %v4071, %v4068, %v4070 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v4073 = stablehlo.reshape %v4072 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4074 = stablehlo.reshape %v178 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4075 = stablehlo.slice %v197 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4076 = stablehlo.slice %v197 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4077 = stablehlo.broadcast_in_dim %v4075, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4078 = stablehlo.broadcast_in_dim %v4076, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4079 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v4080 = stablehlo.add %v4078, %v4079 : tensor<128x64x56x56xf32>
    %v4081 = stablehlo.rsqrt %v4080 : tensor<128x64x56x56xf32>
    %v4082 = stablehlo.subtract %v4074, %v4077 : tensor<128x64x56x56xf32>
    %v4083 = stablehlo.multiply %v4082, %v4081 : tensor<128x64x56x56xf32>
    %v4084 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4085 = stablehlo.reshape %v4073 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4086 = stablehlo.multiply %v4084, %v4085 : tensor<128x64x56x56xf32>
    %v4087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4088 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v4089 = stablehlo.reduce(%v4086 init: %v4087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4090 = stablehlo.divide %v4089, %v4088 : tensor<64xf32>
    %v4091 = stablehlo.multiply %v4083, %v4086 : tensor<128x64x56x56xf32>
    %v4092 = stablehlo.reduce(%v4091 init: %v4087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4093 = stablehlo.divide %v4092, %v4088 : tensor<64xf32>
    %v4094 = stablehlo.concatenate %v4090, %v4093, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4095 = stablehlo.concatenate %v197, %v4094, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b1g2dst = "stablehlo.all_reduce"(%v4095) ({
    ^bb0(%aras1b1g2dst: tensor<f32>, %arbs1b1g2dst: tensor<f32>):
      %aradds1b1g2dst = stablehlo.add %aras1b1g2dst, %arbs1b1g2dst : tensor<f32>
      stablehlo.return %aradds1b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g2dst = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans1b1g2dst = stablehlo.divide %arsums1b1g2dst, %arns1b1g2dst : tensor<256xf32>
    %v4096 = stablehlo.reshape %v178 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4097 = stablehlo.slice %armeans1b1g2dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4098 = stablehlo.slice %armeans1b1g2dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4099 = stablehlo.slice %armeans1b1g2dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4100 = stablehlo.slice %armeans1b1g2dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4101 = stablehlo.broadcast_in_dim %v4097, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4102 = stablehlo.broadcast_in_dim %v4098, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4103 = stablehlo.broadcast_in_dim %v4099, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4104 = stablehlo.broadcast_in_dim %v4100, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4105 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v4106 = stablehlo.add %v4102, %v4105 : tensor<128x64x56x56xf32>
    %v4107 = stablehlo.rsqrt %v4106 : tensor<128x64x56x56xf32>
    %v4108 = stablehlo.subtract %v4096, %v4101 : tensor<128x64x56x56xf32>
    %v4109 = stablehlo.multiply %v4108, %v4107 : tensor<128x64x56x56xf32>
    %v4110 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4111 = stablehlo.reshape %v4073 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4112 = stablehlo.multiply %v4110, %v4111 : tensor<128x64x56x56xf32>
    %v4113 = stablehlo.subtract %v4112, %v4103 : tensor<128x64x56x56xf32>
    %v4114 = stablehlo.multiply %v4109, %v4104 : tensor<128x64x56x56xf32>
    %v4115 = stablehlo.subtract %v4113, %v4114 : tensor<128x64x56x56xf32>
    %v4116 = stablehlo.multiply %v4107, %v4115 : tensor<128x64x56x56xf32>
    %v4117 = stablehlo.reshape %v4116 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4118 = stablehlo.reshape %v4117 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4119 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4120 = stablehlo.transpose %v4119, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4121 = stablehlo.convolution(%v4118, %v4120)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v4122 = stablehlo.reshape %v4121 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4123 = stablehlo.reshape %v4122 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4124 = stablehlo.reshape %v171 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4125 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v4126 = stablehlo.compare GT, %v4124, %v4125 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v4127 = stablehlo.select %v4126, %v4123, %v4125 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v4128 = stablehlo.reshape %v4127 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4129 = stablehlo.reshape %v137 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4130 = stablehlo.slice %v156 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4131 = stablehlo.slice %v156 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4132 = stablehlo.broadcast_in_dim %v4130, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4133 = stablehlo.broadcast_in_dim %v4131, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4134 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v4135 = stablehlo.add %v4133, %v4134 : tensor<128x64x56x56xf32>
    %v4136 = stablehlo.rsqrt %v4135 : tensor<128x64x56x56xf32>
    %v4137 = stablehlo.subtract %v4129, %v4132 : tensor<128x64x56x56xf32>
    %v4138 = stablehlo.multiply %v4137, %v4136 : tensor<128x64x56x56xf32>
    %v4139 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4140 = stablehlo.reshape %v4128 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4141 = stablehlo.multiply %v4139, %v4140 : tensor<128x64x56x56xf32>
    %v4142 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4143 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v4144 = stablehlo.reduce(%v4141 init: %v4142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4145 = stablehlo.divide %v4144, %v4143 : tensor<64xf32>
    %v4146 = stablehlo.multiply %v4138, %v4141 : tensor<128x64x56x56xf32>
    %v4147 = stablehlo.reduce(%v4146 init: %v4142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4148 = stablehlo.divide %v4147, %v4143 : tensor<64xf32>
    %v4149 = stablehlo.concatenate %v4145, %v4148, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4150 = stablehlo.concatenate %v156, %v4149, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b1g1dst = "stablehlo.all_reduce"(%v4150) ({
    ^bb0(%aras1b1g1dst: tensor<f32>, %arbs1b1g1dst: tensor<f32>):
      %aradds1b1g1dst = stablehlo.add %aras1b1g1dst, %arbs1b1g1dst : tensor<f32>
      stablehlo.return %aradds1b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g1dst = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans1b1g1dst = stablehlo.divide %arsums1b1g1dst, %arns1b1g1dst : tensor<256xf32>
    %v4151 = stablehlo.reshape %v137 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4152 = stablehlo.slice %armeans1b1g1dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4153 = stablehlo.slice %armeans1b1g1dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4154 = stablehlo.slice %armeans1b1g1dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4155 = stablehlo.slice %armeans1b1g1dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4156 = stablehlo.broadcast_in_dim %v4152, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4157 = stablehlo.broadcast_in_dim %v4153, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4158 = stablehlo.broadcast_in_dim %v4154, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4159 = stablehlo.broadcast_in_dim %v4155, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4160 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v4161 = stablehlo.add %v4157, %v4160 : tensor<128x64x56x56xf32>
    %v4162 = stablehlo.rsqrt %v4161 : tensor<128x64x56x56xf32>
    %v4163 = stablehlo.subtract %v4151, %v4156 : tensor<128x64x56x56xf32>
    %v4164 = stablehlo.multiply %v4163, %v4162 : tensor<128x64x56x56xf32>
    %v4165 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4166 = stablehlo.reshape %v4128 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4167 = stablehlo.multiply %v4165, %v4166 : tensor<128x64x56x56xf32>
    %v4168 = stablehlo.subtract %v4167, %v4158 : tensor<128x64x56x56xf32>
    %v4169 = stablehlo.multiply %v4164, %v4159 : tensor<128x64x56x56xf32>
    %v4170 = stablehlo.subtract %v4168, %v4169 : tensor<128x64x56x56xf32>
    %v4171 = stablehlo.multiply %v4162, %v4170 : tensor<128x64x56x56xf32>
    %v4172 = stablehlo.reshape %v4171 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4173 = stablehlo.reshape %v4172 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4174 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4175 = stablehlo.transpose %v4174, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4176 = stablehlo.convolution(%v4173, %v4175)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v4177 = stablehlo.reshape %v4176 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4178 = stablehlo.reshape %v4177 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4179 = stablehlo.reshape %v4073 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4180 = stablehlo.add %v4178, %v4179 : tensor<128x64x56x56xf32>
    %v4181 = stablehlo.reshape %v4180 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4182 = stablehlo.reshape %v132 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4183 = stablehlo.reshape %v4172 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4184 = stablehlo.transpose %v4182, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v4185 = stablehlo.transpose %v4183, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v4186 = stablehlo.convolution(%v4184, %v4185)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4187 = stablehlo.transpose %v4186, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4188 = stablehlo.reshape %v137 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4189 = stablehlo.slice %v156 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4190 = stablehlo.slice %v156 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4191 = stablehlo.broadcast_in_dim %v4189, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4192 = stablehlo.broadcast_in_dim %v4190, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4193 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v4194 = stablehlo.add %v4192, %v4193 : tensor<128x64x56x56xf32>
    %v4195 = stablehlo.rsqrt %v4194 : tensor<128x64x56x56xf32>
    %v4196 = stablehlo.subtract %v4188, %v4191 : tensor<128x64x56x56xf32>
    %v4197 = stablehlo.multiply %v4196, %v4195 : tensor<128x64x56x56xf32>
    %v4198 = stablehlo.reshape %v4128 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4199 = stablehlo.multiply %v4198, %v4197 : tensor<128x64x56x56xf32>
    %v4200 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4201 = stablehlo.reduce(%v4199 init: %v4200) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4202 = stablehlo.reshape %v4128 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4204 = stablehlo.reduce(%v4202 init: %v4203) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4205 = stablehlo.reshape %v173 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4206 = stablehlo.reshape %v4117 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4207 = stablehlo.transpose %v4205, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v4208 = stablehlo.transpose %v4206, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v4209 = stablehlo.convolution(%v4207, %v4208)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4210 = stablehlo.transpose %v4209, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4211 = stablehlo.reshape %v178 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4212 = stablehlo.slice %v197 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4213 = stablehlo.slice %v197 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4214 = stablehlo.broadcast_in_dim %v4212, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4215 = stablehlo.broadcast_in_dim %v4213, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4216 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v4217 = stablehlo.add %v4215, %v4216 : tensor<128x64x56x56xf32>
    %v4218 = stablehlo.rsqrt %v4217 : tensor<128x64x56x56xf32>
    %v4219 = stablehlo.subtract %v4211, %v4214 : tensor<128x64x56x56xf32>
    %v4220 = stablehlo.multiply %v4219, %v4218 : tensor<128x64x56x56xf32>
    %v4221 = stablehlo.reshape %v4073 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4222 = stablehlo.multiply %v4221, %v4220 : tensor<128x64x56x56xf32>
    %v4223 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4224 = stablehlo.reduce(%v4222 init: %v4223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4225 = stablehlo.reshape %v4073 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4227 = stablehlo.reduce(%v4225 init: %v4226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4228 = stablehlo.reshape %v4181 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4229 = stablehlo.reshape %v128 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4230 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v4231 = stablehlo.compare GT, %v4229, %v4230 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v4232 = stablehlo.select %v4231, %v4228, %v4230 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v4233 = stablehlo.reshape %v4232 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4234 = stablehlo.reshape %v90 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4235 = stablehlo.slice %v109 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4236 = stablehlo.slice %v109 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4237 = stablehlo.broadcast_in_dim %v4235, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4238 = stablehlo.broadcast_in_dim %v4236, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4239 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v4240 = stablehlo.add %v4238, %v4239 : tensor<128x64x56x56xf32>
    %v4241 = stablehlo.rsqrt %v4240 : tensor<128x64x56x56xf32>
    %v4242 = stablehlo.subtract %v4234, %v4237 : tensor<128x64x56x56xf32>
    %v4243 = stablehlo.multiply %v4242, %v4241 : tensor<128x64x56x56xf32>
    %v4244 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4245 = stablehlo.reshape %v4233 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4246 = stablehlo.multiply %v4244, %v4245 : tensor<128x64x56x56xf32>
    %v4247 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4248 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v4249 = stablehlo.reduce(%v4246 init: %v4247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4250 = stablehlo.divide %v4249, %v4248 : tensor<64xf32>
    %v4251 = stablehlo.multiply %v4243, %v4246 : tensor<128x64x56x56xf32>
    %v4252 = stablehlo.reduce(%v4251 init: %v4247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4253 = stablehlo.divide %v4252, %v4248 : tensor<64xf32>
    %v4254 = stablehlo.concatenate %v4250, %v4253, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4255 = stablehlo.concatenate %v109, %v4254, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b0g2dst = "stablehlo.all_reduce"(%v4255) ({
    ^bb0(%aras1b0g2dst: tensor<f32>, %arbs1b0g2dst: tensor<f32>):
      %aradds1b0g2dst = stablehlo.add %aras1b0g2dst, %arbs1b0g2dst : tensor<f32>
      stablehlo.return %aradds1b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g2dst = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans1b0g2dst = stablehlo.divide %arsums1b0g2dst, %arns1b0g2dst : tensor<256xf32>
    %v4256 = stablehlo.reshape %v90 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4257 = stablehlo.slice %armeans1b0g2dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4258 = stablehlo.slice %armeans1b0g2dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4259 = stablehlo.slice %armeans1b0g2dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4260 = stablehlo.slice %armeans1b0g2dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4261 = stablehlo.broadcast_in_dim %v4257, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4262 = stablehlo.broadcast_in_dim %v4258, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4263 = stablehlo.broadcast_in_dim %v4259, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4264 = stablehlo.broadcast_in_dim %v4260, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4265 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v4266 = stablehlo.add %v4262, %v4265 : tensor<128x64x56x56xf32>
    %v4267 = stablehlo.rsqrt %v4266 : tensor<128x64x56x56xf32>
    %v4268 = stablehlo.subtract %v4256, %v4261 : tensor<128x64x56x56xf32>
    %v4269 = stablehlo.multiply %v4268, %v4267 : tensor<128x64x56x56xf32>
    %v4270 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4271 = stablehlo.reshape %v4233 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4272 = stablehlo.multiply %v4270, %v4271 : tensor<128x64x56x56xf32>
    %v4273 = stablehlo.subtract %v4272, %v4263 : tensor<128x64x56x56xf32>
    %v4274 = stablehlo.multiply %v4269, %v4264 : tensor<128x64x56x56xf32>
    %v4275 = stablehlo.subtract %v4273, %v4274 : tensor<128x64x56x56xf32>
    %v4276 = stablehlo.multiply %v4267, %v4275 : tensor<128x64x56x56xf32>
    %v4277 = stablehlo.reshape %v4276 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4278 = stablehlo.reshape %v4277 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4279 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4280 = stablehlo.transpose %v4279, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4281 = stablehlo.convolution(%v4278, %v4280)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v4282 = stablehlo.reshape %v4281 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4283 = stablehlo.reshape %v4282 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4284 = stablehlo.reshape %v83 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4285 = stablehlo.constant dense<0.0> : tensor<128x64x56x56xf32>
    %v4286 = stablehlo.compare GT, %v4284, %v4285 : (tensor<128x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xi1>
    %v4287 = stablehlo.select %v4286, %v4283, %v4285 : tensor<128x64x56x56xi1>, tensor<128x64x56x56xf32>
    %v4288 = stablehlo.reshape %v4287 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4289 = stablehlo.reshape %v49 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4290 = stablehlo.slice %v68 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4291 = stablehlo.slice %v68 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4292 = stablehlo.broadcast_in_dim %v4290, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4293 = stablehlo.broadcast_in_dim %v4291, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4294 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v4295 = stablehlo.add %v4293, %v4294 : tensor<128x64x56x56xf32>
    %v4296 = stablehlo.rsqrt %v4295 : tensor<128x64x56x56xf32>
    %v4297 = stablehlo.subtract %v4289, %v4292 : tensor<128x64x56x56xf32>
    %v4298 = stablehlo.multiply %v4297, %v4296 : tensor<128x64x56x56xf32>
    %v4299 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4300 = stablehlo.reshape %v4288 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4301 = stablehlo.multiply %v4299, %v4300 : tensor<128x64x56x56xf32>
    %v4302 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4303 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v4304 = stablehlo.reduce(%v4301 init: %v4302) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4305 = stablehlo.divide %v4304, %v4303 : tensor<64xf32>
    %v4306 = stablehlo.multiply %v4298, %v4301 : tensor<128x64x56x56xf32>
    %v4307 = stablehlo.reduce(%v4306 init: %v4302) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4308 = stablehlo.divide %v4307, %v4303 : tensor<64xf32>
    %v4309 = stablehlo.concatenate %v4305, %v4308, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4310 = stablehlo.concatenate %v68, %v4309, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b0g1dst = "stablehlo.all_reduce"(%v4310) ({
    ^bb0(%aras1b0g1dst: tensor<f32>, %arbs1b0g1dst: tensor<f32>):
      %aradds1b0g1dst = stablehlo.add %aras1b0g1dst, %arbs1b0g1dst : tensor<f32>
      stablehlo.return %aradds1b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g1dst = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans1b0g1dst = stablehlo.divide %arsums1b0g1dst, %arns1b0g1dst : tensor<256xf32>
    %v4311 = stablehlo.reshape %v49 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4312 = stablehlo.slice %armeans1b0g1dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4313 = stablehlo.slice %armeans1b0g1dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4314 = stablehlo.slice %armeans1b0g1dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4315 = stablehlo.slice %armeans1b0g1dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4316 = stablehlo.broadcast_in_dim %v4312, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4317 = stablehlo.broadcast_in_dim %v4313, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4318 = stablehlo.broadcast_in_dim %v4314, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4319 = stablehlo.broadcast_in_dim %v4315, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4320 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v4321 = stablehlo.add %v4317, %v4320 : tensor<128x64x56x56xf32>
    %v4322 = stablehlo.rsqrt %v4321 : tensor<128x64x56x56xf32>
    %v4323 = stablehlo.subtract %v4311, %v4316 : tensor<128x64x56x56xf32>
    %v4324 = stablehlo.multiply %v4323, %v4322 : tensor<128x64x56x56xf32>
    %v4325 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4326 = stablehlo.reshape %v4288 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4327 = stablehlo.multiply %v4325, %v4326 : tensor<128x64x56x56xf32>
    %v4328 = stablehlo.subtract %v4327, %v4318 : tensor<128x64x56x56xf32>
    %v4329 = stablehlo.multiply %v4324, %v4319 : tensor<128x64x56x56xf32>
    %v4330 = stablehlo.subtract %v4328, %v4329 : tensor<128x64x56x56xf32>
    %v4331 = stablehlo.multiply %v4322, %v4330 : tensor<128x64x56x56xf32>
    %v4332 = stablehlo.reshape %v4331 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4333 = stablehlo.reshape %v4332 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4334 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4335 = stablehlo.transpose %v4334, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4336 = stablehlo.convolution(%v4333, %v4335)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x56x56xf32>
    %v4337 = stablehlo.reshape %v4336 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4338 = stablehlo.reshape %v4337 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4339 = stablehlo.reshape %v4233 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4340 = stablehlo.add %v4338, %v4339 : tensor<128x64x56x56xf32>
    %v4341 = stablehlo.reshape %v4340 : (tensor<128x64x56x56xf32>) -> tensor<128x200704xf32>
    %v4342 = stablehlo.reshape %v44 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4343 = stablehlo.reshape %v4332 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4344 = stablehlo.transpose %v4342, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v4345 = stablehlo.transpose %v4343, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v4346 = stablehlo.convolution(%v4344, %v4345)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4347 = stablehlo.transpose %v4346, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4348 = stablehlo.reshape %v49 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4349 = stablehlo.slice %v68 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4350 = stablehlo.slice %v68 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4351 = stablehlo.broadcast_in_dim %v4349, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4352 = stablehlo.broadcast_in_dim %v4350, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4353 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v4354 = stablehlo.add %v4352, %v4353 : tensor<128x64x56x56xf32>
    %v4355 = stablehlo.rsqrt %v4354 : tensor<128x64x56x56xf32>
    %v4356 = stablehlo.subtract %v4348, %v4351 : tensor<128x64x56x56xf32>
    %v4357 = stablehlo.multiply %v4356, %v4355 : tensor<128x64x56x56xf32>
    %v4358 = stablehlo.reshape %v4288 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4359 = stablehlo.multiply %v4358, %v4357 : tensor<128x64x56x56xf32>
    %v4360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4361 = stablehlo.reduce(%v4359 init: %v4360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4362 = stablehlo.reshape %v4288 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4364 = stablehlo.reduce(%v4362 init: %v4363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4365 = stablehlo.reshape %v85 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4366 = stablehlo.reshape %v4277 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4367 = stablehlo.transpose %v4365, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v4368 = stablehlo.transpose %v4366, dims = [1, 0, 2, 3] : (tensor<128x64x56x56xf32>) -> tensor<64x128x56x56xf32>
    %v4369 = stablehlo.convolution(%v4367, %v4368)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4370 = stablehlo.transpose %v4369, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4371 = stablehlo.reshape %v90 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4372 = stablehlo.slice %v109 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4373 = stablehlo.slice %v109 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4374 = stablehlo.broadcast_in_dim %v4372, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4375 = stablehlo.broadcast_in_dim %v4373, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %v4376 = stablehlo.constant dense<1.0e-05> : tensor<128x64x56x56xf32>
    %v4377 = stablehlo.add %v4375, %v4376 : tensor<128x64x56x56xf32>
    %v4378 = stablehlo.rsqrt %v4377 : tensor<128x64x56x56xf32>
    %v4379 = stablehlo.subtract %v4371, %v4374 : tensor<128x64x56x56xf32>
    %v4380 = stablehlo.multiply %v4379, %v4378 : tensor<128x64x56x56xf32>
    %v4381 = stablehlo.reshape %v4233 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4382 = stablehlo.multiply %v4381, %v4380 : tensor<128x64x56x56xf32>
    %v4383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4384 = stablehlo.reduce(%v4382 init: %v4383) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4385 = stablehlo.reshape %v4233 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4386 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4387 = stablehlo.reduce(%v4385 init: %v4386) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4388 = stablehlo.reshape %v40 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v4389 = stablehlo.reshape %v4341 : (tensor<128x200704xf32>) -> tensor<128x64x56x56xf32>
    %v4390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4391 = "stablehlo.select_and_scatter"(%v4388, %v4389, %v4390) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<128x64x112x112xf32>, tensor<128x64x56x56xf32>, tensor<f32>) -> tensor<128x64x112x112xf32>
    %v4392 = stablehlo.reshape %v4391 : (tensor<128x64x112x112xf32>) -> tensor<128x802816xf32>
    %v4393 = stablehlo.reshape %v4392 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v4394 = stablehlo.reshape %v38 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v4395 = stablehlo.constant dense<0.0> : tensor<128x64x112x112xf32>
    %v4396 = stablehlo.compare GT, %v4394, %v4395 : (tensor<128x64x112x112xf32>, tensor<128x64x112x112xf32>) -> tensor<128x64x112x112xi1>
    %v4397 = stablehlo.select %v4396, %v4393, %v4395 : tensor<128x64x112x112xi1>, tensor<128x64x112x112xf32>
    %v4398 = stablehlo.reshape %v4397 : (tensor<128x64x112x112xf32>) -> tensor<128x802816xf32>
    %v4399 = stablehlo.reshape %v4 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v4400 = stablehlo.slice %v23 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4401 = stablehlo.slice %v23 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4402 = stablehlo.broadcast_in_dim %v4400, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v4403 = stablehlo.broadcast_in_dim %v4401, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v4404 = stablehlo.constant dense<1.0e-05> : tensor<128x64x112x112xf32>
    %v4405 = stablehlo.add %v4403, %v4404 : tensor<128x64x112x112xf32>
    %v4406 = stablehlo.rsqrt %v4405 : tensor<128x64x112x112xf32>
    %v4407 = stablehlo.subtract %v4399, %v4402 : tensor<128x64x112x112xf32>
    %v4408 = stablehlo.multiply %v4407, %v4406 : tensor<128x64x112x112xf32>
    %v4409 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v4410 = stablehlo.reshape %v4398 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v4411 = stablehlo.multiply %v4409, %v4410 : tensor<128x64x112x112xf32>
    %v4412 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4413 = stablehlo.constant dense<1605632.0> : tensor<64xf32>
    %v4414 = stablehlo.reduce(%v4411 init: %v4412) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4415 = stablehlo.divide %v4414, %v4413 : tensor<64xf32>
    %v4416 = stablehlo.multiply %v4408, %v4411 : tensor<128x64x112x112xf32>
    %v4417 = stablehlo.reduce(%v4416 init: %v4412) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4418 = stablehlo.divide %v4417, %v4413 : tensor<64xf32>
    %v4419 = stablehlo.concatenate %v4415, %v4418, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4420 = stablehlo.concatenate %v23, %v4419, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsumsgdst = "stablehlo.all_reduce"(%v4420) ({
    ^bb0(%arasgdst: tensor<f32>, %arbsgdst: tensor<f32>):
      %araddsgdst = stablehlo.add %arasgdst, %arbsgdst : tensor<f32>
      stablehlo.return %araddsgdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnsgdst = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeansgdst = stablehlo.divide %arsumsgdst, %arnsgdst : tensor<256xf32>
    %v4421 = stablehlo.reshape %v4 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v4422 = stablehlo.slice %armeansgdst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4423 = stablehlo.slice %armeansgdst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4424 = stablehlo.slice %armeansgdst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4425 = stablehlo.slice %armeansgdst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4426 = stablehlo.broadcast_in_dim %v4422, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v4427 = stablehlo.broadcast_in_dim %v4423, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v4428 = stablehlo.broadcast_in_dim %v4424, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v4429 = stablehlo.broadcast_in_dim %v4425, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v4430 = stablehlo.constant dense<1.0e-05> : tensor<128x64x112x112xf32>
    %v4431 = stablehlo.add %v4427, %v4430 : tensor<128x64x112x112xf32>
    %v4432 = stablehlo.rsqrt %v4431 : tensor<128x64x112x112xf32>
    %v4433 = stablehlo.subtract %v4421, %v4426 : tensor<128x64x112x112xf32>
    %v4434 = stablehlo.multiply %v4433, %v4432 : tensor<128x64x112x112xf32>
    %v4435 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v4436 = stablehlo.reshape %v4398 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v4437 = stablehlo.multiply %v4435, %v4436 : tensor<128x64x112x112xf32>
    %v4438 = stablehlo.subtract %v4437, %v4428 : tensor<128x64x112x112xf32>
    %v4439 = stablehlo.multiply %v4434, %v4429 : tensor<128x64x112x112xf32>
    %v4440 = stablehlo.subtract %v4438, %v4439 : tensor<128x64x112x112xf32>
    %v4441 = stablehlo.multiply %v4432, %v4440 : tensor<128x64x112x112xf32>
    %v4442 = stablehlo.reshape %v4441 : (tensor<128x64x112x112xf32>) -> tensor<128x802816xf32>
    %v4443 = stablehlo.reshape %x : (tensor<128x150528xf32>) -> tensor<128x3x224x224xf32>
    %v4444 = stablehlo.reshape %v4442 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v4445 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4446 = stablehlo.pad %v4444, %v4445, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<128x64x224x224xf32>
    %v4447 = stablehlo.transpose %v4443, dims = [1, 0, 2, 3] : (tensor<128x3x224x224xf32>) -> tensor<3x128x224x224xf32>
    %v4448 = stablehlo.transpose %v4446, dims = [1, 0, 2, 3] : (tensor<128x64x224x224xf32>) -> tensor<64x128x224x224xf32>
    %v4449 = stablehlo.convolution(%v4447, %v4448)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x224x224xf32>, tensor<64x128x224x224xf32>) -> tensor<3x64x7x7xf32>
    %v4450 = stablehlo.transpose %v4449, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v4451 = stablehlo.reshape %v4 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v4452 = stablehlo.slice %v23 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4453 = stablehlo.slice %v23 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4454 = stablehlo.broadcast_in_dim %v4452, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v4455 = stablehlo.broadcast_in_dim %v4453, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %v4456 = stablehlo.constant dense<1.0e-05> : tensor<128x64x112x112xf32>
    %v4457 = stablehlo.add %v4455, %v4456 : tensor<128x64x112x112xf32>
    %v4458 = stablehlo.rsqrt %v4457 : tensor<128x64x112x112xf32>
    %v4459 = stablehlo.subtract %v4451, %v4454 : tensor<128x64x112x112xf32>
    %v4460 = stablehlo.multiply %v4459, %v4458 : tensor<128x64x112x112xf32>
    %v4461 = stablehlo.reshape %v4398 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v4462 = stablehlo.multiply %v4461, %v4460 : tensor<128x64x112x112xf32>
    %v4463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4464 = stablehlo.reduce(%v4462 init: %v4463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4465 = stablehlo.reshape %v4398 : (tensor<128x802816xf32>) -> tensor<128x64x112x112xf32>
    %v4466 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4467 = stablehlo.reduce(%v4465 init: %v4466) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4468 = stablehlo.slice %v23 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4469 = stablehlo.slice %v23 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4470 = stablehlo.slice %v68 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4471 = stablehlo.slice %v68 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4472 = stablehlo.slice %v109 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4473 = stablehlo.slice %v109 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4474 = stablehlo.slice %v156 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4475 = stablehlo.slice %v156 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4476 = stablehlo.slice %v197 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4477 = stablehlo.slice %v197 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4478 = stablehlo.slice %v244 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4479 = stablehlo.slice %v244 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4480 = stablehlo.slice %v285 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4481 = stablehlo.slice %v285 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4482 = stablehlo.slice %v332 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4483 = stablehlo.slice %v332 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4484 = stablehlo.slice %v373 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4485 = stablehlo.slice %v373 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4486 = stablehlo.slice %v412 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4487 = stablehlo.slice %v412 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4488 = stablehlo.slice %v454 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4489 = stablehlo.slice %v454 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4490 = stablehlo.slice %v495 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4491 = stablehlo.slice %v495 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4492 = stablehlo.slice %v542 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4493 = stablehlo.slice %v542 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4494 = stablehlo.slice %v583 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4495 = stablehlo.slice %v583 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4496 = stablehlo.slice %v630 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4497 = stablehlo.slice %v630 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4498 = stablehlo.slice %v671 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4499 = stablehlo.slice %v671 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4500 = stablehlo.slice %v718 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4501 = stablehlo.slice %v718 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4502 = stablehlo.slice %v759 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4503 = stablehlo.slice %v759 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4504 = stablehlo.slice %v798 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4505 = stablehlo.slice %v798 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4506 = stablehlo.slice %v840 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4507 = stablehlo.slice %v840 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4508 = stablehlo.slice %v881 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4509 = stablehlo.slice %v881 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4510 = stablehlo.slice %v928 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4511 = stablehlo.slice %v928 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4512 = stablehlo.slice %v969 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4513 = stablehlo.slice %v969 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4514 = stablehlo.slice %v1016 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4515 = stablehlo.slice %v1016 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4516 = stablehlo.slice %v1057 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4517 = stablehlo.slice %v1057 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4518 = stablehlo.slice %v1104 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4519 = stablehlo.slice %v1104 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4520 = stablehlo.slice %v1145 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4521 = stablehlo.slice %v1145 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4522 = stablehlo.slice %v1192 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4523 = stablehlo.slice %v1192 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4524 = stablehlo.slice %v1233 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4525 = stablehlo.slice %v1233 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4526 = stablehlo.slice %v1280 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4527 = stablehlo.slice %v1280 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4528 = stablehlo.slice %v1321 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4529 = stablehlo.slice %v1321 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4530 = stablehlo.slice %v1360 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4531 = stablehlo.slice %v1360 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4532 = stablehlo.slice %v1402 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4533 = stablehlo.slice %v1402 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4534 = stablehlo.slice %v1443 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4535 = stablehlo.slice %v1443 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4536 = stablehlo.slice %v1490 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4537 = stablehlo.slice %v1490 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4538 = stablehlo.slice %v1531 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4539 = stablehlo.slice %v1531 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %arsumsW = "stablehlo.all_reduce"(%v4450) ({
    ^bb0(%arasW: tensor<f32>, %arbsW: tensor<f32>):
      %araddsW = stablehlo.add %arasW, %arbsW : tensor<f32>
      stablehlo.return %araddsW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
    %arnsW = stablehlo.constant dense<2.0> : tensor<64x3x7x7xf32>
    %armeansW = stablehlo.divide %arsumsW, %arnsW : tensor<64x3x7x7xf32>
    %v4540 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4541 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4542 = stablehlo.multiply %v4540, %sWm : tensor<64x3x7x7xf32>
    %v4543 = stablehlo.multiply %v4541, %armeansW : tensor<64x3x7x7xf32>
    %v4544 = stablehlo.add %v4542, %v4543 : tensor<64x3x7x7xf32>
    %v4545 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4546 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4547 = stablehlo.multiply %v4545, %sWv : tensor<64x3x7x7xf32>
    %v4548 = stablehlo.multiply %armeansW, %armeansW : tensor<64x3x7x7xf32>
    %v4549 = stablehlo.multiply %v4546, %v4548 : tensor<64x3x7x7xf32>
    %v4550 = stablehlo.add %v4547, %v4549 : tensor<64x3x7x7xf32>
    %v4551 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4552 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4553 = stablehlo.multiply %v4551, %sWm : tensor<64x3x7x7xf32>
    %v4554 = stablehlo.multiply %v4552, %armeansW : tensor<64x3x7x7xf32>
    %v4555 = stablehlo.add %v4553, %v4554 : tensor<64x3x7x7xf32>
    %v4556 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4557 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4558 = stablehlo.multiply %v4556, %sWv : tensor<64x3x7x7xf32>
    %v4559 = stablehlo.multiply %armeansW, %armeansW : tensor<64x3x7x7xf32>
    %v4560 = stablehlo.multiply %v4557, %v4559 : tensor<64x3x7x7xf32>
    %v4561 = stablehlo.add %v4558, %v4560 : tensor<64x3x7x7xf32>
    %v4562 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4563 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4564 = stablehlo.divide %v4555, %v4562 : tensor<64x3x7x7xf32>
    %v4565 = stablehlo.divide %v4561, %v4563 : tensor<64x3x7x7xf32>
    %v4566 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4567 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4568 = stablehlo.sqrt %v4565 : tensor<64x3x7x7xf32>
    %v4569 = stablehlo.add %v4568, %v4567 : tensor<64x3x7x7xf32>
    %v4570 = stablehlo.divide %v4564, %v4569 : tensor<64x3x7x7xf32>
    %v4571 = stablehlo.multiply %v4566, %v4570 : tensor<64x3x7x7xf32>
    %v4572 = stablehlo.subtract %sW, %v4571 : tensor<64x3x7x7xf32>
    %v4573 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4574 = stablehlo.multiply %v4573, %v4566 : tensor<64x3x7x7xf32>
    %v4575 = stablehlo.multiply %v4574, %sW : tensor<64x3x7x7xf32>
    %v4576 = stablehlo.subtract %v4572, %v4575 : tensor<64x3x7x7xf32>
    %arsumsg = "stablehlo.all_reduce"(%v4464) ({
    ^bb0(%arasg: tensor<f32>, %arbsg: tensor<f32>):
      %araddsg = stablehlo.add %arasg, %arbsg : tensor<f32>
      stablehlo.return %araddsg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsg = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeansg = stablehlo.divide %arsumsg, %arnsg : tensor<64xf32>
    %v4577 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4578 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4579 = stablehlo.multiply %v4577, %sgm : tensor<64xf32>
    %v4580 = stablehlo.multiply %v4578, %armeansg : tensor<64xf32>
    %v4581 = stablehlo.add %v4579, %v4580 : tensor<64xf32>
    %v4582 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4583 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4584 = stablehlo.multiply %v4582, %sgv : tensor<64xf32>
    %v4585 = stablehlo.multiply %armeansg, %armeansg : tensor<64xf32>
    %v4586 = stablehlo.multiply %v4583, %v4585 : tensor<64xf32>
    %v4587 = stablehlo.add %v4584, %v4586 : tensor<64xf32>
    %v4588 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4589 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4590 = stablehlo.multiply %v4588, %sgm : tensor<64xf32>
    %v4591 = stablehlo.multiply %v4589, %armeansg : tensor<64xf32>
    %v4592 = stablehlo.add %v4590, %v4591 : tensor<64xf32>
    %v4593 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4594 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4595 = stablehlo.multiply %v4593, %sgv : tensor<64xf32>
    %v4596 = stablehlo.multiply %armeansg, %armeansg : tensor<64xf32>
    %v4597 = stablehlo.multiply %v4594, %v4596 : tensor<64xf32>
    %v4598 = stablehlo.add %v4595, %v4597 : tensor<64xf32>
    %v4599 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4600 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4601 = stablehlo.divide %v4592, %v4599 : tensor<64xf32>
    %v4602 = stablehlo.divide %v4598, %v4600 : tensor<64xf32>
    %v4603 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4604 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4605 = stablehlo.sqrt %v4602 : tensor<64xf32>
    %v4606 = stablehlo.add %v4605, %v4604 : tensor<64xf32>
    %v4607 = stablehlo.divide %v4601, %v4606 : tensor<64xf32>
    %v4608 = stablehlo.multiply %v4603, %v4607 : tensor<64xf32>
    %v4609 = stablehlo.subtract %sg, %v4608 : tensor<64xf32>
    %v4610 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4611 = stablehlo.multiply %v4610, %v4603 : tensor<64xf32>
    %v4612 = stablehlo.multiply %v4611, %sg : tensor<64xf32>
    %v4613 = stablehlo.subtract %v4609, %v4612 : tensor<64xf32>
    %arsumsbt = "stablehlo.all_reduce"(%v4467) ({
    ^bb0(%arasbt: tensor<f32>, %arbsbt: tensor<f32>):
      %araddsbt = stablehlo.add %arasbt, %arbsbt : tensor<f32>
      stablehlo.return %araddsbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsbt = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeansbt = stablehlo.divide %arsumsbt, %arnsbt : tensor<64xf32>
    %v4614 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4615 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4616 = stablehlo.multiply %v4614, %sbtm : tensor<64xf32>
    %v4617 = stablehlo.multiply %v4615, %armeansbt : tensor<64xf32>
    %v4618 = stablehlo.add %v4616, %v4617 : tensor<64xf32>
    %v4619 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4620 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4621 = stablehlo.multiply %v4619, %sbtv : tensor<64xf32>
    %v4622 = stablehlo.multiply %armeansbt, %armeansbt : tensor<64xf32>
    %v4623 = stablehlo.multiply %v4620, %v4622 : tensor<64xf32>
    %v4624 = stablehlo.add %v4621, %v4623 : tensor<64xf32>
    %v4625 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4626 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4627 = stablehlo.multiply %v4625, %sbtm : tensor<64xf32>
    %v4628 = stablehlo.multiply %v4626, %armeansbt : tensor<64xf32>
    %v4629 = stablehlo.add %v4627, %v4628 : tensor<64xf32>
    %v4630 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4631 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4632 = stablehlo.multiply %v4630, %sbtv : tensor<64xf32>
    %v4633 = stablehlo.multiply %armeansbt, %armeansbt : tensor<64xf32>
    %v4634 = stablehlo.multiply %v4631, %v4633 : tensor<64xf32>
    %v4635 = stablehlo.add %v4632, %v4634 : tensor<64xf32>
    %v4636 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4637 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4638 = stablehlo.divide %v4629, %v4636 : tensor<64xf32>
    %v4639 = stablehlo.divide %v4635, %v4637 : tensor<64xf32>
    %v4640 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4641 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4642 = stablehlo.sqrt %v4639 : tensor<64xf32>
    %v4643 = stablehlo.add %v4642, %v4641 : tensor<64xf32>
    %v4644 = stablehlo.divide %v4638, %v4643 : tensor<64xf32>
    %v4645 = stablehlo.multiply %v4640, %v4644 : tensor<64xf32>
    %v4646 = stablehlo.subtract %sbt, %v4645 : tensor<64xf32>
    %v4647 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4648 = stablehlo.multiply %v4647, %v4640 : tensor<64xf32>
    %v4649 = stablehlo.multiply %v4648, %sbt : tensor<64xf32>
    %v4650 = stablehlo.subtract %v4646, %v4649 : tensor<64xf32>
    %arsums1b0W1 = "stablehlo.all_reduce"(%v4347) ({
    ^bb0(%aras1b0W1: tensor<f32>, %arbs1b0W1: tensor<f32>):
      %aradds1b0W1 = stablehlo.add %aras1b0W1, %arbs1b0W1 : tensor<f32>
      stablehlo.return %aradds1b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W1 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b0W1 = stablehlo.divide %arsums1b0W1, %arns1b0W1 : tensor<64x64x3x3xf32>
    %v4651 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4652 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4653 = stablehlo.multiply %v4651, %s1b0W1m : tensor<64x64x3x3xf32>
    %v4654 = stablehlo.multiply %v4652, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4655 = stablehlo.add %v4653, %v4654 : tensor<64x64x3x3xf32>
    %v4656 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4657 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4658 = stablehlo.multiply %v4656, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4659 = stablehlo.multiply %armeans1b0W1, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4660 = stablehlo.multiply %v4657, %v4659 : tensor<64x64x3x3xf32>
    %v4661 = stablehlo.add %v4658, %v4660 : tensor<64x64x3x3xf32>
    %v4662 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4663 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4664 = stablehlo.multiply %v4662, %s1b0W1m : tensor<64x64x3x3xf32>
    %v4665 = stablehlo.multiply %v4663, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4666 = stablehlo.add %v4664, %v4665 : tensor<64x64x3x3xf32>
    %v4667 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4668 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4669 = stablehlo.multiply %v4667, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4670 = stablehlo.multiply %armeans1b0W1, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4671 = stablehlo.multiply %v4668, %v4670 : tensor<64x64x3x3xf32>
    %v4672 = stablehlo.add %v4669, %v4671 : tensor<64x64x3x3xf32>
    %v4673 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4674 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4675 = stablehlo.divide %v4666, %v4673 : tensor<64x64x3x3xf32>
    %v4676 = stablehlo.divide %v4672, %v4674 : tensor<64x64x3x3xf32>
    %v4677 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4678 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4679 = stablehlo.sqrt %v4676 : tensor<64x64x3x3xf32>
    %v4680 = stablehlo.add %v4679, %v4678 : tensor<64x64x3x3xf32>
    %v4681 = stablehlo.divide %v4675, %v4680 : tensor<64x64x3x3xf32>
    %v4682 = stablehlo.multiply %v4677, %v4681 : tensor<64x64x3x3xf32>
    %v4683 = stablehlo.subtract %s1b0W1, %v4682 : tensor<64x64x3x3xf32>
    %v4684 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4685 = stablehlo.multiply %v4684, %v4677 : tensor<64x64x3x3xf32>
    %v4686 = stablehlo.multiply %v4685, %s1b0W1 : tensor<64x64x3x3xf32>
    %v4687 = stablehlo.subtract %v4683, %v4686 : tensor<64x64x3x3xf32>
    %arsums1b0g1 = "stablehlo.all_reduce"(%v4361) ({
    ^bb0(%aras1b0g1: tensor<f32>, %arbs1b0g1: tensor<f32>):
      %aradds1b0g1 = stablehlo.add %aras1b0g1, %arbs1b0g1 : tensor<f32>
      stablehlo.return %aradds1b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g1 = stablehlo.divide %arsums1b0g1, %arns1b0g1 : tensor<64xf32>
    %v4688 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4689 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4690 = stablehlo.multiply %v4688, %s1b0g1m : tensor<64xf32>
    %v4691 = stablehlo.multiply %v4689, %armeans1b0g1 : tensor<64xf32>
    %v4692 = stablehlo.add %v4690, %v4691 : tensor<64xf32>
    %v4693 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4694 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4695 = stablehlo.multiply %v4693, %s1b0g1v : tensor<64xf32>
    %v4696 = stablehlo.multiply %armeans1b0g1, %armeans1b0g1 : tensor<64xf32>
    %v4697 = stablehlo.multiply %v4694, %v4696 : tensor<64xf32>
    %v4698 = stablehlo.add %v4695, %v4697 : tensor<64xf32>
    %v4699 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4700 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4701 = stablehlo.multiply %v4699, %s1b0g1m : tensor<64xf32>
    %v4702 = stablehlo.multiply %v4700, %armeans1b0g1 : tensor<64xf32>
    %v4703 = stablehlo.add %v4701, %v4702 : tensor<64xf32>
    %v4704 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4705 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4706 = stablehlo.multiply %v4704, %s1b0g1v : tensor<64xf32>
    %v4707 = stablehlo.multiply %armeans1b0g1, %armeans1b0g1 : tensor<64xf32>
    %v4708 = stablehlo.multiply %v4705, %v4707 : tensor<64xf32>
    %v4709 = stablehlo.add %v4706, %v4708 : tensor<64xf32>
    %v4710 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4711 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4712 = stablehlo.divide %v4703, %v4710 : tensor<64xf32>
    %v4713 = stablehlo.divide %v4709, %v4711 : tensor<64xf32>
    %v4714 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4715 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4716 = stablehlo.sqrt %v4713 : tensor<64xf32>
    %v4717 = stablehlo.add %v4716, %v4715 : tensor<64xf32>
    %v4718 = stablehlo.divide %v4712, %v4717 : tensor<64xf32>
    %v4719 = stablehlo.multiply %v4714, %v4718 : tensor<64xf32>
    %v4720 = stablehlo.subtract %s1b0g1, %v4719 : tensor<64xf32>
    %v4721 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4722 = stablehlo.multiply %v4721, %v4714 : tensor<64xf32>
    %v4723 = stablehlo.multiply %v4722, %s1b0g1 : tensor<64xf32>
    %v4724 = stablehlo.subtract %v4720, %v4723 : tensor<64xf32>
    %arsums1b0bt1 = "stablehlo.all_reduce"(%v4364) ({
    ^bb0(%aras1b0bt1: tensor<f32>, %arbs1b0bt1: tensor<f32>):
      %aradds1b0bt1 = stablehlo.add %aras1b0bt1, %arbs1b0bt1 : tensor<f32>
      stablehlo.return %aradds1b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0bt1 = stablehlo.divide %arsums1b0bt1, %arns1b0bt1 : tensor<64xf32>
    %v4725 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4726 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4727 = stablehlo.multiply %v4725, %s1b0bt1m : tensor<64xf32>
    %v4728 = stablehlo.multiply %v4726, %armeans1b0bt1 : tensor<64xf32>
    %v4729 = stablehlo.add %v4727, %v4728 : tensor<64xf32>
    %v4730 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4731 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4732 = stablehlo.multiply %v4730, %s1b0bt1v : tensor<64xf32>
    %v4733 = stablehlo.multiply %armeans1b0bt1, %armeans1b0bt1 : tensor<64xf32>
    %v4734 = stablehlo.multiply %v4731, %v4733 : tensor<64xf32>
    %v4735 = stablehlo.add %v4732, %v4734 : tensor<64xf32>
    %v4736 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4737 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4738 = stablehlo.multiply %v4736, %s1b0bt1m : tensor<64xf32>
    %v4739 = stablehlo.multiply %v4737, %armeans1b0bt1 : tensor<64xf32>
    %v4740 = stablehlo.add %v4738, %v4739 : tensor<64xf32>
    %v4741 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4742 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4743 = stablehlo.multiply %v4741, %s1b0bt1v : tensor<64xf32>
    %v4744 = stablehlo.multiply %armeans1b0bt1, %armeans1b0bt1 : tensor<64xf32>
    %v4745 = stablehlo.multiply %v4742, %v4744 : tensor<64xf32>
    %v4746 = stablehlo.add %v4743, %v4745 : tensor<64xf32>
    %v4747 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4748 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4749 = stablehlo.divide %v4740, %v4747 : tensor<64xf32>
    %v4750 = stablehlo.divide %v4746, %v4748 : tensor<64xf32>
    %v4751 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4752 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4753 = stablehlo.sqrt %v4750 : tensor<64xf32>
    %v4754 = stablehlo.add %v4753, %v4752 : tensor<64xf32>
    %v4755 = stablehlo.divide %v4749, %v4754 : tensor<64xf32>
    %v4756 = stablehlo.multiply %v4751, %v4755 : tensor<64xf32>
    %v4757 = stablehlo.subtract %s1b0bt1, %v4756 : tensor<64xf32>
    %v4758 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4759 = stablehlo.multiply %v4758, %v4751 : tensor<64xf32>
    %v4760 = stablehlo.multiply %v4759, %s1b0bt1 : tensor<64xf32>
    %v4761 = stablehlo.subtract %v4757, %v4760 : tensor<64xf32>
    %arsums1b0W2 = "stablehlo.all_reduce"(%v4370) ({
    ^bb0(%aras1b0W2: tensor<f32>, %arbs1b0W2: tensor<f32>):
      %aradds1b0W2 = stablehlo.add %aras1b0W2, %arbs1b0W2 : tensor<f32>
      stablehlo.return %aradds1b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W2 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b0W2 = stablehlo.divide %arsums1b0W2, %arns1b0W2 : tensor<64x64x3x3xf32>
    %v4762 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4763 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4764 = stablehlo.multiply %v4762, %s1b0W2m : tensor<64x64x3x3xf32>
    %v4765 = stablehlo.multiply %v4763, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4766 = stablehlo.add %v4764, %v4765 : tensor<64x64x3x3xf32>
    %v4767 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4768 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4769 = stablehlo.multiply %v4767, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4770 = stablehlo.multiply %armeans1b0W2, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4771 = stablehlo.multiply %v4768, %v4770 : tensor<64x64x3x3xf32>
    %v4772 = stablehlo.add %v4769, %v4771 : tensor<64x64x3x3xf32>
    %v4773 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4774 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4775 = stablehlo.multiply %v4773, %s1b0W2m : tensor<64x64x3x3xf32>
    %v4776 = stablehlo.multiply %v4774, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4777 = stablehlo.add %v4775, %v4776 : tensor<64x64x3x3xf32>
    %v4778 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4779 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4780 = stablehlo.multiply %v4778, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4781 = stablehlo.multiply %armeans1b0W2, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4782 = stablehlo.multiply %v4779, %v4781 : tensor<64x64x3x3xf32>
    %v4783 = stablehlo.add %v4780, %v4782 : tensor<64x64x3x3xf32>
    %v4784 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4785 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4786 = stablehlo.divide %v4777, %v4784 : tensor<64x64x3x3xf32>
    %v4787 = stablehlo.divide %v4783, %v4785 : tensor<64x64x3x3xf32>
    %v4788 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4789 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4790 = stablehlo.sqrt %v4787 : tensor<64x64x3x3xf32>
    %v4791 = stablehlo.add %v4790, %v4789 : tensor<64x64x3x3xf32>
    %v4792 = stablehlo.divide %v4786, %v4791 : tensor<64x64x3x3xf32>
    %v4793 = stablehlo.multiply %v4788, %v4792 : tensor<64x64x3x3xf32>
    %v4794 = stablehlo.subtract %s1b0W2, %v4793 : tensor<64x64x3x3xf32>
    %v4795 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4796 = stablehlo.multiply %v4795, %v4788 : tensor<64x64x3x3xf32>
    %v4797 = stablehlo.multiply %v4796, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4798 = stablehlo.subtract %v4794, %v4797 : tensor<64x64x3x3xf32>
    %arsums1b0g2 = "stablehlo.all_reduce"(%v4384) ({
    ^bb0(%aras1b0g2: tensor<f32>, %arbs1b0g2: tensor<f32>):
      %aradds1b0g2 = stablehlo.add %aras1b0g2, %arbs1b0g2 : tensor<f32>
      stablehlo.return %aradds1b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g2 = stablehlo.divide %arsums1b0g2, %arns1b0g2 : tensor<64xf32>
    %v4799 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4800 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4801 = stablehlo.multiply %v4799, %s1b0g2m : tensor<64xf32>
    %v4802 = stablehlo.multiply %v4800, %armeans1b0g2 : tensor<64xf32>
    %v4803 = stablehlo.add %v4801, %v4802 : tensor<64xf32>
    %v4804 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4805 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4806 = stablehlo.multiply %v4804, %s1b0g2v : tensor<64xf32>
    %v4807 = stablehlo.multiply %armeans1b0g2, %armeans1b0g2 : tensor<64xf32>
    %v4808 = stablehlo.multiply %v4805, %v4807 : tensor<64xf32>
    %v4809 = stablehlo.add %v4806, %v4808 : tensor<64xf32>
    %v4810 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4811 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4812 = stablehlo.multiply %v4810, %s1b0g2m : tensor<64xf32>
    %v4813 = stablehlo.multiply %v4811, %armeans1b0g2 : tensor<64xf32>
    %v4814 = stablehlo.add %v4812, %v4813 : tensor<64xf32>
    %v4815 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4816 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4817 = stablehlo.multiply %v4815, %s1b0g2v : tensor<64xf32>
    %v4818 = stablehlo.multiply %armeans1b0g2, %armeans1b0g2 : tensor<64xf32>
    %v4819 = stablehlo.multiply %v4816, %v4818 : tensor<64xf32>
    %v4820 = stablehlo.add %v4817, %v4819 : tensor<64xf32>
    %v4821 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4822 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4823 = stablehlo.divide %v4814, %v4821 : tensor<64xf32>
    %v4824 = stablehlo.divide %v4820, %v4822 : tensor<64xf32>
    %v4825 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4826 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4827 = stablehlo.sqrt %v4824 : tensor<64xf32>
    %v4828 = stablehlo.add %v4827, %v4826 : tensor<64xf32>
    %v4829 = stablehlo.divide %v4823, %v4828 : tensor<64xf32>
    %v4830 = stablehlo.multiply %v4825, %v4829 : tensor<64xf32>
    %v4831 = stablehlo.subtract %s1b0g2, %v4830 : tensor<64xf32>
    %v4832 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4833 = stablehlo.multiply %v4832, %v4825 : tensor<64xf32>
    %v4834 = stablehlo.multiply %v4833, %s1b0g2 : tensor<64xf32>
    %v4835 = stablehlo.subtract %v4831, %v4834 : tensor<64xf32>
    %arsums1b0bt2 = "stablehlo.all_reduce"(%v4387) ({
    ^bb0(%aras1b0bt2: tensor<f32>, %arbs1b0bt2: tensor<f32>):
      %aradds1b0bt2 = stablehlo.add %aras1b0bt2, %arbs1b0bt2 : tensor<f32>
      stablehlo.return %aradds1b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0bt2 = stablehlo.divide %arsums1b0bt2, %arns1b0bt2 : tensor<64xf32>
    %v4836 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4837 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4838 = stablehlo.multiply %v4836, %s1b0bt2m : tensor<64xf32>
    %v4839 = stablehlo.multiply %v4837, %armeans1b0bt2 : tensor<64xf32>
    %v4840 = stablehlo.add %v4838, %v4839 : tensor<64xf32>
    %v4841 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4842 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4843 = stablehlo.multiply %v4841, %s1b0bt2v : tensor<64xf32>
    %v4844 = stablehlo.multiply %armeans1b0bt2, %armeans1b0bt2 : tensor<64xf32>
    %v4845 = stablehlo.multiply %v4842, %v4844 : tensor<64xf32>
    %v4846 = stablehlo.add %v4843, %v4845 : tensor<64xf32>
    %v4847 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4848 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4849 = stablehlo.multiply %v4847, %s1b0bt2m : tensor<64xf32>
    %v4850 = stablehlo.multiply %v4848, %armeans1b0bt2 : tensor<64xf32>
    %v4851 = stablehlo.add %v4849, %v4850 : tensor<64xf32>
    %v4852 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4853 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4854 = stablehlo.multiply %v4852, %s1b0bt2v : tensor<64xf32>
    %v4855 = stablehlo.multiply %armeans1b0bt2, %armeans1b0bt2 : tensor<64xf32>
    %v4856 = stablehlo.multiply %v4853, %v4855 : tensor<64xf32>
    %v4857 = stablehlo.add %v4854, %v4856 : tensor<64xf32>
    %v4858 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4859 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4860 = stablehlo.divide %v4851, %v4858 : tensor<64xf32>
    %v4861 = stablehlo.divide %v4857, %v4859 : tensor<64xf32>
    %v4862 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4863 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4864 = stablehlo.sqrt %v4861 : tensor<64xf32>
    %v4865 = stablehlo.add %v4864, %v4863 : tensor<64xf32>
    %v4866 = stablehlo.divide %v4860, %v4865 : tensor<64xf32>
    %v4867 = stablehlo.multiply %v4862, %v4866 : tensor<64xf32>
    %v4868 = stablehlo.subtract %s1b0bt2, %v4867 : tensor<64xf32>
    %v4869 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4870 = stablehlo.multiply %v4869, %v4862 : tensor<64xf32>
    %v4871 = stablehlo.multiply %v4870, %s1b0bt2 : tensor<64xf32>
    %v4872 = stablehlo.subtract %v4868, %v4871 : tensor<64xf32>
    %arsums1b1W1 = "stablehlo.all_reduce"(%v4187) ({
    ^bb0(%aras1b1W1: tensor<f32>, %arbs1b1W1: tensor<f32>):
      %aradds1b1W1 = stablehlo.add %aras1b1W1, %arbs1b1W1 : tensor<f32>
      stablehlo.return %aradds1b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W1 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b1W1 = stablehlo.divide %arsums1b1W1, %arns1b1W1 : tensor<64x64x3x3xf32>
    %v4873 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4874 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4875 = stablehlo.multiply %v4873, %s1b1W1m : tensor<64x64x3x3xf32>
    %v4876 = stablehlo.multiply %v4874, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4877 = stablehlo.add %v4875, %v4876 : tensor<64x64x3x3xf32>
    %v4878 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4879 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4880 = stablehlo.multiply %v4878, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4881 = stablehlo.multiply %armeans1b1W1, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4882 = stablehlo.multiply %v4879, %v4881 : tensor<64x64x3x3xf32>
    %v4883 = stablehlo.add %v4880, %v4882 : tensor<64x64x3x3xf32>
    %v4884 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4885 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4886 = stablehlo.multiply %v4884, %s1b1W1m : tensor<64x64x3x3xf32>
    %v4887 = stablehlo.multiply %v4885, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4888 = stablehlo.add %v4886, %v4887 : tensor<64x64x3x3xf32>
    %v4889 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4890 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4891 = stablehlo.multiply %v4889, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4892 = stablehlo.multiply %armeans1b1W1, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4893 = stablehlo.multiply %v4890, %v4892 : tensor<64x64x3x3xf32>
    %v4894 = stablehlo.add %v4891, %v4893 : tensor<64x64x3x3xf32>
    %v4895 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4896 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4897 = stablehlo.divide %v4888, %v4895 : tensor<64x64x3x3xf32>
    %v4898 = stablehlo.divide %v4894, %v4896 : tensor<64x64x3x3xf32>
    %v4899 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4900 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4901 = stablehlo.sqrt %v4898 : tensor<64x64x3x3xf32>
    %v4902 = stablehlo.add %v4901, %v4900 : tensor<64x64x3x3xf32>
    %v4903 = stablehlo.divide %v4897, %v4902 : tensor<64x64x3x3xf32>
    %v4904 = stablehlo.multiply %v4899, %v4903 : tensor<64x64x3x3xf32>
    %v4905 = stablehlo.subtract %s1b1W1, %v4904 : tensor<64x64x3x3xf32>
    %v4906 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4907 = stablehlo.multiply %v4906, %v4899 : tensor<64x64x3x3xf32>
    %v4908 = stablehlo.multiply %v4907, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4909 = stablehlo.subtract %v4905, %v4908 : tensor<64x64x3x3xf32>
    %arsums1b1g1 = "stablehlo.all_reduce"(%v4201) ({
    ^bb0(%aras1b1g1: tensor<f32>, %arbs1b1g1: tensor<f32>):
      %aradds1b1g1 = stablehlo.add %aras1b1g1, %arbs1b1g1 : tensor<f32>
      stablehlo.return %aradds1b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g1 = stablehlo.divide %arsums1b1g1, %arns1b1g1 : tensor<64xf32>
    %v4910 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4911 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4912 = stablehlo.multiply %v4910, %s1b1g1m : tensor<64xf32>
    %v4913 = stablehlo.multiply %v4911, %armeans1b1g1 : tensor<64xf32>
    %v4914 = stablehlo.add %v4912, %v4913 : tensor<64xf32>
    %v4915 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4916 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4917 = stablehlo.multiply %v4915, %s1b1g1v : tensor<64xf32>
    %v4918 = stablehlo.multiply %armeans1b1g1, %armeans1b1g1 : tensor<64xf32>
    %v4919 = stablehlo.multiply %v4916, %v4918 : tensor<64xf32>
    %v4920 = stablehlo.add %v4917, %v4919 : tensor<64xf32>
    %v4921 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4922 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4923 = stablehlo.multiply %v4921, %s1b1g1m : tensor<64xf32>
    %v4924 = stablehlo.multiply %v4922, %armeans1b1g1 : tensor<64xf32>
    %v4925 = stablehlo.add %v4923, %v4924 : tensor<64xf32>
    %v4926 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4927 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4928 = stablehlo.multiply %v4926, %s1b1g1v : tensor<64xf32>
    %v4929 = stablehlo.multiply %armeans1b1g1, %armeans1b1g1 : tensor<64xf32>
    %v4930 = stablehlo.multiply %v4927, %v4929 : tensor<64xf32>
    %v4931 = stablehlo.add %v4928, %v4930 : tensor<64xf32>
    %v4932 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4933 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4934 = stablehlo.divide %v4925, %v4932 : tensor<64xf32>
    %v4935 = stablehlo.divide %v4931, %v4933 : tensor<64xf32>
    %v4936 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4937 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4938 = stablehlo.sqrt %v4935 : tensor<64xf32>
    %v4939 = stablehlo.add %v4938, %v4937 : tensor<64xf32>
    %v4940 = stablehlo.divide %v4934, %v4939 : tensor<64xf32>
    %v4941 = stablehlo.multiply %v4936, %v4940 : tensor<64xf32>
    %v4942 = stablehlo.subtract %s1b1g1, %v4941 : tensor<64xf32>
    %v4943 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4944 = stablehlo.multiply %v4943, %v4936 : tensor<64xf32>
    %v4945 = stablehlo.multiply %v4944, %s1b1g1 : tensor<64xf32>
    %v4946 = stablehlo.subtract %v4942, %v4945 : tensor<64xf32>
    %arsums1b1bt1 = "stablehlo.all_reduce"(%v4204) ({
    ^bb0(%aras1b1bt1: tensor<f32>, %arbs1b1bt1: tensor<f32>):
      %aradds1b1bt1 = stablehlo.add %aras1b1bt1, %arbs1b1bt1 : tensor<f32>
      stablehlo.return %aradds1b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1bt1 = stablehlo.divide %arsums1b1bt1, %arns1b1bt1 : tensor<64xf32>
    %v4947 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4948 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4949 = stablehlo.multiply %v4947, %s1b1bt1m : tensor<64xf32>
    %v4950 = stablehlo.multiply %v4948, %armeans1b1bt1 : tensor<64xf32>
    %v4951 = stablehlo.add %v4949, %v4950 : tensor<64xf32>
    %v4952 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4953 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4954 = stablehlo.multiply %v4952, %s1b1bt1v : tensor<64xf32>
    %v4955 = stablehlo.multiply %armeans1b1bt1, %armeans1b1bt1 : tensor<64xf32>
    %v4956 = stablehlo.multiply %v4953, %v4955 : tensor<64xf32>
    %v4957 = stablehlo.add %v4954, %v4956 : tensor<64xf32>
    %v4958 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4959 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4960 = stablehlo.multiply %v4958, %s1b1bt1m : tensor<64xf32>
    %v4961 = stablehlo.multiply %v4959, %armeans1b1bt1 : tensor<64xf32>
    %v4962 = stablehlo.add %v4960, %v4961 : tensor<64xf32>
    %v4963 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4964 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4965 = stablehlo.multiply %v4963, %s1b1bt1v : tensor<64xf32>
    %v4966 = stablehlo.multiply %armeans1b1bt1, %armeans1b1bt1 : tensor<64xf32>
    %v4967 = stablehlo.multiply %v4964, %v4966 : tensor<64xf32>
    %v4968 = stablehlo.add %v4965, %v4967 : tensor<64xf32>
    %v4969 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4970 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4971 = stablehlo.divide %v4962, %v4969 : tensor<64xf32>
    %v4972 = stablehlo.divide %v4968, %v4970 : tensor<64xf32>
    %v4973 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4974 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4975 = stablehlo.sqrt %v4972 : tensor<64xf32>
    %v4976 = stablehlo.add %v4975, %v4974 : tensor<64xf32>
    %v4977 = stablehlo.divide %v4971, %v4976 : tensor<64xf32>
    %v4978 = stablehlo.multiply %v4973, %v4977 : tensor<64xf32>
    %v4979 = stablehlo.subtract %s1b1bt1, %v4978 : tensor<64xf32>
    %v4980 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4981 = stablehlo.multiply %v4980, %v4973 : tensor<64xf32>
    %v4982 = stablehlo.multiply %v4981, %s1b1bt1 : tensor<64xf32>
    %v4983 = stablehlo.subtract %v4979, %v4982 : tensor<64xf32>
    %arsums1b1W2 = "stablehlo.all_reduce"(%v4210) ({
    ^bb0(%aras1b1W2: tensor<f32>, %arbs1b1W2: tensor<f32>):
      %aradds1b1W2 = stablehlo.add %aras1b1W2, %arbs1b1W2 : tensor<f32>
      stablehlo.return %aradds1b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W2 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b1W2 = stablehlo.divide %arsums1b1W2, %arns1b1W2 : tensor<64x64x3x3xf32>
    %v4984 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4985 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4986 = stablehlo.multiply %v4984, %s1b1W2m : tensor<64x64x3x3xf32>
    %v4987 = stablehlo.multiply %v4985, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v4988 = stablehlo.add %v4986, %v4987 : tensor<64x64x3x3xf32>
    %v4989 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4990 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4991 = stablehlo.multiply %v4989, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4992 = stablehlo.multiply %armeans1b1W2, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v4993 = stablehlo.multiply %v4990, %v4992 : tensor<64x64x3x3xf32>
    %v4994 = stablehlo.add %v4991, %v4993 : tensor<64x64x3x3xf32>
    %v4995 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4996 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4997 = stablehlo.multiply %v4995, %s1b1W2m : tensor<64x64x3x3xf32>
    %v4998 = stablehlo.multiply %v4996, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v4999 = stablehlo.add %v4997, %v4998 : tensor<64x64x3x3xf32>
    %v5000 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5001 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5002 = stablehlo.multiply %v5000, %s1b1W2v : tensor<64x64x3x3xf32>
    %v5003 = stablehlo.multiply %armeans1b1W2, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v5004 = stablehlo.multiply %v5001, %v5003 : tensor<64x64x3x3xf32>
    %v5005 = stablehlo.add %v5002, %v5004 : tensor<64x64x3x3xf32>
    %v5006 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5007 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5008 = stablehlo.divide %v4999, %v5006 : tensor<64x64x3x3xf32>
    %v5009 = stablehlo.divide %v5005, %v5007 : tensor<64x64x3x3xf32>
    %v5010 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5011 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5012 = stablehlo.sqrt %v5009 : tensor<64x64x3x3xf32>
    %v5013 = stablehlo.add %v5012, %v5011 : tensor<64x64x3x3xf32>
    %v5014 = stablehlo.divide %v5008, %v5013 : tensor<64x64x3x3xf32>
    %v5015 = stablehlo.multiply %v5010, %v5014 : tensor<64x64x3x3xf32>
    %v5016 = stablehlo.subtract %s1b1W2, %v5015 : tensor<64x64x3x3xf32>
    %v5017 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5018 = stablehlo.multiply %v5017, %v5010 : tensor<64x64x3x3xf32>
    %v5019 = stablehlo.multiply %v5018, %s1b1W2 : tensor<64x64x3x3xf32>
    %v5020 = stablehlo.subtract %v5016, %v5019 : tensor<64x64x3x3xf32>
    %arsums1b1g2 = "stablehlo.all_reduce"(%v4224) ({
    ^bb0(%aras1b1g2: tensor<f32>, %arbs1b1g2: tensor<f32>):
      %aradds1b1g2 = stablehlo.add %aras1b1g2, %arbs1b1g2 : tensor<f32>
      stablehlo.return %aradds1b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g2 = stablehlo.divide %arsums1b1g2, %arns1b1g2 : tensor<64xf32>
    %v5021 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5022 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5023 = stablehlo.multiply %v5021, %s1b1g2m : tensor<64xf32>
    %v5024 = stablehlo.multiply %v5022, %armeans1b1g2 : tensor<64xf32>
    %v5025 = stablehlo.add %v5023, %v5024 : tensor<64xf32>
    %v5026 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5027 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5028 = stablehlo.multiply %v5026, %s1b1g2v : tensor<64xf32>
    %v5029 = stablehlo.multiply %armeans1b1g2, %armeans1b1g2 : tensor<64xf32>
    %v5030 = stablehlo.multiply %v5027, %v5029 : tensor<64xf32>
    %v5031 = stablehlo.add %v5028, %v5030 : tensor<64xf32>
    %v5032 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5033 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5034 = stablehlo.multiply %v5032, %s1b1g2m : tensor<64xf32>
    %v5035 = stablehlo.multiply %v5033, %armeans1b1g2 : tensor<64xf32>
    %v5036 = stablehlo.add %v5034, %v5035 : tensor<64xf32>
    %v5037 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5038 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5039 = stablehlo.multiply %v5037, %s1b1g2v : tensor<64xf32>
    %v5040 = stablehlo.multiply %armeans1b1g2, %armeans1b1g2 : tensor<64xf32>
    %v5041 = stablehlo.multiply %v5038, %v5040 : tensor<64xf32>
    %v5042 = stablehlo.add %v5039, %v5041 : tensor<64xf32>
    %v5043 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5044 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5045 = stablehlo.divide %v5036, %v5043 : tensor<64xf32>
    %v5046 = stablehlo.divide %v5042, %v5044 : tensor<64xf32>
    %v5047 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5048 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5049 = stablehlo.sqrt %v5046 : tensor<64xf32>
    %v5050 = stablehlo.add %v5049, %v5048 : tensor<64xf32>
    %v5051 = stablehlo.divide %v5045, %v5050 : tensor<64xf32>
    %v5052 = stablehlo.multiply %v5047, %v5051 : tensor<64xf32>
    %v5053 = stablehlo.subtract %s1b1g2, %v5052 : tensor<64xf32>
    %v5054 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5055 = stablehlo.multiply %v5054, %v5047 : tensor<64xf32>
    %v5056 = stablehlo.multiply %v5055, %s1b1g2 : tensor<64xf32>
    %v5057 = stablehlo.subtract %v5053, %v5056 : tensor<64xf32>
    %arsums1b1bt2 = "stablehlo.all_reduce"(%v4227) ({
    ^bb0(%aras1b1bt2: tensor<f32>, %arbs1b1bt2: tensor<f32>):
      %aradds1b1bt2 = stablehlo.add %aras1b1bt2, %arbs1b1bt2 : tensor<f32>
      stablehlo.return %aradds1b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1bt2 = stablehlo.divide %arsums1b1bt2, %arns1b1bt2 : tensor<64xf32>
    %v5058 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5059 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5060 = stablehlo.multiply %v5058, %s1b1bt2m : tensor<64xf32>
    %v5061 = stablehlo.multiply %v5059, %armeans1b1bt2 : tensor<64xf32>
    %v5062 = stablehlo.add %v5060, %v5061 : tensor<64xf32>
    %v5063 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5064 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5065 = stablehlo.multiply %v5063, %s1b1bt2v : tensor<64xf32>
    %v5066 = stablehlo.multiply %armeans1b1bt2, %armeans1b1bt2 : tensor<64xf32>
    %v5067 = stablehlo.multiply %v5064, %v5066 : tensor<64xf32>
    %v5068 = stablehlo.add %v5065, %v5067 : tensor<64xf32>
    %v5069 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5070 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5071 = stablehlo.multiply %v5069, %s1b1bt2m : tensor<64xf32>
    %v5072 = stablehlo.multiply %v5070, %armeans1b1bt2 : tensor<64xf32>
    %v5073 = stablehlo.add %v5071, %v5072 : tensor<64xf32>
    %v5074 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5075 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5076 = stablehlo.multiply %v5074, %s1b1bt2v : tensor<64xf32>
    %v5077 = stablehlo.multiply %armeans1b1bt2, %armeans1b1bt2 : tensor<64xf32>
    %v5078 = stablehlo.multiply %v5075, %v5077 : tensor<64xf32>
    %v5079 = stablehlo.add %v5076, %v5078 : tensor<64xf32>
    %v5080 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5081 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5082 = stablehlo.divide %v5073, %v5080 : tensor<64xf32>
    %v5083 = stablehlo.divide %v5079, %v5081 : tensor<64xf32>
    %v5084 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5085 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5086 = stablehlo.sqrt %v5083 : tensor<64xf32>
    %v5087 = stablehlo.add %v5086, %v5085 : tensor<64xf32>
    %v5088 = stablehlo.divide %v5082, %v5087 : tensor<64xf32>
    %v5089 = stablehlo.multiply %v5084, %v5088 : tensor<64xf32>
    %v5090 = stablehlo.subtract %s1b1bt2, %v5089 : tensor<64xf32>
    %v5091 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5092 = stablehlo.multiply %v5091, %v5084 : tensor<64xf32>
    %v5093 = stablehlo.multiply %v5092, %s1b1bt2 : tensor<64xf32>
    %v5094 = stablehlo.subtract %v5090, %v5093 : tensor<64xf32>
    %arsums1b2W1 = "stablehlo.all_reduce"(%v4027) ({
    ^bb0(%aras1b2W1: tensor<f32>, %arbs1b2W1: tensor<f32>):
      %aradds1b2W1 = stablehlo.add %aras1b2W1, %arbs1b2W1 : tensor<f32>
      stablehlo.return %aradds1b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W1 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b2W1 = stablehlo.divide %arsums1b2W1, %arns1b2W1 : tensor<64x64x3x3xf32>
    %v5095 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5096 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5097 = stablehlo.multiply %v5095, %s1b2W1m : tensor<64x64x3x3xf32>
    %v5098 = stablehlo.multiply %v5096, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v5099 = stablehlo.add %v5097, %v5098 : tensor<64x64x3x3xf32>
    %v5100 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5101 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5102 = stablehlo.multiply %v5100, %s1b2W1v : tensor<64x64x3x3xf32>
    %v5103 = stablehlo.multiply %armeans1b2W1, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v5104 = stablehlo.multiply %v5101, %v5103 : tensor<64x64x3x3xf32>
    %v5105 = stablehlo.add %v5102, %v5104 : tensor<64x64x3x3xf32>
    %v5106 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5107 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5108 = stablehlo.multiply %v5106, %s1b2W1m : tensor<64x64x3x3xf32>
    %v5109 = stablehlo.multiply %v5107, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v5110 = stablehlo.add %v5108, %v5109 : tensor<64x64x3x3xf32>
    %v5111 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5112 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5113 = stablehlo.multiply %v5111, %s1b2W1v : tensor<64x64x3x3xf32>
    %v5114 = stablehlo.multiply %armeans1b2W1, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v5115 = stablehlo.multiply %v5112, %v5114 : tensor<64x64x3x3xf32>
    %v5116 = stablehlo.add %v5113, %v5115 : tensor<64x64x3x3xf32>
    %v5117 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5118 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5119 = stablehlo.divide %v5110, %v5117 : tensor<64x64x3x3xf32>
    %v5120 = stablehlo.divide %v5116, %v5118 : tensor<64x64x3x3xf32>
    %v5121 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5122 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5123 = stablehlo.sqrt %v5120 : tensor<64x64x3x3xf32>
    %v5124 = stablehlo.add %v5123, %v5122 : tensor<64x64x3x3xf32>
    %v5125 = stablehlo.divide %v5119, %v5124 : tensor<64x64x3x3xf32>
    %v5126 = stablehlo.multiply %v5121, %v5125 : tensor<64x64x3x3xf32>
    %v5127 = stablehlo.subtract %s1b2W1, %v5126 : tensor<64x64x3x3xf32>
    %v5128 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5129 = stablehlo.multiply %v5128, %v5121 : tensor<64x64x3x3xf32>
    %v5130 = stablehlo.multiply %v5129, %s1b2W1 : tensor<64x64x3x3xf32>
    %v5131 = stablehlo.subtract %v5127, %v5130 : tensor<64x64x3x3xf32>
    %arsums1b2g1 = "stablehlo.all_reduce"(%v4041) ({
    ^bb0(%aras1b2g1: tensor<f32>, %arbs1b2g1: tensor<f32>):
      %aradds1b2g1 = stablehlo.add %aras1b2g1, %arbs1b2g1 : tensor<f32>
      stablehlo.return %aradds1b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g1 = stablehlo.divide %arsums1b2g1, %arns1b2g1 : tensor<64xf32>
    %v5132 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5133 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5134 = stablehlo.multiply %v5132, %s1b2g1m : tensor<64xf32>
    %v5135 = stablehlo.multiply %v5133, %armeans1b2g1 : tensor<64xf32>
    %v5136 = stablehlo.add %v5134, %v5135 : tensor<64xf32>
    %v5137 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5138 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5139 = stablehlo.multiply %v5137, %s1b2g1v : tensor<64xf32>
    %v5140 = stablehlo.multiply %armeans1b2g1, %armeans1b2g1 : tensor<64xf32>
    %v5141 = stablehlo.multiply %v5138, %v5140 : tensor<64xf32>
    %v5142 = stablehlo.add %v5139, %v5141 : tensor<64xf32>
    %v5143 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5144 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5145 = stablehlo.multiply %v5143, %s1b2g1m : tensor<64xf32>
    %v5146 = stablehlo.multiply %v5144, %armeans1b2g1 : tensor<64xf32>
    %v5147 = stablehlo.add %v5145, %v5146 : tensor<64xf32>
    %v5148 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5149 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5150 = stablehlo.multiply %v5148, %s1b2g1v : tensor<64xf32>
    %v5151 = stablehlo.multiply %armeans1b2g1, %armeans1b2g1 : tensor<64xf32>
    %v5152 = stablehlo.multiply %v5149, %v5151 : tensor<64xf32>
    %v5153 = stablehlo.add %v5150, %v5152 : tensor<64xf32>
    %v5154 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5155 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5156 = stablehlo.divide %v5147, %v5154 : tensor<64xf32>
    %v5157 = stablehlo.divide %v5153, %v5155 : tensor<64xf32>
    %v5158 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5159 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5160 = stablehlo.sqrt %v5157 : tensor<64xf32>
    %v5161 = stablehlo.add %v5160, %v5159 : tensor<64xf32>
    %v5162 = stablehlo.divide %v5156, %v5161 : tensor<64xf32>
    %v5163 = stablehlo.multiply %v5158, %v5162 : tensor<64xf32>
    %v5164 = stablehlo.subtract %s1b2g1, %v5163 : tensor<64xf32>
    %v5165 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5166 = stablehlo.multiply %v5165, %v5158 : tensor<64xf32>
    %v5167 = stablehlo.multiply %v5166, %s1b2g1 : tensor<64xf32>
    %v5168 = stablehlo.subtract %v5164, %v5167 : tensor<64xf32>
    %arsums1b2bt1 = "stablehlo.all_reduce"(%v4044) ({
    ^bb0(%aras1b2bt1: tensor<f32>, %arbs1b2bt1: tensor<f32>):
      %aradds1b2bt1 = stablehlo.add %aras1b2bt1, %arbs1b2bt1 : tensor<f32>
      stablehlo.return %aradds1b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2bt1 = stablehlo.divide %arsums1b2bt1, %arns1b2bt1 : tensor<64xf32>
    %v5169 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5170 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5171 = stablehlo.multiply %v5169, %s1b2bt1m : tensor<64xf32>
    %v5172 = stablehlo.multiply %v5170, %armeans1b2bt1 : tensor<64xf32>
    %v5173 = stablehlo.add %v5171, %v5172 : tensor<64xf32>
    %v5174 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5175 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5176 = stablehlo.multiply %v5174, %s1b2bt1v : tensor<64xf32>
    %v5177 = stablehlo.multiply %armeans1b2bt1, %armeans1b2bt1 : tensor<64xf32>
    %v5178 = stablehlo.multiply %v5175, %v5177 : tensor<64xf32>
    %v5179 = stablehlo.add %v5176, %v5178 : tensor<64xf32>
    %v5180 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5181 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5182 = stablehlo.multiply %v5180, %s1b2bt1m : tensor<64xf32>
    %v5183 = stablehlo.multiply %v5181, %armeans1b2bt1 : tensor<64xf32>
    %v5184 = stablehlo.add %v5182, %v5183 : tensor<64xf32>
    %v5185 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5186 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5187 = stablehlo.multiply %v5185, %s1b2bt1v : tensor<64xf32>
    %v5188 = stablehlo.multiply %armeans1b2bt1, %armeans1b2bt1 : tensor<64xf32>
    %v5189 = stablehlo.multiply %v5186, %v5188 : tensor<64xf32>
    %v5190 = stablehlo.add %v5187, %v5189 : tensor<64xf32>
    %v5191 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5192 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5193 = stablehlo.divide %v5184, %v5191 : tensor<64xf32>
    %v5194 = stablehlo.divide %v5190, %v5192 : tensor<64xf32>
    %v5195 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5196 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5197 = stablehlo.sqrt %v5194 : tensor<64xf32>
    %v5198 = stablehlo.add %v5197, %v5196 : tensor<64xf32>
    %v5199 = stablehlo.divide %v5193, %v5198 : tensor<64xf32>
    %v5200 = stablehlo.multiply %v5195, %v5199 : tensor<64xf32>
    %v5201 = stablehlo.subtract %s1b2bt1, %v5200 : tensor<64xf32>
    %v5202 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5203 = stablehlo.multiply %v5202, %v5195 : tensor<64xf32>
    %v5204 = stablehlo.multiply %v5203, %s1b2bt1 : tensor<64xf32>
    %v5205 = stablehlo.subtract %v5201, %v5204 : tensor<64xf32>
    %arsums1b2W2 = "stablehlo.all_reduce"(%v4050) ({
    ^bb0(%aras1b2W2: tensor<f32>, %arbs1b2W2: tensor<f32>):
      %aradds1b2W2 = stablehlo.add %aras1b2W2, %arbs1b2W2 : tensor<f32>
      stablehlo.return %aradds1b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W2 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b2W2 = stablehlo.divide %arsums1b2W2, %arns1b2W2 : tensor<64x64x3x3xf32>
    %v5206 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5207 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5208 = stablehlo.multiply %v5206, %s1b2W2m : tensor<64x64x3x3xf32>
    %v5209 = stablehlo.multiply %v5207, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v5210 = stablehlo.add %v5208, %v5209 : tensor<64x64x3x3xf32>
    %v5211 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5212 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5213 = stablehlo.multiply %v5211, %s1b2W2v : tensor<64x64x3x3xf32>
    %v5214 = stablehlo.multiply %armeans1b2W2, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v5215 = stablehlo.multiply %v5212, %v5214 : tensor<64x64x3x3xf32>
    %v5216 = stablehlo.add %v5213, %v5215 : tensor<64x64x3x3xf32>
    %v5217 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5218 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5219 = stablehlo.multiply %v5217, %s1b2W2m : tensor<64x64x3x3xf32>
    %v5220 = stablehlo.multiply %v5218, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v5221 = stablehlo.add %v5219, %v5220 : tensor<64x64x3x3xf32>
    %v5222 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5223 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5224 = stablehlo.multiply %v5222, %s1b2W2v : tensor<64x64x3x3xf32>
    %v5225 = stablehlo.multiply %armeans1b2W2, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v5226 = stablehlo.multiply %v5223, %v5225 : tensor<64x64x3x3xf32>
    %v5227 = stablehlo.add %v5224, %v5226 : tensor<64x64x3x3xf32>
    %v5228 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5229 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5230 = stablehlo.divide %v5221, %v5228 : tensor<64x64x3x3xf32>
    %v5231 = stablehlo.divide %v5227, %v5229 : tensor<64x64x3x3xf32>
    %v5232 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5233 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5234 = stablehlo.sqrt %v5231 : tensor<64x64x3x3xf32>
    %v5235 = stablehlo.add %v5234, %v5233 : tensor<64x64x3x3xf32>
    %v5236 = stablehlo.divide %v5230, %v5235 : tensor<64x64x3x3xf32>
    %v5237 = stablehlo.multiply %v5232, %v5236 : tensor<64x64x3x3xf32>
    %v5238 = stablehlo.subtract %s1b2W2, %v5237 : tensor<64x64x3x3xf32>
    %v5239 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5240 = stablehlo.multiply %v5239, %v5232 : tensor<64x64x3x3xf32>
    %v5241 = stablehlo.multiply %v5240, %s1b2W2 : tensor<64x64x3x3xf32>
    %v5242 = stablehlo.subtract %v5238, %v5241 : tensor<64x64x3x3xf32>
    %arsums1b2g2 = "stablehlo.all_reduce"(%v4064) ({
    ^bb0(%aras1b2g2: tensor<f32>, %arbs1b2g2: tensor<f32>):
      %aradds1b2g2 = stablehlo.add %aras1b2g2, %arbs1b2g2 : tensor<f32>
      stablehlo.return %aradds1b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g2 = stablehlo.divide %arsums1b2g2, %arns1b2g2 : tensor<64xf32>
    %v5243 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5244 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5245 = stablehlo.multiply %v5243, %s1b2g2m : tensor<64xf32>
    %v5246 = stablehlo.multiply %v5244, %armeans1b2g2 : tensor<64xf32>
    %v5247 = stablehlo.add %v5245, %v5246 : tensor<64xf32>
    %v5248 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5249 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5250 = stablehlo.multiply %v5248, %s1b2g2v : tensor<64xf32>
    %v5251 = stablehlo.multiply %armeans1b2g2, %armeans1b2g2 : tensor<64xf32>
    %v5252 = stablehlo.multiply %v5249, %v5251 : tensor<64xf32>
    %v5253 = stablehlo.add %v5250, %v5252 : tensor<64xf32>
    %v5254 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5255 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5256 = stablehlo.multiply %v5254, %s1b2g2m : tensor<64xf32>
    %v5257 = stablehlo.multiply %v5255, %armeans1b2g2 : tensor<64xf32>
    %v5258 = stablehlo.add %v5256, %v5257 : tensor<64xf32>
    %v5259 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5260 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5261 = stablehlo.multiply %v5259, %s1b2g2v : tensor<64xf32>
    %v5262 = stablehlo.multiply %armeans1b2g2, %armeans1b2g2 : tensor<64xf32>
    %v5263 = stablehlo.multiply %v5260, %v5262 : tensor<64xf32>
    %v5264 = stablehlo.add %v5261, %v5263 : tensor<64xf32>
    %v5265 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5266 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5267 = stablehlo.divide %v5258, %v5265 : tensor<64xf32>
    %v5268 = stablehlo.divide %v5264, %v5266 : tensor<64xf32>
    %v5269 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5270 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5271 = stablehlo.sqrt %v5268 : tensor<64xf32>
    %v5272 = stablehlo.add %v5271, %v5270 : tensor<64xf32>
    %v5273 = stablehlo.divide %v5267, %v5272 : tensor<64xf32>
    %v5274 = stablehlo.multiply %v5269, %v5273 : tensor<64xf32>
    %v5275 = stablehlo.subtract %s1b2g2, %v5274 : tensor<64xf32>
    %v5276 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5277 = stablehlo.multiply %v5276, %v5269 : tensor<64xf32>
    %v5278 = stablehlo.multiply %v5277, %s1b2g2 : tensor<64xf32>
    %v5279 = stablehlo.subtract %v5275, %v5278 : tensor<64xf32>
    %arsums1b2bt2 = "stablehlo.all_reduce"(%v4067) ({
    ^bb0(%aras1b2bt2: tensor<f32>, %arbs1b2bt2: tensor<f32>):
      %aradds1b2bt2 = stablehlo.add %aras1b2bt2, %arbs1b2bt2 : tensor<f32>
      stablehlo.return %aradds1b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2bt2 = stablehlo.divide %arsums1b2bt2, %arns1b2bt2 : tensor<64xf32>
    %v5280 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5281 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5282 = stablehlo.multiply %v5280, %s1b2bt2m : tensor<64xf32>
    %v5283 = stablehlo.multiply %v5281, %armeans1b2bt2 : tensor<64xf32>
    %v5284 = stablehlo.add %v5282, %v5283 : tensor<64xf32>
    %v5285 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5286 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5287 = stablehlo.multiply %v5285, %s1b2bt2v : tensor<64xf32>
    %v5288 = stablehlo.multiply %armeans1b2bt2, %armeans1b2bt2 : tensor<64xf32>
    %v5289 = stablehlo.multiply %v5286, %v5288 : tensor<64xf32>
    %v5290 = stablehlo.add %v5287, %v5289 : tensor<64xf32>
    %v5291 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5292 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5293 = stablehlo.multiply %v5291, %s1b2bt2m : tensor<64xf32>
    %v5294 = stablehlo.multiply %v5292, %armeans1b2bt2 : tensor<64xf32>
    %v5295 = stablehlo.add %v5293, %v5294 : tensor<64xf32>
    %v5296 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5297 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5298 = stablehlo.multiply %v5296, %s1b2bt2v : tensor<64xf32>
    %v5299 = stablehlo.multiply %armeans1b2bt2, %armeans1b2bt2 : tensor<64xf32>
    %v5300 = stablehlo.multiply %v5297, %v5299 : tensor<64xf32>
    %v5301 = stablehlo.add %v5298, %v5300 : tensor<64xf32>
    %v5302 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5303 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5304 = stablehlo.divide %v5295, %v5302 : tensor<64xf32>
    %v5305 = stablehlo.divide %v5301, %v5303 : tensor<64xf32>
    %v5306 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5307 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5308 = stablehlo.sqrt %v5305 : tensor<64xf32>
    %v5309 = stablehlo.add %v5308, %v5307 : tensor<64xf32>
    %v5310 = stablehlo.divide %v5304, %v5309 : tensor<64xf32>
    %v5311 = stablehlo.multiply %v5306, %v5310 : tensor<64xf32>
    %v5312 = stablehlo.subtract %s1b2bt2, %v5311 : tensor<64xf32>
    %v5313 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5314 = stablehlo.multiply %v5313, %v5306 : tensor<64xf32>
    %v5315 = stablehlo.multiply %v5314, %s1b2bt2 : tensor<64xf32>
    %v5316 = stablehlo.subtract %v5312, %v5315 : tensor<64xf32>
    %arsumd2W1 = "stablehlo.all_reduce"(%v3842) ({
    ^bb0(%arad2W1: tensor<f32>, %arbd2W1: tensor<f32>):
      %araddd2W1 = stablehlo.add %arad2W1, %arbd2W1 : tensor<f32>
      stablehlo.return %araddd2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf32>
    %arnd2W1 = stablehlo.constant dense<2.0> : tensor<128x64x3x3xf32>
    %armeand2W1 = stablehlo.divide %arsumd2W1, %arnd2W1 : tensor<128x64x3x3xf32>
    %v5317 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5318 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5319 = stablehlo.multiply %v5317, %d2W1m : tensor<128x64x3x3xf32>
    %v5320 = stablehlo.multiply %v5318, %armeand2W1 : tensor<128x64x3x3xf32>
    %v5321 = stablehlo.add %v5319, %v5320 : tensor<128x64x3x3xf32>
    %v5322 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5323 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5324 = stablehlo.multiply %v5322, %d2W1v : tensor<128x64x3x3xf32>
    %v5325 = stablehlo.multiply %armeand2W1, %armeand2W1 : tensor<128x64x3x3xf32>
    %v5326 = stablehlo.multiply %v5323, %v5325 : tensor<128x64x3x3xf32>
    %v5327 = stablehlo.add %v5324, %v5326 : tensor<128x64x3x3xf32>
    %v5328 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5329 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5330 = stablehlo.multiply %v5328, %d2W1m : tensor<128x64x3x3xf32>
    %v5331 = stablehlo.multiply %v5329, %armeand2W1 : tensor<128x64x3x3xf32>
    %v5332 = stablehlo.add %v5330, %v5331 : tensor<128x64x3x3xf32>
    %v5333 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5334 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5335 = stablehlo.multiply %v5333, %d2W1v : tensor<128x64x3x3xf32>
    %v5336 = stablehlo.multiply %armeand2W1, %armeand2W1 : tensor<128x64x3x3xf32>
    %v5337 = stablehlo.multiply %v5334, %v5336 : tensor<128x64x3x3xf32>
    %v5338 = stablehlo.add %v5335, %v5337 : tensor<128x64x3x3xf32>
    %v5339 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5340 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5341 = stablehlo.divide %v5332, %v5339 : tensor<128x64x3x3xf32>
    %v5342 = stablehlo.divide %v5338, %v5340 : tensor<128x64x3x3xf32>
    %v5343 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5344 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5345 = stablehlo.sqrt %v5342 : tensor<128x64x3x3xf32>
    %v5346 = stablehlo.add %v5345, %v5344 : tensor<128x64x3x3xf32>
    %v5347 = stablehlo.divide %v5341, %v5346 : tensor<128x64x3x3xf32>
    %v5348 = stablehlo.multiply %v5343, %v5347 : tensor<128x64x3x3xf32>
    %v5349 = stablehlo.subtract %d2W1, %v5348 : tensor<128x64x3x3xf32>
    %v5350 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5351 = stablehlo.multiply %v5350, %v5343 : tensor<128x64x3x3xf32>
    %v5352 = stablehlo.multiply %v5351, %d2W1 : tensor<128x64x3x3xf32>
    %v5353 = stablehlo.subtract %v5349, %v5352 : tensor<128x64x3x3xf32>
    %arsumd2g1 = "stablehlo.all_reduce"(%v3856) ({
    ^bb0(%arad2g1: tensor<f32>, %arbd2g1: tensor<f32>):
      %araddd2g1 = stablehlo.add %arad2g1, %arbd2g1 : tensor<f32>
      stablehlo.return %araddd2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g1 = stablehlo.divide %arsumd2g1, %arnd2g1 : tensor<128xf32>
    %v5354 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5355 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5356 = stablehlo.multiply %v5354, %d2g1m : tensor<128xf32>
    %v5357 = stablehlo.multiply %v5355, %armeand2g1 : tensor<128xf32>
    %v5358 = stablehlo.add %v5356, %v5357 : tensor<128xf32>
    %v5359 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5360 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5361 = stablehlo.multiply %v5359, %d2g1v : tensor<128xf32>
    %v5362 = stablehlo.multiply %armeand2g1, %armeand2g1 : tensor<128xf32>
    %v5363 = stablehlo.multiply %v5360, %v5362 : tensor<128xf32>
    %v5364 = stablehlo.add %v5361, %v5363 : tensor<128xf32>
    %v5365 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5366 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5367 = stablehlo.multiply %v5365, %d2g1m : tensor<128xf32>
    %v5368 = stablehlo.multiply %v5366, %armeand2g1 : tensor<128xf32>
    %v5369 = stablehlo.add %v5367, %v5368 : tensor<128xf32>
    %v5370 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5371 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5372 = stablehlo.multiply %v5370, %d2g1v : tensor<128xf32>
    %v5373 = stablehlo.multiply %armeand2g1, %armeand2g1 : tensor<128xf32>
    %v5374 = stablehlo.multiply %v5371, %v5373 : tensor<128xf32>
    %v5375 = stablehlo.add %v5372, %v5374 : tensor<128xf32>
    %v5376 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5377 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5378 = stablehlo.divide %v5369, %v5376 : tensor<128xf32>
    %v5379 = stablehlo.divide %v5375, %v5377 : tensor<128xf32>
    %v5380 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5381 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5382 = stablehlo.sqrt %v5379 : tensor<128xf32>
    %v5383 = stablehlo.add %v5382, %v5381 : tensor<128xf32>
    %v5384 = stablehlo.divide %v5378, %v5383 : tensor<128xf32>
    %v5385 = stablehlo.multiply %v5380, %v5384 : tensor<128xf32>
    %v5386 = stablehlo.subtract %d2g1, %v5385 : tensor<128xf32>
    %v5387 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5388 = stablehlo.multiply %v5387, %v5380 : tensor<128xf32>
    %v5389 = stablehlo.multiply %v5388, %d2g1 : tensor<128xf32>
    %v5390 = stablehlo.subtract %v5386, %v5389 : tensor<128xf32>
    %arsumd2bt1 = "stablehlo.all_reduce"(%v3859) ({
    ^bb0(%arad2bt1: tensor<f32>, %arbd2bt1: tensor<f32>):
      %araddd2bt1 = stablehlo.add %arad2bt1, %arbd2bt1 : tensor<f32>
      stablehlo.return %araddd2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2bt1 = stablehlo.divide %arsumd2bt1, %arnd2bt1 : tensor<128xf32>
    %v5391 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5392 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5393 = stablehlo.multiply %v5391, %d2bt1m : tensor<128xf32>
    %v5394 = stablehlo.multiply %v5392, %armeand2bt1 : tensor<128xf32>
    %v5395 = stablehlo.add %v5393, %v5394 : tensor<128xf32>
    %v5396 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5397 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5398 = stablehlo.multiply %v5396, %d2bt1v : tensor<128xf32>
    %v5399 = stablehlo.multiply %armeand2bt1, %armeand2bt1 : tensor<128xf32>
    %v5400 = stablehlo.multiply %v5397, %v5399 : tensor<128xf32>
    %v5401 = stablehlo.add %v5398, %v5400 : tensor<128xf32>
    %v5402 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5403 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5404 = stablehlo.multiply %v5402, %d2bt1m : tensor<128xf32>
    %v5405 = stablehlo.multiply %v5403, %armeand2bt1 : tensor<128xf32>
    %v5406 = stablehlo.add %v5404, %v5405 : tensor<128xf32>
    %v5407 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5408 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5409 = stablehlo.multiply %v5407, %d2bt1v : tensor<128xf32>
    %v5410 = stablehlo.multiply %armeand2bt1, %armeand2bt1 : tensor<128xf32>
    %v5411 = stablehlo.multiply %v5408, %v5410 : tensor<128xf32>
    %v5412 = stablehlo.add %v5409, %v5411 : tensor<128xf32>
    %v5413 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5414 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5415 = stablehlo.divide %v5406, %v5413 : tensor<128xf32>
    %v5416 = stablehlo.divide %v5412, %v5414 : tensor<128xf32>
    %v5417 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5418 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5419 = stablehlo.sqrt %v5416 : tensor<128xf32>
    %v5420 = stablehlo.add %v5419, %v5418 : tensor<128xf32>
    %v5421 = stablehlo.divide %v5415, %v5420 : tensor<128xf32>
    %v5422 = stablehlo.multiply %v5417, %v5421 : tensor<128xf32>
    %v5423 = stablehlo.subtract %d2bt1, %v5422 : tensor<128xf32>
    %v5424 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5425 = stablehlo.multiply %v5424, %v5417 : tensor<128xf32>
    %v5426 = stablehlo.multiply %v5425, %d2bt1 : tensor<128xf32>
    %v5427 = stablehlo.subtract %v5423, %v5426 : tensor<128xf32>
    %arsumd2W2 = "stablehlo.all_reduce"(%v3865) ({
    ^bb0(%arad2W2: tensor<f32>, %arbd2W2: tensor<f32>):
      %araddd2W2 = stablehlo.add %arad2W2, %arbd2W2 : tensor<f32>
      stablehlo.return %araddd2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arnd2W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeand2W2 = stablehlo.divide %arsumd2W2, %arnd2W2 : tensor<128x128x3x3xf32>
    %v5428 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5429 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5430 = stablehlo.multiply %v5428, %d2W2m : tensor<128x128x3x3xf32>
    %v5431 = stablehlo.multiply %v5429, %armeand2W2 : tensor<128x128x3x3xf32>
    %v5432 = stablehlo.add %v5430, %v5431 : tensor<128x128x3x3xf32>
    %v5433 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5434 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5435 = stablehlo.multiply %v5433, %d2W2v : tensor<128x128x3x3xf32>
    %v5436 = stablehlo.multiply %armeand2W2, %armeand2W2 : tensor<128x128x3x3xf32>
    %v5437 = stablehlo.multiply %v5434, %v5436 : tensor<128x128x3x3xf32>
    %v5438 = stablehlo.add %v5435, %v5437 : tensor<128x128x3x3xf32>
    %v5439 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5440 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5441 = stablehlo.multiply %v5439, %d2W2m : tensor<128x128x3x3xf32>
    %v5442 = stablehlo.multiply %v5440, %armeand2W2 : tensor<128x128x3x3xf32>
    %v5443 = stablehlo.add %v5441, %v5442 : tensor<128x128x3x3xf32>
    %v5444 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5445 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5446 = stablehlo.multiply %v5444, %d2W2v : tensor<128x128x3x3xf32>
    %v5447 = stablehlo.multiply %armeand2W2, %armeand2W2 : tensor<128x128x3x3xf32>
    %v5448 = stablehlo.multiply %v5445, %v5447 : tensor<128x128x3x3xf32>
    %v5449 = stablehlo.add %v5446, %v5448 : tensor<128x128x3x3xf32>
    %v5450 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5451 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5452 = stablehlo.divide %v5443, %v5450 : tensor<128x128x3x3xf32>
    %v5453 = stablehlo.divide %v5449, %v5451 : tensor<128x128x3x3xf32>
    %v5454 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5455 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5456 = stablehlo.sqrt %v5453 : tensor<128x128x3x3xf32>
    %v5457 = stablehlo.add %v5456, %v5455 : tensor<128x128x3x3xf32>
    %v5458 = stablehlo.divide %v5452, %v5457 : tensor<128x128x3x3xf32>
    %v5459 = stablehlo.multiply %v5454, %v5458 : tensor<128x128x3x3xf32>
    %v5460 = stablehlo.subtract %d2W2, %v5459 : tensor<128x128x3x3xf32>
    %v5461 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5462 = stablehlo.multiply %v5461, %v5454 : tensor<128x128x3x3xf32>
    %v5463 = stablehlo.multiply %v5462, %d2W2 : tensor<128x128x3x3xf32>
    %v5464 = stablehlo.subtract %v5460, %v5463 : tensor<128x128x3x3xf32>
    %arsumd2g2 = "stablehlo.all_reduce"(%v3879) ({
    ^bb0(%arad2g2: tensor<f32>, %arbd2g2: tensor<f32>):
      %araddd2g2 = stablehlo.add %arad2g2, %arbd2g2 : tensor<f32>
      stablehlo.return %araddd2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g2 = stablehlo.divide %arsumd2g2, %arnd2g2 : tensor<128xf32>
    %v5465 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5466 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5467 = stablehlo.multiply %v5465, %d2g2m : tensor<128xf32>
    %v5468 = stablehlo.multiply %v5466, %armeand2g2 : tensor<128xf32>
    %v5469 = stablehlo.add %v5467, %v5468 : tensor<128xf32>
    %v5470 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5471 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5472 = stablehlo.multiply %v5470, %d2g2v : tensor<128xf32>
    %v5473 = stablehlo.multiply %armeand2g2, %armeand2g2 : tensor<128xf32>
    %v5474 = stablehlo.multiply %v5471, %v5473 : tensor<128xf32>
    %v5475 = stablehlo.add %v5472, %v5474 : tensor<128xf32>
    %v5476 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5477 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5478 = stablehlo.multiply %v5476, %d2g2m : tensor<128xf32>
    %v5479 = stablehlo.multiply %v5477, %armeand2g2 : tensor<128xf32>
    %v5480 = stablehlo.add %v5478, %v5479 : tensor<128xf32>
    %v5481 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5482 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5483 = stablehlo.multiply %v5481, %d2g2v : tensor<128xf32>
    %v5484 = stablehlo.multiply %armeand2g2, %armeand2g2 : tensor<128xf32>
    %v5485 = stablehlo.multiply %v5482, %v5484 : tensor<128xf32>
    %v5486 = stablehlo.add %v5483, %v5485 : tensor<128xf32>
    %v5487 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5488 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5489 = stablehlo.divide %v5480, %v5487 : tensor<128xf32>
    %v5490 = stablehlo.divide %v5486, %v5488 : tensor<128xf32>
    %v5491 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5492 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5493 = stablehlo.sqrt %v5490 : tensor<128xf32>
    %v5494 = stablehlo.add %v5493, %v5492 : tensor<128xf32>
    %v5495 = stablehlo.divide %v5489, %v5494 : tensor<128xf32>
    %v5496 = stablehlo.multiply %v5491, %v5495 : tensor<128xf32>
    %v5497 = stablehlo.subtract %d2g2, %v5496 : tensor<128xf32>
    %v5498 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5499 = stablehlo.multiply %v5498, %v5491 : tensor<128xf32>
    %v5500 = stablehlo.multiply %v5499, %d2g2 : tensor<128xf32>
    %v5501 = stablehlo.subtract %v5497, %v5500 : tensor<128xf32>
    %arsumd2bt2 = "stablehlo.all_reduce"(%v3882) ({
    ^bb0(%arad2bt2: tensor<f32>, %arbd2bt2: tensor<f32>):
      %araddd2bt2 = stablehlo.add %arad2bt2, %arbd2bt2 : tensor<f32>
      stablehlo.return %araddd2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2bt2 = stablehlo.divide %arsumd2bt2, %arnd2bt2 : tensor<128xf32>
    %v5502 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5503 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5504 = stablehlo.multiply %v5502, %d2bt2m : tensor<128xf32>
    %v5505 = stablehlo.multiply %v5503, %armeand2bt2 : tensor<128xf32>
    %v5506 = stablehlo.add %v5504, %v5505 : tensor<128xf32>
    %v5507 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5508 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5509 = stablehlo.multiply %v5507, %d2bt2v : tensor<128xf32>
    %v5510 = stablehlo.multiply %armeand2bt2, %armeand2bt2 : tensor<128xf32>
    %v5511 = stablehlo.multiply %v5508, %v5510 : tensor<128xf32>
    %v5512 = stablehlo.add %v5509, %v5511 : tensor<128xf32>
    %v5513 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5514 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5515 = stablehlo.multiply %v5513, %d2bt2m : tensor<128xf32>
    %v5516 = stablehlo.multiply %v5514, %armeand2bt2 : tensor<128xf32>
    %v5517 = stablehlo.add %v5515, %v5516 : tensor<128xf32>
    %v5518 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5519 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5520 = stablehlo.multiply %v5518, %d2bt2v : tensor<128xf32>
    %v5521 = stablehlo.multiply %armeand2bt2, %armeand2bt2 : tensor<128xf32>
    %v5522 = stablehlo.multiply %v5519, %v5521 : tensor<128xf32>
    %v5523 = stablehlo.add %v5520, %v5522 : tensor<128xf32>
    %v5524 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5525 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5526 = stablehlo.divide %v5517, %v5524 : tensor<128xf32>
    %v5527 = stablehlo.divide %v5523, %v5525 : tensor<128xf32>
    %v5528 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5529 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5530 = stablehlo.sqrt %v5527 : tensor<128xf32>
    %v5531 = stablehlo.add %v5530, %v5529 : tensor<128xf32>
    %v5532 = stablehlo.divide %v5526, %v5531 : tensor<128xf32>
    %v5533 = stablehlo.multiply %v5528, %v5532 : tensor<128xf32>
    %v5534 = stablehlo.subtract %d2bt2, %v5533 : tensor<128xf32>
    %v5535 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5536 = stablehlo.multiply %v5535, %v5528 : tensor<128xf32>
    %v5537 = stablehlo.multiply %v5536, %d2bt2 : tensor<128xf32>
    %v5538 = stablehlo.subtract %v5534, %v5537 : tensor<128xf32>
    %arsumd2Wp = "stablehlo.all_reduce"(%v3890) ({
    ^bb0(%arad2Wp: tensor<f32>, %arbd2Wp: tensor<f32>):
      %araddd2Wp = stablehlo.add %arad2Wp, %arbd2Wp : tensor<f32>
      stablehlo.return %araddd2Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf32>
    %arnd2Wp = stablehlo.constant dense<2.0> : tensor<128x64x1x1xf32>
    %armeand2Wp = stablehlo.divide %arsumd2Wp, %arnd2Wp : tensor<128x64x1x1xf32>
    %v5539 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5540 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5541 = stablehlo.multiply %v5539, %d2Wpm : tensor<128x64x1x1xf32>
    %v5542 = stablehlo.multiply %v5540, %armeand2Wp : tensor<128x64x1x1xf32>
    %v5543 = stablehlo.add %v5541, %v5542 : tensor<128x64x1x1xf32>
    %v5544 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5545 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5546 = stablehlo.multiply %v5544, %d2Wpv : tensor<128x64x1x1xf32>
    %v5547 = stablehlo.multiply %armeand2Wp, %armeand2Wp : tensor<128x64x1x1xf32>
    %v5548 = stablehlo.multiply %v5545, %v5547 : tensor<128x64x1x1xf32>
    %v5549 = stablehlo.add %v5546, %v5548 : tensor<128x64x1x1xf32>
    %v5550 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5551 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5552 = stablehlo.multiply %v5550, %d2Wpm : tensor<128x64x1x1xf32>
    %v5553 = stablehlo.multiply %v5551, %armeand2Wp : tensor<128x64x1x1xf32>
    %v5554 = stablehlo.add %v5552, %v5553 : tensor<128x64x1x1xf32>
    %v5555 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5556 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5557 = stablehlo.multiply %v5555, %d2Wpv : tensor<128x64x1x1xf32>
    %v5558 = stablehlo.multiply %armeand2Wp, %armeand2Wp : tensor<128x64x1x1xf32>
    %v5559 = stablehlo.multiply %v5556, %v5558 : tensor<128x64x1x1xf32>
    %v5560 = stablehlo.add %v5557, %v5559 : tensor<128x64x1x1xf32>
    %v5561 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5562 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5563 = stablehlo.divide %v5554, %v5561 : tensor<128x64x1x1xf32>
    %v5564 = stablehlo.divide %v5560, %v5562 : tensor<128x64x1x1xf32>
    %v5565 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5566 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5567 = stablehlo.sqrt %v5564 : tensor<128x64x1x1xf32>
    %v5568 = stablehlo.add %v5567, %v5566 : tensor<128x64x1x1xf32>
    %v5569 = stablehlo.divide %v5563, %v5568 : tensor<128x64x1x1xf32>
    %v5570 = stablehlo.multiply %v5565, %v5569 : tensor<128x64x1x1xf32>
    %v5571 = stablehlo.subtract %d2Wp, %v5570 : tensor<128x64x1x1xf32>
    %v5572 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5573 = stablehlo.multiply %v5572, %v5565 : tensor<128x64x1x1xf32>
    %v5574 = stablehlo.multiply %v5573, %d2Wp : tensor<128x64x1x1xf32>
    %v5575 = stablehlo.subtract %v5571, %v5574 : tensor<128x64x1x1xf32>
    %arsumd2gp = "stablehlo.all_reduce"(%v3904) ({
    ^bb0(%arad2gp: tensor<f32>, %arbd2gp: tensor<f32>):
      %araddd2gp = stablehlo.add %arad2gp, %arbd2gp : tensor<f32>
      stablehlo.return %araddd2gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2gp = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2gp = stablehlo.divide %arsumd2gp, %arnd2gp : tensor<128xf32>
    %v5576 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5577 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5578 = stablehlo.multiply %v5576, %d2gpm : tensor<128xf32>
    %v5579 = stablehlo.multiply %v5577, %armeand2gp : tensor<128xf32>
    %v5580 = stablehlo.add %v5578, %v5579 : tensor<128xf32>
    %v5581 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5582 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5583 = stablehlo.multiply %v5581, %d2gpv : tensor<128xf32>
    %v5584 = stablehlo.multiply %armeand2gp, %armeand2gp : tensor<128xf32>
    %v5585 = stablehlo.multiply %v5582, %v5584 : tensor<128xf32>
    %v5586 = stablehlo.add %v5583, %v5585 : tensor<128xf32>
    %v5587 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5588 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5589 = stablehlo.multiply %v5587, %d2gpm : tensor<128xf32>
    %v5590 = stablehlo.multiply %v5588, %armeand2gp : tensor<128xf32>
    %v5591 = stablehlo.add %v5589, %v5590 : tensor<128xf32>
    %v5592 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5593 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5594 = stablehlo.multiply %v5592, %d2gpv : tensor<128xf32>
    %v5595 = stablehlo.multiply %armeand2gp, %armeand2gp : tensor<128xf32>
    %v5596 = stablehlo.multiply %v5593, %v5595 : tensor<128xf32>
    %v5597 = stablehlo.add %v5594, %v5596 : tensor<128xf32>
    %v5598 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5599 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5600 = stablehlo.divide %v5591, %v5598 : tensor<128xf32>
    %v5601 = stablehlo.divide %v5597, %v5599 : tensor<128xf32>
    %v5602 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5603 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5604 = stablehlo.sqrt %v5601 : tensor<128xf32>
    %v5605 = stablehlo.add %v5604, %v5603 : tensor<128xf32>
    %v5606 = stablehlo.divide %v5600, %v5605 : tensor<128xf32>
    %v5607 = stablehlo.multiply %v5602, %v5606 : tensor<128xf32>
    %v5608 = stablehlo.subtract %d2gp, %v5607 : tensor<128xf32>
    %v5609 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5610 = stablehlo.multiply %v5609, %v5602 : tensor<128xf32>
    %v5611 = stablehlo.multiply %v5610, %d2gp : tensor<128xf32>
    %v5612 = stablehlo.subtract %v5608, %v5611 : tensor<128xf32>
    %arsumd2btp = "stablehlo.all_reduce"(%v3907) ({
    ^bb0(%arad2btp: tensor<f32>, %arbd2btp: tensor<f32>):
      %araddd2btp = stablehlo.add %arad2btp, %arbd2btp : tensor<f32>
      stablehlo.return %araddd2btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2btp = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2btp = stablehlo.divide %arsumd2btp, %arnd2btp : tensor<128xf32>
    %v5613 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5614 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5615 = stablehlo.multiply %v5613, %d2btpm : tensor<128xf32>
    %v5616 = stablehlo.multiply %v5614, %armeand2btp : tensor<128xf32>
    %v5617 = stablehlo.add %v5615, %v5616 : tensor<128xf32>
    %v5618 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5619 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5620 = stablehlo.multiply %v5618, %d2btpv : tensor<128xf32>
    %v5621 = stablehlo.multiply %armeand2btp, %armeand2btp : tensor<128xf32>
    %v5622 = stablehlo.multiply %v5619, %v5621 : tensor<128xf32>
    %v5623 = stablehlo.add %v5620, %v5622 : tensor<128xf32>
    %v5624 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5625 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5626 = stablehlo.multiply %v5624, %d2btpm : tensor<128xf32>
    %v5627 = stablehlo.multiply %v5625, %armeand2btp : tensor<128xf32>
    %v5628 = stablehlo.add %v5626, %v5627 : tensor<128xf32>
    %v5629 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5630 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5631 = stablehlo.multiply %v5629, %d2btpv : tensor<128xf32>
    %v5632 = stablehlo.multiply %armeand2btp, %armeand2btp : tensor<128xf32>
    %v5633 = stablehlo.multiply %v5630, %v5632 : tensor<128xf32>
    %v5634 = stablehlo.add %v5631, %v5633 : tensor<128xf32>
    %v5635 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5636 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5637 = stablehlo.divide %v5628, %v5635 : tensor<128xf32>
    %v5638 = stablehlo.divide %v5634, %v5636 : tensor<128xf32>
    %v5639 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5640 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5641 = stablehlo.sqrt %v5638 : tensor<128xf32>
    %v5642 = stablehlo.add %v5641, %v5640 : tensor<128xf32>
    %v5643 = stablehlo.divide %v5637, %v5642 : tensor<128xf32>
    %v5644 = stablehlo.multiply %v5639, %v5643 : tensor<128xf32>
    %v5645 = stablehlo.subtract %d2btp, %v5644 : tensor<128xf32>
    %v5646 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5647 = stablehlo.multiply %v5646, %v5639 : tensor<128xf32>
    %v5648 = stablehlo.multiply %v5647, %d2btp : tensor<128xf32>
    %v5649 = stablehlo.subtract %v5645, %v5648 : tensor<128xf32>
    %arsums2b0W1 = "stablehlo.all_reduce"(%v3627) ({
    ^bb0(%aras2b0W1: tensor<f32>, %arbs2b0W1: tensor<f32>):
      %aradds2b0W1 = stablehlo.add %aras2b0W1, %arbs2b0W1 : tensor<f32>
      stablehlo.return %aradds2b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W1 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b0W1 = stablehlo.divide %arsums2b0W1, %arns2b0W1 : tensor<128x128x3x3xf32>
    %v5650 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5651 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5652 = stablehlo.multiply %v5650, %s2b0W1m : tensor<128x128x3x3xf32>
    %v5653 = stablehlo.multiply %v5651, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5654 = stablehlo.add %v5652, %v5653 : tensor<128x128x3x3xf32>
    %v5655 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5656 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5657 = stablehlo.multiply %v5655, %s2b0W1v : tensor<128x128x3x3xf32>
    %v5658 = stablehlo.multiply %armeans2b0W1, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5659 = stablehlo.multiply %v5656, %v5658 : tensor<128x128x3x3xf32>
    %v5660 = stablehlo.add %v5657, %v5659 : tensor<128x128x3x3xf32>
    %v5661 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5662 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5663 = stablehlo.multiply %v5661, %s2b0W1m : tensor<128x128x3x3xf32>
    %v5664 = stablehlo.multiply %v5662, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5665 = stablehlo.add %v5663, %v5664 : tensor<128x128x3x3xf32>
    %v5666 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5667 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5668 = stablehlo.multiply %v5666, %s2b0W1v : tensor<128x128x3x3xf32>
    %v5669 = stablehlo.multiply %armeans2b0W1, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5670 = stablehlo.multiply %v5667, %v5669 : tensor<128x128x3x3xf32>
    %v5671 = stablehlo.add %v5668, %v5670 : tensor<128x128x3x3xf32>
    %v5672 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5673 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5674 = stablehlo.divide %v5665, %v5672 : tensor<128x128x3x3xf32>
    %v5675 = stablehlo.divide %v5671, %v5673 : tensor<128x128x3x3xf32>
    %v5676 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5677 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5678 = stablehlo.sqrt %v5675 : tensor<128x128x3x3xf32>
    %v5679 = stablehlo.add %v5678, %v5677 : tensor<128x128x3x3xf32>
    %v5680 = stablehlo.divide %v5674, %v5679 : tensor<128x128x3x3xf32>
    %v5681 = stablehlo.multiply %v5676, %v5680 : tensor<128x128x3x3xf32>
    %v5682 = stablehlo.subtract %s2b0W1, %v5681 : tensor<128x128x3x3xf32>
    %v5683 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5684 = stablehlo.multiply %v5683, %v5676 : tensor<128x128x3x3xf32>
    %v5685 = stablehlo.multiply %v5684, %s2b0W1 : tensor<128x128x3x3xf32>
    %v5686 = stablehlo.subtract %v5682, %v5685 : tensor<128x128x3x3xf32>
    %arsums2b0g1 = "stablehlo.all_reduce"(%v3641) ({
    ^bb0(%aras2b0g1: tensor<f32>, %arbs2b0g1: tensor<f32>):
      %aradds2b0g1 = stablehlo.add %aras2b0g1, %arbs2b0g1 : tensor<f32>
      stablehlo.return %aradds2b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g1 = stablehlo.divide %arsums2b0g1, %arns2b0g1 : tensor<128xf32>
    %v5687 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5688 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5689 = stablehlo.multiply %v5687, %s2b0g1m : tensor<128xf32>
    %v5690 = stablehlo.multiply %v5688, %armeans2b0g1 : tensor<128xf32>
    %v5691 = stablehlo.add %v5689, %v5690 : tensor<128xf32>
    %v5692 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5693 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5694 = stablehlo.multiply %v5692, %s2b0g1v : tensor<128xf32>
    %v5695 = stablehlo.multiply %armeans2b0g1, %armeans2b0g1 : tensor<128xf32>
    %v5696 = stablehlo.multiply %v5693, %v5695 : tensor<128xf32>
    %v5697 = stablehlo.add %v5694, %v5696 : tensor<128xf32>
    %v5698 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5699 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5700 = stablehlo.multiply %v5698, %s2b0g1m : tensor<128xf32>
    %v5701 = stablehlo.multiply %v5699, %armeans2b0g1 : tensor<128xf32>
    %v5702 = stablehlo.add %v5700, %v5701 : tensor<128xf32>
    %v5703 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5704 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5705 = stablehlo.multiply %v5703, %s2b0g1v : tensor<128xf32>
    %v5706 = stablehlo.multiply %armeans2b0g1, %armeans2b0g1 : tensor<128xf32>
    %v5707 = stablehlo.multiply %v5704, %v5706 : tensor<128xf32>
    %v5708 = stablehlo.add %v5705, %v5707 : tensor<128xf32>
    %v5709 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5710 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5711 = stablehlo.divide %v5702, %v5709 : tensor<128xf32>
    %v5712 = stablehlo.divide %v5708, %v5710 : tensor<128xf32>
    %v5713 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5714 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5715 = stablehlo.sqrt %v5712 : tensor<128xf32>
    %v5716 = stablehlo.add %v5715, %v5714 : tensor<128xf32>
    %v5717 = stablehlo.divide %v5711, %v5716 : tensor<128xf32>
    %v5718 = stablehlo.multiply %v5713, %v5717 : tensor<128xf32>
    %v5719 = stablehlo.subtract %s2b0g1, %v5718 : tensor<128xf32>
    %v5720 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5721 = stablehlo.multiply %v5720, %v5713 : tensor<128xf32>
    %v5722 = stablehlo.multiply %v5721, %s2b0g1 : tensor<128xf32>
    %v5723 = stablehlo.subtract %v5719, %v5722 : tensor<128xf32>
    %arsums2b0bt1 = "stablehlo.all_reduce"(%v3644) ({
    ^bb0(%aras2b0bt1: tensor<f32>, %arbs2b0bt1: tensor<f32>):
      %aradds2b0bt1 = stablehlo.add %aras2b0bt1, %arbs2b0bt1 : tensor<f32>
      stablehlo.return %aradds2b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0bt1 = stablehlo.divide %arsums2b0bt1, %arns2b0bt1 : tensor<128xf32>
    %v5724 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5725 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5726 = stablehlo.multiply %v5724, %s2b0bt1m : tensor<128xf32>
    %v5727 = stablehlo.multiply %v5725, %armeans2b0bt1 : tensor<128xf32>
    %v5728 = stablehlo.add %v5726, %v5727 : tensor<128xf32>
    %v5729 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5730 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5731 = stablehlo.multiply %v5729, %s2b0bt1v : tensor<128xf32>
    %v5732 = stablehlo.multiply %armeans2b0bt1, %armeans2b0bt1 : tensor<128xf32>
    %v5733 = stablehlo.multiply %v5730, %v5732 : tensor<128xf32>
    %v5734 = stablehlo.add %v5731, %v5733 : tensor<128xf32>
    %v5735 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5736 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5737 = stablehlo.multiply %v5735, %s2b0bt1m : tensor<128xf32>
    %v5738 = stablehlo.multiply %v5736, %armeans2b0bt1 : tensor<128xf32>
    %v5739 = stablehlo.add %v5737, %v5738 : tensor<128xf32>
    %v5740 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5741 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5742 = stablehlo.multiply %v5740, %s2b0bt1v : tensor<128xf32>
    %v5743 = stablehlo.multiply %armeans2b0bt1, %armeans2b0bt1 : tensor<128xf32>
    %v5744 = stablehlo.multiply %v5741, %v5743 : tensor<128xf32>
    %v5745 = stablehlo.add %v5742, %v5744 : tensor<128xf32>
    %v5746 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5747 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5748 = stablehlo.divide %v5739, %v5746 : tensor<128xf32>
    %v5749 = stablehlo.divide %v5745, %v5747 : tensor<128xf32>
    %v5750 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5751 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5752 = stablehlo.sqrt %v5749 : tensor<128xf32>
    %v5753 = stablehlo.add %v5752, %v5751 : tensor<128xf32>
    %v5754 = stablehlo.divide %v5748, %v5753 : tensor<128xf32>
    %v5755 = stablehlo.multiply %v5750, %v5754 : tensor<128xf32>
    %v5756 = stablehlo.subtract %s2b0bt1, %v5755 : tensor<128xf32>
    %v5757 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5758 = stablehlo.multiply %v5757, %v5750 : tensor<128xf32>
    %v5759 = stablehlo.multiply %v5758, %s2b0bt1 : tensor<128xf32>
    %v5760 = stablehlo.subtract %v5756, %v5759 : tensor<128xf32>
    %arsums2b0W2 = "stablehlo.all_reduce"(%v3650) ({
    ^bb0(%aras2b0W2: tensor<f32>, %arbs2b0W2: tensor<f32>):
      %aradds2b0W2 = stablehlo.add %aras2b0W2, %arbs2b0W2 : tensor<f32>
      stablehlo.return %aradds2b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b0W2 = stablehlo.divide %arsums2b0W2, %arns2b0W2 : tensor<128x128x3x3xf32>
    %v5761 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5762 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5763 = stablehlo.multiply %v5761, %s2b0W2m : tensor<128x128x3x3xf32>
    %v5764 = stablehlo.multiply %v5762, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5765 = stablehlo.add %v5763, %v5764 : tensor<128x128x3x3xf32>
    %v5766 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5767 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5768 = stablehlo.multiply %v5766, %s2b0W2v : tensor<128x128x3x3xf32>
    %v5769 = stablehlo.multiply %armeans2b0W2, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5770 = stablehlo.multiply %v5767, %v5769 : tensor<128x128x3x3xf32>
    %v5771 = stablehlo.add %v5768, %v5770 : tensor<128x128x3x3xf32>
    %v5772 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5773 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5774 = stablehlo.multiply %v5772, %s2b0W2m : tensor<128x128x3x3xf32>
    %v5775 = stablehlo.multiply %v5773, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5776 = stablehlo.add %v5774, %v5775 : tensor<128x128x3x3xf32>
    %v5777 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5778 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5779 = stablehlo.multiply %v5777, %s2b0W2v : tensor<128x128x3x3xf32>
    %v5780 = stablehlo.multiply %armeans2b0W2, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5781 = stablehlo.multiply %v5778, %v5780 : tensor<128x128x3x3xf32>
    %v5782 = stablehlo.add %v5779, %v5781 : tensor<128x128x3x3xf32>
    %v5783 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5784 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5785 = stablehlo.divide %v5776, %v5783 : tensor<128x128x3x3xf32>
    %v5786 = stablehlo.divide %v5782, %v5784 : tensor<128x128x3x3xf32>
    %v5787 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5788 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5789 = stablehlo.sqrt %v5786 : tensor<128x128x3x3xf32>
    %v5790 = stablehlo.add %v5789, %v5788 : tensor<128x128x3x3xf32>
    %v5791 = stablehlo.divide %v5785, %v5790 : tensor<128x128x3x3xf32>
    %v5792 = stablehlo.multiply %v5787, %v5791 : tensor<128x128x3x3xf32>
    %v5793 = stablehlo.subtract %s2b0W2, %v5792 : tensor<128x128x3x3xf32>
    %v5794 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5795 = stablehlo.multiply %v5794, %v5787 : tensor<128x128x3x3xf32>
    %v5796 = stablehlo.multiply %v5795, %s2b0W2 : tensor<128x128x3x3xf32>
    %v5797 = stablehlo.subtract %v5793, %v5796 : tensor<128x128x3x3xf32>
    %arsums2b0g2 = "stablehlo.all_reduce"(%v3664) ({
    ^bb0(%aras2b0g2: tensor<f32>, %arbs2b0g2: tensor<f32>):
      %aradds2b0g2 = stablehlo.add %aras2b0g2, %arbs2b0g2 : tensor<f32>
      stablehlo.return %aradds2b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g2 = stablehlo.divide %arsums2b0g2, %arns2b0g2 : tensor<128xf32>
    %v5798 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5799 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5800 = stablehlo.multiply %v5798, %s2b0g2m : tensor<128xf32>
    %v5801 = stablehlo.multiply %v5799, %armeans2b0g2 : tensor<128xf32>
    %v5802 = stablehlo.add %v5800, %v5801 : tensor<128xf32>
    %v5803 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5804 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5805 = stablehlo.multiply %v5803, %s2b0g2v : tensor<128xf32>
    %v5806 = stablehlo.multiply %armeans2b0g2, %armeans2b0g2 : tensor<128xf32>
    %v5807 = stablehlo.multiply %v5804, %v5806 : tensor<128xf32>
    %v5808 = stablehlo.add %v5805, %v5807 : tensor<128xf32>
    %v5809 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5810 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5811 = stablehlo.multiply %v5809, %s2b0g2m : tensor<128xf32>
    %v5812 = stablehlo.multiply %v5810, %armeans2b0g2 : tensor<128xf32>
    %v5813 = stablehlo.add %v5811, %v5812 : tensor<128xf32>
    %v5814 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5815 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5816 = stablehlo.multiply %v5814, %s2b0g2v : tensor<128xf32>
    %v5817 = stablehlo.multiply %armeans2b0g2, %armeans2b0g2 : tensor<128xf32>
    %v5818 = stablehlo.multiply %v5815, %v5817 : tensor<128xf32>
    %v5819 = stablehlo.add %v5816, %v5818 : tensor<128xf32>
    %v5820 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5821 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5822 = stablehlo.divide %v5813, %v5820 : tensor<128xf32>
    %v5823 = stablehlo.divide %v5819, %v5821 : tensor<128xf32>
    %v5824 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5825 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5826 = stablehlo.sqrt %v5823 : tensor<128xf32>
    %v5827 = stablehlo.add %v5826, %v5825 : tensor<128xf32>
    %v5828 = stablehlo.divide %v5822, %v5827 : tensor<128xf32>
    %v5829 = stablehlo.multiply %v5824, %v5828 : tensor<128xf32>
    %v5830 = stablehlo.subtract %s2b0g2, %v5829 : tensor<128xf32>
    %v5831 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5832 = stablehlo.multiply %v5831, %v5824 : tensor<128xf32>
    %v5833 = stablehlo.multiply %v5832, %s2b0g2 : tensor<128xf32>
    %v5834 = stablehlo.subtract %v5830, %v5833 : tensor<128xf32>
    %arsums2b0bt2 = "stablehlo.all_reduce"(%v3667) ({
    ^bb0(%aras2b0bt2: tensor<f32>, %arbs2b0bt2: tensor<f32>):
      %aradds2b0bt2 = stablehlo.add %aras2b0bt2, %arbs2b0bt2 : tensor<f32>
      stablehlo.return %aradds2b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0bt2 = stablehlo.divide %arsums2b0bt2, %arns2b0bt2 : tensor<128xf32>
    %v5835 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5836 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5837 = stablehlo.multiply %v5835, %s2b0bt2m : tensor<128xf32>
    %v5838 = stablehlo.multiply %v5836, %armeans2b0bt2 : tensor<128xf32>
    %v5839 = stablehlo.add %v5837, %v5838 : tensor<128xf32>
    %v5840 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5841 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5842 = stablehlo.multiply %v5840, %s2b0bt2v : tensor<128xf32>
    %v5843 = stablehlo.multiply %armeans2b0bt2, %armeans2b0bt2 : tensor<128xf32>
    %v5844 = stablehlo.multiply %v5841, %v5843 : tensor<128xf32>
    %v5845 = stablehlo.add %v5842, %v5844 : tensor<128xf32>
    %v5846 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5847 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5848 = stablehlo.multiply %v5846, %s2b0bt2m : tensor<128xf32>
    %v5849 = stablehlo.multiply %v5847, %armeans2b0bt2 : tensor<128xf32>
    %v5850 = stablehlo.add %v5848, %v5849 : tensor<128xf32>
    %v5851 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5852 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5853 = stablehlo.multiply %v5851, %s2b0bt2v : tensor<128xf32>
    %v5854 = stablehlo.multiply %armeans2b0bt2, %armeans2b0bt2 : tensor<128xf32>
    %v5855 = stablehlo.multiply %v5852, %v5854 : tensor<128xf32>
    %v5856 = stablehlo.add %v5853, %v5855 : tensor<128xf32>
    %v5857 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5858 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5859 = stablehlo.divide %v5850, %v5857 : tensor<128xf32>
    %v5860 = stablehlo.divide %v5856, %v5858 : tensor<128xf32>
    %v5861 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5862 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5863 = stablehlo.sqrt %v5860 : tensor<128xf32>
    %v5864 = stablehlo.add %v5863, %v5862 : tensor<128xf32>
    %v5865 = stablehlo.divide %v5859, %v5864 : tensor<128xf32>
    %v5866 = stablehlo.multiply %v5861, %v5865 : tensor<128xf32>
    %v5867 = stablehlo.subtract %s2b0bt2, %v5866 : tensor<128xf32>
    %v5868 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5869 = stablehlo.multiply %v5868, %v5861 : tensor<128xf32>
    %v5870 = stablehlo.multiply %v5869, %s2b0bt2 : tensor<128xf32>
    %v5871 = stablehlo.subtract %v5867, %v5870 : tensor<128xf32>
    %arsums2b1W1 = "stablehlo.all_reduce"(%v3467) ({
    ^bb0(%aras2b1W1: tensor<f32>, %arbs2b1W1: tensor<f32>):
      %aradds2b1W1 = stablehlo.add %aras2b1W1, %arbs2b1W1 : tensor<f32>
      stablehlo.return %aradds2b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W1 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b1W1 = stablehlo.divide %arsums2b1W1, %arns2b1W1 : tensor<128x128x3x3xf32>
    %v5872 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5873 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5874 = stablehlo.multiply %v5872, %s2b1W1m : tensor<128x128x3x3xf32>
    %v5875 = stablehlo.multiply %v5873, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5876 = stablehlo.add %v5874, %v5875 : tensor<128x128x3x3xf32>
    %v5877 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5878 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5879 = stablehlo.multiply %v5877, %s2b1W1v : tensor<128x128x3x3xf32>
    %v5880 = stablehlo.multiply %armeans2b1W1, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5881 = stablehlo.multiply %v5878, %v5880 : tensor<128x128x3x3xf32>
    %v5882 = stablehlo.add %v5879, %v5881 : tensor<128x128x3x3xf32>
    %v5883 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5884 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5885 = stablehlo.multiply %v5883, %s2b1W1m : tensor<128x128x3x3xf32>
    %v5886 = stablehlo.multiply %v5884, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5887 = stablehlo.add %v5885, %v5886 : tensor<128x128x3x3xf32>
    %v5888 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5889 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5890 = stablehlo.multiply %v5888, %s2b1W1v : tensor<128x128x3x3xf32>
    %v5891 = stablehlo.multiply %armeans2b1W1, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5892 = stablehlo.multiply %v5889, %v5891 : tensor<128x128x3x3xf32>
    %v5893 = stablehlo.add %v5890, %v5892 : tensor<128x128x3x3xf32>
    %v5894 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5895 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5896 = stablehlo.divide %v5887, %v5894 : tensor<128x128x3x3xf32>
    %v5897 = stablehlo.divide %v5893, %v5895 : tensor<128x128x3x3xf32>
    %v5898 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5899 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5900 = stablehlo.sqrt %v5897 : tensor<128x128x3x3xf32>
    %v5901 = stablehlo.add %v5900, %v5899 : tensor<128x128x3x3xf32>
    %v5902 = stablehlo.divide %v5896, %v5901 : tensor<128x128x3x3xf32>
    %v5903 = stablehlo.multiply %v5898, %v5902 : tensor<128x128x3x3xf32>
    %v5904 = stablehlo.subtract %s2b1W1, %v5903 : tensor<128x128x3x3xf32>
    %v5905 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5906 = stablehlo.multiply %v5905, %v5898 : tensor<128x128x3x3xf32>
    %v5907 = stablehlo.multiply %v5906, %s2b1W1 : tensor<128x128x3x3xf32>
    %v5908 = stablehlo.subtract %v5904, %v5907 : tensor<128x128x3x3xf32>
    %arsums2b1g1 = "stablehlo.all_reduce"(%v3481) ({
    ^bb0(%aras2b1g1: tensor<f32>, %arbs2b1g1: tensor<f32>):
      %aradds2b1g1 = stablehlo.add %aras2b1g1, %arbs2b1g1 : tensor<f32>
      stablehlo.return %aradds2b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g1 = stablehlo.divide %arsums2b1g1, %arns2b1g1 : tensor<128xf32>
    %v5909 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5910 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5911 = stablehlo.multiply %v5909, %s2b1g1m : tensor<128xf32>
    %v5912 = stablehlo.multiply %v5910, %armeans2b1g1 : tensor<128xf32>
    %v5913 = stablehlo.add %v5911, %v5912 : tensor<128xf32>
    %v5914 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5915 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5916 = stablehlo.multiply %v5914, %s2b1g1v : tensor<128xf32>
    %v5917 = stablehlo.multiply %armeans2b1g1, %armeans2b1g1 : tensor<128xf32>
    %v5918 = stablehlo.multiply %v5915, %v5917 : tensor<128xf32>
    %v5919 = stablehlo.add %v5916, %v5918 : tensor<128xf32>
    %v5920 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5921 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5922 = stablehlo.multiply %v5920, %s2b1g1m : tensor<128xf32>
    %v5923 = stablehlo.multiply %v5921, %armeans2b1g1 : tensor<128xf32>
    %v5924 = stablehlo.add %v5922, %v5923 : tensor<128xf32>
    %v5925 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5926 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5927 = stablehlo.multiply %v5925, %s2b1g1v : tensor<128xf32>
    %v5928 = stablehlo.multiply %armeans2b1g1, %armeans2b1g1 : tensor<128xf32>
    %v5929 = stablehlo.multiply %v5926, %v5928 : tensor<128xf32>
    %v5930 = stablehlo.add %v5927, %v5929 : tensor<128xf32>
    %v5931 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5932 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5933 = stablehlo.divide %v5924, %v5931 : tensor<128xf32>
    %v5934 = stablehlo.divide %v5930, %v5932 : tensor<128xf32>
    %v5935 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5936 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5937 = stablehlo.sqrt %v5934 : tensor<128xf32>
    %v5938 = stablehlo.add %v5937, %v5936 : tensor<128xf32>
    %v5939 = stablehlo.divide %v5933, %v5938 : tensor<128xf32>
    %v5940 = stablehlo.multiply %v5935, %v5939 : tensor<128xf32>
    %v5941 = stablehlo.subtract %s2b1g1, %v5940 : tensor<128xf32>
    %v5942 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5943 = stablehlo.multiply %v5942, %v5935 : tensor<128xf32>
    %v5944 = stablehlo.multiply %v5943, %s2b1g1 : tensor<128xf32>
    %v5945 = stablehlo.subtract %v5941, %v5944 : tensor<128xf32>
    %arsums2b1bt1 = "stablehlo.all_reduce"(%v3484) ({
    ^bb0(%aras2b1bt1: tensor<f32>, %arbs2b1bt1: tensor<f32>):
      %aradds2b1bt1 = stablehlo.add %aras2b1bt1, %arbs2b1bt1 : tensor<f32>
      stablehlo.return %aradds2b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1bt1 = stablehlo.divide %arsums2b1bt1, %arns2b1bt1 : tensor<128xf32>
    %v5946 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5947 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5948 = stablehlo.multiply %v5946, %s2b1bt1m : tensor<128xf32>
    %v5949 = stablehlo.multiply %v5947, %armeans2b1bt1 : tensor<128xf32>
    %v5950 = stablehlo.add %v5948, %v5949 : tensor<128xf32>
    %v5951 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5952 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5953 = stablehlo.multiply %v5951, %s2b1bt1v : tensor<128xf32>
    %v5954 = stablehlo.multiply %armeans2b1bt1, %armeans2b1bt1 : tensor<128xf32>
    %v5955 = stablehlo.multiply %v5952, %v5954 : tensor<128xf32>
    %v5956 = stablehlo.add %v5953, %v5955 : tensor<128xf32>
    %v5957 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5958 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5959 = stablehlo.multiply %v5957, %s2b1bt1m : tensor<128xf32>
    %v5960 = stablehlo.multiply %v5958, %armeans2b1bt1 : tensor<128xf32>
    %v5961 = stablehlo.add %v5959, %v5960 : tensor<128xf32>
    %v5962 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5963 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5964 = stablehlo.multiply %v5962, %s2b1bt1v : tensor<128xf32>
    %v5965 = stablehlo.multiply %armeans2b1bt1, %armeans2b1bt1 : tensor<128xf32>
    %v5966 = stablehlo.multiply %v5963, %v5965 : tensor<128xf32>
    %v5967 = stablehlo.add %v5964, %v5966 : tensor<128xf32>
    %v5968 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5969 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5970 = stablehlo.divide %v5961, %v5968 : tensor<128xf32>
    %v5971 = stablehlo.divide %v5967, %v5969 : tensor<128xf32>
    %v5972 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5973 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5974 = stablehlo.sqrt %v5971 : tensor<128xf32>
    %v5975 = stablehlo.add %v5974, %v5973 : tensor<128xf32>
    %v5976 = stablehlo.divide %v5970, %v5975 : tensor<128xf32>
    %v5977 = stablehlo.multiply %v5972, %v5976 : tensor<128xf32>
    %v5978 = stablehlo.subtract %s2b1bt1, %v5977 : tensor<128xf32>
    %v5979 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5980 = stablehlo.multiply %v5979, %v5972 : tensor<128xf32>
    %v5981 = stablehlo.multiply %v5980, %s2b1bt1 : tensor<128xf32>
    %v5982 = stablehlo.subtract %v5978, %v5981 : tensor<128xf32>
    %arsums2b1W2 = "stablehlo.all_reduce"(%v3490) ({
    ^bb0(%aras2b1W2: tensor<f32>, %arbs2b1W2: tensor<f32>):
      %aradds2b1W2 = stablehlo.add %aras2b1W2, %arbs2b1W2 : tensor<f32>
      stablehlo.return %aradds2b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b1W2 = stablehlo.divide %arsums2b1W2, %arns2b1W2 : tensor<128x128x3x3xf32>
    %v5983 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5984 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5985 = stablehlo.multiply %v5983, %s2b1W2m : tensor<128x128x3x3xf32>
    %v5986 = stablehlo.multiply %v5984, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v5987 = stablehlo.add %v5985, %v5986 : tensor<128x128x3x3xf32>
    %v5988 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5989 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5990 = stablehlo.multiply %v5988, %s2b1W2v : tensor<128x128x3x3xf32>
    %v5991 = stablehlo.multiply %armeans2b1W2, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v5992 = stablehlo.multiply %v5989, %v5991 : tensor<128x128x3x3xf32>
    %v5993 = stablehlo.add %v5990, %v5992 : tensor<128x128x3x3xf32>
    %v5994 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5995 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5996 = stablehlo.multiply %v5994, %s2b1W2m : tensor<128x128x3x3xf32>
    %v5997 = stablehlo.multiply %v5995, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v5998 = stablehlo.add %v5996, %v5997 : tensor<128x128x3x3xf32>
    %v5999 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6000 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6001 = stablehlo.multiply %v5999, %s2b1W2v : tensor<128x128x3x3xf32>
    %v6002 = stablehlo.multiply %armeans2b1W2, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v6003 = stablehlo.multiply %v6000, %v6002 : tensor<128x128x3x3xf32>
    %v6004 = stablehlo.add %v6001, %v6003 : tensor<128x128x3x3xf32>
    %v6005 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6006 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6007 = stablehlo.divide %v5998, %v6005 : tensor<128x128x3x3xf32>
    %v6008 = stablehlo.divide %v6004, %v6006 : tensor<128x128x3x3xf32>
    %v6009 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6010 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6011 = stablehlo.sqrt %v6008 : tensor<128x128x3x3xf32>
    %v6012 = stablehlo.add %v6011, %v6010 : tensor<128x128x3x3xf32>
    %v6013 = stablehlo.divide %v6007, %v6012 : tensor<128x128x3x3xf32>
    %v6014 = stablehlo.multiply %v6009, %v6013 : tensor<128x128x3x3xf32>
    %v6015 = stablehlo.subtract %s2b1W2, %v6014 : tensor<128x128x3x3xf32>
    %v6016 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6017 = stablehlo.multiply %v6016, %v6009 : tensor<128x128x3x3xf32>
    %v6018 = stablehlo.multiply %v6017, %s2b1W2 : tensor<128x128x3x3xf32>
    %v6019 = stablehlo.subtract %v6015, %v6018 : tensor<128x128x3x3xf32>
    %arsums2b1g2 = "stablehlo.all_reduce"(%v3504) ({
    ^bb0(%aras2b1g2: tensor<f32>, %arbs2b1g2: tensor<f32>):
      %aradds2b1g2 = stablehlo.add %aras2b1g2, %arbs2b1g2 : tensor<f32>
      stablehlo.return %aradds2b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g2 = stablehlo.divide %arsums2b1g2, %arns2b1g2 : tensor<128xf32>
    %v6020 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6021 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6022 = stablehlo.multiply %v6020, %s2b1g2m : tensor<128xf32>
    %v6023 = stablehlo.multiply %v6021, %armeans2b1g2 : tensor<128xf32>
    %v6024 = stablehlo.add %v6022, %v6023 : tensor<128xf32>
    %v6025 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6026 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6027 = stablehlo.multiply %v6025, %s2b1g2v : tensor<128xf32>
    %v6028 = stablehlo.multiply %armeans2b1g2, %armeans2b1g2 : tensor<128xf32>
    %v6029 = stablehlo.multiply %v6026, %v6028 : tensor<128xf32>
    %v6030 = stablehlo.add %v6027, %v6029 : tensor<128xf32>
    %v6031 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6032 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6033 = stablehlo.multiply %v6031, %s2b1g2m : tensor<128xf32>
    %v6034 = stablehlo.multiply %v6032, %armeans2b1g2 : tensor<128xf32>
    %v6035 = stablehlo.add %v6033, %v6034 : tensor<128xf32>
    %v6036 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6037 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6038 = stablehlo.multiply %v6036, %s2b1g2v : tensor<128xf32>
    %v6039 = stablehlo.multiply %armeans2b1g2, %armeans2b1g2 : tensor<128xf32>
    %v6040 = stablehlo.multiply %v6037, %v6039 : tensor<128xf32>
    %v6041 = stablehlo.add %v6038, %v6040 : tensor<128xf32>
    %v6042 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6043 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6044 = stablehlo.divide %v6035, %v6042 : tensor<128xf32>
    %v6045 = stablehlo.divide %v6041, %v6043 : tensor<128xf32>
    %v6046 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6047 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6048 = stablehlo.sqrt %v6045 : tensor<128xf32>
    %v6049 = stablehlo.add %v6048, %v6047 : tensor<128xf32>
    %v6050 = stablehlo.divide %v6044, %v6049 : tensor<128xf32>
    %v6051 = stablehlo.multiply %v6046, %v6050 : tensor<128xf32>
    %v6052 = stablehlo.subtract %s2b1g2, %v6051 : tensor<128xf32>
    %v6053 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6054 = stablehlo.multiply %v6053, %v6046 : tensor<128xf32>
    %v6055 = stablehlo.multiply %v6054, %s2b1g2 : tensor<128xf32>
    %v6056 = stablehlo.subtract %v6052, %v6055 : tensor<128xf32>
    %arsums2b1bt2 = "stablehlo.all_reduce"(%v3507) ({
    ^bb0(%aras2b1bt2: tensor<f32>, %arbs2b1bt2: tensor<f32>):
      %aradds2b1bt2 = stablehlo.add %aras2b1bt2, %arbs2b1bt2 : tensor<f32>
      stablehlo.return %aradds2b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1bt2 = stablehlo.divide %arsums2b1bt2, %arns2b1bt2 : tensor<128xf32>
    %v6057 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6058 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6059 = stablehlo.multiply %v6057, %s2b1bt2m : tensor<128xf32>
    %v6060 = stablehlo.multiply %v6058, %armeans2b1bt2 : tensor<128xf32>
    %v6061 = stablehlo.add %v6059, %v6060 : tensor<128xf32>
    %v6062 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6063 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6064 = stablehlo.multiply %v6062, %s2b1bt2v : tensor<128xf32>
    %v6065 = stablehlo.multiply %armeans2b1bt2, %armeans2b1bt2 : tensor<128xf32>
    %v6066 = stablehlo.multiply %v6063, %v6065 : tensor<128xf32>
    %v6067 = stablehlo.add %v6064, %v6066 : tensor<128xf32>
    %v6068 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6069 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6070 = stablehlo.multiply %v6068, %s2b1bt2m : tensor<128xf32>
    %v6071 = stablehlo.multiply %v6069, %armeans2b1bt2 : tensor<128xf32>
    %v6072 = stablehlo.add %v6070, %v6071 : tensor<128xf32>
    %v6073 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6074 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6075 = stablehlo.multiply %v6073, %s2b1bt2v : tensor<128xf32>
    %v6076 = stablehlo.multiply %armeans2b1bt2, %armeans2b1bt2 : tensor<128xf32>
    %v6077 = stablehlo.multiply %v6074, %v6076 : tensor<128xf32>
    %v6078 = stablehlo.add %v6075, %v6077 : tensor<128xf32>
    %v6079 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6080 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6081 = stablehlo.divide %v6072, %v6079 : tensor<128xf32>
    %v6082 = stablehlo.divide %v6078, %v6080 : tensor<128xf32>
    %v6083 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6084 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6085 = stablehlo.sqrt %v6082 : tensor<128xf32>
    %v6086 = stablehlo.add %v6085, %v6084 : tensor<128xf32>
    %v6087 = stablehlo.divide %v6081, %v6086 : tensor<128xf32>
    %v6088 = stablehlo.multiply %v6083, %v6087 : tensor<128xf32>
    %v6089 = stablehlo.subtract %s2b1bt2, %v6088 : tensor<128xf32>
    %v6090 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6091 = stablehlo.multiply %v6090, %v6083 : tensor<128xf32>
    %v6092 = stablehlo.multiply %v6091, %s2b1bt2 : tensor<128xf32>
    %v6093 = stablehlo.subtract %v6089, %v6092 : tensor<128xf32>
    %arsums2b2W1 = "stablehlo.all_reduce"(%v3307) ({
    ^bb0(%aras2b2W1: tensor<f32>, %arbs2b2W1: tensor<f32>):
      %aradds2b2W1 = stablehlo.add %aras2b2W1, %arbs2b2W1 : tensor<f32>
      stablehlo.return %aradds2b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W1 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b2W1 = stablehlo.divide %arsums2b2W1, %arns2b2W1 : tensor<128x128x3x3xf32>
    %v6094 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6095 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6096 = stablehlo.multiply %v6094, %s2b2W1m : tensor<128x128x3x3xf32>
    %v6097 = stablehlo.multiply %v6095, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v6098 = stablehlo.add %v6096, %v6097 : tensor<128x128x3x3xf32>
    %v6099 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6100 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6101 = stablehlo.multiply %v6099, %s2b2W1v : tensor<128x128x3x3xf32>
    %v6102 = stablehlo.multiply %armeans2b2W1, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v6103 = stablehlo.multiply %v6100, %v6102 : tensor<128x128x3x3xf32>
    %v6104 = stablehlo.add %v6101, %v6103 : tensor<128x128x3x3xf32>
    %v6105 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6106 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6107 = stablehlo.multiply %v6105, %s2b2W1m : tensor<128x128x3x3xf32>
    %v6108 = stablehlo.multiply %v6106, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v6109 = stablehlo.add %v6107, %v6108 : tensor<128x128x3x3xf32>
    %v6110 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6111 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6112 = stablehlo.multiply %v6110, %s2b2W1v : tensor<128x128x3x3xf32>
    %v6113 = stablehlo.multiply %armeans2b2W1, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v6114 = stablehlo.multiply %v6111, %v6113 : tensor<128x128x3x3xf32>
    %v6115 = stablehlo.add %v6112, %v6114 : tensor<128x128x3x3xf32>
    %v6116 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6117 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6118 = stablehlo.divide %v6109, %v6116 : tensor<128x128x3x3xf32>
    %v6119 = stablehlo.divide %v6115, %v6117 : tensor<128x128x3x3xf32>
    %v6120 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6121 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6122 = stablehlo.sqrt %v6119 : tensor<128x128x3x3xf32>
    %v6123 = stablehlo.add %v6122, %v6121 : tensor<128x128x3x3xf32>
    %v6124 = stablehlo.divide %v6118, %v6123 : tensor<128x128x3x3xf32>
    %v6125 = stablehlo.multiply %v6120, %v6124 : tensor<128x128x3x3xf32>
    %v6126 = stablehlo.subtract %s2b2W1, %v6125 : tensor<128x128x3x3xf32>
    %v6127 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6128 = stablehlo.multiply %v6127, %v6120 : tensor<128x128x3x3xf32>
    %v6129 = stablehlo.multiply %v6128, %s2b2W1 : tensor<128x128x3x3xf32>
    %v6130 = stablehlo.subtract %v6126, %v6129 : tensor<128x128x3x3xf32>
    %arsums2b2g1 = "stablehlo.all_reduce"(%v3321) ({
    ^bb0(%aras2b2g1: tensor<f32>, %arbs2b2g1: tensor<f32>):
      %aradds2b2g1 = stablehlo.add %aras2b2g1, %arbs2b2g1 : tensor<f32>
      stablehlo.return %aradds2b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g1 = stablehlo.divide %arsums2b2g1, %arns2b2g1 : tensor<128xf32>
    %v6131 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6132 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6133 = stablehlo.multiply %v6131, %s2b2g1m : tensor<128xf32>
    %v6134 = stablehlo.multiply %v6132, %armeans2b2g1 : tensor<128xf32>
    %v6135 = stablehlo.add %v6133, %v6134 : tensor<128xf32>
    %v6136 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6137 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6138 = stablehlo.multiply %v6136, %s2b2g1v : tensor<128xf32>
    %v6139 = stablehlo.multiply %armeans2b2g1, %armeans2b2g1 : tensor<128xf32>
    %v6140 = stablehlo.multiply %v6137, %v6139 : tensor<128xf32>
    %v6141 = stablehlo.add %v6138, %v6140 : tensor<128xf32>
    %v6142 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6143 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6144 = stablehlo.multiply %v6142, %s2b2g1m : tensor<128xf32>
    %v6145 = stablehlo.multiply %v6143, %armeans2b2g1 : tensor<128xf32>
    %v6146 = stablehlo.add %v6144, %v6145 : tensor<128xf32>
    %v6147 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6148 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6149 = stablehlo.multiply %v6147, %s2b2g1v : tensor<128xf32>
    %v6150 = stablehlo.multiply %armeans2b2g1, %armeans2b2g1 : tensor<128xf32>
    %v6151 = stablehlo.multiply %v6148, %v6150 : tensor<128xf32>
    %v6152 = stablehlo.add %v6149, %v6151 : tensor<128xf32>
    %v6153 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6154 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6155 = stablehlo.divide %v6146, %v6153 : tensor<128xf32>
    %v6156 = stablehlo.divide %v6152, %v6154 : tensor<128xf32>
    %v6157 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6158 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6159 = stablehlo.sqrt %v6156 : tensor<128xf32>
    %v6160 = stablehlo.add %v6159, %v6158 : tensor<128xf32>
    %v6161 = stablehlo.divide %v6155, %v6160 : tensor<128xf32>
    %v6162 = stablehlo.multiply %v6157, %v6161 : tensor<128xf32>
    %v6163 = stablehlo.subtract %s2b2g1, %v6162 : tensor<128xf32>
    %v6164 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6165 = stablehlo.multiply %v6164, %v6157 : tensor<128xf32>
    %v6166 = stablehlo.multiply %v6165, %s2b2g1 : tensor<128xf32>
    %v6167 = stablehlo.subtract %v6163, %v6166 : tensor<128xf32>
    %arsums2b2bt1 = "stablehlo.all_reduce"(%v3324) ({
    ^bb0(%aras2b2bt1: tensor<f32>, %arbs2b2bt1: tensor<f32>):
      %aradds2b2bt1 = stablehlo.add %aras2b2bt1, %arbs2b2bt1 : tensor<f32>
      stablehlo.return %aradds2b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2bt1 = stablehlo.divide %arsums2b2bt1, %arns2b2bt1 : tensor<128xf32>
    %v6168 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6169 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6170 = stablehlo.multiply %v6168, %s2b2bt1m : tensor<128xf32>
    %v6171 = stablehlo.multiply %v6169, %armeans2b2bt1 : tensor<128xf32>
    %v6172 = stablehlo.add %v6170, %v6171 : tensor<128xf32>
    %v6173 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6174 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6175 = stablehlo.multiply %v6173, %s2b2bt1v : tensor<128xf32>
    %v6176 = stablehlo.multiply %armeans2b2bt1, %armeans2b2bt1 : tensor<128xf32>
    %v6177 = stablehlo.multiply %v6174, %v6176 : tensor<128xf32>
    %v6178 = stablehlo.add %v6175, %v6177 : tensor<128xf32>
    %v6179 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6180 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6181 = stablehlo.multiply %v6179, %s2b2bt1m : tensor<128xf32>
    %v6182 = stablehlo.multiply %v6180, %armeans2b2bt1 : tensor<128xf32>
    %v6183 = stablehlo.add %v6181, %v6182 : tensor<128xf32>
    %v6184 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6185 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6186 = stablehlo.multiply %v6184, %s2b2bt1v : tensor<128xf32>
    %v6187 = stablehlo.multiply %armeans2b2bt1, %armeans2b2bt1 : tensor<128xf32>
    %v6188 = stablehlo.multiply %v6185, %v6187 : tensor<128xf32>
    %v6189 = stablehlo.add %v6186, %v6188 : tensor<128xf32>
    %v6190 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6191 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6192 = stablehlo.divide %v6183, %v6190 : tensor<128xf32>
    %v6193 = stablehlo.divide %v6189, %v6191 : tensor<128xf32>
    %v6194 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6195 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6196 = stablehlo.sqrt %v6193 : tensor<128xf32>
    %v6197 = stablehlo.add %v6196, %v6195 : tensor<128xf32>
    %v6198 = stablehlo.divide %v6192, %v6197 : tensor<128xf32>
    %v6199 = stablehlo.multiply %v6194, %v6198 : tensor<128xf32>
    %v6200 = stablehlo.subtract %s2b2bt1, %v6199 : tensor<128xf32>
    %v6201 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6202 = stablehlo.multiply %v6201, %v6194 : tensor<128xf32>
    %v6203 = stablehlo.multiply %v6202, %s2b2bt1 : tensor<128xf32>
    %v6204 = stablehlo.subtract %v6200, %v6203 : tensor<128xf32>
    %arsums2b2W2 = "stablehlo.all_reduce"(%v3330) ({
    ^bb0(%aras2b2W2: tensor<f32>, %arbs2b2W2: tensor<f32>):
      %aradds2b2W2 = stablehlo.add %aras2b2W2, %arbs2b2W2 : tensor<f32>
      stablehlo.return %aradds2b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b2W2 = stablehlo.divide %arsums2b2W2, %arns2b2W2 : tensor<128x128x3x3xf32>
    %v6205 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6206 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6207 = stablehlo.multiply %v6205, %s2b2W2m : tensor<128x128x3x3xf32>
    %v6208 = stablehlo.multiply %v6206, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v6209 = stablehlo.add %v6207, %v6208 : tensor<128x128x3x3xf32>
    %v6210 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6211 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6212 = stablehlo.multiply %v6210, %s2b2W2v : tensor<128x128x3x3xf32>
    %v6213 = stablehlo.multiply %armeans2b2W2, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v6214 = stablehlo.multiply %v6211, %v6213 : tensor<128x128x3x3xf32>
    %v6215 = stablehlo.add %v6212, %v6214 : tensor<128x128x3x3xf32>
    %v6216 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6217 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6218 = stablehlo.multiply %v6216, %s2b2W2m : tensor<128x128x3x3xf32>
    %v6219 = stablehlo.multiply %v6217, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v6220 = stablehlo.add %v6218, %v6219 : tensor<128x128x3x3xf32>
    %v6221 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6222 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6223 = stablehlo.multiply %v6221, %s2b2W2v : tensor<128x128x3x3xf32>
    %v6224 = stablehlo.multiply %armeans2b2W2, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v6225 = stablehlo.multiply %v6222, %v6224 : tensor<128x128x3x3xf32>
    %v6226 = stablehlo.add %v6223, %v6225 : tensor<128x128x3x3xf32>
    %v6227 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6228 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6229 = stablehlo.divide %v6220, %v6227 : tensor<128x128x3x3xf32>
    %v6230 = stablehlo.divide %v6226, %v6228 : tensor<128x128x3x3xf32>
    %v6231 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6232 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6233 = stablehlo.sqrt %v6230 : tensor<128x128x3x3xf32>
    %v6234 = stablehlo.add %v6233, %v6232 : tensor<128x128x3x3xf32>
    %v6235 = stablehlo.divide %v6229, %v6234 : tensor<128x128x3x3xf32>
    %v6236 = stablehlo.multiply %v6231, %v6235 : tensor<128x128x3x3xf32>
    %v6237 = stablehlo.subtract %s2b2W2, %v6236 : tensor<128x128x3x3xf32>
    %v6238 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6239 = stablehlo.multiply %v6238, %v6231 : tensor<128x128x3x3xf32>
    %v6240 = stablehlo.multiply %v6239, %s2b2W2 : tensor<128x128x3x3xf32>
    %v6241 = stablehlo.subtract %v6237, %v6240 : tensor<128x128x3x3xf32>
    %arsums2b2g2 = "stablehlo.all_reduce"(%v3344) ({
    ^bb0(%aras2b2g2: tensor<f32>, %arbs2b2g2: tensor<f32>):
      %aradds2b2g2 = stablehlo.add %aras2b2g2, %arbs2b2g2 : tensor<f32>
      stablehlo.return %aradds2b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g2 = stablehlo.divide %arsums2b2g2, %arns2b2g2 : tensor<128xf32>
    %v6242 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6243 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6244 = stablehlo.multiply %v6242, %s2b2g2m : tensor<128xf32>
    %v6245 = stablehlo.multiply %v6243, %armeans2b2g2 : tensor<128xf32>
    %v6246 = stablehlo.add %v6244, %v6245 : tensor<128xf32>
    %v6247 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6248 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6249 = stablehlo.multiply %v6247, %s2b2g2v : tensor<128xf32>
    %v6250 = stablehlo.multiply %armeans2b2g2, %armeans2b2g2 : tensor<128xf32>
    %v6251 = stablehlo.multiply %v6248, %v6250 : tensor<128xf32>
    %v6252 = stablehlo.add %v6249, %v6251 : tensor<128xf32>
    %v6253 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6254 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6255 = stablehlo.multiply %v6253, %s2b2g2m : tensor<128xf32>
    %v6256 = stablehlo.multiply %v6254, %armeans2b2g2 : tensor<128xf32>
    %v6257 = stablehlo.add %v6255, %v6256 : tensor<128xf32>
    %v6258 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6259 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6260 = stablehlo.multiply %v6258, %s2b2g2v : tensor<128xf32>
    %v6261 = stablehlo.multiply %armeans2b2g2, %armeans2b2g2 : tensor<128xf32>
    %v6262 = stablehlo.multiply %v6259, %v6261 : tensor<128xf32>
    %v6263 = stablehlo.add %v6260, %v6262 : tensor<128xf32>
    %v6264 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6265 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6266 = stablehlo.divide %v6257, %v6264 : tensor<128xf32>
    %v6267 = stablehlo.divide %v6263, %v6265 : tensor<128xf32>
    %v6268 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6269 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6270 = stablehlo.sqrt %v6267 : tensor<128xf32>
    %v6271 = stablehlo.add %v6270, %v6269 : tensor<128xf32>
    %v6272 = stablehlo.divide %v6266, %v6271 : tensor<128xf32>
    %v6273 = stablehlo.multiply %v6268, %v6272 : tensor<128xf32>
    %v6274 = stablehlo.subtract %s2b2g2, %v6273 : tensor<128xf32>
    %v6275 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6276 = stablehlo.multiply %v6275, %v6268 : tensor<128xf32>
    %v6277 = stablehlo.multiply %v6276, %s2b2g2 : tensor<128xf32>
    %v6278 = stablehlo.subtract %v6274, %v6277 : tensor<128xf32>
    %arsums2b2bt2 = "stablehlo.all_reduce"(%v3347) ({
    ^bb0(%aras2b2bt2: tensor<f32>, %arbs2b2bt2: tensor<f32>):
      %aradds2b2bt2 = stablehlo.add %aras2b2bt2, %arbs2b2bt2 : tensor<f32>
      stablehlo.return %aradds2b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2bt2 = stablehlo.divide %arsums2b2bt2, %arns2b2bt2 : tensor<128xf32>
    %v6279 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6280 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6281 = stablehlo.multiply %v6279, %s2b2bt2m : tensor<128xf32>
    %v6282 = stablehlo.multiply %v6280, %armeans2b2bt2 : tensor<128xf32>
    %v6283 = stablehlo.add %v6281, %v6282 : tensor<128xf32>
    %v6284 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6285 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6286 = stablehlo.multiply %v6284, %s2b2bt2v : tensor<128xf32>
    %v6287 = stablehlo.multiply %armeans2b2bt2, %armeans2b2bt2 : tensor<128xf32>
    %v6288 = stablehlo.multiply %v6285, %v6287 : tensor<128xf32>
    %v6289 = stablehlo.add %v6286, %v6288 : tensor<128xf32>
    %v6290 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6291 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6292 = stablehlo.multiply %v6290, %s2b2bt2m : tensor<128xf32>
    %v6293 = stablehlo.multiply %v6291, %armeans2b2bt2 : tensor<128xf32>
    %v6294 = stablehlo.add %v6292, %v6293 : tensor<128xf32>
    %v6295 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6296 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6297 = stablehlo.multiply %v6295, %s2b2bt2v : tensor<128xf32>
    %v6298 = stablehlo.multiply %armeans2b2bt2, %armeans2b2bt2 : tensor<128xf32>
    %v6299 = stablehlo.multiply %v6296, %v6298 : tensor<128xf32>
    %v6300 = stablehlo.add %v6297, %v6299 : tensor<128xf32>
    %v6301 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6302 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6303 = stablehlo.divide %v6294, %v6301 : tensor<128xf32>
    %v6304 = stablehlo.divide %v6300, %v6302 : tensor<128xf32>
    %v6305 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6306 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6307 = stablehlo.sqrt %v6304 : tensor<128xf32>
    %v6308 = stablehlo.add %v6307, %v6306 : tensor<128xf32>
    %v6309 = stablehlo.divide %v6303, %v6308 : tensor<128xf32>
    %v6310 = stablehlo.multiply %v6305, %v6309 : tensor<128xf32>
    %v6311 = stablehlo.subtract %s2b2bt2, %v6310 : tensor<128xf32>
    %v6312 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6313 = stablehlo.multiply %v6312, %v6305 : tensor<128xf32>
    %v6314 = stablehlo.multiply %v6313, %s2b2bt2 : tensor<128xf32>
    %v6315 = stablehlo.subtract %v6311, %v6314 : tensor<128xf32>
    %arsumd3W1 = "stablehlo.all_reduce"(%v3122) ({
    ^bb0(%arad3W1: tensor<f32>, %arbd3W1: tensor<f32>):
      %araddd3W1 = stablehlo.add %arad3W1, %arbd3W1 : tensor<f32>
      stablehlo.return %araddd3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf32>
    %arnd3W1 = stablehlo.constant dense<2.0> : tensor<256x128x3x3xf32>
    %armeand3W1 = stablehlo.divide %arsumd3W1, %arnd3W1 : tensor<256x128x3x3xf32>
    %v6316 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6317 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6318 = stablehlo.multiply %v6316, %d3W1m : tensor<256x128x3x3xf32>
    %v6319 = stablehlo.multiply %v6317, %armeand3W1 : tensor<256x128x3x3xf32>
    %v6320 = stablehlo.add %v6318, %v6319 : tensor<256x128x3x3xf32>
    %v6321 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6322 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6323 = stablehlo.multiply %v6321, %d3W1v : tensor<256x128x3x3xf32>
    %v6324 = stablehlo.multiply %armeand3W1, %armeand3W1 : tensor<256x128x3x3xf32>
    %v6325 = stablehlo.multiply %v6322, %v6324 : tensor<256x128x3x3xf32>
    %v6326 = stablehlo.add %v6323, %v6325 : tensor<256x128x3x3xf32>
    %v6327 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6328 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6329 = stablehlo.multiply %v6327, %d3W1m : tensor<256x128x3x3xf32>
    %v6330 = stablehlo.multiply %v6328, %armeand3W1 : tensor<256x128x3x3xf32>
    %v6331 = stablehlo.add %v6329, %v6330 : tensor<256x128x3x3xf32>
    %v6332 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6333 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6334 = stablehlo.multiply %v6332, %d3W1v : tensor<256x128x3x3xf32>
    %v6335 = stablehlo.multiply %armeand3W1, %armeand3W1 : tensor<256x128x3x3xf32>
    %v6336 = stablehlo.multiply %v6333, %v6335 : tensor<256x128x3x3xf32>
    %v6337 = stablehlo.add %v6334, %v6336 : tensor<256x128x3x3xf32>
    %v6338 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6339 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6340 = stablehlo.divide %v6331, %v6338 : tensor<256x128x3x3xf32>
    %v6341 = stablehlo.divide %v6337, %v6339 : tensor<256x128x3x3xf32>
    %v6342 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6343 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6344 = stablehlo.sqrt %v6341 : tensor<256x128x3x3xf32>
    %v6345 = stablehlo.add %v6344, %v6343 : tensor<256x128x3x3xf32>
    %v6346 = stablehlo.divide %v6340, %v6345 : tensor<256x128x3x3xf32>
    %v6347 = stablehlo.multiply %v6342, %v6346 : tensor<256x128x3x3xf32>
    %v6348 = stablehlo.subtract %d3W1, %v6347 : tensor<256x128x3x3xf32>
    %v6349 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6350 = stablehlo.multiply %v6349, %v6342 : tensor<256x128x3x3xf32>
    %v6351 = stablehlo.multiply %v6350, %d3W1 : tensor<256x128x3x3xf32>
    %v6352 = stablehlo.subtract %v6348, %v6351 : tensor<256x128x3x3xf32>
    %arsumd3g1 = "stablehlo.all_reduce"(%v3136) ({
    ^bb0(%arad3g1: tensor<f32>, %arbd3g1: tensor<f32>):
      %araddd3g1 = stablehlo.add %arad3g1, %arbd3g1 : tensor<f32>
      stablehlo.return %araddd3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g1 = stablehlo.divide %arsumd3g1, %arnd3g1 : tensor<256xf32>
    %v6353 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6354 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6355 = stablehlo.multiply %v6353, %d3g1m : tensor<256xf32>
    %v6356 = stablehlo.multiply %v6354, %armeand3g1 : tensor<256xf32>
    %v6357 = stablehlo.add %v6355, %v6356 : tensor<256xf32>
    %v6358 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6359 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6360 = stablehlo.multiply %v6358, %d3g1v : tensor<256xf32>
    %v6361 = stablehlo.multiply %armeand3g1, %armeand3g1 : tensor<256xf32>
    %v6362 = stablehlo.multiply %v6359, %v6361 : tensor<256xf32>
    %v6363 = stablehlo.add %v6360, %v6362 : tensor<256xf32>
    %v6364 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6365 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6366 = stablehlo.multiply %v6364, %d3g1m : tensor<256xf32>
    %v6367 = stablehlo.multiply %v6365, %armeand3g1 : tensor<256xf32>
    %v6368 = stablehlo.add %v6366, %v6367 : tensor<256xf32>
    %v6369 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6370 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6371 = stablehlo.multiply %v6369, %d3g1v : tensor<256xf32>
    %v6372 = stablehlo.multiply %armeand3g1, %armeand3g1 : tensor<256xf32>
    %v6373 = stablehlo.multiply %v6370, %v6372 : tensor<256xf32>
    %v6374 = stablehlo.add %v6371, %v6373 : tensor<256xf32>
    %v6375 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6376 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6377 = stablehlo.divide %v6368, %v6375 : tensor<256xf32>
    %v6378 = stablehlo.divide %v6374, %v6376 : tensor<256xf32>
    %v6379 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6380 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6381 = stablehlo.sqrt %v6378 : tensor<256xf32>
    %v6382 = stablehlo.add %v6381, %v6380 : tensor<256xf32>
    %v6383 = stablehlo.divide %v6377, %v6382 : tensor<256xf32>
    %v6384 = stablehlo.multiply %v6379, %v6383 : tensor<256xf32>
    %v6385 = stablehlo.subtract %d3g1, %v6384 : tensor<256xf32>
    %v6386 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6387 = stablehlo.multiply %v6386, %v6379 : tensor<256xf32>
    %v6388 = stablehlo.multiply %v6387, %d3g1 : tensor<256xf32>
    %v6389 = stablehlo.subtract %v6385, %v6388 : tensor<256xf32>
    %arsumd3bt1 = "stablehlo.all_reduce"(%v3139) ({
    ^bb0(%arad3bt1: tensor<f32>, %arbd3bt1: tensor<f32>):
      %araddd3bt1 = stablehlo.add %arad3bt1, %arbd3bt1 : tensor<f32>
      stablehlo.return %araddd3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3bt1 = stablehlo.divide %arsumd3bt1, %arnd3bt1 : tensor<256xf32>
    %v6390 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6391 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6392 = stablehlo.multiply %v6390, %d3bt1m : tensor<256xf32>
    %v6393 = stablehlo.multiply %v6391, %armeand3bt1 : tensor<256xf32>
    %v6394 = stablehlo.add %v6392, %v6393 : tensor<256xf32>
    %v6395 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6396 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6397 = stablehlo.multiply %v6395, %d3bt1v : tensor<256xf32>
    %v6398 = stablehlo.multiply %armeand3bt1, %armeand3bt1 : tensor<256xf32>
    %v6399 = stablehlo.multiply %v6396, %v6398 : tensor<256xf32>
    %v6400 = stablehlo.add %v6397, %v6399 : tensor<256xf32>
    %v6401 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6402 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6403 = stablehlo.multiply %v6401, %d3bt1m : tensor<256xf32>
    %v6404 = stablehlo.multiply %v6402, %armeand3bt1 : tensor<256xf32>
    %v6405 = stablehlo.add %v6403, %v6404 : tensor<256xf32>
    %v6406 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6407 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6408 = stablehlo.multiply %v6406, %d3bt1v : tensor<256xf32>
    %v6409 = stablehlo.multiply %armeand3bt1, %armeand3bt1 : tensor<256xf32>
    %v6410 = stablehlo.multiply %v6407, %v6409 : tensor<256xf32>
    %v6411 = stablehlo.add %v6408, %v6410 : tensor<256xf32>
    %v6412 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6413 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6414 = stablehlo.divide %v6405, %v6412 : tensor<256xf32>
    %v6415 = stablehlo.divide %v6411, %v6413 : tensor<256xf32>
    %v6416 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6417 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6418 = stablehlo.sqrt %v6415 : tensor<256xf32>
    %v6419 = stablehlo.add %v6418, %v6417 : tensor<256xf32>
    %v6420 = stablehlo.divide %v6414, %v6419 : tensor<256xf32>
    %v6421 = stablehlo.multiply %v6416, %v6420 : tensor<256xf32>
    %v6422 = stablehlo.subtract %d3bt1, %v6421 : tensor<256xf32>
    %v6423 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6424 = stablehlo.multiply %v6423, %v6416 : tensor<256xf32>
    %v6425 = stablehlo.multiply %v6424, %d3bt1 : tensor<256xf32>
    %v6426 = stablehlo.subtract %v6422, %v6425 : tensor<256xf32>
    %arsumd3W2 = "stablehlo.all_reduce"(%v3145) ({
    ^bb0(%arad3W2: tensor<f32>, %arbd3W2: tensor<f32>):
      %araddd3W2 = stablehlo.add %arad3W2, %arbd3W2 : tensor<f32>
      stablehlo.return %araddd3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arnd3W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeand3W2 = stablehlo.divide %arsumd3W2, %arnd3W2 : tensor<256x256x3x3xf32>
    %v6427 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6428 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6429 = stablehlo.multiply %v6427, %d3W2m : tensor<256x256x3x3xf32>
    %v6430 = stablehlo.multiply %v6428, %armeand3W2 : tensor<256x256x3x3xf32>
    %v6431 = stablehlo.add %v6429, %v6430 : tensor<256x256x3x3xf32>
    %v6432 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6433 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6434 = stablehlo.multiply %v6432, %d3W2v : tensor<256x256x3x3xf32>
    %v6435 = stablehlo.multiply %armeand3W2, %armeand3W2 : tensor<256x256x3x3xf32>
    %v6436 = stablehlo.multiply %v6433, %v6435 : tensor<256x256x3x3xf32>
    %v6437 = stablehlo.add %v6434, %v6436 : tensor<256x256x3x3xf32>
    %v6438 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6439 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6440 = stablehlo.multiply %v6438, %d3W2m : tensor<256x256x3x3xf32>
    %v6441 = stablehlo.multiply %v6439, %armeand3W2 : tensor<256x256x3x3xf32>
    %v6442 = stablehlo.add %v6440, %v6441 : tensor<256x256x3x3xf32>
    %v6443 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6444 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6445 = stablehlo.multiply %v6443, %d3W2v : tensor<256x256x3x3xf32>
    %v6446 = stablehlo.multiply %armeand3W2, %armeand3W2 : tensor<256x256x3x3xf32>
    %v6447 = stablehlo.multiply %v6444, %v6446 : tensor<256x256x3x3xf32>
    %v6448 = stablehlo.add %v6445, %v6447 : tensor<256x256x3x3xf32>
    %v6449 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6450 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6451 = stablehlo.divide %v6442, %v6449 : tensor<256x256x3x3xf32>
    %v6452 = stablehlo.divide %v6448, %v6450 : tensor<256x256x3x3xf32>
    %v6453 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6454 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6455 = stablehlo.sqrt %v6452 : tensor<256x256x3x3xf32>
    %v6456 = stablehlo.add %v6455, %v6454 : tensor<256x256x3x3xf32>
    %v6457 = stablehlo.divide %v6451, %v6456 : tensor<256x256x3x3xf32>
    %v6458 = stablehlo.multiply %v6453, %v6457 : tensor<256x256x3x3xf32>
    %v6459 = stablehlo.subtract %d3W2, %v6458 : tensor<256x256x3x3xf32>
    %v6460 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6461 = stablehlo.multiply %v6460, %v6453 : tensor<256x256x3x3xf32>
    %v6462 = stablehlo.multiply %v6461, %d3W2 : tensor<256x256x3x3xf32>
    %v6463 = stablehlo.subtract %v6459, %v6462 : tensor<256x256x3x3xf32>
    %arsumd3g2 = "stablehlo.all_reduce"(%v3159) ({
    ^bb0(%arad3g2: tensor<f32>, %arbd3g2: tensor<f32>):
      %araddd3g2 = stablehlo.add %arad3g2, %arbd3g2 : tensor<f32>
      stablehlo.return %araddd3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g2 = stablehlo.divide %arsumd3g2, %arnd3g2 : tensor<256xf32>
    %v6464 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6465 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6466 = stablehlo.multiply %v6464, %d3g2m : tensor<256xf32>
    %v6467 = stablehlo.multiply %v6465, %armeand3g2 : tensor<256xf32>
    %v6468 = stablehlo.add %v6466, %v6467 : tensor<256xf32>
    %v6469 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6470 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6471 = stablehlo.multiply %v6469, %d3g2v : tensor<256xf32>
    %v6472 = stablehlo.multiply %armeand3g2, %armeand3g2 : tensor<256xf32>
    %v6473 = stablehlo.multiply %v6470, %v6472 : tensor<256xf32>
    %v6474 = stablehlo.add %v6471, %v6473 : tensor<256xf32>
    %v6475 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6476 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6477 = stablehlo.multiply %v6475, %d3g2m : tensor<256xf32>
    %v6478 = stablehlo.multiply %v6476, %armeand3g2 : tensor<256xf32>
    %v6479 = stablehlo.add %v6477, %v6478 : tensor<256xf32>
    %v6480 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6481 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6482 = stablehlo.multiply %v6480, %d3g2v : tensor<256xf32>
    %v6483 = stablehlo.multiply %armeand3g2, %armeand3g2 : tensor<256xf32>
    %v6484 = stablehlo.multiply %v6481, %v6483 : tensor<256xf32>
    %v6485 = stablehlo.add %v6482, %v6484 : tensor<256xf32>
    %v6486 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6487 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6488 = stablehlo.divide %v6479, %v6486 : tensor<256xf32>
    %v6489 = stablehlo.divide %v6485, %v6487 : tensor<256xf32>
    %v6490 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6491 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6492 = stablehlo.sqrt %v6489 : tensor<256xf32>
    %v6493 = stablehlo.add %v6492, %v6491 : tensor<256xf32>
    %v6494 = stablehlo.divide %v6488, %v6493 : tensor<256xf32>
    %v6495 = stablehlo.multiply %v6490, %v6494 : tensor<256xf32>
    %v6496 = stablehlo.subtract %d3g2, %v6495 : tensor<256xf32>
    %v6497 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6498 = stablehlo.multiply %v6497, %v6490 : tensor<256xf32>
    %v6499 = stablehlo.multiply %v6498, %d3g2 : tensor<256xf32>
    %v6500 = stablehlo.subtract %v6496, %v6499 : tensor<256xf32>
    %arsumd3bt2 = "stablehlo.all_reduce"(%v3162) ({
    ^bb0(%arad3bt2: tensor<f32>, %arbd3bt2: tensor<f32>):
      %araddd3bt2 = stablehlo.add %arad3bt2, %arbd3bt2 : tensor<f32>
      stablehlo.return %araddd3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3bt2 = stablehlo.divide %arsumd3bt2, %arnd3bt2 : tensor<256xf32>
    %v6501 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6502 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6503 = stablehlo.multiply %v6501, %d3bt2m : tensor<256xf32>
    %v6504 = stablehlo.multiply %v6502, %armeand3bt2 : tensor<256xf32>
    %v6505 = stablehlo.add %v6503, %v6504 : tensor<256xf32>
    %v6506 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6507 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6508 = stablehlo.multiply %v6506, %d3bt2v : tensor<256xf32>
    %v6509 = stablehlo.multiply %armeand3bt2, %armeand3bt2 : tensor<256xf32>
    %v6510 = stablehlo.multiply %v6507, %v6509 : tensor<256xf32>
    %v6511 = stablehlo.add %v6508, %v6510 : tensor<256xf32>
    %v6512 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6513 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6514 = stablehlo.multiply %v6512, %d3bt2m : tensor<256xf32>
    %v6515 = stablehlo.multiply %v6513, %armeand3bt2 : tensor<256xf32>
    %v6516 = stablehlo.add %v6514, %v6515 : tensor<256xf32>
    %v6517 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6518 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6519 = stablehlo.multiply %v6517, %d3bt2v : tensor<256xf32>
    %v6520 = stablehlo.multiply %armeand3bt2, %armeand3bt2 : tensor<256xf32>
    %v6521 = stablehlo.multiply %v6518, %v6520 : tensor<256xf32>
    %v6522 = stablehlo.add %v6519, %v6521 : tensor<256xf32>
    %v6523 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6524 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6525 = stablehlo.divide %v6516, %v6523 : tensor<256xf32>
    %v6526 = stablehlo.divide %v6522, %v6524 : tensor<256xf32>
    %v6527 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6528 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6529 = stablehlo.sqrt %v6526 : tensor<256xf32>
    %v6530 = stablehlo.add %v6529, %v6528 : tensor<256xf32>
    %v6531 = stablehlo.divide %v6525, %v6530 : tensor<256xf32>
    %v6532 = stablehlo.multiply %v6527, %v6531 : tensor<256xf32>
    %v6533 = stablehlo.subtract %d3bt2, %v6532 : tensor<256xf32>
    %v6534 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6535 = stablehlo.multiply %v6534, %v6527 : tensor<256xf32>
    %v6536 = stablehlo.multiply %v6535, %d3bt2 : tensor<256xf32>
    %v6537 = stablehlo.subtract %v6533, %v6536 : tensor<256xf32>
    %arsumd3Wp = "stablehlo.all_reduce"(%v3170) ({
    ^bb0(%arad3Wp: tensor<f32>, %arbd3Wp: tensor<f32>):
      %araddd3Wp = stablehlo.add %arad3Wp, %arbd3Wp : tensor<f32>
      stablehlo.return %araddd3Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf32>
    %arnd3Wp = stablehlo.constant dense<2.0> : tensor<256x128x1x1xf32>
    %armeand3Wp = stablehlo.divide %arsumd3Wp, %arnd3Wp : tensor<256x128x1x1xf32>
    %v6538 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6539 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6540 = stablehlo.multiply %v6538, %d3Wpm : tensor<256x128x1x1xf32>
    %v6541 = stablehlo.multiply %v6539, %armeand3Wp : tensor<256x128x1x1xf32>
    %v6542 = stablehlo.add %v6540, %v6541 : tensor<256x128x1x1xf32>
    %v6543 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6544 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6545 = stablehlo.multiply %v6543, %d3Wpv : tensor<256x128x1x1xf32>
    %v6546 = stablehlo.multiply %armeand3Wp, %armeand3Wp : tensor<256x128x1x1xf32>
    %v6547 = stablehlo.multiply %v6544, %v6546 : tensor<256x128x1x1xf32>
    %v6548 = stablehlo.add %v6545, %v6547 : tensor<256x128x1x1xf32>
    %v6549 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6550 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6551 = stablehlo.multiply %v6549, %d3Wpm : tensor<256x128x1x1xf32>
    %v6552 = stablehlo.multiply %v6550, %armeand3Wp : tensor<256x128x1x1xf32>
    %v6553 = stablehlo.add %v6551, %v6552 : tensor<256x128x1x1xf32>
    %v6554 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6555 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6556 = stablehlo.multiply %v6554, %d3Wpv : tensor<256x128x1x1xf32>
    %v6557 = stablehlo.multiply %armeand3Wp, %armeand3Wp : tensor<256x128x1x1xf32>
    %v6558 = stablehlo.multiply %v6555, %v6557 : tensor<256x128x1x1xf32>
    %v6559 = stablehlo.add %v6556, %v6558 : tensor<256x128x1x1xf32>
    %v6560 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6561 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6562 = stablehlo.divide %v6553, %v6560 : tensor<256x128x1x1xf32>
    %v6563 = stablehlo.divide %v6559, %v6561 : tensor<256x128x1x1xf32>
    %v6564 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6565 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6566 = stablehlo.sqrt %v6563 : tensor<256x128x1x1xf32>
    %v6567 = stablehlo.add %v6566, %v6565 : tensor<256x128x1x1xf32>
    %v6568 = stablehlo.divide %v6562, %v6567 : tensor<256x128x1x1xf32>
    %v6569 = stablehlo.multiply %v6564, %v6568 : tensor<256x128x1x1xf32>
    %v6570 = stablehlo.subtract %d3Wp, %v6569 : tensor<256x128x1x1xf32>
    %v6571 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v6572 = stablehlo.multiply %v6571, %v6564 : tensor<256x128x1x1xf32>
    %v6573 = stablehlo.multiply %v6572, %d3Wp : tensor<256x128x1x1xf32>
    %v6574 = stablehlo.subtract %v6570, %v6573 : tensor<256x128x1x1xf32>
    %arsumd3gp = "stablehlo.all_reduce"(%v3184) ({
    ^bb0(%arad3gp: tensor<f32>, %arbd3gp: tensor<f32>):
      %araddd3gp = stablehlo.add %arad3gp, %arbd3gp : tensor<f32>
      stablehlo.return %araddd3gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3gp = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3gp = stablehlo.divide %arsumd3gp, %arnd3gp : tensor<256xf32>
    %v6575 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6576 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6577 = stablehlo.multiply %v6575, %d3gpm : tensor<256xf32>
    %v6578 = stablehlo.multiply %v6576, %armeand3gp : tensor<256xf32>
    %v6579 = stablehlo.add %v6577, %v6578 : tensor<256xf32>
    %v6580 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6581 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6582 = stablehlo.multiply %v6580, %d3gpv : tensor<256xf32>
    %v6583 = stablehlo.multiply %armeand3gp, %armeand3gp : tensor<256xf32>
    %v6584 = stablehlo.multiply %v6581, %v6583 : tensor<256xf32>
    %v6585 = stablehlo.add %v6582, %v6584 : tensor<256xf32>
    %v6586 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6587 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6588 = stablehlo.multiply %v6586, %d3gpm : tensor<256xf32>
    %v6589 = stablehlo.multiply %v6587, %armeand3gp : tensor<256xf32>
    %v6590 = stablehlo.add %v6588, %v6589 : tensor<256xf32>
    %v6591 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6592 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6593 = stablehlo.multiply %v6591, %d3gpv : tensor<256xf32>
    %v6594 = stablehlo.multiply %armeand3gp, %armeand3gp : tensor<256xf32>
    %v6595 = stablehlo.multiply %v6592, %v6594 : tensor<256xf32>
    %v6596 = stablehlo.add %v6593, %v6595 : tensor<256xf32>
    %v6597 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6598 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6599 = stablehlo.divide %v6590, %v6597 : tensor<256xf32>
    %v6600 = stablehlo.divide %v6596, %v6598 : tensor<256xf32>
    %v6601 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6602 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6603 = stablehlo.sqrt %v6600 : tensor<256xf32>
    %v6604 = stablehlo.add %v6603, %v6602 : tensor<256xf32>
    %v6605 = stablehlo.divide %v6599, %v6604 : tensor<256xf32>
    %v6606 = stablehlo.multiply %v6601, %v6605 : tensor<256xf32>
    %v6607 = stablehlo.subtract %d3gp, %v6606 : tensor<256xf32>
    %v6608 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6609 = stablehlo.multiply %v6608, %v6601 : tensor<256xf32>
    %v6610 = stablehlo.multiply %v6609, %d3gp : tensor<256xf32>
    %v6611 = stablehlo.subtract %v6607, %v6610 : tensor<256xf32>
    %arsumd3btp = "stablehlo.all_reduce"(%v3187) ({
    ^bb0(%arad3btp: tensor<f32>, %arbd3btp: tensor<f32>):
      %araddd3btp = stablehlo.add %arad3btp, %arbd3btp : tensor<f32>
      stablehlo.return %araddd3btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3btp = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3btp = stablehlo.divide %arsumd3btp, %arnd3btp : tensor<256xf32>
    %v6612 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6613 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6614 = stablehlo.multiply %v6612, %d3btpm : tensor<256xf32>
    %v6615 = stablehlo.multiply %v6613, %armeand3btp : tensor<256xf32>
    %v6616 = stablehlo.add %v6614, %v6615 : tensor<256xf32>
    %v6617 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6618 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6619 = stablehlo.multiply %v6617, %d3btpv : tensor<256xf32>
    %v6620 = stablehlo.multiply %armeand3btp, %armeand3btp : tensor<256xf32>
    %v6621 = stablehlo.multiply %v6618, %v6620 : tensor<256xf32>
    %v6622 = stablehlo.add %v6619, %v6621 : tensor<256xf32>
    %v6623 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6624 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6625 = stablehlo.multiply %v6623, %d3btpm : tensor<256xf32>
    %v6626 = stablehlo.multiply %v6624, %armeand3btp : tensor<256xf32>
    %v6627 = stablehlo.add %v6625, %v6626 : tensor<256xf32>
    %v6628 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6629 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6630 = stablehlo.multiply %v6628, %d3btpv : tensor<256xf32>
    %v6631 = stablehlo.multiply %armeand3btp, %armeand3btp : tensor<256xf32>
    %v6632 = stablehlo.multiply %v6629, %v6631 : tensor<256xf32>
    %v6633 = stablehlo.add %v6630, %v6632 : tensor<256xf32>
    %v6634 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6635 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6636 = stablehlo.divide %v6627, %v6634 : tensor<256xf32>
    %v6637 = stablehlo.divide %v6633, %v6635 : tensor<256xf32>
    %v6638 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6639 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6640 = stablehlo.sqrt %v6637 : tensor<256xf32>
    %v6641 = stablehlo.add %v6640, %v6639 : tensor<256xf32>
    %v6642 = stablehlo.divide %v6636, %v6641 : tensor<256xf32>
    %v6643 = stablehlo.multiply %v6638, %v6642 : tensor<256xf32>
    %v6644 = stablehlo.subtract %d3btp, %v6643 : tensor<256xf32>
    %v6645 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6646 = stablehlo.multiply %v6645, %v6638 : tensor<256xf32>
    %v6647 = stablehlo.multiply %v6646, %d3btp : tensor<256xf32>
    %v6648 = stablehlo.subtract %v6644, %v6647 : tensor<256xf32>
    %arsums3b0W1 = "stablehlo.all_reduce"(%v2907) ({
    ^bb0(%aras3b0W1: tensor<f32>, %arbs3b0W1: tensor<f32>):
      %aradds3b0W1 = stablehlo.add %aras3b0W1, %arbs3b0W1 : tensor<f32>
      stablehlo.return %aradds3b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b0W1 = stablehlo.divide %arsums3b0W1, %arns3b0W1 : tensor<256x256x3x3xf32>
    %v6649 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6650 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6651 = stablehlo.multiply %v6649, %s3b0W1m : tensor<256x256x3x3xf32>
    %v6652 = stablehlo.multiply %v6650, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6653 = stablehlo.add %v6651, %v6652 : tensor<256x256x3x3xf32>
    %v6654 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6655 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6656 = stablehlo.multiply %v6654, %s3b0W1v : tensor<256x256x3x3xf32>
    %v6657 = stablehlo.multiply %armeans3b0W1, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6658 = stablehlo.multiply %v6655, %v6657 : tensor<256x256x3x3xf32>
    %v6659 = stablehlo.add %v6656, %v6658 : tensor<256x256x3x3xf32>
    %v6660 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6661 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6662 = stablehlo.multiply %v6660, %s3b0W1m : tensor<256x256x3x3xf32>
    %v6663 = stablehlo.multiply %v6661, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6664 = stablehlo.add %v6662, %v6663 : tensor<256x256x3x3xf32>
    %v6665 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6666 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6667 = stablehlo.multiply %v6665, %s3b0W1v : tensor<256x256x3x3xf32>
    %v6668 = stablehlo.multiply %armeans3b0W1, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v6669 = stablehlo.multiply %v6666, %v6668 : tensor<256x256x3x3xf32>
    %v6670 = stablehlo.add %v6667, %v6669 : tensor<256x256x3x3xf32>
    %v6671 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6672 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6673 = stablehlo.divide %v6664, %v6671 : tensor<256x256x3x3xf32>
    %v6674 = stablehlo.divide %v6670, %v6672 : tensor<256x256x3x3xf32>
    %v6675 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6676 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6677 = stablehlo.sqrt %v6674 : tensor<256x256x3x3xf32>
    %v6678 = stablehlo.add %v6677, %v6676 : tensor<256x256x3x3xf32>
    %v6679 = stablehlo.divide %v6673, %v6678 : tensor<256x256x3x3xf32>
    %v6680 = stablehlo.multiply %v6675, %v6679 : tensor<256x256x3x3xf32>
    %v6681 = stablehlo.subtract %s3b0W1, %v6680 : tensor<256x256x3x3xf32>
    %v6682 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6683 = stablehlo.multiply %v6682, %v6675 : tensor<256x256x3x3xf32>
    %v6684 = stablehlo.multiply %v6683, %s3b0W1 : tensor<256x256x3x3xf32>
    %v6685 = stablehlo.subtract %v6681, %v6684 : tensor<256x256x3x3xf32>
    %arsums3b0g1 = "stablehlo.all_reduce"(%v2921) ({
    ^bb0(%aras3b0g1: tensor<f32>, %arbs3b0g1: tensor<f32>):
      %aradds3b0g1 = stablehlo.add %aras3b0g1, %arbs3b0g1 : tensor<f32>
      stablehlo.return %aradds3b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g1 = stablehlo.divide %arsums3b0g1, %arns3b0g1 : tensor<256xf32>
    %v6686 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6687 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6688 = stablehlo.multiply %v6686, %s3b0g1m : tensor<256xf32>
    %v6689 = stablehlo.multiply %v6687, %armeans3b0g1 : tensor<256xf32>
    %v6690 = stablehlo.add %v6688, %v6689 : tensor<256xf32>
    %v6691 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6692 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6693 = stablehlo.multiply %v6691, %s3b0g1v : tensor<256xf32>
    %v6694 = stablehlo.multiply %armeans3b0g1, %armeans3b0g1 : tensor<256xf32>
    %v6695 = stablehlo.multiply %v6692, %v6694 : tensor<256xf32>
    %v6696 = stablehlo.add %v6693, %v6695 : tensor<256xf32>
    %v6697 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6698 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6699 = stablehlo.multiply %v6697, %s3b0g1m : tensor<256xf32>
    %v6700 = stablehlo.multiply %v6698, %armeans3b0g1 : tensor<256xf32>
    %v6701 = stablehlo.add %v6699, %v6700 : tensor<256xf32>
    %v6702 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6703 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6704 = stablehlo.multiply %v6702, %s3b0g1v : tensor<256xf32>
    %v6705 = stablehlo.multiply %armeans3b0g1, %armeans3b0g1 : tensor<256xf32>
    %v6706 = stablehlo.multiply %v6703, %v6705 : tensor<256xf32>
    %v6707 = stablehlo.add %v6704, %v6706 : tensor<256xf32>
    %v6708 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6709 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6710 = stablehlo.divide %v6701, %v6708 : tensor<256xf32>
    %v6711 = stablehlo.divide %v6707, %v6709 : tensor<256xf32>
    %v6712 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6713 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6714 = stablehlo.sqrt %v6711 : tensor<256xf32>
    %v6715 = stablehlo.add %v6714, %v6713 : tensor<256xf32>
    %v6716 = stablehlo.divide %v6710, %v6715 : tensor<256xf32>
    %v6717 = stablehlo.multiply %v6712, %v6716 : tensor<256xf32>
    %v6718 = stablehlo.subtract %s3b0g1, %v6717 : tensor<256xf32>
    %v6719 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6720 = stablehlo.multiply %v6719, %v6712 : tensor<256xf32>
    %v6721 = stablehlo.multiply %v6720, %s3b0g1 : tensor<256xf32>
    %v6722 = stablehlo.subtract %v6718, %v6721 : tensor<256xf32>
    %arsums3b0bt1 = "stablehlo.all_reduce"(%v2924) ({
    ^bb0(%aras3b0bt1: tensor<f32>, %arbs3b0bt1: tensor<f32>):
      %aradds3b0bt1 = stablehlo.add %aras3b0bt1, %arbs3b0bt1 : tensor<f32>
      stablehlo.return %aradds3b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0bt1 = stablehlo.divide %arsums3b0bt1, %arns3b0bt1 : tensor<256xf32>
    %v6723 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6724 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6725 = stablehlo.multiply %v6723, %s3b0bt1m : tensor<256xf32>
    %v6726 = stablehlo.multiply %v6724, %armeans3b0bt1 : tensor<256xf32>
    %v6727 = stablehlo.add %v6725, %v6726 : tensor<256xf32>
    %v6728 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6729 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6730 = stablehlo.multiply %v6728, %s3b0bt1v : tensor<256xf32>
    %v6731 = stablehlo.multiply %armeans3b0bt1, %armeans3b0bt1 : tensor<256xf32>
    %v6732 = stablehlo.multiply %v6729, %v6731 : tensor<256xf32>
    %v6733 = stablehlo.add %v6730, %v6732 : tensor<256xf32>
    %v6734 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6735 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6736 = stablehlo.multiply %v6734, %s3b0bt1m : tensor<256xf32>
    %v6737 = stablehlo.multiply %v6735, %armeans3b0bt1 : tensor<256xf32>
    %v6738 = stablehlo.add %v6736, %v6737 : tensor<256xf32>
    %v6739 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6740 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6741 = stablehlo.multiply %v6739, %s3b0bt1v : tensor<256xf32>
    %v6742 = stablehlo.multiply %armeans3b0bt1, %armeans3b0bt1 : tensor<256xf32>
    %v6743 = stablehlo.multiply %v6740, %v6742 : tensor<256xf32>
    %v6744 = stablehlo.add %v6741, %v6743 : tensor<256xf32>
    %v6745 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6746 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6747 = stablehlo.divide %v6738, %v6745 : tensor<256xf32>
    %v6748 = stablehlo.divide %v6744, %v6746 : tensor<256xf32>
    %v6749 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6750 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6751 = stablehlo.sqrt %v6748 : tensor<256xf32>
    %v6752 = stablehlo.add %v6751, %v6750 : tensor<256xf32>
    %v6753 = stablehlo.divide %v6747, %v6752 : tensor<256xf32>
    %v6754 = stablehlo.multiply %v6749, %v6753 : tensor<256xf32>
    %v6755 = stablehlo.subtract %s3b0bt1, %v6754 : tensor<256xf32>
    %v6756 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6757 = stablehlo.multiply %v6756, %v6749 : tensor<256xf32>
    %v6758 = stablehlo.multiply %v6757, %s3b0bt1 : tensor<256xf32>
    %v6759 = stablehlo.subtract %v6755, %v6758 : tensor<256xf32>
    %arsums3b0W2 = "stablehlo.all_reduce"(%v2930) ({
    ^bb0(%aras3b0W2: tensor<f32>, %arbs3b0W2: tensor<f32>):
      %aradds3b0W2 = stablehlo.add %aras3b0W2, %arbs3b0W2 : tensor<f32>
      stablehlo.return %aradds3b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b0W2 = stablehlo.divide %arsums3b0W2, %arns3b0W2 : tensor<256x256x3x3xf32>
    %v6760 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6761 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6762 = stablehlo.multiply %v6760, %s3b0W2m : tensor<256x256x3x3xf32>
    %v6763 = stablehlo.multiply %v6761, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6764 = stablehlo.add %v6762, %v6763 : tensor<256x256x3x3xf32>
    %v6765 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6766 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6767 = stablehlo.multiply %v6765, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6768 = stablehlo.multiply %armeans3b0W2, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6769 = stablehlo.multiply %v6766, %v6768 : tensor<256x256x3x3xf32>
    %v6770 = stablehlo.add %v6767, %v6769 : tensor<256x256x3x3xf32>
    %v6771 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6772 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6773 = stablehlo.multiply %v6771, %s3b0W2m : tensor<256x256x3x3xf32>
    %v6774 = stablehlo.multiply %v6772, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6775 = stablehlo.add %v6773, %v6774 : tensor<256x256x3x3xf32>
    %v6776 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6777 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6778 = stablehlo.multiply %v6776, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6779 = stablehlo.multiply %armeans3b0W2, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6780 = stablehlo.multiply %v6777, %v6779 : tensor<256x256x3x3xf32>
    %v6781 = stablehlo.add %v6778, %v6780 : tensor<256x256x3x3xf32>
    %v6782 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6783 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6784 = stablehlo.divide %v6775, %v6782 : tensor<256x256x3x3xf32>
    %v6785 = stablehlo.divide %v6781, %v6783 : tensor<256x256x3x3xf32>
    %v6786 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6787 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6788 = stablehlo.sqrt %v6785 : tensor<256x256x3x3xf32>
    %v6789 = stablehlo.add %v6788, %v6787 : tensor<256x256x3x3xf32>
    %v6790 = stablehlo.divide %v6784, %v6789 : tensor<256x256x3x3xf32>
    %v6791 = stablehlo.multiply %v6786, %v6790 : tensor<256x256x3x3xf32>
    %v6792 = stablehlo.subtract %s3b0W2, %v6791 : tensor<256x256x3x3xf32>
    %v6793 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6794 = stablehlo.multiply %v6793, %v6786 : tensor<256x256x3x3xf32>
    %v6795 = stablehlo.multiply %v6794, %s3b0W2 : tensor<256x256x3x3xf32>
    %v6796 = stablehlo.subtract %v6792, %v6795 : tensor<256x256x3x3xf32>
    %arsums3b0g2 = "stablehlo.all_reduce"(%v2944) ({
    ^bb0(%aras3b0g2: tensor<f32>, %arbs3b0g2: tensor<f32>):
      %aradds3b0g2 = stablehlo.add %aras3b0g2, %arbs3b0g2 : tensor<f32>
      stablehlo.return %aradds3b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g2 = stablehlo.divide %arsums3b0g2, %arns3b0g2 : tensor<256xf32>
    %v6797 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6798 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6799 = stablehlo.multiply %v6797, %s3b0g2m : tensor<256xf32>
    %v6800 = stablehlo.multiply %v6798, %armeans3b0g2 : tensor<256xf32>
    %v6801 = stablehlo.add %v6799, %v6800 : tensor<256xf32>
    %v6802 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6803 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6804 = stablehlo.multiply %v6802, %s3b0g2v : tensor<256xf32>
    %v6805 = stablehlo.multiply %armeans3b0g2, %armeans3b0g2 : tensor<256xf32>
    %v6806 = stablehlo.multiply %v6803, %v6805 : tensor<256xf32>
    %v6807 = stablehlo.add %v6804, %v6806 : tensor<256xf32>
    %v6808 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6809 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6810 = stablehlo.multiply %v6808, %s3b0g2m : tensor<256xf32>
    %v6811 = stablehlo.multiply %v6809, %armeans3b0g2 : tensor<256xf32>
    %v6812 = stablehlo.add %v6810, %v6811 : tensor<256xf32>
    %v6813 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6814 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6815 = stablehlo.multiply %v6813, %s3b0g2v : tensor<256xf32>
    %v6816 = stablehlo.multiply %armeans3b0g2, %armeans3b0g2 : tensor<256xf32>
    %v6817 = stablehlo.multiply %v6814, %v6816 : tensor<256xf32>
    %v6818 = stablehlo.add %v6815, %v6817 : tensor<256xf32>
    %v6819 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6820 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6821 = stablehlo.divide %v6812, %v6819 : tensor<256xf32>
    %v6822 = stablehlo.divide %v6818, %v6820 : tensor<256xf32>
    %v6823 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6824 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6825 = stablehlo.sqrt %v6822 : tensor<256xf32>
    %v6826 = stablehlo.add %v6825, %v6824 : tensor<256xf32>
    %v6827 = stablehlo.divide %v6821, %v6826 : tensor<256xf32>
    %v6828 = stablehlo.multiply %v6823, %v6827 : tensor<256xf32>
    %v6829 = stablehlo.subtract %s3b0g2, %v6828 : tensor<256xf32>
    %v6830 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6831 = stablehlo.multiply %v6830, %v6823 : tensor<256xf32>
    %v6832 = stablehlo.multiply %v6831, %s3b0g2 : tensor<256xf32>
    %v6833 = stablehlo.subtract %v6829, %v6832 : tensor<256xf32>
    %arsums3b0bt2 = "stablehlo.all_reduce"(%v2947) ({
    ^bb0(%aras3b0bt2: tensor<f32>, %arbs3b0bt2: tensor<f32>):
      %aradds3b0bt2 = stablehlo.add %aras3b0bt2, %arbs3b0bt2 : tensor<f32>
      stablehlo.return %aradds3b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0bt2 = stablehlo.divide %arsums3b0bt2, %arns3b0bt2 : tensor<256xf32>
    %v6834 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6835 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6836 = stablehlo.multiply %v6834, %s3b0bt2m : tensor<256xf32>
    %v6837 = stablehlo.multiply %v6835, %armeans3b0bt2 : tensor<256xf32>
    %v6838 = stablehlo.add %v6836, %v6837 : tensor<256xf32>
    %v6839 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6840 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6841 = stablehlo.multiply %v6839, %s3b0bt2v : tensor<256xf32>
    %v6842 = stablehlo.multiply %armeans3b0bt2, %armeans3b0bt2 : tensor<256xf32>
    %v6843 = stablehlo.multiply %v6840, %v6842 : tensor<256xf32>
    %v6844 = stablehlo.add %v6841, %v6843 : tensor<256xf32>
    %v6845 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6846 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6847 = stablehlo.multiply %v6845, %s3b0bt2m : tensor<256xf32>
    %v6848 = stablehlo.multiply %v6846, %armeans3b0bt2 : tensor<256xf32>
    %v6849 = stablehlo.add %v6847, %v6848 : tensor<256xf32>
    %v6850 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6851 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6852 = stablehlo.multiply %v6850, %s3b0bt2v : tensor<256xf32>
    %v6853 = stablehlo.multiply %armeans3b0bt2, %armeans3b0bt2 : tensor<256xf32>
    %v6854 = stablehlo.multiply %v6851, %v6853 : tensor<256xf32>
    %v6855 = stablehlo.add %v6852, %v6854 : tensor<256xf32>
    %v6856 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6857 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6858 = stablehlo.divide %v6849, %v6856 : tensor<256xf32>
    %v6859 = stablehlo.divide %v6855, %v6857 : tensor<256xf32>
    %v6860 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6861 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6862 = stablehlo.sqrt %v6859 : tensor<256xf32>
    %v6863 = stablehlo.add %v6862, %v6861 : tensor<256xf32>
    %v6864 = stablehlo.divide %v6858, %v6863 : tensor<256xf32>
    %v6865 = stablehlo.multiply %v6860, %v6864 : tensor<256xf32>
    %v6866 = stablehlo.subtract %s3b0bt2, %v6865 : tensor<256xf32>
    %v6867 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6868 = stablehlo.multiply %v6867, %v6860 : tensor<256xf32>
    %v6869 = stablehlo.multiply %v6868, %s3b0bt2 : tensor<256xf32>
    %v6870 = stablehlo.subtract %v6866, %v6869 : tensor<256xf32>
    %arsums3b1W1 = "stablehlo.all_reduce"(%v2747) ({
    ^bb0(%aras3b1W1: tensor<f32>, %arbs3b1W1: tensor<f32>):
      %aradds3b1W1 = stablehlo.add %aras3b1W1, %arbs3b1W1 : tensor<f32>
      stablehlo.return %aradds3b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b1W1 = stablehlo.divide %arsums3b1W1, %arns3b1W1 : tensor<256x256x3x3xf32>
    %v6871 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6872 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6873 = stablehlo.multiply %v6871, %s3b1W1m : tensor<256x256x3x3xf32>
    %v6874 = stablehlo.multiply %v6872, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6875 = stablehlo.add %v6873, %v6874 : tensor<256x256x3x3xf32>
    %v6876 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6877 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6878 = stablehlo.multiply %v6876, %s3b1W1v : tensor<256x256x3x3xf32>
    %v6879 = stablehlo.multiply %armeans3b1W1, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6880 = stablehlo.multiply %v6877, %v6879 : tensor<256x256x3x3xf32>
    %v6881 = stablehlo.add %v6878, %v6880 : tensor<256x256x3x3xf32>
    %v6882 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6883 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6884 = stablehlo.multiply %v6882, %s3b1W1m : tensor<256x256x3x3xf32>
    %v6885 = stablehlo.multiply %v6883, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6886 = stablehlo.add %v6884, %v6885 : tensor<256x256x3x3xf32>
    %v6887 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6888 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6889 = stablehlo.multiply %v6887, %s3b1W1v : tensor<256x256x3x3xf32>
    %v6890 = stablehlo.multiply %armeans3b1W1, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v6891 = stablehlo.multiply %v6888, %v6890 : tensor<256x256x3x3xf32>
    %v6892 = stablehlo.add %v6889, %v6891 : tensor<256x256x3x3xf32>
    %v6893 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6894 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6895 = stablehlo.divide %v6886, %v6893 : tensor<256x256x3x3xf32>
    %v6896 = stablehlo.divide %v6892, %v6894 : tensor<256x256x3x3xf32>
    %v6897 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6898 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6899 = stablehlo.sqrt %v6896 : tensor<256x256x3x3xf32>
    %v6900 = stablehlo.add %v6899, %v6898 : tensor<256x256x3x3xf32>
    %v6901 = stablehlo.divide %v6895, %v6900 : tensor<256x256x3x3xf32>
    %v6902 = stablehlo.multiply %v6897, %v6901 : tensor<256x256x3x3xf32>
    %v6903 = stablehlo.subtract %s3b1W1, %v6902 : tensor<256x256x3x3xf32>
    %v6904 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6905 = stablehlo.multiply %v6904, %v6897 : tensor<256x256x3x3xf32>
    %v6906 = stablehlo.multiply %v6905, %s3b1W1 : tensor<256x256x3x3xf32>
    %v6907 = stablehlo.subtract %v6903, %v6906 : tensor<256x256x3x3xf32>
    %arsums3b1g1 = "stablehlo.all_reduce"(%v2761) ({
    ^bb0(%aras3b1g1: tensor<f32>, %arbs3b1g1: tensor<f32>):
      %aradds3b1g1 = stablehlo.add %aras3b1g1, %arbs3b1g1 : tensor<f32>
      stablehlo.return %aradds3b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g1 = stablehlo.divide %arsums3b1g1, %arns3b1g1 : tensor<256xf32>
    %v6908 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6909 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6910 = stablehlo.multiply %v6908, %s3b1g1m : tensor<256xf32>
    %v6911 = stablehlo.multiply %v6909, %armeans3b1g1 : tensor<256xf32>
    %v6912 = stablehlo.add %v6910, %v6911 : tensor<256xf32>
    %v6913 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6914 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6915 = stablehlo.multiply %v6913, %s3b1g1v : tensor<256xf32>
    %v6916 = stablehlo.multiply %armeans3b1g1, %armeans3b1g1 : tensor<256xf32>
    %v6917 = stablehlo.multiply %v6914, %v6916 : tensor<256xf32>
    %v6918 = stablehlo.add %v6915, %v6917 : tensor<256xf32>
    %v6919 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6920 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6921 = stablehlo.multiply %v6919, %s3b1g1m : tensor<256xf32>
    %v6922 = stablehlo.multiply %v6920, %armeans3b1g1 : tensor<256xf32>
    %v6923 = stablehlo.add %v6921, %v6922 : tensor<256xf32>
    %v6924 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6925 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6926 = stablehlo.multiply %v6924, %s3b1g1v : tensor<256xf32>
    %v6927 = stablehlo.multiply %armeans3b1g1, %armeans3b1g1 : tensor<256xf32>
    %v6928 = stablehlo.multiply %v6925, %v6927 : tensor<256xf32>
    %v6929 = stablehlo.add %v6926, %v6928 : tensor<256xf32>
    %v6930 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6931 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6932 = stablehlo.divide %v6923, %v6930 : tensor<256xf32>
    %v6933 = stablehlo.divide %v6929, %v6931 : tensor<256xf32>
    %v6934 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6935 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6936 = stablehlo.sqrt %v6933 : tensor<256xf32>
    %v6937 = stablehlo.add %v6936, %v6935 : tensor<256xf32>
    %v6938 = stablehlo.divide %v6932, %v6937 : tensor<256xf32>
    %v6939 = stablehlo.multiply %v6934, %v6938 : tensor<256xf32>
    %v6940 = stablehlo.subtract %s3b1g1, %v6939 : tensor<256xf32>
    %v6941 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6942 = stablehlo.multiply %v6941, %v6934 : tensor<256xf32>
    %v6943 = stablehlo.multiply %v6942, %s3b1g1 : tensor<256xf32>
    %v6944 = stablehlo.subtract %v6940, %v6943 : tensor<256xf32>
    %arsums3b1bt1 = "stablehlo.all_reduce"(%v2764) ({
    ^bb0(%aras3b1bt1: tensor<f32>, %arbs3b1bt1: tensor<f32>):
      %aradds3b1bt1 = stablehlo.add %aras3b1bt1, %arbs3b1bt1 : tensor<f32>
      stablehlo.return %aradds3b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1bt1 = stablehlo.divide %arsums3b1bt1, %arns3b1bt1 : tensor<256xf32>
    %v6945 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6946 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6947 = stablehlo.multiply %v6945, %s3b1bt1m : tensor<256xf32>
    %v6948 = stablehlo.multiply %v6946, %armeans3b1bt1 : tensor<256xf32>
    %v6949 = stablehlo.add %v6947, %v6948 : tensor<256xf32>
    %v6950 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6951 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6952 = stablehlo.multiply %v6950, %s3b1bt1v : tensor<256xf32>
    %v6953 = stablehlo.multiply %armeans3b1bt1, %armeans3b1bt1 : tensor<256xf32>
    %v6954 = stablehlo.multiply %v6951, %v6953 : tensor<256xf32>
    %v6955 = stablehlo.add %v6952, %v6954 : tensor<256xf32>
    %v6956 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6957 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6958 = stablehlo.multiply %v6956, %s3b1bt1m : tensor<256xf32>
    %v6959 = stablehlo.multiply %v6957, %armeans3b1bt1 : tensor<256xf32>
    %v6960 = stablehlo.add %v6958, %v6959 : tensor<256xf32>
    %v6961 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6962 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6963 = stablehlo.multiply %v6961, %s3b1bt1v : tensor<256xf32>
    %v6964 = stablehlo.multiply %armeans3b1bt1, %armeans3b1bt1 : tensor<256xf32>
    %v6965 = stablehlo.multiply %v6962, %v6964 : tensor<256xf32>
    %v6966 = stablehlo.add %v6963, %v6965 : tensor<256xf32>
    %v6967 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6968 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6969 = stablehlo.divide %v6960, %v6967 : tensor<256xf32>
    %v6970 = stablehlo.divide %v6966, %v6968 : tensor<256xf32>
    %v6971 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6972 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6973 = stablehlo.sqrt %v6970 : tensor<256xf32>
    %v6974 = stablehlo.add %v6973, %v6972 : tensor<256xf32>
    %v6975 = stablehlo.divide %v6969, %v6974 : tensor<256xf32>
    %v6976 = stablehlo.multiply %v6971, %v6975 : tensor<256xf32>
    %v6977 = stablehlo.subtract %s3b1bt1, %v6976 : tensor<256xf32>
    %v6978 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6979 = stablehlo.multiply %v6978, %v6971 : tensor<256xf32>
    %v6980 = stablehlo.multiply %v6979, %s3b1bt1 : tensor<256xf32>
    %v6981 = stablehlo.subtract %v6977, %v6980 : tensor<256xf32>
    %arsums3b1W2 = "stablehlo.all_reduce"(%v2770) ({
    ^bb0(%aras3b1W2: tensor<f32>, %arbs3b1W2: tensor<f32>):
      %aradds3b1W2 = stablehlo.add %aras3b1W2, %arbs3b1W2 : tensor<f32>
      stablehlo.return %aradds3b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b1W2 = stablehlo.divide %arsums3b1W2, %arns3b1W2 : tensor<256x256x3x3xf32>
    %v6982 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6983 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6984 = stablehlo.multiply %v6982, %s3b1W2m : tensor<256x256x3x3xf32>
    %v6985 = stablehlo.multiply %v6983, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v6986 = stablehlo.add %v6984, %v6985 : tensor<256x256x3x3xf32>
    %v6987 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6988 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6989 = stablehlo.multiply %v6987, %s3b1W2v : tensor<256x256x3x3xf32>
    %v6990 = stablehlo.multiply %armeans3b1W2, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v6991 = stablehlo.multiply %v6988, %v6990 : tensor<256x256x3x3xf32>
    %v6992 = stablehlo.add %v6989, %v6991 : tensor<256x256x3x3xf32>
    %v6993 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6994 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6995 = stablehlo.multiply %v6993, %s3b1W2m : tensor<256x256x3x3xf32>
    %v6996 = stablehlo.multiply %v6994, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v6997 = stablehlo.add %v6995, %v6996 : tensor<256x256x3x3xf32>
    %v6998 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6999 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7000 = stablehlo.multiply %v6998, %s3b1W2v : tensor<256x256x3x3xf32>
    %v7001 = stablehlo.multiply %armeans3b1W2, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v7002 = stablehlo.multiply %v6999, %v7001 : tensor<256x256x3x3xf32>
    %v7003 = stablehlo.add %v7000, %v7002 : tensor<256x256x3x3xf32>
    %v7004 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7005 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7006 = stablehlo.divide %v6997, %v7004 : tensor<256x256x3x3xf32>
    %v7007 = stablehlo.divide %v7003, %v7005 : tensor<256x256x3x3xf32>
    %v7008 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7009 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7010 = stablehlo.sqrt %v7007 : tensor<256x256x3x3xf32>
    %v7011 = stablehlo.add %v7010, %v7009 : tensor<256x256x3x3xf32>
    %v7012 = stablehlo.divide %v7006, %v7011 : tensor<256x256x3x3xf32>
    %v7013 = stablehlo.multiply %v7008, %v7012 : tensor<256x256x3x3xf32>
    %v7014 = stablehlo.subtract %s3b1W2, %v7013 : tensor<256x256x3x3xf32>
    %v7015 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7016 = stablehlo.multiply %v7015, %v7008 : tensor<256x256x3x3xf32>
    %v7017 = stablehlo.multiply %v7016, %s3b1W2 : tensor<256x256x3x3xf32>
    %v7018 = stablehlo.subtract %v7014, %v7017 : tensor<256x256x3x3xf32>
    %arsums3b1g2 = "stablehlo.all_reduce"(%v2784) ({
    ^bb0(%aras3b1g2: tensor<f32>, %arbs3b1g2: tensor<f32>):
      %aradds3b1g2 = stablehlo.add %aras3b1g2, %arbs3b1g2 : tensor<f32>
      stablehlo.return %aradds3b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g2 = stablehlo.divide %arsums3b1g2, %arns3b1g2 : tensor<256xf32>
    %v7019 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7020 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7021 = stablehlo.multiply %v7019, %s3b1g2m : tensor<256xf32>
    %v7022 = stablehlo.multiply %v7020, %armeans3b1g2 : tensor<256xf32>
    %v7023 = stablehlo.add %v7021, %v7022 : tensor<256xf32>
    %v7024 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7025 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7026 = stablehlo.multiply %v7024, %s3b1g2v : tensor<256xf32>
    %v7027 = stablehlo.multiply %armeans3b1g2, %armeans3b1g2 : tensor<256xf32>
    %v7028 = stablehlo.multiply %v7025, %v7027 : tensor<256xf32>
    %v7029 = stablehlo.add %v7026, %v7028 : tensor<256xf32>
    %v7030 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7031 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7032 = stablehlo.multiply %v7030, %s3b1g2m : tensor<256xf32>
    %v7033 = stablehlo.multiply %v7031, %armeans3b1g2 : tensor<256xf32>
    %v7034 = stablehlo.add %v7032, %v7033 : tensor<256xf32>
    %v7035 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7036 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7037 = stablehlo.multiply %v7035, %s3b1g2v : tensor<256xf32>
    %v7038 = stablehlo.multiply %armeans3b1g2, %armeans3b1g2 : tensor<256xf32>
    %v7039 = stablehlo.multiply %v7036, %v7038 : tensor<256xf32>
    %v7040 = stablehlo.add %v7037, %v7039 : tensor<256xf32>
    %v7041 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7042 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7043 = stablehlo.divide %v7034, %v7041 : tensor<256xf32>
    %v7044 = stablehlo.divide %v7040, %v7042 : tensor<256xf32>
    %v7045 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7046 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7047 = stablehlo.sqrt %v7044 : tensor<256xf32>
    %v7048 = stablehlo.add %v7047, %v7046 : tensor<256xf32>
    %v7049 = stablehlo.divide %v7043, %v7048 : tensor<256xf32>
    %v7050 = stablehlo.multiply %v7045, %v7049 : tensor<256xf32>
    %v7051 = stablehlo.subtract %s3b1g2, %v7050 : tensor<256xf32>
    %v7052 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7053 = stablehlo.multiply %v7052, %v7045 : tensor<256xf32>
    %v7054 = stablehlo.multiply %v7053, %s3b1g2 : tensor<256xf32>
    %v7055 = stablehlo.subtract %v7051, %v7054 : tensor<256xf32>
    %arsums3b1bt2 = "stablehlo.all_reduce"(%v2787) ({
    ^bb0(%aras3b1bt2: tensor<f32>, %arbs3b1bt2: tensor<f32>):
      %aradds3b1bt2 = stablehlo.add %aras3b1bt2, %arbs3b1bt2 : tensor<f32>
      stablehlo.return %aradds3b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1bt2 = stablehlo.divide %arsums3b1bt2, %arns3b1bt2 : tensor<256xf32>
    %v7056 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7057 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7058 = stablehlo.multiply %v7056, %s3b1bt2m : tensor<256xf32>
    %v7059 = stablehlo.multiply %v7057, %armeans3b1bt2 : tensor<256xf32>
    %v7060 = stablehlo.add %v7058, %v7059 : tensor<256xf32>
    %v7061 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7062 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7063 = stablehlo.multiply %v7061, %s3b1bt2v : tensor<256xf32>
    %v7064 = stablehlo.multiply %armeans3b1bt2, %armeans3b1bt2 : tensor<256xf32>
    %v7065 = stablehlo.multiply %v7062, %v7064 : tensor<256xf32>
    %v7066 = stablehlo.add %v7063, %v7065 : tensor<256xf32>
    %v7067 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7068 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7069 = stablehlo.multiply %v7067, %s3b1bt2m : tensor<256xf32>
    %v7070 = stablehlo.multiply %v7068, %armeans3b1bt2 : tensor<256xf32>
    %v7071 = stablehlo.add %v7069, %v7070 : tensor<256xf32>
    %v7072 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7073 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7074 = stablehlo.multiply %v7072, %s3b1bt2v : tensor<256xf32>
    %v7075 = stablehlo.multiply %armeans3b1bt2, %armeans3b1bt2 : tensor<256xf32>
    %v7076 = stablehlo.multiply %v7073, %v7075 : tensor<256xf32>
    %v7077 = stablehlo.add %v7074, %v7076 : tensor<256xf32>
    %v7078 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7079 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7080 = stablehlo.divide %v7071, %v7078 : tensor<256xf32>
    %v7081 = stablehlo.divide %v7077, %v7079 : tensor<256xf32>
    %v7082 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7083 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7084 = stablehlo.sqrt %v7081 : tensor<256xf32>
    %v7085 = stablehlo.add %v7084, %v7083 : tensor<256xf32>
    %v7086 = stablehlo.divide %v7080, %v7085 : tensor<256xf32>
    %v7087 = stablehlo.multiply %v7082, %v7086 : tensor<256xf32>
    %v7088 = stablehlo.subtract %s3b1bt2, %v7087 : tensor<256xf32>
    %v7089 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7090 = stablehlo.multiply %v7089, %v7082 : tensor<256xf32>
    %v7091 = stablehlo.multiply %v7090, %s3b1bt2 : tensor<256xf32>
    %v7092 = stablehlo.subtract %v7088, %v7091 : tensor<256xf32>
    %arsums3b2W1 = "stablehlo.all_reduce"(%v2587) ({
    ^bb0(%aras3b2W1: tensor<f32>, %arbs3b2W1: tensor<f32>):
      %aradds3b2W1 = stablehlo.add %aras3b2W1, %arbs3b2W1 : tensor<f32>
      stablehlo.return %aradds3b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b2W1 = stablehlo.divide %arsums3b2W1, %arns3b2W1 : tensor<256x256x3x3xf32>
    %v7093 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7094 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7095 = stablehlo.multiply %v7093, %s3b2W1m : tensor<256x256x3x3xf32>
    %v7096 = stablehlo.multiply %v7094, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v7097 = stablehlo.add %v7095, %v7096 : tensor<256x256x3x3xf32>
    %v7098 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7099 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7100 = stablehlo.multiply %v7098, %s3b2W1v : tensor<256x256x3x3xf32>
    %v7101 = stablehlo.multiply %armeans3b2W1, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v7102 = stablehlo.multiply %v7099, %v7101 : tensor<256x256x3x3xf32>
    %v7103 = stablehlo.add %v7100, %v7102 : tensor<256x256x3x3xf32>
    %v7104 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7105 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7106 = stablehlo.multiply %v7104, %s3b2W1m : tensor<256x256x3x3xf32>
    %v7107 = stablehlo.multiply %v7105, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v7108 = stablehlo.add %v7106, %v7107 : tensor<256x256x3x3xf32>
    %v7109 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7110 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7111 = stablehlo.multiply %v7109, %s3b2W1v : tensor<256x256x3x3xf32>
    %v7112 = stablehlo.multiply %armeans3b2W1, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v7113 = stablehlo.multiply %v7110, %v7112 : tensor<256x256x3x3xf32>
    %v7114 = stablehlo.add %v7111, %v7113 : tensor<256x256x3x3xf32>
    %v7115 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7116 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7117 = stablehlo.divide %v7108, %v7115 : tensor<256x256x3x3xf32>
    %v7118 = stablehlo.divide %v7114, %v7116 : tensor<256x256x3x3xf32>
    %v7119 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7120 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7121 = stablehlo.sqrt %v7118 : tensor<256x256x3x3xf32>
    %v7122 = stablehlo.add %v7121, %v7120 : tensor<256x256x3x3xf32>
    %v7123 = stablehlo.divide %v7117, %v7122 : tensor<256x256x3x3xf32>
    %v7124 = stablehlo.multiply %v7119, %v7123 : tensor<256x256x3x3xf32>
    %v7125 = stablehlo.subtract %s3b2W1, %v7124 : tensor<256x256x3x3xf32>
    %v7126 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7127 = stablehlo.multiply %v7126, %v7119 : tensor<256x256x3x3xf32>
    %v7128 = stablehlo.multiply %v7127, %s3b2W1 : tensor<256x256x3x3xf32>
    %v7129 = stablehlo.subtract %v7125, %v7128 : tensor<256x256x3x3xf32>
    %arsums3b2g1 = "stablehlo.all_reduce"(%v2601) ({
    ^bb0(%aras3b2g1: tensor<f32>, %arbs3b2g1: tensor<f32>):
      %aradds3b2g1 = stablehlo.add %aras3b2g1, %arbs3b2g1 : tensor<f32>
      stablehlo.return %aradds3b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g1 = stablehlo.divide %arsums3b2g1, %arns3b2g1 : tensor<256xf32>
    %v7130 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7131 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7132 = stablehlo.multiply %v7130, %s3b2g1m : tensor<256xf32>
    %v7133 = stablehlo.multiply %v7131, %armeans3b2g1 : tensor<256xf32>
    %v7134 = stablehlo.add %v7132, %v7133 : tensor<256xf32>
    %v7135 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7136 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7137 = stablehlo.multiply %v7135, %s3b2g1v : tensor<256xf32>
    %v7138 = stablehlo.multiply %armeans3b2g1, %armeans3b2g1 : tensor<256xf32>
    %v7139 = stablehlo.multiply %v7136, %v7138 : tensor<256xf32>
    %v7140 = stablehlo.add %v7137, %v7139 : tensor<256xf32>
    %v7141 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7142 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7143 = stablehlo.multiply %v7141, %s3b2g1m : tensor<256xf32>
    %v7144 = stablehlo.multiply %v7142, %armeans3b2g1 : tensor<256xf32>
    %v7145 = stablehlo.add %v7143, %v7144 : tensor<256xf32>
    %v7146 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7147 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7148 = stablehlo.multiply %v7146, %s3b2g1v : tensor<256xf32>
    %v7149 = stablehlo.multiply %armeans3b2g1, %armeans3b2g1 : tensor<256xf32>
    %v7150 = stablehlo.multiply %v7147, %v7149 : tensor<256xf32>
    %v7151 = stablehlo.add %v7148, %v7150 : tensor<256xf32>
    %v7152 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7153 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7154 = stablehlo.divide %v7145, %v7152 : tensor<256xf32>
    %v7155 = stablehlo.divide %v7151, %v7153 : tensor<256xf32>
    %v7156 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7157 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7158 = stablehlo.sqrt %v7155 : tensor<256xf32>
    %v7159 = stablehlo.add %v7158, %v7157 : tensor<256xf32>
    %v7160 = stablehlo.divide %v7154, %v7159 : tensor<256xf32>
    %v7161 = stablehlo.multiply %v7156, %v7160 : tensor<256xf32>
    %v7162 = stablehlo.subtract %s3b2g1, %v7161 : tensor<256xf32>
    %v7163 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7164 = stablehlo.multiply %v7163, %v7156 : tensor<256xf32>
    %v7165 = stablehlo.multiply %v7164, %s3b2g1 : tensor<256xf32>
    %v7166 = stablehlo.subtract %v7162, %v7165 : tensor<256xf32>
    %arsums3b2bt1 = "stablehlo.all_reduce"(%v2604) ({
    ^bb0(%aras3b2bt1: tensor<f32>, %arbs3b2bt1: tensor<f32>):
      %aradds3b2bt1 = stablehlo.add %aras3b2bt1, %arbs3b2bt1 : tensor<f32>
      stablehlo.return %aradds3b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2bt1 = stablehlo.divide %arsums3b2bt1, %arns3b2bt1 : tensor<256xf32>
    %v7167 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7168 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7169 = stablehlo.multiply %v7167, %s3b2bt1m : tensor<256xf32>
    %v7170 = stablehlo.multiply %v7168, %armeans3b2bt1 : tensor<256xf32>
    %v7171 = stablehlo.add %v7169, %v7170 : tensor<256xf32>
    %v7172 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7173 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7174 = stablehlo.multiply %v7172, %s3b2bt1v : tensor<256xf32>
    %v7175 = stablehlo.multiply %armeans3b2bt1, %armeans3b2bt1 : tensor<256xf32>
    %v7176 = stablehlo.multiply %v7173, %v7175 : tensor<256xf32>
    %v7177 = stablehlo.add %v7174, %v7176 : tensor<256xf32>
    %v7178 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7179 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7180 = stablehlo.multiply %v7178, %s3b2bt1m : tensor<256xf32>
    %v7181 = stablehlo.multiply %v7179, %armeans3b2bt1 : tensor<256xf32>
    %v7182 = stablehlo.add %v7180, %v7181 : tensor<256xf32>
    %v7183 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7184 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7185 = stablehlo.multiply %v7183, %s3b2bt1v : tensor<256xf32>
    %v7186 = stablehlo.multiply %armeans3b2bt1, %armeans3b2bt1 : tensor<256xf32>
    %v7187 = stablehlo.multiply %v7184, %v7186 : tensor<256xf32>
    %v7188 = stablehlo.add %v7185, %v7187 : tensor<256xf32>
    %v7189 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7190 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7191 = stablehlo.divide %v7182, %v7189 : tensor<256xf32>
    %v7192 = stablehlo.divide %v7188, %v7190 : tensor<256xf32>
    %v7193 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7194 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7195 = stablehlo.sqrt %v7192 : tensor<256xf32>
    %v7196 = stablehlo.add %v7195, %v7194 : tensor<256xf32>
    %v7197 = stablehlo.divide %v7191, %v7196 : tensor<256xf32>
    %v7198 = stablehlo.multiply %v7193, %v7197 : tensor<256xf32>
    %v7199 = stablehlo.subtract %s3b2bt1, %v7198 : tensor<256xf32>
    %v7200 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7201 = stablehlo.multiply %v7200, %v7193 : tensor<256xf32>
    %v7202 = stablehlo.multiply %v7201, %s3b2bt1 : tensor<256xf32>
    %v7203 = stablehlo.subtract %v7199, %v7202 : tensor<256xf32>
    %arsums3b2W2 = "stablehlo.all_reduce"(%v2610) ({
    ^bb0(%aras3b2W2: tensor<f32>, %arbs3b2W2: tensor<f32>):
      %aradds3b2W2 = stablehlo.add %aras3b2W2, %arbs3b2W2 : tensor<f32>
      stablehlo.return %aradds3b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b2W2 = stablehlo.divide %arsums3b2W2, %arns3b2W2 : tensor<256x256x3x3xf32>
    %v7204 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7205 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7206 = stablehlo.multiply %v7204, %s3b2W2m : tensor<256x256x3x3xf32>
    %v7207 = stablehlo.multiply %v7205, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v7208 = stablehlo.add %v7206, %v7207 : tensor<256x256x3x3xf32>
    %v7209 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7210 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7211 = stablehlo.multiply %v7209, %s3b2W2v : tensor<256x256x3x3xf32>
    %v7212 = stablehlo.multiply %armeans3b2W2, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v7213 = stablehlo.multiply %v7210, %v7212 : tensor<256x256x3x3xf32>
    %v7214 = stablehlo.add %v7211, %v7213 : tensor<256x256x3x3xf32>
    %v7215 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7216 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7217 = stablehlo.multiply %v7215, %s3b2W2m : tensor<256x256x3x3xf32>
    %v7218 = stablehlo.multiply %v7216, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v7219 = stablehlo.add %v7217, %v7218 : tensor<256x256x3x3xf32>
    %v7220 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7221 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7222 = stablehlo.multiply %v7220, %s3b2W2v : tensor<256x256x3x3xf32>
    %v7223 = stablehlo.multiply %armeans3b2W2, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v7224 = stablehlo.multiply %v7221, %v7223 : tensor<256x256x3x3xf32>
    %v7225 = stablehlo.add %v7222, %v7224 : tensor<256x256x3x3xf32>
    %v7226 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7227 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7228 = stablehlo.divide %v7219, %v7226 : tensor<256x256x3x3xf32>
    %v7229 = stablehlo.divide %v7225, %v7227 : tensor<256x256x3x3xf32>
    %v7230 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7231 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7232 = stablehlo.sqrt %v7229 : tensor<256x256x3x3xf32>
    %v7233 = stablehlo.add %v7232, %v7231 : tensor<256x256x3x3xf32>
    %v7234 = stablehlo.divide %v7228, %v7233 : tensor<256x256x3x3xf32>
    %v7235 = stablehlo.multiply %v7230, %v7234 : tensor<256x256x3x3xf32>
    %v7236 = stablehlo.subtract %s3b2W2, %v7235 : tensor<256x256x3x3xf32>
    %v7237 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7238 = stablehlo.multiply %v7237, %v7230 : tensor<256x256x3x3xf32>
    %v7239 = stablehlo.multiply %v7238, %s3b2W2 : tensor<256x256x3x3xf32>
    %v7240 = stablehlo.subtract %v7236, %v7239 : tensor<256x256x3x3xf32>
    %arsums3b2g2 = "stablehlo.all_reduce"(%v2624) ({
    ^bb0(%aras3b2g2: tensor<f32>, %arbs3b2g2: tensor<f32>):
      %aradds3b2g2 = stablehlo.add %aras3b2g2, %arbs3b2g2 : tensor<f32>
      stablehlo.return %aradds3b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g2 = stablehlo.divide %arsums3b2g2, %arns3b2g2 : tensor<256xf32>
    %v7241 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7242 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7243 = stablehlo.multiply %v7241, %s3b2g2m : tensor<256xf32>
    %v7244 = stablehlo.multiply %v7242, %armeans3b2g2 : tensor<256xf32>
    %v7245 = stablehlo.add %v7243, %v7244 : tensor<256xf32>
    %v7246 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7247 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7248 = stablehlo.multiply %v7246, %s3b2g2v : tensor<256xf32>
    %v7249 = stablehlo.multiply %armeans3b2g2, %armeans3b2g2 : tensor<256xf32>
    %v7250 = stablehlo.multiply %v7247, %v7249 : tensor<256xf32>
    %v7251 = stablehlo.add %v7248, %v7250 : tensor<256xf32>
    %v7252 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7253 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7254 = stablehlo.multiply %v7252, %s3b2g2m : tensor<256xf32>
    %v7255 = stablehlo.multiply %v7253, %armeans3b2g2 : tensor<256xf32>
    %v7256 = stablehlo.add %v7254, %v7255 : tensor<256xf32>
    %v7257 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7258 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7259 = stablehlo.multiply %v7257, %s3b2g2v : tensor<256xf32>
    %v7260 = stablehlo.multiply %armeans3b2g2, %armeans3b2g2 : tensor<256xf32>
    %v7261 = stablehlo.multiply %v7258, %v7260 : tensor<256xf32>
    %v7262 = stablehlo.add %v7259, %v7261 : tensor<256xf32>
    %v7263 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7264 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7265 = stablehlo.divide %v7256, %v7263 : tensor<256xf32>
    %v7266 = stablehlo.divide %v7262, %v7264 : tensor<256xf32>
    %v7267 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7268 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7269 = stablehlo.sqrt %v7266 : tensor<256xf32>
    %v7270 = stablehlo.add %v7269, %v7268 : tensor<256xf32>
    %v7271 = stablehlo.divide %v7265, %v7270 : tensor<256xf32>
    %v7272 = stablehlo.multiply %v7267, %v7271 : tensor<256xf32>
    %v7273 = stablehlo.subtract %s3b2g2, %v7272 : tensor<256xf32>
    %v7274 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7275 = stablehlo.multiply %v7274, %v7267 : tensor<256xf32>
    %v7276 = stablehlo.multiply %v7275, %s3b2g2 : tensor<256xf32>
    %v7277 = stablehlo.subtract %v7273, %v7276 : tensor<256xf32>
    %arsums3b2bt2 = "stablehlo.all_reduce"(%v2627) ({
    ^bb0(%aras3b2bt2: tensor<f32>, %arbs3b2bt2: tensor<f32>):
      %aradds3b2bt2 = stablehlo.add %aras3b2bt2, %arbs3b2bt2 : tensor<f32>
      stablehlo.return %aradds3b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2bt2 = stablehlo.divide %arsums3b2bt2, %arns3b2bt2 : tensor<256xf32>
    %v7278 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7279 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7280 = stablehlo.multiply %v7278, %s3b2bt2m : tensor<256xf32>
    %v7281 = stablehlo.multiply %v7279, %armeans3b2bt2 : tensor<256xf32>
    %v7282 = stablehlo.add %v7280, %v7281 : tensor<256xf32>
    %v7283 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7284 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7285 = stablehlo.multiply %v7283, %s3b2bt2v : tensor<256xf32>
    %v7286 = stablehlo.multiply %armeans3b2bt2, %armeans3b2bt2 : tensor<256xf32>
    %v7287 = stablehlo.multiply %v7284, %v7286 : tensor<256xf32>
    %v7288 = stablehlo.add %v7285, %v7287 : tensor<256xf32>
    %v7289 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7290 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7291 = stablehlo.multiply %v7289, %s3b2bt2m : tensor<256xf32>
    %v7292 = stablehlo.multiply %v7290, %armeans3b2bt2 : tensor<256xf32>
    %v7293 = stablehlo.add %v7291, %v7292 : tensor<256xf32>
    %v7294 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7295 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7296 = stablehlo.multiply %v7294, %s3b2bt2v : tensor<256xf32>
    %v7297 = stablehlo.multiply %armeans3b2bt2, %armeans3b2bt2 : tensor<256xf32>
    %v7298 = stablehlo.multiply %v7295, %v7297 : tensor<256xf32>
    %v7299 = stablehlo.add %v7296, %v7298 : tensor<256xf32>
    %v7300 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7301 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7302 = stablehlo.divide %v7293, %v7300 : tensor<256xf32>
    %v7303 = stablehlo.divide %v7299, %v7301 : tensor<256xf32>
    %v7304 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7305 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7306 = stablehlo.sqrt %v7303 : tensor<256xf32>
    %v7307 = stablehlo.add %v7306, %v7305 : tensor<256xf32>
    %v7308 = stablehlo.divide %v7302, %v7307 : tensor<256xf32>
    %v7309 = stablehlo.multiply %v7304, %v7308 : tensor<256xf32>
    %v7310 = stablehlo.subtract %s3b2bt2, %v7309 : tensor<256xf32>
    %v7311 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7312 = stablehlo.multiply %v7311, %v7304 : tensor<256xf32>
    %v7313 = stablehlo.multiply %v7312, %s3b2bt2 : tensor<256xf32>
    %v7314 = stablehlo.subtract %v7310, %v7313 : tensor<256xf32>
    %arsums3b3W1 = "stablehlo.all_reduce"(%v2427) ({
    ^bb0(%aras3b3W1: tensor<f32>, %arbs3b3W1: tensor<f32>):
      %aradds3b3W1 = stablehlo.add %aras3b3W1, %arbs3b3W1 : tensor<f32>
      stablehlo.return %aradds3b3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b3W1 = stablehlo.divide %arsums3b3W1, %arns3b3W1 : tensor<256x256x3x3xf32>
    %v7315 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7316 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7317 = stablehlo.multiply %v7315, %s3b3W1m : tensor<256x256x3x3xf32>
    %v7318 = stablehlo.multiply %v7316, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v7319 = stablehlo.add %v7317, %v7318 : tensor<256x256x3x3xf32>
    %v7320 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7321 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7322 = stablehlo.multiply %v7320, %s3b3W1v : tensor<256x256x3x3xf32>
    %v7323 = stablehlo.multiply %armeans3b3W1, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v7324 = stablehlo.multiply %v7321, %v7323 : tensor<256x256x3x3xf32>
    %v7325 = stablehlo.add %v7322, %v7324 : tensor<256x256x3x3xf32>
    %v7326 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7327 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7328 = stablehlo.multiply %v7326, %s3b3W1m : tensor<256x256x3x3xf32>
    %v7329 = stablehlo.multiply %v7327, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v7330 = stablehlo.add %v7328, %v7329 : tensor<256x256x3x3xf32>
    %v7331 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7332 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7333 = stablehlo.multiply %v7331, %s3b3W1v : tensor<256x256x3x3xf32>
    %v7334 = stablehlo.multiply %armeans3b3W1, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v7335 = stablehlo.multiply %v7332, %v7334 : tensor<256x256x3x3xf32>
    %v7336 = stablehlo.add %v7333, %v7335 : tensor<256x256x3x3xf32>
    %v7337 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7338 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7339 = stablehlo.divide %v7330, %v7337 : tensor<256x256x3x3xf32>
    %v7340 = stablehlo.divide %v7336, %v7338 : tensor<256x256x3x3xf32>
    %v7341 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7342 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7343 = stablehlo.sqrt %v7340 : tensor<256x256x3x3xf32>
    %v7344 = stablehlo.add %v7343, %v7342 : tensor<256x256x3x3xf32>
    %v7345 = stablehlo.divide %v7339, %v7344 : tensor<256x256x3x3xf32>
    %v7346 = stablehlo.multiply %v7341, %v7345 : tensor<256x256x3x3xf32>
    %v7347 = stablehlo.subtract %s3b3W1, %v7346 : tensor<256x256x3x3xf32>
    %v7348 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7349 = stablehlo.multiply %v7348, %v7341 : tensor<256x256x3x3xf32>
    %v7350 = stablehlo.multiply %v7349, %s3b3W1 : tensor<256x256x3x3xf32>
    %v7351 = stablehlo.subtract %v7347, %v7350 : tensor<256x256x3x3xf32>
    %arsums3b3g1 = "stablehlo.all_reduce"(%v2441) ({
    ^bb0(%aras3b3g1: tensor<f32>, %arbs3b3g1: tensor<f32>):
      %aradds3b3g1 = stablehlo.add %aras3b3g1, %arbs3b3g1 : tensor<f32>
      stablehlo.return %aradds3b3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g1 = stablehlo.divide %arsums3b3g1, %arns3b3g1 : tensor<256xf32>
    %v7352 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7353 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7354 = stablehlo.multiply %v7352, %s3b3g1m : tensor<256xf32>
    %v7355 = stablehlo.multiply %v7353, %armeans3b3g1 : tensor<256xf32>
    %v7356 = stablehlo.add %v7354, %v7355 : tensor<256xf32>
    %v7357 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7358 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7359 = stablehlo.multiply %v7357, %s3b3g1v : tensor<256xf32>
    %v7360 = stablehlo.multiply %armeans3b3g1, %armeans3b3g1 : tensor<256xf32>
    %v7361 = stablehlo.multiply %v7358, %v7360 : tensor<256xf32>
    %v7362 = stablehlo.add %v7359, %v7361 : tensor<256xf32>
    %v7363 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7364 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7365 = stablehlo.multiply %v7363, %s3b3g1m : tensor<256xf32>
    %v7366 = stablehlo.multiply %v7364, %armeans3b3g1 : tensor<256xf32>
    %v7367 = stablehlo.add %v7365, %v7366 : tensor<256xf32>
    %v7368 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7369 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7370 = stablehlo.multiply %v7368, %s3b3g1v : tensor<256xf32>
    %v7371 = stablehlo.multiply %armeans3b3g1, %armeans3b3g1 : tensor<256xf32>
    %v7372 = stablehlo.multiply %v7369, %v7371 : tensor<256xf32>
    %v7373 = stablehlo.add %v7370, %v7372 : tensor<256xf32>
    %v7374 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7375 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7376 = stablehlo.divide %v7367, %v7374 : tensor<256xf32>
    %v7377 = stablehlo.divide %v7373, %v7375 : tensor<256xf32>
    %v7378 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7379 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7380 = stablehlo.sqrt %v7377 : tensor<256xf32>
    %v7381 = stablehlo.add %v7380, %v7379 : tensor<256xf32>
    %v7382 = stablehlo.divide %v7376, %v7381 : tensor<256xf32>
    %v7383 = stablehlo.multiply %v7378, %v7382 : tensor<256xf32>
    %v7384 = stablehlo.subtract %s3b3g1, %v7383 : tensor<256xf32>
    %v7385 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7386 = stablehlo.multiply %v7385, %v7378 : tensor<256xf32>
    %v7387 = stablehlo.multiply %v7386, %s3b3g1 : tensor<256xf32>
    %v7388 = stablehlo.subtract %v7384, %v7387 : tensor<256xf32>
    %arsums3b3bt1 = "stablehlo.all_reduce"(%v2444) ({
    ^bb0(%aras3b3bt1: tensor<f32>, %arbs3b3bt1: tensor<f32>):
      %aradds3b3bt1 = stablehlo.add %aras3b3bt1, %arbs3b3bt1 : tensor<f32>
      stablehlo.return %aradds3b3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3bt1 = stablehlo.divide %arsums3b3bt1, %arns3b3bt1 : tensor<256xf32>
    %v7389 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7390 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7391 = stablehlo.multiply %v7389, %s3b3bt1m : tensor<256xf32>
    %v7392 = stablehlo.multiply %v7390, %armeans3b3bt1 : tensor<256xf32>
    %v7393 = stablehlo.add %v7391, %v7392 : tensor<256xf32>
    %v7394 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7395 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7396 = stablehlo.multiply %v7394, %s3b3bt1v : tensor<256xf32>
    %v7397 = stablehlo.multiply %armeans3b3bt1, %armeans3b3bt1 : tensor<256xf32>
    %v7398 = stablehlo.multiply %v7395, %v7397 : tensor<256xf32>
    %v7399 = stablehlo.add %v7396, %v7398 : tensor<256xf32>
    %v7400 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7401 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7402 = stablehlo.multiply %v7400, %s3b3bt1m : tensor<256xf32>
    %v7403 = stablehlo.multiply %v7401, %armeans3b3bt1 : tensor<256xf32>
    %v7404 = stablehlo.add %v7402, %v7403 : tensor<256xf32>
    %v7405 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7406 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7407 = stablehlo.multiply %v7405, %s3b3bt1v : tensor<256xf32>
    %v7408 = stablehlo.multiply %armeans3b3bt1, %armeans3b3bt1 : tensor<256xf32>
    %v7409 = stablehlo.multiply %v7406, %v7408 : tensor<256xf32>
    %v7410 = stablehlo.add %v7407, %v7409 : tensor<256xf32>
    %v7411 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7412 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7413 = stablehlo.divide %v7404, %v7411 : tensor<256xf32>
    %v7414 = stablehlo.divide %v7410, %v7412 : tensor<256xf32>
    %v7415 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7416 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7417 = stablehlo.sqrt %v7414 : tensor<256xf32>
    %v7418 = stablehlo.add %v7417, %v7416 : tensor<256xf32>
    %v7419 = stablehlo.divide %v7413, %v7418 : tensor<256xf32>
    %v7420 = stablehlo.multiply %v7415, %v7419 : tensor<256xf32>
    %v7421 = stablehlo.subtract %s3b3bt1, %v7420 : tensor<256xf32>
    %v7422 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7423 = stablehlo.multiply %v7422, %v7415 : tensor<256xf32>
    %v7424 = stablehlo.multiply %v7423, %s3b3bt1 : tensor<256xf32>
    %v7425 = stablehlo.subtract %v7421, %v7424 : tensor<256xf32>
    %arsums3b3W2 = "stablehlo.all_reduce"(%v2450) ({
    ^bb0(%aras3b3W2: tensor<f32>, %arbs3b3W2: tensor<f32>):
      %aradds3b3W2 = stablehlo.add %aras3b3W2, %arbs3b3W2 : tensor<f32>
      stablehlo.return %aradds3b3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b3W2 = stablehlo.divide %arsums3b3W2, %arns3b3W2 : tensor<256x256x3x3xf32>
    %v7426 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7427 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7428 = stablehlo.multiply %v7426, %s3b3W2m : tensor<256x256x3x3xf32>
    %v7429 = stablehlo.multiply %v7427, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v7430 = stablehlo.add %v7428, %v7429 : tensor<256x256x3x3xf32>
    %v7431 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7432 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7433 = stablehlo.multiply %v7431, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7434 = stablehlo.multiply %armeans3b3W2, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v7435 = stablehlo.multiply %v7432, %v7434 : tensor<256x256x3x3xf32>
    %v7436 = stablehlo.add %v7433, %v7435 : tensor<256x256x3x3xf32>
    %v7437 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7438 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7439 = stablehlo.multiply %v7437, %s3b3W2m : tensor<256x256x3x3xf32>
    %v7440 = stablehlo.multiply %v7438, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v7441 = stablehlo.add %v7439, %v7440 : tensor<256x256x3x3xf32>
    %v7442 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7443 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7444 = stablehlo.multiply %v7442, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7445 = stablehlo.multiply %armeans3b3W2, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v7446 = stablehlo.multiply %v7443, %v7445 : tensor<256x256x3x3xf32>
    %v7447 = stablehlo.add %v7444, %v7446 : tensor<256x256x3x3xf32>
    %v7448 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7449 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7450 = stablehlo.divide %v7441, %v7448 : tensor<256x256x3x3xf32>
    %v7451 = stablehlo.divide %v7447, %v7449 : tensor<256x256x3x3xf32>
    %v7452 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7453 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7454 = stablehlo.sqrt %v7451 : tensor<256x256x3x3xf32>
    %v7455 = stablehlo.add %v7454, %v7453 : tensor<256x256x3x3xf32>
    %v7456 = stablehlo.divide %v7450, %v7455 : tensor<256x256x3x3xf32>
    %v7457 = stablehlo.multiply %v7452, %v7456 : tensor<256x256x3x3xf32>
    %v7458 = stablehlo.subtract %s3b3W2, %v7457 : tensor<256x256x3x3xf32>
    %v7459 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7460 = stablehlo.multiply %v7459, %v7452 : tensor<256x256x3x3xf32>
    %v7461 = stablehlo.multiply %v7460, %s3b3W2 : tensor<256x256x3x3xf32>
    %v7462 = stablehlo.subtract %v7458, %v7461 : tensor<256x256x3x3xf32>
    %arsums3b3g2 = "stablehlo.all_reduce"(%v2464) ({
    ^bb0(%aras3b3g2: tensor<f32>, %arbs3b3g2: tensor<f32>):
      %aradds3b3g2 = stablehlo.add %aras3b3g2, %arbs3b3g2 : tensor<f32>
      stablehlo.return %aradds3b3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g2 = stablehlo.divide %arsums3b3g2, %arns3b3g2 : tensor<256xf32>
    %v7463 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7464 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7465 = stablehlo.multiply %v7463, %s3b3g2m : tensor<256xf32>
    %v7466 = stablehlo.multiply %v7464, %armeans3b3g2 : tensor<256xf32>
    %v7467 = stablehlo.add %v7465, %v7466 : tensor<256xf32>
    %v7468 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7469 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7470 = stablehlo.multiply %v7468, %s3b3g2v : tensor<256xf32>
    %v7471 = stablehlo.multiply %armeans3b3g2, %armeans3b3g2 : tensor<256xf32>
    %v7472 = stablehlo.multiply %v7469, %v7471 : tensor<256xf32>
    %v7473 = stablehlo.add %v7470, %v7472 : tensor<256xf32>
    %v7474 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7475 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7476 = stablehlo.multiply %v7474, %s3b3g2m : tensor<256xf32>
    %v7477 = stablehlo.multiply %v7475, %armeans3b3g2 : tensor<256xf32>
    %v7478 = stablehlo.add %v7476, %v7477 : tensor<256xf32>
    %v7479 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7480 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7481 = stablehlo.multiply %v7479, %s3b3g2v : tensor<256xf32>
    %v7482 = stablehlo.multiply %armeans3b3g2, %armeans3b3g2 : tensor<256xf32>
    %v7483 = stablehlo.multiply %v7480, %v7482 : tensor<256xf32>
    %v7484 = stablehlo.add %v7481, %v7483 : tensor<256xf32>
    %v7485 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7486 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7487 = stablehlo.divide %v7478, %v7485 : tensor<256xf32>
    %v7488 = stablehlo.divide %v7484, %v7486 : tensor<256xf32>
    %v7489 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7490 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7491 = stablehlo.sqrt %v7488 : tensor<256xf32>
    %v7492 = stablehlo.add %v7491, %v7490 : tensor<256xf32>
    %v7493 = stablehlo.divide %v7487, %v7492 : tensor<256xf32>
    %v7494 = stablehlo.multiply %v7489, %v7493 : tensor<256xf32>
    %v7495 = stablehlo.subtract %s3b3g2, %v7494 : tensor<256xf32>
    %v7496 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7497 = stablehlo.multiply %v7496, %v7489 : tensor<256xf32>
    %v7498 = stablehlo.multiply %v7497, %s3b3g2 : tensor<256xf32>
    %v7499 = stablehlo.subtract %v7495, %v7498 : tensor<256xf32>
    %arsums3b3bt2 = "stablehlo.all_reduce"(%v2467) ({
    ^bb0(%aras3b3bt2: tensor<f32>, %arbs3b3bt2: tensor<f32>):
      %aradds3b3bt2 = stablehlo.add %aras3b3bt2, %arbs3b3bt2 : tensor<f32>
      stablehlo.return %aradds3b3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3bt2 = stablehlo.divide %arsums3b3bt2, %arns3b3bt2 : tensor<256xf32>
    %v7500 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7501 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7502 = stablehlo.multiply %v7500, %s3b3bt2m : tensor<256xf32>
    %v7503 = stablehlo.multiply %v7501, %armeans3b3bt2 : tensor<256xf32>
    %v7504 = stablehlo.add %v7502, %v7503 : tensor<256xf32>
    %v7505 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7506 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7507 = stablehlo.multiply %v7505, %s3b3bt2v : tensor<256xf32>
    %v7508 = stablehlo.multiply %armeans3b3bt2, %armeans3b3bt2 : tensor<256xf32>
    %v7509 = stablehlo.multiply %v7506, %v7508 : tensor<256xf32>
    %v7510 = stablehlo.add %v7507, %v7509 : tensor<256xf32>
    %v7511 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7512 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7513 = stablehlo.multiply %v7511, %s3b3bt2m : tensor<256xf32>
    %v7514 = stablehlo.multiply %v7512, %armeans3b3bt2 : tensor<256xf32>
    %v7515 = stablehlo.add %v7513, %v7514 : tensor<256xf32>
    %v7516 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7517 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7518 = stablehlo.multiply %v7516, %s3b3bt2v : tensor<256xf32>
    %v7519 = stablehlo.multiply %armeans3b3bt2, %armeans3b3bt2 : tensor<256xf32>
    %v7520 = stablehlo.multiply %v7517, %v7519 : tensor<256xf32>
    %v7521 = stablehlo.add %v7518, %v7520 : tensor<256xf32>
    %v7522 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7523 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7524 = stablehlo.divide %v7515, %v7522 : tensor<256xf32>
    %v7525 = stablehlo.divide %v7521, %v7523 : tensor<256xf32>
    %v7526 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7527 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7528 = stablehlo.sqrt %v7525 : tensor<256xf32>
    %v7529 = stablehlo.add %v7528, %v7527 : tensor<256xf32>
    %v7530 = stablehlo.divide %v7524, %v7529 : tensor<256xf32>
    %v7531 = stablehlo.multiply %v7526, %v7530 : tensor<256xf32>
    %v7532 = stablehlo.subtract %s3b3bt2, %v7531 : tensor<256xf32>
    %v7533 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7534 = stablehlo.multiply %v7533, %v7526 : tensor<256xf32>
    %v7535 = stablehlo.multiply %v7534, %s3b3bt2 : tensor<256xf32>
    %v7536 = stablehlo.subtract %v7532, %v7535 : tensor<256xf32>
    %arsums3b4W1 = "stablehlo.all_reduce"(%v2267) ({
    ^bb0(%aras3b4W1: tensor<f32>, %arbs3b4W1: tensor<f32>):
      %aradds3b4W1 = stablehlo.add %aras3b4W1, %arbs3b4W1 : tensor<f32>
      stablehlo.return %aradds3b4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b4W1 = stablehlo.divide %arsums3b4W1, %arns3b4W1 : tensor<256x256x3x3xf32>
    %v7537 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7538 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7539 = stablehlo.multiply %v7537, %s3b4W1m : tensor<256x256x3x3xf32>
    %v7540 = stablehlo.multiply %v7538, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v7541 = stablehlo.add %v7539, %v7540 : tensor<256x256x3x3xf32>
    %v7542 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7543 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7544 = stablehlo.multiply %v7542, %s3b4W1v : tensor<256x256x3x3xf32>
    %v7545 = stablehlo.multiply %armeans3b4W1, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v7546 = stablehlo.multiply %v7543, %v7545 : tensor<256x256x3x3xf32>
    %v7547 = stablehlo.add %v7544, %v7546 : tensor<256x256x3x3xf32>
    %v7548 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7549 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7550 = stablehlo.multiply %v7548, %s3b4W1m : tensor<256x256x3x3xf32>
    %v7551 = stablehlo.multiply %v7549, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v7552 = stablehlo.add %v7550, %v7551 : tensor<256x256x3x3xf32>
    %v7553 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7554 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7555 = stablehlo.multiply %v7553, %s3b4W1v : tensor<256x256x3x3xf32>
    %v7556 = stablehlo.multiply %armeans3b4W1, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v7557 = stablehlo.multiply %v7554, %v7556 : tensor<256x256x3x3xf32>
    %v7558 = stablehlo.add %v7555, %v7557 : tensor<256x256x3x3xf32>
    %v7559 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7560 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7561 = stablehlo.divide %v7552, %v7559 : tensor<256x256x3x3xf32>
    %v7562 = stablehlo.divide %v7558, %v7560 : tensor<256x256x3x3xf32>
    %v7563 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7564 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7565 = stablehlo.sqrt %v7562 : tensor<256x256x3x3xf32>
    %v7566 = stablehlo.add %v7565, %v7564 : tensor<256x256x3x3xf32>
    %v7567 = stablehlo.divide %v7561, %v7566 : tensor<256x256x3x3xf32>
    %v7568 = stablehlo.multiply %v7563, %v7567 : tensor<256x256x3x3xf32>
    %v7569 = stablehlo.subtract %s3b4W1, %v7568 : tensor<256x256x3x3xf32>
    %v7570 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7571 = stablehlo.multiply %v7570, %v7563 : tensor<256x256x3x3xf32>
    %v7572 = stablehlo.multiply %v7571, %s3b4W1 : tensor<256x256x3x3xf32>
    %v7573 = stablehlo.subtract %v7569, %v7572 : tensor<256x256x3x3xf32>
    %arsums3b4g1 = "stablehlo.all_reduce"(%v2281) ({
    ^bb0(%aras3b4g1: tensor<f32>, %arbs3b4g1: tensor<f32>):
      %aradds3b4g1 = stablehlo.add %aras3b4g1, %arbs3b4g1 : tensor<f32>
      stablehlo.return %aradds3b4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g1 = stablehlo.divide %arsums3b4g1, %arns3b4g1 : tensor<256xf32>
    %v7574 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7575 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7576 = stablehlo.multiply %v7574, %s3b4g1m : tensor<256xf32>
    %v7577 = stablehlo.multiply %v7575, %armeans3b4g1 : tensor<256xf32>
    %v7578 = stablehlo.add %v7576, %v7577 : tensor<256xf32>
    %v7579 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7580 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7581 = stablehlo.multiply %v7579, %s3b4g1v : tensor<256xf32>
    %v7582 = stablehlo.multiply %armeans3b4g1, %armeans3b4g1 : tensor<256xf32>
    %v7583 = stablehlo.multiply %v7580, %v7582 : tensor<256xf32>
    %v7584 = stablehlo.add %v7581, %v7583 : tensor<256xf32>
    %v7585 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7586 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7587 = stablehlo.multiply %v7585, %s3b4g1m : tensor<256xf32>
    %v7588 = stablehlo.multiply %v7586, %armeans3b4g1 : tensor<256xf32>
    %v7589 = stablehlo.add %v7587, %v7588 : tensor<256xf32>
    %v7590 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7591 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7592 = stablehlo.multiply %v7590, %s3b4g1v : tensor<256xf32>
    %v7593 = stablehlo.multiply %armeans3b4g1, %armeans3b4g1 : tensor<256xf32>
    %v7594 = stablehlo.multiply %v7591, %v7593 : tensor<256xf32>
    %v7595 = stablehlo.add %v7592, %v7594 : tensor<256xf32>
    %v7596 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7597 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7598 = stablehlo.divide %v7589, %v7596 : tensor<256xf32>
    %v7599 = stablehlo.divide %v7595, %v7597 : tensor<256xf32>
    %v7600 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7601 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7602 = stablehlo.sqrt %v7599 : tensor<256xf32>
    %v7603 = stablehlo.add %v7602, %v7601 : tensor<256xf32>
    %v7604 = stablehlo.divide %v7598, %v7603 : tensor<256xf32>
    %v7605 = stablehlo.multiply %v7600, %v7604 : tensor<256xf32>
    %v7606 = stablehlo.subtract %s3b4g1, %v7605 : tensor<256xf32>
    %v7607 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7608 = stablehlo.multiply %v7607, %v7600 : tensor<256xf32>
    %v7609 = stablehlo.multiply %v7608, %s3b4g1 : tensor<256xf32>
    %v7610 = stablehlo.subtract %v7606, %v7609 : tensor<256xf32>
    %arsums3b4bt1 = "stablehlo.all_reduce"(%v2284) ({
    ^bb0(%aras3b4bt1: tensor<f32>, %arbs3b4bt1: tensor<f32>):
      %aradds3b4bt1 = stablehlo.add %aras3b4bt1, %arbs3b4bt1 : tensor<f32>
      stablehlo.return %aradds3b4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4bt1 = stablehlo.divide %arsums3b4bt1, %arns3b4bt1 : tensor<256xf32>
    %v7611 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7612 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7613 = stablehlo.multiply %v7611, %s3b4bt1m : tensor<256xf32>
    %v7614 = stablehlo.multiply %v7612, %armeans3b4bt1 : tensor<256xf32>
    %v7615 = stablehlo.add %v7613, %v7614 : tensor<256xf32>
    %v7616 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7617 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7618 = stablehlo.multiply %v7616, %s3b4bt1v : tensor<256xf32>
    %v7619 = stablehlo.multiply %armeans3b4bt1, %armeans3b4bt1 : tensor<256xf32>
    %v7620 = stablehlo.multiply %v7617, %v7619 : tensor<256xf32>
    %v7621 = stablehlo.add %v7618, %v7620 : tensor<256xf32>
    %v7622 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7623 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7624 = stablehlo.multiply %v7622, %s3b4bt1m : tensor<256xf32>
    %v7625 = stablehlo.multiply %v7623, %armeans3b4bt1 : tensor<256xf32>
    %v7626 = stablehlo.add %v7624, %v7625 : tensor<256xf32>
    %v7627 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7628 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7629 = stablehlo.multiply %v7627, %s3b4bt1v : tensor<256xf32>
    %v7630 = stablehlo.multiply %armeans3b4bt1, %armeans3b4bt1 : tensor<256xf32>
    %v7631 = stablehlo.multiply %v7628, %v7630 : tensor<256xf32>
    %v7632 = stablehlo.add %v7629, %v7631 : tensor<256xf32>
    %v7633 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7634 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7635 = stablehlo.divide %v7626, %v7633 : tensor<256xf32>
    %v7636 = stablehlo.divide %v7632, %v7634 : tensor<256xf32>
    %v7637 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7638 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7639 = stablehlo.sqrt %v7636 : tensor<256xf32>
    %v7640 = stablehlo.add %v7639, %v7638 : tensor<256xf32>
    %v7641 = stablehlo.divide %v7635, %v7640 : tensor<256xf32>
    %v7642 = stablehlo.multiply %v7637, %v7641 : tensor<256xf32>
    %v7643 = stablehlo.subtract %s3b4bt1, %v7642 : tensor<256xf32>
    %v7644 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7645 = stablehlo.multiply %v7644, %v7637 : tensor<256xf32>
    %v7646 = stablehlo.multiply %v7645, %s3b4bt1 : tensor<256xf32>
    %v7647 = stablehlo.subtract %v7643, %v7646 : tensor<256xf32>
    %arsums3b4W2 = "stablehlo.all_reduce"(%v2290) ({
    ^bb0(%aras3b4W2: tensor<f32>, %arbs3b4W2: tensor<f32>):
      %aradds3b4W2 = stablehlo.add %aras3b4W2, %arbs3b4W2 : tensor<f32>
      stablehlo.return %aradds3b4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b4W2 = stablehlo.divide %arsums3b4W2, %arns3b4W2 : tensor<256x256x3x3xf32>
    %v7648 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7649 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7650 = stablehlo.multiply %v7648, %s3b4W2m : tensor<256x256x3x3xf32>
    %v7651 = stablehlo.multiply %v7649, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7652 = stablehlo.add %v7650, %v7651 : tensor<256x256x3x3xf32>
    %v7653 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7654 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7655 = stablehlo.multiply %v7653, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7656 = stablehlo.multiply %armeans3b4W2, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7657 = stablehlo.multiply %v7654, %v7656 : tensor<256x256x3x3xf32>
    %v7658 = stablehlo.add %v7655, %v7657 : tensor<256x256x3x3xf32>
    %v7659 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7660 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7661 = stablehlo.multiply %v7659, %s3b4W2m : tensor<256x256x3x3xf32>
    %v7662 = stablehlo.multiply %v7660, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7663 = stablehlo.add %v7661, %v7662 : tensor<256x256x3x3xf32>
    %v7664 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7665 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7666 = stablehlo.multiply %v7664, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7667 = stablehlo.multiply %armeans3b4W2, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7668 = stablehlo.multiply %v7665, %v7667 : tensor<256x256x3x3xf32>
    %v7669 = stablehlo.add %v7666, %v7668 : tensor<256x256x3x3xf32>
    %v7670 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7671 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7672 = stablehlo.divide %v7663, %v7670 : tensor<256x256x3x3xf32>
    %v7673 = stablehlo.divide %v7669, %v7671 : tensor<256x256x3x3xf32>
    %v7674 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7675 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7676 = stablehlo.sqrt %v7673 : tensor<256x256x3x3xf32>
    %v7677 = stablehlo.add %v7676, %v7675 : tensor<256x256x3x3xf32>
    %v7678 = stablehlo.divide %v7672, %v7677 : tensor<256x256x3x3xf32>
    %v7679 = stablehlo.multiply %v7674, %v7678 : tensor<256x256x3x3xf32>
    %v7680 = stablehlo.subtract %s3b4W2, %v7679 : tensor<256x256x3x3xf32>
    %v7681 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7682 = stablehlo.multiply %v7681, %v7674 : tensor<256x256x3x3xf32>
    %v7683 = stablehlo.multiply %v7682, %s3b4W2 : tensor<256x256x3x3xf32>
    %v7684 = stablehlo.subtract %v7680, %v7683 : tensor<256x256x3x3xf32>
    %arsums3b4g2 = "stablehlo.all_reduce"(%v2304) ({
    ^bb0(%aras3b4g2: tensor<f32>, %arbs3b4g2: tensor<f32>):
      %aradds3b4g2 = stablehlo.add %aras3b4g2, %arbs3b4g2 : tensor<f32>
      stablehlo.return %aradds3b4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g2 = stablehlo.divide %arsums3b4g2, %arns3b4g2 : tensor<256xf32>
    %v7685 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7686 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7687 = stablehlo.multiply %v7685, %s3b4g2m : tensor<256xf32>
    %v7688 = stablehlo.multiply %v7686, %armeans3b4g2 : tensor<256xf32>
    %v7689 = stablehlo.add %v7687, %v7688 : tensor<256xf32>
    %v7690 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7691 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7692 = stablehlo.multiply %v7690, %s3b4g2v : tensor<256xf32>
    %v7693 = stablehlo.multiply %armeans3b4g2, %armeans3b4g2 : tensor<256xf32>
    %v7694 = stablehlo.multiply %v7691, %v7693 : tensor<256xf32>
    %v7695 = stablehlo.add %v7692, %v7694 : tensor<256xf32>
    %v7696 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7697 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7698 = stablehlo.multiply %v7696, %s3b4g2m : tensor<256xf32>
    %v7699 = stablehlo.multiply %v7697, %armeans3b4g2 : tensor<256xf32>
    %v7700 = stablehlo.add %v7698, %v7699 : tensor<256xf32>
    %v7701 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7702 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7703 = stablehlo.multiply %v7701, %s3b4g2v : tensor<256xf32>
    %v7704 = stablehlo.multiply %armeans3b4g2, %armeans3b4g2 : tensor<256xf32>
    %v7705 = stablehlo.multiply %v7702, %v7704 : tensor<256xf32>
    %v7706 = stablehlo.add %v7703, %v7705 : tensor<256xf32>
    %v7707 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7708 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7709 = stablehlo.divide %v7700, %v7707 : tensor<256xf32>
    %v7710 = stablehlo.divide %v7706, %v7708 : tensor<256xf32>
    %v7711 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7712 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7713 = stablehlo.sqrt %v7710 : tensor<256xf32>
    %v7714 = stablehlo.add %v7713, %v7712 : tensor<256xf32>
    %v7715 = stablehlo.divide %v7709, %v7714 : tensor<256xf32>
    %v7716 = stablehlo.multiply %v7711, %v7715 : tensor<256xf32>
    %v7717 = stablehlo.subtract %s3b4g2, %v7716 : tensor<256xf32>
    %v7718 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7719 = stablehlo.multiply %v7718, %v7711 : tensor<256xf32>
    %v7720 = stablehlo.multiply %v7719, %s3b4g2 : tensor<256xf32>
    %v7721 = stablehlo.subtract %v7717, %v7720 : tensor<256xf32>
    %arsums3b4bt2 = "stablehlo.all_reduce"(%v2307) ({
    ^bb0(%aras3b4bt2: tensor<f32>, %arbs3b4bt2: tensor<f32>):
      %aradds3b4bt2 = stablehlo.add %aras3b4bt2, %arbs3b4bt2 : tensor<f32>
      stablehlo.return %aradds3b4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4bt2 = stablehlo.divide %arsums3b4bt2, %arns3b4bt2 : tensor<256xf32>
    %v7722 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7723 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7724 = stablehlo.multiply %v7722, %s3b4bt2m : tensor<256xf32>
    %v7725 = stablehlo.multiply %v7723, %armeans3b4bt2 : tensor<256xf32>
    %v7726 = stablehlo.add %v7724, %v7725 : tensor<256xf32>
    %v7727 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7728 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7729 = stablehlo.multiply %v7727, %s3b4bt2v : tensor<256xf32>
    %v7730 = stablehlo.multiply %armeans3b4bt2, %armeans3b4bt2 : tensor<256xf32>
    %v7731 = stablehlo.multiply %v7728, %v7730 : tensor<256xf32>
    %v7732 = stablehlo.add %v7729, %v7731 : tensor<256xf32>
    %v7733 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7734 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7735 = stablehlo.multiply %v7733, %s3b4bt2m : tensor<256xf32>
    %v7736 = stablehlo.multiply %v7734, %armeans3b4bt2 : tensor<256xf32>
    %v7737 = stablehlo.add %v7735, %v7736 : tensor<256xf32>
    %v7738 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7739 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7740 = stablehlo.multiply %v7738, %s3b4bt2v : tensor<256xf32>
    %v7741 = stablehlo.multiply %armeans3b4bt2, %armeans3b4bt2 : tensor<256xf32>
    %v7742 = stablehlo.multiply %v7739, %v7741 : tensor<256xf32>
    %v7743 = stablehlo.add %v7740, %v7742 : tensor<256xf32>
    %v7744 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7745 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7746 = stablehlo.divide %v7737, %v7744 : tensor<256xf32>
    %v7747 = stablehlo.divide %v7743, %v7745 : tensor<256xf32>
    %v7748 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7749 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7750 = stablehlo.sqrt %v7747 : tensor<256xf32>
    %v7751 = stablehlo.add %v7750, %v7749 : tensor<256xf32>
    %v7752 = stablehlo.divide %v7746, %v7751 : tensor<256xf32>
    %v7753 = stablehlo.multiply %v7748, %v7752 : tensor<256xf32>
    %v7754 = stablehlo.subtract %s3b4bt2, %v7753 : tensor<256xf32>
    %v7755 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7756 = stablehlo.multiply %v7755, %v7748 : tensor<256xf32>
    %v7757 = stablehlo.multiply %v7756, %s3b4bt2 : tensor<256xf32>
    %v7758 = stablehlo.subtract %v7754, %v7757 : tensor<256xf32>
    %arsumd4W1 = "stablehlo.all_reduce"(%v2082) ({
    ^bb0(%arad4W1: tensor<f32>, %arbd4W1: tensor<f32>):
      %araddd4W1 = stablehlo.add %arad4W1, %arbd4W1 : tensor<f32>
      stablehlo.return %araddd4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf32>
    %arnd4W1 = stablehlo.constant dense<2.0> : tensor<512x256x3x3xf32>
    %armeand4W1 = stablehlo.divide %arsumd4W1, %arnd4W1 : tensor<512x256x3x3xf32>
    %v7759 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7760 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7761 = stablehlo.multiply %v7759, %d4W1m : tensor<512x256x3x3xf32>
    %v7762 = stablehlo.multiply %v7760, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7763 = stablehlo.add %v7761, %v7762 : tensor<512x256x3x3xf32>
    %v7764 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7765 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7766 = stablehlo.multiply %v7764, %d4W1v : tensor<512x256x3x3xf32>
    %v7767 = stablehlo.multiply %armeand4W1, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7768 = stablehlo.multiply %v7765, %v7767 : tensor<512x256x3x3xf32>
    %v7769 = stablehlo.add %v7766, %v7768 : tensor<512x256x3x3xf32>
    %v7770 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7771 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7772 = stablehlo.multiply %v7770, %d4W1m : tensor<512x256x3x3xf32>
    %v7773 = stablehlo.multiply %v7771, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7774 = stablehlo.add %v7772, %v7773 : tensor<512x256x3x3xf32>
    %v7775 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7776 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7777 = stablehlo.multiply %v7775, %d4W1v : tensor<512x256x3x3xf32>
    %v7778 = stablehlo.multiply %armeand4W1, %armeand4W1 : tensor<512x256x3x3xf32>
    %v7779 = stablehlo.multiply %v7776, %v7778 : tensor<512x256x3x3xf32>
    %v7780 = stablehlo.add %v7777, %v7779 : tensor<512x256x3x3xf32>
    %v7781 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7782 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7783 = stablehlo.divide %v7774, %v7781 : tensor<512x256x3x3xf32>
    %v7784 = stablehlo.divide %v7780, %v7782 : tensor<512x256x3x3xf32>
    %v7785 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7786 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7787 = stablehlo.sqrt %v7784 : tensor<512x256x3x3xf32>
    %v7788 = stablehlo.add %v7787, %v7786 : tensor<512x256x3x3xf32>
    %v7789 = stablehlo.divide %v7783, %v7788 : tensor<512x256x3x3xf32>
    %v7790 = stablehlo.multiply %v7785, %v7789 : tensor<512x256x3x3xf32>
    %v7791 = stablehlo.subtract %d4W1, %v7790 : tensor<512x256x3x3xf32>
    %v7792 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v7793 = stablehlo.multiply %v7792, %v7785 : tensor<512x256x3x3xf32>
    %v7794 = stablehlo.multiply %v7793, %d4W1 : tensor<512x256x3x3xf32>
    %v7795 = stablehlo.subtract %v7791, %v7794 : tensor<512x256x3x3xf32>
    %arsumd4g1 = "stablehlo.all_reduce"(%v2096) ({
    ^bb0(%arad4g1: tensor<f32>, %arbd4g1: tensor<f32>):
      %araddd4g1 = stablehlo.add %arad4g1, %arbd4g1 : tensor<f32>
      stablehlo.return %araddd4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g1 = stablehlo.divide %arsumd4g1, %arnd4g1 : tensor<512xf32>
    %v7796 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7797 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7798 = stablehlo.multiply %v7796, %d4g1m : tensor<512xf32>
    %v7799 = stablehlo.multiply %v7797, %armeand4g1 : tensor<512xf32>
    %v7800 = stablehlo.add %v7798, %v7799 : tensor<512xf32>
    %v7801 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7802 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7803 = stablehlo.multiply %v7801, %d4g1v : tensor<512xf32>
    %v7804 = stablehlo.multiply %armeand4g1, %armeand4g1 : tensor<512xf32>
    %v7805 = stablehlo.multiply %v7802, %v7804 : tensor<512xf32>
    %v7806 = stablehlo.add %v7803, %v7805 : tensor<512xf32>
    %v7807 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7808 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7809 = stablehlo.multiply %v7807, %d4g1m : tensor<512xf32>
    %v7810 = stablehlo.multiply %v7808, %armeand4g1 : tensor<512xf32>
    %v7811 = stablehlo.add %v7809, %v7810 : tensor<512xf32>
    %v7812 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7813 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7814 = stablehlo.multiply %v7812, %d4g1v : tensor<512xf32>
    %v7815 = stablehlo.multiply %armeand4g1, %armeand4g1 : tensor<512xf32>
    %v7816 = stablehlo.multiply %v7813, %v7815 : tensor<512xf32>
    %v7817 = stablehlo.add %v7814, %v7816 : tensor<512xf32>
    %v7818 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7819 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7820 = stablehlo.divide %v7811, %v7818 : tensor<512xf32>
    %v7821 = stablehlo.divide %v7817, %v7819 : tensor<512xf32>
    %v7822 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7823 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7824 = stablehlo.sqrt %v7821 : tensor<512xf32>
    %v7825 = stablehlo.add %v7824, %v7823 : tensor<512xf32>
    %v7826 = stablehlo.divide %v7820, %v7825 : tensor<512xf32>
    %v7827 = stablehlo.multiply %v7822, %v7826 : tensor<512xf32>
    %v7828 = stablehlo.subtract %d4g1, %v7827 : tensor<512xf32>
    %v7829 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7830 = stablehlo.multiply %v7829, %v7822 : tensor<512xf32>
    %v7831 = stablehlo.multiply %v7830, %d4g1 : tensor<512xf32>
    %v7832 = stablehlo.subtract %v7828, %v7831 : tensor<512xf32>
    %arsumd4bt1 = "stablehlo.all_reduce"(%v2099) ({
    ^bb0(%arad4bt1: tensor<f32>, %arbd4bt1: tensor<f32>):
      %araddd4bt1 = stablehlo.add %arad4bt1, %arbd4bt1 : tensor<f32>
      stablehlo.return %araddd4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4bt1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4bt1 = stablehlo.divide %arsumd4bt1, %arnd4bt1 : tensor<512xf32>
    %v7833 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7834 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7835 = stablehlo.multiply %v7833, %d4bt1m : tensor<512xf32>
    %v7836 = stablehlo.multiply %v7834, %armeand4bt1 : tensor<512xf32>
    %v7837 = stablehlo.add %v7835, %v7836 : tensor<512xf32>
    %v7838 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7839 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7840 = stablehlo.multiply %v7838, %d4bt1v : tensor<512xf32>
    %v7841 = stablehlo.multiply %armeand4bt1, %armeand4bt1 : tensor<512xf32>
    %v7842 = stablehlo.multiply %v7839, %v7841 : tensor<512xf32>
    %v7843 = stablehlo.add %v7840, %v7842 : tensor<512xf32>
    %v7844 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7845 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7846 = stablehlo.multiply %v7844, %d4bt1m : tensor<512xf32>
    %v7847 = stablehlo.multiply %v7845, %armeand4bt1 : tensor<512xf32>
    %v7848 = stablehlo.add %v7846, %v7847 : tensor<512xf32>
    %v7849 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7850 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7851 = stablehlo.multiply %v7849, %d4bt1v : tensor<512xf32>
    %v7852 = stablehlo.multiply %armeand4bt1, %armeand4bt1 : tensor<512xf32>
    %v7853 = stablehlo.multiply %v7850, %v7852 : tensor<512xf32>
    %v7854 = stablehlo.add %v7851, %v7853 : tensor<512xf32>
    %v7855 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7856 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7857 = stablehlo.divide %v7848, %v7855 : tensor<512xf32>
    %v7858 = stablehlo.divide %v7854, %v7856 : tensor<512xf32>
    %v7859 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7860 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7861 = stablehlo.sqrt %v7858 : tensor<512xf32>
    %v7862 = stablehlo.add %v7861, %v7860 : tensor<512xf32>
    %v7863 = stablehlo.divide %v7857, %v7862 : tensor<512xf32>
    %v7864 = stablehlo.multiply %v7859, %v7863 : tensor<512xf32>
    %v7865 = stablehlo.subtract %d4bt1, %v7864 : tensor<512xf32>
    %v7866 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7867 = stablehlo.multiply %v7866, %v7859 : tensor<512xf32>
    %v7868 = stablehlo.multiply %v7867, %d4bt1 : tensor<512xf32>
    %v7869 = stablehlo.subtract %v7865, %v7868 : tensor<512xf32>
    %arsumd4W2 = "stablehlo.all_reduce"(%v2105) ({
    ^bb0(%arad4W2: tensor<f32>, %arbd4W2: tensor<f32>):
      %araddd4W2 = stablehlo.add %arad4W2, %arbd4W2 : tensor<f32>
      stablehlo.return %araddd4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arnd4W2 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeand4W2 = stablehlo.divide %arsumd4W2, %arnd4W2 : tensor<512x512x3x3xf32>
    %v7870 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7871 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7872 = stablehlo.multiply %v7870, %d4W2m : tensor<512x512x3x3xf32>
    %v7873 = stablehlo.multiply %v7871, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7874 = stablehlo.add %v7872, %v7873 : tensor<512x512x3x3xf32>
    %v7875 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7876 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7877 = stablehlo.multiply %v7875, %d4W2v : tensor<512x512x3x3xf32>
    %v7878 = stablehlo.multiply %armeand4W2, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7879 = stablehlo.multiply %v7876, %v7878 : tensor<512x512x3x3xf32>
    %v7880 = stablehlo.add %v7877, %v7879 : tensor<512x512x3x3xf32>
    %v7881 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7882 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7883 = stablehlo.multiply %v7881, %d4W2m : tensor<512x512x3x3xf32>
    %v7884 = stablehlo.multiply %v7882, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7885 = stablehlo.add %v7883, %v7884 : tensor<512x512x3x3xf32>
    %v7886 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7887 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7888 = stablehlo.multiply %v7886, %d4W2v : tensor<512x512x3x3xf32>
    %v7889 = stablehlo.multiply %armeand4W2, %armeand4W2 : tensor<512x512x3x3xf32>
    %v7890 = stablehlo.multiply %v7887, %v7889 : tensor<512x512x3x3xf32>
    %v7891 = stablehlo.add %v7888, %v7890 : tensor<512x512x3x3xf32>
    %v7892 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7893 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7894 = stablehlo.divide %v7885, %v7892 : tensor<512x512x3x3xf32>
    %v7895 = stablehlo.divide %v7891, %v7893 : tensor<512x512x3x3xf32>
    %v7896 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7897 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7898 = stablehlo.sqrt %v7895 : tensor<512x512x3x3xf32>
    %v7899 = stablehlo.add %v7898, %v7897 : tensor<512x512x3x3xf32>
    %v7900 = stablehlo.divide %v7894, %v7899 : tensor<512x512x3x3xf32>
    %v7901 = stablehlo.multiply %v7896, %v7900 : tensor<512x512x3x3xf32>
    %v7902 = stablehlo.subtract %d4W2, %v7901 : tensor<512x512x3x3xf32>
    %v7903 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7904 = stablehlo.multiply %v7903, %v7896 : tensor<512x512x3x3xf32>
    %v7905 = stablehlo.multiply %v7904, %d4W2 : tensor<512x512x3x3xf32>
    %v7906 = stablehlo.subtract %v7902, %v7905 : tensor<512x512x3x3xf32>
    %arsumd4g2 = "stablehlo.all_reduce"(%v2119) ({
    ^bb0(%arad4g2: tensor<f32>, %arbd4g2: tensor<f32>):
      %araddd4g2 = stablehlo.add %arad4g2, %arbd4g2 : tensor<f32>
      stablehlo.return %araddd4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g2 = stablehlo.divide %arsumd4g2, %arnd4g2 : tensor<512xf32>
    %v7907 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7908 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7909 = stablehlo.multiply %v7907, %d4g2m : tensor<512xf32>
    %v7910 = stablehlo.multiply %v7908, %armeand4g2 : tensor<512xf32>
    %v7911 = stablehlo.add %v7909, %v7910 : tensor<512xf32>
    %v7912 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7913 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7914 = stablehlo.multiply %v7912, %d4g2v : tensor<512xf32>
    %v7915 = stablehlo.multiply %armeand4g2, %armeand4g2 : tensor<512xf32>
    %v7916 = stablehlo.multiply %v7913, %v7915 : tensor<512xf32>
    %v7917 = stablehlo.add %v7914, %v7916 : tensor<512xf32>
    %v7918 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7919 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7920 = stablehlo.multiply %v7918, %d4g2m : tensor<512xf32>
    %v7921 = stablehlo.multiply %v7919, %armeand4g2 : tensor<512xf32>
    %v7922 = stablehlo.add %v7920, %v7921 : tensor<512xf32>
    %v7923 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7924 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7925 = stablehlo.multiply %v7923, %d4g2v : tensor<512xf32>
    %v7926 = stablehlo.multiply %armeand4g2, %armeand4g2 : tensor<512xf32>
    %v7927 = stablehlo.multiply %v7924, %v7926 : tensor<512xf32>
    %v7928 = stablehlo.add %v7925, %v7927 : tensor<512xf32>
    %v7929 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7930 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7931 = stablehlo.divide %v7922, %v7929 : tensor<512xf32>
    %v7932 = stablehlo.divide %v7928, %v7930 : tensor<512xf32>
    %v7933 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7934 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7935 = stablehlo.sqrt %v7932 : tensor<512xf32>
    %v7936 = stablehlo.add %v7935, %v7934 : tensor<512xf32>
    %v7937 = stablehlo.divide %v7931, %v7936 : tensor<512xf32>
    %v7938 = stablehlo.multiply %v7933, %v7937 : tensor<512xf32>
    %v7939 = stablehlo.subtract %d4g2, %v7938 : tensor<512xf32>
    %v7940 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7941 = stablehlo.multiply %v7940, %v7933 : tensor<512xf32>
    %v7942 = stablehlo.multiply %v7941, %d4g2 : tensor<512xf32>
    %v7943 = stablehlo.subtract %v7939, %v7942 : tensor<512xf32>
    %arsumd4bt2 = "stablehlo.all_reduce"(%v2122) ({
    ^bb0(%arad4bt2: tensor<f32>, %arbd4bt2: tensor<f32>):
      %araddd4bt2 = stablehlo.add %arad4bt2, %arbd4bt2 : tensor<f32>
      stablehlo.return %araddd4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4bt2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4bt2 = stablehlo.divide %arsumd4bt2, %arnd4bt2 : tensor<512xf32>
    %v7944 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7945 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7946 = stablehlo.multiply %v7944, %d4bt2m : tensor<512xf32>
    %v7947 = stablehlo.multiply %v7945, %armeand4bt2 : tensor<512xf32>
    %v7948 = stablehlo.add %v7946, %v7947 : tensor<512xf32>
    %v7949 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7950 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7951 = stablehlo.multiply %v7949, %d4bt2v : tensor<512xf32>
    %v7952 = stablehlo.multiply %armeand4bt2, %armeand4bt2 : tensor<512xf32>
    %v7953 = stablehlo.multiply %v7950, %v7952 : tensor<512xf32>
    %v7954 = stablehlo.add %v7951, %v7953 : tensor<512xf32>
    %v7955 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7956 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7957 = stablehlo.multiply %v7955, %d4bt2m : tensor<512xf32>
    %v7958 = stablehlo.multiply %v7956, %armeand4bt2 : tensor<512xf32>
    %v7959 = stablehlo.add %v7957, %v7958 : tensor<512xf32>
    %v7960 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7961 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7962 = stablehlo.multiply %v7960, %d4bt2v : tensor<512xf32>
    %v7963 = stablehlo.multiply %armeand4bt2, %armeand4bt2 : tensor<512xf32>
    %v7964 = stablehlo.multiply %v7961, %v7963 : tensor<512xf32>
    %v7965 = stablehlo.add %v7962, %v7964 : tensor<512xf32>
    %v7966 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7967 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7968 = stablehlo.divide %v7959, %v7966 : tensor<512xf32>
    %v7969 = stablehlo.divide %v7965, %v7967 : tensor<512xf32>
    %v7970 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7971 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7972 = stablehlo.sqrt %v7969 : tensor<512xf32>
    %v7973 = stablehlo.add %v7972, %v7971 : tensor<512xf32>
    %v7974 = stablehlo.divide %v7968, %v7973 : tensor<512xf32>
    %v7975 = stablehlo.multiply %v7970, %v7974 : tensor<512xf32>
    %v7976 = stablehlo.subtract %d4bt2, %v7975 : tensor<512xf32>
    %v7977 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7978 = stablehlo.multiply %v7977, %v7970 : tensor<512xf32>
    %v7979 = stablehlo.multiply %v7978, %d4bt2 : tensor<512xf32>
    %v7980 = stablehlo.subtract %v7976, %v7979 : tensor<512xf32>
    %arsumd4Wp = "stablehlo.all_reduce"(%v2130) ({
    ^bb0(%arad4Wp: tensor<f32>, %arbd4Wp: tensor<f32>):
      %araddd4Wp = stablehlo.add %arad4Wp, %arbd4Wp : tensor<f32>
      stablehlo.return %araddd4Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf32>
    %arnd4Wp = stablehlo.constant dense<2.0> : tensor<512x256x1x1xf32>
    %armeand4Wp = stablehlo.divide %arsumd4Wp, %arnd4Wp : tensor<512x256x1x1xf32>
    %v7981 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7982 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7983 = stablehlo.multiply %v7981, %d4Wpm : tensor<512x256x1x1xf32>
    %v7984 = stablehlo.multiply %v7982, %armeand4Wp : tensor<512x256x1x1xf32>
    %v7985 = stablehlo.add %v7983, %v7984 : tensor<512x256x1x1xf32>
    %v7986 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7987 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7988 = stablehlo.multiply %v7986, %d4Wpv : tensor<512x256x1x1xf32>
    %v7989 = stablehlo.multiply %armeand4Wp, %armeand4Wp : tensor<512x256x1x1xf32>
    %v7990 = stablehlo.multiply %v7987, %v7989 : tensor<512x256x1x1xf32>
    %v7991 = stablehlo.add %v7988, %v7990 : tensor<512x256x1x1xf32>
    %v7992 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7993 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7994 = stablehlo.multiply %v7992, %d4Wpm : tensor<512x256x1x1xf32>
    %v7995 = stablehlo.multiply %v7993, %armeand4Wp : tensor<512x256x1x1xf32>
    %v7996 = stablehlo.add %v7994, %v7995 : tensor<512x256x1x1xf32>
    %v7997 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7998 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7999 = stablehlo.multiply %v7997, %d4Wpv : tensor<512x256x1x1xf32>
    %v8000 = stablehlo.multiply %armeand4Wp, %armeand4Wp : tensor<512x256x1x1xf32>
    %v8001 = stablehlo.multiply %v7998, %v8000 : tensor<512x256x1x1xf32>
    %v8002 = stablehlo.add %v7999, %v8001 : tensor<512x256x1x1xf32>
    %v8003 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8004 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8005 = stablehlo.divide %v7996, %v8003 : tensor<512x256x1x1xf32>
    %v8006 = stablehlo.divide %v8002, %v8004 : tensor<512x256x1x1xf32>
    %v8007 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8008 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8009 = stablehlo.sqrt %v8006 : tensor<512x256x1x1xf32>
    %v8010 = stablehlo.add %v8009, %v8008 : tensor<512x256x1x1xf32>
    %v8011 = stablehlo.divide %v8005, %v8010 : tensor<512x256x1x1xf32>
    %v8012 = stablehlo.multiply %v8007, %v8011 : tensor<512x256x1x1xf32>
    %v8013 = stablehlo.subtract %d4Wp, %v8012 : tensor<512x256x1x1xf32>
    %v8014 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v8015 = stablehlo.multiply %v8014, %v8007 : tensor<512x256x1x1xf32>
    %v8016 = stablehlo.multiply %v8015, %d4Wp : tensor<512x256x1x1xf32>
    %v8017 = stablehlo.subtract %v8013, %v8016 : tensor<512x256x1x1xf32>
    %arsumd4gp = "stablehlo.all_reduce"(%v2144) ({
    ^bb0(%arad4gp: tensor<f32>, %arbd4gp: tensor<f32>):
      %araddd4gp = stablehlo.add %arad4gp, %arbd4gp : tensor<f32>
      stablehlo.return %araddd4gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4gp = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4gp = stablehlo.divide %arsumd4gp, %arnd4gp : tensor<512xf32>
    %v8018 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8019 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8020 = stablehlo.multiply %v8018, %d4gpm : tensor<512xf32>
    %v8021 = stablehlo.multiply %v8019, %armeand4gp : tensor<512xf32>
    %v8022 = stablehlo.add %v8020, %v8021 : tensor<512xf32>
    %v8023 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8024 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8025 = stablehlo.multiply %v8023, %d4gpv : tensor<512xf32>
    %v8026 = stablehlo.multiply %armeand4gp, %armeand4gp : tensor<512xf32>
    %v8027 = stablehlo.multiply %v8024, %v8026 : tensor<512xf32>
    %v8028 = stablehlo.add %v8025, %v8027 : tensor<512xf32>
    %v8029 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8030 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8031 = stablehlo.multiply %v8029, %d4gpm : tensor<512xf32>
    %v8032 = stablehlo.multiply %v8030, %armeand4gp : tensor<512xf32>
    %v8033 = stablehlo.add %v8031, %v8032 : tensor<512xf32>
    %v8034 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8035 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8036 = stablehlo.multiply %v8034, %d4gpv : tensor<512xf32>
    %v8037 = stablehlo.multiply %armeand4gp, %armeand4gp : tensor<512xf32>
    %v8038 = stablehlo.multiply %v8035, %v8037 : tensor<512xf32>
    %v8039 = stablehlo.add %v8036, %v8038 : tensor<512xf32>
    %v8040 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8041 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8042 = stablehlo.divide %v8033, %v8040 : tensor<512xf32>
    %v8043 = stablehlo.divide %v8039, %v8041 : tensor<512xf32>
    %v8044 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8045 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8046 = stablehlo.sqrt %v8043 : tensor<512xf32>
    %v8047 = stablehlo.add %v8046, %v8045 : tensor<512xf32>
    %v8048 = stablehlo.divide %v8042, %v8047 : tensor<512xf32>
    %v8049 = stablehlo.multiply %v8044, %v8048 : tensor<512xf32>
    %v8050 = stablehlo.subtract %d4gp, %v8049 : tensor<512xf32>
    %v8051 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8052 = stablehlo.multiply %v8051, %v8044 : tensor<512xf32>
    %v8053 = stablehlo.multiply %v8052, %d4gp : tensor<512xf32>
    %v8054 = stablehlo.subtract %v8050, %v8053 : tensor<512xf32>
    %arsumd4btp = "stablehlo.all_reduce"(%v2147) ({
    ^bb0(%arad4btp: tensor<f32>, %arbd4btp: tensor<f32>):
      %araddd4btp = stablehlo.add %arad4btp, %arbd4btp : tensor<f32>
      stablehlo.return %araddd4btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4btp = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4btp = stablehlo.divide %arsumd4btp, %arnd4btp : tensor<512xf32>
    %v8055 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8056 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8057 = stablehlo.multiply %v8055, %d4btpm : tensor<512xf32>
    %v8058 = stablehlo.multiply %v8056, %armeand4btp : tensor<512xf32>
    %v8059 = stablehlo.add %v8057, %v8058 : tensor<512xf32>
    %v8060 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8061 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8062 = stablehlo.multiply %v8060, %d4btpv : tensor<512xf32>
    %v8063 = stablehlo.multiply %armeand4btp, %armeand4btp : tensor<512xf32>
    %v8064 = stablehlo.multiply %v8061, %v8063 : tensor<512xf32>
    %v8065 = stablehlo.add %v8062, %v8064 : tensor<512xf32>
    %v8066 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8067 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8068 = stablehlo.multiply %v8066, %d4btpm : tensor<512xf32>
    %v8069 = stablehlo.multiply %v8067, %armeand4btp : tensor<512xf32>
    %v8070 = stablehlo.add %v8068, %v8069 : tensor<512xf32>
    %v8071 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8072 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8073 = stablehlo.multiply %v8071, %d4btpv : tensor<512xf32>
    %v8074 = stablehlo.multiply %armeand4btp, %armeand4btp : tensor<512xf32>
    %v8075 = stablehlo.multiply %v8072, %v8074 : tensor<512xf32>
    %v8076 = stablehlo.add %v8073, %v8075 : tensor<512xf32>
    %v8077 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8078 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8079 = stablehlo.divide %v8070, %v8077 : tensor<512xf32>
    %v8080 = stablehlo.divide %v8076, %v8078 : tensor<512xf32>
    %v8081 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8082 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8083 = stablehlo.sqrt %v8080 : tensor<512xf32>
    %v8084 = stablehlo.add %v8083, %v8082 : tensor<512xf32>
    %v8085 = stablehlo.divide %v8079, %v8084 : tensor<512xf32>
    %v8086 = stablehlo.multiply %v8081, %v8085 : tensor<512xf32>
    %v8087 = stablehlo.subtract %d4btp, %v8086 : tensor<512xf32>
    %v8088 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8089 = stablehlo.multiply %v8088, %v8081 : tensor<512xf32>
    %v8090 = stablehlo.multiply %v8089, %d4btp : tensor<512xf32>
    %v8091 = stablehlo.subtract %v8087, %v8090 : tensor<512xf32>
    %arsums4b0W1 = "stablehlo.all_reduce"(%v1867) ({
    ^bb0(%aras4b0W1: tensor<f32>, %arbs4b0W1: tensor<f32>):
      %aradds4b0W1 = stablehlo.add %aras4b0W1, %arbs4b0W1 : tensor<f32>
      stablehlo.return %aradds4b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W1 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b0W1 = stablehlo.divide %arsums4b0W1, %arns4b0W1 : tensor<512x512x3x3xf32>
    %v8092 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8093 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8094 = stablehlo.multiply %v8092, %s4b0W1m : tensor<512x512x3x3xf32>
    %v8095 = stablehlo.multiply %v8093, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v8096 = stablehlo.add %v8094, %v8095 : tensor<512x512x3x3xf32>
    %v8097 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8098 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8099 = stablehlo.multiply %v8097, %s4b0W1v : tensor<512x512x3x3xf32>
    %v8100 = stablehlo.multiply %armeans4b0W1, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v8101 = stablehlo.multiply %v8098, %v8100 : tensor<512x512x3x3xf32>
    %v8102 = stablehlo.add %v8099, %v8101 : tensor<512x512x3x3xf32>
    %v8103 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8104 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8105 = stablehlo.multiply %v8103, %s4b0W1m : tensor<512x512x3x3xf32>
    %v8106 = stablehlo.multiply %v8104, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v8107 = stablehlo.add %v8105, %v8106 : tensor<512x512x3x3xf32>
    %v8108 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8109 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8110 = stablehlo.multiply %v8108, %s4b0W1v : tensor<512x512x3x3xf32>
    %v8111 = stablehlo.multiply %armeans4b0W1, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v8112 = stablehlo.multiply %v8109, %v8111 : tensor<512x512x3x3xf32>
    %v8113 = stablehlo.add %v8110, %v8112 : tensor<512x512x3x3xf32>
    %v8114 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8115 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8116 = stablehlo.divide %v8107, %v8114 : tensor<512x512x3x3xf32>
    %v8117 = stablehlo.divide %v8113, %v8115 : tensor<512x512x3x3xf32>
    %v8118 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8119 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8120 = stablehlo.sqrt %v8117 : tensor<512x512x3x3xf32>
    %v8121 = stablehlo.add %v8120, %v8119 : tensor<512x512x3x3xf32>
    %v8122 = stablehlo.divide %v8116, %v8121 : tensor<512x512x3x3xf32>
    %v8123 = stablehlo.multiply %v8118, %v8122 : tensor<512x512x3x3xf32>
    %v8124 = stablehlo.subtract %s4b0W1, %v8123 : tensor<512x512x3x3xf32>
    %v8125 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8126 = stablehlo.multiply %v8125, %v8118 : tensor<512x512x3x3xf32>
    %v8127 = stablehlo.multiply %v8126, %s4b0W1 : tensor<512x512x3x3xf32>
    %v8128 = stablehlo.subtract %v8124, %v8127 : tensor<512x512x3x3xf32>
    %arsums4b0g1 = "stablehlo.all_reduce"(%v1881) ({
    ^bb0(%aras4b0g1: tensor<f32>, %arbs4b0g1: tensor<f32>):
      %aradds4b0g1 = stablehlo.add %aras4b0g1, %arbs4b0g1 : tensor<f32>
      stablehlo.return %aradds4b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g1 = stablehlo.divide %arsums4b0g1, %arns4b0g1 : tensor<512xf32>
    %v8129 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8130 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8131 = stablehlo.multiply %v8129, %s4b0g1m : tensor<512xf32>
    %v8132 = stablehlo.multiply %v8130, %armeans4b0g1 : tensor<512xf32>
    %v8133 = stablehlo.add %v8131, %v8132 : tensor<512xf32>
    %v8134 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8135 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8136 = stablehlo.multiply %v8134, %s4b0g1v : tensor<512xf32>
    %v8137 = stablehlo.multiply %armeans4b0g1, %armeans4b0g1 : tensor<512xf32>
    %v8138 = stablehlo.multiply %v8135, %v8137 : tensor<512xf32>
    %v8139 = stablehlo.add %v8136, %v8138 : tensor<512xf32>
    %v8140 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8141 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8142 = stablehlo.multiply %v8140, %s4b0g1m : tensor<512xf32>
    %v8143 = stablehlo.multiply %v8141, %armeans4b0g1 : tensor<512xf32>
    %v8144 = stablehlo.add %v8142, %v8143 : tensor<512xf32>
    %v8145 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8146 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8147 = stablehlo.multiply %v8145, %s4b0g1v : tensor<512xf32>
    %v8148 = stablehlo.multiply %armeans4b0g1, %armeans4b0g1 : tensor<512xf32>
    %v8149 = stablehlo.multiply %v8146, %v8148 : tensor<512xf32>
    %v8150 = stablehlo.add %v8147, %v8149 : tensor<512xf32>
    %v8151 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8152 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8153 = stablehlo.divide %v8144, %v8151 : tensor<512xf32>
    %v8154 = stablehlo.divide %v8150, %v8152 : tensor<512xf32>
    %v8155 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8156 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8157 = stablehlo.sqrt %v8154 : tensor<512xf32>
    %v8158 = stablehlo.add %v8157, %v8156 : tensor<512xf32>
    %v8159 = stablehlo.divide %v8153, %v8158 : tensor<512xf32>
    %v8160 = stablehlo.multiply %v8155, %v8159 : tensor<512xf32>
    %v8161 = stablehlo.subtract %s4b0g1, %v8160 : tensor<512xf32>
    %v8162 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8163 = stablehlo.multiply %v8162, %v8155 : tensor<512xf32>
    %v8164 = stablehlo.multiply %v8163, %s4b0g1 : tensor<512xf32>
    %v8165 = stablehlo.subtract %v8161, %v8164 : tensor<512xf32>
    %arsums4b0bt1 = "stablehlo.all_reduce"(%v1884) ({
    ^bb0(%aras4b0bt1: tensor<f32>, %arbs4b0bt1: tensor<f32>):
      %aradds4b0bt1 = stablehlo.add %aras4b0bt1, %arbs4b0bt1 : tensor<f32>
      stablehlo.return %aradds4b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0bt1 = stablehlo.divide %arsums4b0bt1, %arns4b0bt1 : tensor<512xf32>
    %v8166 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8167 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8168 = stablehlo.multiply %v8166, %s4b0bt1m : tensor<512xf32>
    %v8169 = stablehlo.multiply %v8167, %armeans4b0bt1 : tensor<512xf32>
    %v8170 = stablehlo.add %v8168, %v8169 : tensor<512xf32>
    %v8171 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8172 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8173 = stablehlo.multiply %v8171, %s4b0bt1v : tensor<512xf32>
    %v8174 = stablehlo.multiply %armeans4b0bt1, %armeans4b0bt1 : tensor<512xf32>
    %v8175 = stablehlo.multiply %v8172, %v8174 : tensor<512xf32>
    %v8176 = stablehlo.add %v8173, %v8175 : tensor<512xf32>
    %v8177 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8178 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8179 = stablehlo.multiply %v8177, %s4b0bt1m : tensor<512xf32>
    %v8180 = stablehlo.multiply %v8178, %armeans4b0bt1 : tensor<512xf32>
    %v8181 = stablehlo.add %v8179, %v8180 : tensor<512xf32>
    %v8182 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8183 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8184 = stablehlo.multiply %v8182, %s4b0bt1v : tensor<512xf32>
    %v8185 = stablehlo.multiply %armeans4b0bt1, %armeans4b0bt1 : tensor<512xf32>
    %v8186 = stablehlo.multiply %v8183, %v8185 : tensor<512xf32>
    %v8187 = stablehlo.add %v8184, %v8186 : tensor<512xf32>
    %v8188 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8189 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8190 = stablehlo.divide %v8181, %v8188 : tensor<512xf32>
    %v8191 = stablehlo.divide %v8187, %v8189 : tensor<512xf32>
    %v8192 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8193 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8194 = stablehlo.sqrt %v8191 : tensor<512xf32>
    %v8195 = stablehlo.add %v8194, %v8193 : tensor<512xf32>
    %v8196 = stablehlo.divide %v8190, %v8195 : tensor<512xf32>
    %v8197 = stablehlo.multiply %v8192, %v8196 : tensor<512xf32>
    %v8198 = stablehlo.subtract %s4b0bt1, %v8197 : tensor<512xf32>
    %v8199 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8200 = stablehlo.multiply %v8199, %v8192 : tensor<512xf32>
    %v8201 = stablehlo.multiply %v8200, %s4b0bt1 : tensor<512xf32>
    %v8202 = stablehlo.subtract %v8198, %v8201 : tensor<512xf32>
    %arsums4b0W2 = "stablehlo.all_reduce"(%v1890) ({
    ^bb0(%aras4b0W2: tensor<f32>, %arbs4b0W2: tensor<f32>):
      %aradds4b0W2 = stablehlo.add %aras4b0W2, %arbs4b0W2 : tensor<f32>
      stablehlo.return %aradds4b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W2 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b0W2 = stablehlo.divide %arsums4b0W2, %arns4b0W2 : tensor<512x512x3x3xf32>
    %v8203 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8204 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8205 = stablehlo.multiply %v8203, %s4b0W2m : tensor<512x512x3x3xf32>
    %v8206 = stablehlo.multiply %v8204, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v8207 = stablehlo.add %v8205, %v8206 : tensor<512x512x3x3xf32>
    %v8208 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8209 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8210 = stablehlo.multiply %v8208, %s4b0W2v : tensor<512x512x3x3xf32>
    %v8211 = stablehlo.multiply %armeans4b0W2, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v8212 = stablehlo.multiply %v8209, %v8211 : tensor<512x512x3x3xf32>
    %v8213 = stablehlo.add %v8210, %v8212 : tensor<512x512x3x3xf32>
    %v8214 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8215 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8216 = stablehlo.multiply %v8214, %s4b0W2m : tensor<512x512x3x3xf32>
    %v8217 = stablehlo.multiply %v8215, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v8218 = stablehlo.add %v8216, %v8217 : tensor<512x512x3x3xf32>
    %v8219 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8220 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8221 = stablehlo.multiply %v8219, %s4b0W2v : tensor<512x512x3x3xf32>
    %v8222 = stablehlo.multiply %armeans4b0W2, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v8223 = stablehlo.multiply %v8220, %v8222 : tensor<512x512x3x3xf32>
    %v8224 = stablehlo.add %v8221, %v8223 : tensor<512x512x3x3xf32>
    %v8225 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8226 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8227 = stablehlo.divide %v8218, %v8225 : tensor<512x512x3x3xf32>
    %v8228 = stablehlo.divide %v8224, %v8226 : tensor<512x512x3x3xf32>
    %v8229 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8230 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8231 = stablehlo.sqrt %v8228 : tensor<512x512x3x3xf32>
    %v8232 = stablehlo.add %v8231, %v8230 : tensor<512x512x3x3xf32>
    %v8233 = stablehlo.divide %v8227, %v8232 : tensor<512x512x3x3xf32>
    %v8234 = stablehlo.multiply %v8229, %v8233 : tensor<512x512x3x3xf32>
    %v8235 = stablehlo.subtract %s4b0W2, %v8234 : tensor<512x512x3x3xf32>
    %v8236 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8237 = stablehlo.multiply %v8236, %v8229 : tensor<512x512x3x3xf32>
    %v8238 = stablehlo.multiply %v8237, %s4b0W2 : tensor<512x512x3x3xf32>
    %v8239 = stablehlo.subtract %v8235, %v8238 : tensor<512x512x3x3xf32>
    %arsums4b0g2 = "stablehlo.all_reduce"(%v1904) ({
    ^bb0(%aras4b0g2: tensor<f32>, %arbs4b0g2: tensor<f32>):
      %aradds4b0g2 = stablehlo.add %aras4b0g2, %arbs4b0g2 : tensor<f32>
      stablehlo.return %aradds4b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g2 = stablehlo.divide %arsums4b0g2, %arns4b0g2 : tensor<512xf32>
    %v8240 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8241 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8242 = stablehlo.multiply %v8240, %s4b0g2m : tensor<512xf32>
    %v8243 = stablehlo.multiply %v8241, %armeans4b0g2 : tensor<512xf32>
    %v8244 = stablehlo.add %v8242, %v8243 : tensor<512xf32>
    %v8245 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8246 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8247 = stablehlo.multiply %v8245, %s4b0g2v : tensor<512xf32>
    %v8248 = stablehlo.multiply %armeans4b0g2, %armeans4b0g2 : tensor<512xf32>
    %v8249 = stablehlo.multiply %v8246, %v8248 : tensor<512xf32>
    %v8250 = stablehlo.add %v8247, %v8249 : tensor<512xf32>
    %v8251 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8252 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8253 = stablehlo.multiply %v8251, %s4b0g2m : tensor<512xf32>
    %v8254 = stablehlo.multiply %v8252, %armeans4b0g2 : tensor<512xf32>
    %v8255 = stablehlo.add %v8253, %v8254 : tensor<512xf32>
    %v8256 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8257 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8258 = stablehlo.multiply %v8256, %s4b0g2v : tensor<512xf32>
    %v8259 = stablehlo.multiply %armeans4b0g2, %armeans4b0g2 : tensor<512xf32>
    %v8260 = stablehlo.multiply %v8257, %v8259 : tensor<512xf32>
    %v8261 = stablehlo.add %v8258, %v8260 : tensor<512xf32>
    %v8262 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8263 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8264 = stablehlo.divide %v8255, %v8262 : tensor<512xf32>
    %v8265 = stablehlo.divide %v8261, %v8263 : tensor<512xf32>
    %v8266 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8267 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8268 = stablehlo.sqrt %v8265 : tensor<512xf32>
    %v8269 = stablehlo.add %v8268, %v8267 : tensor<512xf32>
    %v8270 = stablehlo.divide %v8264, %v8269 : tensor<512xf32>
    %v8271 = stablehlo.multiply %v8266, %v8270 : tensor<512xf32>
    %v8272 = stablehlo.subtract %s4b0g2, %v8271 : tensor<512xf32>
    %v8273 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8274 = stablehlo.multiply %v8273, %v8266 : tensor<512xf32>
    %v8275 = stablehlo.multiply %v8274, %s4b0g2 : tensor<512xf32>
    %v8276 = stablehlo.subtract %v8272, %v8275 : tensor<512xf32>
    %arsums4b0bt2 = "stablehlo.all_reduce"(%v1907) ({
    ^bb0(%aras4b0bt2: tensor<f32>, %arbs4b0bt2: tensor<f32>):
      %aradds4b0bt2 = stablehlo.add %aras4b0bt2, %arbs4b0bt2 : tensor<f32>
      stablehlo.return %aradds4b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0bt2 = stablehlo.divide %arsums4b0bt2, %arns4b0bt2 : tensor<512xf32>
    %v8277 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8278 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8279 = stablehlo.multiply %v8277, %s4b0bt2m : tensor<512xf32>
    %v8280 = stablehlo.multiply %v8278, %armeans4b0bt2 : tensor<512xf32>
    %v8281 = stablehlo.add %v8279, %v8280 : tensor<512xf32>
    %v8282 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8283 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8284 = stablehlo.multiply %v8282, %s4b0bt2v : tensor<512xf32>
    %v8285 = stablehlo.multiply %armeans4b0bt2, %armeans4b0bt2 : tensor<512xf32>
    %v8286 = stablehlo.multiply %v8283, %v8285 : tensor<512xf32>
    %v8287 = stablehlo.add %v8284, %v8286 : tensor<512xf32>
    %v8288 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8289 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8290 = stablehlo.multiply %v8288, %s4b0bt2m : tensor<512xf32>
    %v8291 = stablehlo.multiply %v8289, %armeans4b0bt2 : tensor<512xf32>
    %v8292 = stablehlo.add %v8290, %v8291 : tensor<512xf32>
    %v8293 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8294 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8295 = stablehlo.multiply %v8293, %s4b0bt2v : tensor<512xf32>
    %v8296 = stablehlo.multiply %armeans4b0bt2, %armeans4b0bt2 : tensor<512xf32>
    %v8297 = stablehlo.multiply %v8294, %v8296 : tensor<512xf32>
    %v8298 = stablehlo.add %v8295, %v8297 : tensor<512xf32>
    %v8299 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8300 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8301 = stablehlo.divide %v8292, %v8299 : tensor<512xf32>
    %v8302 = stablehlo.divide %v8298, %v8300 : tensor<512xf32>
    %v8303 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8304 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8305 = stablehlo.sqrt %v8302 : tensor<512xf32>
    %v8306 = stablehlo.add %v8305, %v8304 : tensor<512xf32>
    %v8307 = stablehlo.divide %v8301, %v8306 : tensor<512xf32>
    %v8308 = stablehlo.multiply %v8303, %v8307 : tensor<512xf32>
    %v8309 = stablehlo.subtract %s4b0bt2, %v8308 : tensor<512xf32>
    %v8310 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8311 = stablehlo.multiply %v8310, %v8303 : tensor<512xf32>
    %v8312 = stablehlo.multiply %v8311, %s4b0bt2 : tensor<512xf32>
    %v8313 = stablehlo.subtract %v8309, %v8312 : tensor<512xf32>
    %arsums4b1W1 = "stablehlo.all_reduce"(%v1707) ({
    ^bb0(%aras4b1W1: tensor<f32>, %arbs4b1W1: tensor<f32>):
      %aradds4b1W1 = stablehlo.add %aras4b1W1, %arbs4b1W1 : tensor<f32>
      stablehlo.return %aradds4b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W1 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b1W1 = stablehlo.divide %arsums4b1W1, %arns4b1W1 : tensor<512x512x3x3xf32>
    %v8314 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8315 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8316 = stablehlo.multiply %v8314, %s4b1W1m : tensor<512x512x3x3xf32>
    %v8317 = stablehlo.multiply %v8315, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v8318 = stablehlo.add %v8316, %v8317 : tensor<512x512x3x3xf32>
    %v8319 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8320 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8321 = stablehlo.multiply %v8319, %s4b1W1v : tensor<512x512x3x3xf32>
    %v8322 = stablehlo.multiply %armeans4b1W1, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v8323 = stablehlo.multiply %v8320, %v8322 : tensor<512x512x3x3xf32>
    %v8324 = stablehlo.add %v8321, %v8323 : tensor<512x512x3x3xf32>
    %v8325 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8326 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8327 = stablehlo.multiply %v8325, %s4b1W1m : tensor<512x512x3x3xf32>
    %v8328 = stablehlo.multiply %v8326, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v8329 = stablehlo.add %v8327, %v8328 : tensor<512x512x3x3xf32>
    %v8330 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8331 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8332 = stablehlo.multiply %v8330, %s4b1W1v : tensor<512x512x3x3xf32>
    %v8333 = stablehlo.multiply %armeans4b1W1, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v8334 = stablehlo.multiply %v8331, %v8333 : tensor<512x512x3x3xf32>
    %v8335 = stablehlo.add %v8332, %v8334 : tensor<512x512x3x3xf32>
    %v8336 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8337 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8338 = stablehlo.divide %v8329, %v8336 : tensor<512x512x3x3xf32>
    %v8339 = stablehlo.divide %v8335, %v8337 : tensor<512x512x3x3xf32>
    %v8340 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8341 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8342 = stablehlo.sqrt %v8339 : tensor<512x512x3x3xf32>
    %v8343 = stablehlo.add %v8342, %v8341 : tensor<512x512x3x3xf32>
    %v8344 = stablehlo.divide %v8338, %v8343 : tensor<512x512x3x3xf32>
    %v8345 = stablehlo.multiply %v8340, %v8344 : tensor<512x512x3x3xf32>
    %v8346 = stablehlo.subtract %s4b1W1, %v8345 : tensor<512x512x3x3xf32>
    %v8347 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8348 = stablehlo.multiply %v8347, %v8340 : tensor<512x512x3x3xf32>
    %v8349 = stablehlo.multiply %v8348, %s4b1W1 : tensor<512x512x3x3xf32>
    %v8350 = stablehlo.subtract %v8346, %v8349 : tensor<512x512x3x3xf32>
    %arsums4b1g1 = "stablehlo.all_reduce"(%v1721) ({
    ^bb0(%aras4b1g1: tensor<f32>, %arbs4b1g1: tensor<f32>):
      %aradds4b1g1 = stablehlo.add %aras4b1g1, %arbs4b1g1 : tensor<f32>
      stablehlo.return %aradds4b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g1 = stablehlo.divide %arsums4b1g1, %arns4b1g1 : tensor<512xf32>
    %v8351 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8352 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8353 = stablehlo.multiply %v8351, %s4b1g1m : tensor<512xf32>
    %v8354 = stablehlo.multiply %v8352, %armeans4b1g1 : tensor<512xf32>
    %v8355 = stablehlo.add %v8353, %v8354 : tensor<512xf32>
    %v8356 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8357 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8358 = stablehlo.multiply %v8356, %s4b1g1v : tensor<512xf32>
    %v8359 = stablehlo.multiply %armeans4b1g1, %armeans4b1g1 : tensor<512xf32>
    %v8360 = stablehlo.multiply %v8357, %v8359 : tensor<512xf32>
    %v8361 = stablehlo.add %v8358, %v8360 : tensor<512xf32>
    %v8362 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8363 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8364 = stablehlo.multiply %v8362, %s4b1g1m : tensor<512xf32>
    %v8365 = stablehlo.multiply %v8363, %armeans4b1g1 : tensor<512xf32>
    %v8366 = stablehlo.add %v8364, %v8365 : tensor<512xf32>
    %v8367 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8368 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8369 = stablehlo.multiply %v8367, %s4b1g1v : tensor<512xf32>
    %v8370 = stablehlo.multiply %armeans4b1g1, %armeans4b1g1 : tensor<512xf32>
    %v8371 = stablehlo.multiply %v8368, %v8370 : tensor<512xf32>
    %v8372 = stablehlo.add %v8369, %v8371 : tensor<512xf32>
    %v8373 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8374 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8375 = stablehlo.divide %v8366, %v8373 : tensor<512xf32>
    %v8376 = stablehlo.divide %v8372, %v8374 : tensor<512xf32>
    %v8377 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8378 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8379 = stablehlo.sqrt %v8376 : tensor<512xf32>
    %v8380 = stablehlo.add %v8379, %v8378 : tensor<512xf32>
    %v8381 = stablehlo.divide %v8375, %v8380 : tensor<512xf32>
    %v8382 = stablehlo.multiply %v8377, %v8381 : tensor<512xf32>
    %v8383 = stablehlo.subtract %s4b1g1, %v8382 : tensor<512xf32>
    %v8384 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8385 = stablehlo.multiply %v8384, %v8377 : tensor<512xf32>
    %v8386 = stablehlo.multiply %v8385, %s4b1g1 : tensor<512xf32>
    %v8387 = stablehlo.subtract %v8383, %v8386 : tensor<512xf32>
    %arsums4b1bt1 = "stablehlo.all_reduce"(%v1724) ({
    ^bb0(%aras4b1bt1: tensor<f32>, %arbs4b1bt1: tensor<f32>):
      %aradds4b1bt1 = stablehlo.add %aras4b1bt1, %arbs4b1bt1 : tensor<f32>
      stablehlo.return %aradds4b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1bt1 = stablehlo.divide %arsums4b1bt1, %arns4b1bt1 : tensor<512xf32>
    %v8388 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8389 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8390 = stablehlo.multiply %v8388, %s4b1bt1m : tensor<512xf32>
    %v8391 = stablehlo.multiply %v8389, %armeans4b1bt1 : tensor<512xf32>
    %v8392 = stablehlo.add %v8390, %v8391 : tensor<512xf32>
    %v8393 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8394 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8395 = stablehlo.multiply %v8393, %s4b1bt1v : tensor<512xf32>
    %v8396 = stablehlo.multiply %armeans4b1bt1, %armeans4b1bt1 : tensor<512xf32>
    %v8397 = stablehlo.multiply %v8394, %v8396 : tensor<512xf32>
    %v8398 = stablehlo.add %v8395, %v8397 : tensor<512xf32>
    %v8399 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8400 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8401 = stablehlo.multiply %v8399, %s4b1bt1m : tensor<512xf32>
    %v8402 = stablehlo.multiply %v8400, %armeans4b1bt1 : tensor<512xf32>
    %v8403 = stablehlo.add %v8401, %v8402 : tensor<512xf32>
    %v8404 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8405 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8406 = stablehlo.multiply %v8404, %s4b1bt1v : tensor<512xf32>
    %v8407 = stablehlo.multiply %armeans4b1bt1, %armeans4b1bt1 : tensor<512xf32>
    %v8408 = stablehlo.multiply %v8405, %v8407 : tensor<512xf32>
    %v8409 = stablehlo.add %v8406, %v8408 : tensor<512xf32>
    %v8410 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8411 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8412 = stablehlo.divide %v8403, %v8410 : tensor<512xf32>
    %v8413 = stablehlo.divide %v8409, %v8411 : tensor<512xf32>
    %v8414 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8415 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8416 = stablehlo.sqrt %v8413 : tensor<512xf32>
    %v8417 = stablehlo.add %v8416, %v8415 : tensor<512xf32>
    %v8418 = stablehlo.divide %v8412, %v8417 : tensor<512xf32>
    %v8419 = stablehlo.multiply %v8414, %v8418 : tensor<512xf32>
    %v8420 = stablehlo.subtract %s4b1bt1, %v8419 : tensor<512xf32>
    %v8421 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8422 = stablehlo.multiply %v8421, %v8414 : tensor<512xf32>
    %v8423 = stablehlo.multiply %v8422, %s4b1bt1 : tensor<512xf32>
    %v8424 = stablehlo.subtract %v8420, %v8423 : tensor<512xf32>
    %arsums4b1W2 = "stablehlo.all_reduce"(%v1730) ({
    ^bb0(%aras4b1W2: tensor<f32>, %arbs4b1W2: tensor<f32>):
      %aradds4b1W2 = stablehlo.add %aras4b1W2, %arbs4b1W2 : tensor<f32>
      stablehlo.return %aradds4b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W2 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b1W2 = stablehlo.divide %arsums4b1W2, %arns4b1W2 : tensor<512x512x3x3xf32>
    %v8425 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8426 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8427 = stablehlo.multiply %v8425, %s4b1W2m : tensor<512x512x3x3xf32>
    %v8428 = stablehlo.multiply %v8426, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v8429 = stablehlo.add %v8427, %v8428 : tensor<512x512x3x3xf32>
    %v8430 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8431 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8432 = stablehlo.multiply %v8430, %s4b1W2v : tensor<512x512x3x3xf32>
    %v8433 = stablehlo.multiply %armeans4b1W2, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v8434 = stablehlo.multiply %v8431, %v8433 : tensor<512x512x3x3xf32>
    %v8435 = stablehlo.add %v8432, %v8434 : tensor<512x512x3x3xf32>
    %v8436 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8437 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8438 = stablehlo.multiply %v8436, %s4b1W2m : tensor<512x512x3x3xf32>
    %v8439 = stablehlo.multiply %v8437, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v8440 = stablehlo.add %v8438, %v8439 : tensor<512x512x3x3xf32>
    %v8441 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8442 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8443 = stablehlo.multiply %v8441, %s4b1W2v : tensor<512x512x3x3xf32>
    %v8444 = stablehlo.multiply %armeans4b1W2, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v8445 = stablehlo.multiply %v8442, %v8444 : tensor<512x512x3x3xf32>
    %v8446 = stablehlo.add %v8443, %v8445 : tensor<512x512x3x3xf32>
    %v8447 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8448 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8449 = stablehlo.divide %v8440, %v8447 : tensor<512x512x3x3xf32>
    %v8450 = stablehlo.divide %v8446, %v8448 : tensor<512x512x3x3xf32>
    %v8451 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8452 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8453 = stablehlo.sqrt %v8450 : tensor<512x512x3x3xf32>
    %v8454 = stablehlo.add %v8453, %v8452 : tensor<512x512x3x3xf32>
    %v8455 = stablehlo.divide %v8449, %v8454 : tensor<512x512x3x3xf32>
    %v8456 = stablehlo.multiply %v8451, %v8455 : tensor<512x512x3x3xf32>
    %v8457 = stablehlo.subtract %s4b1W2, %v8456 : tensor<512x512x3x3xf32>
    %v8458 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8459 = stablehlo.multiply %v8458, %v8451 : tensor<512x512x3x3xf32>
    %v8460 = stablehlo.multiply %v8459, %s4b1W2 : tensor<512x512x3x3xf32>
    %v8461 = stablehlo.subtract %v8457, %v8460 : tensor<512x512x3x3xf32>
    %arsums4b1g2 = "stablehlo.all_reduce"(%v1744) ({
    ^bb0(%aras4b1g2: tensor<f32>, %arbs4b1g2: tensor<f32>):
      %aradds4b1g2 = stablehlo.add %aras4b1g2, %arbs4b1g2 : tensor<f32>
      stablehlo.return %aradds4b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g2 = stablehlo.divide %arsums4b1g2, %arns4b1g2 : tensor<512xf32>
    %v8462 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8463 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8464 = stablehlo.multiply %v8462, %s4b1g2m : tensor<512xf32>
    %v8465 = stablehlo.multiply %v8463, %armeans4b1g2 : tensor<512xf32>
    %v8466 = stablehlo.add %v8464, %v8465 : tensor<512xf32>
    %v8467 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8468 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8469 = stablehlo.multiply %v8467, %s4b1g2v : tensor<512xf32>
    %v8470 = stablehlo.multiply %armeans4b1g2, %armeans4b1g2 : tensor<512xf32>
    %v8471 = stablehlo.multiply %v8468, %v8470 : tensor<512xf32>
    %v8472 = stablehlo.add %v8469, %v8471 : tensor<512xf32>
    %v8473 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8474 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8475 = stablehlo.multiply %v8473, %s4b1g2m : tensor<512xf32>
    %v8476 = stablehlo.multiply %v8474, %armeans4b1g2 : tensor<512xf32>
    %v8477 = stablehlo.add %v8475, %v8476 : tensor<512xf32>
    %v8478 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8479 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8480 = stablehlo.multiply %v8478, %s4b1g2v : tensor<512xf32>
    %v8481 = stablehlo.multiply %armeans4b1g2, %armeans4b1g2 : tensor<512xf32>
    %v8482 = stablehlo.multiply %v8479, %v8481 : tensor<512xf32>
    %v8483 = stablehlo.add %v8480, %v8482 : tensor<512xf32>
    %v8484 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8485 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8486 = stablehlo.divide %v8477, %v8484 : tensor<512xf32>
    %v8487 = stablehlo.divide %v8483, %v8485 : tensor<512xf32>
    %v8488 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8489 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8490 = stablehlo.sqrt %v8487 : tensor<512xf32>
    %v8491 = stablehlo.add %v8490, %v8489 : tensor<512xf32>
    %v8492 = stablehlo.divide %v8486, %v8491 : tensor<512xf32>
    %v8493 = stablehlo.multiply %v8488, %v8492 : tensor<512xf32>
    %v8494 = stablehlo.subtract %s4b1g2, %v8493 : tensor<512xf32>
    %v8495 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8496 = stablehlo.multiply %v8495, %v8488 : tensor<512xf32>
    %v8497 = stablehlo.multiply %v8496, %s4b1g2 : tensor<512xf32>
    %v8498 = stablehlo.subtract %v8494, %v8497 : tensor<512xf32>
    %arsums4b1bt2 = "stablehlo.all_reduce"(%v1747) ({
    ^bb0(%aras4b1bt2: tensor<f32>, %arbs4b1bt2: tensor<f32>):
      %aradds4b1bt2 = stablehlo.add %aras4b1bt2, %arbs4b1bt2 : tensor<f32>
      stablehlo.return %aradds4b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1bt2 = stablehlo.divide %arsums4b1bt2, %arns4b1bt2 : tensor<512xf32>
    %v8499 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8500 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8501 = stablehlo.multiply %v8499, %s4b1bt2m : tensor<512xf32>
    %v8502 = stablehlo.multiply %v8500, %armeans4b1bt2 : tensor<512xf32>
    %v8503 = stablehlo.add %v8501, %v8502 : tensor<512xf32>
    %v8504 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8505 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8506 = stablehlo.multiply %v8504, %s4b1bt2v : tensor<512xf32>
    %v8507 = stablehlo.multiply %armeans4b1bt2, %armeans4b1bt2 : tensor<512xf32>
    %v8508 = stablehlo.multiply %v8505, %v8507 : tensor<512xf32>
    %v8509 = stablehlo.add %v8506, %v8508 : tensor<512xf32>
    %v8510 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8511 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8512 = stablehlo.multiply %v8510, %s4b1bt2m : tensor<512xf32>
    %v8513 = stablehlo.multiply %v8511, %armeans4b1bt2 : tensor<512xf32>
    %v8514 = stablehlo.add %v8512, %v8513 : tensor<512xf32>
    %v8515 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8516 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8517 = stablehlo.multiply %v8515, %s4b1bt2v : tensor<512xf32>
    %v8518 = stablehlo.multiply %armeans4b1bt2, %armeans4b1bt2 : tensor<512xf32>
    %v8519 = stablehlo.multiply %v8516, %v8518 : tensor<512xf32>
    %v8520 = stablehlo.add %v8517, %v8519 : tensor<512xf32>
    %v8521 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8522 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8523 = stablehlo.divide %v8514, %v8521 : tensor<512xf32>
    %v8524 = stablehlo.divide %v8520, %v8522 : tensor<512xf32>
    %v8525 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8526 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8527 = stablehlo.sqrt %v8524 : tensor<512xf32>
    %v8528 = stablehlo.add %v8527, %v8526 : tensor<512xf32>
    %v8529 = stablehlo.divide %v8523, %v8528 : tensor<512xf32>
    %v8530 = stablehlo.multiply %v8525, %v8529 : tensor<512xf32>
    %v8531 = stablehlo.subtract %s4b1bt2, %v8530 : tensor<512xf32>
    %v8532 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8533 = stablehlo.multiply %v8532, %v8525 : tensor<512xf32>
    %v8534 = stablehlo.multiply %v8533, %s4b1bt2 : tensor<512xf32>
    %v8535 = stablehlo.subtract %v8531, %v8534 : tensor<512xf32>
    %arsumWd = "stablehlo.all_reduce"(%v1581) ({
    ^bb0(%araWd: tensor<f32>, %arbWd: tensor<f32>):
      %araddWd = stablehlo.add %araWd, %arbWd : tensor<f32>
      stablehlo.return %araddWd : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x10xf32>) -> tensor<512x10xf32>
    %arnWd = stablehlo.constant dense<2.0> : tensor<512x10xf32>
    %armeanWd = stablehlo.divide %arsumWd, %arnWd : tensor<512x10xf32>
    %v8536 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8537 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8538 = stablehlo.multiply %v8536, %Wdm : tensor<512x10xf32>
    %v8539 = stablehlo.multiply %v8537, %armeanWd : tensor<512x10xf32>
    %v8540 = stablehlo.add %v8538, %v8539 : tensor<512x10xf32>
    %v8541 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8542 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8543 = stablehlo.multiply %v8541, %Wdv : tensor<512x10xf32>
    %v8544 = stablehlo.multiply %armeanWd, %armeanWd : tensor<512x10xf32>
    %v8545 = stablehlo.multiply %v8542, %v8544 : tensor<512x10xf32>
    %v8546 = stablehlo.add %v8543, %v8545 : tensor<512x10xf32>
    %v8547 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8548 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8549 = stablehlo.multiply %v8547, %Wdm : tensor<512x10xf32>
    %v8550 = stablehlo.multiply %v8548, %armeanWd : tensor<512x10xf32>
    %v8551 = stablehlo.add %v8549, %v8550 : tensor<512x10xf32>
    %v8552 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8553 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8554 = stablehlo.multiply %v8552, %Wdv : tensor<512x10xf32>
    %v8555 = stablehlo.multiply %armeanWd, %armeanWd : tensor<512x10xf32>
    %v8556 = stablehlo.multiply %v8553, %v8555 : tensor<512x10xf32>
    %v8557 = stablehlo.add %v8554, %v8556 : tensor<512x10xf32>
    %v8558 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8559 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8560 = stablehlo.divide %v8551, %v8558 : tensor<512x10xf32>
    %v8561 = stablehlo.divide %v8557, %v8559 : tensor<512x10xf32>
    %v8562 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8563 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8564 = stablehlo.sqrt %v8561 : tensor<512x10xf32>
    %v8565 = stablehlo.add %v8564, %v8563 : tensor<512x10xf32>
    %v8566 = stablehlo.divide %v8560, %v8565 : tensor<512x10xf32>
    %v8567 = stablehlo.multiply %v8562, %v8566 : tensor<512x10xf32>
    %v8568 = stablehlo.subtract %Wd, %v8567 : tensor<512x10xf32>
    %v8569 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v8570 = stablehlo.multiply %v8569, %v8562 : tensor<512x10xf32>
    %v8571 = stablehlo.multiply %v8570, %Wd : tensor<512x10xf32>
    %v8572 = stablehlo.subtract %v8568, %v8571 : tensor<512x10xf32>
    %arsumbd = "stablehlo.all_reduce"(%v1583) ({
    ^bb0(%arabd: tensor<f32>, %arbbd: tensor<f32>):
      %araddbd = stablehlo.add %arabd, %arbbd : tensor<f32>
      stablehlo.return %araddbd : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<10xf32>) -> tensor<10xf32>
    %arnbd = stablehlo.constant dense<2.0> : tensor<10xf32>
    %armeanbd = stablehlo.divide %arsumbd, %arnbd : tensor<10xf32>
    %v8573 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8574 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8575 = stablehlo.multiply %v8573, %bdm : tensor<10xf32>
    %v8576 = stablehlo.multiply %v8574, %armeanbd : tensor<10xf32>
    %v8577 = stablehlo.add %v8575, %v8576 : tensor<10xf32>
    %v8578 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8579 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8580 = stablehlo.multiply %v8578, %bdv : tensor<10xf32>
    %v8581 = stablehlo.multiply %armeanbd, %armeanbd : tensor<10xf32>
    %v8582 = stablehlo.multiply %v8579, %v8581 : tensor<10xf32>
    %v8583 = stablehlo.add %v8580, %v8582 : tensor<10xf32>
    %v8584 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8585 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8586 = stablehlo.multiply %v8584, %bdm : tensor<10xf32>
    %v8587 = stablehlo.multiply %v8585, %armeanbd : tensor<10xf32>
    %v8588 = stablehlo.add %v8586, %v8587 : tensor<10xf32>
    %v8589 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8590 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8591 = stablehlo.multiply %v8589, %bdv : tensor<10xf32>
    %v8592 = stablehlo.multiply %armeanbd, %armeanbd : tensor<10xf32>
    %v8593 = stablehlo.multiply %v8590, %v8592 : tensor<10xf32>
    %v8594 = stablehlo.add %v8591, %v8593 : tensor<10xf32>
    %v8595 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8596 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8597 = stablehlo.divide %v8588, %v8595 : tensor<10xf32>
    %v8598 = stablehlo.divide %v8594, %v8596 : tensor<10xf32>
    %v8599 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8600 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8601 = stablehlo.sqrt %v8598 : tensor<10xf32>
    %v8602 = stablehlo.add %v8601, %v8600 : tensor<10xf32>
    %v8603 = stablehlo.divide %v8597, %v8602 : tensor<10xf32>
    %v8604 = stablehlo.multiply %v8599, %v8603 : tensor<10xf32>
    %v8605 = stablehlo.subtract %bd, %v8604 : tensor<10xf32>
    %v8606 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v8607 = stablehlo.multiply %v8606, %v8599 : tensor<10xf32>
    %v8608 = stablehlo.multiply %v8607, %bd : tensor<10xf32>
    %v8609 = stablehlo.subtract %v8605, %v8608 : tensor<10xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1569 : tensor<128x10xf32>
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
    return %v4576, %v4613, %v4650, %v4687, %v4724, %v4761, %v4798, %v4835, %v4872, %v4909, %v4946, %v4983, %v5020, %v5057, %v5094, %v5131, %v5168, %v5205, %v5242, %v5279, %v5316, %v5353, %v5390, %v5427, %v5464, %v5501, %v5538, %v5575, %v5612, %v5649, %v5686, %v5723, %v5760, %v5797, %v5834, %v5871, %v5908, %v5945, %v5982, %v6019, %v6056, %v6093, %v6130, %v6167, %v6204, %v6241, %v6278, %v6315, %v6352, %v6389, %v6426, %v6463, %v6500, %v6537, %v6574, %v6611, %v6648, %v6685, %v6722, %v6759, %v6796, %v6833, %v6870, %v6907, %v6944, %v6981, %v7018, %v7055, %v7092, %v7129, %v7166, %v7203, %v7240, %v7277, %v7314, %v7351, %v7388, %v7425, %v7462, %v7499, %v7536, %v7573, %v7610, %v7647, %v7684, %v7721, %v7758, %v7795, %v7832, %v7869, %v7906, %v7943, %v7980, %v8017, %v8054, %v8091, %v8128, %v8165, %v8202, %v8239, %v8276, %v8313, %v8350, %v8387, %v8424, %v8461, %v8498, %v8535, %v8572, %v8609, %v4544, %v4581, %v4618, %v4655, %v4692, %v4729, %v4766, %v4803, %v4840, %v4877, %v4914, %v4951, %v4988, %v5025, %v5062, %v5099, %v5136, %v5173, %v5210, %v5247, %v5284, %v5321, %v5358, %v5395, %v5432, %v5469, %v5506, %v5543, %v5580, %v5617, %v5654, %v5691, %v5728, %v5765, %v5802, %v5839, %v5876, %v5913, %v5950, %v5987, %v6024, %v6061, %v6098, %v6135, %v6172, %v6209, %v6246, %v6283, %v6320, %v6357, %v6394, %v6431, %v6468, %v6505, %v6542, %v6579, %v6616, %v6653, %v6690, %v6727, %v6764, %v6801, %v6838, %v6875, %v6912, %v6949, %v6986, %v7023, %v7060, %v7097, %v7134, %v7171, %v7208, %v7245, %v7282, %v7319, %v7356, %v7393, %v7430, %v7467, %v7504, %v7541, %v7578, %v7615, %v7652, %v7689, %v7726, %v7763, %v7800, %v7837, %v7874, %v7911, %v7948, %v7985, %v8022, %v8059, %v8096, %v8133, %v8170, %v8207, %v8244, %v8281, %v8318, %v8355, %v8392, %v8429, %v8466, %v8503, %v8540, %v8577, %v4550, %v4587, %v4624, %v4661, %v4698, %v4735, %v4772, %v4809, %v4846, %v4883, %v4920, %v4957, %v4994, %v5031, %v5068, %v5105, %v5142, %v5179, %v5216, %v5253, %v5290, %v5327, %v5364, %v5401, %v5438, %v5475, %v5512, %v5549, %v5586, %v5623, %v5660, %v5697, %v5734, %v5771, %v5808, %v5845, %v5882, %v5919, %v5956, %v5993, %v6030, %v6067, %v6104, %v6141, %v6178, %v6215, %v6252, %v6289, %v6326, %v6363, %v6400, %v6437, %v6474, %v6511, %v6548, %v6585, %v6622, %v6659, %v6696, %v6733, %v6770, %v6807, %v6844, %v6881, %v6918, %v6955, %v6992, %v7029, %v7066, %v7103, %v7140, %v7177, %v7214, %v7251, %v7288, %v7325, %v7362, %v7399, %v7436, %v7473, %v7510, %v7547, %v7584, %v7621, %v7658, %v7695, %v7732, %v7769, %v7806, %v7843, %v7880, %v7917, %v7954, %v7991, %v8028, %v8065, %v8102, %v8139, %v8176, %v8213, %v8250, %v8287, %v8324, %v8361, %v8398, %v8435, %v8472, %v8509, %v8546, %v8583, %loss, %bc1, %bc2, %v4468, %v4469, %v4470, %v4471, %v4472, %v4473, %v4474, %v4475, %v4476, %v4477, %v4478, %v4479, %v4480, %v4481, %v4482, %v4483, %v4484, %v4485, %v4486, %v4487, %v4488, %v4489, %v4490, %v4491, %v4492, %v4493, %v4494, %v4495, %v4496, %v4497, %v4498, %v4499, %v4500, %v4501, %v4502, %v4503, %v4504, %v4505, %v4506, %v4507, %v4508, %v4509, %v4510, %v4511, %v4512, %v4513, %v4514, %v4515, %v4516, %v4517, %v4518, %v4519, %v4520, %v4521, %v4522, %v4523, %v4524, %v4525, %v4526, %v4527, %v4528, %v4529, %v4530, %v4531, %v4532, %v4533, %v4534, %v4535, %v4536, %v4537, %v4538, %v4539 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}

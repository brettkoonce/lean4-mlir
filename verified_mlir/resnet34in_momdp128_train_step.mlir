module @m {
  func.func @resnet34in_momdp128_train_step(%x: tensor<128x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x1x1xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x1x1xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x1x1xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x1000xf32>, %bd: tensor<1000xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x3x3xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b1W1m: tensor<64x64x3x3xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b2W1m: tensor<64x64x3x3xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %d2W1m: tensor<128x64x3x3xf32>, %d2g1m: tensor<128xf32>, %d2bt1m: tensor<128xf32>, %d2W2m: tensor<128x128x3x3xf32>, %d2g2m: tensor<128xf32>, %d2bt2m: tensor<128xf32>, %d2Wpm: tensor<128x64x1x1xf32>, %d2gpm: tensor<128xf32>, %d2btpm: tensor<128xf32>, %s2b0W1m: tensor<128x128x3x3xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b1W1m: tensor<128x128x3x3xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b2W1m: tensor<128x128x3x3xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %d3W1m: tensor<256x128x3x3xf32>, %d3g1m: tensor<256xf32>, %d3bt1m: tensor<256xf32>, %d3W2m: tensor<256x256x3x3xf32>, %d3g2m: tensor<256xf32>, %d3bt2m: tensor<256xf32>, %d3Wpm: tensor<256x128x1x1xf32>, %d3gpm: tensor<256xf32>, %d3btpm: tensor<256xf32>, %s3b0W1m: tensor<256x256x3x3xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b1W1m: tensor<256x256x3x3xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b2W1m: tensor<256x256x3x3xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b3W1m: tensor<256x256x3x3xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b4W1m: tensor<256x256x3x3xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %d4W1m: tensor<512x256x3x3xf32>, %d4g1m: tensor<512xf32>, %d4bt1m: tensor<512xf32>, %d4W2m: tensor<512x512x3x3xf32>, %d4g2m: tensor<512xf32>, %d4bt2m: tensor<512xf32>, %d4Wpm: tensor<512x256x1x1xf32>, %d4gpm: tensor<512xf32>, %d4btpm: tensor<512xf32>, %s4b0W1m: tensor<512x512x3x3xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b1W1m: tensor<512x512x3x3xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %Wdm: tensor<512x1000xf32>, %bdm: tensor<1000xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x3x3xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b1W1v: tensor<64x64x3x3xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b2W1v: tensor<64x64x3x3xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %d2W1v: tensor<128x64x3x3xf32>, %d2g1v: tensor<128xf32>, %d2bt1v: tensor<128xf32>, %d2W2v: tensor<128x128x3x3xf32>, %d2g2v: tensor<128xf32>, %d2bt2v: tensor<128xf32>, %d2Wpv: tensor<128x64x1x1xf32>, %d2gpv: tensor<128xf32>, %d2btpv: tensor<128xf32>, %s2b0W1v: tensor<128x128x3x3xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b1W1v: tensor<128x128x3x3xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b2W1v: tensor<128x128x3x3xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %d3W1v: tensor<256x128x3x3xf32>, %d3g1v: tensor<256xf32>, %d3bt1v: tensor<256xf32>, %d3W2v: tensor<256x256x3x3xf32>, %d3g2v: tensor<256xf32>, %d3bt2v: tensor<256xf32>, %d3Wpv: tensor<256x128x1x1xf32>, %d3gpv: tensor<256xf32>, %d3btpv: tensor<256xf32>, %s3b0W1v: tensor<256x256x3x3xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b1W1v: tensor<256x256x3x3xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b2W1v: tensor<256x256x3x3xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b3W1v: tensor<256x256x3x3xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b4W1v: tensor<256x256x3x3xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %d4W1v: tensor<512x256x3x3xf32>, %d4g1v: tensor<512xf32>, %d4bt1v: tensor<512xf32>, %d4W2v: tensor<512x512x3x3xf32>, %d4g2v: tensor<512xf32>, %d4bt2v: tensor<512xf32>, %d4Wpv: tensor<512x256x1x1xf32>, %d4gpv: tensor<512xf32>, %d4btpv: tensor<512xf32>, %s4b0W1v: tensor<512x512x3x3xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b1W1v: tensor<512x512x3x3xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %Wdv: tensor<512x1000xf32>, %bdv: tensor<1000xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %d2n1mui: tensor<128xf32>, %d2n1vari: tensor<128xf32>, %d2n2mui: tensor<128xf32>, %d2n2vari: tensor<128xf32>, %d2npmui: tensor<128xf32>, %d2npvari: tensor<128xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %d3n1mui: tensor<256xf32>, %d3n1vari: tensor<256xf32>, %d3n2mui: tensor<256xf32>, %d3n2vari: tensor<256xf32>, %d3npmui: tensor<256xf32>, %d3npvari: tensor<256xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %d4n1mui: tensor<512xf32>, %d4n1vari: tensor<512xf32>, %d4n2mui: tensor<512xf32>, %d4n2vari: tensor<512xf32>, %d4npmui: tensor<512xf32>, %d4npvari: tensor<512xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %onehot: tensor<128x1000xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) {
    // ── ResNet-34 batch-BN heavy-ball momentum + coupled L2 train step, DATA-PARALLEL over 2 replicas ──
    // Every line is pretty(verified AST node), the per-parameter `%arsum*` all_reduce /
    // `%armean*` blocks included: pretty(allReduceMeanF), whose den is the replica MEAN of
    // the per-replica gradient nodes (4d piece 2). BatchNorm is SYNCHRONISED: every BN
    // layer all-reduces its mu, then var_r + (mu_r - mu)^2 (bnBatchVarAtB, Chan's parallel
    // variance), before normalising with the global [mu | var] (bnSyncF); its
    // backward all-reduces the two dy-reductions (bnSyncDyStatsB -> bnSyncBack), and the gamma
    // gradient reads the same global x-hat (bnSyncGammaGradB). Each replica therefore computes
    // its shard of the GLOBAL-batch function, and this step IS the single-device step at the
    // global batch N x b: proved as ResNet34SyncTieB.r34_net_syncTiedB (every all-reduced
    // gradient) and StableHLO.resnet34FwdGraphSync_full_shard (the forward), both in
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
    %v1560 = stablehlo.dot_general %v1559, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x1000xf32>) -> tensor<128x1000xf32>
    %v1561 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<128x1000xf32>
    %v1562 = stablehlo.add %v1560, %v1561 : tensor<128x1000xf32>
    %v1563 = stablehlo.reshape %v1562 : (tensor<128x1000xf32>) -> tensor<128x1x1000xf32>
    %v1564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1565 = stablehlo.exponential %v1563 : tensor<128x1x1000xf32>
    %v1566 = stablehlo.reduce(%v1565 init: %v1564) applies stablehlo.add across dimensions = [2] : (tensor<128x1x1000xf32>, tensor<f32>) -> tensor<128x1xf32>
    %v1567 = stablehlo.broadcast_in_dim %v1566, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x1x1000xf32>
    %v1568 = stablehlo.divide %v1565, %v1567 : tensor<128x1x1000xf32>
    %v1569 = stablehlo.reshape %v1568 : (tensor<128x1x1000xf32>) -> tensor<128x1000xf32>
    %v1570 = stablehlo.subtract %v1569, %onehot : tensor<128x1000xf32>
    %v1571 = stablehlo.constant dense<0.100000> : tensor<128x1000xf32>
    %v1572 = stablehlo.multiply %onehot, %v1571 : tensor<128x1000xf32>
    %v1573 = stablehlo.add %v1570, %v1572 : tensor<128x1000xf32>
    %v1574 = stablehlo.constant dense<-0.000100> : tensor<128x1000xf32>
    %v1575 = stablehlo.add %v1573, %v1574 : tensor<128x1000xf32>
    %v1576 = stablehlo.constant dense<128.0> : tensor<128x1000xf32>
    %v1577 = stablehlo.divide %v1575, %v1576 : tensor<128x1000xf32>
    %v1578 = stablehlo.reshape %v1577 : (tensor<128x1000xf32>) -> tensor<128x1x1000xf32>
    %v1579 = stablehlo.dot_general %v1578, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x1000xf32>, tensor<512x1000xf32>) -> tensor<128x1x512xf32>
    %v1580 = stablehlo.reshape %v1579 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v1581 = stablehlo.dot_general %v1559, %v1577, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x1000xf32>) -> tensor<512x1000xf32>
    %v1582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1583 = stablehlo.reduce(%v1577 init: %v1582) applies stablehlo.add across dimensions = [0] : (tensor<128x1000xf32>, tensor<f32>) -> tensor<1000xf32>
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
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %arsumsW = "stablehlo.all_reduce"(%v4450) ({
    ^bb0(%arasW: tensor<f32>, %arbsW: tensor<f32>):
      %araddsW = stablehlo.add %arasW, %arbsW : tensor<f32>
      stablehlo.return %araddsW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
    %arnsW = stablehlo.constant dense<2.0> : tensor<64x3x7x7xf32>
    %armeansW = stablehlo.divide %arsumsW, %arnsW : tensor<64x3x7x7xf32>
    %v4540 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4541 = stablehlo.multiply %v4540, %sW : tensor<64x3x7x7xf32>
    %v4542 = stablehlo.add %v4541, %armeansW : tensor<64x3x7x7xf32>
    %v4543 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4544 = stablehlo.multiply %v4543, %sWv : tensor<64x3x7x7xf32>
    %v4545 = stablehlo.add %v4544, %v4542 : tensor<64x3x7x7xf32>
    %v4546 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4547 = stablehlo.multiply %v4546, %v4545 : tensor<64x3x7x7xf32>
    %v4548 = stablehlo.subtract %sW, %v4547 : tensor<64x3x7x7xf32>
    %arsumsg = "stablehlo.all_reduce"(%v4464) ({
    ^bb0(%arasg: tensor<f32>, %arbsg: tensor<f32>):
      %araddsg = stablehlo.add %arasg, %arbsg : tensor<f32>
      stablehlo.return %araddsg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsg = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeansg = stablehlo.divide %arsumsg, %arnsg : tensor<64xf32>
    %v4549 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4550 = stablehlo.multiply %v4549, %sg : tensor<64xf32>
    %v4551 = stablehlo.add %v4550, %armeansg : tensor<64xf32>
    %v4552 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4553 = stablehlo.multiply %v4552, %sgv : tensor<64xf32>
    %v4554 = stablehlo.add %v4553, %v4551 : tensor<64xf32>
    %v4555 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4556 = stablehlo.multiply %v4555, %v4554 : tensor<64xf32>
    %v4557 = stablehlo.subtract %sg, %v4556 : tensor<64xf32>
    %arsumsbt = "stablehlo.all_reduce"(%v4467) ({
    ^bb0(%arasbt: tensor<f32>, %arbsbt: tensor<f32>):
      %araddsbt = stablehlo.add %arasbt, %arbsbt : tensor<f32>
      stablehlo.return %araddsbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsbt = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeansbt = stablehlo.divide %arsumsbt, %arnsbt : tensor<64xf32>
    %v4558 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4559 = stablehlo.multiply %v4558, %sbt : tensor<64xf32>
    %v4560 = stablehlo.add %v4559, %armeansbt : tensor<64xf32>
    %v4561 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4562 = stablehlo.multiply %v4561, %sbtv : tensor<64xf32>
    %v4563 = stablehlo.add %v4562, %v4560 : tensor<64xf32>
    %v4564 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4565 = stablehlo.multiply %v4564, %v4563 : tensor<64xf32>
    %v4566 = stablehlo.subtract %sbt, %v4565 : tensor<64xf32>
    %arsums1b0W1 = "stablehlo.all_reduce"(%v4347) ({
    ^bb0(%aras1b0W1: tensor<f32>, %arbs1b0W1: tensor<f32>):
      %aradds1b0W1 = stablehlo.add %aras1b0W1, %arbs1b0W1 : tensor<f32>
      stablehlo.return %aradds1b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W1 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b0W1 = stablehlo.divide %arsums1b0W1, %arns1b0W1 : tensor<64x64x3x3xf32>
    %v4567 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4568 = stablehlo.multiply %v4567, %s1b0W1 : tensor<64x64x3x3xf32>
    %v4569 = stablehlo.add %v4568, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4570 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4571 = stablehlo.multiply %v4570, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4572 = stablehlo.add %v4571, %v4569 : tensor<64x64x3x3xf32>
    %v4573 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4574 = stablehlo.multiply %v4573, %v4572 : tensor<64x64x3x3xf32>
    %v4575 = stablehlo.subtract %s1b0W1, %v4574 : tensor<64x64x3x3xf32>
    %arsums1b0g1 = "stablehlo.all_reduce"(%v4361) ({
    ^bb0(%aras1b0g1: tensor<f32>, %arbs1b0g1: tensor<f32>):
      %aradds1b0g1 = stablehlo.add %aras1b0g1, %arbs1b0g1 : tensor<f32>
      stablehlo.return %aradds1b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g1 = stablehlo.divide %arsums1b0g1, %arns1b0g1 : tensor<64xf32>
    %v4576 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4577 = stablehlo.multiply %v4576, %s1b0g1 : tensor<64xf32>
    %v4578 = stablehlo.add %v4577, %armeans1b0g1 : tensor<64xf32>
    %v4579 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4580 = stablehlo.multiply %v4579, %s1b0g1v : tensor<64xf32>
    %v4581 = stablehlo.add %v4580, %v4578 : tensor<64xf32>
    %v4582 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4583 = stablehlo.multiply %v4582, %v4581 : tensor<64xf32>
    %v4584 = stablehlo.subtract %s1b0g1, %v4583 : tensor<64xf32>
    %arsums1b0bt1 = "stablehlo.all_reduce"(%v4364) ({
    ^bb0(%aras1b0bt1: tensor<f32>, %arbs1b0bt1: tensor<f32>):
      %aradds1b0bt1 = stablehlo.add %aras1b0bt1, %arbs1b0bt1 : tensor<f32>
      stablehlo.return %aradds1b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0bt1 = stablehlo.divide %arsums1b0bt1, %arns1b0bt1 : tensor<64xf32>
    %v4585 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4586 = stablehlo.multiply %v4585, %s1b0bt1 : tensor<64xf32>
    %v4587 = stablehlo.add %v4586, %armeans1b0bt1 : tensor<64xf32>
    %v4588 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4589 = stablehlo.multiply %v4588, %s1b0bt1v : tensor<64xf32>
    %v4590 = stablehlo.add %v4589, %v4587 : tensor<64xf32>
    %v4591 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4592 = stablehlo.multiply %v4591, %v4590 : tensor<64xf32>
    %v4593 = stablehlo.subtract %s1b0bt1, %v4592 : tensor<64xf32>
    %arsums1b0W2 = "stablehlo.all_reduce"(%v4370) ({
    ^bb0(%aras1b0W2: tensor<f32>, %arbs1b0W2: tensor<f32>):
      %aradds1b0W2 = stablehlo.add %aras1b0W2, %arbs1b0W2 : tensor<f32>
      stablehlo.return %aradds1b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W2 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b0W2 = stablehlo.divide %arsums1b0W2, %arns1b0W2 : tensor<64x64x3x3xf32>
    %v4594 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4595 = stablehlo.multiply %v4594, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4596 = stablehlo.add %v4595, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4597 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4598 = stablehlo.multiply %v4597, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4599 = stablehlo.add %v4598, %v4596 : tensor<64x64x3x3xf32>
    %v4600 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4601 = stablehlo.multiply %v4600, %v4599 : tensor<64x64x3x3xf32>
    %v4602 = stablehlo.subtract %s1b0W2, %v4601 : tensor<64x64x3x3xf32>
    %arsums1b0g2 = "stablehlo.all_reduce"(%v4384) ({
    ^bb0(%aras1b0g2: tensor<f32>, %arbs1b0g2: tensor<f32>):
      %aradds1b0g2 = stablehlo.add %aras1b0g2, %arbs1b0g2 : tensor<f32>
      stablehlo.return %aradds1b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0g2 = stablehlo.divide %arsums1b0g2, %arns1b0g2 : tensor<64xf32>
    %v4603 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4604 = stablehlo.multiply %v4603, %s1b0g2 : tensor<64xf32>
    %v4605 = stablehlo.add %v4604, %armeans1b0g2 : tensor<64xf32>
    %v4606 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4607 = stablehlo.multiply %v4606, %s1b0g2v : tensor<64xf32>
    %v4608 = stablehlo.add %v4607, %v4605 : tensor<64xf32>
    %v4609 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4610 = stablehlo.multiply %v4609, %v4608 : tensor<64xf32>
    %v4611 = stablehlo.subtract %s1b0g2, %v4610 : tensor<64xf32>
    %arsums1b0bt2 = "stablehlo.all_reduce"(%v4387) ({
    ^bb0(%aras1b0bt2: tensor<f32>, %arbs1b0bt2: tensor<f32>):
      %aradds1b0bt2 = stablehlo.add %aras1b0bt2, %arbs1b0bt2 : tensor<f32>
      stablehlo.return %aradds1b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b0bt2 = stablehlo.divide %arsums1b0bt2, %arns1b0bt2 : tensor<64xf32>
    %v4612 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4613 = stablehlo.multiply %v4612, %s1b0bt2 : tensor<64xf32>
    %v4614 = stablehlo.add %v4613, %armeans1b0bt2 : tensor<64xf32>
    %v4615 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4616 = stablehlo.multiply %v4615, %s1b0bt2v : tensor<64xf32>
    %v4617 = stablehlo.add %v4616, %v4614 : tensor<64xf32>
    %v4618 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4619 = stablehlo.multiply %v4618, %v4617 : tensor<64xf32>
    %v4620 = stablehlo.subtract %s1b0bt2, %v4619 : tensor<64xf32>
    %arsums1b1W1 = "stablehlo.all_reduce"(%v4187) ({
    ^bb0(%aras1b1W1: tensor<f32>, %arbs1b1W1: tensor<f32>):
      %aradds1b1W1 = stablehlo.add %aras1b1W1, %arbs1b1W1 : tensor<f32>
      stablehlo.return %aradds1b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W1 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b1W1 = stablehlo.divide %arsums1b1W1, %arns1b1W1 : tensor<64x64x3x3xf32>
    %v4621 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4622 = stablehlo.multiply %v4621, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4623 = stablehlo.add %v4622, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4624 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4625 = stablehlo.multiply %v4624, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4626 = stablehlo.add %v4625, %v4623 : tensor<64x64x3x3xf32>
    %v4627 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4628 = stablehlo.multiply %v4627, %v4626 : tensor<64x64x3x3xf32>
    %v4629 = stablehlo.subtract %s1b1W1, %v4628 : tensor<64x64x3x3xf32>
    %arsums1b1g1 = "stablehlo.all_reduce"(%v4201) ({
    ^bb0(%aras1b1g1: tensor<f32>, %arbs1b1g1: tensor<f32>):
      %aradds1b1g1 = stablehlo.add %aras1b1g1, %arbs1b1g1 : tensor<f32>
      stablehlo.return %aradds1b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g1 = stablehlo.divide %arsums1b1g1, %arns1b1g1 : tensor<64xf32>
    %v4630 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4631 = stablehlo.multiply %v4630, %s1b1g1 : tensor<64xf32>
    %v4632 = stablehlo.add %v4631, %armeans1b1g1 : tensor<64xf32>
    %v4633 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4634 = stablehlo.multiply %v4633, %s1b1g1v : tensor<64xf32>
    %v4635 = stablehlo.add %v4634, %v4632 : tensor<64xf32>
    %v4636 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4637 = stablehlo.multiply %v4636, %v4635 : tensor<64xf32>
    %v4638 = stablehlo.subtract %s1b1g1, %v4637 : tensor<64xf32>
    %arsums1b1bt1 = "stablehlo.all_reduce"(%v4204) ({
    ^bb0(%aras1b1bt1: tensor<f32>, %arbs1b1bt1: tensor<f32>):
      %aradds1b1bt1 = stablehlo.add %aras1b1bt1, %arbs1b1bt1 : tensor<f32>
      stablehlo.return %aradds1b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1bt1 = stablehlo.divide %arsums1b1bt1, %arns1b1bt1 : tensor<64xf32>
    %v4639 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4640 = stablehlo.multiply %v4639, %s1b1bt1 : tensor<64xf32>
    %v4641 = stablehlo.add %v4640, %armeans1b1bt1 : tensor<64xf32>
    %v4642 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4643 = stablehlo.multiply %v4642, %s1b1bt1v : tensor<64xf32>
    %v4644 = stablehlo.add %v4643, %v4641 : tensor<64xf32>
    %v4645 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4646 = stablehlo.multiply %v4645, %v4644 : tensor<64xf32>
    %v4647 = stablehlo.subtract %s1b1bt1, %v4646 : tensor<64xf32>
    %arsums1b1W2 = "stablehlo.all_reduce"(%v4210) ({
    ^bb0(%aras1b1W2: tensor<f32>, %arbs1b1W2: tensor<f32>):
      %aradds1b1W2 = stablehlo.add %aras1b1W2, %arbs1b1W2 : tensor<f32>
      stablehlo.return %aradds1b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W2 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b1W2 = stablehlo.divide %arsums1b1W2, %arns1b1W2 : tensor<64x64x3x3xf32>
    %v4648 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4649 = stablehlo.multiply %v4648, %s1b1W2 : tensor<64x64x3x3xf32>
    %v4650 = stablehlo.add %v4649, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v4651 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4652 = stablehlo.multiply %v4651, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4653 = stablehlo.add %v4652, %v4650 : tensor<64x64x3x3xf32>
    %v4654 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4655 = stablehlo.multiply %v4654, %v4653 : tensor<64x64x3x3xf32>
    %v4656 = stablehlo.subtract %s1b1W2, %v4655 : tensor<64x64x3x3xf32>
    %arsums1b1g2 = "stablehlo.all_reduce"(%v4224) ({
    ^bb0(%aras1b1g2: tensor<f32>, %arbs1b1g2: tensor<f32>):
      %aradds1b1g2 = stablehlo.add %aras1b1g2, %arbs1b1g2 : tensor<f32>
      stablehlo.return %aradds1b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1g2 = stablehlo.divide %arsums1b1g2, %arns1b1g2 : tensor<64xf32>
    %v4657 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4658 = stablehlo.multiply %v4657, %s1b1g2 : tensor<64xf32>
    %v4659 = stablehlo.add %v4658, %armeans1b1g2 : tensor<64xf32>
    %v4660 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4661 = stablehlo.multiply %v4660, %s1b1g2v : tensor<64xf32>
    %v4662 = stablehlo.add %v4661, %v4659 : tensor<64xf32>
    %v4663 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4664 = stablehlo.multiply %v4663, %v4662 : tensor<64xf32>
    %v4665 = stablehlo.subtract %s1b1g2, %v4664 : tensor<64xf32>
    %arsums1b1bt2 = "stablehlo.all_reduce"(%v4227) ({
    ^bb0(%aras1b1bt2: tensor<f32>, %arbs1b1bt2: tensor<f32>):
      %aradds1b1bt2 = stablehlo.add %aras1b1bt2, %arbs1b1bt2 : tensor<f32>
      stablehlo.return %aradds1b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b1bt2 = stablehlo.divide %arsums1b1bt2, %arns1b1bt2 : tensor<64xf32>
    %v4666 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4667 = stablehlo.multiply %v4666, %s1b1bt2 : tensor<64xf32>
    %v4668 = stablehlo.add %v4667, %armeans1b1bt2 : tensor<64xf32>
    %v4669 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4670 = stablehlo.multiply %v4669, %s1b1bt2v : tensor<64xf32>
    %v4671 = stablehlo.add %v4670, %v4668 : tensor<64xf32>
    %v4672 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4673 = stablehlo.multiply %v4672, %v4671 : tensor<64xf32>
    %v4674 = stablehlo.subtract %s1b1bt2, %v4673 : tensor<64xf32>
    %arsums1b2W1 = "stablehlo.all_reduce"(%v4027) ({
    ^bb0(%aras1b2W1: tensor<f32>, %arbs1b2W1: tensor<f32>):
      %aradds1b2W1 = stablehlo.add %aras1b2W1, %arbs1b2W1 : tensor<f32>
      stablehlo.return %aradds1b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W1 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b2W1 = stablehlo.divide %arsums1b2W1, %arns1b2W1 : tensor<64x64x3x3xf32>
    %v4675 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4676 = stablehlo.multiply %v4675, %s1b2W1 : tensor<64x64x3x3xf32>
    %v4677 = stablehlo.add %v4676, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v4678 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4679 = stablehlo.multiply %v4678, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4680 = stablehlo.add %v4679, %v4677 : tensor<64x64x3x3xf32>
    %v4681 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4682 = stablehlo.multiply %v4681, %v4680 : tensor<64x64x3x3xf32>
    %v4683 = stablehlo.subtract %s1b2W1, %v4682 : tensor<64x64x3x3xf32>
    %arsums1b2g1 = "stablehlo.all_reduce"(%v4041) ({
    ^bb0(%aras1b2g1: tensor<f32>, %arbs1b2g1: tensor<f32>):
      %aradds1b2g1 = stablehlo.add %aras1b2g1, %arbs1b2g1 : tensor<f32>
      stablehlo.return %aradds1b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g1 = stablehlo.divide %arsums1b2g1, %arns1b2g1 : tensor<64xf32>
    %v4684 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4685 = stablehlo.multiply %v4684, %s1b2g1 : tensor<64xf32>
    %v4686 = stablehlo.add %v4685, %armeans1b2g1 : tensor<64xf32>
    %v4687 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4688 = stablehlo.multiply %v4687, %s1b2g1v : tensor<64xf32>
    %v4689 = stablehlo.add %v4688, %v4686 : tensor<64xf32>
    %v4690 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4691 = stablehlo.multiply %v4690, %v4689 : tensor<64xf32>
    %v4692 = stablehlo.subtract %s1b2g1, %v4691 : tensor<64xf32>
    %arsums1b2bt1 = "stablehlo.all_reduce"(%v4044) ({
    ^bb0(%aras1b2bt1: tensor<f32>, %arbs1b2bt1: tensor<f32>):
      %aradds1b2bt1 = stablehlo.add %aras1b2bt1, %arbs1b2bt1 : tensor<f32>
      stablehlo.return %aradds1b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt1 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2bt1 = stablehlo.divide %arsums1b2bt1, %arns1b2bt1 : tensor<64xf32>
    %v4693 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4694 = stablehlo.multiply %v4693, %s1b2bt1 : tensor<64xf32>
    %v4695 = stablehlo.add %v4694, %armeans1b2bt1 : tensor<64xf32>
    %v4696 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4697 = stablehlo.multiply %v4696, %s1b2bt1v : tensor<64xf32>
    %v4698 = stablehlo.add %v4697, %v4695 : tensor<64xf32>
    %v4699 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4700 = stablehlo.multiply %v4699, %v4698 : tensor<64xf32>
    %v4701 = stablehlo.subtract %s1b2bt1, %v4700 : tensor<64xf32>
    %arsums1b2W2 = "stablehlo.all_reduce"(%v4050) ({
    ^bb0(%aras1b2W2: tensor<f32>, %arbs1b2W2: tensor<f32>):
      %aradds1b2W2 = stablehlo.add %aras1b2W2, %arbs1b2W2 : tensor<f32>
      stablehlo.return %aradds1b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W2 = stablehlo.constant dense<2.0> : tensor<64x64x3x3xf32>
    %armeans1b2W2 = stablehlo.divide %arsums1b2W2, %arns1b2W2 : tensor<64x64x3x3xf32>
    %v4702 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4703 = stablehlo.multiply %v4702, %s1b2W2 : tensor<64x64x3x3xf32>
    %v4704 = stablehlo.add %v4703, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v4705 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4706 = stablehlo.multiply %v4705, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4707 = stablehlo.add %v4706, %v4704 : tensor<64x64x3x3xf32>
    %v4708 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4709 = stablehlo.multiply %v4708, %v4707 : tensor<64x64x3x3xf32>
    %v4710 = stablehlo.subtract %s1b2W2, %v4709 : tensor<64x64x3x3xf32>
    %arsums1b2g2 = "stablehlo.all_reduce"(%v4064) ({
    ^bb0(%aras1b2g2: tensor<f32>, %arbs1b2g2: tensor<f32>):
      %aradds1b2g2 = stablehlo.add %aras1b2g2, %arbs1b2g2 : tensor<f32>
      stablehlo.return %aradds1b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2g2 = stablehlo.divide %arsums1b2g2, %arns1b2g2 : tensor<64xf32>
    %v4711 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4712 = stablehlo.multiply %v4711, %s1b2g2 : tensor<64xf32>
    %v4713 = stablehlo.add %v4712, %armeans1b2g2 : tensor<64xf32>
    %v4714 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4715 = stablehlo.multiply %v4714, %s1b2g2v : tensor<64xf32>
    %v4716 = stablehlo.add %v4715, %v4713 : tensor<64xf32>
    %v4717 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4718 = stablehlo.multiply %v4717, %v4716 : tensor<64xf32>
    %v4719 = stablehlo.subtract %s1b2g2, %v4718 : tensor<64xf32>
    %arsums1b2bt2 = "stablehlo.all_reduce"(%v4067) ({
    ^bb0(%aras1b2bt2: tensor<f32>, %arbs1b2bt2: tensor<f32>):
      %aradds1b2bt2 = stablehlo.add %aras1b2bt2, %arbs1b2bt2 : tensor<f32>
      stablehlo.return %aradds1b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt2 = stablehlo.constant dense<2.0> : tensor<64xf32>
    %armeans1b2bt2 = stablehlo.divide %arsums1b2bt2, %arns1b2bt2 : tensor<64xf32>
    %v4720 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4721 = stablehlo.multiply %v4720, %s1b2bt2 : tensor<64xf32>
    %v4722 = stablehlo.add %v4721, %armeans1b2bt2 : tensor<64xf32>
    %v4723 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4724 = stablehlo.multiply %v4723, %s1b2bt2v : tensor<64xf32>
    %v4725 = stablehlo.add %v4724, %v4722 : tensor<64xf32>
    %v4726 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4727 = stablehlo.multiply %v4726, %v4725 : tensor<64xf32>
    %v4728 = stablehlo.subtract %s1b2bt2, %v4727 : tensor<64xf32>
    %arsumd2W1 = "stablehlo.all_reduce"(%v3842) ({
    ^bb0(%arad2W1: tensor<f32>, %arbd2W1: tensor<f32>):
      %araddd2W1 = stablehlo.add %arad2W1, %arbd2W1 : tensor<f32>
      stablehlo.return %araddd2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf32>
    %arnd2W1 = stablehlo.constant dense<2.0> : tensor<128x64x3x3xf32>
    %armeand2W1 = stablehlo.divide %arsumd2W1, %arnd2W1 : tensor<128x64x3x3xf32>
    %v4729 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4730 = stablehlo.multiply %v4729, %d2W1 : tensor<128x64x3x3xf32>
    %v4731 = stablehlo.add %v4730, %armeand2W1 : tensor<128x64x3x3xf32>
    %v4732 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4733 = stablehlo.multiply %v4732, %d2W1v : tensor<128x64x3x3xf32>
    %v4734 = stablehlo.add %v4733, %v4731 : tensor<128x64x3x3xf32>
    %v4735 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4736 = stablehlo.multiply %v4735, %v4734 : tensor<128x64x3x3xf32>
    %v4737 = stablehlo.subtract %d2W1, %v4736 : tensor<128x64x3x3xf32>
    %arsumd2g1 = "stablehlo.all_reduce"(%v3856) ({
    ^bb0(%arad2g1: tensor<f32>, %arbd2g1: tensor<f32>):
      %araddd2g1 = stablehlo.add %arad2g1, %arbd2g1 : tensor<f32>
      stablehlo.return %araddd2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g1 = stablehlo.divide %arsumd2g1, %arnd2g1 : tensor<128xf32>
    %v4738 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4739 = stablehlo.multiply %v4738, %d2g1 : tensor<128xf32>
    %v4740 = stablehlo.add %v4739, %armeand2g1 : tensor<128xf32>
    %v4741 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4742 = stablehlo.multiply %v4741, %d2g1v : tensor<128xf32>
    %v4743 = stablehlo.add %v4742, %v4740 : tensor<128xf32>
    %v4744 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4745 = stablehlo.multiply %v4744, %v4743 : tensor<128xf32>
    %v4746 = stablehlo.subtract %d2g1, %v4745 : tensor<128xf32>
    %arsumd2bt1 = "stablehlo.all_reduce"(%v3859) ({
    ^bb0(%arad2bt1: tensor<f32>, %arbd2bt1: tensor<f32>):
      %araddd2bt1 = stablehlo.add %arad2bt1, %arbd2bt1 : tensor<f32>
      stablehlo.return %araddd2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2bt1 = stablehlo.divide %arsumd2bt1, %arnd2bt1 : tensor<128xf32>
    %v4747 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4748 = stablehlo.multiply %v4747, %d2bt1 : tensor<128xf32>
    %v4749 = stablehlo.add %v4748, %armeand2bt1 : tensor<128xf32>
    %v4750 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4751 = stablehlo.multiply %v4750, %d2bt1v : tensor<128xf32>
    %v4752 = stablehlo.add %v4751, %v4749 : tensor<128xf32>
    %v4753 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4754 = stablehlo.multiply %v4753, %v4752 : tensor<128xf32>
    %v4755 = stablehlo.subtract %d2bt1, %v4754 : tensor<128xf32>
    %arsumd2W2 = "stablehlo.all_reduce"(%v3865) ({
    ^bb0(%arad2W2: tensor<f32>, %arbd2W2: tensor<f32>):
      %araddd2W2 = stablehlo.add %arad2W2, %arbd2W2 : tensor<f32>
      stablehlo.return %araddd2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arnd2W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeand2W2 = stablehlo.divide %arsumd2W2, %arnd2W2 : tensor<128x128x3x3xf32>
    %v4756 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4757 = stablehlo.multiply %v4756, %d2W2 : tensor<128x128x3x3xf32>
    %v4758 = stablehlo.add %v4757, %armeand2W2 : tensor<128x128x3x3xf32>
    %v4759 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4760 = stablehlo.multiply %v4759, %d2W2v : tensor<128x128x3x3xf32>
    %v4761 = stablehlo.add %v4760, %v4758 : tensor<128x128x3x3xf32>
    %v4762 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4763 = stablehlo.multiply %v4762, %v4761 : tensor<128x128x3x3xf32>
    %v4764 = stablehlo.subtract %d2W2, %v4763 : tensor<128x128x3x3xf32>
    %arsumd2g2 = "stablehlo.all_reduce"(%v3879) ({
    ^bb0(%arad2g2: tensor<f32>, %arbd2g2: tensor<f32>):
      %araddd2g2 = stablehlo.add %arad2g2, %arbd2g2 : tensor<f32>
      stablehlo.return %araddd2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2g2 = stablehlo.divide %arsumd2g2, %arnd2g2 : tensor<128xf32>
    %v4765 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4766 = stablehlo.multiply %v4765, %d2g2 : tensor<128xf32>
    %v4767 = stablehlo.add %v4766, %armeand2g2 : tensor<128xf32>
    %v4768 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4769 = stablehlo.multiply %v4768, %d2g2v : tensor<128xf32>
    %v4770 = stablehlo.add %v4769, %v4767 : tensor<128xf32>
    %v4771 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4772 = stablehlo.multiply %v4771, %v4770 : tensor<128xf32>
    %v4773 = stablehlo.subtract %d2g2, %v4772 : tensor<128xf32>
    %arsumd2bt2 = "stablehlo.all_reduce"(%v3882) ({
    ^bb0(%arad2bt2: tensor<f32>, %arbd2bt2: tensor<f32>):
      %araddd2bt2 = stablehlo.add %arad2bt2, %arbd2bt2 : tensor<f32>
      stablehlo.return %araddd2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2bt2 = stablehlo.divide %arsumd2bt2, %arnd2bt2 : tensor<128xf32>
    %v4774 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4775 = stablehlo.multiply %v4774, %d2bt2 : tensor<128xf32>
    %v4776 = stablehlo.add %v4775, %armeand2bt2 : tensor<128xf32>
    %v4777 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4778 = stablehlo.multiply %v4777, %d2bt2v : tensor<128xf32>
    %v4779 = stablehlo.add %v4778, %v4776 : tensor<128xf32>
    %v4780 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4781 = stablehlo.multiply %v4780, %v4779 : tensor<128xf32>
    %v4782 = stablehlo.subtract %d2bt2, %v4781 : tensor<128xf32>
    %arsumd2Wp = "stablehlo.all_reduce"(%v3890) ({
    ^bb0(%arad2Wp: tensor<f32>, %arbd2Wp: tensor<f32>):
      %araddd2Wp = stablehlo.add %arad2Wp, %arbd2Wp : tensor<f32>
      stablehlo.return %araddd2Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf32>
    %arnd2Wp = stablehlo.constant dense<2.0> : tensor<128x64x1x1xf32>
    %armeand2Wp = stablehlo.divide %arsumd2Wp, %arnd2Wp : tensor<128x64x1x1xf32>
    %v4783 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4784 = stablehlo.multiply %v4783, %d2Wp : tensor<128x64x1x1xf32>
    %v4785 = stablehlo.add %v4784, %armeand2Wp : tensor<128x64x1x1xf32>
    %v4786 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4787 = stablehlo.multiply %v4786, %d2Wpv : tensor<128x64x1x1xf32>
    %v4788 = stablehlo.add %v4787, %v4785 : tensor<128x64x1x1xf32>
    %v4789 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4790 = stablehlo.multiply %v4789, %v4788 : tensor<128x64x1x1xf32>
    %v4791 = stablehlo.subtract %d2Wp, %v4790 : tensor<128x64x1x1xf32>
    %arsumd2gp = "stablehlo.all_reduce"(%v3904) ({
    ^bb0(%arad2gp: tensor<f32>, %arbd2gp: tensor<f32>):
      %araddd2gp = stablehlo.add %arad2gp, %arbd2gp : tensor<f32>
      stablehlo.return %araddd2gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2gp = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2gp = stablehlo.divide %arsumd2gp, %arnd2gp : tensor<128xf32>
    %v4792 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4793 = stablehlo.multiply %v4792, %d2gp : tensor<128xf32>
    %v4794 = stablehlo.add %v4793, %armeand2gp : tensor<128xf32>
    %v4795 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4796 = stablehlo.multiply %v4795, %d2gpv : tensor<128xf32>
    %v4797 = stablehlo.add %v4796, %v4794 : tensor<128xf32>
    %v4798 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4799 = stablehlo.multiply %v4798, %v4797 : tensor<128xf32>
    %v4800 = stablehlo.subtract %d2gp, %v4799 : tensor<128xf32>
    %arsumd2btp = "stablehlo.all_reduce"(%v3907) ({
    ^bb0(%arad2btp: tensor<f32>, %arbd2btp: tensor<f32>):
      %araddd2btp = stablehlo.add %arad2btp, %arbd2btp : tensor<f32>
      stablehlo.return %araddd2btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2btp = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeand2btp = stablehlo.divide %arsumd2btp, %arnd2btp : tensor<128xf32>
    %v4801 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4802 = stablehlo.multiply %v4801, %d2btp : tensor<128xf32>
    %v4803 = stablehlo.add %v4802, %armeand2btp : tensor<128xf32>
    %v4804 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4805 = stablehlo.multiply %v4804, %d2btpv : tensor<128xf32>
    %v4806 = stablehlo.add %v4805, %v4803 : tensor<128xf32>
    %v4807 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4808 = stablehlo.multiply %v4807, %v4806 : tensor<128xf32>
    %v4809 = stablehlo.subtract %d2btp, %v4808 : tensor<128xf32>
    %arsums2b0W1 = "stablehlo.all_reduce"(%v3627) ({
    ^bb0(%aras2b0W1: tensor<f32>, %arbs2b0W1: tensor<f32>):
      %aradds2b0W1 = stablehlo.add %aras2b0W1, %arbs2b0W1 : tensor<f32>
      stablehlo.return %aradds2b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W1 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b0W1 = stablehlo.divide %arsums2b0W1, %arns2b0W1 : tensor<128x128x3x3xf32>
    %v4810 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4811 = stablehlo.multiply %v4810, %s2b0W1 : tensor<128x128x3x3xf32>
    %v4812 = stablehlo.add %v4811, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v4813 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4814 = stablehlo.multiply %v4813, %s2b0W1v : tensor<128x128x3x3xf32>
    %v4815 = stablehlo.add %v4814, %v4812 : tensor<128x128x3x3xf32>
    %v4816 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4817 = stablehlo.multiply %v4816, %v4815 : tensor<128x128x3x3xf32>
    %v4818 = stablehlo.subtract %s2b0W1, %v4817 : tensor<128x128x3x3xf32>
    %arsums2b0g1 = "stablehlo.all_reduce"(%v3641) ({
    ^bb0(%aras2b0g1: tensor<f32>, %arbs2b0g1: tensor<f32>):
      %aradds2b0g1 = stablehlo.add %aras2b0g1, %arbs2b0g1 : tensor<f32>
      stablehlo.return %aradds2b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g1 = stablehlo.divide %arsums2b0g1, %arns2b0g1 : tensor<128xf32>
    %v4819 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4820 = stablehlo.multiply %v4819, %s2b0g1 : tensor<128xf32>
    %v4821 = stablehlo.add %v4820, %armeans2b0g1 : tensor<128xf32>
    %v4822 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4823 = stablehlo.multiply %v4822, %s2b0g1v : tensor<128xf32>
    %v4824 = stablehlo.add %v4823, %v4821 : tensor<128xf32>
    %v4825 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4826 = stablehlo.multiply %v4825, %v4824 : tensor<128xf32>
    %v4827 = stablehlo.subtract %s2b0g1, %v4826 : tensor<128xf32>
    %arsums2b0bt1 = "stablehlo.all_reduce"(%v3644) ({
    ^bb0(%aras2b0bt1: tensor<f32>, %arbs2b0bt1: tensor<f32>):
      %aradds2b0bt1 = stablehlo.add %aras2b0bt1, %arbs2b0bt1 : tensor<f32>
      stablehlo.return %aradds2b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0bt1 = stablehlo.divide %arsums2b0bt1, %arns2b0bt1 : tensor<128xf32>
    %v4828 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4829 = stablehlo.multiply %v4828, %s2b0bt1 : tensor<128xf32>
    %v4830 = stablehlo.add %v4829, %armeans2b0bt1 : tensor<128xf32>
    %v4831 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4832 = stablehlo.multiply %v4831, %s2b0bt1v : tensor<128xf32>
    %v4833 = stablehlo.add %v4832, %v4830 : tensor<128xf32>
    %v4834 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4835 = stablehlo.multiply %v4834, %v4833 : tensor<128xf32>
    %v4836 = stablehlo.subtract %s2b0bt1, %v4835 : tensor<128xf32>
    %arsums2b0W2 = "stablehlo.all_reduce"(%v3650) ({
    ^bb0(%aras2b0W2: tensor<f32>, %arbs2b0W2: tensor<f32>):
      %aradds2b0W2 = stablehlo.add %aras2b0W2, %arbs2b0W2 : tensor<f32>
      stablehlo.return %aradds2b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b0W2 = stablehlo.divide %arsums2b0W2, %arns2b0W2 : tensor<128x128x3x3xf32>
    %v4837 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4838 = stablehlo.multiply %v4837, %s2b0W2 : tensor<128x128x3x3xf32>
    %v4839 = stablehlo.add %v4838, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v4840 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4841 = stablehlo.multiply %v4840, %s2b0W2v : tensor<128x128x3x3xf32>
    %v4842 = stablehlo.add %v4841, %v4839 : tensor<128x128x3x3xf32>
    %v4843 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4844 = stablehlo.multiply %v4843, %v4842 : tensor<128x128x3x3xf32>
    %v4845 = stablehlo.subtract %s2b0W2, %v4844 : tensor<128x128x3x3xf32>
    %arsums2b0g2 = "stablehlo.all_reduce"(%v3664) ({
    ^bb0(%aras2b0g2: tensor<f32>, %arbs2b0g2: tensor<f32>):
      %aradds2b0g2 = stablehlo.add %aras2b0g2, %arbs2b0g2 : tensor<f32>
      stablehlo.return %aradds2b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0g2 = stablehlo.divide %arsums2b0g2, %arns2b0g2 : tensor<128xf32>
    %v4846 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4847 = stablehlo.multiply %v4846, %s2b0g2 : tensor<128xf32>
    %v4848 = stablehlo.add %v4847, %armeans2b0g2 : tensor<128xf32>
    %v4849 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4850 = stablehlo.multiply %v4849, %s2b0g2v : tensor<128xf32>
    %v4851 = stablehlo.add %v4850, %v4848 : tensor<128xf32>
    %v4852 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4853 = stablehlo.multiply %v4852, %v4851 : tensor<128xf32>
    %v4854 = stablehlo.subtract %s2b0g2, %v4853 : tensor<128xf32>
    %arsums2b0bt2 = "stablehlo.all_reduce"(%v3667) ({
    ^bb0(%aras2b0bt2: tensor<f32>, %arbs2b0bt2: tensor<f32>):
      %aradds2b0bt2 = stablehlo.add %aras2b0bt2, %arbs2b0bt2 : tensor<f32>
      stablehlo.return %aradds2b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b0bt2 = stablehlo.divide %arsums2b0bt2, %arns2b0bt2 : tensor<128xf32>
    %v4855 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4856 = stablehlo.multiply %v4855, %s2b0bt2 : tensor<128xf32>
    %v4857 = stablehlo.add %v4856, %armeans2b0bt2 : tensor<128xf32>
    %v4858 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4859 = stablehlo.multiply %v4858, %s2b0bt2v : tensor<128xf32>
    %v4860 = stablehlo.add %v4859, %v4857 : tensor<128xf32>
    %v4861 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4862 = stablehlo.multiply %v4861, %v4860 : tensor<128xf32>
    %v4863 = stablehlo.subtract %s2b0bt2, %v4862 : tensor<128xf32>
    %arsums2b1W1 = "stablehlo.all_reduce"(%v3467) ({
    ^bb0(%aras2b1W1: tensor<f32>, %arbs2b1W1: tensor<f32>):
      %aradds2b1W1 = stablehlo.add %aras2b1W1, %arbs2b1W1 : tensor<f32>
      stablehlo.return %aradds2b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W1 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b1W1 = stablehlo.divide %arsums2b1W1, %arns2b1W1 : tensor<128x128x3x3xf32>
    %v4864 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4865 = stablehlo.multiply %v4864, %s2b1W1 : tensor<128x128x3x3xf32>
    %v4866 = stablehlo.add %v4865, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v4867 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4868 = stablehlo.multiply %v4867, %s2b1W1v : tensor<128x128x3x3xf32>
    %v4869 = stablehlo.add %v4868, %v4866 : tensor<128x128x3x3xf32>
    %v4870 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4871 = stablehlo.multiply %v4870, %v4869 : tensor<128x128x3x3xf32>
    %v4872 = stablehlo.subtract %s2b1W1, %v4871 : tensor<128x128x3x3xf32>
    %arsums2b1g1 = "stablehlo.all_reduce"(%v3481) ({
    ^bb0(%aras2b1g1: tensor<f32>, %arbs2b1g1: tensor<f32>):
      %aradds2b1g1 = stablehlo.add %aras2b1g1, %arbs2b1g1 : tensor<f32>
      stablehlo.return %aradds2b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g1 = stablehlo.divide %arsums2b1g1, %arns2b1g1 : tensor<128xf32>
    %v4873 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4874 = stablehlo.multiply %v4873, %s2b1g1 : tensor<128xf32>
    %v4875 = stablehlo.add %v4874, %armeans2b1g1 : tensor<128xf32>
    %v4876 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4877 = stablehlo.multiply %v4876, %s2b1g1v : tensor<128xf32>
    %v4878 = stablehlo.add %v4877, %v4875 : tensor<128xf32>
    %v4879 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4880 = stablehlo.multiply %v4879, %v4878 : tensor<128xf32>
    %v4881 = stablehlo.subtract %s2b1g1, %v4880 : tensor<128xf32>
    %arsums2b1bt1 = "stablehlo.all_reduce"(%v3484) ({
    ^bb0(%aras2b1bt1: tensor<f32>, %arbs2b1bt1: tensor<f32>):
      %aradds2b1bt1 = stablehlo.add %aras2b1bt1, %arbs2b1bt1 : tensor<f32>
      stablehlo.return %aradds2b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1bt1 = stablehlo.divide %arsums2b1bt1, %arns2b1bt1 : tensor<128xf32>
    %v4882 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4883 = stablehlo.multiply %v4882, %s2b1bt1 : tensor<128xf32>
    %v4884 = stablehlo.add %v4883, %armeans2b1bt1 : tensor<128xf32>
    %v4885 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4886 = stablehlo.multiply %v4885, %s2b1bt1v : tensor<128xf32>
    %v4887 = stablehlo.add %v4886, %v4884 : tensor<128xf32>
    %v4888 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4889 = stablehlo.multiply %v4888, %v4887 : tensor<128xf32>
    %v4890 = stablehlo.subtract %s2b1bt1, %v4889 : tensor<128xf32>
    %arsums2b1W2 = "stablehlo.all_reduce"(%v3490) ({
    ^bb0(%aras2b1W2: tensor<f32>, %arbs2b1W2: tensor<f32>):
      %aradds2b1W2 = stablehlo.add %aras2b1W2, %arbs2b1W2 : tensor<f32>
      stablehlo.return %aradds2b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b1W2 = stablehlo.divide %arsums2b1W2, %arns2b1W2 : tensor<128x128x3x3xf32>
    %v4891 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4892 = stablehlo.multiply %v4891, %s2b1W2 : tensor<128x128x3x3xf32>
    %v4893 = stablehlo.add %v4892, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v4894 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4895 = stablehlo.multiply %v4894, %s2b1W2v : tensor<128x128x3x3xf32>
    %v4896 = stablehlo.add %v4895, %v4893 : tensor<128x128x3x3xf32>
    %v4897 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4898 = stablehlo.multiply %v4897, %v4896 : tensor<128x128x3x3xf32>
    %v4899 = stablehlo.subtract %s2b1W2, %v4898 : tensor<128x128x3x3xf32>
    %arsums2b1g2 = "stablehlo.all_reduce"(%v3504) ({
    ^bb0(%aras2b1g2: tensor<f32>, %arbs2b1g2: tensor<f32>):
      %aradds2b1g2 = stablehlo.add %aras2b1g2, %arbs2b1g2 : tensor<f32>
      stablehlo.return %aradds2b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1g2 = stablehlo.divide %arsums2b1g2, %arns2b1g2 : tensor<128xf32>
    %v4900 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4901 = stablehlo.multiply %v4900, %s2b1g2 : tensor<128xf32>
    %v4902 = stablehlo.add %v4901, %armeans2b1g2 : tensor<128xf32>
    %v4903 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4904 = stablehlo.multiply %v4903, %s2b1g2v : tensor<128xf32>
    %v4905 = stablehlo.add %v4904, %v4902 : tensor<128xf32>
    %v4906 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4907 = stablehlo.multiply %v4906, %v4905 : tensor<128xf32>
    %v4908 = stablehlo.subtract %s2b1g2, %v4907 : tensor<128xf32>
    %arsums2b1bt2 = "stablehlo.all_reduce"(%v3507) ({
    ^bb0(%aras2b1bt2: tensor<f32>, %arbs2b1bt2: tensor<f32>):
      %aradds2b1bt2 = stablehlo.add %aras2b1bt2, %arbs2b1bt2 : tensor<f32>
      stablehlo.return %aradds2b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b1bt2 = stablehlo.divide %arsums2b1bt2, %arns2b1bt2 : tensor<128xf32>
    %v4909 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4910 = stablehlo.multiply %v4909, %s2b1bt2 : tensor<128xf32>
    %v4911 = stablehlo.add %v4910, %armeans2b1bt2 : tensor<128xf32>
    %v4912 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4913 = stablehlo.multiply %v4912, %s2b1bt2v : tensor<128xf32>
    %v4914 = stablehlo.add %v4913, %v4911 : tensor<128xf32>
    %v4915 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4916 = stablehlo.multiply %v4915, %v4914 : tensor<128xf32>
    %v4917 = stablehlo.subtract %s2b1bt2, %v4916 : tensor<128xf32>
    %arsums2b2W1 = "stablehlo.all_reduce"(%v3307) ({
    ^bb0(%aras2b2W1: tensor<f32>, %arbs2b2W1: tensor<f32>):
      %aradds2b2W1 = stablehlo.add %aras2b2W1, %arbs2b2W1 : tensor<f32>
      stablehlo.return %aradds2b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W1 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b2W1 = stablehlo.divide %arsums2b2W1, %arns2b2W1 : tensor<128x128x3x3xf32>
    %v4918 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4919 = stablehlo.multiply %v4918, %s2b2W1 : tensor<128x128x3x3xf32>
    %v4920 = stablehlo.add %v4919, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v4921 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4922 = stablehlo.multiply %v4921, %s2b2W1v : tensor<128x128x3x3xf32>
    %v4923 = stablehlo.add %v4922, %v4920 : tensor<128x128x3x3xf32>
    %v4924 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4925 = stablehlo.multiply %v4924, %v4923 : tensor<128x128x3x3xf32>
    %v4926 = stablehlo.subtract %s2b2W1, %v4925 : tensor<128x128x3x3xf32>
    %arsums2b2g1 = "stablehlo.all_reduce"(%v3321) ({
    ^bb0(%aras2b2g1: tensor<f32>, %arbs2b2g1: tensor<f32>):
      %aradds2b2g1 = stablehlo.add %aras2b2g1, %arbs2b2g1 : tensor<f32>
      stablehlo.return %aradds2b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g1 = stablehlo.divide %arsums2b2g1, %arns2b2g1 : tensor<128xf32>
    %v4927 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4928 = stablehlo.multiply %v4927, %s2b2g1 : tensor<128xf32>
    %v4929 = stablehlo.add %v4928, %armeans2b2g1 : tensor<128xf32>
    %v4930 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4931 = stablehlo.multiply %v4930, %s2b2g1v : tensor<128xf32>
    %v4932 = stablehlo.add %v4931, %v4929 : tensor<128xf32>
    %v4933 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4934 = stablehlo.multiply %v4933, %v4932 : tensor<128xf32>
    %v4935 = stablehlo.subtract %s2b2g1, %v4934 : tensor<128xf32>
    %arsums2b2bt1 = "stablehlo.all_reduce"(%v3324) ({
    ^bb0(%aras2b2bt1: tensor<f32>, %arbs2b2bt1: tensor<f32>):
      %aradds2b2bt1 = stablehlo.add %aras2b2bt1, %arbs2b2bt1 : tensor<f32>
      stablehlo.return %aradds2b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt1 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2bt1 = stablehlo.divide %arsums2b2bt1, %arns2b2bt1 : tensor<128xf32>
    %v4936 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4937 = stablehlo.multiply %v4936, %s2b2bt1 : tensor<128xf32>
    %v4938 = stablehlo.add %v4937, %armeans2b2bt1 : tensor<128xf32>
    %v4939 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4940 = stablehlo.multiply %v4939, %s2b2bt1v : tensor<128xf32>
    %v4941 = stablehlo.add %v4940, %v4938 : tensor<128xf32>
    %v4942 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4943 = stablehlo.multiply %v4942, %v4941 : tensor<128xf32>
    %v4944 = stablehlo.subtract %s2b2bt1, %v4943 : tensor<128xf32>
    %arsums2b2W2 = "stablehlo.all_reduce"(%v3330) ({
    ^bb0(%aras2b2W2: tensor<f32>, %arbs2b2W2: tensor<f32>):
      %aradds2b2W2 = stablehlo.add %aras2b2W2, %arbs2b2W2 : tensor<f32>
      stablehlo.return %aradds2b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W2 = stablehlo.constant dense<2.0> : tensor<128x128x3x3xf32>
    %armeans2b2W2 = stablehlo.divide %arsums2b2W2, %arns2b2W2 : tensor<128x128x3x3xf32>
    %v4945 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4946 = stablehlo.multiply %v4945, %s2b2W2 : tensor<128x128x3x3xf32>
    %v4947 = stablehlo.add %v4946, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v4948 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4949 = stablehlo.multiply %v4948, %s2b2W2v : tensor<128x128x3x3xf32>
    %v4950 = stablehlo.add %v4949, %v4947 : tensor<128x128x3x3xf32>
    %v4951 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4952 = stablehlo.multiply %v4951, %v4950 : tensor<128x128x3x3xf32>
    %v4953 = stablehlo.subtract %s2b2W2, %v4952 : tensor<128x128x3x3xf32>
    %arsums2b2g2 = "stablehlo.all_reduce"(%v3344) ({
    ^bb0(%aras2b2g2: tensor<f32>, %arbs2b2g2: tensor<f32>):
      %aradds2b2g2 = stablehlo.add %aras2b2g2, %arbs2b2g2 : tensor<f32>
      stablehlo.return %aradds2b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2g2 = stablehlo.divide %arsums2b2g2, %arns2b2g2 : tensor<128xf32>
    %v4954 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4955 = stablehlo.multiply %v4954, %s2b2g2 : tensor<128xf32>
    %v4956 = stablehlo.add %v4955, %armeans2b2g2 : tensor<128xf32>
    %v4957 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4958 = stablehlo.multiply %v4957, %s2b2g2v : tensor<128xf32>
    %v4959 = stablehlo.add %v4958, %v4956 : tensor<128xf32>
    %v4960 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4961 = stablehlo.multiply %v4960, %v4959 : tensor<128xf32>
    %v4962 = stablehlo.subtract %s2b2g2, %v4961 : tensor<128xf32>
    %arsums2b2bt2 = "stablehlo.all_reduce"(%v3347) ({
    ^bb0(%aras2b2bt2: tensor<f32>, %arbs2b2bt2: tensor<f32>):
      %aradds2b2bt2 = stablehlo.add %aras2b2bt2, %arbs2b2bt2 : tensor<f32>
      stablehlo.return %aradds2b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt2 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %armeans2b2bt2 = stablehlo.divide %arsums2b2bt2, %arns2b2bt2 : tensor<128xf32>
    %v4963 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4964 = stablehlo.multiply %v4963, %s2b2bt2 : tensor<128xf32>
    %v4965 = stablehlo.add %v4964, %armeans2b2bt2 : tensor<128xf32>
    %v4966 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4967 = stablehlo.multiply %v4966, %s2b2bt2v : tensor<128xf32>
    %v4968 = stablehlo.add %v4967, %v4965 : tensor<128xf32>
    %v4969 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4970 = stablehlo.multiply %v4969, %v4968 : tensor<128xf32>
    %v4971 = stablehlo.subtract %s2b2bt2, %v4970 : tensor<128xf32>
    %arsumd3W1 = "stablehlo.all_reduce"(%v3122) ({
    ^bb0(%arad3W1: tensor<f32>, %arbd3W1: tensor<f32>):
      %araddd3W1 = stablehlo.add %arad3W1, %arbd3W1 : tensor<f32>
      stablehlo.return %araddd3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf32>
    %arnd3W1 = stablehlo.constant dense<2.0> : tensor<256x128x3x3xf32>
    %armeand3W1 = stablehlo.divide %arsumd3W1, %arnd3W1 : tensor<256x128x3x3xf32>
    %v4972 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4973 = stablehlo.multiply %v4972, %d3W1 : tensor<256x128x3x3xf32>
    %v4974 = stablehlo.add %v4973, %armeand3W1 : tensor<256x128x3x3xf32>
    %v4975 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4976 = stablehlo.multiply %v4975, %d3W1v : tensor<256x128x3x3xf32>
    %v4977 = stablehlo.add %v4976, %v4974 : tensor<256x128x3x3xf32>
    %v4978 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4979 = stablehlo.multiply %v4978, %v4977 : tensor<256x128x3x3xf32>
    %v4980 = stablehlo.subtract %d3W1, %v4979 : tensor<256x128x3x3xf32>
    %arsumd3g1 = "stablehlo.all_reduce"(%v3136) ({
    ^bb0(%arad3g1: tensor<f32>, %arbd3g1: tensor<f32>):
      %araddd3g1 = stablehlo.add %arad3g1, %arbd3g1 : tensor<f32>
      stablehlo.return %araddd3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g1 = stablehlo.divide %arsumd3g1, %arnd3g1 : tensor<256xf32>
    %v4981 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4982 = stablehlo.multiply %v4981, %d3g1 : tensor<256xf32>
    %v4983 = stablehlo.add %v4982, %armeand3g1 : tensor<256xf32>
    %v4984 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4985 = stablehlo.multiply %v4984, %d3g1v : tensor<256xf32>
    %v4986 = stablehlo.add %v4985, %v4983 : tensor<256xf32>
    %v4987 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4988 = stablehlo.multiply %v4987, %v4986 : tensor<256xf32>
    %v4989 = stablehlo.subtract %d3g1, %v4988 : tensor<256xf32>
    %arsumd3bt1 = "stablehlo.all_reduce"(%v3139) ({
    ^bb0(%arad3bt1: tensor<f32>, %arbd3bt1: tensor<f32>):
      %araddd3bt1 = stablehlo.add %arad3bt1, %arbd3bt1 : tensor<f32>
      stablehlo.return %araddd3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3bt1 = stablehlo.divide %arsumd3bt1, %arnd3bt1 : tensor<256xf32>
    %v4990 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4991 = stablehlo.multiply %v4990, %d3bt1 : tensor<256xf32>
    %v4992 = stablehlo.add %v4991, %armeand3bt1 : tensor<256xf32>
    %v4993 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4994 = stablehlo.multiply %v4993, %d3bt1v : tensor<256xf32>
    %v4995 = stablehlo.add %v4994, %v4992 : tensor<256xf32>
    %v4996 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4997 = stablehlo.multiply %v4996, %v4995 : tensor<256xf32>
    %v4998 = stablehlo.subtract %d3bt1, %v4997 : tensor<256xf32>
    %arsumd3W2 = "stablehlo.all_reduce"(%v3145) ({
    ^bb0(%arad3W2: tensor<f32>, %arbd3W2: tensor<f32>):
      %araddd3W2 = stablehlo.add %arad3W2, %arbd3W2 : tensor<f32>
      stablehlo.return %araddd3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arnd3W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeand3W2 = stablehlo.divide %arsumd3W2, %arnd3W2 : tensor<256x256x3x3xf32>
    %v4999 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5000 = stablehlo.multiply %v4999, %d3W2 : tensor<256x256x3x3xf32>
    %v5001 = stablehlo.add %v5000, %armeand3W2 : tensor<256x256x3x3xf32>
    %v5002 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5003 = stablehlo.multiply %v5002, %d3W2v : tensor<256x256x3x3xf32>
    %v5004 = stablehlo.add %v5003, %v5001 : tensor<256x256x3x3xf32>
    %v5005 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5006 = stablehlo.multiply %v5005, %v5004 : tensor<256x256x3x3xf32>
    %v5007 = stablehlo.subtract %d3W2, %v5006 : tensor<256x256x3x3xf32>
    %arsumd3g2 = "stablehlo.all_reduce"(%v3159) ({
    ^bb0(%arad3g2: tensor<f32>, %arbd3g2: tensor<f32>):
      %araddd3g2 = stablehlo.add %arad3g2, %arbd3g2 : tensor<f32>
      stablehlo.return %araddd3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3g2 = stablehlo.divide %arsumd3g2, %arnd3g2 : tensor<256xf32>
    %v5008 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5009 = stablehlo.multiply %v5008, %d3g2 : tensor<256xf32>
    %v5010 = stablehlo.add %v5009, %armeand3g2 : tensor<256xf32>
    %v5011 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5012 = stablehlo.multiply %v5011, %d3g2v : tensor<256xf32>
    %v5013 = stablehlo.add %v5012, %v5010 : tensor<256xf32>
    %v5014 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5015 = stablehlo.multiply %v5014, %v5013 : tensor<256xf32>
    %v5016 = stablehlo.subtract %d3g2, %v5015 : tensor<256xf32>
    %arsumd3bt2 = "stablehlo.all_reduce"(%v3162) ({
    ^bb0(%arad3bt2: tensor<f32>, %arbd3bt2: tensor<f32>):
      %araddd3bt2 = stablehlo.add %arad3bt2, %arbd3bt2 : tensor<f32>
      stablehlo.return %araddd3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3bt2 = stablehlo.divide %arsumd3bt2, %arnd3bt2 : tensor<256xf32>
    %v5017 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5018 = stablehlo.multiply %v5017, %d3bt2 : tensor<256xf32>
    %v5019 = stablehlo.add %v5018, %armeand3bt2 : tensor<256xf32>
    %v5020 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5021 = stablehlo.multiply %v5020, %d3bt2v : tensor<256xf32>
    %v5022 = stablehlo.add %v5021, %v5019 : tensor<256xf32>
    %v5023 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5024 = stablehlo.multiply %v5023, %v5022 : tensor<256xf32>
    %v5025 = stablehlo.subtract %d3bt2, %v5024 : tensor<256xf32>
    %arsumd3Wp = "stablehlo.all_reduce"(%v3170) ({
    ^bb0(%arad3Wp: tensor<f32>, %arbd3Wp: tensor<f32>):
      %araddd3Wp = stablehlo.add %arad3Wp, %arbd3Wp : tensor<f32>
      stablehlo.return %araddd3Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf32>
    %arnd3Wp = stablehlo.constant dense<2.0> : tensor<256x128x1x1xf32>
    %armeand3Wp = stablehlo.divide %arsumd3Wp, %arnd3Wp : tensor<256x128x1x1xf32>
    %v5026 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5027 = stablehlo.multiply %v5026, %d3Wp : tensor<256x128x1x1xf32>
    %v5028 = stablehlo.add %v5027, %armeand3Wp : tensor<256x128x1x1xf32>
    %v5029 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5030 = stablehlo.multiply %v5029, %d3Wpv : tensor<256x128x1x1xf32>
    %v5031 = stablehlo.add %v5030, %v5028 : tensor<256x128x1x1xf32>
    %v5032 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5033 = stablehlo.multiply %v5032, %v5031 : tensor<256x128x1x1xf32>
    %v5034 = stablehlo.subtract %d3Wp, %v5033 : tensor<256x128x1x1xf32>
    %arsumd3gp = "stablehlo.all_reduce"(%v3184) ({
    ^bb0(%arad3gp: tensor<f32>, %arbd3gp: tensor<f32>):
      %araddd3gp = stablehlo.add %arad3gp, %arbd3gp : tensor<f32>
      stablehlo.return %araddd3gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3gp = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3gp = stablehlo.divide %arsumd3gp, %arnd3gp : tensor<256xf32>
    %v5035 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5036 = stablehlo.multiply %v5035, %d3gp : tensor<256xf32>
    %v5037 = stablehlo.add %v5036, %armeand3gp : tensor<256xf32>
    %v5038 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5039 = stablehlo.multiply %v5038, %d3gpv : tensor<256xf32>
    %v5040 = stablehlo.add %v5039, %v5037 : tensor<256xf32>
    %v5041 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5042 = stablehlo.multiply %v5041, %v5040 : tensor<256xf32>
    %v5043 = stablehlo.subtract %d3gp, %v5042 : tensor<256xf32>
    %arsumd3btp = "stablehlo.all_reduce"(%v3187) ({
    ^bb0(%arad3btp: tensor<f32>, %arbd3btp: tensor<f32>):
      %araddd3btp = stablehlo.add %arad3btp, %arbd3btp : tensor<f32>
      stablehlo.return %araddd3btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3btp = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeand3btp = stablehlo.divide %arsumd3btp, %arnd3btp : tensor<256xf32>
    %v5044 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5045 = stablehlo.multiply %v5044, %d3btp : tensor<256xf32>
    %v5046 = stablehlo.add %v5045, %armeand3btp : tensor<256xf32>
    %v5047 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5048 = stablehlo.multiply %v5047, %d3btpv : tensor<256xf32>
    %v5049 = stablehlo.add %v5048, %v5046 : tensor<256xf32>
    %v5050 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5051 = stablehlo.multiply %v5050, %v5049 : tensor<256xf32>
    %v5052 = stablehlo.subtract %d3btp, %v5051 : tensor<256xf32>
    %arsums3b0W1 = "stablehlo.all_reduce"(%v2907) ({
    ^bb0(%aras3b0W1: tensor<f32>, %arbs3b0W1: tensor<f32>):
      %aradds3b0W1 = stablehlo.add %aras3b0W1, %arbs3b0W1 : tensor<f32>
      stablehlo.return %aradds3b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b0W1 = stablehlo.divide %arsums3b0W1, %arns3b0W1 : tensor<256x256x3x3xf32>
    %v5053 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5054 = stablehlo.multiply %v5053, %s3b0W1 : tensor<256x256x3x3xf32>
    %v5055 = stablehlo.add %v5054, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v5056 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5057 = stablehlo.multiply %v5056, %s3b0W1v : tensor<256x256x3x3xf32>
    %v5058 = stablehlo.add %v5057, %v5055 : tensor<256x256x3x3xf32>
    %v5059 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5060 = stablehlo.multiply %v5059, %v5058 : tensor<256x256x3x3xf32>
    %v5061 = stablehlo.subtract %s3b0W1, %v5060 : tensor<256x256x3x3xf32>
    %arsums3b0g1 = "stablehlo.all_reduce"(%v2921) ({
    ^bb0(%aras3b0g1: tensor<f32>, %arbs3b0g1: tensor<f32>):
      %aradds3b0g1 = stablehlo.add %aras3b0g1, %arbs3b0g1 : tensor<f32>
      stablehlo.return %aradds3b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g1 = stablehlo.divide %arsums3b0g1, %arns3b0g1 : tensor<256xf32>
    %v5062 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5063 = stablehlo.multiply %v5062, %s3b0g1 : tensor<256xf32>
    %v5064 = stablehlo.add %v5063, %armeans3b0g1 : tensor<256xf32>
    %v5065 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5066 = stablehlo.multiply %v5065, %s3b0g1v : tensor<256xf32>
    %v5067 = stablehlo.add %v5066, %v5064 : tensor<256xf32>
    %v5068 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5069 = stablehlo.multiply %v5068, %v5067 : tensor<256xf32>
    %v5070 = stablehlo.subtract %s3b0g1, %v5069 : tensor<256xf32>
    %arsums3b0bt1 = "stablehlo.all_reduce"(%v2924) ({
    ^bb0(%aras3b0bt1: tensor<f32>, %arbs3b0bt1: tensor<f32>):
      %aradds3b0bt1 = stablehlo.add %aras3b0bt1, %arbs3b0bt1 : tensor<f32>
      stablehlo.return %aradds3b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0bt1 = stablehlo.divide %arsums3b0bt1, %arns3b0bt1 : tensor<256xf32>
    %v5071 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5072 = stablehlo.multiply %v5071, %s3b0bt1 : tensor<256xf32>
    %v5073 = stablehlo.add %v5072, %armeans3b0bt1 : tensor<256xf32>
    %v5074 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5075 = stablehlo.multiply %v5074, %s3b0bt1v : tensor<256xf32>
    %v5076 = stablehlo.add %v5075, %v5073 : tensor<256xf32>
    %v5077 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5078 = stablehlo.multiply %v5077, %v5076 : tensor<256xf32>
    %v5079 = stablehlo.subtract %s3b0bt1, %v5078 : tensor<256xf32>
    %arsums3b0W2 = "stablehlo.all_reduce"(%v2930) ({
    ^bb0(%aras3b0W2: tensor<f32>, %arbs3b0W2: tensor<f32>):
      %aradds3b0W2 = stablehlo.add %aras3b0W2, %arbs3b0W2 : tensor<f32>
      stablehlo.return %aradds3b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b0W2 = stablehlo.divide %arsums3b0W2, %arns3b0W2 : tensor<256x256x3x3xf32>
    %v5080 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5081 = stablehlo.multiply %v5080, %s3b0W2 : tensor<256x256x3x3xf32>
    %v5082 = stablehlo.add %v5081, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v5083 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5084 = stablehlo.multiply %v5083, %s3b0W2v : tensor<256x256x3x3xf32>
    %v5085 = stablehlo.add %v5084, %v5082 : tensor<256x256x3x3xf32>
    %v5086 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5087 = stablehlo.multiply %v5086, %v5085 : tensor<256x256x3x3xf32>
    %v5088 = stablehlo.subtract %s3b0W2, %v5087 : tensor<256x256x3x3xf32>
    %arsums3b0g2 = "stablehlo.all_reduce"(%v2944) ({
    ^bb0(%aras3b0g2: tensor<f32>, %arbs3b0g2: tensor<f32>):
      %aradds3b0g2 = stablehlo.add %aras3b0g2, %arbs3b0g2 : tensor<f32>
      stablehlo.return %aradds3b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0g2 = stablehlo.divide %arsums3b0g2, %arns3b0g2 : tensor<256xf32>
    %v5089 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5090 = stablehlo.multiply %v5089, %s3b0g2 : tensor<256xf32>
    %v5091 = stablehlo.add %v5090, %armeans3b0g2 : tensor<256xf32>
    %v5092 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5093 = stablehlo.multiply %v5092, %s3b0g2v : tensor<256xf32>
    %v5094 = stablehlo.add %v5093, %v5091 : tensor<256xf32>
    %v5095 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5096 = stablehlo.multiply %v5095, %v5094 : tensor<256xf32>
    %v5097 = stablehlo.subtract %s3b0g2, %v5096 : tensor<256xf32>
    %arsums3b0bt2 = "stablehlo.all_reduce"(%v2947) ({
    ^bb0(%aras3b0bt2: tensor<f32>, %arbs3b0bt2: tensor<f32>):
      %aradds3b0bt2 = stablehlo.add %aras3b0bt2, %arbs3b0bt2 : tensor<f32>
      stablehlo.return %aradds3b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b0bt2 = stablehlo.divide %arsums3b0bt2, %arns3b0bt2 : tensor<256xf32>
    %v5098 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5099 = stablehlo.multiply %v5098, %s3b0bt2 : tensor<256xf32>
    %v5100 = stablehlo.add %v5099, %armeans3b0bt2 : tensor<256xf32>
    %v5101 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5102 = stablehlo.multiply %v5101, %s3b0bt2v : tensor<256xf32>
    %v5103 = stablehlo.add %v5102, %v5100 : tensor<256xf32>
    %v5104 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5105 = stablehlo.multiply %v5104, %v5103 : tensor<256xf32>
    %v5106 = stablehlo.subtract %s3b0bt2, %v5105 : tensor<256xf32>
    %arsums3b1W1 = "stablehlo.all_reduce"(%v2747) ({
    ^bb0(%aras3b1W1: tensor<f32>, %arbs3b1W1: tensor<f32>):
      %aradds3b1W1 = stablehlo.add %aras3b1W1, %arbs3b1W1 : tensor<f32>
      stablehlo.return %aradds3b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b1W1 = stablehlo.divide %arsums3b1W1, %arns3b1W1 : tensor<256x256x3x3xf32>
    %v5107 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5108 = stablehlo.multiply %v5107, %s3b1W1 : tensor<256x256x3x3xf32>
    %v5109 = stablehlo.add %v5108, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v5110 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5111 = stablehlo.multiply %v5110, %s3b1W1v : tensor<256x256x3x3xf32>
    %v5112 = stablehlo.add %v5111, %v5109 : tensor<256x256x3x3xf32>
    %v5113 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5114 = stablehlo.multiply %v5113, %v5112 : tensor<256x256x3x3xf32>
    %v5115 = stablehlo.subtract %s3b1W1, %v5114 : tensor<256x256x3x3xf32>
    %arsums3b1g1 = "stablehlo.all_reduce"(%v2761) ({
    ^bb0(%aras3b1g1: tensor<f32>, %arbs3b1g1: tensor<f32>):
      %aradds3b1g1 = stablehlo.add %aras3b1g1, %arbs3b1g1 : tensor<f32>
      stablehlo.return %aradds3b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g1 = stablehlo.divide %arsums3b1g1, %arns3b1g1 : tensor<256xf32>
    %v5116 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5117 = stablehlo.multiply %v5116, %s3b1g1 : tensor<256xf32>
    %v5118 = stablehlo.add %v5117, %armeans3b1g1 : tensor<256xf32>
    %v5119 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5120 = stablehlo.multiply %v5119, %s3b1g1v : tensor<256xf32>
    %v5121 = stablehlo.add %v5120, %v5118 : tensor<256xf32>
    %v5122 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5123 = stablehlo.multiply %v5122, %v5121 : tensor<256xf32>
    %v5124 = stablehlo.subtract %s3b1g1, %v5123 : tensor<256xf32>
    %arsums3b1bt1 = "stablehlo.all_reduce"(%v2764) ({
    ^bb0(%aras3b1bt1: tensor<f32>, %arbs3b1bt1: tensor<f32>):
      %aradds3b1bt1 = stablehlo.add %aras3b1bt1, %arbs3b1bt1 : tensor<f32>
      stablehlo.return %aradds3b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1bt1 = stablehlo.divide %arsums3b1bt1, %arns3b1bt1 : tensor<256xf32>
    %v5125 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5126 = stablehlo.multiply %v5125, %s3b1bt1 : tensor<256xf32>
    %v5127 = stablehlo.add %v5126, %armeans3b1bt1 : tensor<256xf32>
    %v5128 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5129 = stablehlo.multiply %v5128, %s3b1bt1v : tensor<256xf32>
    %v5130 = stablehlo.add %v5129, %v5127 : tensor<256xf32>
    %v5131 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5132 = stablehlo.multiply %v5131, %v5130 : tensor<256xf32>
    %v5133 = stablehlo.subtract %s3b1bt1, %v5132 : tensor<256xf32>
    %arsums3b1W2 = "stablehlo.all_reduce"(%v2770) ({
    ^bb0(%aras3b1W2: tensor<f32>, %arbs3b1W2: tensor<f32>):
      %aradds3b1W2 = stablehlo.add %aras3b1W2, %arbs3b1W2 : tensor<f32>
      stablehlo.return %aradds3b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b1W2 = stablehlo.divide %arsums3b1W2, %arns3b1W2 : tensor<256x256x3x3xf32>
    %v5134 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5135 = stablehlo.multiply %v5134, %s3b1W2 : tensor<256x256x3x3xf32>
    %v5136 = stablehlo.add %v5135, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v5137 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5138 = stablehlo.multiply %v5137, %s3b1W2v : tensor<256x256x3x3xf32>
    %v5139 = stablehlo.add %v5138, %v5136 : tensor<256x256x3x3xf32>
    %v5140 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5141 = stablehlo.multiply %v5140, %v5139 : tensor<256x256x3x3xf32>
    %v5142 = stablehlo.subtract %s3b1W2, %v5141 : tensor<256x256x3x3xf32>
    %arsums3b1g2 = "stablehlo.all_reduce"(%v2784) ({
    ^bb0(%aras3b1g2: tensor<f32>, %arbs3b1g2: tensor<f32>):
      %aradds3b1g2 = stablehlo.add %aras3b1g2, %arbs3b1g2 : tensor<f32>
      stablehlo.return %aradds3b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1g2 = stablehlo.divide %arsums3b1g2, %arns3b1g2 : tensor<256xf32>
    %v5143 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5144 = stablehlo.multiply %v5143, %s3b1g2 : tensor<256xf32>
    %v5145 = stablehlo.add %v5144, %armeans3b1g2 : tensor<256xf32>
    %v5146 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5147 = stablehlo.multiply %v5146, %s3b1g2v : tensor<256xf32>
    %v5148 = stablehlo.add %v5147, %v5145 : tensor<256xf32>
    %v5149 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5150 = stablehlo.multiply %v5149, %v5148 : tensor<256xf32>
    %v5151 = stablehlo.subtract %s3b1g2, %v5150 : tensor<256xf32>
    %arsums3b1bt2 = "stablehlo.all_reduce"(%v2787) ({
    ^bb0(%aras3b1bt2: tensor<f32>, %arbs3b1bt2: tensor<f32>):
      %aradds3b1bt2 = stablehlo.add %aras3b1bt2, %arbs3b1bt2 : tensor<f32>
      stablehlo.return %aradds3b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b1bt2 = stablehlo.divide %arsums3b1bt2, %arns3b1bt2 : tensor<256xf32>
    %v5152 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5153 = stablehlo.multiply %v5152, %s3b1bt2 : tensor<256xf32>
    %v5154 = stablehlo.add %v5153, %armeans3b1bt2 : tensor<256xf32>
    %v5155 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5156 = stablehlo.multiply %v5155, %s3b1bt2v : tensor<256xf32>
    %v5157 = stablehlo.add %v5156, %v5154 : tensor<256xf32>
    %v5158 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5159 = stablehlo.multiply %v5158, %v5157 : tensor<256xf32>
    %v5160 = stablehlo.subtract %s3b1bt2, %v5159 : tensor<256xf32>
    %arsums3b2W1 = "stablehlo.all_reduce"(%v2587) ({
    ^bb0(%aras3b2W1: tensor<f32>, %arbs3b2W1: tensor<f32>):
      %aradds3b2W1 = stablehlo.add %aras3b2W1, %arbs3b2W1 : tensor<f32>
      stablehlo.return %aradds3b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b2W1 = stablehlo.divide %arsums3b2W1, %arns3b2W1 : tensor<256x256x3x3xf32>
    %v5161 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5162 = stablehlo.multiply %v5161, %s3b2W1 : tensor<256x256x3x3xf32>
    %v5163 = stablehlo.add %v5162, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v5164 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5165 = stablehlo.multiply %v5164, %s3b2W1v : tensor<256x256x3x3xf32>
    %v5166 = stablehlo.add %v5165, %v5163 : tensor<256x256x3x3xf32>
    %v5167 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5168 = stablehlo.multiply %v5167, %v5166 : tensor<256x256x3x3xf32>
    %v5169 = stablehlo.subtract %s3b2W1, %v5168 : tensor<256x256x3x3xf32>
    %arsums3b2g1 = "stablehlo.all_reduce"(%v2601) ({
    ^bb0(%aras3b2g1: tensor<f32>, %arbs3b2g1: tensor<f32>):
      %aradds3b2g1 = stablehlo.add %aras3b2g1, %arbs3b2g1 : tensor<f32>
      stablehlo.return %aradds3b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g1 = stablehlo.divide %arsums3b2g1, %arns3b2g1 : tensor<256xf32>
    %v5170 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5171 = stablehlo.multiply %v5170, %s3b2g1 : tensor<256xf32>
    %v5172 = stablehlo.add %v5171, %armeans3b2g1 : tensor<256xf32>
    %v5173 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5174 = stablehlo.multiply %v5173, %s3b2g1v : tensor<256xf32>
    %v5175 = stablehlo.add %v5174, %v5172 : tensor<256xf32>
    %v5176 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5177 = stablehlo.multiply %v5176, %v5175 : tensor<256xf32>
    %v5178 = stablehlo.subtract %s3b2g1, %v5177 : tensor<256xf32>
    %arsums3b2bt1 = "stablehlo.all_reduce"(%v2604) ({
    ^bb0(%aras3b2bt1: tensor<f32>, %arbs3b2bt1: tensor<f32>):
      %aradds3b2bt1 = stablehlo.add %aras3b2bt1, %arbs3b2bt1 : tensor<f32>
      stablehlo.return %aradds3b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2bt1 = stablehlo.divide %arsums3b2bt1, %arns3b2bt1 : tensor<256xf32>
    %v5179 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5180 = stablehlo.multiply %v5179, %s3b2bt1 : tensor<256xf32>
    %v5181 = stablehlo.add %v5180, %armeans3b2bt1 : tensor<256xf32>
    %v5182 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5183 = stablehlo.multiply %v5182, %s3b2bt1v : tensor<256xf32>
    %v5184 = stablehlo.add %v5183, %v5181 : tensor<256xf32>
    %v5185 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5186 = stablehlo.multiply %v5185, %v5184 : tensor<256xf32>
    %v5187 = stablehlo.subtract %s3b2bt1, %v5186 : tensor<256xf32>
    %arsums3b2W2 = "stablehlo.all_reduce"(%v2610) ({
    ^bb0(%aras3b2W2: tensor<f32>, %arbs3b2W2: tensor<f32>):
      %aradds3b2W2 = stablehlo.add %aras3b2W2, %arbs3b2W2 : tensor<f32>
      stablehlo.return %aradds3b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b2W2 = stablehlo.divide %arsums3b2W2, %arns3b2W2 : tensor<256x256x3x3xf32>
    %v5188 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5189 = stablehlo.multiply %v5188, %s3b2W2 : tensor<256x256x3x3xf32>
    %v5190 = stablehlo.add %v5189, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v5191 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5192 = stablehlo.multiply %v5191, %s3b2W2v : tensor<256x256x3x3xf32>
    %v5193 = stablehlo.add %v5192, %v5190 : tensor<256x256x3x3xf32>
    %v5194 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5195 = stablehlo.multiply %v5194, %v5193 : tensor<256x256x3x3xf32>
    %v5196 = stablehlo.subtract %s3b2W2, %v5195 : tensor<256x256x3x3xf32>
    %arsums3b2g2 = "stablehlo.all_reduce"(%v2624) ({
    ^bb0(%aras3b2g2: tensor<f32>, %arbs3b2g2: tensor<f32>):
      %aradds3b2g2 = stablehlo.add %aras3b2g2, %arbs3b2g2 : tensor<f32>
      stablehlo.return %aradds3b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2g2 = stablehlo.divide %arsums3b2g2, %arns3b2g2 : tensor<256xf32>
    %v5197 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5198 = stablehlo.multiply %v5197, %s3b2g2 : tensor<256xf32>
    %v5199 = stablehlo.add %v5198, %armeans3b2g2 : tensor<256xf32>
    %v5200 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5201 = stablehlo.multiply %v5200, %s3b2g2v : tensor<256xf32>
    %v5202 = stablehlo.add %v5201, %v5199 : tensor<256xf32>
    %v5203 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5204 = stablehlo.multiply %v5203, %v5202 : tensor<256xf32>
    %v5205 = stablehlo.subtract %s3b2g2, %v5204 : tensor<256xf32>
    %arsums3b2bt2 = "stablehlo.all_reduce"(%v2627) ({
    ^bb0(%aras3b2bt2: tensor<f32>, %arbs3b2bt2: tensor<f32>):
      %aradds3b2bt2 = stablehlo.add %aras3b2bt2, %arbs3b2bt2 : tensor<f32>
      stablehlo.return %aradds3b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b2bt2 = stablehlo.divide %arsums3b2bt2, %arns3b2bt2 : tensor<256xf32>
    %v5206 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5207 = stablehlo.multiply %v5206, %s3b2bt2 : tensor<256xf32>
    %v5208 = stablehlo.add %v5207, %armeans3b2bt2 : tensor<256xf32>
    %v5209 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5210 = stablehlo.multiply %v5209, %s3b2bt2v : tensor<256xf32>
    %v5211 = stablehlo.add %v5210, %v5208 : tensor<256xf32>
    %v5212 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5213 = stablehlo.multiply %v5212, %v5211 : tensor<256xf32>
    %v5214 = stablehlo.subtract %s3b2bt2, %v5213 : tensor<256xf32>
    %arsums3b3W1 = "stablehlo.all_reduce"(%v2427) ({
    ^bb0(%aras3b3W1: tensor<f32>, %arbs3b3W1: tensor<f32>):
      %aradds3b3W1 = stablehlo.add %aras3b3W1, %arbs3b3W1 : tensor<f32>
      stablehlo.return %aradds3b3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b3W1 = stablehlo.divide %arsums3b3W1, %arns3b3W1 : tensor<256x256x3x3xf32>
    %v5215 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5216 = stablehlo.multiply %v5215, %s3b3W1 : tensor<256x256x3x3xf32>
    %v5217 = stablehlo.add %v5216, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v5218 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5219 = stablehlo.multiply %v5218, %s3b3W1v : tensor<256x256x3x3xf32>
    %v5220 = stablehlo.add %v5219, %v5217 : tensor<256x256x3x3xf32>
    %v5221 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5222 = stablehlo.multiply %v5221, %v5220 : tensor<256x256x3x3xf32>
    %v5223 = stablehlo.subtract %s3b3W1, %v5222 : tensor<256x256x3x3xf32>
    %arsums3b3g1 = "stablehlo.all_reduce"(%v2441) ({
    ^bb0(%aras3b3g1: tensor<f32>, %arbs3b3g1: tensor<f32>):
      %aradds3b3g1 = stablehlo.add %aras3b3g1, %arbs3b3g1 : tensor<f32>
      stablehlo.return %aradds3b3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g1 = stablehlo.divide %arsums3b3g1, %arns3b3g1 : tensor<256xf32>
    %v5224 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5225 = stablehlo.multiply %v5224, %s3b3g1 : tensor<256xf32>
    %v5226 = stablehlo.add %v5225, %armeans3b3g1 : tensor<256xf32>
    %v5227 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5228 = stablehlo.multiply %v5227, %s3b3g1v : tensor<256xf32>
    %v5229 = stablehlo.add %v5228, %v5226 : tensor<256xf32>
    %v5230 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5231 = stablehlo.multiply %v5230, %v5229 : tensor<256xf32>
    %v5232 = stablehlo.subtract %s3b3g1, %v5231 : tensor<256xf32>
    %arsums3b3bt1 = "stablehlo.all_reduce"(%v2444) ({
    ^bb0(%aras3b3bt1: tensor<f32>, %arbs3b3bt1: tensor<f32>):
      %aradds3b3bt1 = stablehlo.add %aras3b3bt1, %arbs3b3bt1 : tensor<f32>
      stablehlo.return %aradds3b3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3bt1 = stablehlo.divide %arsums3b3bt1, %arns3b3bt1 : tensor<256xf32>
    %v5233 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5234 = stablehlo.multiply %v5233, %s3b3bt1 : tensor<256xf32>
    %v5235 = stablehlo.add %v5234, %armeans3b3bt1 : tensor<256xf32>
    %v5236 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5237 = stablehlo.multiply %v5236, %s3b3bt1v : tensor<256xf32>
    %v5238 = stablehlo.add %v5237, %v5235 : tensor<256xf32>
    %v5239 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5240 = stablehlo.multiply %v5239, %v5238 : tensor<256xf32>
    %v5241 = stablehlo.subtract %s3b3bt1, %v5240 : tensor<256xf32>
    %arsums3b3W2 = "stablehlo.all_reduce"(%v2450) ({
    ^bb0(%aras3b3W2: tensor<f32>, %arbs3b3W2: tensor<f32>):
      %aradds3b3W2 = stablehlo.add %aras3b3W2, %arbs3b3W2 : tensor<f32>
      stablehlo.return %aradds3b3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b3W2 = stablehlo.divide %arsums3b3W2, %arns3b3W2 : tensor<256x256x3x3xf32>
    %v5242 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5243 = stablehlo.multiply %v5242, %s3b3W2 : tensor<256x256x3x3xf32>
    %v5244 = stablehlo.add %v5243, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v5245 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5246 = stablehlo.multiply %v5245, %s3b3W2v : tensor<256x256x3x3xf32>
    %v5247 = stablehlo.add %v5246, %v5244 : tensor<256x256x3x3xf32>
    %v5248 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5249 = stablehlo.multiply %v5248, %v5247 : tensor<256x256x3x3xf32>
    %v5250 = stablehlo.subtract %s3b3W2, %v5249 : tensor<256x256x3x3xf32>
    %arsums3b3g2 = "stablehlo.all_reduce"(%v2464) ({
    ^bb0(%aras3b3g2: tensor<f32>, %arbs3b3g2: tensor<f32>):
      %aradds3b3g2 = stablehlo.add %aras3b3g2, %arbs3b3g2 : tensor<f32>
      stablehlo.return %aradds3b3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3g2 = stablehlo.divide %arsums3b3g2, %arns3b3g2 : tensor<256xf32>
    %v5251 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5252 = stablehlo.multiply %v5251, %s3b3g2 : tensor<256xf32>
    %v5253 = stablehlo.add %v5252, %armeans3b3g2 : tensor<256xf32>
    %v5254 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5255 = stablehlo.multiply %v5254, %s3b3g2v : tensor<256xf32>
    %v5256 = stablehlo.add %v5255, %v5253 : tensor<256xf32>
    %v5257 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5258 = stablehlo.multiply %v5257, %v5256 : tensor<256xf32>
    %v5259 = stablehlo.subtract %s3b3g2, %v5258 : tensor<256xf32>
    %arsums3b3bt2 = "stablehlo.all_reduce"(%v2467) ({
    ^bb0(%aras3b3bt2: tensor<f32>, %arbs3b3bt2: tensor<f32>):
      %aradds3b3bt2 = stablehlo.add %aras3b3bt2, %arbs3b3bt2 : tensor<f32>
      stablehlo.return %aradds3b3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b3bt2 = stablehlo.divide %arsums3b3bt2, %arns3b3bt2 : tensor<256xf32>
    %v5260 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5261 = stablehlo.multiply %v5260, %s3b3bt2 : tensor<256xf32>
    %v5262 = stablehlo.add %v5261, %armeans3b3bt2 : tensor<256xf32>
    %v5263 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5264 = stablehlo.multiply %v5263, %s3b3bt2v : tensor<256xf32>
    %v5265 = stablehlo.add %v5264, %v5262 : tensor<256xf32>
    %v5266 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5267 = stablehlo.multiply %v5266, %v5265 : tensor<256xf32>
    %v5268 = stablehlo.subtract %s3b3bt2, %v5267 : tensor<256xf32>
    %arsums3b4W1 = "stablehlo.all_reduce"(%v2267) ({
    ^bb0(%aras3b4W1: tensor<f32>, %arbs3b4W1: tensor<f32>):
      %aradds3b4W1 = stablehlo.add %aras3b4W1, %arbs3b4W1 : tensor<f32>
      stablehlo.return %aradds3b4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W1 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b4W1 = stablehlo.divide %arsums3b4W1, %arns3b4W1 : tensor<256x256x3x3xf32>
    %v5269 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5270 = stablehlo.multiply %v5269, %s3b4W1 : tensor<256x256x3x3xf32>
    %v5271 = stablehlo.add %v5270, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v5272 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5273 = stablehlo.multiply %v5272, %s3b4W1v : tensor<256x256x3x3xf32>
    %v5274 = stablehlo.add %v5273, %v5271 : tensor<256x256x3x3xf32>
    %v5275 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5276 = stablehlo.multiply %v5275, %v5274 : tensor<256x256x3x3xf32>
    %v5277 = stablehlo.subtract %s3b4W1, %v5276 : tensor<256x256x3x3xf32>
    %arsums3b4g1 = "stablehlo.all_reduce"(%v2281) ({
    ^bb0(%aras3b4g1: tensor<f32>, %arbs3b4g1: tensor<f32>):
      %aradds3b4g1 = stablehlo.add %aras3b4g1, %arbs3b4g1 : tensor<f32>
      stablehlo.return %aradds3b4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g1 = stablehlo.divide %arsums3b4g1, %arns3b4g1 : tensor<256xf32>
    %v5278 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5279 = stablehlo.multiply %v5278, %s3b4g1 : tensor<256xf32>
    %v5280 = stablehlo.add %v5279, %armeans3b4g1 : tensor<256xf32>
    %v5281 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5282 = stablehlo.multiply %v5281, %s3b4g1v : tensor<256xf32>
    %v5283 = stablehlo.add %v5282, %v5280 : tensor<256xf32>
    %v5284 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5285 = stablehlo.multiply %v5284, %v5283 : tensor<256xf32>
    %v5286 = stablehlo.subtract %s3b4g1, %v5285 : tensor<256xf32>
    %arsums3b4bt1 = "stablehlo.all_reduce"(%v2284) ({
    ^bb0(%aras3b4bt1: tensor<f32>, %arbs3b4bt1: tensor<f32>):
      %aradds3b4bt1 = stablehlo.add %aras3b4bt1, %arbs3b4bt1 : tensor<f32>
      stablehlo.return %aradds3b4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt1 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4bt1 = stablehlo.divide %arsums3b4bt1, %arns3b4bt1 : tensor<256xf32>
    %v5287 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5288 = stablehlo.multiply %v5287, %s3b4bt1 : tensor<256xf32>
    %v5289 = stablehlo.add %v5288, %armeans3b4bt1 : tensor<256xf32>
    %v5290 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5291 = stablehlo.multiply %v5290, %s3b4bt1v : tensor<256xf32>
    %v5292 = stablehlo.add %v5291, %v5289 : tensor<256xf32>
    %v5293 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5294 = stablehlo.multiply %v5293, %v5292 : tensor<256xf32>
    %v5295 = stablehlo.subtract %s3b4bt1, %v5294 : tensor<256xf32>
    %arsums3b4W2 = "stablehlo.all_reduce"(%v2290) ({
    ^bb0(%aras3b4W2: tensor<f32>, %arbs3b4W2: tensor<f32>):
      %aradds3b4W2 = stablehlo.add %aras3b4W2, %arbs3b4W2 : tensor<f32>
      stablehlo.return %aradds3b4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W2 = stablehlo.constant dense<2.0> : tensor<256x256x3x3xf32>
    %armeans3b4W2 = stablehlo.divide %arsums3b4W2, %arns3b4W2 : tensor<256x256x3x3xf32>
    %v5296 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5297 = stablehlo.multiply %v5296, %s3b4W2 : tensor<256x256x3x3xf32>
    %v5298 = stablehlo.add %v5297, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v5299 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5300 = stablehlo.multiply %v5299, %s3b4W2v : tensor<256x256x3x3xf32>
    %v5301 = stablehlo.add %v5300, %v5298 : tensor<256x256x3x3xf32>
    %v5302 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5303 = stablehlo.multiply %v5302, %v5301 : tensor<256x256x3x3xf32>
    %v5304 = stablehlo.subtract %s3b4W2, %v5303 : tensor<256x256x3x3xf32>
    %arsums3b4g2 = "stablehlo.all_reduce"(%v2304) ({
    ^bb0(%aras3b4g2: tensor<f32>, %arbs3b4g2: tensor<f32>):
      %aradds3b4g2 = stablehlo.add %aras3b4g2, %arbs3b4g2 : tensor<f32>
      stablehlo.return %aradds3b4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4g2 = stablehlo.divide %arsums3b4g2, %arns3b4g2 : tensor<256xf32>
    %v5305 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5306 = stablehlo.multiply %v5305, %s3b4g2 : tensor<256xf32>
    %v5307 = stablehlo.add %v5306, %armeans3b4g2 : tensor<256xf32>
    %v5308 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5309 = stablehlo.multiply %v5308, %s3b4g2v : tensor<256xf32>
    %v5310 = stablehlo.add %v5309, %v5307 : tensor<256xf32>
    %v5311 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5312 = stablehlo.multiply %v5311, %v5310 : tensor<256xf32>
    %v5313 = stablehlo.subtract %s3b4g2, %v5312 : tensor<256xf32>
    %arsums3b4bt2 = "stablehlo.all_reduce"(%v2307) ({
    ^bb0(%aras3b4bt2: tensor<f32>, %arbs3b4bt2: tensor<f32>):
      %aradds3b4bt2 = stablehlo.add %aras3b4bt2, %arbs3b4bt2 : tensor<f32>
      stablehlo.return %aradds3b4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt2 = stablehlo.constant dense<2.0> : tensor<256xf32>
    %armeans3b4bt2 = stablehlo.divide %arsums3b4bt2, %arns3b4bt2 : tensor<256xf32>
    %v5314 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5315 = stablehlo.multiply %v5314, %s3b4bt2 : tensor<256xf32>
    %v5316 = stablehlo.add %v5315, %armeans3b4bt2 : tensor<256xf32>
    %v5317 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5318 = stablehlo.multiply %v5317, %s3b4bt2v : tensor<256xf32>
    %v5319 = stablehlo.add %v5318, %v5316 : tensor<256xf32>
    %v5320 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5321 = stablehlo.multiply %v5320, %v5319 : tensor<256xf32>
    %v5322 = stablehlo.subtract %s3b4bt2, %v5321 : tensor<256xf32>
    %arsumd4W1 = "stablehlo.all_reduce"(%v2082) ({
    ^bb0(%arad4W1: tensor<f32>, %arbd4W1: tensor<f32>):
      %araddd4W1 = stablehlo.add %arad4W1, %arbd4W1 : tensor<f32>
      stablehlo.return %araddd4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf32>
    %arnd4W1 = stablehlo.constant dense<2.0> : tensor<512x256x3x3xf32>
    %armeand4W1 = stablehlo.divide %arsumd4W1, %arnd4W1 : tensor<512x256x3x3xf32>
    %v5323 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5324 = stablehlo.multiply %v5323, %d4W1 : tensor<512x256x3x3xf32>
    %v5325 = stablehlo.add %v5324, %armeand4W1 : tensor<512x256x3x3xf32>
    %v5326 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5327 = stablehlo.multiply %v5326, %d4W1v : tensor<512x256x3x3xf32>
    %v5328 = stablehlo.add %v5327, %v5325 : tensor<512x256x3x3xf32>
    %v5329 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5330 = stablehlo.multiply %v5329, %v5328 : tensor<512x256x3x3xf32>
    %v5331 = stablehlo.subtract %d4W1, %v5330 : tensor<512x256x3x3xf32>
    %arsumd4g1 = "stablehlo.all_reduce"(%v2096) ({
    ^bb0(%arad4g1: tensor<f32>, %arbd4g1: tensor<f32>):
      %araddd4g1 = stablehlo.add %arad4g1, %arbd4g1 : tensor<f32>
      stablehlo.return %araddd4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g1 = stablehlo.divide %arsumd4g1, %arnd4g1 : tensor<512xf32>
    %v5332 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5333 = stablehlo.multiply %v5332, %d4g1 : tensor<512xf32>
    %v5334 = stablehlo.add %v5333, %armeand4g1 : tensor<512xf32>
    %v5335 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5336 = stablehlo.multiply %v5335, %d4g1v : tensor<512xf32>
    %v5337 = stablehlo.add %v5336, %v5334 : tensor<512xf32>
    %v5338 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5339 = stablehlo.multiply %v5338, %v5337 : tensor<512xf32>
    %v5340 = stablehlo.subtract %d4g1, %v5339 : tensor<512xf32>
    %arsumd4bt1 = "stablehlo.all_reduce"(%v2099) ({
    ^bb0(%arad4bt1: tensor<f32>, %arbd4bt1: tensor<f32>):
      %araddd4bt1 = stablehlo.add %arad4bt1, %arbd4bt1 : tensor<f32>
      stablehlo.return %araddd4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4bt1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4bt1 = stablehlo.divide %arsumd4bt1, %arnd4bt1 : tensor<512xf32>
    %v5341 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5342 = stablehlo.multiply %v5341, %d4bt1 : tensor<512xf32>
    %v5343 = stablehlo.add %v5342, %armeand4bt1 : tensor<512xf32>
    %v5344 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5345 = stablehlo.multiply %v5344, %d4bt1v : tensor<512xf32>
    %v5346 = stablehlo.add %v5345, %v5343 : tensor<512xf32>
    %v5347 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5348 = stablehlo.multiply %v5347, %v5346 : tensor<512xf32>
    %v5349 = stablehlo.subtract %d4bt1, %v5348 : tensor<512xf32>
    %arsumd4W2 = "stablehlo.all_reduce"(%v2105) ({
    ^bb0(%arad4W2: tensor<f32>, %arbd4W2: tensor<f32>):
      %araddd4W2 = stablehlo.add %arad4W2, %arbd4W2 : tensor<f32>
      stablehlo.return %araddd4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arnd4W2 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeand4W2 = stablehlo.divide %arsumd4W2, %arnd4W2 : tensor<512x512x3x3xf32>
    %v5350 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5351 = stablehlo.multiply %v5350, %d4W2 : tensor<512x512x3x3xf32>
    %v5352 = stablehlo.add %v5351, %armeand4W2 : tensor<512x512x3x3xf32>
    %v5353 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5354 = stablehlo.multiply %v5353, %d4W2v : tensor<512x512x3x3xf32>
    %v5355 = stablehlo.add %v5354, %v5352 : tensor<512x512x3x3xf32>
    %v5356 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5357 = stablehlo.multiply %v5356, %v5355 : tensor<512x512x3x3xf32>
    %v5358 = stablehlo.subtract %d4W2, %v5357 : tensor<512x512x3x3xf32>
    %arsumd4g2 = "stablehlo.all_reduce"(%v2119) ({
    ^bb0(%arad4g2: tensor<f32>, %arbd4g2: tensor<f32>):
      %araddd4g2 = stablehlo.add %arad4g2, %arbd4g2 : tensor<f32>
      stablehlo.return %araddd4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4g2 = stablehlo.divide %arsumd4g2, %arnd4g2 : tensor<512xf32>
    %v5359 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5360 = stablehlo.multiply %v5359, %d4g2 : tensor<512xf32>
    %v5361 = stablehlo.add %v5360, %armeand4g2 : tensor<512xf32>
    %v5362 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5363 = stablehlo.multiply %v5362, %d4g2v : tensor<512xf32>
    %v5364 = stablehlo.add %v5363, %v5361 : tensor<512xf32>
    %v5365 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5366 = stablehlo.multiply %v5365, %v5364 : tensor<512xf32>
    %v5367 = stablehlo.subtract %d4g2, %v5366 : tensor<512xf32>
    %arsumd4bt2 = "stablehlo.all_reduce"(%v2122) ({
    ^bb0(%arad4bt2: tensor<f32>, %arbd4bt2: tensor<f32>):
      %araddd4bt2 = stablehlo.add %arad4bt2, %arbd4bt2 : tensor<f32>
      stablehlo.return %araddd4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4bt2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4bt2 = stablehlo.divide %arsumd4bt2, %arnd4bt2 : tensor<512xf32>
    %v5368 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5369 = stablehlo.multiply %v5368, %d4bt2 : tensor<512xf32>
    %v5370 = stablehlo.add %v5369, %armeand4bt2 : tensor<512xf32>
    %v5371 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5372 = stablehlo.multiply %v5371, %d4bt2v : tensor<512xf32>
    %v5373 = stablehlo.add %v5372, %v5370 : tensor<512xf32>
    %v5374 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5375 = stablehlo.multiply %v5374, %v5373 : tensor<512xf32>
    %v5376 = stablehlo.subtract %d4bt2, %v5375 : tensor<512xf32>
    %arsumd4Wp = "stablehlo.all_reduce"(%v2130) ({
    ^bb0(%arad4Wp: tensor<f32>, %arbd4Wp: tensor<f32>):
      %araddd4Wp = stablehlo.add %arad4Wp, %arbd4Wp : tensor<f32>
      stablehlo.return %araddd4Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf32>
    %arnd4Wp = stablehlo.constant dense<2.0> : tensor<512x256x1x1xf32>
    %armeand4Wp = stablehlo.divide %arsumd4Wp, %arnd4Wp : tensor<512x256x1x1xf32>
    %v5377 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5378 = stablehlo.multiply %v5377, %d4Wp : tensor<512x256x1x1xf32>
    %v5379 = stablehlo.add %v5378, %armeand4Wp : tensor<512x256x1x1xf32>
    %v5380 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5381 = stablehlo.multiply %v5380, %d4Wpv : tensor<512x256x1x1xf32>
    %v5382 = stablehlo.add %v5381, %v5379 : tensor<512x256x1x1xf32>
    %v5383 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5384 = stablehlo.multiply %v5383, %v5382 : tensor<512x256x1x1xf32>
    %v5385 = stablehlo.subtract %d4Wp, %v5384 : tensor<512x256x1x1xf32>
    %arsumd4gp = "stablehlo.all_reduce"(%v2144) ({
    ^bb0(%arad4gp: tensor<f32>, %arbd4gp: tensor<f32>):
      %araddd4gp = stablehlo.add %arad4gp, %arbd4gp : tensor<f32>
      stablehlo.return %araddd4gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4gp = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4gp = stablehlo.divide %arsumd4gp, %arnd4gp : tensor<512xf32>
    %v5386 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5387 = stablehlo.multiply %v5386, %d4gp : tensor<512xf32>
    %v5388 = stablehlo.add %v5387, %armeand4gp : tensor<512xf32>
    %v5389 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5390 = stablehlo.multiply %v5389, %d4gpv : tensor<512xf32>
    %v5391 = stablehlo.add %v5390, %v5388 : tensor<512xf32>
    %v5392 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5393 = stablehlo.multiply %v5392, %v5391 : tensor<512xf32>
    %v5394 = stablehlo.subtract %d4gp, %v5393 : tensor<512xf32>
    %arsumd4btp = "stablehlo.all_reduce"(%v2147) ({
    ^bb0(%arad4btp: tensor<f32>, %arbd4btp: tensor<f32>):
      %araddd4btp = stablehlo.add %arad4btp, %arbd4btp : tensor<f32>
      stablehlo.return %araddd4btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4btp = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeand4btp = stablehlo.divide %arsumd4btp, %arnd4btp : tensor<512xf32>
    %v5395 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5396 = stablehlo.multiply %v5395, %d4btp : tensor<512xf32>
    %v5397 = stablehlo.add %v5396, %armeand4btp : tensor<512xf32>
    %v5398 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5399 = stablehlo.multiply %v5398, %d4btpv : tensor<512xf32>
    %v5400 = stablehlo.add %v5399, %v5397 : tensor<512xf32>
    %v5401 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5402 = stablehlo.multiply %v5401, %v5400 : tensor<512xf32>
    %v5403 = stablehlo.subtract %d4btp, %v5402 : tensor<512xf32>
    %arsums4b0W1 = "stablehlo.all_reduce"(%v1867) ({
    ^bb0(%aras4b0W1: tensor<f32>, %arbs4b0W1: tensor<f32>):
      %aradds4b0W1 = stablehlo.add %aras4b0W1, %arbs4b0W1 : tensor<f32>
      stablehlo.return %aradds4b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W1 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b0W1 = stablehlo.divide %arsums4b0W1, %arns4b0W1 : tensor<512x512x3x3xf32>
    %v5404 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5405 = stablehlo.multiply %v5404, %s4b0W1 : tensor<512x512x3x3xf32>
    %v5406 = stablehlo.add %v5405, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v5407 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5408 = stablehlo.multiply %v5407, %s4b0W1v : tensor<512x512x3x3xf32>
    %v5409 = stablehlo.add %v5408, %v5406 : tensor<512x512x3x3xf32>
    %v5410 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5411 = stablehlo.multiply %v5410, %v5409 : tensor<512x512x3x3xf32>
    %v5412 = stablehlo.subtract %s4b0W1, %v5411 : tensor<512x512x3x3xf32>
    %arsums4b0g1 = "stablehlo.all_reduce"(%v1881) ({
    ^bb0(%aras4b0g1: tensor<f32>, %arbs4b0g1: tensor<f32>):
      %aradds4b0g1 = stablehlo.add %aras4b0g1, %arbs4b0g1 : tensor<f32>
      stablehlo.return %aradds4b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g1 = stablehlo.divide %arsums4b0g1, %arns4b0g1 : tensor<512xf32>
    %v5413 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5414 = stablehlo.multiply %v5413, %s4b0g1 : tensor<512xf32>
    %v5415 = stablehlo.add %v5414, %armeans4b0g1 : tensor<512xf32>
    %v5416 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5417 = stablehlo.multiply %v5416, %s4b0g1v : tensor<512xf32>
    %v5418 = stablehlo.add %v5417, %v5415 : tensor<512xf32>
    %v5419 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5420 = stablehlo.multiply %v5419, %v5418 : tensor<512xf32>
    %v5421 = stablehlo.subtract %s4b0g1, %v5420 : tensor<512xf32>
    %arsums4b0bt1 = "stablehlo.all_reduce"(%v1884) ({
    ^bb0(%aras4b0bt1: tensor<f32>, %arbs4b0bt1: tensor<f32>):
      %aradds4b0bt1 = stablehlo.add %aras4b0bt1, %arbs4b0bt1 : tensor<f32>
      stablehlo.return %aradds4b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0bt1 = stablehlo.divide %arsums4b0bt1, %arns4b0bt1 : tensor<512xf32>
    %v5422 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5423 = stablehlo.multiply %v5422, %s4b0bt1 : tensor<512xf32>
    %v5424 = stablehlo.add %v5423, %armeans4b0bt1 : tensor<512xf32>
    %v5425 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5426 = stablehlo.multiply %v5425, %s4b0bt1v : tensor<512xf32>
    %v5427 = stablehlo.add %v5426, %v5424 : tensor<512xf32>
    %v5428 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5429 = stablehlo.multiply %v5428, %v5427 : tensor<512xf32>
    %v5430 = stablehlo.subtract %s4b0bt1, %v5429 : tensor<512xf32>
    %arsums4b0W2 = "stablehlo.all_reduce"(%v1890) ({
    ^bb0(%aras4b0W2: tensor<f32>, %arbs4b0W2: tensor<f32>):
      %aradds4b0W2 = stablehlo.add %aras4b0W2, %arbs4b0W2 : tensor<f32>
      stablehlo.return %aradds4b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W2 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b0W2 = stablehlo.divide %arsums4b0W2, %arns4b0W2 : tensor<512x512x3x3xf32>
    %v5431 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5432 = stablehlo.multiply %v5431, %s4b0W2 : tensor<512x512x3x3xf32>
    %v5433 = stablehlo.add %v5432, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v5434 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5435 = stablehlo.multiply %v5434, %s4b0W2v : tensor<512x512x3x3xf32>
    %v5436 = stablehlo.add %v5435, %v5433 : tensor<512x512x3x3xf32>
    %v5437 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5438 = stablehlo.multiply %v5437, %v5436 : tensor<512x512x3x3xf32>
    %v5439 = stablehlo.subtract %s4b0W2, %v5438 : tensor<512x512x3x3xf32>
    %arsums4b0g2 = "stablehlo.all_reduce"(%v1904) ({
    ^bb0(%aras4b0g2: tensor<f32>, %arbs4b0g2: tensor<f32>):
      %aradds4b0g2 = stablehlo.add %aras4b0g2, %arbs4b0g2 : tensor<f32>
      stablehlo.return %aradds4b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0g2 = stablehlo.divide %arsums4b0g2, %arns4b0g2 : tensor<512xf32>
    %v5440 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5441 = stablehlo.multiply %v5440, %s4b0g2 : tensor<512xf32>
    %v5442 = stablehlo.add %v5441, %armeans4b0g2 : tensor<512xf32>
    %v5443 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5444 = stablehlo.multiply %v5443, %s4b0g2v : tensor<512xf32>
    %v5445 = stablehlo.add %v5444, %v5442 : tensor<512xf32>
    %v5446 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5447 = stablehlo.multiply %v5446, %v5445 : tensor<512xf32>
    %v5448 = stablehlo.subtract %s4b0g2, %v5447 : tensor<512xf32>
    %arsums4b0bt2 = "stablehlo.all_reduce"(%v1907) ({
    ^bb0(%aras4b0bt2: tensor<f32>, %arbs4b0bt2: tensor<f32>):
      %aradds4b0bt2 = stablehlo.add %aras4b0bt2, %arbs4b0bt2 : tensor<f32>
      stablehlo.return %aradds4b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b0bt2 = stablehlo.divide %arsums4b0bt2, %arns4b0bt2 : tensor<512xf32>
    %v5449 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5450 = stablehlo.multiply %v5449, %s4b0bt2 : tensor<512xf32>
    %v5451 = stablehlo.add %v5450, %armeans4b0bt2 : tensor<512xf32>
    %v5452 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5453 = stablehlo.multiply %v5452, %s4b0bt2v : tensor<512xf32>
    %v5454 = stablehlo.add %v5453, %v5451 : tensor<512xf32>
    %v5455 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5456 = stablehlo.multiply %v5455, %v5454 : tensor<512xf32>
    %v5457 = stablehlo.subtract %s4b0bt2, %v5456 : tensor<512xf32>
    %arsums4b1W1 = "stablehlo.all_reduce"(%v1707) ({
    ^bb0(%aras4b1W1: tensor<f32>, %arbs4b1W1: tensor<f32>):
      %aradds4b1W1 = stablehlo.add %aras4b1W1, %arbs4b1W1 : tensor<f32>
      stablehlo.return %aradds4b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W1 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b1W1 = stablehlo.divide %arsums4b1W1, %arns4b1W1 : tensor<512x512x3x3xf32>
    %v5458 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5459 = stablehlo.multiply %v5458, %s4b1W1 : tensor<512x512x3x3xf32>
    %v5460 = stablehlo.add %v5459, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v5461 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5462 = stablehlo.multiply %v5461, %s4b1W1v : tensor<512x512x3x3xf32>
    %v5463 = stablehlo.add %v5462, %v5460 : tensor<512x512x3x3xf32>
    %v5464 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5465 = stablehlo.multiply %v5464, %v5463 : tensor<512x512x3x3xf32>
    %v5466 = stablehlo.subtract %s4b1W1, %v5465 : tensor<512x512x3x3xf32>
    %arsums4b1g1 = "stablehlo.all_reduce"(%v1721) ({
    ^bb0(%aras4b1g1: tensor<f32>, %arbs4b1g1: tensor<f32>):
      %aradds4b1g1 = stablehlo.add %aras4b1g1, %arbs4b1g1 : tensor<f32>
      stablehlo.return %aradds4b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g1 = stablehlo.divide %arsums4b1g1, %arns4b1g1 : tensor<512xf32>
    %v5467 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5468 = stablehlo.multiply %v5467, %s4b1g1 : tensor<512xf32>
    %v5469 = stablehlo.add %v5468, %armeans4b1g1 : tensor<512xf32>
    %v5470 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5471 = stablehlo.multiply %v5470, %s4b1g1v : tensor<512xf32>
    %v5472 = stablehlo.add %v5471, %v5469 : tensor<512xf32>
    %v5473 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5474 = stablehlo.multiply %v5473, %v5472 : tensor<512xf32>
    %v5475 = stablehlo.subtract %s4b1g1, %v5474 : tensor<512xf32>
    %arsums4b1bt1 = "stablehlo.all_reduce"(%v1724) ({
    ^bb0(%aras4b1bt1: tensor<f32>, %arbs4b1bt1: tensor<f32>):
      %aradds4b1bt1 = stablehlo.add %aras4b1bt1, %arbs4b1bt1 : tensor<f32>
      stablehlo.return %aradds4b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt1 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1bt1 = stablehlo.divide %arsums4b1bt1, %arns4b1bt1 : tensor<512xf32>
    %v5476 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5477 = stablehlo.multiply %v5476, %s4b1bt1 : tensor<512xf32>
    %v5478 = stablehlo.add %v5477, %armeans4b1bt1 : tensor<512xf32>
    %v5479 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5480 = stablehlo.multiply %v5479, %s4b1bt1v : tensor<512xf32>
    %v5481 = stablehlo.add %v5480, %v5478 : tensor<512xf32>
    %v5482 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5483 = stablehlo.multiply %v5482, %v5481 : tensor<512xf32>
    %v5484 = stablehlo.subtract %s4b1bt1, %v5483 : tensor<512xf32>
    %arsums4b1W2 = "stablehlo.all_reduce"(%v1730) ({
    ^bb0(%aras4b1W2: tensor<f32>, %arbs4b1W2: tensor<f32>):
      %aradds4b1W2 = stablehlo.add %aras4b1W2, %arbs4b1W2 : tensor<f32>
      stablehlo.return %aradds4b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W2 = stablehlo.constant dense<2.0> : tensor<512x512x3x3xf32>
    %armeans4b1W2 = stablehlo.divide %arsums4b1W2, %arns4b1W2 : tensor<512x512x3x3xf32>
    %v5485 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5486 = stablehlo.multiply %v5485, %s4b1W2 : tensor<512x512x3x3xf32>
    %v5487 = stablehlo.add %v5486, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v5488 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5489 = stablehlo.multiply %v5488, %s4b1W2v : tensor<512x512x3x3xf32>
    %v5490 = stablehlo.add %v5489, %v5487 : tensor<512x512x3x3xf32>
    %v5491 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5492 = stablehlo.multiply %v5491, %v5490 : tensor<512x512x3x3xf32>
    %v5493 = stablehlo.subtract %s4b1W2, %v5492 : tensor<512x512x3x3xf32>
    %arsums4b1g2 = "stablehlo.all_reduce"(%v1744) ({
    ^bb0(%aras4b1g2: tensor<f32>, %arbs4b1g2: tensor<f32>):
      %aradds4b1g2 = stablehlo.add %aras4b1g2, %arbs4b1g2 : tensor<f32>
      stablehlo.return %aradds4b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1g2 = stablehlo.divide %arsums4b1g2, %arns4b1g2 : tensor<512xf32>
    %v5494 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5495 = stablehlo.multiply %v5494, %s4b1g2 : tensor<512xf32>
    %v5496 = stablehlo.add %v5495, %armeans4b1g2 : tensor<512xf32>
    %v5497 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5498 = stablehlo.multiply %v5497, %s4b1g2v : tensor<512xf32>
    %v5499 = stablehlo.add %v5498, %v5496 : tensor<512xf32>
    %v5500 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5501 = stablehlo.multiply %v5500, %v5499 : tensor<512xf32>
    %v5502 = stablehlo.subtract %s4b1g2, %v5501 : tensor<512xf32>
    %arsums4b1bt2 = "stablehlo.all_reduce"(%v1747) ({
    ^bb0(%aras4b1bt2: tensor<f32>, %arbs4b1bt2: tensor<f32>):
      %aradds4b1bt2 = stablehlo.add %aras4b1bt2, %arbs4b1bt2 : tensor<f32>
      stablehlo.return %aradds4b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt2 = stablehlo.constant dense<2.0> : tensor<512xf32>
    %armeans4b1bt2 = stablehlo.divide %arsums4b1bt2, %arns4b1bt2 : tensor<512xf32>
    %v5503 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5504 = stablehlo.multiply %v5503, %s4b1bt2 : tensor<512xf32>
    %v5505 = stablehlo.add %v5504, %armeans4b1bt2 : tensor<512xf32>
    %v5506 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5507 = stablehlo.multiply %v5506, %s4b1bt2v : tensor<512xf32>
    %v5508 = stablehlo.add %v5507, %v5505 : tensor<512xf32>
    %v5509 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5510 = stablehlo.multiply %v5509, %v5508 : tensor<512xf32>
    %v5511 = stablehlo.subtract %s4b1bt2, %v5510 : tensor<512xf32>
    %arsumWd = "stablehlo.all_reduce"(%v1581) ({
    ^bb0(%araWd: tensor<f32>, %arbWd: tensor<f32>):
      %araddWd = stablehlo.add %araWd, %arbWd : tensor<f32>
      stablehlo.return %araddWd : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<512x1000xf32>) -> tensor<512x1000xf32>
    %arnWd = stablehlo.constant dense<2.0> : tensor<512x1000xf32>
    %armeanWd = stablehlo.divide %arsumWd, %arnWd : tensor<512x1000xf32>
    %v5512 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %v5513 = stablehlo.multiply %v5512, %Wd : tensor<512x1000xf32>
    %v5514 = stablehlo.add %v5513, %armeanWd : tensor<512x1000xf32>
    %v5515 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %v5516 = stablehlo.multiply %v5515, %Wdv : tensor<512x1000xf32>
    %v5517 = stablehlo.add %v5516, %v5514 : tensor<512x1000xf32>
    %v5518 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %v5519 = stablehlo.multiply %v5518, %v5517 : tensor<512x1000xf32>
    %v5520 = stablehlo.subtract %Wd, %v5519 : tensor<512x1000xf32>
    %arsumbd = "stablehlo.all_reduce"(%v1583) ({
    ^bb0(%arabd: tensor<f32>, %arbbd: tensor<f32>):
      %araddbd = stablehlo.add %arabd, %arbbd : tensor<f32>
      stablehlo.return %araddbd : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1000xf32>) -> tensor<1000xf32>
    %arnbd = stablehlo.constant dense<2.0> : tensor<1000xf32>
    %armeanbd = stablehlo.divide %arsumbd, %arnbd : tensor<1000xf32>
    %v5521 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v5522 = stablehlo.multiply %v5521, %bd : tensor<1000xf32>
    %v5523 = stablehlo.add %v5522, %armeanbd : tensor<1000xf32>
    %v5524 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v5525 = stablehlo.multiply %v5524, %bdv : tensor<1000xf32>
    %v5526 = stablehlo.add %v5525, %v5523 : tensor<1000xf32>
    %v5527 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v5528 = stablehlo.multiply %v5527, %v5526 : tensor<1000xf32>
    %v5529 = stablehlo.subtract %bd, %v5528 : tensor<1000xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1569 : tensor<128x1000xf32>
    %lohll = stablehlo.multiply %onehot, %llog : tensor<128x1000xf32>
    %lt1s = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<128x1000xf32>, tensor<f32>) -> tensor<128xf32>
    %llsr = stablehlo.reduce(%llog init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<128x1000xf32>, tensor<f32>) -> tensor<128xf32>
    %lomac = stablehlo.constant dense<0.900000> : tensor<128xf32>
    %laKc = stablehlo.constant dense<0.000100> : tensor<128xf32>
    %llt1 = stablehlo.multiply %lomac, %lt1s : tensor<128xf32>
    %llt2 = stablehlo.multiply %laKc, %llsr : tensor<128xf32>
    %llpe = stablehlo.add %llt1, %llt2 : tensor<128xf32>
    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    %lbfc = stablehlo.constant dense<128.0> : tensor<f32>
    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>
    %loss = stablehlo.negate %lossm : tensor<f32>
    return %v4548, %v4557, %v4566, %v4575, %v4584, %v4593, %v4602, %v4611, %v4620, %v4629, %v4638, %v4647, %v4656, %v4665, %v4674, %v4683, %v4692, %v4701, %v4710, %v4719, %v4728, %v4737, %v4746, %v4755, %v4764, %v4773, %v4782, %v4791, %v4800, %v4809, %v4818, %v4827, %v4836, %v4845, %v4854, %v4863, %v4872, %v4881, %v4890, %v4899, %v4908, %v4917, %v4926, %v4935, %v4944, %v4953, %v4962, %v4971, %v4980, %v4989, %v4998, %v5007, %v5016, %v5025, %v5034, %v5043, %v5052, %v5061, %v5070, %v5079, %v5088, %v5097, %v5106, %v5115, %v5124, %v5133, %v5142, %v5151, %v5160, %v5169, %v5178, %v5187, %v5196, %v5205, %v5214, %v5223, %v5232, %v5241, %v5250, %v5259, %v5268, %v5277, %v5286, %v5295, %v5304, %v5313, %v5322, %v5331, %v5340, %v5349, %v5358, %v5367, %v5376, %v5385, %v5394, %v5403, %v5412, %v5421, %v5430, %v5439, %v5448, %v5457, %v5466, %v5475, %v5484, %v5493, %v5502, %v5511, %v5520, %v5529, %sWm, %sgm, %sbtm, %s1b0W1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0g2m, %s1b0bt2m, %s1b1W1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1g2m, %s1b1bt2m, %s1b2W1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2g2m, %s1b2bt2m, %d2W1m, %d2g1m, %d2bt1m, %d2W2m, %d2g2m, %d2bt2m, %d2Wpm, %d2gpm, %d2btpm, %s2b0W1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0g2m, %s2b0bt2m, %s2b1W1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1g2m, %s2b1bt2m, %s2b2W1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2g2m, %s2b2bt2m, %d3W1m, %d3g1m, %d3bt1m, %d3W2m, %d3g2m, %d3bt2m, %d3Wpm, %d3gpm, %d3btpm, %s3b0W1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0g2m, %s3b0bt2m, %s3b1W1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1g2m, %s3b1bt2m, %s3b2W1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2g2m, %s3b2bt2m, %s3b3W1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3g2m, %s3b3bt2m, %s3b4W1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4g2m, %s3b4bt2m, %d4W1m, %d4g1m, %d4bt1m, %d4W2m, %d4g2m, %d4bt2m, %d4Wpm, %d4gpm, %d4btpm, %s4b0W1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0g2m, %s4b0bt2m, %s4b1W1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1g2m, %s4b1bt2m, %Wdm, %bdm, %v4545, %v4554, %v4563, %v4572, %v4581, %v4590, %v4599, %v4608, %v4617, %v4626, %v4635, %v4644, %v4653, %v4662, %v4671, %v4680, %v4689, %v4698, %v4707, %v4716, %v4725, %v4734, %v4743, %v4752, %v4761, %v4770, %v4779, %v4788, %v4797, %v4806, %v4815, %v4824, %v4833, %v4842, %v4851, %v4860, %v4869, %v4878, %v4887, %v4896, %v4905, %v4914, %v4923, %v4932, %v4941, %v4950, %v4959, %v4968, %v4977, %v4986, %v4995, %v5004, %v5013, %v5022, %v5031, %v5040, %v5049, %v5058, %v5067, %v5076, %v5085, %v5094, %v5103, %v5112, %v5121, %v5130, %v5139, %v5148, %v5157, %v5166, %v5175, %v5184, %v5193, %v5202, %v5211, %v5220, %v5229, %v5238, %v5247, %v5256, %v5265, %v5274, %v5283, %v5292, %v5301, %v5310, %v5319, %v5328, %v5337, %v5346, %v5355, %v5364, %v5373, %v5382, %v5391, %v5400, %v5409, %v5418, %v5427, %v5436, %v5445, %v5454, %v5463, %v5472, %v5481, %v5490, %v5499, %v5508, %v5517, %v5526, %loss, %bc1, %bc2, %v4468, %v4469, %v4470, %v4471, %v4472, %v4473, %v4474, %v4475, %v4476, %v4477, %v4478, %v4479, %v4480, %v4481, %v4482, %v4483, %v4484, %v4485, %v4486, %v4487, %v4488, %v4489, %v4490, %v4491, %v4492, %v4493, %v4494, %v4495, %v4496, %v4497, %v4498, %v4499, %v4500, %v4501, %v4502, %v4503, %v4504, %v4505, %v4506, %v4507, %v4508, %v4509, %v4510, %v4511, %v4512, %v4513, %v4514, %v4515, %v4516, %v4517, %v4518, %v4519, %v4520, %v4521, %v4522, %v4523, %v4524, %v4525, %v4526, %v4527, %v4528, %v4529, %v4530, %v4531, %v4532, %v4533, %v4534, %v4535, %v4536, %v4537, %v4538, %v4539 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}

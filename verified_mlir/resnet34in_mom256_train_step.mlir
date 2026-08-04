module @m {
  func.func @resnet34in_mom256_train_step(%x: tensor<256x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x1x1xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x1x1xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x1x1xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x1000xf32>, %bd: tensor<1000xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x3x3xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b1W1m: tensor<64x64x3x3xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b2W1m: tensor<64x64x3x3xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %d2W1m: tensor<128x64x3x3xf32>, %d2g1m: tensor<128xf32>, %d2bt1m: tensor<128xf32>, %d2W2m: tensor<128x128x3x3xf32>, %d2g2m: tensor<128xf32>, %d2bt2m: tensor<128xf32>, %d2Wpm: tensor<128x64x1x1xf32>, %d2gpm: tensor<128xf32>, %d2btpm: tensor<128xf32>, %s2b0W1m: tensor<128x128x3x3xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b1W1m: tensor<128x128x3x3xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b2W1m: tensor<128x128x3x3xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %d3W1m: tensor<256x128x3x3xf32>, %d3g1m: tensor<256xf32>, %d3bt1m: tensor<256xf32>, %d3W2m: tensor<256x256x3x3xf32>, %d3g2m: tensor<256xf32>, %d3bt2m: tensor<256xf32>, %d3Wpm: tensor<256x128x1x1xf32>, %d3gpm: tensor<256xf32>, %d3btpm: tensor<256xf32>, %s3b0W1m: tensor<256x256x3x3xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b1W1m: tensor<256x256x3x3xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b2W1m: tensor<256x256x3x3xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b3W1m: tensor<256x256x3x3xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b4W1m: tensor<256x256x3x3xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %d4W1m: tensor<512x256x3x3xf32>, %d4g1m: tensor<512xf32>, %d4bt1m: tensor<512xf32>, %d4W2m: tensor<512x512x3x3xf32>, %d4g2m: tensor<512xf32>, %d4bt2m: tensor<512xf32>, %d4Wpm: tensor<512x256x1x1xf32>, %d4gpm: tensor<512xf32>, %d4btpm: tensor<512xf32>, %s4b0W1m: tensor<512x512x3x3xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b1W1m: tensor<512x512x3x3xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %Wdm: tensor<512x1000xf32>, %bdm: tensor<1000xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x3x3xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b1W1v: tensor<64x64x3x3xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b2W1v: tensor<64x64x3x3xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %d2W1v: tensor<128x64x3x3xf32>, %d2g1v: tensor<128xf32>, %d2bt1v: tensor<128xf32>, %d2W2v: tensor<128x128x3x3xf32>, %d2g2v: tensor<128xf32>, %d2bt2v: tensor<128xf32>, %d2Wpv: tensor<128x64x1x1xf32>, %d2gpv: tensor<128xf32>, %d2btpv: tensor<128xf32>, %s2b0W1v: tensor<128x128x3x3xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b1W1v: tensor<128x128x3x3xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b2W1v: tensor<128x128x3x3xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %d3W1v: tensor<256x128x3x3xf32>, %d3g1v: tensor<256xf32>, %d3bt1v: tensor<256xf32>, %d3W2v: tensor<256x256x3x3xf32>, %d3g2v: tensor<256xf32>, %d3bt2v: tensor<256xf32>, %d3Wpv: tensor<256x128x1x1xf32>, %d3gpv: tensor<256xf32>, %d3btpv: tensor<256xf32>, %s3b0W1v: tensor<256x256x3x3xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b1W1v: tensor<256x256x3x3xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b2W1v: tensor<256x256x3x3xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b3W1v: tensor<256x256x3x3xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b4W1v: tensor<256x256x3x3xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %d4W1v: tensor<512x256x3x3xf32>, %d4g1v: tensor<512xf32>, %d4bt1v: tensor<512xf32>, %d4W2v: tensor<512x512x3x3xf32>, %d4g2v: tensor<512xf32>, %d4bt2v: tensor<512xf32>, %d4Wpv: tensor<512x256x1x1xf32>, %d4gpv: tensor<512xf32>, %d4btpv: tensor<512xf32>, %s4b0W1v: tensor<512x512x3x3xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b1W1v: tensor<512x512x3x3xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %Wdv: tensor<512x1000xf32>, %bdv: tensor<1000xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %d2n1mui: tensor<128xf32>, %d2n1vari: tensor<128xf32>, %d2n2mui: tensor<128xf32>, %d2n2vari: tensor<128xf32>, %d2npmui: tensor<128xf32>, %d2npvari: tensor<128xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %d3n1mui: tensor<256xf32>, %d3n1vari: tensor<256xf32>, %d3n2mui: tensor<256xf32>, %d3n2vari: tensor<256xf32>, %d3npmui: tensor<256xf32>, %d3npvari: tensor<256xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %d4n1mui: tensor<512xf32>, %d4n1vari: tensor<512xf32>, %d4n2mui: tensor<512xf32>, %d4n2vari: tensor<512xf32>, %d4npmui: tensor<512xf32>, %d4npvari: tensor<512xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %onehot: tensor<256x1000xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) {
    // ── ResNet-34 batch-BN heavy-ball momentum + coupled L2 train step: every line is pretty(verified AST node) ──
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb64 = stablehlo.constant dense<0.0> : tensor<64xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb512 = stablehlo.constant dense<0.0> : tensor<512xf32>
    %v0 = stablehlo.reshape %x : (tensor<256x150528xf32>) -> tensor<256x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<256x64x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<256x64x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-05> : tensor<256x64x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<256x64x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<256x64x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<256x64x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<256x64x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<256x64x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<256x64x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<256x64x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<256x64x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<256x64x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v25 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v26 = stablehlo.maximum %v24, %v25 : tensor<256x802816xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v28 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v29 = "stablehlo.reduce_window"(%v27, %v28) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<256x64x56x56xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v32 = stablehlo.convolution(%v31, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v33 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<256x64x56x56xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<f32>
    %v38 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v39 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v40 = stablehlo.reduce(%v36 init: %v37) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v41 = stablehlo.broadcast_in_dim %v40, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v42 = stablehlo.divide %v41, %v38 : tensor<256x64x56x56xf32>
    %v43 = stablehlo.subtract %v36, %v42 : tensor<256x64x56x56xf32>
    %v44 = stablehlo.multiply %v43, %v43 : tensor<256x64x56x56xf32>
    %v45 = stablehlo.reduce(%v44 init: %v37) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v46 = stablehlo.broadcast_in_dim %v45, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v47 = stablehlo.divide %v46, %v38 : tensor<256x64x56x56xf32>
    %v48 = stablehlo.add %v47, %v39 : tensor<256x64x56x56xf32>
    %v49 = stablehlo.rsqrt %v48 : tensor<256x64x56x56xf32>
    %v50 = stablehlo.multiply %v43, %v49 : tensor<256x64x56x56xf32>
    %v51 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v52 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v53 = stablehlo.multiply %v50, %v51 : tensor<256x64x56x56xf32>
    %v54 = stablehlo.add %v53, %v52 : tensor<256x64x56x56xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v56 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v57 = stablehlo.maximum %v55, %v56 : tensor<256x200704xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v59 = stablehlo.convolution(%v58, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v60 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<256x64x56x56xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v64 = stablehlo.constant dense<0.0> : tensor<f32>
    %v65 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v66 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v67 = stablehlo.reduce(%v63 init: %v64) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v69 = stablehlo.divide %v68, %v65 : tensor<256x64x56x56xf32>
    %v70 = stablehlo.subtract %v63, %v69 : tensor<256x64x56x56xf32>
    %v71 = stablehlo.multiply %v70, %v70 : tensor<256x64x56x56xf32>
    %v72 = stablehlo.reduce(%v71 init: %v64) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v73 = stablehlo.broadcast_in_dim %v72, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v74 = stablehlo.divide %v73, %v65 : tensor<256x64x56x56xf32>
    %v75 = stablehlo.add %v74, %v66 : tensor<256x64x56x56xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<256x64x56x56xf32>
    %v77 = stablehlo.multiply %v70, %v76 : tensor<256x64x56x56xf32>
    %v78 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v79 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v80 = stablehlo.multiply %v77, %v78 : tensor<256x64x56x56xf32>
    %v81 = stablehlo.add %v80, %v79 : tensor<256x64x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v83 = stablehlo.add %v82, %v30 : tensor<256x200704xf32>
    %v84 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v85 = stablehlo.maximum %v83, %v84 : tensor<256x200704xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v87 = stablehlo.convolution(%v86, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v88 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v89 = stablehlo.add %v87, %v88 : tensor<256x64x56x56xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<f32>
    %v93 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v94 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v95 = stablehlo.reduce(%v91 init: %v92) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v96 = stablehlo.broadcast_in_dim %v95, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v97 = stablehlo.divide %v96, %v93 : tensor<256x64x56x56xf32>
    %v98 = stablehlo.subtract %v91, %v97 : tensor<256x64x56x56xf32>
    %v99 = stablehlo.multiply %v98, %v98 : tensor<256x64x56x56xf32>
    %v100 = stablehlo.reduce(%v99 init: %v92) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v101 = stablehlo.broadcast_in_dim %v100, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v102 = stablehlo.divide %v101, %v93 : tensor<256x64x56x56xf32>
    %v103 = stablehlo.add %v102, %v94 : tensor<256x64x56x56xf32>
    %v104 = stablehlo.rsqrt %v103 : tensor<256x64x56x56xf32>
    %v105 = stablehlo.multiply %v98, %v104 : tensor<256x64x56x56xf32>
    %v106 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v107 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v108 = stablehlo.multiply %v105, %v106 : tensor<256x64x56x56xf32>
    %v109 = stablehlo.add %v108, %v107 : tensor<256x64x56x56xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v111 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v112 = stablehlo.maximum %v110, %v111 : tensor<256x200704xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v114 = stablehlo.convolution(%v113, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v115 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v116 = stablehlo.add %v114, %v115 : tensor<256x64x56x56xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v120 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v121 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v122 = stablehlo.reduce(%v118 init: %v119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v123 = stablehlo.broadcast_in_dim %v122, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v124 = stablehlo.divide %v123, %v120 : tensor<256x64x56x56xf32>
    %v125 = stablehlo.subtract %v118, %v124 : tensor<256x64x56x56xf32>
    %v126 = stablehlo.multiply %v125, %v125 : tensor<256x64x56x56xf32>
    %v127 = stablehlo.reduce(%v126 init: %v119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v128 = stablehlo.broadcast_in_dim %v127, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v129 = stablehlo.divide %v128, %v120 : tensor<256x64x56x56xf32>
    %v130 = stablehlo.add %v129, %v121 : tensor<256x64x56x56xf32>
    %v131 = stablehlo.rsqrt %v130 : tensor<256x64x56x56xf32>
    %v132 = stablehlo.multiply %v125, %v131 : tensor<256x64x56x56xf32>
    %v133 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v134 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v135 = stablehlo.multiply %v132, %v133 : tensor<256x64x56x56xf32>
    %v136 = stablehlo.add %v135, %v134 : tensor<256x64x56x56xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v138 = stablehlo.add %v137, %v85 : tensor<256x200704xf32>
    %v139 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v140 = stablehlo.maximum %v138, %v139 : tensor<256x200704xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v142 = stablehlo.convolution(%v141, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v143 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v144 = stablehlo.add %v142, %v143 : tensor<256x64x56x56xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v148 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v149 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v150 = stablehlo.reduce(%v146 init: %v147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v151 = stablehlo.broadcast_in_dim %v150, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v152 = stablehlo.divide %v151, %v148 : tensor<256x64x56x56xf32>
    %v153 = stablehlo.subtract %v146, %v152 : tensor<256x64x56x56xf32>
    %v154 = stablehlo.multiply %v153, %v153 : tensor<256x64x56x56xf32>
    %v155 = stablehlo.reduce(%v154 init: %v147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v156 = stablehlo.broadcast_in_dim %v155, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v157 = stablehlo.divide %v156, %v148 : tensor<256x64x56x56xf32>
    %v158 = stablehlo.add %v157, %v149 : tensor<256x64x56x56xf32>
    %v159 = stablehlo.rsqrt %v158 : tensor<256x64x56x56xf32>
    %v160 = stablehlo.multiply %v153, %v159 : tensor<256x64x56x56xf32>
    %v161 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v162 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v163 = stablehlo.multiply %v160, %v161 : tensor<256x64x56x56xf32>
    %v164 = stablehlo.add %v163, %v162 : tensor<256x64x56x56xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v166 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v167 = stablehlo.maximum %v165, %v166 : tensor<256x200704xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v169 = stablehlo.convolution(%v168, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v170 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v171 = stablehlo.add %v169, %v170 : tensor<256x64x56x56xf32>
    %v172 = stablehlo.reshape %v171 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v174 = stablehlo.constant dense<0.0> : tensor<f32>
    %v175 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v176 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v177 = stablehlo.reduce(%v173 init: %v174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v178 = stablehlo.broadcast_in_dim %v177, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v179 = stablehlo.divide %v178, %v175 : tensor<256x64x56x56xf32>
    %v180 = stablehlo.subtract %v173, %v179 : tensor<256x64x56x56xf32>
    %v181 = stablehlo.multiply %v180, %v180 : tensor<256x64x56x56xf32>
    %v182 = stablehlo.reduce(%v181 init: %v174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v183 = stablehlo.broadcast_in_dim %v182, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v184 = stablehlo.divide %v183, %v175 : tensor<256x64x56x56xf32>
    %v185 = stablehlo.add %v184, %v176 : tensor<256x64x56x56xf32>
    %v186 = stablehlo.rsqrt %v185 : tensor<256x64x56x56xf32>
    %v187 = stablehlo.multiply %v180, %v186 : tensor<256x64x56x56xf32>
    %v188 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v189 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v190 = stablehlo.multiply %v187, %v188 : tensor<256x64x56x56xf32>
    %v191 = stablehlo.add %v190, %v189 : tensor<256x64x56x56xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v193 = stablehlo.add %v192, %v140 : tensor<256x200704xf32>
    %v194 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v195 = stablehlo.maximum %v193, %v194 : tensor<256x200704xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v197 = stablehlo.convolution(%v196, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v198 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v199 = stablehlo.add %v197, %v198 : tensor<256x128x28x28xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v203 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v204 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v205 = stablehlo.reduce(%v201 init: %v202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v206 = stablehlo.broadcast_in_dim %v205, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v207 = stablehlo.divide %v206, %v203 : tensor<256x128x28x28xf32>
    %v208 = stablehlo.subtract %v201, %v207 : tensor<256x128x28x28xf32>
    %v209 = stablehlo.multiply %v208, %v208 : tensor<256x128x28x28xf32>
    %v210 = stablehlo.reduce(%v209 init: %v202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v211 = stablehlo.broadcast_in_dim %v210, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v212 = stablehlo.divide %v211, %v203 : tensor<256x128x28x28xf32>
    %v213 = stablehlo.add %v212, %v204 : tensor<256x128x28x28xf32>
    %v214 = stablehlo.rsqrt %v213 : tensor<256x128x28x28xf32>
    %v215 = stablehlo.multiply %v208, %v214 : tensor<256x128x28x28xf32>
    %v216 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v217 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v218 = stablehlo.multiply %v215, %v216 : tensor<256x128x28x28xf32>
    %v219 = stablehlo.add %v218, %v217 : tensor<256x128x28x28xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v221 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v222 = stablehlo.maximum %v220, %v221 : tensor<256x100352xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v224 = stablehlo.convolution(%v223, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v225 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v226 = stablehlo.add %v224, %v225 : tensor<256x128x28x28xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v230 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v231 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v232 = stablehlo.reduce(%v228 init: %v229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v233 = stablehlo.broadcast_in_dim %v232, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v234 = stablehlo.divide %v233, %v230 : tensor<256x128x28x28xf32>
    %v235 = stablehlo.subtract %v228, %v234 : tensor<256x128x28x28xf32>
    %v236 = stablehlo.multiply %v235, %v235 : tensor<256x128x28x28xf32>
    %v237 = stablehlo.reduce(%v236 init: %v229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v238 = stablehlo.broadcast_in_dim %v237, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v239 = stablehlo.divide %v238, %v230 : tensor<256x128x28x28xf32>
    %v240 = stablehlo.add %v239, %v231 : tensor<256x128x28x28xf32>
    %v241 = stablehlo.rsqrt %v240 : tensor<256x128x28x28xf32>
    %v242 = stablehlo.multiply %v235, %v241 : tensor<256x128x28x28xf32>
    %v243 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v244 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v245 = stablehlo.multiply %v242, %v243 : tensor<256x128x28x28xf32>
    %v246 = stablehlo.add %v245, %v244 : tensor<256x128x28x28xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v248 = stablehlo.reshape %v195 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v249 = stablehlo.convolution(%v248, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<128x64x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v250 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v251 = stablehlo.add %v249, %v250 : tensor<256x128x28x28xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v255 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v256 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v257 = stablehlo.reduce(%v253 init: %v254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v258 = stablehlo.broadcast_in_dim %v257, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v259 = stablehlo.divide %v258, %v255 : tensor<256x128x28x28xf32>
    %v260 = stablehlo.subtract %v253, %v259 : tensor<256x128x28x28xf32>
    %v261 = stablehlo.multiply %v260, %v260 : tensor<256x128x28x28xf32>
    %v262 = stablehlo.reduce(%v261 init: %v254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v263 = stablehlo.broadcast_in_dim %v262, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v264 = stablehlo.divide %v263, %v255 : tensor<256x128x28x28xf32>
    %v265 = stablehlo.add %v264, %v256 : tensor<256x128x28x28xf32>
    %v266 = stablehlo.rsqrt %v265 : tensor<256x128x28x28xf32>
    %v267 = stablehlo.multiply %v260, %v266 : tensor<256x128x28x28xf32>
    %v268 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v269 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v270 = stablehlo.multiply %v267, %v268 : tensor<256x128x28x28xf32>
    %v271 = stablehlo.add %v270, %v269 : tensor<256x128x28x28xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v273 = stablehlo.add %v247, %v272 : tensor<256x100352xf32>
    %v274 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v275 = stablehlo.maximum %v273, %v274 : tensor<256x100352xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v277 = stablehlo.convolution(%v276, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v278 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v279 = stablehlo.add %v277, %v278 : tensor<256x128x28x28xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v282 = stablehlo.constant dense<0.0> : tensor<f32>
    %v283 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v284 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v285 = stablehlo.reduce(%v281 init: %v282) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v286 = stablehlo.broadcast_in_dim %v285, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v287 = stablehlo.divide %v286, %v283 : tensor<256x128x28x28xf32>
    %v288 = stablehlo.subtract %v281, %v287 : tensor<256x128x28x28xf32>
    %v289 = stablehlo.multiply %v288, %v288 : tensor<256x128x28x28xf32>
    %v290 = stablehlo.reduce(%v289 init: %v282) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v291 = stablehlo.broadcast_in_dim %v290, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v292 = stablehlo.divide %v291, %v283 : tensor<256x128x28x28xf32>
    %v293 = stablehlo.add %v292, %v284 : tensor<256x128x28x28xf32>
    %v294 = stablehlo.rsqrt %v293 : tensor<256x128x28x28xf32>
    %v295 = stablehlo.multiply %v288, %v294 : tensor<256x128x28x28xf32>
    %v296 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v297 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v298 = stablehlo.multiply %v295, %v296 : tensor<256x128x28x28xf32>
    %v299 = stablehlo.add %v298, %v297 : tensor<256x128x28x28xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v301 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v302 = stablehlo.maximum %v300, %v301 : tensor<256x100352xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v304 = stablehlo.convolution(%v303, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v305 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v306 = stablehlo.add %v304, %v305 : tensor<256x128x28x28xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v310 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v311 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v312 = stablehlo.reduce(%v308 init: %v309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v313 = stablehlo.broadcast_in_dim %v312, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v314 = stablehlo.divide %v313, %v310 : tensor<256x128x28x28xf32>
    %v315 = stablehlo.subtract %v308, %v314 : tensor<256x128x28x28xf32>
    %v316 = stablehlo.multiply %v315, %v315 : tensor<256x128x28x28xf32>
    %v317 = stablehlo.reduce(%v316 init: %v309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v318 = stablehlo.broadcast_in_dim %v317, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v319 = stablehlo.divide %v318, %v310 : tensor<256x128x28x28xf32>
    %v320 = stablehlo.add %v319, %v311 : tensor<256x128x28x28xf32>
    %v321 = stablehlo.rsqrt %v320 : tensor<256x128x28x28xf32>
    %v322 = stablehlo.multiply %v315, %v321 : tensor<256x128x28x28xf32>
    %v323 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v324 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v325 = stablehlo.multiply %v322, %v323 : tensor<256x128x28x28xf32>
    %v326 = stablehlo.add %v325, %v324 : tensor<256x128x28x28xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v328 = stablehlo.add %v327, %v275 : tensor<256x100352xf32>
    %v329 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v330 = stablehlo.maximum %v328, %v329 : tensor<256x100352xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v332 = stablehlo.convolution(%v331, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v333 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v334 = stablehlo.add %v332, %v333 : tensor<256x128x28x28xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v338 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v339 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v340 = stablehlo.reduce(%v336 init: %v337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v341 = stablehlo.broadcast_in_dim %v340, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v342 = stablehlo.divide %v341, %v338 : tensor<256x128x28x28xf32>
    %v343 = stablehlo.subtract %v336, %v342 : tensor<256x128x28x28xf32>
    %v344 = stablehlo.multiply %v343, %v343 : tensor<256x128x28x28xf32>
    %v345 = stablehlo.reduce(%v344 init: %v337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v346 = stablehlo.broadcast_in_dim %v345, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v347 = stablehlo.divide %v346, %v338 : tensor<256x128x28x28xf32>
    %v348 = stablehlo.add %v347, %v339 : tensor<256x128x28x28xf32>
    %v349 = stablehlo.rsqrt %v348 : tensor<256x128x28x28xf32>
    %v350 = stablehlo.multiply %v343, %v349 : tensor<256x128x28x28xf32>
    %v351 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v352 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v353 = stablehlo.multiply %v350, %v351 : tensor<256x128x28x28xf32>
    %v354 = stablehlo.add %v353, %v352 : tensor<256x128x28x28xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v356 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v357 = stablehlo.maximum %v355, %v356 : tensor<256x100352xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v359 = stablehlo.convolution(%v358, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v360 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v361 = stablehlo.add %v359, %v360 : tensor<256x128x28x28xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v365 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v366 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v367 = stablehlo.reduce(%v363 init: %v364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v368 = stablehlo.broadcast_in_dim %v367, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v369 = stablehlo.divide %v368, %v365 : tensor<256x128x28x28xf32>
    %v370 = stablehlo.subtract %v363, %v369 : tensor<256x128x28x28xf32>
    %v371 = stablehlo.multiply %v370, %v370 : tensor<256x128x28x28xf32>
    %v372 = stablehlo.reduce(%v371 init: %v364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v373 = stablehlo.broadcast_in_dim %v372, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v374 = stablehlo.divide %v373, %v365 : tensor<256x128x28x28xf32>
    %v375 = stablehlo.add %v374, %v366 : tensor<256x128x28x28xf32>
    %v376 = stablehlo.rsqrt %v375 : tensor<256x128x28x28xf32>
    %v377 = stablehlo.multiply %v370, %v376 : tensor<256x128x28x28xf32>
    %v378 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v379 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v380 = stablehlo.multiply %v377, %v378 : tensor<256x128x28x28xf32>
    %v381 = stablehlo.add %v380, %v379 : tensor<256x128x28x28xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v383 = stablehlo.add %v382, %v330 : tensor<256x100352xf32>
    %v384 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v385 = stablehlo.maximum %v383, %v384 : tensor<256x100352xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v387 = stablehlo.convolution(%v386, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v388 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v389 = stablehlo.add %v387, %v388 : tensor<256x128x28x28xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v393 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v394 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v395 = stablehlo.reduce(%v391 init: %v392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v396 = stablehlo.broadcast_in_dim %v395, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v397 = stablehlo.divide %v396, %v393 : tensor<256x128x28x28xf32>
    %v398 = stablehlo.subtract %v391, %v397 : tensor<256x128x28x28xf32>
    %v399 = stablehlo.multiply %v398, %v398 : tensor<256x128x28x28xf32>
    %v400 = stablehlo.reduce(%v399 init: %v392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v401 = stablehlo.broadcast_in_dim %v400, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v402 = stablehlo.divide %v401, %v393 : tensor<256x128x28x28xf32>
    %v403 = stablehlo.add %v402, %v394 : tensor<256x128x28x28xf32>
    %v404 = stablehlo.rsqrt %v403 : tensor<256x128x28x28xf32>
    %v405 = stablehlo.multiply %v398, %v404 : tensor<256x128x28x28xf32>
    %v406 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v407 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v408 = stablehlo.multiply %v405, %v406 : tensor<256x128x28x28xf32>
    %v409 = stablehlo.add %v408, %v407 : tensor<256x128x28x28xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v411 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v412 = stablehlo.maximum %v410, %v411 : tensor<256x100352xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v414 = stablehlo.convolution(%v413, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v415 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v416 = stablehlo.add %v414, %v415 : tensor<256x128x28x28xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v419 = stablehlo.constant dense<0.0> : tensor<f32>
    %v420 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v421 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v422 = stablehlo.reduce(%v418 init: %v419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v423 = stablehlo.broadcast_in_dim %v422, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v424 = stablehlo.divide %v423, %v420 : tensor<256x128x28x28xf32>
    %v425 = stablehlo.subtract %v418, %v424 : tensor<256x128x28x28xf32>
    %v426 = stablehlo.multiply %v425, %v425 : tensor<256x128x28x28xf32>
    %v427 = stablehlo.reduce(%v426 init: %v419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v428 = stablehlo.broadcast_in_dim %v427, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v429 = stablehlo.divide %v428, %v420 : tensor<256x128x28x28xf32>
    %v430 = stablehlo.add %v429, %v421 : tensor<256x128x28x28xf32>
    %v431 = stablehlo.rsqrt %v430 : tensor<256x128x28x28xf32>
    %v432 = stablehlo.multiply %v425, %v431 : tensor<256x128x28x28xf32>
    %v433 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v434 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v435 = stablehlo.multiply %v432, %v433 : tensor<256x128x28x28xf32>
    %v436 = stablehlo.add %v435, %v434 : tensor<256x128x28x28xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v438 = stablehlo.add %v437, %v385 : tensor<256x100352xf32>
    %v439 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v440 = stablehlo.maximum %v438, %v439 : tensor<256x100352xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v442 = stablehlo.convolution(%v441, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v443 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v444 = stablehlo.add %v442, %v443 : tensor<256x256x14x14xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v448 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v449 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v450 = stablehlo.reduce(%v446 init: %v447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v451 = stablehlo.broadcast_in_dim %v450, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v452 = stablehlo.divide %v451, %v448 : tensor<256x256x14x14xf32>
    %v453 = stablehlo.subtract %v446, %v452 : tensor<256x256x14x14xf32>
    %v454 = stablehlo.multiply %v453, %v453 : tensor<256x256x14x14xf32>
    %v455 = stablehlo.reduce(%v454 init: %v447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v456 = stablehlo.broadcast_in_dim %v455, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v457 = stablehlo.divide %v456, %v448 : tensor<256x256x14x14xf32>
    %v458 = stablehlo.add %v457, %v449 : tensor<256x256x14x14xf32>
    %v459 = stablehlo.rsqrt %v458 : tensor<256x256x14x14xf32>
    %v460 = stablehlo.multiply %v453, %v459 : tensor<256x256x14x14xf32>
    %v461 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v462 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v463 = stablehlo.multiply %v460, %v461 : tensor<256x256x14x14xf32>
    %v464 = stablehlo.add %v463, %v462 : tensor<256x256x14x14xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v466 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v467 = stablehlo.maximum %v465, %v466 : tensor<256x50176xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v469 = stablehlo.convolution(%v468, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v470 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v471 = stablehlo.add %v469, %v470 : tensor<256x256x14x14xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v475 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v476 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v477 = stablehlo.reduce(%v473 init: %v474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v478 = stablehlo.broadcast_in_dim %v477, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v479 = stablehlo.divide %v478, %v475 : tensor<256x256x14x14xf32>
    %v480 = stablehlo.subtract %v473, %v479 : tensor<256x256x14x14xf32>
    %v481 = stablehlo.multiply %v480, %v480 : tensor<256x256x14x14xf32>
    %v482 = stablehlo.reduce(%v481 init: %v474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v483 = stablehlo.broadcast_in_dim %v482, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v484 = stablehlo.divide %v483, %v475 : tensor<256x256x14x14xf32>
    %v485 = stablehlo.add %v484, %v476 : tensor<256x256x14x14xf32>
    %v486 = stablehlo.rsqrt %v485 : tensor<256x256x14x14xf32>
    %v487 = stablehlo.multiply %v480, %v486 : tensor<256x256x14x14xf32>
    %v488 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v489 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v490 = stablehlo.multiply %v487, %v488 : tensor<256x256x14x14xf32>
    %v491 = stablehlo.add %v490, %v489 : tensor<256x256x14x14xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v493 = stablehlo.reshape %v440 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v494 = stablehlo.convolution(%v493, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<256x128x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v495 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v496 = stablehlo.add %v494, %v495 : tensor<256x256x14x14xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v499 = stablehlo.constant dense<0.0> : tensor<f32>
    %v500 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v501 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v502 = stablehlo.reduce(%v498 init: %v499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v503 = stablehlo.broadcast_in_dim %v502, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v504 = stablehlo.divide %v503, %v500 : tensor<256x256x14x14xf32>
    %v505 = stablehlo.subtract %v498, %v504 : tensor<256x256x14x14xf32>
    %v506 = stablehlo.multiply %v505, %v505 : tensor<256x256x14x14xf32>
    %v507 = stablehlo.reduce(%v506 init: %v499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v508 = stablehlo.broadcast_in_dim %v507, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v509 = stablehlo.divide %v508, %v500 : tensor<256x256x14x14xf32>
    %v510 = stablehlo.add %v509, %v501 : tensor<256x256x14x14xf32>
    %v511 = stablehlo.rsqrt %v510 : tensor<256x256x14x14xf32>
    %v512 = stablehlo.multiply %v505, %v511 : tensor<256x256x14x14xf32>
    %v513 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v514 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v515 = stablehlo.multiply %v512, %v513 : tensor<256x256x14x14xf32>
    %v516 = stablehlo.add %v515, %v514 : tensor<256x256x14x14xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v518 = stablehlo.add %v492, %v517 : tensor<256x50176xf32>
    %v519 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v520 = stablehlo.maximum %v518, %v519 : tensor<256x50176xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v522 = stablehlo.convolution(%v521, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v523 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v524 = stablehlo.add %v522, %v523 : tensor<256x256x14x14xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v528 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v529 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v530 = stablehlo.reduce(%v526 init: %v527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v531 = stablehlo.broadcast_in_dim %v530, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v532 = stablehlo.divide %v531, %v528 : tensor<256x256x14x14xf32>
    %v533 = stablehlo.subtract %v526, %v532 : tensor<256x256x14x14xf32>
    %v534 = stablehlo.multiply %v533, %v533 : tensor<256x256x14x14xf32>
    %v535 = stablehlo.reduce(%v534 init: %v527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v536 = stablehlo.broadcast_in_dim %v535, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v537 = stablehlo.divide %v536, %v528 : tensor<256x256x14x14xf32>
    %v538 = stablehlo.add %v537, %v529 : tensor<256x256x14x14xf32>
    %v539 = stablehlo.rsqrt %v538 : tensor<256x256x14x14xf32>
    %v540 = stablehlo.multiply %v533, %v539 : tensor<256x256x14x14xf32>
    %v541 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v542 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v543 = stablehlo.multiply %v540, %v541 : tensor<256x256x14x14xf32>
    %v544 = stablehlo.add %v543, %v542 : tensor<256x256x14x14xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v546 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v547 = stablehlo.maximum %v545, %v546 : tensor<256x50176xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v549 = stablehlo.convolution(%v548, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v550 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v551 = stablehlo.add %v549, %v550 : tensor<256x256x14x14xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v555 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v556 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v557 = stablehlo.reduce(%v553 init: %v554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v558 = stablehlo.broadcast_in_dim %v557, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v559 = stablehlo.divide %v558, %v555 : tensor<256x256x14x14xf32>
    %v560 = stablehlo.subtract %v553, %v559 : tensor<256x256x14x14xf32>
    %v561 = stablehlo.multiply %v560, %v560 : tensor<256x256x14x14xf32>
    %v562 = stablehlo.reduce(%v561 init: %v554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v563 = stablehlo.broadcast_in_dim %v562, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v564 = stablehlo.divide %v563, %v555 : tensor<256x256x14x14xf32>
    %v565 = stablehlo.add %v564, %v556 : tensor<256x256x14x14xf32>
    %v566 = stablehlo.rsqrt %v565 : tensor<256x256x14x14xf32>
    %v567 = stablehlo.multiply %v560, %v566 : tensor<256x256x14x14xf32>
    %v568 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v569 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v570 = stablehlo.multiply %v567, %v568 : tensor<256x256x14x14xf32>
    %v571 = stablehlo.add %v570, %v569 : tensor<256x256x14x14xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v573 = stablehlo.add %v572, %v520 : tensor<256x50176xf32>
    %v574 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v575 = stablehlo.maximum %v573, %v574 : tensor<256x50176xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v577 = stablehlo.convolution(%v576, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v578 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v579 = stablehlo.add %v577, %v578 : tensor<256x256x14x14xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v583 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v584 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v585 = stablehlo.reduce(%v581 init: %v582) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v586 = stablehlo.broadcast_in_dim %v585, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v587 = stablehlo.divide %v586, %v583 : tensor<256x256x14x14xf32>
    %v588 = stablehlo.subtract %v581, %v587 : tensor<256x256x14x14xf32>
    %v589 = stablehlo.multiply %v588, %v588 : tensor<256x256x14x14xf32>
    %v590 = stablehlo.reduce(%v589 init: %v582) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v591 = stablehlo.broadcast_in_dim %v590, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v592 = stablehlo.divide %v591, %v583 : tensor<256x256x14x14xf32>
    %v593 = stablehlo.add %v592, %v584 : tensor<256x256x14x14xf32>
    %v594 = stablehlo.rsqrt %v593 : tensor<256x256x14x14xf32>
    %v595 = stablehlo.multiply %v588, %v594 : tensor<256x256x14x14xf32>
    %v596 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v597 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v598 = stablehlo.multiply %v595, %v596 : tensor<256x256x14x14xf32>
    %v599 = stablehlo.add %v598, %v597 : tensor<256x256x14x14xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v601 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v602 = stablehlo.maximum %v600, %v601 : tensor<256x50176xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v604 = stablehlo.convolution(%v603, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v605 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v606 = stablehlo.add %v604, %v605 : tensor<256x256x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v610 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v611 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v612 = stablehlo.reduce(%v608 init: %v609) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v613 = stablehlo.broadcast_in_dim %v612, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v614 = stablehlo.divide %v613, %v610 : tensor<256x256x14x14xf32>
    %v615 = stablehlo.subtract %v608, %v614 : tensor<256x256x14x14xf32>
    %v616 = stablehlo.multiply %v615, %v615 : tensor<256x256x14x14xf32>
    %v617 = stablehlo.reduce(%v616 init: %v609) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v619 = stablehlo.divide %v618, %v610 : tensor<256x256x14x14xf32>
    %v620 = stablehlo.add %v619, %v611 : tensor<256x256x14x14xf32>
    %v621 = stablehlo.rsqrt %v620 : tensor<256x256x14x14xf32>
    %v622 = stablehlo.multiply %v615, %v621 : tensor<256x256x14x14xf32>
    %v623 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v624 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v625 = stablehlo.multiply %v622, %v623 : tensor<256x256x14x14xf32>
    %v626 = stablehlo.add %v625, %v624 : tensor<256x256x14x14xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v628 = stablehlo.add %v627, %v575 : tensor<256x50176xf32>
    %v629 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v630 = stablehlo.maximum %v628, %v629 : tensor<256x50176xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v632 = stablehlo.convolution(%v631, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v633 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v634 = stablehlo.add %v632, %v633 : tensor<256x256x14x14xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v637 = stablehlo.constant dense<0.0> : tensor<f32>
    %v638 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v639 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v640 = stablehlo.reduce(%v636 init: %v637) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v641 = stablehlo.broadcast_in_dim %v640, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v642 = stablehlo.divide %v641, %v638 : tensor<256x256x14x14xf32>
    %v643 = stablehlo.subtract %v636, %v642 : tensor<256x256x14x14xf32>
    %v644 = stablehlo.multiply %v643, %v643 : tensor<256x256x14x14xf32>
    %v645 = stablehlo.reduce(%v644 init: %v637) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v646 = stablehlo.broadcast_in_dim %v645, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v647 = stablehlo.divide %v646, %v638 : tensor<256x256x14x14xf32>
    %v648 = stablehlo.add %v647, %v639 : tensor<256x256x14x14xf32>
    %v649 = stablehlo.rsqrt %v648 : tensor<256x256x14x14xf32>
    %v650 = stablehlo.multiply %v643, %v649 : tensor<256x256x14x14xf32>
    %v651 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v652 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v653 = stablehlo.multiply %v650, %v651 : tensor<256x256x14x14xf32>
    %v654 = stablehlo.add %v653, %v652 : tensor<256x256x14x14xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v656 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v657 = stablehlo.maximum %v655, %v656 : tensor<256x50176xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v659 = stablehlo.convolution(%v658, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v660 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v661 = stablehlo.add %v659, %v660 : tensor<256x256x14x14xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v665 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v666 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v667 = stablehlo.reduce(%v663 init: %v664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v668 = stablehlo.broadcast_in_dim %v667, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v669 = stablehlo.divide %v668, %v665 : tensor<256x256x14x14xf32>
    %v670 = stablehlo.subtract %v663, %v669 : tensor<256x256x14x14xf32>
    %v671 = stablehlo.multiply %v670, %v670 : tensor<256x256x14x14xf32>
    %v672 = stablehlo.reduce(%v671 init: %v664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v673 = stablehlo.broadcast_in_dim %v672, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v674 = stablehlo.divide %v673, %v665 : tensor<256x256x14x14xf32>
    %v675 = stablehlo.add %v674, %v666 : tensor<256x256x14x14xf32>
    %v676 = stablehlo.rsqrt %v675 : tensor<256x256x14x14xf32>
    %v677 = stablehlo.multiply %v670, %v676 : tensor<256x256x14x14xf32>
    %v678 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v679 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v680 = stablehlo.multiply %v677, %v678 : tensor<256x256x14x14xf32>
    %v681 = stablehlo.add %v680, %v679 : tensor<256x256x14x14xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v683 = stablehlo.add %v682, %v630 : tensor<256x50176xf32>
    %v684 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v685 = stablehlo.maximum %v683, %v684 : tensor<256x50176xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v687 = stablehlo.convolution(%v686, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v688 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v689 = stablehlo.add %v687, %v688 : tensor<256x256x14x14xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v693 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v694 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v695 = stablehlo.reduce(%v691 init: %v692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v696 = stablehlo.broadcast_in_dim %v695, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v697 = stablehlo.divide %v696, %v693 : tensor<256x256x14x14xf32>
    %v698 = stablehlo.subtract %v691, %v697 : tensor<256x256x14x14xf32>
    %v699 = stablehlo.multiply %v698, %v698 : tensor<256x256x14x14xf32>
    %v700 = stablehlo.reduce(%v699 init: %v692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v701 = stablehlo.broadcast_in_dim %v700, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v702 = stablehlo.divide %v701, %v693 : tensor<256x256x14x14xf32>
    %v703 = stablehlo.add %v702, %v694 : tensor<256x256x14x14xf32>
    %v704 = stablehlo.rsqrt %v703 : tensor<256x256x14x14xf32>
    %v705 = stablehlo.multiply %v698, %v704 : tensor<256x256x14x14xf32>
    %v706 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v707 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v708 = stablehlo.multiply %v705, %v706 : tensor<256x256x14x14xf32>
    %v709 = stablehlo.add %v708, %v707 : tensor<256x256x14x14xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v711 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v712 = stablehlo.maximum %v710, %v711 : tensor<256x50176xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v714 = stablehlo.convolution(%v713, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v715 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v716 = stablehlo.add %v714, %v715 : tensor<256x256x14x14xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v720 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v721 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v722 = stablehlo.reduce(%v718 init: %v719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v723 = stablehlo.broadcast_in_dim %v722, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v724 = stablehlo.divide %v723, %v720 : tensor<256x256x14x14xf32>
    %v725 = stablehlo.subtract %v718, %v724 : tensor<256x256x14x14xf32>
    %v726 = stablehlo.multiply %v725, %v725 : tensor<256x256x14x14xf32>
    %v727 = stablehlo.reduce(%v726 init: %v719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v728 = stablehlo.broadcast_in_dim %v727, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v729 = stablehlo.divide %v728, %v720 : tensor<256x256x14x14xf32>
    %v730 = stablehlo.add %v729, %v721 : tensor<256x256x14x14xf32>
    %v731 = stablehlo.rsqrt %v730 : tensor<256x256x14x14xf32>
    %v732 = stablehlo.multiply %v725, %v731 : tensor<256x256x14x14xf32>
    %v733 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v735 = stablehlo.multiply %v732, %v733 : tensor<256x256x14x14xf32>
    %v736 = stablehlo.add %v735, %v734 : tensor<256x256x14x14xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v738 = stablehlo.add %v737, %v685 : tensor<256x50176xf32>
    %v739 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v740 = stablehlo.maximum %v738, %v739 : tensor<256x50176xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v742 = stablehlo.convolution(%v741, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v743 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v744 = stablehlo.add %v742, %v743 : tensor<256x256x14x14xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v748 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v749 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v750 = stablehlo.reduce(%v746 init: %v747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v751 = stablehlo.broadcast_in_dim %v750, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v752 = stablehlo.divide %v751, %v748 : tensor<256x256x14x14xf32>
    %v753 = stablehlo.subtract %v746, %v752 : tensor<256x256x14x14xf32>
    %v754 = stablehlo.multiply %v753, %v753 : tensor<256x256x14x14xf32>
    %v755 = stablehlo.reduce(%v754 init: %v747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v756 = stablehlo.broadcast_in_dim %v755, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v757 = stablehlo.divide %v756, %v748 : tensor<256x256x14x14xf32>
    %v758 = stablehlo.add %v757, %v749 : tensor<256x256x14x14xf32>
    %v759 = stablehlo.rsqrt %v758 : tensor<256x256x14x14xf32>
    %v760 = stablehlo.multiply %v753, %v759 : tensor<256x256x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v762 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v763 = stablehlo.multiply %v760, %v761 : tensor<256x256x14x14xf32>
    %v764 = stablehlo.add %v763, %v762 : tensor<256x256x14x14xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v766 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v767 = stablehlo.maximum %v765, %v766 : tensor<256x50176xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v769 = stablehlo.convolution(%v768, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v770 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v771 = stablehlo.add %v769, %v770 : tensor<256x256x14x14xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v775 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v776 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v777 = stablehlo.reduce(%v773 init: %v774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v778 = stablehlo.broadcast_in_dim %v777, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v779 = stablehlo.divide %v778, %v775 : tensor<256x256x14x14xf32>
    %v780 = stablehlo.subtract %v773, %v779 : tensor<256x256x14x14xf32>
    %v781 = stablehlo.multiply %v780, %v780 : tensor<256x256x14x14xf32>
    %v782 = stablehlo.reduce(%v781 init: %v774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v783 = stablehlo.broadcast_in_dim %v782, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v784 = stablehlo.divide %v783, %v775 : tensor<256x256x14x14xf32>
    %v785 = stablehlo.add %v784, %v776 : tensor<256x256x14x14xf32>
    %v786 = stablehlo.rsqrt %v785 : tensor<256x256x14x14xf32>
    %v787 = stablehlo.multiply %v780, %v786 : tensor<256x256x14x14xf32>
    %v788 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v789 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v790 = stablehlo.multiply %v787, %v788 : tensor<256x256x14x14xf32>
    %v791 = stablehlo.add %v790, %v789 : tensor<256x256x14x14xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v793 = stablehlo.add %v792, %v740 : tensor<256x50176xf32>
    %v794 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v795 = stablehlo.maximum %v793, %v794 : tensor<256x50176xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v797 = stablehlo.convolution(%v796, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v798 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v799 = stablehlo.add %v797, %v798 : tensor<256x512x7x7xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v803 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v804 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v805 = stablehlo.reduce(%v801 init: %v802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v806 = stablehlo.broadcast_in_dim %v805, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v807 = stablehlo.divide %v806, %v803 : tensor<256x512x7x7xf32>
    %v808 = stablehlo.subtract %v801, %v807 : tensor<256x512x7x7xf32>
    %v809 = stablehlo.multiply %v808, %v808 : tensor<256x512x7x7xf32>
    %v810 = stablehlo.reduce(%v809 init: %v802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v811 = stablehlo.broadcast_in_dim %v810, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v812 = stablehlo.divide %v811, %v803 : tensor<256x512x7x7xf32>
    %v813 = stablehlo.add %v812, %v804 : tensor<256x512x7x7xf32>
    %v814 = stablehlo.rsqrt %v813 : tensor<256x512x7x7xf32>
    %v815 = stablehlo.multiply %v808, %v814 : tensor<256x512x7x7xf32>
    %v816 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v817 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v818 = stablehlo.multiply %v815, %v816 : tensor<256x512x7x7xf32>
    %v819 = stablehlo.add %v818, %v817 : tensor<256x512x7x7xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v821 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v822 = stablehlo.maximum %v820, %v821 : tensor<256x25088xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v824 = stablehlo.convolution(%v823, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v825 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v826 = stablehlo.add %v824, %v825 : tensor<256x512x7x7xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v830 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v831 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v832 = stablehlo.reduce(%v828 init: %v829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v833 = stablehlo.broadcast_in_dim %v832, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v834 = stablehlo.divide %v833, %v830 : tensor<256x512x7x7xf32>
    %v835 = stablehlo.subtract %v828, %v834 : tensor<256x512x7x7xf32>
    %v836 = stablehlo.multiply %v835, %v835 : tensor<256x512x7x7xf32>
    %v837 = stablehlo.reduce(%v836 init: %v829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v838 = stablehlo.broadcast_in_dim %v837, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v839 = stablehlo.divide %v838, %v830 : tensor<256x512x7x7xf32>
    %v840 = stablehlo.add %v839, %v831 : tensor<256x512x7x7xf32>
    %v841 = stablehlo.rsqrt %v840 : tensor<256x512x7x7xf32>
    %v842 = stablehlo.multiply %v835, %v841 : tensor<256x512x7x7xf32>
    %v843 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v844 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v845 = stablehlo.multiply %v842, %v843 : tensor<256x512x7x7xf32>
    %v846 = stablehlo.add %v845, %v844 : tensor<256x512x7x7xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v848 = stablehlo.reshape %v795 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v849 = stablehlo.convolution(%v848, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<512x256x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v850 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v851 = stablehlo.add %v849, %v850 : tensor<256x512x7x7xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v855 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v856 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v857 = stablehlo.reduce(%v853 init: %v854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v858 = stablehlo.broadcast_in_dim %v857, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v859 = stablehlo.divide %v858, %v855 : tensor<256x512x7x7xf32>
    %v860 = stablehlo.subtract %v853, %v859 : tensor<256x512x7x7xf32>
    %v861 = stablehlo.multiply %v860, %v860 : tensor<256x512x7x7xf32>
    %v862 = stablehlo.reduce(%v861 init: %v854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v863 = stablehlo.broadcast_in_dim %v862, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v864 = stablehlo.divide %v863, %v855 : tensor<256x512x7x7xf32>
    %v865 = stablehlo.add %v864, %v856 : tensor<256x512x7x7xf32>
    %v866 = stablehlo.rsqrt %v865 : tensor<256x512x7x7xf32>
    %v867 = stablehlo.multiply %v860, %v866 : tensor<256x512x7x7xf32>
    %v868 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v869 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v870 = stablehlo.multiply %v867, %v868 : tensor<256x512x7x7xf32>
    %v871 = stablehlo.add %v870, %v869 : tensor<256x512x7x7xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v873 = stablehlo.add %v847, %v872 : tensor<256x25088xf32>
    %v874 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v875 = stablehlo.maximum %v873, %v874 : tensor<256x25088xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v877 = stablehlo.convolution(%v876, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v878 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v879 = stablehlo.add %v877, %v878 : tensor<256x512x7x7xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v883 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v884 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v885 = stablehlo.reduce(%v881 init: %v882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v886 = stablehlo.broadcast_in_dim %v885, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v887 = stablehlo.divide %v886, %v883 : tensor<256x512x7x7xf32>
    %v888 = stablehlo.subtract %v881, %v887 : tensor<256x512x7x7xf32>
    %v889 = stablehlo.multiply %v888, %v888 : tensor<256x512x7x7xf32>
    %v890 = stablehlo.reduce(%v889 init: %v882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v891 = stablehlo.broadcast_in_dim %v890, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v892 = stablehlo.divide %v891, %v883 : tensor<256x512x7x7xf32>
    %v893 = stablehlo.add %v892, %v884 : tensor<256x512x7x7xf32>
    %v894 = stablehlo.rsqrt %v893 : tensor<256x512x7x7xf32>
    %v895 = stablehlo.multiply %v888, %v894 : tensor<256x512x7x7xf32>
    %v896 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v897 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v898 = stablehlo.multiply %v895, %v896 : tensor<256x512x7x7xf32>
    %v899 = stablehlo.add %v898, %v897 : tensor<256x512x7x7xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v901 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v902 = stablehlo.maximum %v900, %v901 : tensor<256x25088xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v904 = stablehlo.convolution(%v903, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v905 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v906 = stablehlo.add %v904, %v905 : tensor<256x512x7x7xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v910 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v911 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v912 = stablehlo.reduce(%v908 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v913 = stablehlo.broadcast_in_dim %v912, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v914 = stablehlo.divide %v913, %v910 : tensor<256x512x7x7xf32>
    %v915 = stablehlo.subtract %v908, %v914 : tensor<256x512x7x7xf32>
    %v916 = stablehlo.multiply %v915, %v915 : tensor<256x512x7x7xf32>
    %v917 = stablehlo.reduce(%v916 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v918 = stablehlo.broadcast_in_dim %v917, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v919 = stablehlo.divide %v918, %v910 : tensor<256x512x7x7xf32>
    %v920 = stablehlo.add %v919, %v911 : tensor<256x512x7x7xf32>
    %v921 = stablehlo.rsqrt %v920 : tensor<256x512x7x7xf32>
    %v922 = stablehlo.multiply %v915, %v921 : tensor<256x512x7x7xf32>
    %v923 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v924 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v925 = stablehlo.multiply %v922, %v923 : tensor<256x512x7x7xf32>
    %v926 = stablehlo.add %v925, %v924 : tensor<256x512x7x7xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v928 = stablehlo.add %v927, %v875 : tensor<256x25088xf32>
    %v929 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v930 = stablehlo.maximum %v928, %v929 : tensor<256x25088xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v932 = stablehlo.convolution(%v931, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v933 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v934 = stablehlo.add %v932, %v933 : tensor<256x512x7x7xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v938 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v939 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v940 = stablehlo.reduce(%v936 init: %v937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v941 = stablehlo.broadcast_in_dim %v940, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v942 = stablehlo.divide %v941, %v938 : tensor<256x512x7x7xf32>
    %v943 = stablehlo.subtract %v936, %v942 : tensor<256x512x7x7xf32>
    %v944 = stablehlo.multiply %v943, %v943 : tensor<256x512x7x7xf32>
    %v945 = stablehlo.reduce(%v944 init: %v937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v946 = stablehlo.broadcast_in_dim %v945, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v947 = stablehlo.divide %v946, %v938 : tensor<256x512x7x7xf32>
    %v948 = stablehlo.add %v947, %v939 : tensor<256x512x7x7xf32>
    %v949 = stablehlo.rsqrt %v948 : tensor<256x512x7x7xf32>
    %v950 = stablehlo.multiply %v943, %v949 : tensor<256x512x7x7xf32>
    %v951 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v952 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v953 = stablehlo.multiply %v950, %v951 : tensor<256x512x7x7xf32>
    %v954 = stablehlo.add %v953, %v952 : tensor<256x512x7x7xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v956 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v957 = stablehlo.maximum %v955, %v956 : tensor<256x25088xf32>
    %v958 = stablehlo.reshape %v957 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v959 = stablehlo.convolution(%v958, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v960 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v961 = stablehlo.add %v959, %v960 : tensor<256x512x7x7xf32>
    %v962 = stablehlo.reshape %v961 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v965 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v966 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v967 = stablehlo.reduce(%v963 init: %v964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v968 = stablehlo.broadcast_in_dim %v967, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v969 = stablehlo.divide %v968, %v965 : tensor<256x512x7x7xf32>
    %v970 = stablehlo.subtract %v963, %v969 : tensor<256x512x7x7xf32>
    %v971 = stablehlo.multiply %v970, %v970 : tensor<256x512x7x7xf32>
    %v972 = stablehlo.reduce(%v971 init: %v964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v973 = stablehlo.broadcast_in_dim %v972, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v974 = stablehlo.divide %v973, %v965 : tensor<256x512x7x7xf32>
    %v975 = stablehlo.add %v974, %v966 : tensor<256x512x7x7xf32>
    %v976 = stablehlo.rsqrt %v975 : tensor<256x512x7x7xf32>
    %v977 = stablehlo.multiply %v970, %v976 : tensor<256x512x7x7xf32>
    %v978 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v979 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v980 = stablehlo.multiply %v977, %v978 : tensor<256x512x7x7xf32>
    %v981 = stablehlo.add %v980, %v979 : tensor<256x512x7x7xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v983 = stablehlo.add %v982, %v930 : tensor<256x25088xf32>
    %v984 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v985 = stablehlo.maximum %v983, %v984 : tensor<256x25088xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v987 = stablehlo.constant dense<0.0> : tensor<f32>
    %v988 = stablehlo.reduce(%v986 init: %v987) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v989 = stablehlo.constant dense<49.0> : tensor<256x512xf32>
    %v990 = stablehlo.divide %v988, %v989 : tensor<256x512xf32>
    %v991 = stablehlo.dot_general %v990, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x512xf32>, tensor<512x1000xf32>) -> tensor<256x1000xf32>
    %v992 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
    %v993 = stablehlo.add %v991, %v992 : tensor<256x1000xf32>
    %v994 = stablehlo.reshape %v993 : (tensor<256x1000xf32>) -> tensor<256x1x1000xf32>
    %v995 = stablehlo.constant dense<0.0> : tensor<f32>
    %v996 = stablehlo.exponential %v994 : tensor<256x1x1000xf32>
    %v997 = stablehlo.reduce(%v996 init: %v995) applies stablehlo.add across dimensions = [2] : (tensor<256x1x1000xf32>, tensor<f32>) -> tensor<256x1xf32>
    %v998 = stablehlo.broadcast_in_dim %v997, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x1x1000xf32>
    %v999 = stablehlo.divide %v996, %v998 : tensor<256x1x1000xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<256x1x1000xf32>) -> tensor<256x1000xf32>
    %v1001 = stablehlo.subtract %v1000, %onehot : tensor<256x1000xf32>
    %v1002 = stablehlo.constant dense<0.100000> : tensor<256x1000xf32>
    %v1003 = stablehlo.multiply %onehot, %v1002 : tensor<256x1000xf32>
    %v1004 = stablehlo.add %v1001, %v1003 : tensor<256x1000xf32>
    %v1005 = stablehlo.constant dense<-0.000100> : tensor<256x1000xf32>
    %v1006 = stablehlo.add %v1004, %v1005 : tensor<256x1000xf32>
    %v1007 = stablehlo.constant dense<256.0> : tensor<256x1000xf32>
    %v1008 = stablehlo.divide %v1006, %v1007 : tensor<256x1000xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<256x1000xf32>) -> tensor<256x1x1000xf32>
    %v1010 = stablehlo.dot_general %v1009, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x1x1000xf32>, tensor<512x1000xf32>) -> tensor<256x1x512xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<256x1x512xf32>) -> tensor<256x512xf32>
    %v1012 = stablehlo.dot_general %v990, %v1008, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x512xf32>, tensor<256x1000xf32>) -> tensor<512x1000xf32>
    %v1013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1014 = stablehlo.reduce(%v1008 init: %v1013) applies stablehlo.add across dimensions = [0] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<1000xf32>
    %v1015 = stablehlo.broadcast_in_dim %v1011, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v1016 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v1017 = stablehlo.divide %v1015, %v1016 : tensor<256x512x7x7xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1019 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1020 = stablehlo.compare GT, %v983, %v1019 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1021 = stablehlo.select %v1020, %v1018, %v1019 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1022 = stablehlo.reshape %v962 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1024 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1025 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1026 = stablehlo.reduce(%v1022 init: %v1023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1027 = stablehlo.broadcast_in_dim %v1026, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1028 = stablehlo.divide %v1027, %v1024 : tensor<256x512x7x7xf32>
    %v1029 = stablehlo.subtract %v1022, %v1028 : tensor<256x512x7x7xf32>
    %v1030 = stablehlo.multiply %v1029, %v1029 : tensor<256x512x7x7xf32>
    %v1031 = stablehlo.reduce(%v1030 init: %v1023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1032 = stablehlo.broadcast_in_dim %v1031, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1033 = stablehlo.divide %v1032, %v1024 : tensor<256x512x7x7xf32>
    %v1034 = stablehlo.add %v1033, %v1025 : tensor<256x512x7x7xf32>
    %v1035 = stablehlo.rsqrt %v1034 : tensor<256x512x7x7xf32>
    %v1036 = stablehlo.multiply %v1029, %v1035 : tensor<256x512x7x7xf32>
    %v1037 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1038 = stablehlo.reshape %v1021 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1039 = stablehlo.multiply %v1037, %v1038 : tensor<256x512x7x7xf32>
    %v1040 = stablehlo.reduce(%v1039 init: %v1023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1041 = stablehlo.broadcast_in_dim %v1040, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1042 = stablehlo.multiply %v1036, %v1039 : tensor<256x512x7x7xf32>
    %v1043 = stablehlo.reduce(%v1042 init: %v1023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1044 = stablehlo.broadcast_in_dim %v1043, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1045 = stablehlo.multiply %v1039, %v1024 : tensor<256x512x7x7xf32>
    %v1046 = stablehlo.subtract %v1045, %v1041 : tensor<256x512x7x7xf32>
    %v1047 = stablehlo.multiply %v1036, %v1044 : tensor<256x512x7x7xf32>
    %v1048 = stablehlo.subtract %v1046, %v1047 : tensor<256x512x7x7xf32>
    %v1049 = stablehlo.divide %v1035, %v1024 : tensor<256x512x7x7xf32>
    %v1050 = stablehlo.multiply %v1049, %v1048 : tensor<256x512x7x7xf32>
    %v1051 = stablehlo.reshape %v1050 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1053 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1054 = stablehlo.transpose %v1053, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1055 = stablehlo.convolution(%v1052, %v1054)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1057 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1058 = stablehlo.compare GT, %v955, %v1057 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1059 = stablehlo.select %v1058, %v1056, %v1057 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1060 = stablehlo.reshape %v935 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1062 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1063 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1064 = stablehlo.reduce(%v1060 init: %v1061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1065 = stablehlo.broadcast_in_dim %v1064, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1066 = stablehlo.divide %v1065, %v1062 : tensor<256x512x7x7xf32>
    %v1067 = stablehlo.subtract %v1060, %v1066 : tensor<256x512x7x7xf32>
    %v1068 = stablehlo.multiply %v1067, %v1067 : tensor<256x512x7x7xf32>
    %v1069 = stablehlo.reduce(%v1068 init: %v1061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1070 = stablehlo.broadcast_in_dim %v1069, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1071 = stablehlo.divide %v1070, %v1062 : tensor<256x512x7x7xf32>
    %v1072 = stablehlo.add %v1071, %v1063 : tensor<256x512x7x7xf32>
    %v1073 = stablehlo.rsqrt %v1072 : tensor<256x512x7x7xf32>
    %v1074 = stablehlo.multiply %v1067, %v1073 : tensor<256x512x7x7xf32>
    %v1075 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1076 = stablehlo.reshape %v1059 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1077 = stablehlo.multiply %v1075, %v1076 : tensor<256x512x7x7xf32>
    %v1078 = stablehlo.reduce(%v1077 init: %v1061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1079 = stablehlo.broadcast_in_dim %v1078, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1080 = stablehlo.multiply %v1074, %v1077 : tensor<256x512x7x7xf32>
    %v1081 = stablehlo.reduce(%v1080 init: %v1061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1082 = stablehlo.broadcast_in_dim %v1081, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1083 = stablehlo.multiply %v1077, %v1062 : tensor<256x512x7x7xf32>
    %v1084 = stablehlo.subtract %v1083, %v1079 : tensor<256x512x7x7xf32>
    %v1085 = stablehlo.multiply %v1074, %v1082 : tensor<256x512x7x7xf32>
    %v1086 = stablehlo.subtract %v1084, %v1085 : tensor<256x512x7x7xf32>
    %v1087 = stablehlo.divide %v1073, %v1062 : tensor<256x512x7x7xf32>
    %v1088 = stablehlo.multiply %v1087, %v1086 : tensor<256x512x7x7xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1091 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1092 = stablehlo.transpose %v1091, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1093 = stablehlo.convolution(%v1090, %v1092)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1095 = stablehlo.add %v1094, %v1021 : tensor<256x25088xf32>
    %v1096 = stablehlo.reshape %v930 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1097 = stablehlo.reshape %v1089 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1098 = stablehlo.transpose %v1096, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1099 = stablehlo.transpose %v1097, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1100 = stablehlo.convolution(%v1098, %v1099)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1101 = stablehlo.transpose %v1100, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1102 = stablehlo.reshape %v935 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1104 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1105 = stablehlo.reduce(%v1102 init: %v1103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1106 = stablehlo.broadcast_in_dim %v1105, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1107 = stablehlo.divide %v1106, %v1104 : tensor<256x512x7x7xf32>
    %v1108 = stablehlo.subtract %v1102, %v1107 : tensor<256x512x7x7xf32>
    %v1109 = stablehlo.multiply %v1108, %v1108 : tensor<256x512x7x7xf32>
    %v1110 = stablehlo.reduce(%v1109 init: %v1103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1111 = stablehlo.broadcast_in_dim %v1110, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1112 = stablehlo.divide %v1111, %v1104 : tensor<256x512x7x7xf32>
    %v1113 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1114 = stablehlo.add %v1112, %v1113 : tensor<256x512x7x7xf32>
    %v1115 = stablehlo.rsqrt %v1114 : tensor<256x512x7x7xf32>
    %v1116 = stablehlo.multiply %v1108, %v1115 : tensor<256x512x7x7xf32>
    %v1117 = stablehlo.reshape %v1059 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1118 = stablehlo.multiply %v1117, %v1116 : tensor<256x512x7x7xf32>
    %v1119 = stablehlo.reduce(%v1118 init: %v1103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1120 = stablehlo.reshape %v1059 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1122 = stablehlo.reduce(%v1120 init: %v1121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1123 = stablehlo.reshape %v957 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1124 = stablehlo.reshape %v1051 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1125 = stablehlo.transpose %v1123, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1126 = stablehlo.transpose %v1124, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1127 = stablehlo.convolution(%v1125, %v1126)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1128 = stablehlo.transpose %v1127, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1129 = stablehlo.reshape %v962 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1131 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1132 = stablehlo.reduce(%v1129 init: %v1130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1133 = stablehlo.broadcast_in_dim %v1132, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1134 = stablehlo.divide %v1133, %v1131 : tensor<256x512x7x7xf32>
    %v1135 = stablehlo.subtract %v1129, %v1134 : tensor<256x512x7x7xf32>
    %v1136 = stablehlo.multiply %v1135, %v1135 : tensor<256x512x7x7xf32>
    %v1137 = stablehlo.reduce(%v1136 init: %v1130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1138 = stablehlo.broadcast_in_dim %v1137, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1139 = stablehlo.divide %v1138, %v1131 : tensor<256x512x7x7xf32>
    %v1140 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1141 = stablehlo.add %v1139, %v1140 : tensor<256x512x7x7xf32>
    %v1142 = stablehlo.rsqrt %v1141 : tensor<256x512x7x7xf32>
    %v1143 = stablehlo.multiply %v1135, %v1142 : tensor<256x512x7x7xf32>
    %v1144 = stablehlo.reshape %v1021 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1145 = stablehlo.multiply %v1144, %v1143 : tensor<256x512x7x7xf32>
    %v1146 = stablehlo.reduce(%v1145 init: %v1130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1147 = stablehlo.reshape %v1021 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1148 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1149 = stablehlo.reduce(%v1147 init: %v1148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1150 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1151 = stablehlo.compare GT, %v928, %v1150 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1152 = stablehlo.select %v1151, %v1095, %v1150 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1153 = stablehlo.reshape %v907 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1155 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1156 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1157 = stablehlo.reduce(%v1153 init: %v1154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1158 = stablehlo.broadcast_in_dim %v1157, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1159 = stablehlo.divide %v1158, %v1155 : tensor<256x512x7x7xf32>
    %v1160 = stablehlo.subtract %v1153, %v1159 : tensor<256x512x7x7xf32>
    %v1161 = stablehlo.multiply %v1160, %v1160 : tensor<256x512x7x7xf32>
    %v1162 = stablehlo.reduce(%v1161 init: %v1154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1163 = stablehlo.broadcast_in_dim %v1162, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1164 = stablehlo.divide %v1163, %v1155 : tensor<256x512x7x7xf32>
    %v1165 = stablehlo.add %v1164, %v1156 : tensor<256x512x7x7xf32>
    %v1166 = stablehlo.rsqrt %v1165 : tensor<256x512x7x7xf32>
    %v1167 = stablehlo.multiply %v1160, %v1166 : tensor<256x512x7x7xf32>
    %v1168 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1169 = stablehlo.reshape %v1152 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1170 = stablehlo.multiply %v1168, %v1169 : tensor<256x512x7x7xf32>
    %v1171 = stablehlo.reduce(%v1170 init: %v1154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1172 = stablehlo.broadcast_in_dim %v1171, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1173 = stablehlo.multiply %v1167, %v1170 : tensor<256x512x7x7xf32>
    %v1174 = stablehlo.reduce(%v1173 init: %v1154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1175 = stablehlo.broadcast_in_dim %v1174, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1176 = stablehlo.multiply %v1170, %v1155 : tensor<256x512x7x7xf32>
    %v1177 = stablehlo.subtract %v1176, %v1172 : tensor<256x512x7x7xf32>
    %v1178 = stablehlo.multiply %v1167, %v1175 : tensor<256x512x7x7xf32>
    %v1179 = stablehlo.subtract %v1177, %v1178 : tensor<256x512x7x7xf32>
    %v1180 = stablehlo.divide %v1166, %v1155 : tensor<256x512x7x7xf32>
    %v1181 = stablehlo.multiply %v1180, %v1179 : tensor<256x512x7x7xf32>
    %v1182 = stablehlo.reshape %v1181 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1184 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1185 = stablehlo.transpose %v1184, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1186 = stablehlo.convolution(%v1183, %v1185)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1187 = stablehlo.reshape %v1186 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1188 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1189 = stablehlo.compare GT, %v900, %v1188 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1190 = stablehlo.select %v1189, %v1187, %v1188 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1191 = stablehlo.reshape %v880 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1193 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1194 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1195 = stablehlo.reduce(%v1191 init: %v1192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1196 = stablehlo.broadcast_in_dim %v1195, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1197 = stablehlo.divide %v1196, %v1193 : tensor<256x512x7x7xf32>
    %v1198 = stablehlo.subtract %v1191, %v1197 : tensor<256x512x7x7xf32>
    %v1199 = stablehlo.multiply %v1198, %v1198 : tensor<256x512x7x7xf32>
    %v1200 = stablehlo.reduce(%v1199 init: %v1192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1201 = stablehlo.broadcast_in_dim %v1200, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1202 = stablehlo.divide %v1201, %v1193 : tensor<256x512x7x7xf32>
    %v1203 = stablehlo.add %v1202, %v1194 : tensor<256x512x7x7xf32>
    %v1204 = stablehlo.rsqrt %v1203 : tensor<256x512x7x7xf32>
    %v1205 = stablehlo.multiply %v1198, %v1204 : tensor<256x512x7x7xf32>
    %v1206 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1207 = stablehlo.reshape %v1190 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1208 = stablehlo.multiply %v1206, %v1207 : tensor<256x512x7x7xf32>
    %v1209 = stablehlo.reduce(%v1208 init: %v1192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1210 = stablehlo.broadcast_in_dim %v1209, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1211 = stablehlo.multiply %v1205, %v1208 : tensor<256x512x7x7xf32>
    %v1212 = stablehlo.reduce(%v1211 init: %v1192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1213 = stablehlo.broadcast_in_dim %v1212, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1214 = stablehlo.multiply %v1208, %v1193 : tensor<256x512x7x7xf32>
    %v1215 = stablehlo.subtract %v1214, %v1210 : tensor<256x512x7x7xf32>
    %v1216 = stablehlo.multiply %v1205, %v1213 : tensor<256x512x7x7xf32>
    %v1217 = stablehlo.subtract %v1215, %v1216 : tensor<256x512x7x7xf32>
    %v1218 = stablehlo.divide %v1204, %v1193 : tensor<256x512x7x7xf32>
    %v1219 = stablehlo.multiply %v1218, %v1217 : tensor<256x512x7x7xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1222 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1223 = stablehlo.transpose %v1222, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1224 = stablehlo.convolution(%v1221, %v1223)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1226 = stablehlo.add %v1225, %v1152 : tensor<256x25088xf32>
    %v1227 = stablehlo.reshape %v875 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1228 = stablehlo.reshape %v1220 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1229 = stablehlo.transpose %v1227, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1230 = stablehlo.transpose %v1228, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1231 = stablehlo.convolution(%v1229, %v1230)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1232 = stablehlo.transpose %v1231, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1233 = stablehlo.reshape %v880 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1234 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1235 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1236 = stablehlo.reduce(%v1233 init: %v1234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1237 = stablehlo.broadcast_in_dim %v1236, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1238 = stablehlo.divide %v1237, %v1235 : tensor<256x512x7x7xf32>
    %v1239 = stablehlo.subtract %v1233, %v1238 : tensor<256x512x7x7xf32>
    %v1240 = stablehlo.multiply %v1239, %v1239 : tensor<256x512x7x7xf32>
    %v1241 = stablehlo.reduce(%v1240 init: %v1234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1242 = stablehlo.broadcast_in_dim %v1241, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1243 = stablehlo.divide %v1242, %v1235 : tensor<256x512x7x7xf32>
    %v1244 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1245 = stablehlo.add %v1243, %v1244 : tensor<256x512x7x7xf32>
    %v1246 = stablehlo.rsqrt %v1245 : tensor<256x512x7x7xf32>
    %v1247 = stablehlo.multiply %v1239, %v1246 : tensor<256x512x7x7xf32>
    %v1248 = stablehlo.reshape %v1190 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1249 = stablehlo.multiply %v1248, %v1247 : tensor<256x512x7x7xf32>
    %v1250 = stablehlo.reduce(%v1249 init: %v1234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1251 = stablehlo.reshape %v1190 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1252 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1253 = stablehlo.reduce(%v1251 init: %v1252) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1254 = stablehlo.reshape %v902 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1255 = stablehlo.reshape %v1182 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1256 = stablehlo.transpose %v1254, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1257 = stablehlo.transpose %v1255, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1258 = stablehlo.convolution(%v1256, %v1257)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1259 = stablehlo.transpose %v1258, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1260 = stablehlo.reshape %v907 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1262 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1263 = stablehlo.reduce(%v1260 init: %v1261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1264 = stablehlo.broadcast_in_dim %v1263, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1265 = stablehlo.divide %v1264, %v1262 : tensor<256x512x7x7xf32>
    %v1266 = stablehlo.subtract %v1260, %v1265 : tensor<256x512x7x7xf32>
    %v1267 = stablehlo.multiply %v1266, %v1266 : tensor<256x512x7x7xf32>
    %v1268 = stablehlo.reduce(%v1267 init: %v1261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1269 = stablehlo.broadcast_in_dim %v1268, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1270 = stablehlo.divide %v1269, %v1262 : tensor<256x512x7x7xf32>
    %v1271 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1272 = stablehlo.add %v1270, %v1271 : tensor<256x512x7x7xf32>
    %v1273 = stablehlo.rsqrt %v1272 : tensor<256x512x7x7xf32>
    %v1274 = stablehlo.multiply %v1266, %v1273 : tensor<256x512x7x7xf32>
    %v1275 = stablehlo.reshape %v1152 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1276 = stablehlo.multiply %v1275, %v1274 : tensor<256x512x7x7xf32>
    %v1277 = stablehlo.reduce(%v1276 init: %v1261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1278 = stablehlo.reshape %v1152 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1280 = stablehlo.reduce(%v1278 init: %v1279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1281 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1282 = stablehlo.compare GT, %v873, %v1281 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1283 = stablehlo.select %v1282, %v1226, %v1281 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1284 = stablehlo.reshape %v827 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1286 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1287 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1288 = stablehlo.reduce(%v1284 init: %v1285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1289 = stablehlo.broadcast_in_dim %v1288, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1290 = stablehlo.divide %v1289, %v1286 : tensor<256x512x7x7xf32>
    %v1291 = stablehlo.subtract %v1284, %v1290 : tensor<256x512x7x7xf32>
    %v1292 = stablehlo.multiply %v1291, %v1291 : tensor<256x512x7x7xf32>
    %v1293 = stablehlo.reduce(%v1292 init: %v1285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1294 = stablehlo.broadcast_in_dim %v1293, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1295 = stablehlo.divide %v1294, %v1286 : tensor<256x512x7x7xf32>
    %v1296 = stablehlo.add %v1295, %v1287 : tensor<256x512x7x7xf32>
    %v1297 = stablehlo.rsqrt %v1296 : tensor<256x512x7x7xf32>
    %v1298 = stablehlo.multiply %v1291, %v1297 : tensor<256x512x7x7xf32>
    %v1299 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1300 = stablehlo.reshape %v1283 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1301 = stablehlo.multiply %v1299, %v1300 : tensor<256x512x7x7xf32>
    %v1302 = stablehlo.reduce(%v1301 init: %v1285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1303 = stablehlo.broadcast_in_dim %v1302, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1304 = stablehlo.multiply %v1298, %v1301 : tensor<256x512x7x7xf32>
    %v1305 = stablehlo.reduce(%v1304 init: %v1285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1306 = stablehlo.broadcast_in_dim %v1305, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1307 = stablehlo.multiply %v1301, %v1286 : tensor<256x512x7x7xf32>
    %v1308 = stablehlo.subtract %v1307, %v1303 : tensor<256x512x7x7xf32>
    %v1309 = stablehlo.multiply %v1298, %v1306 : tensor<256x512x7x7xf32>
    %v1310 = stablehlo.subtract %v1308, %v1309 : tensor<256x512x7x7xf32>
    %v1311 = stablehlo.divide %v1297, %v1286 : tensor<256x512x7x7xf32>
    %v1312 = stablehlo.multiply %v1311, %v1310 : tensor<256x512x7x7xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1315 = stablehlo.reverse %d4W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1316 = stablehlo.transpose %v1315, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1317 = stablehlo.convolution(%v1314, %v1316)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1319 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1320 = stablehlo.compare GT, %v820, %v1319 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1321 = stablehlo.select %v1320, %v1318, %v1319 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1322 = stablehlo.reshape %v800 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1324 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1325 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1326 = stablehlo.reduce(%v1322 init: %v1323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1327 = stablehlo.broadcast_in_dim %v1326, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1328 = stablehlo.divide %v1327, %v1324 : tensor<256x512x7x7xf32>
    %v1329 = stablehlo.subtract %v1322, %v1328 : tensor<256x512x7x7xf32>
    %v1330 = stablehlo.multiply %v1329, %v1329 : tensor<256x512x7x7xf32>
    %v1331 = stablehlo.reduce(%v1330 init: %v1323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1332 = stablehlo.broadcast_in_dim %v1331, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1333 = stablehlo.divide %v1332, %v1324 : tensor<256x512x7x7xf32>
    %v1334 = stablehlo.add %v1333, %v1325 : tensor<256x512x7x7xf32>
    %v1335 = stablehlo.rsqrt %v1334 : tensor<256x512x7x7xf32>
    %v1336 = stablehlo.multiply %v1329, %v1335 : tensor<256x512x7x7xf32>
    %v1337 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1338 = stablehlo.reshape %v1321 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1339 = stablehlo.multiply %v1337, %v1338 : tensor<256x512x7x7xf32>
    %v1340 = stablehlo.reduce(%v1339 init: %v1323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1341 = stablehlo.broadcast_in_dim %v1340, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1342 = stablehlo.multiply %v1336, %v1339 : tensor<256x512x7x7xf32>
    %v1343 = stablehlo.reduce(%v1342 init: %v1323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1344 = stablehlo.broadcast_in_dim %v1343, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1345 = stablehlo.multiply %v1339, %v1324 : tensor<256x512x7x7xf32>
    %v1346 = stablehlo.subtract %v1345, %v1341 : tensor<256x512x7x7xf32>
    %v1347 = stablehlo.multiply %v1336, %v1344 : tensor<256x512x7x7xf32>
    %v1348 = stablehlo.subtract %v1346, %v1347 : tensor<256x512x7x7xf32>
    %v1349 = stablehlo.divide %v1335, %v1324 : tensor<256x512x7x7xf32>
    %v1350 = stablehlo.multiply %v1349, %v1348 : tensor<256x512x7x7xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1354 = stablehlo.pad %v1352, %v1353, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512x14x14xf32>
    %v1355 = stablehlo.reverse %d4W1, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v1356 = stablehlo.transpose %v1355, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1357 = stablehlo.convolution(%v1354, %v1356)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1359 = stablehlo.reshape %v852 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1361 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1362 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1363 = stablehlo.reduce(%v1359 init: %v1360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1364 = stablehlo.broadcast_in_dim %v1363, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1365 = stablehlo.divide %v1364, %v1361 : tensor<256x512x7x7xf32>
    %v1366 = stablehlo.subtract %v1359, %v1365 : tensor<256x512x7x7xf32>
    %v1367 = stablehlo.multiply %v1366, %v1366 : tensor<256x512x7x7xf32>
    %v1368 = stablehlo.reduce(%v1367 init: %v1360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1369 = stablehlo.broadcast_in_dim %v1368, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1370 = stablehlo.divide %v1369, %v1361 : tensor<256x512x7x7xf32>
    %v1371 = stablehlo.add %v1370, %v1362 : tensor<256x512x7x7xf32>
    %v1372 = stablehlo.rsqrt %v1371 : tensor<256x512x7x7xf32>
    %v1373 = stablehlo.multiply %v1366, %v1372 : tensor<256x512x7x7xf32>
    %v1374 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1375 = stablehlo.reshape %v1283 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1376 = stablehlo.multiply %v1374, %v1375 : tensor<256x512x7x7xf32>
    %v1377 = stablehlo.reduce(%v1376 init: %v1360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1378 = stablehlo.broadcast_in_dim %v1377, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1379 = stablehlo.multiply %v1373, %v1376 : tensor<256x512x7x7xf32>
    %v1380 = stablehlo.reduce(%v1379 init: %v1360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1381 = stablehlo.broadcast_in_dim %v1380, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1382 = stablehlo.multiply %v1376, %v1361 : tensor<256x512x7x7xf32>
    %v1383 = stablehlo.subtract %v1382, %v1378 : tensor<256x512x7x7xf32>
    %v1384 = stablehlo.multiply %v1373, %v1381 : tensor<256x512x7x7xf32>
    %v1385 = stablehlo.subtract %v1383, %v1384 : tensor<256x512x7x7xf32>
    %v1386 = stablehlo.divide %v1372, %v1361 : tensor<256x512x7x7xf32>
    %v1387 = stablehlo.multiply %v1386, %v1385 : tensor<256x512x7x7xf32>
    %v1388 = stablehlo.reshape %v1387 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1389 = stablehlo.reshape %v1388 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1391 = stablehlo.pad %v1389, %v1390, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512x14x14xf32>
    %v1392 = stablehlo.reverse %d4Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v1393 = stablehlo.transpose %v1392, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v1394 = stablehlo.convolution(%v1391, %v1393)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x14x14xf32>, tensor<256x512x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v1395 = stablehlo.reshape %v1394 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1396 = stablehlo.add %v1358, %v1395 : tensor<256x50176xf32>
    %v1397 = stablehlo.reshape %v795 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1398 = stablehlo.reshape %v1351 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1400 = stablehlo.pad %v1398, %v1399, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512x14x14xf32>
    %v1401 = stablehlo.transpose %v1397, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1402 = stablehlo.transpose %v1400, dims = [1, 0, 2, 3] : (tensor<256x512x14x14xf32>) -> tensor<512x256x14x14xf32>
    %v1403 = stablehlo.convolution(%v1401, %v1402)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<512x256x14x14xf32>) -> tensor<256x512x3x3xf32>
    %v1404 = stablehlo.transpose %v1403, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1405 = stablehlo.reshape %v800 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1407 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1408 = stablehlo.reduce(%v1405 init: %v1406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1409 = stablehlo.broadcast_in_dim %v1408, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1410 = stablehlo.divide %v1409, %v1407 : tensor<256x512x7x7xf32>
    %v1411 = stablehlo.subtract %v1405, %v1410 : tensor<256x512x7x7xf32>
    %v1412 = stablehlo.multiply %v1411, %v1411 : tensor<256x512x7x7xf32>
    %v1413 = stablehlo.reduce(%v1412 init: %v1406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1414 = stablehlo.broadcast_in_dim %v1413, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1415 = stablehlo.divide %v1414, %v1407 : tensor<256x512x7x7xf32>
    %v1416 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1417 = stablehlo.add %v1415, %v1416 : tensor<256x512x7x7xf32>
    %v1418 = stablehlo.rsqrt %v1417 : tensor<256x512x7x7xf32>
    %v1419 = stablehlo.multiply %v1411, %v1418 : tensor<256x512x7x7xf32>
    %v1420 = stablehlo.reshape %v1321 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1421 = stablehlo.multiply %v1420, %v1419 : tensor<256x512x7x7xf32>
    %v1422 = stablehlo.reduce(%v1421 init: %v1406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1423 = stablehlo.reshape %v1321 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1424 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1425 = stablehlo.reduce(%v1423 init: %v1424) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1426 = stablehlo.reshape %v822 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1427 = stablehlo.reshape %v1313 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1428 = stablehlo.transpose %v1426, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1429 = stablehlo.transpose %v1427, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1430 = stablehlo.convolution(%v1428, %v1429)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1431 = stablehlo.transpose %v1430, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1432 = stablehlo.reshape %v827 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1433 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1434 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1435 = stablehlo.reduce(%v1432 init: %v1433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1436 = stablehlo.broadcast_in_dim %v1435, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1437 = stablehlo.divide %v1436, %v1434 : tensor<256x512x7x7xf32>
    %v1438 = stablehlo.subtract %v1432, %v1437 : tensor<256x512x7x7xf32>
    %v1439 = stablehlo.multiply %v1438, %v1438 : tensor<256x512x7x7xf32>
    %v1440 = stablehlo.reduce(%v1439 init: %v1433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1441 = stablehlo.broadcast_in_dim %v1440, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1442 = stablehlo.divide %v1441, %v1434 : tensor<256x512x7x7xf32>
    %v1443 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1444 = stablehlo.add %v1442, %v1443 : tensor<256x512x7x7xf32>
    %v1445 = stablehlo.rsqrt %v1444 : tensor<256x512x7x7xf32>
    %v1446 = stablehlo.multiply %v1438, %v1445 : tensor<256x512x7x7xf32>
    %v1447 = stablehlo.reshape %v1283 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1448 = stablehlo.multiply %v1447, %v1446 : tensor<256x512x7x7xf32>
    %v1449 = stablehlo.reduce(%v1448 init: %v1433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1450 = stablehlo.reshape %v1283 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1451 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1452 = stablehlo.reduce(%v1450 init: %v1451) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1453 = stablehlo.reshape %v795 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1454 = stablehlo.reshape %v1388 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1455 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1456 = stablehlo.pad %v1454, %v1455, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512x14x14xf32>
    %v1457 = stablehlo.transpose %v1453, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1458 = stablehlo.transpose %v1456, dims = [1, 0, 2, 3] : (tensor<256x512x14x14xf32>) -> tensor<512x256x14x14xf32>
    %v1459 = stablehlo.convolution(%v1457, %v1458)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<512x256x14x14xf32>) -> tensor<256x512x1x1xf32>
    %v1460 = stablehlo.transpose %v1459, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v1461 = stablehlo.reshape %v852 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1463 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1464 = stablehlo.reduce(%v1461 init: %v1462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1465 = stablehlo.broadcast_in_dim %v1464, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1466 = stablehlo.divide %v1465, %v1463 : tensor<256x512x7x7xf32>
    %v1467 = stablehlo.subtract %v1461, %v1466 : tensor<256x512x7x7xf32>
    %v1468 = stablehlo.multiply %v1467, %v1467 : tensor<256x512x7x7xf32>
    %v1469 = stablehlo.reduce(%v1468 init: %v1462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1470 = stablehlo.broadcast_in_dim %v1469, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1471 = stablehlo.divide %v1470, %v1463 : tensor<256x512x7x7xf32>
    %v1472 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1473 = stablehlo.add %v1471, %v1472 : tensor<256x512x7x7xf32>
    %v1474 = stablehlo.rsqrt %v1473 : tensor<256x512x7x7xf32>
    %v1475 = stablehlo.multiply %v1467, %v1474 : tensor<256x512x7x7xf32>
    %v1476 = stablehlo.reshape %v1283 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1477 = stablehlo.multiply %v1476, %v1475 : tensor<256x512x7x7xf32>
    %v1478 = stablehlo.reduce(%v1477 init: %v1462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1479 = stablehlo.reshape %v1283 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1481 = stablehlo.reduce(%v1479 init: %v1480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1482 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1483 = stablehlo.compare GT, %v793, %v1482 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1484 = stablehlo.select %v1483, %v1396, %v1482 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1485 = stablehlo.reshape %v772 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1486 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1487 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1488 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1489 = stablehlo.reduce(%v1485 init: %v1486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1490 = stablehlo.broadcast_in_dim %v1489, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1491 = stablehlo.divide %v1490, %v1487 : tensor<256x256x14x14xf32>
    %v1492 = stablehlo.subtract %v1485, %v1491 : tensor<256x256x14x14xf32>
    %v1493 = stablehlo.multiply %v1492, %v1492 : tensor<256x256x14x14xf32>
    %v1494 = stablehlo.reduce(%v1493 init: %v1486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1495 = stablehlo.broadcast_in_dim %v1494, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1496 = stablehlo.divide %v1495, %v1487 : tensor<256x256x14x14xf32>
    %v1497 = stablehlo.add %v1496, %v1488 : tensor<256x256x14x14xf32>
    %v1498 = stablehlo.rsqrt %v1497 : tensor<256x256x14x14xf32>
    %v1499 = stablehlo.multiply %v1492, %v1498 : tensor<256x256x14x14xf32>
    %v1500 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1501 = stablehlo.reshape %v1484 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1502 = stablehlo.multiply %v1500, %v1501 : tensor<256x256x14x14xf32>
    %v1503 = stablehlo.reduce(%v1502 init: %v1486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1504 = stablehlo.broadcast_in_dim %v1503, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1505 = stablehlo.multiply %v1499, %v1502 : tensor<256x256x14x14xf32>
    %v1506 = stablehlo.reduce(%v1505 init: %v1486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1507 = stablehlo.broadcast_in_dim %v1506, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1508 = stablehlo.multiply %v1502, %v1487 : tensor<256x256x14x14xf32>
    %v1509 = stablehlo.subtract %v1508, %v1504 : tensor<256x256x14x14xf32>
    %v1510 = stablehlo.multiply %v1499, %v1507 : tensor<256x256x14x14xf32>
    %v1511 = stablehlo.subtract %v1509, %v1510 : tensor<256x256x14x14xf32>
    %v1512 = stablehlo.divide %v1498, %v1487 : tensor<256x256x14x14xf32>
    %v1513 = stablehlo.multiply %v1512, %v1511 : tensor<256x256x14x14xf32>
    %v1514 = stablehlo.reshape %v1513 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1515 = stablehlo.reshape %v1514 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1516 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1517 = stablehlo.transpose %v1516, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1518 = stablehlo.convolution(%v1515, %v1517)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1520 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1521 = stablehlo.compare GT, %v765, %v1520 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1522 = stablehlo.select %v1521, %v1519, %v1520 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1523 = stablehlo.reshape %v745 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1524 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1525 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1526 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1527 = stablehlo.reduce(%v1523 init: %v1524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1528 = stablehlo.broadcast_in_dim %v1527, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1529 = stablehlo.divide %v1528, %v1525 : tensor<256x256x14x14xf32>
    %v1530 = stablehlo.subtract %v1523, %v1529 : tensor<256x256x14x14xf32>
    %v1531 = stablehlo.multiply %v1530, %v1530 : tensor<256x256x14x14xf32>
    %v1532 = stablehlo.reduce(%v1531 init: %v1524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1533 = stablehlo.broadcast_in_dim %v1532, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1534 = stablehlo.divide %v1533, %v1525 : tensor<256x256x14x14xf32>
    %v1535 = stablehlo.add %v1534, %v1526 : tensor<256x256x14x14xf32>
    %v1536 = stablehlo.rsqrt %v1535 : tensor<256x256x14x14xf32>
    %v1537 = stablehlo.multiply %v1530, %v1536 : tensor<256x256x14x14xf32>
    %v1538 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1539 = stablehlo.reshape %v1522 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1540 = stablehlo.multiply %v1538, %v1539 : tensor<256x256x14x14xf32>
    %v1541 = stablehlo.reduce(%v1540 init: %v1524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1542 = stablehlo.broadcast_in_dim %v1541, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1543 = stablehlo.multiply %v1537, %v1540 : tensor<256x256x14x14xf32>
    %v1544 = stablehlo.reduce(%v1543 init: %v1524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1545 = stablehlo.broadcast_in_dim %v1544, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1546 = stablehlo.multiply %v1540, %v1525 : tensor<256x256x14x14xf32>
    %v1547 = stablehlo.subtract %v1546, %v1542 : tensor<256x256x14x14xf32>
    %v1548 = stablehlo.multiply %v1537, %v1545 : tensor<256x256x14x14xf32>
    %v1549 = stablehlo.subtract %v1547, %v1548 : tensor<256x256x14x14xf32>
    %v1550 = stablehlo.divide %v1536, %v1525 : tensor<256x256x14x14xf32>
    %v1551 = stablehlo.multiply %v1550, %v1549 : tensor<256x256x14x14xf32>
    %v1552 = stablehlo.reshape %v1551 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1553 = stablehlo.reshape %v1552 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1554 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1555 = stablehlo.transpose %v1554, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1556 = stablehlo.convolution(%v1553, %v1555)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1557 = stablehlo.reshape %v1556 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1558 = stablehlo.add %v1557, %v1484 : tensor<256x50176xf32>
    %v1559 = stablehlo.reshape %v740 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1560 = stablehlo.reshape %v1552 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1561 = stablehlo.transpose %v1559, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1562 = stablehlo.transpose %v1560, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1563 = stablehlo.convolution(%v1561, %v1562)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1564 = stablehlo.transpose %v1563, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1565 = stablehlo.reshape %v745 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1567 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1568 = stablehlo.reduce(%v1565 init: %v1566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1569 = stablehlo.broadcast_in_dim %v1568, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1570 = stablehlo.divide %v1569, %v1567 : tensor<256x256x14x14xf32>
    %v1571 = stablehlo.subtract %v1565, %v1570 : tensor<256x256x14x14xf32>
    %v1572 = stablehlo.multiply %v1571, %v1571 : tensor<256x256x14x14xf32>
    %v1573 = stablehlo.reduce(%v1572 init: %v1566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1574 = stablehlo.broadcast_in_dim %v1573, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1575 = stablehlo.divide %v1574, %v1567 : tensor<256x256x14x14xf32>
    %v1576 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1577 = stablehlo.add %v1575, %v1576 : tensor<256x256x14x14xf32>
    %v1578 = stablehlo.rsqrt %v1577 : tensor<256x256x14x14xf32>
    %v1579 = stablehlo.multiply %v1571, %v1578 : tensor<256x256x14x14xf32>
    %v1580 = stablehlo.reshape %v1522 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1581 = stablehlo.multiply %v1580, %v1579 : tensor<256x256x14x14xf32>
    %v1582 = stablehlo.reduce(%v1581 init: %v1566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1583 = stablehlo.reshape %v1522 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1584 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1585 = stablehlo.reduce(%v1583 init: %v1584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1586 = stablehlo.reshape %v767 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1587 = stablehlo.reshape %v1514 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1588 = stablehlo.transpose %v1586, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1589 = stablehlo.transpose %v1587, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1590 = stablehlo.convolution(%v1588, %v1589)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1591 = stablehlo.transpose %v1590, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1592 = stablehlo.reshape %v772 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1594 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1595 = stablehlo.reduce(%v1592 init: %v1593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1596 = stablehlo.broadcast_in_dim %v1595, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1597 = stablehlo.divide %v1596, %v1594 : tensor<256x256x14x14xf32>
    %v1598 = stablehlo.subtract %v1592, %v1597 : tensor<256x256x14x14xf32>
    %v1599 = stablehlo.multiply %v1598, %v1598 : tensor<256x256x14x14xf32>
    %v1600 = stablehlo.reduce(%v1599 init: %v1593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1601 = stablehlo.broadcast_in_dim %v1600, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1602 = stablehlo.divide %v1601, %v1594 : tensor<256x256x14x14xf32>
    %v1603 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1604 = stablehlo.add %v1602, %v1603 : tensor<256x256x14x14xf32>
    %v1605 = stablehlo.rsqrt %v1604 : tensor<256x256x14x14xf32>
    %v1606 = stablehlo.multiply %v1598, %v1605 : tensor<256x256x14x14xf32>
    %v1607 = stablehlo.reshape %v1484 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1608 = stablehlo.multiply %v1607, %v1606 : tensor<256x256x14x14xf32>
    %v1609 = stablehlo.reduce(%v1608 init: %v1593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1610 = stablehlo.reshape %v1484 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1611 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1612 = stablehlo.reduce(%v1610 init: %v1611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1613 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1614 = stablehlo.compare GT, %v738, %v1613 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1615 = stablehlo.select %v1614, %v1558, %v1613 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1616 = stablehlo.reshape %v717 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1618 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1619 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1620 = stablehlo.reduce(%v1616 init: %v1617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1621 = stablehlo.broadcast_in_dim %v1620, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1622 = stablehlo.divide %v1621, %v1618 : tensor<256x256x14x14xf32>
    %v1623 = stablehlo.subtract %v1616, %v1622 : tensor<256x256x14x14xf32>
    %v1624 = stablehlo.multiply %v1623, %v1623 : tensor<256x256x14x14xf32>
    %v1625 = stablehlo.reduce(%v1624 init: %v1617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1626 = stablehlo.broadcast_in_dim %v1625, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1627 = stablehlo.divide %v1626, %v1618 : tensor<256x256x14x14xf32>
    %v1628 = stablehlo.add %v1627, %v1619 : tensor<256x256x14x14xf32>
    %v1629 = stablehlo.rsqrt %v1628 : tensor<256x256x14x14xf32>
    %v1630 = stablehlo.multiply %v1623, %v1629 : tensor<256x256x14x14xf32>
    %v1631 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1632 = stablehlo.reshape %v1615 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1633 = stablehlo.multiply %v1631, %v1632 : tensor<256x256x14x14xf32>
    %v1634 = stablehlo.reduce(%v1633 init: %v1617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1635 = stablehlo.broadcast_in_dim %v1634, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1636 = stablehlo.multiply %v1630, %v1633 : tensor<256x256x14x14xf32>
    %v1637 = stablehlo.reduce(%v1636 init: %v1617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1638 = stablehlo.broadcast_in_dim %v1637, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1639 = stablehlo.multiply %v1633, %v1618 : tensor<256x256x14x14xf32>
    %v1640 = stablehlo.subtract %v1639, %v1635 : tensor<256x256x14x14xf32>
    %v1641 = stablehlo.multiply %v1630, %v1638 : tensor<256x256x14x14xf32>
    %v1642 = stablehlo.subtract %v1640, %v1641 : tensor<256x256x14x14xf32>
    %v1643 = stablehlo.divide %v1629, %v1618 : tensor<256x256x14x14xf32>
    %v1644 = stablehlo.multiply %v1643, %v1642 : tensor<256x256x14x14xf32>
    %v1645 = stablehlo.reshape %v1644 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1646 = stablehlo.reshape %v1645 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1647 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1648 = stablehlo.transpose %v1647, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1649 = stablehlo.convolution(%v1646, %v1648)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1650 = stablehlo.reshape %v1649 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1651 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1652 = stablehlo.compare GT, %v710, %v1651 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1653 = stablehlo.select %v1652, %v1650, %v1651 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1654 = stablehlo.reshape %v690 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1655 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1656 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1657 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1658 = stablehlo.reduce(%v1654 init: %v1655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1659 = stablehlo.broadcast_in_dim %v1658, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1660 = stablehlo.divide %v1659, %v1656 : tensor<256x256x14x14xf32>
    %v1661 = stablehlo.subtract %v1654, %v1660 : tensor<256x256x14x14xf32>
    %v1662 = stablehlo.multiply %v1661, %v1661 : tensor<256x256x14x14xf32>
    %v1663 = stablehlo.reduce(%v1662 init: %v1655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1664 = stablehlo.broadcast_in_dim %v1663, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1665 = stablehlo.divide %v1664, %v1656 : tensor<256x256x14x14xf32>
    %v1666 = stablehlo.add %v1665, %v1657 : tensor<256x256x14x14xf32>
    %v1667 = stablehlo.rsqrt %v1666 : tensor<256x256x14x14xf32>
    %v1668 = stablehlo.multiply %v1661, %v1667 : tensor<256x256x14x14xf32>
    %v1669 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1670 = stablehlo.reshape %v1653 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1671 = stablehlo.multiply %v1669, %v1670 : tensor<256x256x14x14xf32>
    %v1672 = stablehlo.reduce(%v1671 init: %v1655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1673 = stablehlo.broadcast_in_dim %v1672, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1674 = stablehlo.multiply %v1668, %v1671 : tensor<256x256x14x14xf32>
    %v1675 = stablehlo.reduce(%v1674 init: %v1655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1676 = stablehlo.broadcast_in_dim %v1675, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1677 = stablehlo.multiply %v1671, %v1656 : tensor<256x256x14x14xf32>
    %v1678 = stablehlo.subtract %v1677, %v1673 : tensor<256x256x14x14xf32>
    %v1679 = stablehlo.multiply %v1668, %v1676 : tensor<256x256x14x14xf32>
    %v1680 = stablehlo.subtract %v1678, %v1679 : tensor<256x256x14x14xf32>
    %v1681 = stablehlo.divide %v1667, %v1656 : tensor<256x256x14x14xf32>
    %v1682 = stablehlo.multiply %v1681, %v1680 : tensor<256x256x14x14xf32>
    %v1683 = stablehlo.reshape %v1682 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1684 = stablehlo.reshape %v1683 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1685 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1686 = stablehlo.transpose %v1685, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1687 = stablehlo.convolution(%v1684, %v1686)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1688 = stablehlo.reshape %v1687 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1689 = stablehlo.add %v1688, %v1615 : tensor<256x50176xf32>
    %v1690 = stablehlo.reshape %v685 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1691 = stablehlo.reshape %v1683 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1692 = stablehlo.transpose %v1690, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1693 = stablehlo.transpose %v1691, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1694 = stablehlo.convolution(%v1692, %v1693)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1695 = stablehlo.transpose %v1694, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1696 = stablehlo.reshape %v690 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1697 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1698 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1699 = stablehlo.reduce(%v1696 init: %v1697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1700 = stablehlo.broadcast_in_dim %v1699, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1701 = stablehlo.divide %v1700, %v1698 : tensor<256x256x14x14xf32>
    %v1702 = stablehlo.subtract %v1696, %v1701 : tensor<256x256x14x14xf32>
    %v1703 = stablehlo.multiply %v1702, %v1702 : tensor<256x256x14x14xf32>
    %v1704 = stablehlo.reduce(%v1703 init: %v1697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1705 = stablehlo.broadcast_in_dim %v1704, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1706 = stablehlo.divide %v1705, %v1698 : tensor<256x256x14x14xf32>
    %v1707 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1708 = stablehlo.add %v1706, %v1707 : tensor<256x256x14x14xf32>
    %v1709 = stablehlo.rsqrt %v1708 : tensor<256x256x14x14xf32>
    %v1710 = stablehlo.multiply %v1702, %v1709 : tensor<256x256x14x14xf32>
    %v1711 = stablehlo.reshape %v1653 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1712 = stablehlo.multiply %v1711, %v1710 : tensor<256x256x14x14xf32>
    %v1713 = stablehlo.reduce(%v1712 init: %v1697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1714 = stablehlo.reshape %v1653 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1716 = stablehlo.reduce(%v1714 init: %v1715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1717 = stablehlo.reshape %v712 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1718 = stablehlo.reshape %v1645 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1719 = stablehlo.transpose %v1717, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1720 = stablehlo.transpose %v1718, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1721 = stablehlo.convolution(%v1719, %v1720)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1722 = stablehlo.transpose %v1721, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1723 = stablehlo.reshape %v717 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1725 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1726 = stablehlo.reduce(%v1723 init: %v1724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1727 = stablehlo.broadcast_in_dim %v1726, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1728 = stablehlo.divide %v1727, %v1725 : tensor<256x256x14x14xf32>
    %v1729 = stablehlo.subtract %v1723, %v1728 : tensor<256x256x14x14xf32>
    %v1730 = stablehlo.multiply %v1729, %v1729 : tensor<256x256x14x14xf32>
    %v1731 = stablehlo.reduce(%v1730 init: %v1724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1732 = stablehlo.broadcast_in_dim %v1731, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1733 = stablehlo.divide %v1732, %v1725 : tensor<256x256x14x14xf32>
    %v1734 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1735 = stablehlo.add %v1733, %v1734 : tensor<256x256x14x14xf32>
    %v1736 = stablehlo.rsqrt %v1735 : tensor<256x256x14x14xf32>
    %v1737 = stablehlo.multiply %v1729, %v1736 : tensor<256x256x14x14xf32>
    %v1738 = stablehlo.reshape %v1615 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1739 = stablehlo.multiply %v1738, %v1737 : tensor<256x256x14x14xf32>
    %v1740 = stablehlo.reduce(%v1739 init: %v1724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1741 = stablehlo.reshape %v1615 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1743 = stablehlo.reduce(%v1741 init: %v1742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1744 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1745 = stablehlo.compare GT, %v683, %v1744 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1746 = stablehlo.select %v1745, %v1689, %v1744 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1747 = stablehlo.reshape %v662 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1748 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1749 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1750 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1751 = stablehlo.reduce(%v1747 init: %v1748) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1752 = stablehlo.broadcast_in_dim %v1751, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1753 = stablehlo.divide %v1752, %v1749 : tensor<256x256x14x14xf32>
    %v1754 = stablehlo.subtract %v1747, %v1753 : tensor<256x256x14x14xf32>
    %v1755 = stablehlo.multiply %v1754, %v1754 : tensor<256x256x14x14xf32>
    %v1756 = stablehlo.reduce(%v1755 init: %v1748) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1757 = stablehlo.broadcast_in_dim %v1756, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1758 = stablehlo.divide %v1757, %v1749 : tensor<256x256x14x14xf32>
    %v1759 = stablehlo.add %v1758, %v1750 : tensor<256x256x14x14xf32>
    %v1760 = stablehlo.rsqrt %v1759 : tensor<256x256x14x14xf32>
    %v1761 = stablehlo.multiply %v1754, %v1760 : tensor<256x256x14x14xf32>
    %v1762 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1763 = stablehlo.reshape %v1746 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1764 = stablehlo.multiply %v1762, %v1763 : tensor<256x256x14x14xf32>
    %v1765 = stablehlo.reduce(%v1764 init: %v1748) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1766 = stablehlo.broadcast_in_dim %v1765, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1767 = stablehlo.multiply %v1761, %v1764 : tensor<256x256x14x14xf32>
    %v1768 = stablehlo.reduce(%v1767 init: %v1748) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1769 = stablehlo.broadcast_in_dim %v1768, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1770 = stablehlo.multiply %v1764, %v1749 : tensor<256x256x14x14xf32>
    %v1771 = stablehlo.subtract %v1770, %v1766 : tensor<256x256x14x14xf32>
    %v1772 = stablehlo.multiply %v1761, %v1769 : tensor<256x256x14x14xf32>
    %v1773 = stablehlo.subtract %v1771, %v1772 : tensor<256x256x14x14xf32>
    %v1774 = stablehlo.divide %v1760, %v1749 : tensor<256x256x14x14xf32>
    %v1775 = stablehlo.multiply %v1774, %v1773 : tensor<256x256x14x14xf32>
    %v1776 = stablehlo.reshape %v1775 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1778 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1779 = stablehlo.transpose %v1778, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1780 = stablehlo.convolution(%v1777, %v1779)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1781 = stablehlo.reshape %v1780 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1782 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1783 = stablehlo.compare GT, %v655, %v1782 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1784 = stablehlo.select %v1783, %v1781, %v1782 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1785 = stablehlo.reshape %v635 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1787 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1788 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1789 = stablehlo.reduce(%v1785 init: %v1786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1790 = stablehlo.broadcast_in_dim %v1789, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1791 = stablehlo.divide %v1790, %v1787 : tensor<256x256x14x14xf32>
    %v1792 = stablehlo.subtract %v1785, %v1791 : tensor<256x256x14x14xf32>
    %v1793 = stablehlo.multiply %v1792, %v1792 : tensor<256x256x14x14xf32>
    %v1794 = stablehlo.reduce(%v1793 init: %v1786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1795 = stablehlo.broadcast_in_dim %v1794, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1796 = stablehlo.divide %v1795, %v1787 : tensor<256x256x14x14xf32>
    %v1797 = stablehlo.add %v1796, %v1788 : tensor<256x256x14x14xf32>
    %v1798 = stablehlo.rsqrt %v1797 : tensor<256x256x14x14xf32>
    %v1799 = stablehlo.multiply %v1792, %v1798 : tensor<256x256x14x14xf32>
    %v1800 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1801 = stablehlo.reshape %v1784 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1802 = stablehlo.multiply %v1800, %v1801 : tensor<256x256x14x14xf32>
    %v1803 = stablehlo.reduce(%v1802 init: %v1786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1804 = stablehlo.broadcast_in_dim %v1803, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1805 = stablehlo.multiply %v1799, %v1802 : tensor<256x256x14x14xf32>
    %v1806 = stablehlo.reduce(%v1805 init: %v1786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1807 = stablehlo.broadcast_in_dim %v1806, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1808 = stablehlo.multiply %v1802, %v1787 : tensor<256x256x14x14xf32>
    %v1809 = stablehlo.subtract %v1808, %v1804 : tensor<256x256x14x14xf32>
    %v1810 = stablehlo.multiply %v1799, %v1807 : tensor<256x256x14x14xf32>
    %v1811 = stablehlo.subtract %v1809, %v1810 : tensor<256x256x14x14xf32>
    %v1812 = stablehlo.divide %v1798, %v1787 : tensor<256x256x14x14xf32>
    %v1813 = stablehlo.multiply %v1812, %v1811 : tensor<256x256x14x14xf32>
    %v1814 = stablehlo.reshape %v1813 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1815 = stablehlo.reshape %v1814 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1816 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1817 = stablehlo.transpose %v1816, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1818 = stablehlo.convolution(%v1815, %v1817)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1819 = stablehlo.reshape %v1818 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1820 = stablehlo.add %v1819, %v1746 : tensor<256x50176xf32>
    %v1821 = stablehlo.reshape %v630 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1822 = stablehlo.reshape %v1814 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1823 = stablehlo.transpose %v1821, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1824 = stablehlo.transpose %v1822, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1825 = stablehlo.convolution(%v1823, %v1824)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1826 = stablehlo.transpose %v1825, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1827 = stablehlo.reshape %v635 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1828 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1829 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1830 = stablehlo.reduce(%v1827 init: %v1828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1831 = stablehlo.broadcast_in_dim %v1830, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1832 = stablehlo.divide %v1831, %v1829 : tensor<256x256x14x14xf32>
    %v1833 = stablehlo.subtract %v1827, %v1832 : tensor<256x256x14x14xf32>
    %v1834 = stablehlo.multiply %v1833, %v1833 : tensor<256x256x14x14xf32>
    %v1835 = stablehlo.reduce(%v1834 init: %v1828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1836 = stablehlo.broadcast_in_dim %v1835, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1837 = stablehlo.divide %v1836, %v1829 : tensor<256x256x14x14xf32>
    %v1838 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1839 = stablehlo.add %v1837, %v1838 : tensor<256x256x14x14xf32>
    %v1840 = stablehlo.rsqrt %v1839 : tensor<256x256x14x14xf32>
    %v1841 = stablehlo.multiply %v1833, %v1840 : tensor<256x256x14x14xf32>
    %v1842 = stablehlo.reshape %v1784 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1843 = stablehlo.multiply %v1842, %v1841 : tensor<256x256x14x14xf32>
    %v1844 = stablehlo.reduce(%v1843 init: %v1828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1845 = stablehlo.reshape %v1784 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1847 = stablehlo.reduce(%v1845 init: %v1846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1848 = stablehlo.reshape %v657 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1849 = stablehlo.reshape %v1776 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1850 = stablehlo.transpose %v1848, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1851 = stablehlo.transpose %v1849, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1852 = stablehlo.convolution(%v1850, %v1851)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1853 = stablehlo.transpose %v1852, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1854 = stablehlo.reshape %v662 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1855 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1856 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1857 = stablehlo.reduce(%v1854 init: %v1855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1858 = stablehlo.broadcast_in_dim %v1857, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1859 = stablehlo.divide %v1858, %v1856 : tensor<256x256x14x14xf32>
    %v1860 = stablehlo.subtract %v1854, %v1859 : tensor<256x256x14x14xf32>
    %v1861 = stablehlo.multiply %v1860, %v1860 : tensor<256x256x14x14xf32>
    %v1862 = stablehlo.reduce(%v1861 init: %v1855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1863 = stablehlo.broadcast_in_dim %v1862, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1864 = stablehlo.divide %v1863, %v1856 : tensor<256x256x14x14xf32>
    %v1865 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1866 = stablehlo.add %v1864, %v1865 : tensor<256x256x14x14xf32>
    %v1867 = stablehlo.rsqrt %v1866 : tensor<256x256x14x14xf32>
    %v1868 = stablehlo.multiply %v1860, %v1867 : tensor<256x256x14x14xf32>
    %v1869 = stablehlo.reshape %v1746 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1870 = stablehlo.multiply %v1869, %v1868 : tensor<256x256x14x14xf32>
    %v1871 = stablehlo.reduce(%v1870 init: %v1855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1872 = stablehlo.reshape %v1746 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1873 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1874 = stablehlo.reduce(%v1872 init: %v1873) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1875 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1876 = stablehlo.compare GT, %v628, %v1875 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1877 = stablehlo.select %v1876, %v1820, %v1875 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1878 = stablehlo.reshape %v607 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1879 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1880 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1881 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1882 = stablehlo.reduce(%v1878 init: %v1879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1883 = stablehlo.broadcast_in_dim %v1882, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1884 = stablehlo.divide %v1883, %v1880 : tensor<256x256x14x14xf32>
    %v1885 = stablehlo.subtract %v1878, %v1884 : tensor<256x256x14x14xf32>
    %v1886 = stablehlo.multiply %v1885, %v1885 : tensor<256x256x14x14xf32>
    %v1887 = stablehlo.reduce(%v1886 init: %v1879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1888 = stablehlo.broadcast_in_dim %v1887, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1889 = stablehlo.divide %v1888, %v1880 : tensor<256x256x14x14xf32>
    %v1890 = stablehlo.add %v1889, %v1881 : tensor<256x256x14x14xf32>
    %v1891 = stablehlo.rsqrt %v1890 : tensor<256x256x14x14xf32>
    %v1892 = stablehlo.multiply %v1885, %v1891 : tensor<256x256x14x14xf32>
    %v1893 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1894 = stablehlo.reshape %v1877 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1895 = stablehlo.multiply %v1893, %v1894 : tensor<256x256x14x14xf32>
    %v1896 = stablehlo.reduce(%v1895 init: %v1879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1897 = stablehlo.broadcast_in_dim %v1896, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1898 = stablehlo.multiply %v1892, %v1895 : tensor<256x256x14x14xf32>
    %v1899 = stablehlo.reduce(%v1898 init: %v1879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1900 = stablehlo.broadcast_in_dim %v1899, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1901 = stablehlo.multiply %v1895, %v1880 : tensor<256x256x14x14xf32>
    %v1902 = stablehlo.subtract %v1901, %v1897 : tensor<256x256x14x14xf32>
    %v1903 = stablehlo.multiply %v1892, %v1900 : tensor<256x256x14x14xf32>
    %v1904 = stablehlo.subtract %v1902, %v1903 : tensor<256x256x14x14xf32>
    %v1905 = stablehlo.divide %v1891, %v1880 : tensor<256x256x14x14xf32>
    %v1906 = stablehlo.multiply %v1905, %v1904 : tensor<256x256x14x14xf32>
    %v1907 = stablehlo.reshape %v1906 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1908 = stablehlo.reshape %v1907 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1909 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1910 = stablehlo.transpose %v1909, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1911 = stablehlo.convolution(%v1908, %v1910)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1912 = stablehlo.reshape %v1911 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1913 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1914 = stablehlo.compare GT, %v600, %v1913 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1915 = stablehlo.select %v1914, %v1912, %v1913 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1916 = stablehlo.reshape %v580 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1918 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1919 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1920 = stablehlo.reduce(%v1916 init: %v1917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1921 = stablehlo.broadcast_in_dim %v1920, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1922 = stablehlo.divide %v1921, %v1918 : tensor<256x256x14x14xf32>
    %v1923 = stablehlo.subtract %v1916, %v1922 : tensor<256x256x14x14xf32>
    %v1924 = stablehlo.multiply %v1923, %v1923 : tensor<256x256x14x14xf32>
    %v1925 = stablehlo.reduce(%v1924 init: %v1917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1926 = stablehlo.broadcast_in_dim %v1925, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1927 = stablehlo.divide %v1926, %v1918 : tensor<256x256x14x14xf32>
    %v1928 = stablehlo.add %v1927, %v1919 : tensor<256x256x14x14xf32>
    %v1929 = stablehlo.rsqrt %v1928 : tensor<256x256x14x14xf32>
    %v1930 = stablehlo.multiply %v1923, %v1929 : tensor<256x256x14x14xf32>
    %v1931 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1932 = stablehlo.reshape %v1915 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1933 = stablehlo.multiply %v1931, %v1932 : tensor<256x256x14x14xf32>
    %v1934 = stablehlo.reduce(%v1933 init: %v1917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1935 = stablehlo.broadcast_in_dim %v1934, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1936 = stablehlo.multiply %v1930, %v1933 : tensor<256x256x14x14xf32>
    %v1937 = stablehlo.reduce(%v1936 init: %v1917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1938 = stablehlo.broadcast_in_dim %v1937, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1939 = stablehlo.multiply %v1933, %v1918 : tensor<256x256x14x14xf32>
    %v1940 = stablehlo.subtract %v1939, %v1935 : tensor<256x256x14x14xf32>
    %v1941 = stablehlo.multiply %v1930, %v1938 : tensor<256x256x14x14xf32>
    %v1942 = stablehlo.subtract %v1940, %v1941 : tensor<256x256x14x14xf32>
    %v1943 = stablehlo.divide %v1929, %v1918 : tensor<256x256x14x14xf32>
    %v1944 = stablehlo.multiply %v1943, %v1942 : tensor<256x256x14x14xf32>
    %v1945 = stablehlo.reshape %v1944 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1946 = stablehlo.reshape %v1945 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1947 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1948 = stablehlo.transpose %v1947, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1949 = stablehlo.convolution(%v1946, %v1948)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1950 = stablehlo.reshape %v1949 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1951 = stablehlo.add %v1950, %v1877 : tensor<256x50176xf32>
    %v1952 = stablehlo.reshape %v575 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1953 = stablehlo.reshape %v1945 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1954 = stablehlo.transpose %v1952, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1955 = stablehlo.transpose %v1953, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1956 = stablehlo.convolution(%v1954, %v1955)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1957 = stablehlo.transpose %v1956, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1958 = stablehlo.reshape %v580 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1960 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1961 = stablehlo.reduce(%v1958 init: %v1959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1962 = stablehlo.broadcast_in_dim %v1961, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1963 = stablehlo.divide %v1962, %v1960 : tensor<256x256x14x14xf32>
    %v1964 = stablehlo.subtract %v1958, %v1963 : tensor<256x256x14x14xf32>
    %v1965 = stablehlo.multiply %v1964, %v1964 : tensor<256x256x14x14xf32>
    %v1966 = stablehlo.reduce(%v1965 init: %v1959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1967 = stablehlo.broadcast_in_dim %v1966, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1968 = stablehlo.divide %v1967, %v1960 : tensor<256x256x14x14xf32>
    %v1969 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1970 = stablehlo.add %v1968, %v1969 : tensor<256x256x14x14xf32>
    %v1971 = stablehlo.rsqrt %v1970 : tensor<256x256x14x14xf32>
    %v1972 = stablehlo.multiply %v1964, %v1971 : tensor<256x256x14x14xf32>
    %v1973 = stablehlo.reshape %v1915 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1974 = stablehlo.multiply %v1973, %v1972 : tensor<256x256x14x14xf32>
    %v1975 = stablehlo.reduce(%v1974 init: %v1959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1976 = stablehlo.reshape %v1915 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1977 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1978 = stablehlo.reduce(%v1976 init: %v1977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1979 = stablehlo.reshape %v602 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1980 = stablehlo.reshape %v1907 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1981 = stablehlo.transpose %v1979, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1982 = stablehlo.transpose %v1980, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1983 = stablehlo.convolution(%v1981, %v1982)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1984 = stablehlo.transpose %v1983, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1985 = stablehlo.reshape %v607 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1987 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1988 = stablehlo.reduce(%v1985 init: %v1986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1989 = stablehlo.broadcast_in_dim %v1988, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1990 = stablehlo.divide %v1989, %v1987 : tensor<256x256x14x14xf32>
    %v1991 = stablehlo.subtract %v1985, %v1990 : tensor<256x256x14x14xf32>
    %v1992 = stablehlo.multiply %v1991, %v1991 : tensor<256x256x14x14xf32>
    %v1993 = stablehlo.reduce(%v1992 init: %v1986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1994 = stablehlo.broadcast_in_dim %v1993, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1995 = stablehlo.divide %v1994, %v1987 : tensor<256x256x14x14xf32>
    %v1996 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1997 = stablehlo.add %v1995, %v1996 : tensor<256x256x14x14xf32>
    %v1998 = stablehlo.rsqrt %v1997 : tensor<256x256x14x14xf32>
    %v1999 = stablehlo.multiply %v1991, %v1998 : tensor<256x256x14x14xf32>
    %v2000 = stablehlo.reshape %v1877 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2001 = stablehlo.multiply %v2000, %v1999 : tensor<256x256x14x14xf32>
    %v2002 = stablehlo.reduce(%v2001 init: %v1986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2003 = stablehlo.reshape %v1877 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2004 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2005 = stablehlo.reduce(%v2003 init: %v2004) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2006 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2007 = stablehlo.compare GT, %v573, %v2006 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2008 = stablehlo.select %v2007, %v1951, %v2006 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2009 = stablehlo.reshape %v552 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2010 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2011 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2012 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2013 = stablehlo.reduce(%v2009 init: %v2010) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2014 = stablehlo.broadcast_in_dim %v2013, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2015 = stablehlo.divide %v2014, %v2011 : tensor<256x256x14x14xf32>
    %v2016 = stablehlo.subtract %v2009, %v2015 : tensor<256x256x14x14xf32>
    %v2017 = stablehlo.multiply %v2016, %v2016 : tensor<256x256x14x14xf32>
    %v2018 = stablehlo.reduce(%v2017 init: %v2010) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2019 = stablehlo.broadcast_in_dim %v2018, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2020 = stablehlo.divide %v2019, %v2011 : tensor<256x256x14x14xf32>
    %v2021 = stablehlo.add %v2020, %v2012 : tensor<256x256x14x14xf32>
    %v2022 = stablehlo.rsqrt %v2021 : tensor<256x256x14x14xf32>
    %v2023 = stablehlo.multiply %v2016, %v2022 : tensor<256x256x14x14xf32>
    %v2024 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2025 = stablehlo.reshape %v2008 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2026 = stablehlo.multiply %v2024, %v2025 : tensor<256x256x14x14xf32>
    %v2027 = stablehlo.reduce(%v2026 init: %v2010) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2028 = stablehlo.broadcast_in_dim %v2027, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2029 = stablehlo.multiply %v2023, %v2026 : tensor<256x256x14x14xf32>
    %v2030 = stablehlo.reduce(%v2029 init: %v2010) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2031 = stablehlo.broadcast_in_dim %v2030, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2032 = stablehlo.multiply %v2026, %v2011 : tensor<256x256x14x14xf32>
    %v2033 = stablehlo.subtract %v2032, %v2028 : tensor<256x256x14x14xf32>
    %v2034 = stablehlo.multiply %v2023, %v2031 : tensor<256x256x14x14xf32>
    %v2035 = stablehlo.subtract %v2033, %v2034 : tensor<256x256x14x14xf32>
    %v2036 = stablehlo.divide %v2022, %v2011 : tensor<256x256x14x14xf32>
    %v2037 = stablehlo.multiply %v2036, %v2035 : tensor<256x256x14x14xf32>
    %v2038 = stablehlo.reshape %v2037 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2039 = stablehlo.reshape %v2038 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2040 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2041 = stablehlo.transpose %v2040, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2042 = stablehlo.convolution(%v2039, %v2041)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2044 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2045 = stablehlo.compare GT, %v545, %v2044 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2046 = stablehlo.select %v2045, %v2043, %v2044 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2047 = stablehlo.reshape %v525 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2048 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2049 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2050 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2051 = stablehlo.reduce(%v2047 init: %v2048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2052 = stablehlo.broadcast_in_dim %v2051, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2053 = stablehlo.divide %v2052, %v2049 : tensor<256x256x14x14xf32>
    %v2054 = stablehlo.subtract %v2047, %v2053 : tensor<256x256x14x14xf32>
    %v2055 = stablehlo.multiply %v2054, %v2054 : tensor<256x256x14x14xf32>
    %v2056 = stablehlo.reduce(%v2055 init: %v2048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2057 = stablehlo.broadcast_in_dim %v2056, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2058 = stablehlo.divide %v2057, %v2049 : tensor<256x256x14x14xf32>
    %v2059 = stablehlo.add %v2058, %v2050 : tensor<256x256x14x14xf32>
    %v2060 = stablehlo.rsqrt %v2059 : tensor<256x256x14x14xf32>
    %v2061 = stablehlo.multiply %v2054, %v2060 : tensor<256x256x14x14xf32>
    %v2062 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2063 = stablehlo.reshape %v2046 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2064 = stablehlo.multiply %v2062, %v2063 : tensor<256x256x14x14xf32>
    %v2065 = stablehlo.reduce(%v2064 init: %v2048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2066 = stablehlo.broadcast_in_dim %v2065, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2067 = stablehlo.multiply %v2061, %v2064 : tensor<256x256x14x14xf32>
    %v2068 = stablehlo.reduce(%v2067 init: %v2048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2069 = stablehlo.broadcast_in_dim %v2068, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2070 = stablehlo.multiply %v2064, %v2049 : tensor<256x256x14x14xf32>
    %v2071 = stablehlo.subtract %v2070, %v2066 : tensor<256x256x14x14xf32>
    %v2072 = stablehlo.multiply %v2061, %v2069 : tensor<256x256x14x14xf32>
    %v2073 = stablehlo.subtract %v2071, %v2072 : tensor<256x256x14x14xf32>
    %v2074 = stablehlo.divide %v2060, %v2049 : tensor<256x256x14x14xf32>
    %v2075 = stablehlo.multiply %v2074, %v2073 : tensor<256x256x14x14xf32>
    %v2076 = stablehlo.reshape %v2075 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2077 = stablehlo.reshape %v2076 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2078 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2079 = stablehlo.transpose %v2078, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2080 = stablehlo.convolution(%v2077, %v2079)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2081 = stablehlo.reshape %v2080 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2082 = stablehlo.add %v2081, %v2008 : tensor<256x50176xf32>
    %v2083 = stablehlo.reshape %v520 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2084 = stablehlo.reshape %v2076 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2085 = stablehlo.transpose %v2083, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2086 = stablehlo.transpose %v2084, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2087 = stablehlo.convolution(%v2085, %v2086)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2088 = stablehlo.transpose %v2087, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2089 = stablehlo.reshape %v525 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2090 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2091 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2092 = stablehlo.reduce(%v2089 init: %v2090) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2093 = stablehlo.broadcast_in_dim %v2092, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2094 = stablehlo.divide %v2093, %v2091 : tensor<256x256x14x14xf32>
    %v2095 = stablehlo.subtract %v2089, %v2094 : tensor<256x256x14x14xf32>
    %v2096 = stablehlo.multiply %v2095, %v2095 : tensor<256x256x14x14xf32>
    %v2097 = stablehlo.reduce(%v2096 init: %v2090) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2098 = stablehlo.broadcast_in_dim %v2097, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2099 = stablehlo.divide %v2098, %v2091 : tensor<256x256x14x14xf32>
    %v2100 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2101 = stablehlo.add %v2099, %v2100 : tensor<256x256x14x14xf32>
    %v2102 = stablehlo.rsqrt %v2101 : tensor<256x256x14x14xf32>
    %v2103 = stablehlo.multiply %v2095, %v2102 : tensor<256x256x14x14xf32>
    %v2104 = stablehlo.reshape %v2046 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2105 = stablehlo.multiply %v2104, %v2103 : tensor<256x256x14x14xf32>
    %v2106 = stablehlo.reduce(%v2105 init: %v2090) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2107 = stablehlo.reshape %v2046 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2108 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2109 = stablehlo.reduce(%v2107 init: %v2108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2110 = stablehlo.reshape %v547 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2111 = stablehlo.reshape %v2038 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2112 = stablehlo.transpose %v2110, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2113 = stablehlo.transpose %v2111, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2114 = stablehlo.convolution(%v2112, %v2113)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2115 = stablehlo.transpose %v2114, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2116 = stablehlo.reshape %v552 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2118 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2119 = stablehlo.reduce(%v2116 init: %v2117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2120 = stablehlo.broadcast_in_dim %v2119, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2121 = stablehlo.divide %v2120, %v2118 : tensor<256x256x14x14xf32>
    %v2122 = stablehlo.subtract %v2116, %v2121 : tensor<256x256x14x14xf32>
    %v2123 = stablehlo.multiply %v2122, %v2122 : tensor<256x256x14x14xf32>
    %v2124 = stablehlo.reduce(%v2123 init: %v2117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2125 = stablehlo.broadcast_in_dim %v2124, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2126 = stablehlo.divide %v2125, %v2118 : tensor<256x256x14x14xf32>
    %v2127 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2128 = stablehlo.add %v2126, %v2127 : tensor<256x256x14x14xf32>
    %v2129 = stablehlo.rsqrt %v2128 : tensor<256x256x14x14xf32>
    %v2130 = stablehlo.multiply %v2122, %v2129 : tensor<256x256x14x14xf32>
    %v2131 = stablehlo.reshape %v2008 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2132 = stablehlo.multiply %v2131, %v2130 : tensor<256x256x14x14xf32>
    %v2133 = stablehlo.reduce(%v2132 init: %v2117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2134 = stablehlo.reshape %v2008 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2135 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2136 = stablehlo.reduce(%v2134 init: %v2135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2137 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2138 = stablehlo.compare GT, %v518, %v2137 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2139 = stablehlo.select %v2138, %v2082, %v2137 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2140 = stablehlo.reshape %v472 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2141 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2142 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2143 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2144 = stablehlo.reduce(%v2140 init: %v2141) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2145 = stablehlo.broadcast_in_dim %v2144, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2146 = stablehlo.divide %v2145, %v2142 : tensor<256x256x14x14xf32>
    %v2147 = stablehlo.subtract %v2140, %v2146 : tensor<256x256x14x14xf32>
    %v2148 = stablehlo.multiply %v2147, %v2147 : tensor<256x256x14x14xf32>
    %v2149 = stablehlo.reduce(%v2148 init: %v2141) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2150 = stablehlo.broadcast_in_dim %v2149, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2151 = stablehlo.divide %v2150, %v2142 : tensor<256x256x14x14xf32>
    %v2152 = stablehlo.add %v2151, %v2143 : tensor<256x256x14x14xf32>
    %v2153 = stablehlo.rsqrt %v2152 : tensor<256x256x14x14xf32>
    %v2154 = stablehlo.multiply %v2147, %v2153 : tensor<256x256x14x14xf32>
    %v2155 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2156 = stablehlo.reshape %v2139 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2157 = stablehlo.multiply %v2155, %v2156 : tensor<256x256x14x14xf32>
    %v2158 = stablehlo.reduce(%v2157 init: %v2141) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2159 = stablehlo.broadcast_in_dim %v2158, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2160 = stablehlo.multiply %v2154, %v2157 : tensor<256x256x14x14xf32>
    %v2161 = stablehlo.reduce(%v2160 init: %v2141) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2162 = stablehlo.broadcast_in_dim %v2161, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2163 = stablehlo.multiply %v2157, %v2142 : tensor<256x256x14x14xf32>
    %v2164 = stablehlo.subtract %v2163, %v2159 : tensor<256x256x14x14xf32>
    %v2165 = stablehlo.multiply %v2154, %v2162 : tensor<256x256x14x14xf32>
    %v2166 = stablehlo.subtract %v2164, %v2165 : tensor<256x256x14x14xf32>
    %v2167 = stablehlo.divide %v2153, %v2142 : tensor<256x256x14x14xf32>
    %v2168 = stablehlo.multiply %v2167, %v2166 : tensor<256x256x14x14xf32>
    %v2169 = stablehlo.reshape %v2168 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2170 = stablehlo.reshape %v2169 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2171 = stablehlo.reverse %d3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2172 = stablehlo.transpose %v2171, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2173 = stablehlo.convolution(%v2170, %v2172)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2174 = stablehlo.reshape %v2173 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2175 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2176 = stablehlo.compare GT, %v465, %v2175 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2177 = stablehlo.select %v2176, %v2174, %v2175 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2178 = stablehlo.reshape %v445 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2179 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2180 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2181 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2182 = stablehlo.reduce(%v2178 init: %v2179) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2183 = stablehlo.broadcast_in_dim %v2182, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2184 = stablehlo.divide %v2183, %v2180 : tensor<256x256x14x14xf32>
    %v2185 = stablehlo.subtract %v2178, %v2184 : tensor<256x256x14x14xf32>
    %v2186 = stablehlo.multiply %v2185, %v2185 : tensor<256x256x14x14xf32>
    %v2187 = stablehlo.reduce(%v2186 init: %v2179) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2188 = stablehlo.broadcast_in_dim %v2187, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2189 = stablehlo.divide %v2188, %v2180 : tensor<256x256x14x14xf32>
    %v2190 = stablehlo.add %v2189, %v2181 : tensor<256x256x14x14xf32>
    %v2191 = stablehlo.rsqrt %v2190 : tensor<256x256x14x14xf32>
    %v2192 = stablehlo.multiply %v2185, %v2191 : tensor<256x256x14x14xf32>
    %v2193 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2194 = stablehlo.reshape %v2177 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2195 = stablehlo.multiply %v2193, %v2194 : tensor<256x256x14x14xf32>
    %v2196 = stablehlo.reduce(%v2195 init: %v2179) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2197 = stablehlo.broadcast_in_dim %v2196, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2198 = stablehlo.multiply %v2192, %v2195 : tensor<256x256x14x14xf32>
    %v2199 = stablehlo.reduce(%v2198 init: %v2179) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2200 = stablehlo.broadcast_in_dim %v2199, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2201 = stablehlo.multiply %v2195, %v2180 : tensor<256x256x14x14xf32>
    %v2202 = stablehlo.subtract %v2201, %v2197 : tensor<256x256x14x14xf32>
    %v2203 = stablehlo.multiply %v2192, %v2200 : tensor<256x256x14x14xf32>
    %v2204 = stablehlo.subtract %v2202, %v2203 : tensor<256x256x14x14xf32>
    %v2205 = stablehlo.divide %v2191, %v2180 : tensor<256x256x14x14xf32>
    %v2206 = stablehlo.multiply %v2205, %v2204 : tensor<256x256x14x14xf32>
    %v2207 = stablehlo.reshape %v2206 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2208 = stablehlo.reshape %v2207 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2209 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2210 = stablehlo.pad %v2208, %v2209, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256x28x28xf32>
    %v2211 = stablehlo.reverse %d3W1, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v2212 = stablehlo.transpose %v2211, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2213 = stablehlo.convolution(%v2210, %v2212)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2214 = stablehlo.reshape %v2213 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2215 = stablehlo.reshape %v497 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2216 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2217 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2218 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2219 = stablehlo.reduce(%v2215 init: %v2216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2220 = stablehlo.broadcast_in_dim %v2219, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2221 = stablehlo.divide %v2220, %v2217 : tensor<256x256x14x14xf32>
    %v2222 = stablehlo.subtract %v2215, %v2221 : tensor<256x256x14x14xf32>
    %v2223 = stablehlo.multiply %v2222, %v2222 : tensor<256x256x14x14xf32>
    %v2224 = stablehlo.reduce(%v2223 init: %v2216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2225 = stablehlo.broadcast_in_dim %v2224, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2226 = stablehlo.divide %v2225, %v2217 : tensor<256x256x14x14xf32>
    %v2227 = stablehlo.add %v2226, %v2218 : tensor<256x256x14x14xf32>
    %v2228 = stablehlo.rsqrt %v2227 : tensor<256x256x14x14xf32>
    %v2229 = stablehlo.multiply %v2222, %v2228 : tensor<256x256x14x14xf32>
    %v2230 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2231 = stablehlo.reshape %v2139 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2232 = stablehlo.multiply %v2230, %v2231 : tensor<256x256x14x14xf32>
    %v2233 = stablehlo.reduce(%v2232 init: %v2216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2234 = stablehlo.broadcast_in_dim %v2233, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2235 = stablehlo.multiply %v2229, %v2232 : tensor<256x256x14x14xf32>
    %v2236 = stablehlo.reduce(%v2235 init: %v2216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2237 = stablehlo.broadcast_in_dim %v2236, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2238 = stablehlo.multiply %v2232, %v2217 : tensor<256x256x14x14xf32>
    %v2239 = stablehlo.subtract %v2238, %v2234 : tensor<256x256x14x14xf32>
    %v2240 = stablehlo.multiply %v2229, %v2237 : tensor<256x256x14x14xf32>
    %v2241 = stablehlo.subtract %v2239, %v2240 : tensor<256x256x14x14xf32>
    %v2242 = stablehlo.divide %v2228, %v2217 : tensor<256x256x14x14xf32>
    %v2243 = stablehlo.multiply %v2242, %v2241 : tensor<256x256x14x14xf32>
    %v2244 = stablehlo.reshape %v2243 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2245 = stablehlo.reshape %v2244 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2247 = stablehlo.pad %v2245, %v2246, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256x28x28xf32>
    %v2248 = stablehlo.reverse %d3Wp, dims = [2, 3] : tensor<256x128x1x1xf32>
    %v2249 = stablehlo.transpose %v2248, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v2250 = stablehlo.convolution(%v2247, %v2249)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<128x256x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v2251 = stablehlo.reshape %v2250 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2252 = stablehlo.add %v2214, %v2251 : tensor<256x100352xf32>
    %v2253 = stablehlo.reshape %v440 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2254 = stablehlo.reshape %v2207 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2255 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2256 = stablehlo.pad %v2254, %v2255, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256x28x28xf32>
    %v2257 = stablehlo.transpose %v2253, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2258 = stablehlo.transpose %v2256, dims = [1, 0, 2, 3] : (tensor<256x256x28x28xf32>) -> tensor<256x256x28x28xf32>
    %v2259 = stablehlo.convolution(%v2257, %v2258)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<256x256x28x28xf32>) -> tensor<128x256x3x3xf32>
    %v2260 = stablehlo.transpose %v2259, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2261 = stablehlo.reshape %v445 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2262 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2263 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2264 = stablehlo.reduce(%v2261 init: %v2262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2265 = stablehlo.broadcast_in_dim %v2264, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2266 = stablehlo.divide %v2265, %v2263 : tensor<256x256x14x14xf32>
    %v2267 = stablehlo.subtract %v2261, %v2266 : tensor<256x256x14x14xf32>
    %v2268 = stablehlo.multiply %v2267, %v2267 : tensor<256x256x14x14xf32>
    %v2269 = stablehlo.reduce(%v2268 init: %v2262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2270 = stablehlo.broadcast_in_dim %v2269, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2271 = stablehlo.divide %v2270, %v2263 : tensor<256x256x14x14xf32>
    %v2272 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2273 = stablehlo.add %v2271, %v2272 : tensor<256x256x14x14xf32>
    %v2274 = stablehlo.rsqrt %v2273 : tensor<256x256x14x14xf32>
    %v2275 = stablehlo.multiply %v2267, %v2274 : tensor<256x256x14x14xf32>
    %v2276 = stablehlo.reshape %v2177 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2277 = stablehlo.multiply %v2276, %v2275 : tensor<256x256x14x14xf32>
    %v2278 = stablehlo.reduce(%v2277 init: %v2262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2279 = stablehlo.reshape %v2177 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2281 = stablehlo.reduce(%v2279 init: %v2280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2282 = stablehlo.reshape %v467 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2283 = stablehlo.reshape %v2169 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2284 = stablehlo.transpose %v2282, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2285 = stablehlo.transpose %v2283, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2286 = stablehlo.convolution(%v2284, %v2285)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2287 = stablehlo.transpose %v2286, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2288 = stablehlo.reshape %v472 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2290 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2291 = stablehlo.reduce(%v2288 init: %v2289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2292 = stablehlo.broadcast_in_dim %v2291, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2293 = stablehlo.divide %v2292, %v2290 : tensor<256x256x14x14xf32>
    %v2294 = stablehlo.subtract %v2288, %v2293 : tensor<256x256x14x14xf32>
    %v2295 = stablehlo.multiply %v2294, %v2294 : tensor<256x256x14x14xf32>
    %v2296 = stablehlo.reduce(%v2295 init: %v2289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2297 = stablehlo.broadcast_in_dim %v2296, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2298 = stablehlo.divide %v2297, %v2290 : tensor<256x256x14x14xf32>
    %v2299 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2300 = stablehlo.add %v2298, %v2299 : tensor<256x256x14x14xf32>
    %v2301 = stablehlo.rsqrt %v2300 : tensor<256x256x14x14xf32>
    %v2302 = stablehlo.multiply %v2294, %v2301 : tensor<256x256x14x14xf32>
    %v2303 = stablehlo.reshape %v2139 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2304 = stablehlo.multiply %v2303, %v2302 : tensor<256x256x14x14xf32>
    %v2305 = stablehlo.reduce(%v2304 init: %v2289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2306 = stablehlo.reshape %v2139 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2307 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2308 = stablehlo.reduce(%v2306 init: %v2307) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2309 = stablehlo.reshape %v440 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2310 = stablehlo.reshape %v2244 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2311 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2312 = stablehlo.pad %v2310, %v2311, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256x28x28xf32>
    %v2313 = stablehlo.transpose %v2309, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2314 = stablehlo.transpose %v2312, dims = [1, 0, 2, 3] : (tensor<256x256x28x28xf32>) -> tensor<256x256x28x28xf32>
    %v2315 = stablehlo.convolution(%v2313, %v2314)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<256x256x28x28xf32>) -> tensor<128x256x1x1xf32>
    %v2316 = stablehlo.transpose %v2315, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v2317 = stablehlo.reshape %v497 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2318 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2319 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2320 = stablehlo.reduce(%v2317 init: %v2318) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2321 = stablehlo.broadcast_in_dim %v2320, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2322 = stablehlo.divide %v2321, %v2319 : tensor<256x256x14x14xf32>
    %v2323 = stablehlo.subtract %v2317, %v2322 : tensor<256x256x14x14xf32>
    %v2324 = stablehlo.multiply %v2323, %v2323 : tensor<256x256x14x14xf32>
    %v2325 = stablehlo.reduce(%v2324 init: %v2318) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2326 = stablehlo.broadcast_in_dim %v2325, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2327 = stablehlo.divide %v2326, %v2319 : tensor<256x256x14x14xf32>
    %v2328 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2329 = stablehlo.add %v2327, %v2328 : tensor<256x256x14x14xf32>
    %v2330 = stablehlo.rsqrt %v2329 : tensor<256x256x14x14xf32>
    %v2331 = stablehlo.multiply %v2323, %v2330 : tensor<256x256x14x14xf32>
    %v2332 = stablehlo.reshape %v2139 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2333 = stablehlo.multiply %v2332, %v2331 : tensor<256x256x14x14xf32>
    %v2334 = stablehlo.reduce(%v2333 init: %v2318) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2335 = stablehlo.reshape %v2139 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2336 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2337 = stablehlo.reduce(%v2335 init: %v2336) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2338 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2339 = stablehlo.compare GT, %v438, %v2338 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2340 = stablehlo.select %v2339, %v2252, %v2338 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2341 = stablehlo.reshape %v417 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2342 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2343 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2344 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2345 = stablehlo.reduce(%v2341 init: %v2342) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2346 = stablehlo.broadcast_in_dim %v2345, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2347 = stablehlo.divide %v2346, %v2343 : tensor<256x128x28x28xf32>
    %v2348 = stablehlo.subtract %v2341, %v2347 : tensor<256x128x28x28xf32>
    %v2349 = stablehlo.multiply %v2348, %v2348 : tensor<256x128x28x28xf32>
    %v2350 = stablehlo.reduce(%v2349 init: %v2342) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2351 = stablehlo.broadcast_in_dim %v2350, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2352 = stablehlo.divide %v2351, %v2343 : tensor<256x128x28x28xf32>
    %v2353 = stablehlo.add %v2352, %v2344 : tensor<256x128x28x28xf32>
    %v2354 = stablehlo.rsqrt %v2353 : tensor<256x128x28x28xf32>
    %v2355 = stablehlo.multiply %v2348, %v2354 : tensor<256x128x28x28xf32>
    %v2356 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2357 = stablehlo.reshape %v2340 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2358 = stablehlo.multiply %v2356, %v2357 : tensor<256x128x28x28xf32>
    %v2359 = stablehlo.reduce(%v2358 init: %v2342) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2360 = stablehlo.broadcast_in_dim %v2359, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2361 = stablehlo.multiply %v2355, %v2358 : tensor<256x128x28x28xf32>
    %v2362 = stablehlo.reduce(%v2361 init: %v2342) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2363 = stablehlo.broadcast_in_dim %v2362, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2364 = stablehlo.multiply %v2358, %v2343 : tensor<256x128x28x28xf32>
    %v2365 = stablehlo.subtract %v2364, %v2360 : tensor<256x128x28x28xf32>
    %v2366 = stablehlo.multiply %v2355, %v2363 : tensor<256x128x28x28xf32>
    %v2367 = stablehlo.subtract %v2365, %v2366 : tensor<256x128x28x28xf32>
    %v2368 = stablehlo.divide %v2354, %v2343 : tensor<256x128x28x28xf32>
    %v2369 = stablehlo.multiply %v2368, %v2367 : tensor<256x128x28x28xf32>
    %v2370 = stablehlo.reshape %v2369 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2371 = stablehlo.reshape %v2370 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2372 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2373 = stablehlo.transpose %v2372, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2374 = stablehlo.convolution(%v2371, %v2373)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2375 = stablehlo.reshape %v2374 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2376 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2377 = stablehlo.compare GT, %v410, %v2376 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2378 = stablehlo.select %v2377, %v2375, %v2376 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2379 = stablehlo.reshape %v390 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2380 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2381 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2382 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2383 = stablehlo.reduce(%v2379 init: %v2380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2384 = stablehlo.broadcast_in_dim %v2383, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2385 = stablehlo.divide %v2384, %v2381 : tensor<256x128x28x28xf32>
    %v2386 = stablehlo.subtract %v2379, %v2385 : tensor<256x128x28x28xf32>
    %v2387 = stablehlo.multiply %v2386, %v2386 : tensor<256x128x28x28xf32>
    %v2388 = stablehlo.reduce(%v2387 init: %v2380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2389 = stablehlo.broadcast_in_dim %v2388, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2390 = stablehlo.divide %v2389, %v2381 : tensor<256x128x28x28xf32>
    %v2391 = stablehlo.add %v2390, %v2382 : tensor<256x128x28x28xf32>
    %v2392 = stablehlo.rsqrt %v2391 : tensor<256x128x28x28xf32>
    %v2393 = stablehlo.multiply %v2386, %v2392 : tensor<256x128x28x28xf32>
    %v2394 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2395 = stablehlo.reshape %v2378 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2396 = stablehlo.multiply %v2394, %v2395 : tensor<256x128x28x28xf32>
    %v2397 = stablehlo.reduce(%v2396 init: %v2380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2398 = stablehlo.broadcast_in_dim %v2397, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2399 = stablehlo.multiply %v2393, %v2396 : tensor<256x128x28x28xf32>
    %v2400 = stablehlo.reduce(%v2399 init: %v2380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2401 = stablehlo.broadcast_in_dim %v2400, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2402 = stablehlo.multiply %v2396, %v2381 : tensor<256x128x28x28xf32>
    %v2403 = stablehlo.subtract %v2402, %v2398 : tensor<256x128x28x28xf32>
    %v2404 = stablehlo.multiply %v2393, %v2401 : tensor<256x128x28x28xf32>
    %v2405 = stablehlo.subtract %v2403, %v2404 : tensor<256x128x28x28xf32>
    %v2406 = stablehlo.divide %v2392, %v2381 : tensor<256x128x28x28xf32>
    %v2407 = stablehlo.multiply %v2406, %v2405 : tensor<256x128x28x28xf32>
    %v2408 = stablehlo.reshape %v2407 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2409 = stablehlo.reshape %v2408 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2410 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2411 = stablehlo.transpose %v2410, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2412 = stablehlo.convolution(%v2409, %v2411)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2413 = stablehlo.reshape %v2412 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2414 = stablehlo.add %v2413, %v2340 : tensor<256x100352xf32>
    %v2415 = stablehlo.reshape %v385 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2416 = stablehlo.reshape %v2408 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2417 = stablehlo.transpose %v2415, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2418 = stablehlo.transpose %v2416, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2419 = stablehlo.convolution(%v2417, %v2418)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2420 = stablehlo.transpose %v2419, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2421 = stablehlo.reshape %v390 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2423 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2424 = stablehlo.reduce(%v2421 init: %v2422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2425 = stablehlo.broadcast_in_dim %v2424, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2426 = stablehlo.divide %v2425, %v2423 : tensor<256x128x28x28xf32>
    %v2427 = stablehlo.subtract %v2421, %v2426 : tensor<256x128x28x28xf32>
    %v2428 = stablehlo.multiply %v2427, %v2427 : tensor<256x128x28x28xf32>
    %v2429 = stablehlo.reduce(%v2428 init: %v2422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2430 = stablehlo.broadcast_in_dim %v2429, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2431 = stablehlo.divide %v2430, %v2423 : tensor<256x128x28x28xf32>
    %v2432 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2433 = stablehlo.add %v2431, %v2432 : tensor<256x128x28x28xf32>
    %v2434 = stablehlo.rsqrt %v2433 : tensor<256x128x28x28xf32>
    %v2435 = stablehlo.multiply %v2427, %v2434 : tensor<256x128x28x28xf32>
    %v2436 = stablehlo.reshape %v2378 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2437 = stablehlo.multiply %v2436, %v2435 : tensor<256x128x28x28xf32>
    %v2438 = stablehlo.reduce(%v2437 init: %v2422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2439 = stablehlo.reshape %v2378 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2440 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2441 = stablehlo.reduce(%v2439 init: %v2440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2442 = stablehlo.reshape %v412 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2443 = stablehlo.reshape %v2370 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2444 = stablehlo.transpose %v2442, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2445 = stablehlo.transpose %v2443, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2446 = stablehlo.convolution(%v2444, %v2445)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2447 = stablehlo.transpose %v2446, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2448 = stablehlo.reshape %v417 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2450 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2451 = stablehlo.reduce(%v2448 init: %v2449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2452 = stablehlo.broadcast_in_dim %v2451, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2453 = stablehlo.divide %v2452, %v2450 : tensor<256x128x28x28xf32>
    %v2454 = stablehlo.subtract %v2448, %v2453 : tensor<256x128x28x28xf32>
    %v2455 = stablehlo.multiply %v2454, %v2454 : tensor<256x128x28x28xf32>
    %v2456 = stablehlo.reduce(%v2455 init: %v2449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2457 = stablehlo.broadcast_in_dim %v2456, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2458 = stablehlo.divide %v2457, %v2450 : tensor<256x128x28x28xf32>
    %v2459 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2460 = stablehlo.add %v2458, %v2459 : tensor<256x128x28x28xf32>
    %v2461 = stablehlo.rsqrt %v2460 : tensor<256x128x28x28xf32>
    %v2462 = stablehlo.multiply %v2454, %v2461 : tensor<256x128x28x28xf32>
    %v2463 = stablehlo.reshape %v2340 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2464 = stablehlo.multiply %v2463, %v2462 : tensor<256x128x28x28xf32>
    %v2465 = stablehlo.reduce(%v2464 init: %v2449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2466 = stablehlo.reshape %v2340 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2467 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2468 = stablehlo.reduce(%v2466 init: %v2467) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2469 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2470 = stablehlo.compare GT, %v383, %v2469 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2471 = stablehlo.select %v2470, %v2414, %v2469 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2472 = stablehlo.reshape %v362 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2474 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2475 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2476 = stablehlo.reduce(%v2472 init: %v2473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2477 = stablehlo.broadcast_in_dim %v2476, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2478 = stablehlo.divide %v2477, %v2474 : tensor<256x128x28x28xf32>
    %v2479 = stablehlo.subtract %v2472, %v2478 : tensor<256x128x28x28xf32>
    %v2480 = stablehlo.multiply %v2479, %v2479 : tensor<256x128x28x28xf32>
    %v2481 = stablehlo.reduce(%v2480 init: %v2473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2482 = stablehlo.broadcast_in_dim %v2481, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2483 = stablehlo.divide %v2482, %v2474 : tensor<256x128x28x28xf32>
    %v2484 = stablehlo.add %v2483, %v2475 : tensor<256x128x28x28xf32>
    %v2485 = stablehlo.rsqrt %v2484 : tensor<256x128x28x28xf32>
    %v2486 = stablehlo.multiply %v2479, %v2485 : tensor<256x128x28x28xf32>
    %v2487 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2488 = stablehlo.reshape %v2471 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2489 = stablehlo.multiply %v2487, %v2488 : tensor<256x128x28x28xf32>
    %v2490 = stablehlo.reduce(%v2489 init: %v2473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2491 = stablehlo.broadcast_in_dim %v2490, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2492 = stablehlo.multiply %v2486, %v2489 : tensor<256x128x28x28xf32>
    %v2493 = stablehlo.reduce(%v2492 init: %v2473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2494 = stablehlo.broadcast_in_dim %v2493, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2495 = stablehlo.multiply %v2489, %v2474 : tensor<256x128x28x28xf32>
    %v2496 = stablehlo.subtract %v2495, %v2491 : tensor<256x128x28x28xf32>
    %v2497 = stablehlo.multiply %v2486, %v2494 : tensor<256x128x28x28xf32>
    %v2498 = stablehlo.subtract %v2496, %v2497 : tensor<256x128x28x28xf32>
    %v2499 = stablehlo.divide %v2485, %v2474 : tensor<256x128x28x28xf32>
    %v2500 = stablehlo.multiply %v2499, %v2498 : tensor<256x128x28x28xf32>
    %v2501 = stablehlo.reshape %v2500 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2502 = stablehlo.reshape %v2501 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2503 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2504 = stablehlo.transpose %v2503, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2505 = stablehlo.convolution(%v2502, %v2504)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2506 = stablehlo.reshape %v2505 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2507 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2508 = stablehlo.compare GT, %v355, %v2507 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2509 = stablehlo.select %v2508, %v2506, %v2507 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2510 = stablehlo.reshape %v335 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2512 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2513 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2514 = stablehlo.reduce(%v2510 init: %v2511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2515 = stablehlo.broadcast_in_dim %v2514, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2516 = stablehlo.divide %v2515, %v2512 : tensor<256x128x28x28xf32>
    %v2517 = stablehlo.subtract %v2510, %v2516 : tensor<256x128x28x28xf32>
    %v2518 = stablehlo.multiply %v2517, %v2517 : tensor<256x128x28x28xf32>
    %v2519 = stablehlo.reduce(%v2518 init: %v2511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2520 = stablehlo.broadcast_in_dim %v2519, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2521 = stablehlo.divide %v2520, %v2512 : tensor<256x128x28x28xf32>
    %v2522 = stablehlo.add %v2521, %v2513 : tensor<256x128x28x28xf32>
    %v2523 = stablehlo.rsqrt %v2522 : tensor<256x128x28x28xf32>
    %v2524 = stablehlo.multiply %v2517, %v2523 : tensor<256x128x28x28xf32>
    %v2525 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2526 = stablehlo.reshape %v2509 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2527 = stablehlo.multiply %v2525, %v2526 : tensor<256x128x28x28xf32>
    %v2528 = stablehlo.reduce(%v2527 init: %v2511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2529 = stablehlo.broadcast_in_dim %v2528, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2530 = stablehlo.multiply %v2524, %v2527 : tensor<256x128x28x28xf32>
    %v2531 = stablehlo.reduce(%v2530 init: %v2511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2532 = stablehlo.broadcast_in_dim %v2531, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2533 = stablehlo.multiply %v2527, %v2512 : tensor<256x128x28x28xf32>
    %v2534 = stablehlo.subtract %v2533, %v2529 : tensor<256x128x28x28xf32>
    %v2535 = stablehlo.multiply %v2524, %v2532 : tensor<256x128x28x28xf32>
    %v2536 = stablehlo.subtract %v2534, %v2535 : tensor<256x128x28x28xf32>
    %v2537 = stablehlo.divide %v2523, %v2512 : tensor<256x128x28x28xf32>
    %v2538 = stablehlo.multiply %v2537, %v2536 : tensor<256x128x28x28xf32>
    %v2539 = stablehlo.reshape %v2538 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2540 = stablehlo.reshape %v2539 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2541 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2542 = stablehlo.transpose %v2541, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2543 = stablehlo.convolution(%v2540, %v2542)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2544 = stablehlo.reshape %v2543 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2545 = stablehlo.add %v2544, %v2471 : tensor<256x100352xf32>
    %v2546 = stablehlo.reshape %v330 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2547 = stablehlo.reshape %v2539 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2548 = stablehlo.transpose %v2546, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2549 = stablehlo.transpose %v2547, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2550 = stablehlo.convolution(%v2548, %v2549)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2551 = stablehlo.transpose %v2550, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2552 = stablehlo.reshape %v335 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2553 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2554 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2555 = stablehlo.reduce(%v2552 init: %v2553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2556 = stablehlo.broadcast_in_dim %v2555, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2557 = stablehlo.divide %v2556, %v2554 : tensor<256x128x28x28xf32>
    %v2558 = stablehlo.subtract %v2552, %v2557 : tensor<256x128x28x28xf32>
    %v2559 = stablehlo.multiply %v2558, %v2558 : tensor<256x128x28x28xf32>
    %v2560 = stablehlo.reduce(%v2559 init: %v2553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2561 = stablehlo.broadcast_in_dim %v2560, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2562 = stablehlo.divide %v2561, %v2554 : tensor<256x128x28x28xf32>
    %v2563 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2564 = stablehlo.add %v2562, %v2563 : tensor<256x128x28x28xf32>
    %v2565 = stablehlo.rsqrt %v2564 : tensor<256x128x28x28xf32>
    %v2566 = stablehlo.multiply %v2558, %v2565 : tensor<256x128x28x28xf32>
    %v2567 = stablehlo.reshape %v2509 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2568 = stablehlo.multiply %v2567, %v2566 : tensor<256x128x28x28xf32>
    %v2569 = stablehlo.reduce(%v2568 init: %v2553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2570 = stablehlo.reshape %v2509 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2571 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2572 = stablehlo.reduce(%v2570 init: %v2571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2573 = stablehlo.reshape %v357 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2574 = stablehlo.reshape %v2501 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2575 = stablehlo.transpose %v2573, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2576 = stablehlo.transpose %v2574, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2577 = stablehlo.convolution(%v2575, %v2576)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2578 = stablehlo.transpose %v2577, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2579 = stablehlo.reshape %v362 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2580 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2581 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2582 = stablehlo.reduce(%v2579 init: %v2580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2583 = stablehlo.broadcast_in_dim %v2582, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2584 = stablehlo.divide %v2583, %v2581 : tensor<256x128x28x28xf32>
    %v2585 = stablehlo.subtract %v2579, %v2584 : tensor<256x128x28x28xf32>
    %v2586 = stablehlo.multiply %v2585, %v2585 : tensor<256x128x28x28xf32>
    %v2587 = stablehlo.reduce(%v2586 init: %v2580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2588 = stablehlo.broadcast_in_dim %v2587, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2589 = stablehlo.divide %v2588, %v2581 : tensor<256x128x28x28xf32>
    %v2590 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2591 = stablehlo.add %v2589, %v2590 : tensor<256x128x28x28xf32>
    %v2592 = stablehlo.rsqrt %v2591 : tensor<256x128x28x28xf32>
    %v2593 = stablehlo.multiply %v2585, %v2592 : tensor<256x128x28x28xf32>
    %v2594 = stablehlo.reshape %v2471 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2595 = stablehlo.multiply %v2594, %v2593 : tensor<256x128x28x28xf32>
    %v2596 = stablehlo.reduce(%v2595 init: %v2580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2597 = stablehlo.reshape %v2471 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2598 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2599 = stablehlo.reduce(%v2597 init: %v2598) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2600 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2601 = stablehlo.compare GT, %v328, %v2600 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2602 = stablehlo.select %v2601, %v2545, %v2600 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2603 = stablehlo.reshape %v307 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2604 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2605 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2606 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2607 = stablehlo.reduce(%v2603 init: %v2604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2608 = stablehlo.broadcast_in_dim %v2607, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2609 = stablehlo.divide %v2608, %v2605 : tensor<256x128x28x28xf32>
    %v2610 = stablehlo.subtract %v2603, %v2609 : tensor<256x128x28x28xf32>
    %v2611 = stablehlo.multiply %v2610, %v2610 : tensor<256x128x28x28xf32>
    %v2612 = stablehlo.reduce(%v2611 init: %v2604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2613 = stablehlo.broadcast_in_dim %v2612, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2614 = stablehlo.divide %v2613, %v2605 : tensor<256x128x28x28xf32>
    %v2615 = stablehlo.add %v2614, %v2606 : tensor<256x128x28x28xf32>
    %v2616 = stablehlo.rsqrt %v2615 : tensor<256x128x28x28xf32>
    %v2617 = stablehlo.multiply %v2610, %v2616 : tensor<256x128x28x28xf32>
    %v2618 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2619 = stablehlo.reshape %v2602 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2620 = stablehlo.multiply %v2618, %v2619 : tensor<256x128x28x28xf32>
    %v2621 = stablehlo.reduce(%v2620 init: %v2604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2622 = stablehlo.broadcast_in_dim %v2621, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2623 = stablehlo.multiply %v2617, %v2620 : tensor<256x128x28x28xf32>
    %v2624 = stablehlo.reduce(%v2623 init: %v2604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2625 = stablehlo.broadcast_in_dim %v2624, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2626 = stablehlo.multiply %v2620, %v2605 : tensor<256x128x28x28xf32>
    %v2627 = stablehlo.subtract %v2626, %v2622 : tensor<256x128x28x28xf32>
    %v2628 = stablehlo.multiply %v2617, %v2625 : tensor<256x128x28x28xf32>
    %v2629 = stablehlo.subtract %v2627, %v2628 : tensor<256x128x28x28xf32>
    %v2630 = stablehlo.divide %v2616, %v2605 : tensor<256x128x28x28xf32>
    %v2631 = stablehlo.multiply %v2630, %v2629 : tensor<256x128x28x28xf32>
    %v2632 = stablehlo.reshape %v2631 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2633 = stablehlo.reshape %v2632 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2634 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2635 = stablehlo.transpose %v2634, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2636 = stablehlo.convolution(%v2633, %v2635)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2637 = stablehlo.reshape %v2636 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2638 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2639 = stablehlo.compare GT, %v300, %v2638 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2640 = stablehlo.select %v2639, %v2637, %v2638 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2641 = stablehlo.reshape %v280 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2642 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2643 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2644 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2645 = stablehlo.reduce(%v2641 init: %v2642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2646 = stablehlo.broadcast_in_dim %v2645, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2647 = stablehlo.divide %v2646, %v2643 : tensor<256x128x28x28xf32>
    %v2648 = stablehlo.subtract %v2641, %v2647 : tensor<256x128x28x28xf32>
    %v2649 = stablehlo.multiply %v2648, %v2648 : tensor<256x128x28x28xf32>
    %v2650 = stablehlo.reduce(%v2649 init: %v2642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2651 = stablehlo.broadcast_in_dim %v2650, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2652 = stablehlo.divide %v2651, %v2643 : tensor<256x128x28x28xf32>
    %v2653 = stablehlo.add %v2652, %v2644 : tensor<256x128x28x28xf32>
    %v2654 = stablehlo.rsqrt %v2653 : tensor<256x128x28x28xf32>
    %v2655 = stablehlo.multiply %v2648, %v2654 : tensor<256x128x28x28xf32>
    %v2656 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2657 = stablehlo.reshape %v2640 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2658 = stablehlo.multiply %v2656, %v2657 : tensor<256x128x28x28xf32>
    %v2659 = stablehlo.reduce(%v2658 init: %v2642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2660 = stablehlo.broadcast_in_dim %v2659, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2661 = stablehlo.multiply %v2655, %v2658 : tensor<256x128x28x28xf32>
    %v2662 = stablehlo.reduce(%v2661 init: %v2642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2663 = stablehlo.broadcast_in_dim %v2662, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2664 = stablehlo.multiply %v2658, %v2643 : tensor<256x128x28x28xf32>
    %v2665 = stablehlo.subtract %v2664, %v2660 : tensor<256x128x28x28xf32>
    %v2666 = stablehlo.multiply %v2655, %v2663 : tensor<256x128x28x28xf32>
    %v2667 = stablehlo.subtract %v2665, %v2666 : tensor<256x128x28x28xf32>
    %v2668 = stablehlo.divide %v2654, %v2643 : tensor<256x128x28x28xf32>
    %v2669 = stablehlo.multiply %v2668, %v2667 : tensor<256x128x28x28xf32>
    %v2670 = stablehlo.reshape %v2669 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2671 = stablehlo.reshape %v2670 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2672 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2673 = stablehlo.transpose %v2672, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2674 = stablehlo.convolution(%v2671, %v2673)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2675 = stablehlo.reshape %v2674 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2676 = stablehlo.add %v2675, %v2602 : tensor<256x100352xf32>
    %v2677 = stablehlo.reshape %v275 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2678 = stablehlo.reshape %v2670 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2679 = stablehlo.transpose %v2677, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2680 = stablehlo.transpose %v2678, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2681 = stablehlo.convolution(%v2679, %v2680)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2682 = stablehlo.transpose %v2681, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2683 = stablehlo.reshape %v280 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2684 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2685 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2686 = stablehlo.reduce(%v2683 init: %v2684) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2687 = stablehlo.broadcast_in_dim %v2686, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2688 = stablehlo.divide %v2687, %v2685 : tensor<256x128x28x28xf32>
    %v2689 = stablehlo.subtract %v2683, %v2688 : tensor<256x128x28x28xf32>
    %v2690 = stablehlo.multiply %v2689, %v2689 : tensor<256x128x28x28xf32>
    %v2691 = stablehlo.reduce(%v2690 init: %v2684) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2692 = stablehlo.broadcast_in_dim %v2691, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2693 = stablehlo.divide %v2692, %v2685 : tensor<256x128x28x28xf32>
    %v2694 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2695 = stablehlo.add %v2693, %v2694 : tensor<256x128x28x28xf32>
    %v2696 = stablehlo.rsqrt %v2695 : tensor<256x128x28x28xf32>
    %v2697 = stablehlo.multiply %v2689, %v2696 : tensor<256x128x28x28xf32>
    %v2698 = stablehlo.reshape %v2640 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2699 = stablehlo.multiply %v2698, %v2697 : tensor<256x128x28x28xf32>
    %v2700 = stablehlo.reduce(%v2699 init: %v2684) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2701 = stablehlo.reshape %v2640 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2703 = stablehlo.reduce(%v2701 init: %v2702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2704 = stablehlo.reshape %v302 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2705 = stablehlo.reshape %v2632 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2706 = stablehlo.transpose %v2704, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2707 = stablehlo.transpose %v2705, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2708 = stablehlo.convolution(%v2706, %v2707)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2709 = stablehlo.transpose %v2708, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2710 = stablehlo.reshape %v307 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2712 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2713 = stablehlo.reduce(%v2710 init: %v2711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2714 = stablehlo.broadcast_in_dim %v2713, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2715 = stablehlo.divide %v2714, %v2712 : tensor<256x128x28x28xf32>
    %v2716 = stablehlo.subtract %v2710, %v2715 : tensor<256x128x28x28xf32>
    %v2717 = stablehlo.multiply %v2716, %v2716 : tensor<256x128x28x28xf32>
    %v2718 = stablehlo.reduce(%v2717 init: %v2711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2719 = stablehlo.broadcast_in_dim %v2718, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2720 = stablehlo.divide %v2719, %v2712 : tensor<256x128x28x28xf32>
    %v2721 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2722 = stablehlo.add %v2720, %v2721 : tensor<256x128x28x28xf32>
    %v2723 = stablehlo.rsqrt %v2722 : tensor<256x128x28x28xf32>
    %v2724 = stablehlo.multiply %v2716, %v2723 : tensor<256x128x28x28xf32>
    %v2725 = stablehlo.reshape %v2602 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2726 = stablehlo.multiply %v2725, %v2724 : tensor<256x128x28x28xf32>
    %v2727 = stablehlo.reduce(%v2726 init: %v2711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2728 = stablehlo.reshape %v2602 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2730 = stablehlo.reduce(%v2728 init: %v2729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2731 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2732 = stablehlo.compare GT, %v273, %v2731 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2733 = stablehlo.select %v2732, %v2676, %v2731 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2734 = stablehlo.reshape %v227 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2736 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2737 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2738 = stablehlo.reduce(%v2734 init: %v2735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2739 = stablehlo.broadcast_in_dim %v2738, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2740 = stablehlo.divide %v2739, %v2736 : tensor<256x128x28x28xf32>
    %v2741 = stablehlo.subtract %v2734, %v2740 : tensor<256x128x28x28xf32>
    %v2742 = stablehlo.multiply %v2741, %v2741 : tensor<256x128x28x28xf32>
    %v2743 = stablehlo.reduce(%v2742 init: %v2735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2744 = stablehlo.broadcast_in_dim %v2743, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2745 = stablehlo.divide %v2744, %v2736 : tensor<256x128x28x28xf32>
    %v2746 = stablehlo.add %v2745, %v2737 : tensor<256x128x28x28xf32>
    %v2747 = stablehlo.rsqrt %v2746 : tensor<256x128x28x28xf32>
    %v2748 = stablehlo.multiply %v2741, %v2747 : tensor<256x128x28x28xf32>
    %v2749 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2750 = stablehlo.reshape %v2733 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2751 = stablehlo.multiply %v2749, %v2750 : tensor<256x128x28x28xf32>
    %v2752 = stablehlo.reduce(%v2751 init: %v2735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2753 = stablehlo.broadcast_in_dim %v2752, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2754 = stablehlo.multiply %v2748, %v2751 : tensor<256x128x28x28xf32>
    %v2755 = stablehlo.reduce(%v2754 init: %v2735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2756 = stablehlo.broadcast_in_dim %v2755, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2757 = stablehlo.multiply %v2751, %v2736 : tensor<256x128x28x28xf32>
    %v2758 = stablehlo.subtract %v2757, %v2753 : tensor<256x128x28x28xf32>
    %v2759 = stablehlo.multiply %v2748, %v2756 : tensor<256x128x28x28xf32>
    %v2760 = stablehlo.subtract %v2758, %v2759 : tensor<256x128x28x28xf32>
    %v2761 = stablehlo.divide %v2747, %v2736 : tensor<256x128x28x28xf32>
    %v2762 = stablehlo.multiply %v2761, %v2760 : tensor<256x128x28x28xf32>
    %v2763 = stablehlo.reshape %v2762 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2764 = stablehlo.reshape %v2763 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2765 = stablehlo.reverse %d2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2766 = stablehlo.transpose %v2765, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2767 = stablehlo.convolution(%v2764, %v2766)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2768 = stablehlo.reshape %v2767 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2769 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2770 = stablehlo.compare GT, %v220, %v2769 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2771 = stablehlo.select %v2770, %v2768, %v2769 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2772 = stablehlo.reshape %v200 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2773 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2774 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2775 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2776 = stablehlo.reduce(%v2772 init: %v2773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2777 = stablehlo.broadcast_in_dim %v2776, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2778 = stablehlo.divide %v2777, %v2774 : tensor<256x128x28x28xf32>
    %v2779 = stablehlo.subtract %v2772, %v2778 : tensor<256x128x28x28xf32>
    %v2780 = stablehlo.multiply %v2779, %v2779 : tensor<256x128x28x28xf32>
    %v2781 = stablehlo.reduce(%v2780 init: %v2773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2782 = stablehlo.broadcast_in_dim %v2781, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2783 = stablehlo.divide %v2782, %v2774 : tensor<256x128x28x28xf32>
    %v2784 = stablehlo.add %v2783, %v2775 : tensor<256x128x28x28xf32>
    %v2785 = stablehlo.rsqrt %v2784 : tensor<256x128x28x28xf32>
    %v2786 = stablehlo.multiply %v2779, %v2785 : tensor<256x128x28x28xf32>
    %v2787 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2788 = stablehlo.reshape %v2771 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2789 = stablehlo.multiply %v2787, %v2788 : tensor<256x128x28x28xf32>
    %v2790 = stablehlo.reduce(%v2789 init: %v2773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2791 = stablehlo.broadcast_in_dim %v2790, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2792 = stablehlo.multiply %v2786, %v2789 : tensor<256x128x28x28xf32>
    %v2793 = stablehlo.reduce(%v2792 init: %v2773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2794 = stablehlo.broadcast_in_dim %v2793, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2795 = stablehlo.multiply %v2789, %v2774 : tensor<256x128x28x28xf32>
    %v2796 = stablehlo.subtract %v2795, %v2791 : tensor<256x128x28x28xf32>
    %v2797 = stablehlo.multiply %v2786, %v2794 : tensor<256x128x28x28xf32>
    %v2798 = stablehlo.subtract %v2796, %v2797 : tensor<256x128x28x28xf32>
    %v2799 = stablehlo.divide %v2785, %v2774 : tensor<256x128x28x28xf32>
    %v2800 = stablehlo.multiply %v2799, %v2798 : tensor<256x128x28x28xf32>
    %v2801 = stablehlo.reshape %v2800 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2802 = stablehlo.reshape %v2801 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2803 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2804 = stablehlo.pad %v2802, %v2803, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128x56x56xf32>
    %v2805 = stablehlo.reverse %d2W1, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v2806 = stablehlo.transpose %v2805, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v2807 = stablehlo.convolution(%v2804, %v2806)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v2808 = stablehlo.reshape %v2807 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v2809 = stablehlo.reshape %v252 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2811 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2812 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2813 = stablehlo.reduce(%v2809 init: %v2810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2814 = stablehlo.broadcast_in_dim %v2813, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2815 = stablehlo.divide %v2814, %v2811 : tensor<256x128x28x28xf32>
    %v2816 = stablehlo.subtract %v2809, %v2815 : tensor<256x128x28x28xf32>
    %v2817 = stablehlo.multiply %v2816, %v2816 : tensor<256x128x28x28xf32>
    %v2818 = stablehlo.reduce(%v2817 init: %v2810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2819 = stablehlo.broadcast_in_dim %v2818, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2820 = stablehlo.divide %v2819, %v2811 : tensor<256x128x28x28xf32>
    %v2821 = stablehlo.add %v2820, %v2812 : tensor<256x128x28x28xf32>
    %v2822 = stablehlo.rsqrt %v2821 : tensor<256x128x28x28xf32>
    %v2823 = stablehlo.multiply %v2816, %v2822 : tensor<256x128x28x28xf32>
    %v2824 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2825 = stablehlo.reshape %v2733 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2826 = stablehlo.multiply %v2824, %v2825 : tensor<256x128x28x28xf32>
    %v2827 = stablehlo.reduce(%v2826 init: %v2810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2828 = stablehlo.broadcast_in_dim %v2827, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2829 = stablehlo.multiply %v2823, %v2826 : tensor<256x128x28x28xf32>
    %v2830 = stablehlo.reduce(%v2829 init: %v2810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2831 = stablehlo.broadcast_in_dim %v2830, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2832 = stablehlo.multiply %v2826, %v2811 : tensor<256x128x28x28xf32>
    %v2833 = stablehlo.subtract %v2832, %v2828 : tensor<256x128x28x28xf32>
    %v2834 = stablehlo.multiply %v2823, %v2831 : tensor<256x128x28x28xf32>
    %v2835 = stablehlo.subtract %v2833, %v2834 : tensor<256x128x28x28xf32>
    %v2836 = stablehlo.divide %v2822, %v2811 : tensor<256x128x28x28xf32>
    %v2837 = stablehlo.multiply %v2836, %v2835 : tensor<256x128x28x28xf32>
    %v2838 = stablehlo.reshape %v2837 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2839 = stablehlo.reshape %v2838 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2840 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2841 = stablehlo.pad %v2839, %v2840, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128x56x56xf32>
    %v2842 = stablehlo.reverse %d2Wp, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v2843 = stablehlo.transpose %v2842, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v2844 = stablehlo.convolution(%v2841, %v2843)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x56x56xf32>, tensor<64x128x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v2845 = stablehlo.reshape %v2844 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v2846 = stablehlo.add %v2808, %v2845 : tensor<256x200704xf32>
    %v2847 = stablehlo.reshape %v195 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v2848 = stablehlo.reshape %v2801 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2850 = stablehlo.pad %v2848, %v2849, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128x56x56xf32>
    %v2851 = stablehlo.transpose %v2847, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v2852 = stablehlo.transpose %v2850, dims = [1, 0, 2, 3] : (tensor<256x128x56x56xf32>) -> tensor<128x256x56x56xf32>
    %v2853 = stablehlo.convolution(%v2851, %v2852)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<128x256x56x56xf32>) -> tensor<64x128x3x3xf32>
    %v2854 = stablehlo.transpose %v2853, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v2855 = stablehlo.reshape %v200 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2857 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2858 = stablehlo.reduce(%v2855 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2859 = stablehlo.broadcast_in_dim %v2858, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2860 = stablehlo.divide %v2859, %v2857 : tensor<256x128x28x28xf32>
    %v2861 = stablehlo.subtract %v2855, %v2860 : tensor<256x128x28x28xf32>
    %v2862 = stablehlo.multiply %v2861, %v2861 : tensor<256x128x28x28xf32>
    %v2863 = stablehlo.reduce(%v2862 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2864 = stablehlo.broadcast_in_dim %v2863, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2865 = stablehlo.divide %v2864, %v2857 : tensor<256x128x28x28xf32>
    %v2866 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2867 = stablehlo.add %v2865, %v2866 : tensor<256x128x28x28xf32>
    %v2868 = stablehlo.rsqrt %v2867 : tensor<256x128x28x28xf32>
    %v2869 = stablehlo.multiply %v2861, %v2868 : tensor<256x128x28x28xf32>
    %v2870 = stablehlo.reshape %v2771 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2871 = stablehlo.multiply %v2870, %v2869 : tensor<256x128x28x28xf32>
    %v2872 = stablehlo.reduce(%v2871 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2873 = stablehlo.reshape %v2771 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2875 = stablehlo.reduce(%v2873 init: %v2874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2876 = stablehlo.reshape %v222 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2877 = stablehlo.reshape %v2763 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2878 = stablehlo.transpose %v2876, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2879 = stablehlo.transpose %v2877, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2880 = stablehlo.convolution(%v2878, %v2879)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2881 = stablehlo.transpose %v2880, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2882 = stablehlo.reshape %v227 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2884 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2885 = stablehlo.reduce(%v2882 init: %v2883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2886 = stablehlo.broadcast_in_dim %v2885, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2887 = stablehlo.divide %v2886, %v2884 : tensor<256x128x28x28xf32>
    %v2888 = stablehlo.subtract %v2882, %v2887 : tensor<256x128x28x28xf32>
    %v2889 = stablehlo.multiply %v2888, %v2888 : tensor<256x128x28x28xf32>
    %v2890 = stablehlo.reduce(%v2889 init: %v2883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2891 = stablehlo.broadcast_in_dim %v2890, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2892 = stablehlo.divide %v2891, %v2884 : tensor<256x128x28x28xf32>
    %v2893 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2894 = stablehlo.add %v2892, %v2893 : tensor<256x128x28x28xf32>
    %v2895 = stablehlo.rsqrt %v2894 : tensor<256x128x28x28xf32>
    %v2896 = stablehlo.multiply %v2888, %v2895 : tensor<256x128x28x28xf32>
    %v2897 = stablehlo.reshape %v2733 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2898 = stablehlo.multiply %v2897, %v2896 : tensor<256x128x28x28xf32>
    %v2899 = stablehlo.reduce(%v2898 init: %v2883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2900 = stablehlo.reshape %v2733 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2901 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2902 = stablehlo.reduce(%v2900 init: %v2901) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2903 = stablehlo.reshape %v195 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v2904 = stablehlo.reshape %v2838 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2906 = stablehlo.pad %v2904, %v2905, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128x56x56xf32>
    %v2907 = stablehlo.transpose %v2903, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v2908 = stablehlo.transpose %v2906, dims = [1, 0, 2, 3] : (tensor<256x128x56x56xf32>) -> tensor<128x256x56x56xf32>
    %v2909 = stablehlo.convolution(%v2907, %v2908)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<128x256x56x56xf32>) -> tensor<64x128x1x1xf32>
    %v2910 = stablehlo.transpose %v2909, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v2911 = stablehlo.reshape %v252 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2912 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2913 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2914 = stablehlo.reduce(%v2911 init: %v2912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2915 = stablehlo.broadcast_in_dim %v2914, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2916 = stablehlo.divide %v2915, %v2913 : tensor<256x128x28x28xf32>
    %v2917 = stablehlo.subtract %v2911, %v2916 : tensor<256x128x28x28xf32>
    %v2918 = stablehlo.multiply %v2917, %v2917 : tensor<256x128x28x28xf32>
    %v2919 = stablehlo.reduce(%v2918 init: %v2912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2920 = stablehlo.broadcast_in_dim %v2919, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2921 = stablehlo.divide %v2920, %v2913 : tensor<256x128x28x28xf32>
    %v2922 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2923 = stablehlo.add %v2921, %v2922 : tensor<256x128x28x28xf32>
    %v2924 = stablehlo.rsqrt %v2923 : tensor<256x128x28x28xf32>
    %v2925 = stablehlo.multiply %v2917, %v2924 : tensor<256x128x28x28xf32>
    %v2926 = stablehlo.reshape %v2733 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2927 = stablehlo.multiply %v2926, %v2925 : tensor<256x128x28x28xf32>
    %v2928 = stablehlo.reduce(%v2927 init: %v2912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2929 = stablehlo.reshape %v2733 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2931 = stablehlo.reduce(%v2929 init: %v2930) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2932 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v2933 = stablehlo.compare GT, %v193, %v2932 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v2934 = stablehlo.select %v2933, %v2846, %v2932 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v2935 = stablehlo.reshape %v172 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v2936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2937 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v2938 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v2939 = stablehlo.reduce(%v2935 init: %v2936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2940 = stablehlo.broadcast_in_dim %v2939, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v2941 = stablehlo.divide %v2940, %v2937 : tensor<256x64x56x56xf32>
    %v2942 = stablehlo.subtract %v2935, %v2941 : tensor<256x64x56x56xf32>
    %v2943 = stablehlo.multiply %v2942, %v2942 : tensor<256x64x56x56xf32>
    %v2944 = stablehlo.reduce(%v2943 init: %v2936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2945 = stablehlo.broadcast_in_dim %v2944, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v2946 = stablehlo.divide %v2945, %v2937 : tensor<256x64x56x56xf32>
    %v2947 = stablehlo.add %v2946, %v2938 : tensor<256x64x56x56xf32>
    %v2948 = stablehlo.rsqrt %v2947 : tensor<256x64x56x56xf32>
    %v2949 = stablehlo.multiply %v2942, %v2948 : tensor<256x64x56x56xf32>
    %v2950 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v2951 = stablehlo.reshape %v2934 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v2952 = stablehlo.multiply %v2950, %v2951 : tensor<256x64x56x56xf32>
    %v2953 = stablehlo.reduce(%v2952 init: %v2936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2954 = stablehlo.broadcast_in_dim %v2953, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v2955 = stablehlo.multiply %v2949, %v2952 : tensor<256x64x56x56xf32>
    %v2956 = stablehlo.reduce(%v2955 init: %v2936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2957 = stablehlo.broadcast_in_dim %v2956, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v2958 = stablehlo.multiply %v2952, %v2937 : tensor<256x64x56x56xf32>
    %v2959 = stablehlo.subtract %v2958, %v2954 : tensor<256x64x56x56xf32>
    %v2960 = stablehlo.multiply %v2949, %v2957 : tensor<256x64x56x56xf32>
    %v2961 = stablehlo.subtract %v2959, %v2960 : tensor<256x64x56x56xf32>
    %v2962 = stablehlo.divide %v2948, %v2937 : tensor<256x64x56x56xf32>
    %v2963 = stablehlo.multiply %v2962, %v2961 : tensor<256x64x56x56xf32>
    %v2964 = stablehlo.reshape %v2963 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v2965 = stablehlo.reshape %v2964 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v2966 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v2967 = stablehlo.transpose %v2966, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v2968 = stablehlo.convolution(%v2965, %v2967)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v2969 = stablehlo.reshape %v2968 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v2970 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v2971 = stablehlo.compare GT, %v165, %v2970 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v2972 = stablehlo.select %v2971, %v2969, %v2970 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v2973 = stablehlo.reshape %v145 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v2974 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2975 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v2976 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v2977 = stablehlo.reduce(%v2973 init: %v2974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2978 = stablehlo.broadcast_in_dim %v2977, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v2979 = stablehlo.divide %v2978, %v2975 : tensor<256x64x56x56xf32>
    %v2980 = stablehlo.subtract %v2973, %v2979 : tensor<256x64x56x56xf32>
    %v2981 = stablehlo.multiply %v2980, %v2980 : tensor<256x64x56x56xf32>
    %v2982 = stablehlo.reduce(%v2981 init: %v2974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2983 = stablehlo.broadcast_in_dim %v2982, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v2984 = stablehlo.divide %v2983, %v2975 : tensor<256x64x56x56xf32>
    %v2985 = stablehlo.add %v2984, %v2976 : tensor<256x64x56x56xf32>
    %v2986 = stablehlo.rsqrt %v2985 : tensor<256x64x56x56xf32>
    %v2987 = stablehlo.multiply %v2980, %v2986 : tensor<256x64x56x56xf32>
    %v2988 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v2989 = stablehlo.reshape %v2972 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v2990 = stablehlo.multiply %v2988, %v2989 : tensor<256x64x56x56xf32>
    %v2991 = stablehlo.reduce(%v2990 init: %v2974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2992 = stablehlo.broadcast_in_dim %v2991, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v2993 = stablehlo.multiply %v2987, %v2990 : tensor<256x64x56x56xf32>
    %v2994 = stablehlo.reduce(%v2993 init: %v2974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2995 = stablehlo.broadcast_in_dim %v2994, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v2996 = stablehlo.multiply %v2990, %v2975 : tensor<256x64x56x56xf32>
    %v2997 = stablehlo.subtract %v2996, %v2992 : tensor<256x64x56x56xf32>
    %v2998 = stablehlo.multiply %v2987, %v2995 : tensor<256x64x56x56xf32>
    %v2999 = stablehlo.subtract %v2997, %v2998 : tensor<256x64x56x56xf32>
    %v3000 = stablehlo.divide %v2986, %v2975 : tensor<256x64x56x56xf32>
    %v3001 = stablehlo.multiply %v3000, %v2999 : tensor<256x64x56x56xf32>
    %v3002 = stablehlo.reshape %v3001 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3003 = stablehlo.reshape %v3002 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3004 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3005 = stablehlo.transpose %v3004, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3006 = stablehlo.convolution(%v3003, %v3005)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v3007 = stablehlo.reshape %v3006 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3008 = stablehlo.add %v3007, %v2934 : tensor<256x200704xf32>
    %v3009 = stablehlo.reshape %v140 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3010 = stablehlo.reshape %v3002 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3011 = stablehlo.transpose %v3009, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3012 = stablehlo.transpose %v3010, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3013 = stablehlo.convolution(%v3011, %v3012)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3014 = stablehlo.transpose %v3013, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3015 = stablehlo.reshape %v145 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3016 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3017 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3018 = stablehlo.reduce(%v3015 init: %v3016) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3019 = stablehlo.broadcast_in_dim %v3018, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3020 = stablehlo.divide %v3019, %v3017 : tensor<256x64x56x56xf32>
    %v3021 = stablehlo.subtract %v3015, %v3020 : tensor<256x64x56x56xf32>
    %v3022 = stablehlo.multiply %v3021, %v3021 : tensor<256x64x56x56xf32>
    %v3023 = stablehlo.reduce(%v3022 init: %v3016) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3024 = stablehlo.broadcast_in_dim %v3023, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3025 = stablehlo.divide %v3024, %v3017 : tensor<256x64x56x56xf32>
    %v3026 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3027 = stablehlo.add %v3025, %v3026 : tensor<256x64x56x56xf32>
    %v3028 = stablehlo.rsqrt %v3027 : tensor<256x64x56x56xf32>
    %v3029 = stablehlo.multiply %v3021, %v3028 : tensor<256x64x56x56xf32>
    %v3030 = stablehlo.reshape %v2972 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3031 = stablehlo.multiply %v3030, %v3029 : tensor<256x64x56x56xf32>
    %v3032 = stablehlo.reduce(%v3031 init: %v3016) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3033 = stablehlo.reshape %v2972 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3034 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3035 = stablehlo.reduce(%v3033 init: %v3034) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3036 = stablehlo.reshape %v167 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3037 = stablehlo.reshape %v2964 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3038 = stablehlo.transpose %v3036, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3039 = stablehlo.transpose %v3037, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3040 = stablehlo.convolution(%v3038, %v3039)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3041 = stablehlo.transpose %v3040, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3042 = stablehlo.reshape %v172 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3044 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3045 = stablehlo.reduce(%v3042 init: %v3043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3046 = stablehlo.broadcast_in_dim %v3045, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3047 = stablehlo.divide %v3046, %v3044 : tensor<256x64x56x56xf32>
    %v3048 = stablehlo.subtract %v3042, %v3047 : tensor<256x64x56x56xf32>
    %v3049 = stablehlo.multiply %v3048, %v3048 : tensor<256x64x56x56xf32>
    %v3050 = stablehlo.reduce(%v3049 init: %v3043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3051 = stablehlo.broadcast_in_dim %v3050, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3052 = stablehlo.divide %v3051, %v3044 : tensor<256x64x56x56xf32>
    %v3053 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3054 = stablehlo.add %v3052, %v3053 : tensor<256x64x56x56xf32>
    %v3055 = stablehlo.rsqrt %v3054 : tensor<256x64x56x56xf32>
    %v3056 = stablehlo.multiply %v3048, %v3055 : tensor<256x64x56x56xf32>
    %v3057 = stablehlo.reshape %v2934 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3058 = stablehlo.multiply %v3057, %v3056 : tensor<256x64x56x56xf32>
    %v3059 = stablehlo.reduce(%v3058 init: %v3043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3060 = stablehlo.reshape %v2934 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3062 = stablehlo.reduce(%v3060 init: %v3061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3063 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v3064 = stablehlo.compare GT, %v138, %v3063 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v3065 = stablehlo.select %v3064, %v3008, %v3063 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v3066 = stablehlo.reshape %v117 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3067 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3068 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3069 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3070 = stablehlo.reduce(%v3066 init: %v3067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3071 = stablehlo.broadcast_in_dim %v3070, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3072 = stablehlo.divide %v3071, %v3068 : tensor<256x64x56x56xf32>
    %v3073 = stablehlo.subtract %v3066, %v3072 : tensor<256x64x56x56xf32>
    %v3074 = stablehlo.multiply %v3073, %v3073 : tensor<256x64x56x56xf32>
    %v3075 = stablehlo.reduce(%v3074 init: %v3067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3076 = stablehlo.broadcast_in_dim %v3075, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3077 = stablehlo.divide %v3076, %v3068 : tensor<256x64x56x56xf32>
    %v3078 = stablehlo.add %v3077, %v3069 : tensor<256x64x56x56xf32>
    %v3079 = stablehlo.rsqrt %v3078 : tensor<256x64x56x56xf32>
    %v3080 = stablehlo.multiply %v3073, %v3079 : tensor<256x64x56x56xf32>
    %v3081 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3082 = stablehlo.reshape %v3065 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3083 = stablehlo.multiply %v3081, %v3082 : tensor<256x64x56x56xf32>
    %v3084 = stablehlo.reduce(%v3083 init: %v3067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3085 = stablehlo.broadcast_in_dim %v3084, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3086 = stablehlo.multiply %v3080, %v3083 : tensor<256x64x56x56xf32>
    %v3087 = stablehlo.reduce(%v3086 init: %v3067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3088 = stablehlo.broadcast_in_dim %v3087, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3089 = stablehlo.multiply %v3083, %v3068 : tensor<256x64x56x56xf32>
    %v3090 = stablehlo.subtract %v3089, %v3085 : tensor<256x64x56x56xf32>
    %v3091 = stablehlo.multiply %v3080, %v3088 : tensor<256x64x56x56xf32>
    %v3092 = stablehlo.subtract %v3090, %v3091 : tensor<256x64x56x56xf32>
    %v3093 = stablehlo.divide %v3079, %v3068 : tensor<256x64x56x56xf32>
    %v3094 = stablehlo.multiply %v3093, %v3092 : tensor<256x64x56x56xf32>
    %v3095 = stablehlo.reshape %v3094 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3096 = stablehlo.reshape %v3095 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3097 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3098 = stablehlo.transpose %v3097, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3099 = stablehlo.convolution(%v3096, %v3098)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v3100 = stablehlo.reshape %v3099 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3101 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v3102 = stablehlo.compare GT, %v110, %v3101 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v3103 = stablehlo.select %v3102, %v3100, %v3101 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v3104 = stablehlo.reshape %v90 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3105 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3106 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3107 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3108 = stablehlo.reduce(%v3104 init: %v3105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3109 = stablehlo.broadcast_in_dim %v3108, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3110 = stablehlo.divide %v3109, %v3106 : tensor<256x64x56x56xf32>
    %v3111 = stablehlo.subtract %v3104, %v3110 : tensor<256x64x56x56xf32>
    %v3112 = stablehlo.multiply %v3111, %v3111 : tensor<256x64x56x56xf32>
    %v3113 = stablehlo.reduce(%v3112 init: %v3105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3114 = stablehlo.broadcast_in_dim %v3113, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3115 = stablehlo.divide %v3114, %v3106 : tensor<256x64x56x56xf32>
    %v3116 = stablehlo.add %v3115, %v3107 : tensor<256x64x56x56xf32>
    %v3117 = stablehlo.rsqrt %v3116 : tensor<256x64x56x56xf32>
    %v3118 = stablehlo.multiply %v3111, %v3117 : tensor<256x64x56x56xf32>
    %v3119 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3120 = stablehlo.reshape %v3103 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3121 = stablehlo.multiply %v3119, %v3120 : tensor<256x64x56x56xf32>
    %v3122 = stablehlo.reduce(%v3121 init: %v3105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3123 = stablehlo.broadcast_in_dim %v3122, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3124 = stablehlo.multiply %v3118, %v3121 : tensor<256x64x56x56xf32>
    %v3125 = stablehlo.reduce(%v3124 init: %v3105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3126 = stablehlo.broadcast_in_dim %v3125, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3127 = stablehlo.multiply %v3121, %v3106 : tensor<256x64x56x56xf32>
    %v3128 = stablehlo.subtract %v3127, %v3123 : tensor<256x64x56x56xf32>
    %v3129 = stablehlo.multiply %v3118, %v3126 : tensor<256x64x56x56xf32>
    %v3130 = stablehlo.subtract %v3128, %v3129 : tensor<256x64x56x56xf32>
    %v3131 = stablehlo.divide %v3117, %v3106 : tensor<256x64x56x56xf32>
    %v3132 = stablehlo.multiply %v3131, %v3130 : tensor<256x64x56x56xf32>
    %v3133 = stablehlo.reshape %v3132 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3134 = stablehlo.reshape %v3133 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3135 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3136 = stablehlo.transpose %v3135, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3137 = stablehlo.convolution(%v3134, %v3136)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v3138 = stablehlo.reshape %v3137 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3139 = stablehlo.add %v3138, %v3065 : tensor<256x200704xf32>
    %v3140 = stablehlo.reshape %v85 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3141 = stablehlo.reshape %v3133 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3142 = stablehlo.transpose %v3140, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3143 = stablehlo.transpose %v3141, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3144 = stablehlo.convolution(%v3142, %v3143)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3145 = stablehlo.transpose %v3144, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3146 = stablehlo.reshape %v90 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3148 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3149 = stablehlo.reduce(%v3146 init: %v3147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3150 = stablehlo.broadcast_in_dim %v3149, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3151 = stablehlo.divide %v3150, %v3148 : tensor<256x64x56x56xf32>
    %v3152 = stablehlo.subtract %v3146, %v3151 : tensor<256x64x56x56xf32>
    %v3153 = stablehlo.multiply %v3152, %v3152 : tensor<256x64x56x56xf32>
    %v3154 = stablehlo.reduce(%v3153 init: %v3147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3155 = stablehlo.broadcast_in_dim %v3154, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3156 = stablehlo.divide %v3155, %v3148 : tensor<256x64x56x56xf32>
    %v3157 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3158 = stablehlo.add %v3156, %v3157 : tensor<256x64x56x56xf32>
    %v3159 = stablehlo.rsqrt %v3158 : tensor<256x64x56x56xf32>
    %v3160 = stablehlo.multiply %v3152, %v3159 : tensor<256x64x56x56xf32>
    %v3161 = stablehlo.reshape %v3103 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3162 = stablehlo.multiply %v3161, %v3160 : tensor<256x64x56x56xf32>
    %v3163 = stablehlo.reduce(%v3162 init: %v3147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3164 = stablehlo.reshape %v3103 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3166 = stablehlo.reduce(%v3164 init: %v3165) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3167 = stablehlo.reshape %v112 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3168 = stablehlo.reshape %v3095 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3169 = stablehlo.transpose %v3167, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3170 = stablehlo.transpose %v3168, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3171 = stablehlo.convolution(%v3169, %v3170)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3172 = stablehlo.transpose %v3171, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3173 = stablehlo.reshape %v117 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3174 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3175 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3176 = stablehlo.reduce(%v3173 init: %v3174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3177 = stablehlo.broadcast_in_dim %v3176, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3178 = stablehlo.divide %v3177, %v3175 : tensor<256x64x56x56xf32>
    %v3179 = stablehlo.subtract %v3173, %v3178 : tensor<256x64x56x56xf32>
    %v3180 = stablehlo.multiply %v3179, %v3179 : tensor<256x64x56x56xf32>
    %v3181 = stablehlo.reduce(%v3180 init: %v3174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3182 = stablehlo.broadcast_in_dim %v3181, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3183 = stablehlo.divide %v3182, %v3175 : tensor<256x64x56x56xf32>
    %v3184 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3185 = stablehlo.add %v3183, %v3184 : tensor<256x64x56x56xf32>
    %v3186 = stablehlo.rsqrt %v3185 : tensor<256x64x56x56xf32>
    %v3187 = stablehlo.multiply %v3179, %v3186 : tensor<256x64x56x56xf32>
    %v3188 = stablehlo.reshape %v3065 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3189 = stablehlo.multiply %v3188, %v3187 : tensor<256x64x56x56xf32>
    %v3190 = stablehlo.reduce(%v3189 init: %v3174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3191 = stablehlo.reshape %v3065 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3193 = stablehlo.reduce(%v3191 init: %v3192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3194 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v3195 = stablehlo.compare GT, %v83, %v3194 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v3196 = stablehlo.select %v3195, %v3139, %v3194 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v3197 = stablehlo.reshape %v62 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3199 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3200 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3201 = stablehlo.reduce(%v3197 init: %v3198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3202 = stablehlo.broadcast_in_dim %v3201, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3203 = stablehlo.divide %v3202, %v3199 : tensor<256x64x56x56xf32>
    %v3204 = stablehlo.subtract %v3197, %v3203 : tensor<256x64x56x56xf32>
    %v3205 = stablehlo.multiply %v3204, %v3204 : tensor<256x64x56x56xf32>
    %v3206 = stablehlo.reduce(%v3205 init: %v3198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3207 = stablehlo.broadcast_in_dim %v3206, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3208 = stablehlo.divide %v3207, %v3199 : tensor<256x64x56x56xf32>
    %v3209 = stablehlo.add %v3208, %v3200 : tensor<256x64x56x56xf32>
    %v3210 = stablehlo.rsqrt %v3209 : tensor<256x64x56x56xf32>
    %v3211 = stablehlo.multiply %v3204, %v3210 : tensor<256x64x56x56xf32>
    %v3212 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3213 = stablehlo.reshape %v3196 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3214 = stablehlo.multiply %v3212, %v3213 : tensor<256x64x56x56xf32>
    %v3215 = stablehlo.reduce(%v3214 init: %v3198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3216 = stablehlo.broadcast_in_dim %v3215, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3217 = stablehlo.multiply %v3211, %v3214 : tensor<256x64x56x56xf32>
    %v3218 = stablehlo.reduce(%v3217 init: %v3198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3219 = stablehlo.broadcast_in_dim %v3218, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3220 = stablehlo.multiply %v3214, %v3199 : tensor<256x64x56x56xf32>
    %v3221 = stablehlo.subtract %v3220, %v3216 : tensor<256x64x56x56xf32>
    %v3222 = stablehlo.multiply %v3211, %v3219 : tensor<256x64x56x56xf32>
    %v3223 = stablehlo.subtract %v3221, %v3222 : tensor<256x64x56x56xf32>
    %v3224 = stablehlo.divide %v3210, %v3199 : tensor<256x64x56x56xf32>
    %v3225 = stablehlo.multiply %v3224, %v3223 : tensor<256x64x56x56xf32>
    %v3226 = stablehlo.reshape %v3225 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3227 = stablehlo.reshape %v3226 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3228 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3229 = stablehlo.transpose %v3228, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3230 = stablehlo.convolution(%v3227, %v3229)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v3231 = stablehlo.reshape %v3230 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3232 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v3233 = stablehlo.compare GT, %v55, %v3232 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v3234 = stablehlo.select %v3233, %v3231, %v3232 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v3235 = stablehlo.reshape %v35 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3236 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3237 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3238 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3239 = stablehlo.reduce(%v3235 init: %v3236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3240 = stablehlo.broadcast_in_dim %v3239, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3241 = stablehlo.divide %v3240, %v3237 : tensor<256x64x56x56xf32>
    %v3242 = stablehlo.subtract %v3235, %v3241 : tensor<256x64x56x56xf32>
    %v3243 = stablehlo.multiply %v3242, %v3242 : tensor<256x64x56x56xf32>
    %v3244 = stablehlo.reduce(%v3243 init: %v3236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3245 = stablehlo.broadcast_in_dim %v3244, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3246 = stablehlo.divide %v3245, %v3237 : tensor<256x64x56x56xf32>
    %v3247 = stablehlo.add %v3246, %v3238 : tensor<256x64x56x56xf32>
    %v3248 = stablehlo.rsqrt %v3247 : tensor<256x64x56x56xf32>
    %v3249 = stablehlo.multiply %v3242, %v3248 : tensor<256x64x56x56xf32>
    %v3250 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3251 = stablehlo.reshape %v3234 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3252 = stablehlo.multiply %v3250, %v3251 : tensor<256x64x56x56xf32>
    %v3253 = stablehlo.reduce(%v3252 init: %v3236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3254 = stablehlo.broadcast_in_dim %v3253, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3255 = stablehlo.multiply %v3249, %v3252 : tensor<256x64x56x56xf32>
    %v3256 = stablehlo.reduce(%v3255 init: %v3236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3257 = stablehlo.broadcast_in_dim %v3256, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3258 = stablehlo.multiply %v3252, %v3237 : tensor<256x64x56x56xf32>
    %v3259 = stablehlo.subtract %v3258, %v3254 : tensor<256x64x56x56xf32>
    %v3260 = stablehlo.multiply %v3249, %v3257 : tensor<256x64x56x56xf32>
    %v3261 = stablehlo.subtract %v3259, %v3260 : tensor<256x64x56x56xf32>
    %v3262 = stablehlo.divide %v3248, %v3237 : tensor<256x64x56x56xf32>
    %v3263 = stablehlo.multiply %v3262, %v3261 : tensor<256x64x56x56xf32>
    %v3264 = stablehlo.reshape %v3263 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3265 = stablehlo.reshape %v3264 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3266 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3267 = stablehlo.transpose %v3266, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3268 = stablehlo.convolution(%v3265, %v3267)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v3269 = stablehlo.reshape %v3268 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3270 = stablehlo.add %v3269, %v3196 : tensor<256x200704xf32>
    %v3271 = stablehlo.reshape %v30 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3272 = stablehlo.reshape %v3264 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3273 = stablehlo.transpose %v3271, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3274 = stablehlo.transpose %v3272, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3275 = stablehlo.convolution(%v3273, %v3274)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3276 = stablehlo.transpose %v3275, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3277 = stablehlo.reshape %v35 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3279 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3280 = stablehlo.reduce(%v3277 init: %v3278) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3281 = stablehlo.broadcast_in_dim %v3280, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3282 = stablehlo.divide %v3281, %v3279 : tensor<256x64x56x56xf32>
    %v3283 = stablehlo.subtract %v3277, %v3282 : tensor<256x64x56x56xf32>
    %v3284 = stablehlo.multiply %v3283, %v3283 : tensor<256x64x56x56xf32>
    %v3285 = stablehlo.reduce(%v3284 init: %v3278) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3286 = stablehlo.broadcast_in_dim %v3285, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3287 = stablehlo.divide %v3286, %v3279 : tensor<256x64x56x56xf32>
    %v3288 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3289 = stablehlo.add %v3287, %v3288 : tensor<256x64x56x56xf32>
    %v3290 = stablehlo.rsqrt %v3289 : tensor<256x64x56x56xf32>
    %v3291 = stablehlo.multiply %v3283, %v3290 : tensor<256x64x56x56xf32>
    %v3292 = stablehlo.reshape %v3234 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3293 = stablehlo.multiply %v3292, %v3291 : tensor<256x64x56x56xf32>
    %v3294 = stablehlo.reduce(%v3293 init: %v3278) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3295 = stablehlo.reshape %v3234 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3296 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3297 = stablehlo.reduce(%v3295 init: %v3296) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3298 = stablehlo.reshape %v57 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3299 = stablehlo.reshape %v3226 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3300 = stablehlo.transpose %v3298, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3301 = stablehlo.transpose %v3299, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3302 = stablehlo.convolution(%v3300, %v3301)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3303 = stablehlo.transpose %v3302, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3304 = stablehlo.reshape %v62 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3305 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3306 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3307 = stablehlo.reduce(%v3304 init: %v3305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3308 = stablehlo.broadcast_in_dim %v3307, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3309 = stablehlo.divide %v3308, %v3306 : tensor<256x64x56x56xf32>
    %v3310 = stablehlo.subtract %v3304, %v3309 : tensor<256x64x56x56xf32>
    %v3311 = stablehlo.multiply %v3310, %v3310 : tensor<256x64x56x56xf32>
    %v3312 = stablehlo.reduce(%v3311 init: %v3305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3313 = stablehlo.broadcast_in_dim %v3312, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3314 = stablehlo.divide %v3313, %v3306 : tensor<256x64x56x56xf32>
    %v3315 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3316 = stablehlo.add %v3314, %v3315 : tensor<256x64x56x56xf32>
    %v3317 = stablehlo.rsqrt %v3316 : tensor<256x64x56x56xf32>
    %v3318 = stablehlo.multiply %v3310, %v3317 : tensor<256x64x56x56xf32>
    %v3319 = stablehlo.reshape %v3196 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3320 = stablehlo.multiply %v3319, %v3318 : tensor<256x64x56x56xf32>
    %v3321 = stablehlo.reduce(%v3320 init: %v3305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3322 = stablehlo.reshape %v3196 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3324 = stablehlo.reduce(%v3322 init: %v3323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3325 = stablehlo.reshape %v26 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3326 = stablehlo.reshape %v3270 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3327 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3328 = "stablehlo.select_and_scatter"(%v3325, %v3326, %v3327) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<256x64x112x112xf32>, tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64x112x112xf32>
    %v3329 = stablehlo.reshape %v3328 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v3330 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v3331 = stablehlo.compare GT, %v24, %v3330 : (tensor<256x802816xf32>, tensor<256x802816xf32>) -> tensor<256x802816xi1>
    %v3332 = stablehlo.select %v3331, %v3329, %v3330 : tensor<256x802816xi1>, tensor<256x802816xf32>
    %v3333 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3335 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v3336 = stablehlo.constant dense<1.0e-05> : tensor<256x64x112x112xf32>
    %v3337 = stablehlo.reduce(%v3333 init: %v3334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3338 = stablehlo.broadcast_in_dim %v3337, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3339 = stablehlo.divide %v3338, %v3335 : tensor<256x64x112x112xf32>
    %v3340 = stablehlo.subtract %v3333, %v3339 : tensor<256x64x112x112xf32>
    %v3341 = stablehlo.multiply %v3340, %v3340 : tensor<256x64x112x112xf32>
    %v3342 = stablehlo.reduce(%v3341 init: %v3334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3343 = stablehlo.broadcast_in_dim %v3342, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3344 = stablehlo.divide %v3343, %v3335 : tensor<256x64x112x112xf32>
    %v3345 = stablehlo.add %v3344, %v3336 : tensor<256x64x112x112xf32>
    %v3346 = stablehlo.rsqrt %v3345 : tensor<256x64x112x112xf32>
    %v3347 = stablehlo.multiply %v3340, %v3346 : tensor<256x64x112x112xf32>
    %v3348 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3349 = stablehlo.reshape %v3332 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3350 = stablehlo.multiply %v3348, %v3349 : tensor<256x64x112x112xf32>
    %v3351 = stablehlo.reduce(%v3350 init: %v3334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3352 = stablehlo.broadcast_in_dim %v3351, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3353 = stablehlo.multiply %v3347, %v3350 : tensor<256x64x112x112xf32>
    %v3354 = stablehlo.reduce(%v3353 init: %v3334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3355 = stablehlo.broadcast_in_dim %v3354, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3356 = stablehlo.multiply %v3350, %v3335 : tensor<256x64x112x112xf32>
    %v3357 = stablehlo.subtract %v3356, %v3352 : tensor<256x64x112x112xf32>
    %v3358 = stablehlo.multiply %v3347, %v3355 : tensor<256x64x112x112xf32>
    %v3359 = stablehlo.subtract %v3357, %v3358 : tensor<256x64x112x112xf32>
    %v3360 = stablehlo.divide %v3346, %v3335 : tensor<256x64x112x112xf32>
    %v3361 = stablehlo.multiply %v3360, %v3359 : tensor<256x64x112x112xf32>
    %v3362 = stablehlo.reshape %v3361 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v3363 = stablehlo.reshape %x : (tensor<256x150528xf32>) -> tensor<256x3x224x224xf32>
    %v3364 = stablehlo.reshape %v3362 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3366 = stablehlo.pad %v3364, %v3365, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<256x64x224x224xf32>
    %v3367 = stablehlo.transpose %v3363, dims = [1, 0, 2, 3] : (tensor<256x3x224x224xf32>) -> tensor<3x256x224x224xf32>
    %v3368 = stablehlo.transpose %v3366, dims = [1, 0, 2, 3] : (tensor<256x64x224x224xf32>) -> tensor<64x256x224x224xf32>
    %v3369 = stablehlo.convolution(%v3367, %v3368)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x256x224x224xf32>, tensor<64x256x224x224xf32>) -> tensor<3x64x7x7xf32>
    %v3370 = stablehlo.transpose %v3369, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v3371 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3372 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3373 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v3374 = stablehlo.reduce(%v3371 init: %v3372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3375 = stablehlo.broadcast_in_dim %v3374, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3376 = stablehlo.divide %v3375, %v3373 : tensor<256x64x112x112xf32>
    %v3377 = stablehlo.subtract %v3371, %v3376 : tensor<256x64x112x112xf32>
    %v3378 = stablehlo.multiply %v3377, %v3377 : tensor<256x64x112x112xf32>
    %v3379 = stablehlo.reduce(%v3378 init: %v3372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3380 = stablehlo.broadcast_in_dim %v3379, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3381 = stablehlo.divide %v3380, %v3373 : tensor<256x64x112x112xf32>
    %v3382 = stablehlo.constant dense<1.0e-05> : tensor<256x64x112x112xf32>
    %v3383 = stablehlo.add %v3381, %v3382 : tensor<256x64x112x112xf32>
    %v3384 = stablehlo.rsqrt %v3383 : tensor<256x64x112x112xf32>
    %v3385 = stablehlo.multiply %v3377, %v3384 : tensor<256x64x112x112xf32>
    %v3386 = stablehlo.reshape %v3332 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3387 = stablehlo.multiply %v3386, %v3385 : tensor<256x64x112x112xf32>
    %v3388 = stablehlo.reduce(%v3387 init: %v3372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3389 = stablehlo.reshape %v3332 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3391 = stablehlo.reduce(%v3389 init: %v3390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3392 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3393 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3394 = stablehlo.constant dense<3211264.0> : tensor<64xf32>
    %v3395 = stablehlo.reduce(%v3392 init: %v3393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3396 = stablehlo.divide %v3395, %v3394 : tensor<64xf32>
    %v3397 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3399 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v3400 = stablehlo.reduce(%v3397 init: %v3398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3401 = stablehlo.broadcast_in_dim %v3400, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3402 = stablehlo.divide %v3401, %v3399 : tensor<256x64x112x112xf32>
    %v3403 = stablehlo.subtract %v3397, %v3402 : tensor<256x64x112x112xf32>
    %v3404 = stablehlo.multiply %v3403, %v3403 : tensor<256x64x112x112xf32>
    %v3405 = stablehlo.reduce(%v3404 init: %v3398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3406 = stablehlo.constant dense<3211264.0> : tensor<64xf32>
    %v3407 = stablehlo.divide %v3405, %v3406 : tensor<64xf32>
    %v3408 = stablehlo.reshape %v35 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3409 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3410 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3411 = stablehlo.reduce(%v3408 init: %v3409) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3412 = stablehlo.divide %v3411, %v3410 : tensor<64xf32>
    %v3413 = stablehlo.reshape %v35 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3414 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3415 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3416 = stablehlo.reduce(%v3413 init: %v3414) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3417 = stablehlo.broadcast_in_dim %v3416, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3418 = stablehlo.divide %v3417, %v3415 : tensor<256x64x56x56xf32>
    %v3419 = stablehlo.subtract %v3413, %v3418 : tensor<256x64x56x56xf32>
    %v3420 = stablehlo.multiply %v3419, %v3419 : tensor<256x64x56x56xf32>
    %v3421 = stablehlo.reduce(%v3420 init: %v3414) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3422 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3423 = stablehlo.divide %v3421, %v3422 : tensor<64xf32>
    %v3424 = stablehlo.reshape %v62 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3425 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3426 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3427 = stablehlo.reduce(%v3424 init: %v3425) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3428 = stablehlo.divide %v3427, %v3426 : tensor<64xf32>
    %v3429 = stablehlo.reshape %v62 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3431 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3432 = stablehlo.reduce(%v3429 init: %v3430) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3433 = stablehlo.broadcast_in_dim %v3432, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3434 = stablehlo.divide %v3433, %v3431 : tensor<256x64x56x56xf32>
    %v3435 = stablehlo.subtract %v3429, %v3434 : tensor<256x64x56x56xf32>
    %v3436 = stablehlo.multiply %v3435, %v3435 : tensor<256x64x56x56xf32>
    %v3437 = stablehlo.reduce(%v3436 init: %v3430) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3438 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3439 = stablehlo.divide %v3437, %v3438 : tensor<64xf32>
    %v3440 = stablehlo.reshape %v90 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3441 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3442 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3443 = stablehlo.reduce(%v3440 init: %v3441) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3444 = stablehlo.divide %v3443, %v3442 : tensor<64xf32>
    %v3445 = stablehlo.reshape %v90 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3446 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3447 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3448 = stablehlo.reduce(%v3445 init: %v3446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3449 = stablehlo.broadcast_in_dim %v3448, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3450 = stablehlo.divide %v3449, %v3447 : tensor<256x64x56x56xf32>
    %v3451 = stablehlo.subtract %v3445, %v3450 : tensor<256x64x56x56xf32>
    %v3452 = stablehlo.multiply %v3451, %v3451 : tensor<256x64x56x56xf32>
    %v3453 = stablehlo.reduce(%v3452 init: %v3446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3454 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3455 = stablehlo.divide %v3453, %v3454 : tensor<64xf32>
    %v3456 = stablehlo.reshape %v117 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3457 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3458 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3459 = stablehlo.reduce(%v3456 init: %v3457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3460 = stablehlo.divide %v3459, %v3458 : tensor<64xf32>
    %v3461 = stablehlo.reshape %v117 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3463 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3464 = stablehlo.reduce(%v3461 init: %v3462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3465 = stablehlo.broadcast_in_dim %v3464, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3466 = stablehlo.divide %v3465, %v3463 : tensor<256x64x56x56xf32>
    %v3467 = stablehlo.subtract %v3461, %v3466 : tensor<256x64x56x56xf32>
    %v3468 = stablehlo.multiply %v3467, %v3467 : tensor<256x64x56x56xf32>
    %v3469 = stablehlo.reduce(%v3468 init: %v3462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3470 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3471 = stablehlo.divide %v3469, %v3470 : tensor<64xf32>
    %v3472 = stablehlo.reshape %v145 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3474 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3475 = stablehlo.reduce(%v3472 init: %v3473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3476 = stablehlo.divide %v3475, %v3474 : tensor<64xf32>
    %v3477 = stablehlo.reshape %v145 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3479 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3480 = stablehlo.reduce(%v3477 init: %v3478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3481 = stablehlo.broadcast_in_dim %v3480, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3482 = stablehlo.divide %v3481, %v3479 : tensor<256x64x56x56xf32>
    %v3483 = stablehlo.subtract %v3477, %v3482 : tensor<256x64x56x56xf32>
    %v3484 = stablehlo.multiply %v3483, %v3483 : tensor<256x64x56x56xf32>
    %v3485 = stablehlo.reduce(%v3484 init: %v3478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3486 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3487 = stablehlo.divide %v3485, %v3486 : tensor<64xf32>
    %v3488 = stablehlo.reshape %v172 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3490 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3491 = stablehlo.reduce(%v3488 init: %v3489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3492 = stablehlo.divide %v3491, %v3490 : tensor<64xf32>
    %v3493 = stablehlo.reshape %v172 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3494 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3495 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3496 = stablehlo.reduce(%v3493 init: %v3494) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3497 = stablehlo.broadcast_in_dim %v3496, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3498 = stablehlo.divide %v3497, %v3495 : tensor<256x64x56x56xf32>
    %v3499 = stablehlo.subtract %v3493, %v3498 : tensor<256x64x56x56xf32>
    %v3500 = stablehlo.multiply %v3499, %v3499 : tensor<256x64x56x56xf32>
    %v3501 = stablehlo.reduce(%v3500 init: %v3494) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3502 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3503 = stablehlo.divide %v3501, %v3502 : tensor<64xf32>
    %v3504 = stablehlo.reshape %v200 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3506 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3507 = stablehlo.reduce(%v3504 init: %v3505) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3508 = stablehlo.divide %v3507, %v3506 : tensor<128xf32>
    %v3509 = stablehlo.reshape %v200 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3510 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3511 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3512 = stablehlo.reduce(%v3509 init: %v3510) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3513 = stablehlo.broadcast_in_dim %v3512, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3514 = stablehlo.divide %v3513, %v3511 : tensor<256x128x28x28xf32>
    %v3515 = stablehlo.subtract %v3509, %v3514 : tensor<256x128x28x28xf32>
    %v3516 = stablehlo.multiply %v3515, %v3515 : tensor<256x128x28x28xf32>
    %v3517 = stablehlo.reduce(%v3516 init: %v3510) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3518 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3519 = stablehlo.divide %v3517, %v3518 : tensor<128xf32>
    %v3520 = stablehlo.reshape %v227 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3521 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3522 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3523 = stablehlo.reduce(%v3520 init: %v3521) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3524 = stablehlo.divide %v3523, %v3522 : tensor<128xf32>
    %v3525 = stablehlo.reshape %v227 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3526 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3527 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3528 = stablehlo.reduce(%v3525 init: %v3526) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3529 = stablehlo.broadcast_in_dim %v3528, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3530 = stablehlo.divide %v3529, %v3527 : tensor<256x128x28x28xf32>
    %v3531 = stablehlo.subtract %v3525, %v3530 : tensor<256x128x28x28xf32>
    %v3532 = stablehlo.multiply %v3531, %v3531 : tensor<256x128x28x28xf32>
    %v3533 = stablehlo.reduce(%v3532 init: %v3526) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3534 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3535 = stablehlo.divide %v3533, %v3534 : tensor<128xf32>
    %v3536 = stablehlo.reshape %v252 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3537 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3538 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3539 = stablehlo.reduce(%v3536 init: %v3537) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3540 = stablehlo.divide %v3539, %v3538 : tensor<128xf32>
    %v3541 = stablehlo.reshape %v252 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3543 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3544 = stablehlo.reduce(%v3541 init: %v3542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3545 = stablehlo.broadcast_in_dim %v3544, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3546 = stablehlo.divide %v3545, %v3543 : tensor<256x128x28x28xf32>
    %v3547 = stablehlo.subtract %v3541, %v3546 : tensor<256x128x28x28xf32>
    %v3548 = stablehlo.multiply %v3547, %v3547 : tensor<256x128x28x28xf32>
    %v3549 = stablehlo.reduce(%v3548 init: %v3542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3550 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3551 = stablehlo.divide %v3549, %v3550 : tensor<128xf32>
    %v3552 = stablehlo.reshape %v280 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3553 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3554 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3555 = stablehlo.reduce(%v3552 init: %v3553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3556 = stablehlo.divide %v3555, %v3554 : tensor<128xf32>
    %v3557 = stablehlo.reshape %v280 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3558 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3559 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3560 = stablehlo.reduce(%v3557 init: %v3558) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3561 = stablehlo.broadcast_in_dim %v3560, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3562 = stablehlo.divide %v3561, %v3559 : tensor<256x128x28x28xf32>
    %v3563 = stablehlo.subtract %v3557, %v3562 : tensor<256x128x28x28xf32>
    %v3564 = stablehlo.multiply %v3563, %v3563 : tensor<256x128x28x28xf32>
    %v3565 = stablehlo.reduce(%v3564 init: %v3558) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3566 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3567 = stablehlo.divide %v3565, %v3566 : tensor<128xf32>
    %v3568 = stablehlo.reshape %v307 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3570 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3571 = stablehlo.reduce(%v3568 init: %v3569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3572 = stablehlo.divide %v3571, %v3570 : tensor<128xf32>
    %v3573 = stablehlo.reshape %v307 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3574 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3575 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3576 = stablehlo.reduce(%v3573 init: %v3574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3577 = stablehlo.broadcast_in_dim %v3576, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3578 = stablehlo.divide %v3577, %v3575 : tensor<256x128x28x28xf32>
    %v3579 = stablehlo.subtract %v3573, %v3578 : tensor<256x128x28x28xf32>
    %v3580 = stablehlo.multiply %v3579, %v3579 : tensor<256x128x28x28xf32>
    %v3581 = stablehlo.reduce(%v3580 init: %v3574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3582 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3583 = stablehlo.divide %v3581, %v3582 : tensor<128xf32>
    %v3584 = stablehlo.reshape %v335 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3586 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3587 = stablehlo.reduce(%v3584 init: %v3585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3588 = stablehlo.divide %v3587, %v3586 : tensor<128xf32>
    %v3589 = stablehlo.reshape %v335 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3591 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3592 = stablehlo.reduce(%v3589 init: %v3590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3593 = stablehlo.broadcast_in_dim %v3592, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3594 = stablehlo.divide %v3593, %v3591 : tensor<256x128x28x28xf32>
    %v3595 = stablehlo.subtract %v3589, %v3594 : tensor<256x128x28x28xf32>
    %v3596 = stablehlo.multiply %v3595, %v3595 : tensor<256x128x28x28xf32>
    %v3597 = stablehlo.reduce(%v3596 init: %v3590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3598 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3599 = stablehlo.divide %v3597, %v3598 : tensor<128xf32>
    %v3600 = stablehlo.reshape %v362 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3601 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3602 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3603 = stablehlo.reduce(%v3600 init: %v3601) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3604 = stablehlo.divide %v3603, %v3602 : tensor<128xf32>
    %v3605 = stablehlo.reshape %v362 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3606 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3607 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3608 = stablehlo.reduce(%v3605 init: %v3606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3609 = stablehlo.broadcast_in_dim %v3608, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3610 = stablehlo.divide %v3609, %v3607 : tensor<256x128x28x28xf32>
    %v3611 = stablehlo.subtract %v3605, %v3610 : tensor<256x128x28x28xf32>
    %v3612 = stablehlo.multiply %v3611, %v3611 : tensor<256x128x28x28xf32>
    %v3613 = stablehlo.reduce(%v3612 init: %v3606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3614 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3615 = stablehlo.divide %v3613, %v3614 : tensor<128xf32>
    %v3616 = stablehlo.reshape %v390 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3618 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3619 = stablehlo.reduce(%v3616 init: %v3617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3620 = stablehlo.divide %v3619, %v3618 : tensor<128xf32>
    %v3621 = stablehlo.reshape %v390 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3622 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3623 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3624 = stablehlo.reduce(%v3621 init: %v3622) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3625 = stablehlo.broadcast_in_dim %v3624, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3626 = stablehlo.divide %v3625, %v3623 : tensor<256x128x28x28xf32>
    %v3627 = stablehlo.subtract %v3621, %v3626 : tensor<256x128x28x28xf32>
    %v3628 = stablehlo.multiply %v3627, %v3627 : tensor<256x128x28x28xf32>
    %v3629 = stablehlo.reduce(%v3628 init: %v3622) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3630 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3631 = stablehlo.divide %v3629, %v3630 : tensor<128xf32>
    %v3632 = stablehlo.reshape %v417 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3633 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3634 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3635 = stablehlo.reduce(%v3632 init: %v3633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3636 = stablehlo.divide %v3635, %v3634 : tensor<128xf32>
    %v3637 = stablehlo.reshape %v417 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3638 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3639 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3640 = stablehlo.reduce(%v3637 init: %v3638) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3641 = stablehlo.broadcast_in_dim %v3640, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3642 = stablehlo.divide %v3641, %v3639 : tensor<256x128x28x28xf32>
    %v3643 = stablehlo.subtract %v3637, %v3642 : tensor<256x128x28x28xf32>
    %v3644 = stablehlo.multiply %v3643, %v3643 : tensor<256x128x28x28xf32>
    %v3645 = stablehlo.reduce(%v3644 init: %v3638) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3646 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3647 = stablehlo.divide %v3645, %v3646 : tensor<128xf32>
    %v3648 = stablehlo.reshape %v445 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3649 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3650 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3651 = stablehlo.reduce(%v3648 init: %v3649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3652 = stablehlo.divide %v3651, %v3650 : tensor<256xf32>
    %v3653 = stablehlo.reshape %v445 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3655 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3656 = stablehlo.reduce(%v3653 init: %v3654) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3657 = stablehlo.broadcast_in_dim %v3656, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3658 = stablehlo.divide %v3657, %v3655 : tensor<256x256x14x14xf32>
    %v3659 = stablehlo.subtract %v3653, %v3658 : tensor<256x256x14x14xf32>
    %v3660 = stablehlo.multiply %v3659, %v3659 : tensor<256x256x14x14xf32>
    %v3661 = stablehlo.reduce(%v3660 init: %v3654) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3662 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3663 = stablehlo.divide %v3661, %v3662 : tensor<256xf32>
    %v3664 = stablehlo.reshape %v472 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3666 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3667 = stablehlo.reduce(%v3664 init: %v3665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3668 = stablehlo.divide %v3667, %v3666 : tensor<256xf32>
    %v3669 = stablehlo.reshape %v472 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3670 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3671 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3672 = stablehlo.reduce(%v3669 init: %v3670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3673 = stablehlo.broadcast_in_dim %v3672, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3674 = stablehlo.divide %v3673, %v3671 : tensor<256x256x14x14xf32>
    %v3675 = stablehlo.subtract %v3669, %v3674 : tensor<256x256x14x14xf32>
    %v3676 = stablehlo.multiply %v3675, %v3675 : tensor<256x256x14x14xf32>
    %v3677 = stablehlo.reduce(%v3676 init: %v3670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3678 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3679 = stablehlo.divide %v3677, %v3678 : tensor<256xf32>
    %v3680 = stablehlo.reshape %v497 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3681 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3682 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3683 = stablehlo.reduce(%v3680 init: %v3681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3684 = stablehlo.divide %v3683, %v3682 : tensor<256xf32>
    %v3685 = stablehlo.reshape %v497 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3687 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3688 = stablehlo.reduce(%v3685 init: %v3686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3689 = stablehlo.broadcast_in_dim %v3688, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3690 = stablehlo.divide %v3689, %v3687 : tensor<256x256x14x14xf32>
    %v3691 = stablehlo.subtract %v3685, %v3690 : tensor<256x256x14x14xf32>
    %v3692 = stablehlo.multiply %v3691, %v3691 : tensor<256x256x14x14xf32>
    %v3693 = stablehlo.reduce(%v3692 init: %v3686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3694 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3695 = stablehlo.divide %v3693, %v3694 : tensor<256xf32>
    %v3696 = stablehlo.reshape %v525 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3697 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3698 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3699 = stablehlo.reduce(%v3696 init: %v3697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3700 = stablehlo.divide %v3699, %v3698 : tensor<256xf32>
    %v3701 = stablehlo.reshape %v525 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3703 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3704 = stablehlo.reduce(%v3701 init: %v3702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3705 = stablehlo.broadcast_in_dim %v3704, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3706 = stablehlo.divide %v3705, %v3703 : tensor<256x256x14x14xf32>
    %v3707 = stablehlo.subtract %v3701, %v3706 : tensor<256x256x14x14xf32>
    %v3708 = stablehlo.multiply %v3707, %v3707 : tensor<256x256x14x14xf32>
    %v3709 = stablehlo.reduce(%v3708 init: %v3702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3710 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3711 = stablehlo.divide %v3709, %v3710 : tensor<256xf32>
    %v3712 = stablehlo.reshape %v552 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3714 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3715 = stablehlo.reduce(%v3712 init: %v3713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3716 = stablehlo.divide %v3715, %v3714 : tensor<256xf32>
    %v3717 = stablehlo.reshape %v552 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3718 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3719 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3720 = stablehlo.reduce(%v3717 init: %v3718) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3721 = stablehlo.broadcast_in_dim %v3720, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3722 = stablehlo.divide %v3721, %v3719 : tensor<256x256x14x14xf32>
    %v3723 = stablehlo.subtract %v3717, %v3722 : tensor<256x256x14x14xf32>
    %v3724 = stablehlo.multiply %v3723, %v3723 : tensor<256x256x14x14xf32>
    %v3725 = stablehlo.reduce(%v3724 init: %v3718) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3726 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3727 = stablehlo.divide %v3725, %v3726 : tensor<256xf32>
    %v3728 = stablehlo.reshape %v580 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3730 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3731 = stablehlo.reduce(%v3728 init: %v3729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3732 = stablehlo.divide %v3731, %v3730 : tensor<256xf32>
    %v3733 = stablehlo.reshape %v580 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3735 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3736 = stablehlo.reduce(%v3733 init: %v3734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3737 = stablehlo.broadcast_in_dim %v3736, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3738 = stablehlo.divide %v3737, %v3735 : tensor<256x256x14x14xf32>
    %v3739 = stablehlo.subtract %v3733, %v3738 : tensor<256x256x14x14xf32>
    %v3740 = stablehlo.multiply %v3739, %v3739 : tensor<256x256x14x14xf32>
    %v3741 = stablehlo.reduce(%v3740 init: %v3734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3742 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3743 = stablehlo.divide %v3741, %v3742 : tensor<256xf32>
    %v3744 = stablehlo.reshape %v607 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3745 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3746 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3747 = stablehlo.reduce(%v3744 init: %v3745) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3748 = stablehlo.divide %v3747, %v3746 : tensor<256xf32>
    %v3749 = stablehlo.reshape %v607 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3750 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3751 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3752 = stablehlo.reduce(%v3749 init: %v3750) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3753 = stablehlo.broadcast_in_dim %v3752, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3754 = stablehlo.divide %v3753, %v3751 : tensor<256x256x14x14xf32>
    %v3755 = stablehlo.subtract %v3749, %v3754 : tensor<256x256x14x14xf32>
    %v3756 = stablehlo.multiply %v3755, %v3755 : tensor<256x256x14x14xf32>
    %v3757 = stablehlo.reduce(%v3756 init: %v3750) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3758 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3759 = stablehlo.divide %v3757, %v3758 : tensor<256xf32>
    %v3760 = stablehlo.reshape %v635 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3761 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3762 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3763 = stablehlo.reduce(%v3760 init: %v3761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3764 = stablehlo.divide %v3763, %v3762 : tensor<256xf32>
    %v3765 = stablehlo.reshape %v635 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3766 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3767 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3768 = stablehlo.reduce(%v3765 init: %v3766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3769 = stablehlo.broadcast_in_dim %v3768, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3770 = stablehlo.divide %v3769, %v3767 : tensor<256x256x14x14xf32>
    %v3771 = stablehlo.subtract %v3765, %v3770 : tensor<256x256x14x14xf32>
    %v3772 = stablehlo.multiply %v3771, %v3771 : tensor<256x256x14x14xf32>
    %v3773 = stablehlo.reduce(%v3772 init: %v3766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3774 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3775 = stablehlo.divide %v3773, %v3774 : tensor<256xf32>
    %v3776 = stablehlo.reshape %v662 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3777 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3778 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3779 = stablehlo.reduce(%v3776 init: %v3777) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3780 = stablehlo.divide %v3779, %v3778 : tensor<256xf32>
    %v3781 = stablehlo.reshape %v662 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3782 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3783 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3784 = stablehlo.reduce(%v3781 init: %v3782) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3785 = stablehlo.broadcast_in_dim %v3784, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3786 = stablehlo.divide %v3785, %v3783 : tensor<256x256x14x14xf32>
    %v3787 = stablehlo.subtract %v3781, %v3786 : tensor<256x256x14x14xf32>
    %v3788 = stablehlo.multiply %v3787, %v3787 : tensor<256x256x14x14xf32>
    %v3789 = stablehlo.reduce(%v3788 init: %v3782) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3790 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3791 = stablehlo.divide %v3789, %v3790 : tensor<256xf32>
    %v3792 = stablehlo.reshape %v690 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3793 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3794 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3795 = stablehlo.reduce(%v3792 init: %v3793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3796 = stablehlo.divide %v3795, %v3794 : tensor<256xf32>
    %v3797 = stablehlo.reshape %v690 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3798 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3799 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3800 = stablehlo.reduce(%v3797 init: %v3798) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3801 = stablehlo.broadcast_in_dim %v3800, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3802 = stablehlo.divide %v3801, %v3799 : tensor<256x256x14x14xf32>
    %v3803 = stablehlo.subtract %v3797, %v3802 : tensor<256x256x14x14xf32>
    %v3804 = stablehlo.multiply %v3803, %v3803 : tensor<256x256x14x14xf32>
    %v3805 = stablehlo.reduce(%v3804 init: %v3798) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3806 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3807 = stablehlo.divide %v3805, %v3806 : tensor<256xf32>
    %v3808 = stablehlo.reshape %v717 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3810 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3811 = stablehlo.reduce(%v3808 init: %v3809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3812 = stablehlo.divide %v3811, %v3810 : tensor<256xf32>
    %v3813 = stablehlo.reshape %v717 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3814 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3815 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3816 = stablehlo.reduce(%v3813 init: %v3814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3817 = stablehlo.broadcast_in_dim %v3816, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3818 = stablehlo.divide %v3817, %v3815 : tensor<256x256x14x14xf32>
    %v3819 = stablehlo.subtract %v3813, %v3818 : tensor<256x256x14x14xf32>
    %v3820 = stablehlo.multiply %v3819, %v3819 : tensor<256x256x14x14xf32>
    %v3821 = stablehlo.reduce(%v3820 init: %v3814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3822 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3823 = stablehlo.divide %v3821, %v3822 : tensor<256xf32>
    %v3824 = stablehlo.reshape %v745 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3825 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3826 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3827 = stablehlo.reduce(%v3824 init: %v3825) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3828 = stablehlo.divide %v3827, %v3826 : tensor<256xf32>
    %v3829 = stablehlo.reshape %v745 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3831 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3832 = stablehlo.reduce(%v3829 init: %v3830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3833 = stablehlo.broadcast_in_dim %v3832, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3834 = stablehlo.divide %v3833, %v3831 : tensor<256x256x14x14xf32>
    %v3835 = stablehlo.subtract %v3829, %v3834 : tensor<256x256x14x14xf32>
    %v3836 = stablehlo.multiply %v3835, %v3835 : tensor<256x256x14x14xf32>
    %v3837 = stablehlo.reduce(%v3836 init: %v3830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3838 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3839 = stablehlo.divide %v3837, %v3838 : tensor<256xf32>
    %v3840 = stablehlo.reshape %v772 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3841 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3842 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3843 = stablehlo.reduce(%v3840 init: %v3841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3844 = stablehlo.divide %v3843, %v3842 : tensor<256xf32>
    %v3845 = stablehlo.reshape %v772 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3847 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3848 = stablehlo.reduce(%v3845 init: %v3846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3849 = stablehlo.broadcast_in_dim %v3848, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3850 = stablehlo.divide %v3849, %v3847 : tensor<256x256x14x14xf32>
    %v3851 = stablehlo.subtract %v3845, %v3850 : tensor<256x256x14x14xf32>
    %v3852 = stablehlo.multiply %v3851, %v3851 : tensor<256x256x14x14xf32>
    %v3853 = stablehlo.reduce(%v3852 init: %v3846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3854 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3855 = stablehlo.divide %v3853, %v3854 : tensor<256xf32>
    %v3856 = stablehlo.reshape %v800 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3857 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3858 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3859 = stablehlo.reduce(%v3856 init: %v3857) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3860 = stablehlo.divide %v3859, %v3858 : tensor<512xf32>
    %v3861 = stablehlo.reshape %v800 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3863 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v3864 = stablehlo.reduce(%v3861 init: %v3862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3865 = stablehlo.broadcast_in_dim %v3864, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v3866 = stablehlo.divide %v3865, %v3863 : tensor<256x512x7x7xf32>
    %v3867 = stablehlo.subtract %v3861, %v3866 : tensor<256x512x7x7xf32>
    %v3868 = stablehlo.multiply %v3867, %v3867 : tensor<256x512x7x7xf32>
    %v3869 = stablehlo.reduce(%v3868 init: %v3862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3870 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3871 = stablehlo.divide %v3869, %v3870 : tensor<512xf32>
    %v3872 = stablehlo.reshape %v827 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3873 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3874 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3875 = stablehlo.reduce(%v3872 init: %v3873) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3876 = stablehlo.divide %v3875, %v3874 : tensor<512xf32>
    %v3877 = stablehlo.reshape %v827 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3878 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3879 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v3880 = stablehlo.reduce(%v3877 init: %v3878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3881 = stablehlo.broadcast_in_dim %v3880, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v3882 = stablehlo.divide %v3881, %v3879 : tensor<256x512x7x7xf32>
    %v3883 = stablehlo.subtract %v3877, %v3882 : tensor<256x512x7x7xf32>
    %v3884 = stablehlo.multiply %v3883, %v3883 : tensor<256x512x7x7xf32>
    %v3885 = stablehlo.reduce(%v3884 init: %v3878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3886 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3887 = stablehlo.divide %v3885, %v3886 : tensor<512xf32>
    %v3888 = stablehlo.reshape %v852 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3889 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3890 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3891 = stablehlo.reduce(%v3888 init: %v3889) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3892 = stablehlo.divide %v3891, %v3890 : tensor<512xf32>
    %v3893 = stablehlo.reshape %v852 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3895 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v3896 = stablehlo.reduce(%v3893 init: %v3894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3897 = stablehlo.broadcast_in_dim %v3896, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v3898 = stablehlo.divide %v3897, %v3895 : tensor<256x512x7x7xf32>
    %v3899 = stablehlo.subtract %v3893, %v3898 : tensor<256x512x7x7xf32>
    %v3900 = stablehlo.multiply %v3899, %v3899 : tensor<256x512x7x7xf32>
    %v3901 = stablehlo.reduce(%v3900 init: %v3894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3902 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3903 = stablehlo.divide %v3901, %v3902 : tensor<512xf32>
    %v3904 = stablehlo.reshape %v880 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3906 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3907 = stablehlo.reduce(%v3904 init: %v3905) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3908 = stablehlo.divide %v3907, %v3906 : tensor<512xf32>
    %v3909 = stablehlo.reshape %v880 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3910 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3911 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v3912 = stablehlo.reduce(%v3909 init: %v3910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3913 = stablehlo.broadcast_in_dim %v3912, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v3914 = stablehlo.divide %v3913, %v3911 : tensor<256x512x7x7xf32>
    %v3915 = stablehlo.subtract %v3909, %v3914 : tensor<256x512x7x7xf32>
    %v3916 = stablehlo.multiply %v3915, %v3915 : tensor<256x512x7x7xf32>
    %v3917 = stablehlo.reduce(%v3916 init: %v3910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3918 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3919 = stablehlo.divide %v3917, %v3918 : tensor<512xf32>
    %v3920 = stablehlo.reshape %v907 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3922 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3923 = stablehlo.reduce(%v3920 init: %v3921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3924 = stablehlo.divide %v3923, %v3922 : tensor<512xf32>
    %v3925 = stablehlo.reshape %v907 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3927 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v3928 = stablehlo.reduce(%v3925 init: %v3926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3929 = stablehlo.broadcast_in_dim %v3928, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v3930 = stablehlo.divide %v3929, %v3927 : tensor<256x512x7x7xf32>
    %v3931 = stablehlo.subtract %v3925, %v3930 : tensor<256x512x7x7xf32>
    %v3932 = stablehlo.multiply %v3931, %v3931 : tensor<256x512x7x7xf32>
    %v3933 = stablehlo.reduce(%v3932 init: %v3926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3934 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3935 = stablehlo.divide %v3933, %v3934 : tensor<512xf32>
    %v3936 = stablehlo.reshape %v935 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3938 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3939 = stablehlo.reduce(%v3936 init: %v3937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3940 = stablehlo.divide %v3939, %v3938 : tensor<512xf32>
    %v3941 = stablehlo.reshape %v935 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3942 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3943 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v3944 = stablehlo.reduce(%v3941 init: %v3942) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3945 = stablehlo.broadcast_in_dim %v3944, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v3946 = stablehlo.divide %v3945, %v3943 : tensor<256x512x7x7xf32>
    %v3947 = stablehlo.subtract %v3941, %v3946 : tensor<256x512x7x7xf32>
    %v3948 = stablehlo.multiply %v3947, %v3947 : tensor<256x512x7x7xf32>
    %v3949 = stablehlo.reduce(%v3948 init: %v3942) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3950 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3951 = stablehlo.divide %v3949, %v3950 : tensor<512xf32>
    %v3952 = stablehlo.reshape %v962 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3953 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3954 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3955 = stablehlo.reduce(%v3952 init: %v3953) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3956 = stablehlo.divide %v3955, %v3954 : tensor<512xf32>
    %v3957 = stablehlo.reshape %v962 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3958 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3959 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v3960 = stablehlo.reduce(%v3957 init: %v3958) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3961 = stablehlo.broadcast_in_dim %v3960, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v3962 = stablehlo.divide %v3961, %v3959 : tensor<256x512x7x7xf32>
    %v3963 = stablehlo.subtract %v3957, %v3962 : tensor<256x512x7x7xf32>
    %v3964 = stablehlo.multiply %v3963, %v3963 : tensor<256x512x7x7xf32>
    %v3965 = stablehlo.reduce(%v3964 init: %v3958) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3966 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3967 = stablehlo.divide %v3965, %v3966 : tensor<512xf32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %v3968 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v3969 = stablehlo.multiply %v3968, %sW : tensor<64x3x7x7xf32>
    %v3970 = stablehlo.add %v3969, %v3370 : tensor<64x3x7x7xf32>
    %v3971 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v3972 = stablehlo.multiply %v3971, %sWv : tensor<64x3x7x7xf32>
    %v3973 = stablehlo.add %v3972, %v3970 : tensor<64x3x7x7xf32>
    %v3974 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v3975 = stablehlo.multiply %v3974, %v3973 : tensor<64x3x7x7xf32>
    %v3976 = stablehlo.subtract %sW, %v3975 : tensor<64x3x7x7xf32>
    %v3977 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v3978 = stablehlo.multiply %v3977, %sg : tensor<64xf32>
    %v3979 = stablehlo.add %v3978, %v3388 : tensor<64xf32>
    %v3980 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v3981 = stablehlo.multiply %v3980, %sgv : tensor<64xf32>
    %v3982 = stablehlo.add %v3981, %v3979 : tensor<64xf32>
    %v3983 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v3984 = stablehlo.multiply %v3983, %v3982 : tensor<64xf32>
    %v3985 = stablehlo.subtract %sg, %v3984 : tensor<64xf32>
    %v3986 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v3987 = stablehlo.multiply %v3986, %sbt : tensor<64xf32>
    %v3988 = stablehlo.add %v3987, %v3391 : tensor<64xf32>
    %v3989 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v3990 = stablehlo.multiply %v3989, %sbtv : tensor<64xf32>
    %v3991 = stablehlo.add %v3990, %v3988 : tensor<64xf32>
    %v3992 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v3993 = stablehlo.multiply %v3992, %v3991 : tensor<64xf32>
    %v3994 = stablehlo.subtract %sbt, %v3993 : tensor<64xf32>
    %v3995 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v3996 = stablehlo.multiply %v3995, %s1b0W1 : tensor<64x64x3x3xf32>
    %v3997 = stablehlo.add %v3996, %v3276 : tensor<64x64x3x3xf32>
    %v3998 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v3999 = stablehlo.multiply %v3998, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4000 = stablehlo.add %v3999, %v3997 : tensor<64x64x3x3xf32>
    %v4001 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4002 = stablehlo.multiply %v4001, %v4000 : tensor<64x64x3x3xf32>
    %v4003 = stablehlo.subtract %s1b0W1, %v4002 : tensor<64x64x3x3xf32>
    %v4004 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4005 = stablehlo.multiply %v4004, %s1b0g1 : tensor<64xf32>
    %v4006 = stablehlo.add %v4005, %v3294 : tensor<64xf32>
    %v4007 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4008 = stablehlo.multiply %v4007, %s1b0g1v : tensor<64xf32>
    %v4009 = stablehlo.add %v4008, %v4006 : tensor<64xf32>
    %v4010 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4011 = stablehlo.multiply %v4010, %v4009 : tensor<64xf32>
    %v4012 = stablehlo.subtract %s1b0g1, %v4011 : tensor<64xf32>
    %v4013 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4014 = stablehlo.multiply %v4013, %s1b0bt1 : tensor<64xf32>
    %v4015 = stablehlo.add %v4014, %v3297 : tensor<64xf32>
    %v4016 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4017 = stablehlo.multiply %v4016, %s1b0bt1v : tensor<64xf32>
    %v4018 = stablehlo.add %v4017, %v4015 : tensor<64xf32>
    %v4019 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4020 = stablehlo.multiply %v4019, %v4018 : tensor<64xf32>
    %v4021 = stablehlo.subtract %s1b0bt1, %v4020 : tensor<64xf32>
    %v4022 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4023 = stablehlo.multiply %v4022, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4024 = stablehlo.add %v4023, %v3303 : tensor<64x64x3x3xf32>
    %v4025 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4026 = stablehlo.multiply %v4025, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4027 = stablehlo.add %v4026, %v4024 : tensor<64x64x3x3xf32>
    %v4028 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4029 = stablehlo.multiply %v4028, %v4027 : tensor<64x64x3x3xf32>
    %v4030 = stablehlo.subtract %s1b0W2, %v4029 : tensor<64x64x3x3xf32>
    %v4031 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4032 = stablehlo.multiply %v4031, %s1b0g2 : tensor<64xf32>
    %v4033 = stablehlo.add %v4032, %v3321 : tensor<64xf32>
    %v4034 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4035 = stablehlo.multiply %v4034, %s1b0g2v : tensor<64xf32>
    %v4036 = stablehlo.add %v4035, %v4033 : tensor<64xf32>
    %v4037 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4038 = stablehlo.multiply %v4037, %v4036 : tensor<64xf32>
    %v4039 = stablehlo.subtract %s1b0g2, %v4038 : tensor<64xf32>
    %v4040 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4041 = stablehlo.multiply %v4040, %s1b0bt2 : tensor<64xf32>
    %v4042 = stablehlo.add %v4041, %v3324 : tensor<64xf32>
    %v4043 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4044 = stablehlo.multiply %v4043, %s1b0bt2v : tensor<64xf32>
    %v4045 = stablehlo.add %v4044, %v4042 : tensor<64xf32>
    %v4046 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4047 = stablehlo.multiply %v4046, %v4045 : tensor<64xf32>
    %v4048 = stablehlo.subtract %s1b0bt2, %v4047 : tensor<64xf32>
    %v4049 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4050 = stablehlo.multiply %v4049, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4051 = stablehlo.add %v4050, %v3145 : tensor<64x64x3x3xf32>
    %v4052 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4053 = stablehlo.multiply %v4052, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4054 = stablehlo.add %v4053, %v4051 : tensor<64x64x3x3xf32>
    %v4055 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4056 = stablehlo.multiply %v4055, %v4054 : tensor<64x64x3x3xf32>
    %v4057 = stablehlo.subtract %s1b1W1, %v4056 : tensor<64x64x3x3xf32>
    %v4058 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4059 = stablehlo.multiply %v4058, %s1b1g1 : tensor<64xf32>
    %v4060 = stablehlo.add %v4059, %v3163 : tensor<64xf32>
    %v4061 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4062 = stablehlo.multiply %v4061, %s1b1g1v : tensor<64xf32>
    %v4063 = stablehlo.add %v4062, %v4060 : tensor<64xf32>
    %v4064 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4065 = stablehlo.multiply %v4064, %v4063 : tensor<64xf32>
    %v4066 = stablehlo.subtract %s1b1g1, %v4065 : tensor<64xf32>
    %v4067 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4068 = stablehlo.multiply %v4067, %s1b1bt1 : tensor<64xf32>
    %v4069 = stablehlo.add %v4068, %v3166 : tensor<64xf32>
    %v4070 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4071 = stablehlo.multiply %v4070, %s1b1bt1v : tensor<64xf32>
    %v4072 = stablehlo.add %v4071, %v4069 : tensor<64xf32>
    %v4073 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4074 = stablehlo.multiply %v4073, %v4072 : tensor<64xf32>
    %v4075 = stablehlo.subtract %s1b1bt1, %v4074 : tensor<64xf32>
    %v4076 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4077 = stablehlo.multiply %v4076, %s1b1W2 : tensor<64x64x3x3xf32>
    %v4078 = stablehlo.add %v4077, %v3172 : tensor<64x64x3x3xf32>
    %v4079 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4080 = stablehlo.multiply %v4079, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4081 = stablehlo.add %v4080, %v4078 : tensor<64x64x3x3xf32>
    %v4082 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4083 = stablehlo.multiply %v4082, %v4081 : tensor<64x64x3x3xf32>
    %v4084 = stablehlo.subtract %s1b1W2, %v4083 : tensor<64x64x3x3xf32>
    %v4085 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4086 = stablehlo.multiply %v4085, %s1b1g2 : tensor<64xf32>
    %v4087 = stablehlo.add %v4086, %v3190 : tensor<64xf32>
    %v4088 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4089 = stablehlo.multiply %v4088, %s1b1g2v : tensor<64xf32>
    %v4090 = stablehlo.add %v4089, %v4087 : tensor<64xf32>
    %v4091 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4092 = stablehlo.multiply %v4091, %v4090 : tensor<64xf32>
    %v4093 = stablehlo.subtract %s1b1g2, %v4092 : tensor<64xf32>
    %v4094 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4095 = stablehlo.multiply %v4094, %s1b1bt2 : tensor<64xf32>
    %v4096 = stablehlo.add %v4095, %v3193 : tensor<64xf32>
    %v4097 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4098 = stablehlo.multiply %v4097, %s1b1bt2v : tensor<64xf32>
    %v4099 = stablehlo.add %v4098, %v4096 : tensor<64xf32>
    %v4100 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4101 = stablehlo.multiply %v4100, %v4099 : tensor<64xf32>
    %v4102 = stablehlo.subtract %s1b1bt2, %v4101 : tensor<64xf32>
    %v4103 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4104 = stablehlo.multiply %v4103, %s1b2W1 : tensor<64x64x3x3xf32>
    %v4105 = stablehlo.add %v4104, %v3014 : tensor<64x64x3x3xf32>
    %v4106 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4107 = stablehlo.multiply %v4106, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4108 = stablehlo.add %v4107, %v4105 : tensor<64x64x3x3xf32>
    %v4109 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4110 = stablehlo.multiply %v4109, %v4108 : tensor<64x64x3x3xf32>
    %v4111 = stablehlo.subtract %s1b2W1, %v4110 : tensor<64x64x3x3xf32>
    %v4112 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4113 = stablehlo.multiply %v4112, %s1b2g1 : tensor<64xf32>
    %v4114 = stablehlo.add %v4113, %v3032 : tensor<64xf32>
    %v4115 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4116 = stablehlo.multiply %v4115, %s1b2g1v : tensor<64xf32>
    %v4117 = stablehlo.add %v4116, %v4114 : tensor<64xf32>
    %v4118 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4119 = stablehlo.multiply %v4118, %v4117 : tensor<64xf32>
    %v4120 = stablehlo.subtract %s1b2g1, %v4119 : tensor<64xf32>
    %v4121 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4122 = stablehlo.multiply %v4121, %s1b2bt1 : tensor<64xf32>
    %v4123 = stablehlo.add %v4122, %v3035 : tensor<64xf32>
    %v4124 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4125 = stablehlo.multiply %v4124, %s1b2bt1v : tensor<64xf32>
    %v4126 = stablehlo.add %v4125, %v4123 : tensor<64xf32>
    %v4127 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4128 = stablehlo.multiply %v4127, %v4126 : tensor<64xf32>
    %v4129 = stablehlo.subtract %s1b2bt1, %v4128 : tensor<64xf32>
    %v4130 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4131 = stablehlo.multiply %v4130, %s1b2W2 : tensor<64x64x3x3xf32>
    %v4132 = stablehlo.add %v4131, %v3041 : tensor<64x64x3x3xf32>
    %v4133 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4134 = stablehlo.multiply %v4133, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4135 = stablehlo.add %v4134, %v4132 : tensor<64x64x3x3xf32>
    %v4136 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4137 = stablehlo.multiply %v4136, %v4135 : tensor<64x64x3x3xf32>
    %v4138 = stablehlo.subtract %s1b2W2, %v4137 : tensor<64x64x3x3xf32>
    %v4139 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4140 = stablehlo.multiply %v4139, %s1b2g2 : tensor<64xf32>
    %v4141 = stablehlo.add %v4140, %v3059 : tensor<64xf32>
    %v4142 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4143 = stablehlo.multiply %v4142, %s1b2g2v : tensor<64xf32>
    %v4144 = stablehlo.add %v4143, %v4141 : tensor<64xf32>
    %v4145 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4146 = stablehlo.multiply %v4145, %v4144 : tensor<64xf32>
    %v4147 = stablehlo.subtract %s1b2g2, %v4146 : tensor<64xf32>
    %v4148 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4149 = stablehlo.multiply %v4148, %s1b2bt2 : tensor<64xf32>
    %v4150 = stablehlo.add %v4149, %v3062 : tensor<64xf32>
    %v4151 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4152 = stablehlo.multiply %v4151, %s1b2bt2v : tensor<64xf32>
    %v4153 = stablehlo.add %v4152, %v4150 : tensor<64xf32>
    %v4154 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4155 = stablehlo.multiply %v4154, %v4153 : tensor<64xf32>
    %v4156 = stablehlo.subtract %s1b2bt2, %v4155 : tensor<64xf32>
    %v4157 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4158 = stablehlo.multiply %v4157, %d2W1 : tensor<128x64x3x3xf32>
    %v4159 = stablehlo.add %v4158, %v2854 : tensor<128x64x3x3xf32>
    %v4160 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4161 = stablehlo.multiply %v4160, %d2W1v : tensor<128x64x3x3xf32>
    %v4162 = stablehlo.add %v4161, %v4159 : tensor<128x64x3x3xf32>
    %v4163 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4164 = stablehlo.multiply %v4163, %v4162 : tensor<128x64x3x3xf32>
    %v4165 = stablehlo.subtract %d2W1, %v4164 : tensor<128x64x3x3xf32>
    %v4166 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4167 = stablehlo.multiply %v4166, %d2g1 : tensor<128xf32>
    %v4168 = stablehlo.add %v4167, %v2872 : tensor<128xf32>
    %v4169 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4170 = stablehlo.multiply %v4169, %d2g1v : tensor<128xf32>
    %v4171 = stablehlo.add %v4170, %v4168 : tensor<128xf32>
    %v4172 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4173 = stablehlo.multiply %v4172, %v4171 : tensor<128xf32>
    %v4174 = stablehlo.subtract %d2g1, %v4173 : tensor<128xf32>
    %v4175 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4176 = stablehlo.multiply %v4175, %d2bt1 : tensor<128xf32>
    %v4177 = stablehlo.add %v4176, %v2875 : tensor<128xf32>
    %v4178 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4179 = stablehlo.multiply %v4178, %d2bt1v : tensor<128xf32>
    %v4180 = stablehlo.add %v4179, %v4177 : tensor<128xf32>
    %v4181 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4182 = stablehlo.multiply %v4181, %v4180 : tensor<128xf32>
    %v4183 = stablehlo.subtract %d2bt1, %v4182 : tensor<128xf32>
    %v4184 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4185 = stablehlo.multiply %v4184, %d2W2 : tensor<128x128x3x3xf32>
    %v4186 = stablehlo.add %v4185, %v2881 : tensor<128x128x3x3xf32>
    %v4187 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4188 = stablehlo.multiply %v4187, %d2W2v : tensor<128x128x3x3xf32>
    %v4189 = stablehlo.add %v4188, %v4186 : tensor<128x128x3x3xf32>
    %v4190 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4191 = stablehlo.multiply %v4190, %v4189 : tensor<128x128x3x3xf32>
    %v4192 = stablehlo.subtract %d2W2, %v4191 : tensor<128x128x3x3xf32>
    %v4193 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4194 = stablehlo.multiply %v4193, %d2g2 : tensor<128xf32>
    %v4195 = stablehlo.add %v4194, %v2899 : tensor<128xf32>
    %v4196 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4197 = stablehlo.multiply %v4196, %d2g2v : tensor<128xf32>
    %v4198 = stablehlo.add %v4197, %v4195 : tensor<128xf32>
    %v4199 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4200 = stablehlo.multiply %v4199, %v4198 : tensor<128xf32>
    %v4201 = stablehlo.subtract %d2g2, %v4200 : tensor<128xf32>
    %v4202 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4203 = stablehlo.multiply %v4202, %d2bt2 : tensor<128xf32>
    %v4204 = stablehlo.add %v4203, %v2902 : tensor<128xf32>
    %v4205 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4206 = stablehlo.multiply %v4205, %d2bt2v : tensor<128xf32>
    %v4207 = stablehlo.add %v4206, %v4204 : tensor<128xf32>
    %v4208 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4209 = stablehlo.multiply %v4208, %v4207 : tensor<128xf32>
    %v4210 = stablehlo.subtract %d2bt2, %v4209 : tensor<128xf32>
    %v4211 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4212 = stablehlo.multiply %v4211, %d2Wp : tensor<128x64x1x1xf32>
    %v4213 = stablehlo.add %v4212, %v2910 : tensor<128x64x1x1xf32>
    %v4214 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4215 = stablehlo.multiply %v4214, %d2Wpv : tensor<128x64x1x1xf32>
    %v4216 = stablehlo.add %v4215, %v4213 : tensor<128x64x1x1xf32>
    %v4217 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4218 = stablehlo.multiply %v4217, %v4216 : tensor<128x64x1x1xf32>
    %v4219 = stablehlo.subtract %d2Wp, %v4218 : tensor<128x64x1x1xf32>
    %v4220 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4221 = stablehlo.multiply %v4220, %d2gp : tensor<128xf32>
    %v4222 = stablehlo.add %v4221, %v2928 : tensor<128xf32>
    %v4223 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4224 = stablehlo.multiply %v4223, %d2gpv : tensor<128xf32>
    %v4225 = stablehlo.add %v4224, %v4222 : tensor<128xf32>
    %v4226 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4227 = stablehlo.multiply %v4226, %v4225 : tensor<128xf32>
    %v4228 = stablehlo.subtract %d2gp, %v4227 : tensor<128xf32>
    %v4229 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4230 = stablehlo.multiply %v4229, %d2btp : tensor<128xf32>
    %v4231 = stablehlo.add %v4230, %v2931 : tensor<128xf32>
    %v4232 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4233 = stablehlo.multiply %v4232, %d2btpv : tensor<128xf32>
    %v4234 = stablehlo.add %v4233, %v4231 : tensor<128xf32>
    %v4235 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4236 = stablehlo.multiply %v4235, %v4234 : tensor<128xf32>
    %v4237 = stablehlo.subtract %d2btp, %v4236 : tensor<128xf32>
    %v4238 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4239 = stablehlo.multiply %v4238, %s2b0W1 : tensor<128x128x3x3xf32>
    %v4240 = stablehlo.add %v4239, %v2682 : tensor<128x128x3x3xf32>
    %v4241 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4242 = stablehlo.multiply %v4241, %s2b0W1v : tensor<128x128x3x3xf32>
    %v4243 = stablehlo.add %v4242, %v4240 : tensor<128x128x3x3xf32>
    %v4244 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4245 = stablehlo.multiply %v4244, %v4243 : tensor<128x128x3x3xf32>
    %v4246 = stablehlo.subtract %s2b0W1, %v4245 : tensor<128x128x3x3xf32>
    %v4247 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4248 = stablehlo.multiply %v4247, %s2b0g1 : tensor<128xf32>
    %v4249 = stablehlo.add %v4248, %v2700 : tensor<128xf32>
    %v4250 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4251 = stablehlo.multiply %v4250, %s2b0g1v : tensor<128xf32>
    %v4252 = stablehlo.add %v4251, %v4249 : tensor<128xf32>
    %v4253 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4254 = stablehlo.multiply %v4253, %v4252 : tensor<128xf32>
    %v4255 = stablehlo.subtract %s2b0g1, %v4254 : tensor<128xf32>
    %v4256 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4257 = stablehlo.multiply %v4256, %s2b0bt1 : tensor<128xf32>
    %v4258 = stablehlo.add %v4257, %v2703 : tensor<128xf32>
    %v4259 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4260 = stablehlo.multiply %v4259, %s2b0bt1v : tensor<128xf32>
    %v4261 = stablehlo.add %v4260, %v4258 : tensor<128xf32>
    %v4262 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4263 = stablehlo.multiply %v4262, %v4261 : tensor<128xf32>
    %v4264 = stablehlo.subtract %s2b0bt1, %v4263 : tensor<128xf32>
    %v4265 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4266 = stablehlo.multiply %v4265, %s2b0W2 : tensor<128x128x3x3xf32>
    %v4267 = stablehlo.add %v4266, %v2709 : tensor<128x128x3x3xf32>
    %v4268 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4269 = stablehlo.multiply %v4268, %s2b0W2v : tensor<128x128x3x3xf32>
    %v4270 = stablehlo.add %v4269, %v4267 : tensor<128x128x3x3xf32>
    %v4271 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4272 = stablehlo.multiply %v4271, %v4270 : tensor<128x128x3x3xf32>
    %v4273 = stablehlo.subtract %s2b0W2, %v4272 : tensor<128x128x3x3xf32>
    %v4274 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4275 = stablehlo.multiply %v4274, %s2b0g2 : tensor<128xf32>
    %v4276 = stablehlo.add %v4275, %v2727 : tensor<128xf32>
    %v4277 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4278 = stablehlo.multiply %v4277, %s2b0g2v : tensor<128xf32>
    %v4279 = stablehlo.add %v4278, %v4276 : tensor<128xf32>
    %v4280 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4281 = stablehlo.multiply %v4280, %v4279 : tensor<128xf32>
    %v4282 = stablehlo.subtract %s2b0g2, %v4281 : tensor<128xf32>
    %v4283 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4284 = stablehlo.multiply %v4283, %s2b0bt2 : tensor<128xf32>
    %v4285 = stablehlo.add %v4284, %v2730 : tensor<128xf32>
    %v4286 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4287 = stablehlo.multiply %v4286, %s2b0bt2v : tensor<128xf32>
    %v4288 = stablehlo.add %v4287, %v4285 : tensor<128xf32>
    %v4289 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4290 = stablehlo.multiply %v4289, %v4288 : tensor<128xf32>
    %v4291 = stablehlo.subtract %s2b0bt2, %v4290 : tensor<128xf32>
    %v4292 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4293 = stablehlo.multiply %v4292, %s2b1W1 : tensor<128x128x3x3xf32>
    %v4294 = stablehlo.add %v4293, %v2551 : tensor<128x128x3x3xf32>
    %v4295 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4296 = stablehlo.multiply %v4295, %s2b1W1v : tensor<128x128x3x3xf32>
    %v4297 = stablehlo.add %v4296, %v4294 : tensor<128x128x3x3xf32>
    %v4298 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4299 = stablehlo.multiply %v4298, %v4297 : tensor<128x128x3x3xf32>
    %v4300 = stablehlo.subtract %s2b1W1, %v4299 : tensor<128x128x3x3xf32>
    %v4301 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4302 = stablehlo.multiply %v4301, %s2b1g1 : tensor<128xf32>
    %v4303 = stablehlo.add %v4302, %v2569 : tensor<128xf32>
    %v4304 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4305 = stablehlo.multiply %v4304, %s2b1g1v : tensor<128xf32>
    %v4306 = stablehlo.add %v4305, %v4303 : tensor<128xf32>
    %v4307 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4308 = stablehlo.multiply %v4307, %v4306 : tensor<128xf32>
    %v4309 = stablehlo.subtract %s2b1g1, %v4308 : tensor<128xf32>
    %v4310 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4311 = stablehlo.multiply %v4310, %s2b1bt1 : tensor<128xf32>
    %v4312 = stablehlo.add %v4311, %v2572 : tensor<128xf32>
    %v4313 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4314 = stablehlo.multiply %v4313, %s2b1bt1v : tensor<128xf32>
    %v4315 = stablehlo.add %v4314, %v4312 : tensor<128xf32>
    %v4316 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4317 = stablehlo.multiply %v4316, %v4315 : tensor<128xf32>
    %v4318 = stablehlo.subtract %s2b1bt1, %v4317 : tensor<128xf32>
    %v4319 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4320 = stablehlo.multiply %v4319, %s2b1W2 : tensor<128x128x3x3xf32>
    %v4321 = stablehlo.add %v4320, %v2578 : tensor<128x128x3x3xf32>
    %v4322 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4323 = stablehlo.multiply %v4322, %s2b1W2v : tensor<128x128x3x3xf32>
    %v4324 = stablehlo.add %v4323, %v4321 : tensor<128x128x3x3xf32>
    %v4325 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4326 = stablehlo.multiply %v4325, %v4324 : tensor<128x128x3x3xf32>
    %v4327 = stablehlo.subtract %s2b1W2, %v4326 : tensor<128x128x3x3xf32>
    %v4328 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4329 = stablehlo.multiply %v4328, %s2b1g2 : tensor<128xf32>
    %v4330 = stablehlo.add %v4329, %v2596 : tensor<128xf32>
    %v4331 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4332 = stablehlo.multiply %v4331, %s2b1g2v : tensor<128xf32>
    %v4333 = stablehlo.add %v4332, %v4330 : tensor<128xf32>
    %v4334 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4335 = stablehlo.multiply %v4334, %v4333 : tensor<128xf32>
    %v4336 = stablehlo.subtract %s2b1g2, %v4335 : tensor<128xf32>
    %v4337 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4338 = stablehlo.multiply %v4337, %s2b1bt2 : tensor<128xf32>
    %v4339 = stablehlo.add %v4338, %v2599 : tensor<128xf32>
    %v4340 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4341 = stablehlo.multiply %v4340, %s2b1bt2v : tensor<128xf32>
    %v4342 = stablehlo.add %v4341, %v4339 : tensor<128xf32>
    %v4343 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4344 = stablehlo.multiply %v4343, %v4342 : tensor<128xf32>
    %v4345 = stablehlo.subtract %s2b1bt2, %v4344 : tensor<128xf32>
    %v4346 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4347 = stablehlo.multiply %v4346, %s2b2W1 : tensor<128x128x3x3xf32>
    %v4348 = stablehlo.add %v4347, %v2420 : tensor<128x128x3x3xf32>
    %v4349 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4350 = stablehlo.multiply %v4349, %s2b2W1v : tensor<128x128x3x3xf32>
    %v4351 = stablehlo.add %v4350, %v4348 : tensor<128x128x3x3xf32>
    %v4352 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4353 = stablehlo.multiply %v4352, %v4351 : tensor<128x128x3x3xf32>
    %v4354 = stablehlo.subtract %s2b2W1, %v4353 : tensor<128x128x3x3xf32>
    %v4355 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4356 = stablehlo.multiply %v4355, %s2b2g1 : tensor<128xf32>
    %v4357 = stablehlo.add %v4356, %v2438 : tensor<128xf32>
    %v4358 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4359 = stablehlo.multiply %v4358, %s2b2g1v : tensor<128xf32>
    %v4360 = stablehlo.add %v4359, %v4357 : tensor<128xf32>
    %v4361 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4362 = stablehlo.multiply %v4361, %v4360 : tensor<128xf32>
    %v4363 = stablehlo.subtract %s2b2g1, %v4362 : tensor<128xf32>
    %v4364 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4365 = stablehlo.multiply %v4364, %s2b2bt1 : tensor<128xf32>
    %v4366 = stablehlo.add %v4365, %v2441 : tensor<128xf32>
    %v4367 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4368 = stablehlo.multiply %v4367, %s2b2bt1v : tensor<128xf32>
    %v4369 = stablehlo.add %v4368, %v4366 : tensor<128xf32>
    %v4370 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4371 = stablehlo.multiply %v4370, %v4369 : tensor<128xf32>
    %v4372 = stablehlo.subtract %s2b2bt1, %v4371 : tensor<128xf32>
    %v4373 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4374 = stablehlo.multiply %v4373, %s2b2W2 : tensor<128x128x3x3xf32>
    %v4375 = stablehlo.add %v4374, %v2447 : tensor<128x128x3x3xf32>
    %v4376 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4377 = stablehlo.multiply %v4376, %s2b2W2v : tensor<128x128x3x3xf32>
    %v4378 = stablehlo.add %v4377, %v4375 : tensor<128x128x3x3xf32>
    %v4379 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4380 = stablehlo.multiply %v4379, %v4378 : tensor<128x128x3x3xf32>
    %v4381 = stablehlo.subtract %s2b2W2, %v4380 : tensor<128x128x3x3xf32>
    %v4382 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4383 = stablehlo.multiply %v4382, %s2b2g2 : tensor<128xf32>
    %v4384 = stablehlo.add %v4383, %v2465 : tensor<128xf32>
    %v4385 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4386 = stablehlo.multiply %v4385, %s2b2g2v : tensor<128xf32>
    %v4387 = stablehlo.add %v4386, %v4384 : tensor<128xf32>
    %v4388 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4389 = stablehlo.multiply %v4388, %v4387 : tensor<128xf32>
    %v4390 = stablehlo.subtract %s2b2g2, %v4389 : tensor<128xf32>
    %v4391 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4392 = stablehlo.multiply %v4391, %s2b2bt2 : tensor<128xf32>
    %v4393 = stablehlo.add %v4392, %v2468 : tensor<128xf32>
    %v4394 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4395 = stablehlo.multiply %v4394, %s2b2bt2v : tensor<128xf32>
    %v4396 = stablehlo.add %v4395, %v4393 : tensor<128xf32>
    %v4397 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4398 = stablehlo.multiply %v4397, %v4396 : tensor<128xf32>
    %v4399 = stablehlo.subtract %s2b2bt2, %v4398 : tensor<128xf32>
    %v4400 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4401 = stablehlo.multiply %v4400, %d3W1 : tensor<256x128x3x3xf32>
    %v4402 = stablehlo.add %v4401, %v2260 : tensor<256x128x3x3xf32>
    %v4403 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4404 = stablehlo.multiply %v4403, %d3W1v : tensor<256x128x3x3xf32>
    %v4405 = stablehlo.add %v4404, %v4402 : tensor<256x128x3x3xf32>
    %v4406 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4407 = stablehlo.multiply %v4406, %v4405 : tensor<256x128x3x3xf32>
    %v4408 = stablehlo.subtract %d3W1, %v4407 : tensor<256x128x3x3xf32>
    %v4409 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4410 = stablehlo.multiply %v4409, %d3g1 : tensor<256xf32>
    %v4411 = stablehlo.add %v4410, %v2278 : tensor<256xf32>
    %v4412 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4413 = stablehlo.multiply %v4412, %d3g1v : tensor<256xf32>
    %v4414 = stablehlo.add %v4413, %v4411 : tensor<256xf32>
    %v4415 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4416 = stablehlo.multiply %v4415, %v4414 : tensor<256xf32>
    %v4417 = stablehlo.subtract %d3g1, %v4416 : tensor<256xf32>
    %v4418 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4419 = stablehlo.multiply %v4418, %d3bt1 : tensor<256xf32>
    %v4420 = stablehlo.add %v4419, %v2281 : tensor<256xf32>
    %v4421 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4422 = stablehlo.multiply %v4421, %d3bt1v : tensor<256xf32>
    %v4423 = stablehlo.add %v4422, %v4420 : tensor<256xf32>
    %v4424 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4425 = stablehlo.multiply %v4424, %v4423 : tensor<256xf32>
    %v4426 = stablehlo.subtract %d3bt1, %v4425 : tensor<256xf32>
    %v4427 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4428 = stablehlo.multiply %v4427, %d3W2 : tensor<256x256x3x3xf32>
    %v4429 = stablehlo.add %v4428, %v2287 : tensor<256x256x3x3xf32>
    %v4430 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4431 = stablehlo.multiply %v4430, %d3W2v : tensor<256x256x3x3xf32>
    %v4432 = stablehlo.add %v4431, %v4429 : tensor<256x256x3x3xf32>
    %v4433 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4434 = stablehlo.multiply %v4433, %v4432 : tensor<256x256x3x3xf32>
    %v4435 = stablehlo.subtract %d3W2, %v4434 : tensor<256x256x3x3xf32>
    %v4436 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4437 = stablehlo.multiply %v4436, %d3g2 : tensor<256xf32>
    %v4438 = stablehlo.add %v4437, %v2305 : tensor<256xf32>
    %v4439 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4440 = stablehlo.multiply %v4439, %d3g2v : tensor<256xf32>
    %v4441 = stablehlo.add %v4440, %v4438 : tensor<256xf32>
    %v4442 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4443 = stablehlo.multiply %v4442, %v4441 : tensor<256xf32>
    %v4444 = stablehlo.subtract %d3g2, %v4443 : tensor<256xf32>
    %v4445 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4446 = stablehlo.multiply %v4445, %d3bt2 : tensor<256xf32>
    %v4447 = stablehlo.add %v4446, %v2308 : tensor<256xf32>
    %v4448 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4449 = stablehlo.multiply %v4448, %d3bt2v : tensor<256xf32>
    %v4450 = stablehlo.add %v4449, %v4447 : tensor<256xf32>
    %v4451 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4452 = stablehlo.multiply %v4451, %v4450 : tensor<256xf32>
    %v4453 = stablehlo.subtract %d3bt2, %v4452 : tensor<256xf32>
    %v4454 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v4455 = stablehlo.multiply %v4454, %d3Wp : tensor<256x128x1x1xf32>
    %v4456 = stablehlo.add %v4455, %v2316 : tensor<256x128x1x1xf32>
    %v4457 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v4458 = stablehlo.multiply %v4457, %d3Wpv : tensor<256x128x1x1xf32>
    %v4459 = stablehlo.add %v4458, %v4456 : tensor<256x128x1x1xf32>
    %v4460 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v4461 = stablehlo.multiply %v4460, %v4459 : tensor<256x128x1x1xf32>
    %v4462 = stablehlo.subtract %d3Wp, %v4461 : tensor<256x128x1x1xf32>
    %v4463 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4464 = stablehlo.multiply %v4463, %d3gp : tensor<256xf32>
    %v4465 = stablehlo.add %v4464, %v2334 : tensor<256xf32>
    %v4466 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4467 = stablehlo.multiply %v4466, %d3gpv : tensor<256xf32>
    %v4468 = stablehlo.add %v4467, %v4465 : tensor<256xf32>
    %v4469 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4470 = stablehlo.multiply %v4469, %v4468 : tensor<256xf32>
    %v4471 = stablehlo.subtract %d3gp, %v4470 : tensor<256xf32>
    %v4472 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4473 = stablehlo.multiply %v4472, %d3btp : tensor<256xf32>
    %v4474 = stablehlo.add %v4473, %v2337 : tensor<256xf32>
    %v4475 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4476 = stablehlo.multiply %v4475, %d3btpv : tensor<256xf32>
    %v4477 = stablehlo.add %v4476, %v4474 : tensor<256xf32>
    %v4478 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4479 = stablehlo.multiply %v4478, %v4477 : tensor<256xf32>
    %v4480 = stablehlo.subtract %d3btp, %v4479 : tensor<256xf32>
    %v4481 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4482 = stablehlo.multiply %v4481, %s3b0W1 : tensor<256x256x3x3xf32>
    %v4483 = stablehlo.add %v4482, %v2088 : tensor<256x256x3x3xf32>
    %v4484 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4485 = stablehlo.multiply %v4484, %s3b0W1v : tensor<256x256x3x3xf32>
    %v4486 = stablehlo.add %v4485, %v4483 : tensor<256x256x3x3xf32>
    %v4487 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4488 = stablehlo.multiply %v4487, %v4486 : tensor<256x256x3x3xf32>
    %v4489 = stablehlo.subtract %s3b0W1, %v4488 : tensor<256x256x3x3xf32>
    %v4490 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4491 = stablehlo.multiply %v4490, %s3b0g1 : tensor<256xf32>
    %v4492 = stablehlo.add %v4491, %v2106 : tensor<256xf32>
    %v4493 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4494 = stablehlo.multiply %v4493, %s3b0g1v : tensor<256xf32>
    %v4495 = stablehlo.add %v4494, %v4492 : tensor<256xf32>
    %v4496 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4497 = stablehlo.multiply %v4496, %v4495 : tensor<256xf32>
    %v4498 = stablehlo.subtract %s3b0g1, %v4497 : tensor<256xf32>
    %v4499 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4500 = stablehlo.multiply %v4499, %s3b0bt1 : tensor<256xf32>
    %v4501 = stablehlo.add %v4500, %v2109 : tensor<256xf32>
    %v4502 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4503 = stablehlo.multiply %v4502, %s3b0bt1v : tensor<256xf32>
    %v4504 = stablehlo.add %v4503, %v4501 : tensor<256xf32>
    %v4505 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4506 = stablehlo.multiply %v4505, %v4504 : tensor<256xf32>
    %v4507 = stablehlo.subtract %s3b0bt1, %v4506 : tensor<256xf32>
    %v4508 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4509 = stablehlo.multiply %v4508, %s3b0W2 : tensor<256x256x3x3xf32>
    %v4510 = stablehlo.add %v4509, %v2115 : tensor<256x256x3x3xf32>
    %v4511 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4512 = stablehlo.multiply %v4511, %s3b0W2v : tensor<256x256x3x3xf32>
    %v4513 = stablehlo.add %v4512, %v4510 : tensor<256x256x3x3xf32>
    %v4514 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4515 = stablehlo.multiply %v4514, %v4513 : tensor<256x256x3x3xf32>
    %v4516 = stablehlo.subtract %s3b0W2, %v4515 : tensor<256x256x3x3xf32>
    %v4517 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4518 = stablehlo.multiply %v4517, %s3b0g2 : tensor<256xf32>
    %v4519 = stablehlo.add %v4518, %v2133 : tensor<256xf32>
    %v4520 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4521 = stablehlo.multiply %v4520, %s3b0g2v : tensor<256xf32>
    %v4522 = stablehlo.add %v4521, %v4519 : tensor<256xf32>
    %v4523 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4524 = stablehlo.multiply %v4523, %v4522 : tensor<256xf32>
    %v4525 = stablehlo.subtract %s3b0g2, %v4524 : tensor<256xf32>
    %v4526 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4527 = stablehlo.multiply %v4526, %s3b0bt2 : tensor<256xf32>
    %v4528 = stablehlo.add %v4527, %v2136 : tensor<256xf32>
    %v4529 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4530 = stablehlo.multiply %v4529, %s3b0bt2v : tensor<256xf32>
    %v4531 = stablehlo.add %v4530, %v4528 : tensor<256xf32>
    %v4532 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4533 = stablehlo.multiply %v4532, %v4531 : tensor<256xf32>
    %v4534 = stablehlo.subtract %s3b0bt2, %v4533 : tensor<256xf32>
    %v4535 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4536 = stablehlo.multiply %v4535, %s3b1W1 : tensor<256x256x3x3xf32>
    %v4537 = stablehlo.add %v4536, %v1957 : tensor<256x256x3x3xf32>
    %v4538 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4539 = stablehlo.multiply %v4538, %s3b1W1v : tensor<256x256x3x3xf32>
    %v4540 = stablehlo.add %v4539, %v4537 : tensor<256x256x3x3xf32>
    %v4541 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4542 = stablehlo.multiply %v4541, %v4540 : tensor<256x256x3x3xf32>
    %v4543 = stablehlo.subtract %s3b1W1, %v4542 : tensor<256x256x3x3xf32>
    %v4544 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4545 = stablehlo.multiply %v4544, %s3b1g1 : tensor<256xf32>
    %v4546 = stablehlo.add %v4545, %v1975 : tensor<256xf32>
    %v4547 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4548 = stablehlo.multiply %v4547, %s3b1g1v : tensor<256xf32>
    %v4549 = stablehlo.add %v4548, %v4546 : tensor<256xf32>
    %v4550 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4551 = stablehlo.multiply %v4550, %v4549 : tensor<256xf32>
    %v4552 = stablehlo.subtract %s3b1g1, %v4551 : tensor<256xf32>
    %v4553 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4554 = stablehlo.multiply %v4553, %s3b1bt1 : tensor<256xf32>
    %v4555 = stablehlo.add %v4554, %v1978 : tensor<256xf32>
    %v4556 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4557 = stablehlo.multiply %v4556, %s3b1bt1v : tensor<256xf32>
    %v4558 = stablehlo.add %v4557, %v4555 : tensor<256xf32>
    %v4559 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4560 = stablehlo.multiply %v4559, %v4558 : tensor<256xf32>
    %v4561 = stablehlo.subtract %s3b1bt1, %v4560 : tensor<256xf32>
    %v4562 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4563 = stablehlo.multiply %v4562, %s3b1W2 : tensor<256x256x3x3xf32>
    %v4564 = stablehlo.add %v4563, %v1984 : tensor<256x256x3x3xf32>
    %v4565 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4566 = stablehlo.multiply %v4565, %s3b1W2v : tensor<256x256x3x3xf32>
    %v4567 = stablehlo.add %v4566, %v4564 : tensor<256x256x3x3xf32>
    %v4568 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4569 = stablehlo.multiply %v4568, %v4567 : tensor<256x256x3x3xf32>
    %v4570 = stablehlo.subtract %s3b1W2, %v4569 : tensor<256x256x3x3xf32>
    %v4571 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4572 = stablehlo.multiply %v4571, %s3b1g2 : tensor<256xf32>
    %v4573 = stablehlo.add %v4572, %v2002 : tensor<256xf32>
    %v4574 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4575 = stablehlo.multiply %v4574, %s3b1g2v : tensor<256xf32>
    %v4576 = stablehlo.add %v4575, %v4573 : tensor<256xf32>
    %v4577 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4578 = stablehlo.multiply %v4577, %v4576 : tensor<256xf32>
    %v4579 = stablehlo.subtract %s3b1g2, %v4578 : tensor<256xf32>
    %v4580 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4581 = stablehlo.multiply %v4580, %s3b1bt2 : tensor<256xf32>
    %v4582 = stablehlo.add %v4581, %v2005 : tensor<256xf32>
    %v4583 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4584 = stablehlo.multiply %v4583, %s3b1bt2v : tensor<256xf32>
    %v4585 = stablehlo.add %v4584, %v4582 : tensor<256xf32>
    %v4586 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4587 = stablehlo.multiply %v4586, %v4585 : tensor<256xf32>
    %v4588 = stablehlo.subtract %s3b1bt2, %v4587 : tensor<256xf32>
    %v4589 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4590 = stablehlo.multiply %v4589, %s3b2W1 : tensor<256x256x3x3xf32>
    %v4591 = stablehlo.add %v4590, %v1826 : tensor<256x256x3x3xf32>
    %v4592 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4593 = stablehlo.multiply %v4592, %s3b2W1v : tensor<256x256x3x3xf32>
    %v4594 = stablehlo.add %v4593, %v4591 : tensor<256x256x3x3xf32>
    %v4595 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4596 = stablehlo.multiply %v4595, %v4594 : tensor<256x256x3x3xf32>
    %v4597 = stablehlo.subtract %s3b2W1, %v4596 : tensor<256x256x3x3xf32>
    %v4598 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4599 = stablehlo.multiply %v4598, %s3b2g1 : tensor<256xf32>
    %v4600 = stablehlo.add %v4599, %v1844 : tensor<256xf32>
    %v4601 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4602 = stablehlo.multiply %v4601, %s3b2g1v : tensor<256xf32>
    %v4603 = stablehlo.add %v4602, %v4600 : tensor<256xf32>
    %v4604 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4605 = stablehlo.multiply %v4604, %v4603 : tensor<256xf32>
    %v4606 = stablehlo.subtract %s3b2g1, %v4605 : tensor<256xf32>
    %v4607 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4608 = stablehlo.multiply %v4607, %s3b2bt1 : tensor<256xf32>
    %v4609 = stablehlo.add %v4608, %v1847 : tensor<256xf32>
    %v4610 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4611 = stablehlo.multiply %v4610, %s3b2bt1v : tensor<256xf32>
    %v4612 = stablehlo.add %v4611, %v4609 : tensor<256xf32>
    %v4613 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4614 = stablehlo.multiply %v4613, %v4612 : tensor<256xf32>
    %v4615 = stablehlo.subtract %s3b2bt1, %v4614 : tensor<256xf32>
    %v4616 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4617 = stablehlo.multiply %v4616, %s3b2W2 : tensor<256x256x3x3xf32>
    %v4618 = stablehlo.add %v4617, %v1853 : tensor<256x256x3x3xf32>
    %v4619 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4620 = stablehlo.multiply %v4619, %s3b2W2v : tensor<256x256x3x3xf32>
    %v4621 = stablehlo.add %v4620, %v4618 : tensor<256x256x3x3xf32>
    %v4622 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4623 = stablehlo.multiply %v4622, %v4621 : tensor<256x256x3x3xf32>
    %v4624 = stablehlo.subtract %s3b2W2, %v4623 : tensor<256x256x3x3xf32>
    %v4625 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4626 = stablehlo.multiply %v4625, %s3b2g2 : tensor<256xf32>
    %v4627 = stablehlo.add %v4626, %v1871 : tensor<256xf32>
    %v4628 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4629 = stablehlo.multiply %v4628, %s3b2g2v : tensor<256xf32>
    %v4630 = stablehlo.add %v4629, %v4627 : tensor<256xf32>
    %v4631 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4632 = stablehlo.multiply %v4631, %v4630 : tensor<256xf32>
    %v4633 = stablehlo.subtract %s3b2g2, %v4632 : tensor<256xf32>
    %v4634 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4635 = stablehlo.multiply %v4634, %s3b2bt2 : tensor<256xf32>
    %v4636 = stablehlo.add %v4635, %v1874 : tensor<256xf32>
    %v4637 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4638 = stablehlo.multiply %v4637, %s3b2bt2v : tensor<256xf32>
    %v4639 = stablehlo.add %v4638, %v4636 : tensor<256xf32>
    %v4640 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4641 = stablehlo.multiply %v4640, %v4639 : tensor<256xf32>
    %v4642 = stablehlo.subtract %s3b2bt2, %v4641 : tensor<256xf32>
    %v4643 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4644 = stablehlo.multiply %v4643, %s3b3W1 : tensor<256x256x3x3xf32>
    %v4645 = stablehlo.add %v4644, %v1695 : tensor<256x256x3x3xf32>
    %v4646 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4647 = stablehlo.multiply %v4646, %s3b3W1v : tensor<256x256x3x3xf32>
    %v4648 = stablehlo.add %v4647, %v4645 : tensor<256x256x3x3xf32>
    %v4649 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4650 = stablehlo.multiply %v4649, %v4648 : tensor<256x256x3x3xf32>
    %v4651 = stablehlo.subtract %s3b3W1, %v4650 : tensor<256x256x3x3xf32>
    %v4652 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4653 = stablehlo.multiply %v4652, %s3b3g1 : tensor<256xf32>
    %v4654 = stablehlo.add %v4653, %v1713 : tensor<256xf32>
    %v4655 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4656 = stablehlo.multiply %v4655, %s3b3g1v : tensor<256xf32>
    %v4657 = stablehlo.add %v4656, %v4654 : tensor<256xf32>
    %v4658 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4659 = stablehlo.multiply %v4658, %v4657 : tensor<256xf32>
    %v4660 = stablehlo.subtract %s3b3g1, %v4659 : tensor<256xf32>
    %v4661 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4662 = stablehlo.multiply %v4661, %s3b3bt1 : tensor<256xf32>
    %v4663 = stablehlo.add %v4662, %v1716 : tensor<256xf32>
    %v4664 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4665 = stablehlo.multiply %v4664, %s3b3bt1v : tensor<256xf32>
    %v4666 = stablehlo.add %v4665, %v4663 : tensor<256xf32>
    %v4667 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4668 = stablehlo.multiply %v4667, %v4666 : tensor<256xf32>
    %v4669 = stablehlo.subtract %s3b3bt1, %v4668 : tensor<256xf32>
    %v4670 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4671 = stablehlo.multiply %v4670, %s3b3W2 : tensor<256x256x3x3xf32>
    %v4672 = stablehlo.add %v4671, %v1722 : tensor<256x256x3x3xf32>
    %v4673 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4674 = stablehlo.multiply %v4673, %s3b3W2v : tensor<256x256x3x3xf32>
    %v4675 = stablehlo.add %v4674, %v4672 : tensor<256x256x3x3xf32>
    %v4676 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4677 = stablehlo.multiply %v4676, %v4675 : tensor<256x256x3x3xf32>
    %v4678 = stablehlo.subtract %s3b3W2, %v4677 : tensor<256x256x3x3xf32>
    %v4679 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4680 = stablehlo.multiply %v4679, %s3b3g2 : tensor<256xf32>
    %v4681 = stablehlo.add %v4680, %v1740 : tensor<256xf32>
    %v4682 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4683 = stablehlo.multiply %v4682, %s3b3g2v : tensor<256xf32>
    %v4684 = stablehlo.add %v4683, %v4681 : tensor<256xf32>
    %v4685 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4686 = stablehlo.multiply %v4685, %v4684 : tensor<256xf32>
    %v4687 = stablehlo.subtract %s3b3g2, %v4686 : tensor<256xf32>
    %v4688 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4689 = stablehlo.multiply %v4688, %s3b3bt2 : tensor<256xf32>
    %v4690 = stablehlo.add %v4689, %v1743 : tensor<256xf32>
    %v4691 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4692 = stablehlo.multiply %v4691, %s3b3bt2v : tensor<256xf32>
    %v4693 = stablehlo.add %v4692, %v4690 : tensor<256xf32>
    %v4694 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4695 = stablehlo.multiply %v4694, %v4693 : tensor<256xf32>
    %v4696 = stablehlo.subtract %s3b3bt2, %v4695 : tensor<256xf32>
    %v4697 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4698 = stablehlo.multiply %v4697, %s3b4W1 : tensor<256x256x3x3xf32>
    %v4699 = stablehlo.add %v4698, %v1564 : tensor<256x256x3x3xf32>
    %v4700 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4701 = stablehlo.multiply %v4700, %s3b4W1v : tensor<256x256x3x3xf32>
    %v4702 = stablehlo.add %v4701, %v4699 : tensor<256x256x3x3xf32>
    %v4703 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4704 = stablehlo.multiply %v4703, %v4702 : tensor<256x256x3x3xf32>
    %v4705 = stablehlo.subtract %s3b4W1, %v4704 : tensor<256x256x3x3xf32>
    %v4706 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4707 = stablehlo.multiply %v4706, %s3b4g1 : tensor<256xf32>
    %v4708 = stablehlo.add %v4707, %v1582 : tensor<256xf32>
    %v4709 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4710 = stablehlo.multiply %v4709, %s3b4g1v : tensor<256xf32>
    %v4711 = stablehlo.add %v4710, %v4708 : tensor<256xf32>
    %v4712 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4713 = stablehlo.multiply %v4712, %v4711 : tensor<256xf32>
    %v4714 = stablehlo.subtract %s3b4g1, %v4713 : tensor<256xf32>
    %v4715 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4716 = stablehlo.multiply %v4715, %s3b4bt1 : tensor<256xf32>
    %v4717 = stablehlo.add %v4716, %v1585 : tensor<256xf32>
    %v4718 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4719 = stablehlo.multiply %v4718, %s3b4bt1v : tensor<256xf32>
    %v4720 = stablehlo.add %v4719, %v4717 : tensor<256xf32>
    %v4721 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4722 = stablehlo.multiply %v4721, %v4720 : tensor<256xf32>
    %v4723 = stablehlo.subtract %s3b4bt1, %v4722 : tensor<256xf32>
    %v4724 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4725 = stablehlo.multiply %v4724, %s3b4W2 : tensor<256x256x3x3xf32>
    %v4726 = stablehlo.add %v4725, %v1591 : tensor<256x256x3x3xf32>
    %v4727 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4728 = stablehlo.multiply %v4727, %s3b4W2v : tensor<256x256x3x3xf32>
    %v4729 = stablehlo.add %v4728, %v4726 : tensor<256x256x3x3xf32>
    %v4730 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4731 = stablehlo.multiply %v4730, %v4729 : tensor<256x256x3x3xf32>
    %v4732 = stablehlo.subtract %s3b4W2, %v4731 : tensor<256x256x3x3xf32>
    %v4733 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4734 = stablehlo.multiply %v4733, %s3b4g2 : tensor<256xf32>
    %v4735 = stablehlo.add %v4734, %v1609 : tensor<256xf32>
    %v4736 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4737 = stablehlo.multiply %v4736, %s3b4g2v : tensor<256xf32>
    %v4738 = stablehlo.add %v4737, %v4735 : tensor<256xf32>
    %v4739 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4740 = stablehlo.multiply %v4739, %v4738 : tensor<256xf32>
    %v4741 = stablehlo.subtract %s3b4g2, %v4740 : tensor<256xf32>
    %v4742 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4743 = stablehlo.multiply %v4742, %s3b4bt2 : tensor<256xf32>
    %v4744 = stablehlo.add %v4743, %v1612 : tensor<256xf32>
    %v4745 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4746 = stablehlo.multiply %v4745, %s3b4bt2v : tensor<256xf32>
    %v4747 = stablehlo.add %v4746, %v4744 : tensor<256xf32>
    %v4748 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4749 = stablehlo.multiply %v4748, %v4747 : tensor<256xf32>
    %v4750 = stablehlo.subtract %s3b4bt2, %v4749 : tensor<256xf32>
    %v4751 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v4752 = stablehlo.multiply %v4751, %d4W1 : tensor<512x256x3x3xf32>
    %v4753 = stablehlo.add %v4752, %v1404 : tensor<512x256x3x3xf32>
    %v4754 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v4755 = stablehlo.multiply %v4754, %d4W1v : tensor<512x256x3x3xf32>
    %v4756 = stablehlo.add %v4755, %v4753 : tensor<512x256x3x3xf32>
    %v4757 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v4758 = stablehlo.multiply %v4757, %v4756 : tensor<512x256x3x3xf32>
    %v4759 = stablehlo.subtract %d4W1, %v4758 : tensor<512x256x3x3xf32>
    %v4760 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4761 = stablehlo.multiply %v4760, %d4g1 : tensor<512xf32>
    %v4762 = stablehlo.add %v4761, %v1422 : tensor<512xf32>
    %v4763 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4764 = stablehlo.multiply %v4763, %d4g1v : tensor<512xf32>
    %v4765 = stablehlo.add %v4764, %v4762 : tensor<512xf32>
    %v4766 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4767 = stablehlo.multiply %v4766, %v4765 : tensor<512xf32>
    %v4768 = stablehlo.subtract %d4g1, %v4767 : tensor<512xf32>
    %v4769 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4770 = stablehlo.multiply %v4769, %d4bt1 : tensor<512xf32>
    %v4771 = stablehlo.add %v4770, %v1425 : tensor<512xf32>
    %v4772 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4773 = stablehlo.multiply %v4772, %d4bt1v : tensor<512xf32>
    %v4774 = stablehlo.add %v4773, %v4771 : tensor<512xf32>
    %v4775 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4776 = stablehlo.multiply %v4775, %v4774 : tensor<512xf32>
    %v4777 = stablehlo.subtract %d4bt1, %v4776 : tensor<512xf32>
    %v4778 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v4779 = stablehlo.multiply %v4778, %d4W2 : tensor<512x512x3x3xf32>
    %v4780 = stablehlo.add %v4779, %v1431 : tensor<512x512x3x3xf32>
    %v4781 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v4782 = stablehlo.multiply %v4781, %d4W2v : tensor<512x512x3x3xf32>
    %v4783 = stablehlo.add %v4782, %v4780 : tensor<512x512x3x3xf32>
    %v4784 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v4785 = stablehlo.multiply %v4784, %v4783 : tensor<512x512x3x3xf32>
    %v4786 = stablehlo.subtract %d4W2, %v4785 : tensor<512x512x3x3xf32>
    %v4787 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4788 = stablehlo.multiply %v4787, %d4g2 : tensor<512xf32>
    %v4789 = stablehlo.add %v4788, %v1449 : tensor<512xf32>
    %v4790 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4791 = stablehlo.multiply %v4790, %d4g2v : tensor<512xf32>
    %v4792 = stablehlo.add %v4791, %v4789 : tensor<512xf32>
    %v4793 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4794 = stablehlo.multiply %v4793, %v4792 : tensor<512xf32>
    %v4795 = stablehlo.subtract %d4g2, %v4794 : tensor<512xf32>
    %v4796 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4797 = stablehlo.multiply %v4796, %d4bt2 : tensor<512xf32>
    %v4798 = stablehlo.add %v4797, %v1452 : tensor<512xf32>
    %v4799 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4800 = stablehlo.multiply %v4799, %d4bt2v : tensor<512xf32>
    %v4801 = stablehlo.add %v4800, %v4798 : tensor<512xf32>
    %v4802 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4803 = stablehlo.multiply %v4802, %v4801 : tensor<512xf32>
    %v4804 = stablehlo.subtract %d4bt2, %v4803 : tensor<512xf32>
    %v4805 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v4806 = stablehlo.multiply %v4805, %d4Wp : tensor<512x256x1x1xf32>
    %v4807 = stablehlo.add %v4806, %v1460 : tensor<512x256x1x1xf32>
    %v4808 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v4809 = stablehlo.multiply %v4808, %d4Wpv : tensor<512x256x1x1xf32>
    %v4810 = stablehlo.add %v4809, %v4807 : tensor<512x256x1x1xf32>
    %v4811 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v4812 = stablehlo.multiply %v4811, %v4810 : tensor<512x256x1x1xf32>
    %v4813 = stablehlo.subtract %d4Wp, %v4812 : tensor<512x256x1x1xf32>
    %v4814 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4815 = stablehlo.multiply %v4814, %d4gp : tensor<512xf32>
    %v4816 = stablehlo.add %v4815, %v1478 : tensor<512xf32>
    %v4817 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4818 = stablehlo.multiply %v4817, %d4gpv : tensor<512xf32>
    %v4819 = stablehlo.add %v4818, %v4816 : tensor<512xf32>
    %v4820 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4821 = stablehlo.multiply %v4820, %v4819 : tensor<512xf32>
    %v4822 = stablehlo.subtract %d4gp, %v4821 : tensor<512xf32>
    %v4823 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4824 = stablehlo.multiply %v4823, %d4btp : tensor<512xf32>
    %v4825 = stablehlo.add %v4824, %v1481 : tensor<512xf32>
    %v4826 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4827 = stablehlo.multiply %v4826, %d4btpv : tensor<512xf32>
    %v4828 = stablehlo.add %v4827, %v4825 : tensor<512xf32>
    %v4829 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4830 = stablehlo.multiply %v4829, %v4828 : tensor<512xf32>
    %v4831 = stablehlo.subtract %d4btp, %v4830 : tensor<512xf32>
    %v4832 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v4833 = stablehlo.multiply %v4832, %s4b0W1 : tensor<512x512x3x3xf32>
    %v4834 = stablehlo.add %v4833, %v1232 : tensor<512x512x3x3xf32>
    %v4835 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v4836 = stablehlo.multiply %v4835, %s4b0W1v : tensor<512x512x3x3xf32>
    %v4837 = stablehlo.add %v4836, %v4834 : tensor<512x512x3x3xf32>
    %v4838 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v4839 = stablehlo.multiply %v4838, %v4837 : tensor<512x512x3x3xf32>
    %v4840 = stablehlo.subtract %s4b0W1, %v4839 : tensor<512x512x3x3xf32>
    %v4841 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4842 = stablehlo.multiply %v4841, %s4b0g1 : tensor<512xf32>
    %v4843 = stablehlo.add %v4842, %v1250 : tensor<512xf32>
    %v4844 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4845 = stablehlo.multiply %v4844, %s4b0g1v : tensor<512xf32>
    %v4846 = stablehlo.add %v4845, %v4843 : tensor<512xf32>
    %v4847 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4848 = stablehlo.multiply %v4847, %v4846 : tensor<512xf32>
    %v4849 = stablehlo.subtract %s4b0g1, %v4848 : tensor<512xf32>
    %v4850 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4851 = stablehlo.multiply %v4850, %s4b0bt1 : tensor<512xf32>
    %v4852 = stablehlo.add %v4851, %v1253 : tensor<512xf32>
    %v4853 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4854 = stablehlo.multiply %v4853, %s4b0bt1v : tensor<512xf32>
    %v4855 = stablehlo.add %v4854, %v4852 : tensor<512xf32>
    %v4856 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4857 = stablehlo.multiply %v4856, %v4855 : tensor<512xf32>
    %v4858 = stablehlo.subtract %s4b0bt1, %v4857 : tensor<512xf32>
    %v4859 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v4860 = stablehlo.multiply %v4859, %s4b0W2 : tensor<512x512x3x3xf32>
    %v4861 = stablehlo.add %v4860, %v1259 : tensor<512x512x3x3xf32>
    %v4862 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v4863 = stablehlo.multiply %v4862, %s4b0W2v : tensor<512x512x3x3xf32>
    %v4864 = stablehlo.add %v4863, %v4861 : tensor<512x512x3x3xf32>
    %v4865 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v4866 = stablehlo.multiply %v4865, %v4864 : tensor<512x512x3x3xf32>
    %v4867 = stablehlo.subtract %s4b0W2, %v4866 : tensor<512x512x3x3xf32>
    %v4868 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4869 = stablehlo.multiply %v4868, %s4b0g2 : tensor<512xf32>
    %v4870 = stablehlo.add %v4869, %v1277 : tensor<512xf32>
    %v4871 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4872 = stablehlo.multiply %v4871, %s4b0g2v : tensor<512xf32>
    %v4873 = stablehlo.add %v4872, %v4870 : tensor<512xf32>
    %v4874 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4875 = stablehlo.multiply %v4874, %v4873 : tensor<512xf32>
    %v4876 = stablehlo.subtract %s4b0g2, %v4875 : tensor<512xf32>
    %v4877 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4878 = stablehlo.multiply %v4877, %s4b0bt2 : tensor<512xf32>
    %v4879 = stablehlo.add %v4878, %v1280 : tensor<512xf32>
    %v4880 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4881 = stablehlo.multiply %v4880, %s4b0bt2v : tensor<512xf32>
    %v4882 = stablehlo.add %v4881, %v4879 : tensor<512xf32>
    %v4883 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4884 = stablehlo.multiply %v4883, %v4882 : tensor<512xf32>
    %v4885 = stablehlo.subtract %s4b0bt2, %v4884 : tensor<512xf32>
    %v4886 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v4887 = stablehlo.multiply %v4886, %s4b1W1 : tensor<512x512x3x3xf32>
    %v4888 = stablehlo.add %v4887, %v1101 : tensor<512x512x3x3xf32>
    %v4889 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v4890 = stablehlo.multiply %v4889, %s4b1W1v : tensor<512x512x3x3xf32>
    %v4891 = stablehlo.add %v4890, %v4888 : tensor<512x512x3x3xf32>
    %v4892 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v4893 = stablehlo.multiply %v4892, %v4891 : tensor<512x512x3x3xf32>
    %v4894 = stablehlo.subtract %s4b1W1, %v4893 : tensor<512x512x3x3xf32>
    %v4895 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4896 = stablehlo.multiply %v4895, %s4b1g1 : tensor<512xf32>
    %v4897 = stablehlo.add %v4896, %v1119 : tensor<512xf32>
    %v4898 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4899 = stablehlo.multiply %v4898, %s4b1g1v : tensor<512xf32>
    %v4900 = stablehlo.add %v4899, %v4897 : tensor<512xf32>
    %v4901 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4902 = stablehlo.multiply %v4901, %v4900 : tensor<512xf32>
    %v4903 = stablehlo.subtract %s4b1g1, %v4902 : tensor<512xf32>
    %v4904 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4905 = stablehlo.multiply %v4904, %s4b1bt1 : tensor<512xf32>
    %v4906 = stablehlo.add %v4905, %v1122 : tensor<512xf32>
    %v4907 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4908 = stablehlo.multiply %v4907, %s4b1bt1v : tensor<512xf32>
    %v4909 = stablehlo.add %v4908, %v4906 : tensor<512xf32>
    %v4910 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4911 = stablehlo.multiply %v4910, %v4909 : tensor<512xf32>
    %v4912 = stablehlo.subtract %s4b1bt1, %v4911 : tensor<512xf32>
    %v4913 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v4914 = stablehlo.multiply %v4913, %s4b1W2 : tensor<512x512x3x3xf32>
    %v4915 = stablehlo.add %v4914, %v1128 : tensor<512x512x3x3xf32>
    %v4916 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v4917 = stablehlo.multiply %v4916, %s4b1W2v : tensor<512x512x3x3xf32>
    %v4918 = stablehlo.add %v4917, %v4915 : tensor<512x512x3x3xf32>
    %v4919 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v4920 = stablehlo.multiply %v4919, %v4918 : tensor<512x512x3x3xf32>
    %v4921 = stablehlo.subtract %s4b1W2, %v4920 : tensor<512x512x3x3xf32>
    %v4922 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4923 = stablehlo.multiply %v4922, %s4b1g2 : tensor<512xf32>
    %v4924 = stablehlo.add %v4923, %v1146 : tensor<512xf32>
    %v4925 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4926 = stablehlo.multiply %v4925, %s4b1g2v : tensor<512xf32>
    %v4927 = stablehlo.add %v4926, %v4924 : tensor<512xf32>
    %v4928 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4929 = stablehlo.multiply %v4928, %v4927 : tensor<512xf32>
    %v4930 = stablehlo.subtract %s4b1g2, %v4929 : tensor<512xf32>
    %v4931 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4932 = stablehlo.multiply %v4931, %s4b1bt2 : tensor<512xf32>
    %v4933 = stablehlo.add %v4932, %v1149 : tensor<512xf32>
    %v4934 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4935 = stablehlo.multiply %v4934, %s4b1bt2v : tensor<512xf32>
    %v4936 = stablehlo.add %v4935, %v4933 : tensor<512xf32>
    %v4937 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v4938 = stablehlo.multiply %v4937, %v4936 : tensor<512xf32>
    %v4939 = stablehlo.subtract %s4b1bt2, %v4938 : tensor<512xf32>
    %v4940 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %v4941 = stablehlo.multiply %v4940, %Wd : tensor<512x1000xf32>
    %v4942 = stablehlo.add %v4941, %v1012 : tensor<512x1000xf32>
    %v4943 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %v4944 = stablehlo.multiply %v4943, %Wdv : tensor<512x1000xf32>
    %v4945 = stablehlo.add %v4944, %v4942 : tensor<512x1000xf32>
    %v4946 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %v4947 = stablehlo.multiply %v4946, %v4945 : tensor<512x1000xf32>
    %v4948 = stablehlo.subtract %Wd, %v4947 : tensor<512x1000xf32>
    %v4949 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v4950 = stablehlo.multiply %v4949, %bd : tensor<1000xf32>
    %v4951 = stablehlo.add %v4950, %v1014 : tensor<1000xf32>
    %v4952 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v4953 = stablehlo.multiply %v4952, %bdv : tensor<1000xf32>
    %v4954 = stablehlo.add %v4953, %v4951 : tensor<1000xf32>
    %v4955 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v4956 = stablehlo.multiply %v4955, %v4954 : tensor<1000xf32>
    %v4957 = stablehlo.subtract %bd, %v4956 : tensor<1000xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1000 : tensor<256x1000xf32>
    %lohll = stablehlo.multiply %onehot, %llog : tensor<256x1000xf32>
    %lt1s = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
    %llsr = stablehlo.reduce(%llog init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
    %lomac = stablehlo.constant dense<0.900000> : tensor<256xf32>
    %laKc = stablehlo.constant dense<0.000100> : tensor<256xf32>
    %llt1 = stablehlo.multiply %lomac, %lt1s : tensor<256xf32>
    %llt2 = stablehlo.multiply %laKc, %llsr : tensor<256xf32>
    %llpe = stablehlo.add %llt1, %llt2 : tensor<256xf32>
    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    %lbfc = stablehlo.constant dense<256.0> : tensor<f32>
    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>
    %loss = stablehlo.negate %lossm : tensor<f32>
    return %v3976, %v3985, %v3994, %v4003, %v4012, %v4021, %v4030, %v4039, %v4048, %v4057, %v4066, %v4075, %v4084, %v4093, %v4102, %v4111, %v4120, %v4129, %v4138, %v4147, %v4156, %v4165, %v4174, %v4183, %v4192, %v4201, %v4210, %v4219, %v4228, %v4237, %v4246, %v4255, %v4264, %v4273, %v4282, %v4291, %v4300, %v4309, %v4318, %v4327, %v4336, %v4345, %v4354, %v4363, %v4372, %v4381, %v4390, %v4399, %v4408, %v4417, %v4426, %v4435, %v4444, %v4453, %v4462, %v4471, %v4480, %v4489, %v4498, %v4507, %v4516, %v4525, %v4534, %v4543, %v4552, %v4561, %v4570, %v4579, %v4588, %v4597, %v4606, %v4615, %v4624, %v4633, %v4642, %v4651, %v4660, %v4669, %v4678, %v4687, %v4696, %v4705, %v4714, %v4723, %v4732, %v4741, %v4750, %v4759, %v4768, %v4777, %v4786, %v4795, %v4804, %v4813, %v4822, %v4831, %v4840, %v4849, %v4858, %v4867, %v4876, %v4885, %v4894, %v4903, %v4912, %v4921, %v4930, %v4939, %v4948, %v4957, %sWm, %sgm, %sbtm, %s1b0W1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0g2m, %s1b0bt2m, %s1b1W1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1g2m, %s1b1bt2m, %s1b2W1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2g2m, %s1b2bt2m, %d2W1m, %d2g1m, %d2bt1m, %d2W2m, %d2g2m, %d2bt2m, %d2Wpm, %d2gpm, %d2btpm, %s2b0W1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0g2m, %s2b0bt2m, %s2b1W1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1g2m, %s2b1bt2m, %s2b2W1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2g2m, %s2b2bt2m, %d3W1m, %d3g1m, %d3bt1m, %d3W2m, %d3g2m, %d3bt2m, %d3Wpm, %d3gpm, %d3btpm, %s3b0W1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0g2m, %s3b0bt2m, %s3b1W1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1g2m, %s3b1bt2m, %s3b2W1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2g2m, %s3b2bt2m, %s3b3W1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3g2m, %s3b3bt2m, %s3b4W1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4g2m, %s3b4bt2m, %d4W1m, %d4g1m, %d4bt1m, %d4W2m, %d4g2m, %d4bt2m, %d4Wpm, %d4gpm, %d4btpm, %s4b0W1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0g2m, %s4b0bt2m, %s4b1W1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1g2m, %s4b1bt2m, %Wdm, %bdm, %v3973, %v3982, %v3991, %v4000, %v4009, %v4018, %v4027, %v4036, %v4045, %v4054, %v4063, %v4072, %v4081, %v4090, %v4099, %v4108, %v4117, %v4126, %v4135, %v4144, %v4153, %v4162, %v4171, %v4180, %v4189, %v4198, %v4207, %v4216, %v4225, %v4234, %v4243, %v4252, %v4261, %v4270, %v4279, %v4288, %v4297, %v4306, %v4315, %v4324, %v4333, %v4342, %v4351, %v4360, %v4369, %v4378, %v4387, %v4396, %v4405, %v4414, %v4423, %v4432, %v4441, %v4450, %v4459, %v4468, %v4477, %v4486, %v4495, %v4504, %v4513, %v4522, %v4531, %v4540, %v4549, %v4558, %v4567, %v4576, %v4585, %v4594, %v4603, %v4612, %v4621, %v4630, %v4639, %v4648, %v4657, %v4666, %v4675, %v4684, %v4693, %v4702, %v4711, %v4720, %v4729, %v4738, %v4747, %v4756, %v4765, %v4774, %v4783, %v4792, %v4801, %v4810, %v4819, %v4828, %v4837, %v4846, %v4855, %v4864, %v4873, %v4882, %v4891, %v4900, %v4909, %v4918, %v4927, %v4936, %v4945, %v4954, %loss, %bc1, %bc2, %v3396, %v3407, %v3412, %v3423, %v3428, %v3439, %v3444, %v3455, %v3460, %v3471, %v3476, %v3487, %v3492, %v3503, %v3508, %v3519, %v3524, %v3535, %v3540, %v3551, %v3556, %v3567, %v3572, %v3583, %v3588, %v3599, %v3604, %v3615, %v3620, %v3631, %v3636, %v3647, %v3652, %v3663, %v3668, %v3679, %v3684, %v3695, %v3700, %v3711, %v3716, %v3727, %v3732, %v3743, %v3748, %v3759, %v3764, %v3775, %v3780, %v3791, %v3796, %v3807, %v3812, %v3823, %v3828, %v3839, %v3844, %v3855, %v3860, %v3871, %v3876, %v3887, %v3892, %v3903, %v3908, %v3919, %v3924, %v3935, %v3940, %v3951, %v3956, %v3967 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}

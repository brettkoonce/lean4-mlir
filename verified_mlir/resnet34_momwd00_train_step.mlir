module @m {
  func.func @resnet34_momwd00_train_step(%x: tensor<32x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x1x1xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x1x1xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x1x1xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x3x3xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b1W1m: tensor<64x64x3x3xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b2W1m: tensor<64x64x3x3xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %d2W1m: tensor<128x64x3x3xf32>, %d2g1m: tensor<128xf32>, %d2bt1m: tensor<128xf32>, %d2W2m: tensor<128x128x3x3xf32>, %d2g2m: tensor<128xf32>, %d2bt2m: tensor<128xf32>, %d2Wpm: tensor<128x64x1x1xf32>, %d2gpm: tensor<128xf32>, %d2btpm: tensor<128xf32>, %s2b0W1m: tensor<128x128x3x3xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b1W1m: tensor<128x128x3x3xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b2W1m: tensor<128x128x3x3xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %d3W1m: tensor<256x128x3x3xf32>, %d3g1m: tensor<256xf32>, %d3bt1m: tensor<256xf32>, %d3W2m: tensor<256x256x3x3xf32>, %d3g2m: tensor<256xf32>, %d3bt2m: tensor<256xf32>, %d3Wpm: tensor<256x128x1x1xf32>, %d3gpm: tensor<256xf32>, %d3btpm: tensor<256xf32>, %s3b0W1m: tensor<256x256x3x3xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b1W1m: tensor<256x256x3x3xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b2W1m: tensor<256x256x3x3xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b3W1m: tensor<256x256x3x3xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b4W1m: tensor<256x256x3x3xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %d4W1m: tensor<512x256x3x3xf32>, %d4g1m: tensor<512xf32>, %d4bt1m: tensor<512xf32>, %d4W2m: tensor<512x512x3x3xf32>, %d4g2m: tensor<512xf32>, %d4bt2m: tensor<512xf32>, %d4Wpm: tensor<512x256x1x1xf32>, %d4gpm: tensor<512xf32>, %d4btpm: tensor<512xf32>, %s4b0W1m: tensor<512x512x3x3xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b1W1m: tensor<512x512x3x3xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %Wdm: tensor<512x10xf32>, %bdm: tensor<10xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x3x3xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b1W1v: tensor<64x64x3x3xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b2W1v: tensor<64x64x3x3xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %d2W1v: tensor<128x64x3x3xf32>, %d2g1v: tensor<128xf32>, %d2bt1v: tensor<128xf32>, %d2W2v: tensor<128x128x3x3xf32>, %d2g2v: tensor<128xf32>, %d2bt2v: tensor<128xf32>, %d2Wpv: tensor<128x64x1x1xf32>, %d2gpv: tensor<128xf32>, %d2btpv: tensor<128xf32>, %s2b0W1v: tensor<128x128x3x3xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b1W1v: tensor<128x128x3x3xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b2W1v: tensor<128x128x3x3xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %d3W1v: tensor<256x128x3x3xf32>, %d3g1v: tensor<256xf32>, %d3bt1v: tensor<256xf32>, %d3W2v: tensor<256x256x3x3xf32>, %d3g2v: tensor<256xf32>, %d3bt2v: tensor<256xf32>, %d3Wpv: tensor<256x128x1x1xf32>, %d3gpv: tensor<256xf32>, %d3btpv: tensor<256xf32>, %s3b0W1v: tensor<256x256x3x3xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b1W1v: tensor<256x256x3x3xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b2W1v: tensor<256x256x3x3xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b3W1v: tensor<256x256x3x3xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b4W1v: tensor<256x256x3x3xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %d4W1v: tensor<512x256x3x3xf32>, %d4g1v: tensor<512xf32>, %d4bt1v: tensor<512xf32>, %d4W2v: tensor<512x512x3x3xf32>, %d4g2v: tensor<512xf32>, %d4bt2v: tensor<512xf32>, %d4Wpv: tensor<512x256x1x1xf32>, %d4gpv: tensor<512xf32>, %d4btpv: tensor<512xf32>, %s4b0W1v: tensor<512x512x3x3xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b1W1v: tensor<512x512x3x3xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %Wdv: tensor<512x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %d2n1mui: tensor<128xf32>, %d2n1vari: tensor<128xf32>, %d2n2mui: tensor<128xf32>, %d2n2vari: tensor<128xf32>, %d2npmui: tensor<128xf32>, %d2npvari: tensor<128xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %d3n1mui: tensor<256xf32>, %d3n1vari: tensor<256xf32>, %d3n2mui: tensor<256xf32>, %d3n2vari: tensor<256xf32>, %d3npmui: tensor<256xf32>, %d3npvari: tensor<256xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %d4n1mui: tensor<512xf32>, %d4n1vari: tensor<512xf32>, %d4n2mui: tensor<512xf32>, %d4n2vari: tensor<512xf32>, %d4npmui: tensor<512xf32>, %d4npvari: tensor<512xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %onehot: tensor<32x10xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) {
    // ── ResNet-34 batch-BN heavy-ball momentum + coupled L2 train step: every op is pretty(verified AST node) except the constants, the input passthroughs and the marked report-only %loss ──
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
    %v3585 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3586 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3587 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3588 = stablehlo.pad %v3583, %v3585, low = [0, 0, 1, 1], high = [0, 0, 1, 1], interior = [0, 0, 0, 0] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3589 = stablehlo.slice %v3588 [0:32, 0:64, 0:112:2, 0:112:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3590 = stablehlo.slice %v3588 [0:32, 0:64, 0:112:2, 1:113:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3591 = stablehlo.slice %v3588 [0:32, 0:64, 0:112:2, 2:114:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3592 = stablehlo.slice %v3588 [0:32, 0:64, 1:113:2, 0:112:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3593 = stablehlo.slice %v3588 [0:32, 0:64, 1:113:2, 1:113:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3594 = stablehlo.slice %v3588 [0:32, 0:64, 1:113:2, 2:114:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3595 = stablehlo.slice %v3588 [0:32, 0:64, 2:114:2, 0:112:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3596 = stablehlo.slice %v3588 [0:32, 0:64, 2:114:2, 1:113:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3597 = stablehlo.slice %v3588 [0:32, 0:64, 2:114:2, 2:114:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3598 = stablehlo.maximum %v3589, %v3590 : tensor<32x64x56x56xf32>
    %v3599 = stablehlo.maximum %v3598, %v3591 : tensor<32x64x56x56xf32>
    %v3600 = stablehlo.maximum %v3599, %v3592 : tensor<32x64x56x56xf32>
    %v3601 = stablehlo.maximum %v3600, %v3593 : tensor<32x64x56x56xf32>
    %v3602 = stablehlo.maximum %v3601, %v3594 : tensor<32x64x56x56xf32>
    %v3603 = stablehlo.maximum %v3602, %v3595 : tensor<32x64x56x56xf32>
    %v3604 = stablehlo.maximum %v3603, %v3596 : tensor<32x64x56x56xf32>
    %v3605 = stablehlo.maximum %v3604, %v3597 : tensor<32x64x56x56xf32>
    %v3606 = stablehlo.compare EQ, %v3589, %v3605 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3607 = stablehlo.select %v3606, %v3584, %v3587 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3608 = stablehlo.pad %v3607, %v3586, low = [0, 0, 0, 0], high = [0, 0, 3, 3], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3609 = stablehlo.compare EQ, %v3590, %v3605 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3610 = stablehlo.not %v3606 : tensor<32x64x56x56xi1>
    %v3611 = stablehlo.and %v3609, %v3610 : tensor<32x64x56x56xi1>
    %v3612 = stablehlo.or %v3606, %v3609 : tensor<32x64x56x56xi1>
    %v3613 = stablehlo.select %v3611, %v3584, %v3587 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3614 = stablehlo.pad %v3613, %v3586, low = [0, 0, 0, 1], high = [0, 0, 3, 2], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3615 = stablehlo.add %v3608, %v3614 : tensor<32x64x114x114xf32>
    %v3616 = stablehlo.compare EQ, %v3591, %v3605 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3617 = stablehlo.not %v3612 : tensor<32x64x56x56xi1>
    %v3618 = stablehlo.and %v3616, %v3617 : tensor<32x64x56x56xi1>
    %v3619 = stablehlo.or %v3612, %v3616 : tensor<32x64x56x56xi1>
    %v3620 = stablehlo.select %v3618, %v3584, %v3587 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3621 = stablehlo.pad %v3620, %v3586, low = [0, 0, 0, 2], high = [0, 0, 3, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3622 = stablehlo.add %v3615, %v3621 : tensor<32x64x114x114xf32>
    %v3623 = stablehlo.compare EQ, %v3592, %v3605 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3624 = stablehlo.not %v3619 : tensor<32x64x56x56xi1>
    %v3625 = stablehlo.and %v3623, %v3624 : tensor<32x64x56x56xi1>
    %v3626 = stablehlo.or %v3619, %v3623 : tensor<32x64x56x56xi1>
    %v3627 = stablehlo.select %v3625, %v3584, %v3587 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3628 = stablehlo.pad %v3627, %v3586, low = [0, 0, 1, 0], high = [0, 0, 2, 3], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3629 = stablehlo.add %v3622, %v3628 : tensor<32x64x114x114xf32>
    %v3630 = stablehlo.compare EQ, %v3593, %v3605 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3631 = stablehlo.not %v3626 : tensor<32x64x56x56xi1>
    %v3632 = stablehlo.and %v3630, %v3631 : tensor<32x64x56x56xi1>
    %v3633 = stablehlo.or %v3626, %v3630 : tensor<32x64x56x56xi1>
    %v3634 = stablehlo.select %v3632, %v3584, %v3587 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3635 = stablehlo.pad %v3634, %v3586, low = [0, 0, 1, 1], high = [0, 0, 2, 2], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3636 = stablehlo.add %v3629, %v3635 : tensor<32x64x114x114xf32>
    %v3637 = stablehlo.compare EQ, %v3594, %v3605 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3638 = stablehlo.not %v3633 : tensor<32x64x56x56xi1>
    %v3639 = stablehlo.and %v3637, %v3638 : tensor<32x64x56x56xi1>
    %v3640 = stablehlo.or %v3633, %v3637 : tensor<32x64x56x56xi1>
    %v3641 = stablehlo.select %v3639, %v3584, %v3587 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3642 = stablehlo.pad %v3641, %v3586, low = [0, 0, 1, 2], high = [0, 0, 2, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3643 = stablehlo.add %v3636, %v3642 : tensor<32x64x114x114xf32>
    %v3644 = stablehlo.compare EQ, %v3595, %v3605 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3645 = stablehlo.not %v3640 : tensor<32x64x56x56xi1>
    %v3646 = stablehlo.and %v3644, %v3645 : tensor<32x64x56x56xi1>
    %v3647 = stablehlo.or %v3640, %v3644 : tensor<32x64x56x56xi1>
    %v3648 = stablehlo.select %v3646, %v3584, %v3587 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3649 = stablehlo.pad %v3648, %v3586, low = [0, 0, 2, 0], high = [0, 0, 1, 3], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3650 = stablehlo.add %v3643, %v3649 : tensor<32x64x114x114xf32>
    %v3651 = stablehlo.compare EQ, %v3596, %v3605 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3652 = stablehlo.not %v3647 : tensor<32x64x56x56xi1>
    %v3653 = stablehlo.and %v3651, %v3652 : tensor<32x64x56x56xi1>
    %v3654 = stablehlo.or %v3647, %v3651 : tensor<32x64x56x56xi1>
    %v3655 = stablehlo.select %v3653, %v3584, %v3587 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3656 = stablehlo.pad %v3655, %v3586, low = [0, 0, 2, 1], high = [0, 0, 1, 2], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3657 = stablehlo.add %v3650, %v3656 : tensor<32x64x114x114xf32>
    %v3658 = stablehlo.compare EQ, %v3597, %v3605 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3659 = stablehlo.not %v3654 : tensor<32x64x56x56xi1>
    %v3660 = stablehlo.and %v3658, %v3659 : tensor<32x64x56x56xi1>
    %v3661 = stablehlo.or %v3654, %v3658 : tensor<32x64x56x56xi1>
    %v3662 = stablehlo.select %v3660, %v3584, %v3587 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3663 = stablehlo.pad %v3662, %v3586, low = [0, 0, 2, 2], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3664 = stablehlo.add %v3657, %v3663 : tensor<32x64x114x114xf32>
    %v3665 = stablehlo.slice %v3664 [0:32, 0:64, 1:113, 1:113] : (tensor<32x64x114x114xf32>) -> tensor<32x64x112x112xf32>
    %v3666 = stablehlo.reshape %v3665 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3667 = stablehlo.reshape %v3666 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3668 = stablehlo.reshape %v24 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3669 = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %v3670 = stablehlo.compare GT, %v3668, %v3669 : (tensor<32x64x112x112xf32>, tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xi1>
    %v3671 = stablehlo.select %v3670, %v3667, %v3669 : tensor<32x64x112x112xi1>, tensor<32x64x112x112xf32>
    %v3672 = stablehlo.reshape %v3671 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3673 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3674 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3675 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3676 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3677 = stablehlo.reduce(%v3673 init: %v3674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3678 = stablehlo.broadcast_in_dim %v3677, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3679 = stablehlo.divide %v3678, %v3675 : tensor<32x64x112x112xf32>
    %v3680 = stablehlo.subtract %v3673, %v3679 : tensor<32x64x112x112xf32>
    %v3681 = stablehlo.multiply %v3680, %v3680 : tensor<32x64x112x112xf32>
    %v3682 = stablehlo.reduce(%v3681 init: %v3674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3683 = stablehlo.broadcast_in_dim %v3682, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3684 = stablehlo.divide %v3683, %v3675 : tensor<32x64x112x112xf32>
    %v3685 = stablehlo.add %v3684, %v3676 : tensor<32x64x112x112xf32>
    %v3686 = stablehlo.rsqrt %v3685 : tensor<32x64x112x112xf32>
    %v3687 = stablehlo.multiply %v3680, %v3686 : tensor<32x64x112x112xf32>
    %v3688 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3689 = stablehlo.reshape %v3672 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3690 = stablehlo.multiply %v3688, %v3689 : tensor<32x64x112x112xf32>
    %v3691 = stablehlo.reduce(%v3690 init: %v3674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3692 = stablehlo.broadcast_in_dim %v3691, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3693 = stablehlo.multiply %v3687, %v3690 : tensor<32x64x112x112xf32>
    %v3694 = stablehlo.reduce(%v3693 init: %v3674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3695 = stablehlo.broadcast_in_dim %v3694, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3696 = stablehlo.multiply %v3690, %v3675 : tensor<32x64x112x112xf32>
    %v3697 = stablehlo.subtract %v3696, %v3692 : tensor<32x64x112x112xf32>
    %v3698 = stablehlo.multiply %v3687, %v3695 : tensor<32x64x112x112xf32>
    %v3699 = stablehlo.subtract %v3697, %v3698 : tensor<32x64x112x112xf32>
    %v3700 = stablehlo.divide %v3686, %v3675 : tensor<32x64x112x112xf32>
    %v3701 = stablehlo.multiply %v3700, %v3699 : tensor<32x64x112x112xf32>
    %v3702 = stablehlo.reshape %v3701 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3703 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v3704 = stablehlo.reshape %v3702 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3705 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3706 = stablehlo.pad %v3704, %v3705, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x224x224xf32>
    %v3707 = stablehlo.transpose %v3703, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v3708 = stablehlo.transpose %v3706, dims = [1, 0, 2, 3] : (tensor<32x64x224x224xf32>) -> tensor<64x32x224x224xf32>
    %v3709 = stablehlo.convolution(%v3707, %v3708)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<64x32x224x224xf32>) -> tensor<3x64x7x7xf32>
    %v3710 = stablehlo.transpose %v3709, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v3711 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3712 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3713 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3714 = stablehlo.reduce(%v3711 init: %v3712) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3715 = stablehlo.broadcast_in_dim %v3714, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3716 = stablehlo.divide %v3715, %v3713 : tensor<32x64x112x112xf32>
    %v3717 = stablehlo.subtract %v3711, %v3716 : tensor<32x64x112x112xf32>
    %v3718 = stablehlo.multiply %v3717, %v3717 : tensor<32x64x112x112xf32>
    %v3719 = stablehlo.reduce(%v3718 init: %v3712) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3720 = stablehlo.broadcast_in_dim %v3719, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3721 = stablehlo.divide %v3720, %v3713 : tensor<32x64x112x112xf32>
    %v3722 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3723 = stablehlo.add %v3721, %v3722 : tensor<32x64x112x112xf32>
    %v3724 = stablehlo.rsqrt %v3723 : tensor<32x64x112x112xf32>
    %v3725 = stablehlo.multiply %v3717, %v3724 : tensor<32x64x112x112xf32>
    %v3726 = stablehlo.reshape %v3672 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3727 = stablehlo.multiply %v3726, %v3725 : tensor<32x64x112x112xf32>
    %v3728 = stablehlo.reduce(%v3727 init: %v3712) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3729 = stablehlo.reshape %v3672 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3731 = stablehlo.reduce(%v3729 init: %v3730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3732 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3733 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3734 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3735 = stablehlo.reduce(%v3732 init: %v3733) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3736 = stablehlo.divide %v3735, %v3734 : tensor<64xf32>
    %v3737 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3738 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3739 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3740 = stablehlo.reduce(%v3737 init: %v3738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3741 = stablehlo.broadcast_in_dim %v3740, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3742 = stablehlo.divide %v3741, %v3739 : tensor<32x64x112x112xf32>
    %v3743 = stablehlo.subtract %v3737, %v3742 : tensor<32x64x112x112xf32>
    %v3744 = stablehlo.multiply %v3743, %v3743 : tensor<32x64x112x112xf32>
    %v3745 = stablehlo.reduce(%v3744 init: %v3738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3746 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3747 = stablehlo.divide %v3745, %v3746 : tensor<64xf32>
    %v3748 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3750 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3751 = stablehlo.reduce(%v3748 init: %v3749) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3752 = stablehlo.divide %v3751, %v3750 : tensor<64xf32>
    %v3753 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3754 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3755 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3756 = stablehlo.reduce(%v3753 init: %v3754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3757 = stablehlo.broadcast_in_dim %v3756, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3758 = stablehlo.divide %v3757, %v3755 : tensor<32x64x56x56xf32>
    %v3759 = stablehlo.subtract %v3753, %v3758 : tensor<32x64x56x56xf32>
    %v3760 = stablehlo.multiply %v3759, %v3759 : tensor<32x64x56x56xf32>
    %v3761 = stablehlo.reduce(%v3760 init: %v3754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3762 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3763 = stablehlo.divide %v3761, %v3762 : tensor<64xf32>
    %v3764 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3765 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3766 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3767 = stablehlo.reduce(%v3764 init: %v3765) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3768 = stablehlo.divide %v3767, %v3766 : tensor<64xf32>
    %v3769 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3770 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3771 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3772 = stablehlo.reduce(%v3769 init: %v3770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3773 = stablehlo.broadcast_in_dim %v3772, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3774 = stablehlo.divide %v3773, %v3771 : tensor<32x64x56x56xf32>
    %v3775 = stablehlo.subtract %v3769, %v3774 : tensor<32x64x56x56xf32>
    %v3776 = stablehlo.multiply %v3775, %v3775 : tensor<32x64x56x56xf32>
    %v3777 = stablehlo.reduce(%v3776 init: %v3770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3778 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3779 = stablehlo.divide %v3777, %v3778 : tensor<64xf32>
    %v3780 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3782 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3783 = stablehlo.reduce(%v3780 init: %v3781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3784 = stablehlo.divide %v3783, %v3782 : tensor<64xf32>
    %v3785 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3787 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3788 = stablehlo.reduce(%v3785 init: %v3786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3789 = stablehlo.broadcast_in_dim %v3788, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3790 = stablehlo.divide %v3789, %v3787 : tensor<32x64x56x56xf32>
    %v3791 = stablehlo.subtract %v3785, %v3790 : tensor<32x64x56x56xf32>
    %v3792 = stablehlo.multiply %v3791, %v3791 : tensor<32x64x56x56xf32>
    %v3793 = stablehlo.reduce(%v3792 init: %v3786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3794 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3795 = stablehlo.divide %v3793, %v3794 : tensor<64xf32>
    %v3796 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3798 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3799 = stablehlo.reduce(%v3796 init: %v3797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3800 = stablehlo.divide %v3799, %v3798 : tensor<64xf32>
    %v3801 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3803 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3804 = stablehlo.reduce(%v3801 init: %v3802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3805 = stablehlo.broadcast_in_dim %v3804, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3806 = stablehlo.divide %v3805, %v3803 : tensor<32x64x56x56xf32>
    %v3807 = stablehlo.subtract %v3801, %v3806 : tensor<32x64x56x56xf32>
    %v3808 = stablehlo.multiply %v3807, %v3807 : tensor<32x64x56x56xf32>
    %v3809 = stablehlo.reduce(%v3808 init: %v3802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3810 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3811 = stablehlo.divide %v3809, %v3810 : tensor<64xf32>
    %v3812 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3813 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3814 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3815 = stablehlo.reduce(%v3812 init: %v3813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3816 = stablehlo.divide %v3815, %v3814 : tensor<64xf32>
    %v3817 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3818 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3819 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3820 = stablehlo.reduce(%v3817 init: %v3818) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3821 = stablehlo.broadcast_in_dim %v3820, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3822 = stablehlo.divide %v3821, %v3819 : tensor<32x64x56x56xf32>
    %v3823 = stablehlo.subtract %v3817, %v3822 : tensor<32x64x56x56xf32>
    %v3824 = stablehlo.multiply %v3823, %v3823 : tensor<32x64x56x56xf32>
    %v3825 = stablehlo.reduce(%v3824 init: %v3818) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3826 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3827 = stablehlo.divide %v3825, %v3826 : tensor<64xf32>
    %v3828 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3830 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3831 = stablehlo.reduce(%v3828 init: %v3829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3832 = stablehlo.divide %v3831, %v3830 : tensor<64xf32>
    %v3833 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3834 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3835 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3836 = stablehlo.reduce(%v3833 init: %v3834) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3837 = stablehlo.broadcast_in_dim %v3836, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3838 = stablehlo.divide %v3837, %v3835 : tensor<32x64x56x56xf32>
    %v3839 = stablehlo.subtract %v3833, %v3838 : tensor<32x64x56x56xf32>
    %v3840 = stablehlo.multiply %v3839, %v3839 : tensor<32x64x56x56xf32>
    %v3841 = stablehlo.reduce(%v3840 init: %v3834) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3842 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3843 = stablehlo.divide %v3841, %v3842 : tensor<64xf32>
    %v3844 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3845 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3846 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3847 = stablehlo.reduce(%v3844 init: %v3845) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3848 = stablehlo.divide %v3847, %v3846 : tensor<128xf32>
    %v3849 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3851 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3852 = stablehlo.reduce(%v3849 init: %v3850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3853 = stablehlo.broadcast_in_dim %v3852, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3854 = stablehlo.divide %v3853, %v3851 : tensor<32x128x28x28xf32>
    %v3855 = stablehlo.subtract %v3849, %v3854 : tensor<32x128x28x28xf32>
    %v3856 = stablehlo.multiply %v3855, %v3855 : tensor<32x128x28x28xf32>
    %v3857 = stablehlo.reduce(%v3856 init: %v3850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3858 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3859 = stablehlo.divide %v3857, %v3858 : tensor<128xf32>
    %v3860 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3862 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3863 = stablehlo.reduce(%v3860 init: %v3861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3864 = stablehlo.divide %v3863, %v3862 : tensor<128xf32>
    %v3865 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3866 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3867 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3868 = stablehlo.reduce(%v3865 init: %v3866) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3869 = stablehlo.broadcast_in_dim %v3868, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3870 = stablehlo.divide %v3869, %v3867 : tensor<32x128x28x28xf32>
    %v3871 = stablehlo.subtract %v3865, %v3870 : tensor<32x128x28x28xf32>
    %v3872 = stablehlo.multiply %v3871, %v3871 : tensor<32x128x28x28xf32>
    %v3873 = stablehlo.reduce(%v3872 init: %v3866) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3874 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3875 = stablehlo.divide %v3873, %v3874 : tensor<128xf32>
    %v3876 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3877 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3878 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3879 = stablehlo.reduce(%v3876 init: %v3877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3880 = stablehlo.divide %v3879, %v3878 : tensor<128xf32>
    %v3881 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3883 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3884 = stablehlo.reduce(%v3881 init: %v3882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3885 = stablehlo.broadcast_in_dim %v3884, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3886 = stablehlo.divide %v3885, %v3883 : tensor<32x128x28x28xf32>
    %v3887 = stablehlo.subtract %v3881, %v3886 : tensor<32x128x28x28xf32>
    %v3888 = stablehlo.multiply %v3887, %v3887 : tensor<32x128x28x28xf32>
    %v3889 = stablehlo.reduce(%v3888 init: %v3882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3890 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3891 = stablehlo.divide %v3889, %v3890 : tensor<128xf32>
    %v3892 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3894 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3895 = stablehlo.reduce(%v3892 init: %v3893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3896 = stablehlo.divide %v3895, %v3894 : tensor<128xf32>
    %v3897 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3898 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3899 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3900 = stablehlo.reduce(%v3897 init: %v3898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3901 = stablehlo.broadcast_in_dim %v3900, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3902 = stablehlo.divide %v3901, %v3899 : tensor<32x128x28x28xf32>
    %v3903 = stablehlo.subtract %v3897, %v3902 : tensor<32x128x28x28xf32>
    %v3904 = stablehlo.multiply %v3903, %v3903 : tensor<32x128x28x28xf32>
    %v3905 = stablehlo.reduce(%v3904 init: %v3898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3906 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3907 = stablehlo.divide %v3905, %v3906 : tensor<128xf32>
    %v3908 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3910 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3911 = stablehlo.reduce(%v3908 init: %v3909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3912 = stablehlo.divide %v3911, %v3910 : tensor<128xf32>
    %v3913 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3914 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3915 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3916 = stablehlo.reduce(%v3913 init: %v3914) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3917 = stablehlo.broadcast_in_dim %v3916, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3918 = stablehlo.divide %v3917, %v3915 : tensor<32x128x28x28xf32>
    %v3919 = stablehlo.subtract %v3913, %v3918 : tensor<32x128x28x28xf32>
    %v3920 = stablehlo.multiply %v3919, %v3919 : tensor<32x128x28x28xf32>
    %v3921 = stablehlo.reduce(%v3920 init: %v3914) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3922 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3923 = stablehlo.divide %v3921, %v3922 : tensor<128xf32>
    %v3924 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3925 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3926 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3927 = stablehlo.reduce(%v3924 init: %v3925) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3928 = stablehlo.divide %v3927, %v3926 : tensor<128xf32>
    %v3929 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3931 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3932 = stablehlo.reduce(%v3929 init: %v3930) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3933 = stablehlo.broadcast_in_dim %v3932, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3934 = stablehlo.divide %v3933, %v3931 : tensor<32x128x28x28xf32>
    %v3935 = stablehlo.subtract %v3929, %v3934 : tensor<32x128x28x28xf32>
    %v3936 = stablehlo.multiply %v3935, %v3935 : tensor<32x128x28x28xf32>
    %v3937 = stablehlo.reduce(%v3936 init: %v3930) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3938 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3939 = stablehlo.divide %v3937, %v3938 : tensor<128xf32>
    %v3940 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3941 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3942 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3943 = stablehlo.reduce(%v3940 init: %v3941) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3944 = stablehlo.divide %v3943, %v3942 : tensor<128xf32>
    %v3945 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3946 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3947 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3948 = stablehlo.reduce(%v3945 init: %v3946) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3949 = stablehlo.broadcast_in_dim %v3948, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3950 = stablehlo.divide %v3949, %v3947 : tensor<32x128x28x28xf32>
    %v3951 = stablehlo.subtract %v3945, %v3950 : tensor<32x128x28x28xf32>
    %v3952 = stablehlo.multiply %v3951, %v3951 : tensor<32x128x28x28xf32>
    %v3953 = stablehlo.reduce(%v3952 init: %v3946) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3954 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3955 = stablehlo.divide %v3953, %v3954 : tensor<128xf32>
    %v3956 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3957 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3958 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3959 = stablehlo.reduce(%v3956 init: %v3957) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3960 = stablehlo.divide %v3959, %v3958 : tensor<128xf32>
    %v3961 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3963 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3964 = stablehlo.reduce(%v3961 init: %v3962) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3965 = stablehlo.broadcast_in_dim %v3964, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3966 = stablehlo.divide %v3965, %v3963 : tensor<32x128x28x28xf32>
    %v3967 = stablehlo.subtract %v3961, %v3966 : tensor<32x128x28x28xf32>
    %v3968 = stablehlo.multiply %v3967, %v3967 : tensor<32x128x28x28xf32>
    %v3969 = stablehlo.reduce(%v3968 init: %v3962) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3970 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3971 = stablehlo.divide %v3969, %v3970 : tensor<128xf32>
    %v3972 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3973 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3974 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3975 = stablehlo.reduce(%v3972 init: %v3973) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3976 = stablehlo.divide %v3975, %v3974 : tensor<128xf32>
    %v3977 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3978 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3979 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3980 = stablehlo.reduce(%v3977 init: %v3978) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3981 = stablehlo.broadcast_in_dim %v3980, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3982 = stablehlo.divide %v3981, %v3979 : tensor<32x128x28x28xf32>
    %v3983 = stablehlo.subtract %v3977, %v3982 : tensor<32x128x28x28xf32>
    %v3984 = stablehlo.multiply %v3983, %v3983 : tensor<32x128x28x28xf32>
    %v3985 = stablehlo.reduce(%v3984 init: %v3978) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3986 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3987 = stablehlo.divide %v3985, %v3986 : tensor<128xf32>
    %v3988 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3989 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3990 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3991 = stablehlo.reduce(%v3988 init: %v3989) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3992 = stablehlo.divide %v3991, %v3990 : tensor<256xf32>
    %v3993 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3994 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3995 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3996 = stablehlo.reduce(%v3993 init: %v3994) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3997 = stablehlo.broadcast_in_dim %v3996, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3998 = stablehlo.divide %v3997, %v3995 : tensor<32x256x14x14xf32>
    %v3999 = stablehlo.subtract %v3993, %v3998 : tensor<32x256x14x14xf32>
    %v4000 = stablehlo.multiply %v3999, %v3999 : tensor<32x256x14x14xf32>
    %v4001 = stablehlo.reduce(%v4000 init: %v3994) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4002 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4003 = stablehlo.divide %v4001, %v4002 : tensor<256xf32>
    %v4004 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4005 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4006 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4007 = stablehlo.reduce(%v4004 init: %v4005) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4008 = stablehlo.divide %v4007, %v4006 : tensor<256xf32>
    %v4009 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4010 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4011 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4012 = stablehlo.reduce(%v4009 init: %v4010) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4013 = stablehlo.broadcast_in_dim %v4012, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4014 = stablehlo.divide %v4013, %v4011 : tensor<32x256x14x14xf32>
    %v4015 = stablehlo.subtract %v4009, %v4014 : tensor<32x256x14x14xf32>
    %v4016 = stablehlo.multiply %v4015, %v4015 : tensor<32x256x14x14xf32>
    %v4017 = stablehlo.reduce(%v4016 init: %v4010) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4018 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4019 = stablehlo.divide %v4017, %v4018 : tensor<256xf32>
    %v4020 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4021 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4022 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4023 = stablehlo.reduce(%v4020 init: %v4021) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4024 = stablehlo.divide %v4023, %v4022 : tensor<256xf32>
    %v4025 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4026 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4027 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4028 = stablehlo.reduce(%v4025 init: %v4026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4029 = stablehlo.broadcast_in_dim %v4028, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4030 = stablehlo.divide %v4029, %v4027 : tensor<32x256x14x14xf32>
    %v4031 = stablehlo.subtract %v4025, %v4030 : tensor<32x256x14x14xf32>
    %v4032 = stablehlo.multiply %v4031, %v4031 : tensor<32x256x14x14xf32>
    %v4033 = stablehlo.reduce(%v4032 init: %v4026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4034 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4035 = stablehlo.divide %v4033, %v4034 : tensor<256xf32>
    %v4036 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4037 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4038 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4039 = stablehlo.reduce(%v4036 init: %v4037) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4040 = stablehlo.divide %v4039, %v4038 : tensor<256xf32>
    %v4041 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4042 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4043 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4044 = stablehlo.reduce(%v4041 init: %v4042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4045 = stablehlo.broadcast_in_dim %v4044, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4046 = stablehlo.divide %v4045, %v4043 : tensor<32x256x14x14xf32>
    %v4047 = stablehlo.subtract %v4041, %v4046 : tensor<32x256x14x14xf32>
    %v4048 = stablehlo.multiply %v4047, %v4047 : tensor<32x256x14x14xf32>
    %v4049 = stablehlo.reduce(%v4048 init: %v4042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4050 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4051 = stablehlo.divide %v4049, %v4050 : tensor<256xf32>
    %v4052 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4053 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4054 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4055 = stablehlo.reduce(%v4052 init: %v4053) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4056 = stablehlo.divide %v4055, %v4054 : tensor<256xf32>
    %v4057 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4058 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4059 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4060 = stablehlo.reduce(%v4057 init: %v4058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4061 = stablehlo.broadcast_in_dim %v4060, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4062 = stablehlo.divide %v4061, %v4059 : tensor<32x256x14x14xf32>
    %v4063 = stablehlo.subtract %v4057, %v4062 : tensor<32x256x14x14xf32>
    %v4064 = stablehlo.multiply %v4063, %v4063 : tensor<32x256x14x14xf32>
    %v4065 = stablehlo.reduce(%v4064 init: %v4058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4066 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4067 = stablehlo.divide %v4065, %v4066 : tensor<256xf32>
    %v4068 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4069 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4070 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4071 = stablehlo.reduce(%v4068 init: %v4069) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4072 = stablehlo.divide %v4071, %v4070 : tensor<256xf32>
    %v4073 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4074 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4075 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4076 = stablehlo.reduce(%v4073 init: %v4074) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4077 = stablehlo.broadcast_in_dim %v4076, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4078 = stablehlo.divide %v4077, %v4075 : tensor<32x256x14x14xf32>
    %v4079 = stablehlo.subtract %v4073, %v4078 : tensor<32x256x14x14xf32>
    %v4080 = stablehlo.multiply %v4079, %v4079 : tensor<32x256x14x14xf32>
    %v4081 = stablehlo.reduce(%v4080 init: %v4074) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4082 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4083 = stablehlo.divide %v4081, %v4082 : tensor<256xf32>
    %v4084 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4086 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4087 = stablehlo.reduce(%v4084 init: %v4085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4088 = stablehlo.divide %v4087, %v4086 : tensor<256xf32>
    %v4089 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4090 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4091 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4092 = stablehlo.reduce(%v4089 init: %v4090) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4093 = stablehlo.broadcast_in_dim %v4092, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4094 = stablehlo.divide %v4093, %v4091 : tensor<32x256x14x14xf32>
    %v4095 = stablehlo.subtract %v4089, %v4094 : tensor<32x256x14x14xf32>
    %v4096 = stablehlo.multiply %v4095, %v4095 : tensor<32x256x14x14xf32>
    %v4097 = stablehlo.reduce(%v4096 init: %v4090) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4098 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4099 = stablehlo.divide %v4097, %v4098 : tensor<256xf32>
    %v4100 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4102 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4103 = stablehlo.reduce(%v4100 init: %v4101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4104 = stablehlo.divide %v4103, %v4102 : tensor<256xf32>
    %v4105 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4107 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4108 = stablehlo.reduce(%v4105 init: %v4106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4109 = stablehlo.broadcast_in_dim %v4108, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4110 = stablehlo.divide %v4109, %v4107 : tensor<32x256x14x14xf32>
    %v4111 = stablehlo.subtract %v4105, %v4110 : tensor<32x256x14x14xf32>
    %v4112 = stablehlo.multiply %v4111, %v4111 : tensor<32x256x14x14xf32>
    %v4113 = stablehlo.reduce(%v4112 init: %v4106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4114 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4115 = stablehlo.divide %v4113, %v4114 : tensor<256xf32>
    %v4116 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4118 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4119 = stablehlo.reduce(%v4116 init: %v4117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4120 = stablehlo.divide %v4119, %v4118 : tensor<256xf32>
    %v4121 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4123 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4124 = stablehlo.reduce(%v4121 init: %v4122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4125 = stablehlo.broadcast_in_dim %v4124, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4126 = stablehlo.divide %v4125, %v4123 : tensor<32x256x14x14xf32>
    %v4127 = stablehlo.subtract %v4121, %v4126 : tensor<32x256x14x14xf32>
    %v4128 = stablehlo.multiply %v4127, %v4127 : tensor<32x256x14x14xf32>
    %v4129 = stablehlo.reduce(%v4128 init: %v4122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4130 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4131 = stablehlo.divide %v4129, %v4130 : tensor<256xf32>
    %v4132 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4134 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4135 = stablehlo.reduce(%v4132 init: %v4133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4136 = stablehlo.divide %v4135, %v4134 : tensor<256xf32>
    %v4137 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4139 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4140 = stablehlo.reduce(%v4137 init: %v4138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4141 = stablehlo.broadcast_in_dim %v4140, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4142 = stablehlo.divide %v4141, %v4139 : tensor<32x256x14x14xf32>
    %v4143 = stablehlo.subtract %v4137, %v4142 : tensor<32x256x14x14xf32>
    %v4144 = stablehlo.multiply %v4143, %v4143 : tensor<32x256x14x14xf32>
    %v4145 = stablehlo.reduce(%v4144 init: %v4138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4146 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4147 = stablehlo.divide %v4145, %v4146 : tensor<256xf32>
    %v4148 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4149 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4150 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4151 = stablehlo.reduce(%v4148 init: %v4149) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4152 = stablehlo.divide %v4151, %v4150 : tensor<256xf32>
    %v4153 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4155 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4156 = stablehlo.reduce(%v4153 init: %v4154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4157 = stablehlo.broadcast_in_dim %v4156, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4158 = stablehlo.divide %v4157, %v4155 : tensor<32x256x14x14xf32>
    %v4159 = stablehlo.subtract %v4153, %v4158 : tensor<32x256x14x14xf32>
    %v4160 = stablehlo.multiply %v4159, %v4159 : tensor<32x256x14x14xf32>
    %v4161 = stablehlo.reduce(%v4160 init: %v4154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4162 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4163 = stablehlo.divide %v4161, %v4162 : tensor<256xf32>
    %v4164 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4166 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4167 = stablehlo.reduce(%v4164 init: %v4165) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4168 = stablehlo.divide %v4167, %v4166 : tensor<256xf32>
    %v4169 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4171 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4172 = stablehlo.reduce(%v4169 init: %v4170) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4173 = stablehlo.broadcast_in_dim %v4172, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4174 = stablehlo.divide %v4173, %v4171 : tensor<32x256x14x14xf32>
    %v4175 = stablehlo.subtract %v4169, %v4174 : tensor<32x256x14x14xf32>
    %v4176 = stablehlo.multiply %v4175, %v4175 : tensor<32x256x14x14xf32>
    %v4177 = stablehlo.reduce(%v4176 init: %v4170) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4178 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4179 = stablehlo.divide %v4177, %v4178 : tensor<256xf32>
    %v4180 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4182 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4183 = stablehlo.reduce(%v4180 init: %v4181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4184 = stablehlo.divide %v4183, %v4182 : tensor<256xf32>
    %v4185 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4187 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4188 = stablehlo.reduce(%v4185 init: %v4186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4189 = stablehlo.broadcast_in_dim %v4188, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4190 = stablehlo.divide %v4189, %v4187 : tensor<32x256x14x14xf32>
    %v4191 = stablehlo.subtract %v4185, %v4190 : tensor<32x256x14x14xf32>
    %v4192 = stablehlo.multiply %v4191, %v4191 : tensor<32x256x14x14xf32>
    %v4193 = stablehlo.reduce(%v4192 init: %v4186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4194 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4195 = stablehlo.divide %v4193, %v4194 : tensor<256xf32>
    %v4196 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4198 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4199 = stablehlo.reduce(%v4196 init: %v4197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4200 = stablehlo.divide %v4199, %v4198 : tensor<512xf32>
    %v4201 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4203 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4204 = stablehlo.reduce(%v4201 init: %v4202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4205 = stablehlo.broadcast_in_dim %v4204, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4206 = stablehlo.divide %v4205, %v4203 : tensor<32x512x7x7xf32>
    %v4207 = stablehlo.subtract %v4201, %v4206 : tensor<32x512x7x7xf32>
    %v4208 = stablehlo.multiply %v4207, %v4207 : tensor<32x512x7x7xf32>
    %v4209 = stablehlo.reduce(%v4208 init: %v4202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4210 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4211 = stablehlo.divide %v4209, %v4210 : tensor<512xf32>
    %v4212 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4213 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4214 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4215 = stablehlo.reduce(%v4212 init: %v4213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4216 = stablehlo.divide %v4215, %v4214 : tensor<512xf32>
    %v4217 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4219 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4220 = stablehlo.reduce(%v4217 init: %v4218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4221 = stablehlo.broadcast_in_dim %v4220, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4222 = stablehlo.divide %v4221, %v4219 : tensor<32x512x7x7xf32>
    %v4223 = stablehlo.subtract %v4217, %v4222 : tensor<32x512x7x7xf32>
    %v4224 = stablehlo.multiply %v4223, %v4223 : tensor<32x512x7x7xf32>
    %v4225 = stablehlo.reduce(%v4224 init: %v4218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4226 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4227 = stablehlo.divide %v4225, %v4226 : tensor<512xf32>
    %v4228 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4230 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4231 = stablehlo.reduce(%v4228 init: %v4229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4232 = stablehlo.divide %v4231, %v4230 : tensor<512xf32>
    %v4233 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4234 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4235 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4236 = stablehlo.reduce(%v4233 init: %v4234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4237 = stablehlo.broadcast_in_dim %v4236, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4238 = stablehlo.divide %v4237, %v4235 : tensor<32x512x7x7xf32>
    %v4239 = stablehlo.subtract %v4233, %v4238 : tensor<32x512x7x7xf32>
    %v4240 = stablehlo.multiply %v4239, %v4239 : tensor<32x512x7x7xf32>
    %v4241 = stablehlo.reduce(%v4240 init: %v4234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4242 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4243 = stablehlo.divide %v4241, %v4242 : tensor<512xf32>
    %v4244 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4246 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4247 = stablehlo.reduce(%v4244 init: %v4245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4248 = stablehlo.divide %v4247, %v4246 : tensor<512xf32>
    %v4249 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4250 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4251 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4252 = stablehlo.reduce(%v4249 init: %v4250) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4253 = stablehlo.broadcast_in_dim %v4252, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4254 = stablehlo.divide %v4253, %v4251 : tensor<32x512x7x7xf32>
    %v4255 = stablehlo.subtract %v4249, %v4254 : tensor<32x512x7x7xf32>
    %v4256 = stablehlo.multiply %v4255, %v4255 : tensor<32x512x7x7xf32>
    %v4257 = stablehlo.reduce(%v4256 init: %v4250) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4258 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4259 = stablehlo.divide %v4257, %v4258 : tensor<512xf32>
    %v4260 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4262 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4263 = stablehlo.reduce(%v4260 init: %v4261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4264 = stablehlo.divide %v4263, %v4262 : tensor<512xf32>
    %v4265 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4266 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4267 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4268 = stablehlo.reduce(%v4265 init: %v4266) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4269 = stablehlo.broadcast_in_dim %v4268, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4270 = stablehlo.divide %v4269, %v4267 : tensor<32x512x7x7xf32>
    %v4271 = stablehlo.subtract %v4265, %v4270 : tensor<32x512x7x7xf32>
    %v4272 = stablehlo.multiply %v4271, %v4271 : tensor<32x512x7x7xf32>
    %v4273 = stablehlo.reduce(%v4272 init: %v4266) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4274 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4275 = stablehlo.divide %v4273, %v4274 : tensor<512xf32>
    %v4276 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4277 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4278 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4279 = stablehlo.reduce(%v4276 init: %v4277) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4280 = stablehlo.divide %v4279, %v4278 : tensor<512xf32>
    %v4281 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4282 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4283 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4284 = stablehlo.reduce(%v4281 init: %v4282) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4285 = stablehlo.broadcast_in_dim %v4284, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4286 = stablehlo.divide %v4285, %v4283 : tensor<32x512x7x7xf32>
    %v4287 = stablehlo.subtract %v4281, %v4286 : tensor<32x512x7x7xf32>
    %v4288 = stablehlo.multiply %v4287, %v4287 : tensor<32x512x7x7xf32>
    %v4289 = stablehlo.reduce(%v4288 init: %v4282) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4290 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4291 = stablehlo.divide %v4289, %v4290 : tensor<512xf32>
    %v4292 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4294 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4295 = stablehlo.reduce(%v4292 init: %v4293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4296 = stablehlo.divide %v4295, %v4294 : tensor<512xf32>
    %v4297 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4299 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4300 = stablehlo.reduce(%v4297 init: %v4298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4301 = stablehlo.broadcast_in_dim %v4300, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4302 = stablehlo.divide %v4301, %v4299 : tensor<32x512x7x7xf32>
    %v4303 = stablehlo.subtract %v4297, %v4302 : tensor<32x512x7x7xf32>
    %v4304 = stablehlo.multiply %v4303, %v4303 : tensor<32x512x7x7xf32>
    %v4305 = stablehlo.reduce(%v4304 init: %v4298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4306 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4307 = stablehlo.divide %v4305, %v4306 : tensor<512xf32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0> : tensor<f32>
    %v4308 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4309 = stablehlo.multiply %v4308, %sW : tensor<64x3x7x7xf32>
    %v4310 = stablehlo.add %v4309, %v3710 : tensor<64x3x7x7xf32>
    %v4311 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4312 = stablehlo.multiply %v4311, %sWv : tensor<64x3x7x7xf32>
    %v4313 = stablehlo.add %v4312, %v4310 : tensor<64x3x7x7xf32>
    %v4314 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4315 = stablehlo.multiply %v4314, %v4313 : tensor<64x3x7x7xf32>
    %v4316 = stablehlo.subtract %sW, %v4315 : tensor<64x3x7x7xf32>
    %v4317 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4318 = stablehlo.multiply %v4317, %sg : tensor<64xf32>
    %v4319 = stablehlo.add %v4318, %v3728 : tensor<64xf32>
    %v4320 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4321 = stablehlo.multiply %v4320, %sgv : tensor<64xf32>
    %v4322 = stablehlo.add %v4321, %v4319 : tensor<64xf32>
    %v4323 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4324 = stablehlo.multiply %v4323, %v4322 : tensor<64xf32>
    %v4325 = stablehlo.subtract %sg, %v4324 : tensor<64xf32>
    %v4326 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4327 = stablehlo.multiply %v4326, %sbt : tensor<64xf32>
    %v4328 = stablehlo.add %v4327, %v3731 : tensor<64xf32>
    %v4329 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4330 = stablehlo.multiply %v4329, %sbtv : tensor<64xf32>
    %v4331 = stablehlo.add %v4330, %v4328 : tensor<64xf32>
    %v4332 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4333 = stablehlo.multiply %v4332, %v4331 : tensor<64xf32>
    %v4334 = stablehlo.subtract %sbt, %v4333 : tensor<64xf32>
    %v4335 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4336 = stablehlo.multiply %v4335, %s1b0W1 : tensor<64x64x3x3xf32>
    %v4337 = stablehlo.add %v4336, %v3534 : tensor<64x64x3x3xf32>
    %v4338 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4339 = stablehlo.multiply %v4338, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4340 = stablehlo.add %v4339, %v4337 : tensor<64x64x3x3xf32>
    %v4341 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4342 = stablehlo.multiply %v4341, %v4340 : tensor<64x64x3x3xf32>
    %v4343 = stablehlo.subtract %s1b0W1, %v4342 : tensor<64x64x3x3xf32>
    %v4344 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4345 = stablehlo.multiply %v4344, %s1b0g1 : tensor<64xf32>
    %v4346 = stablehlo.add %v4345, %v3552 : tensor<64xf32>
    %v4347 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4348 = stablehlo.multiply %v4347, %s1b0g1v : tensor<64xf32>
    %v4349 = stablehlo.add %v4348, %v4346 : tensor<64xf32>
    %v4350 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4351 = stablehlo.multiply %v4350, %v4349 : tensor<64xf32>
    %v4352 = stablehlo.subtract %s1b0g1, %v4351 : tensor<64xf32>
    %v4353 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4354 = stablehlo.multiply %v4353, %s1b0bt1 : tensor<64xf32>
    %v4355 = stablehlo.add %v4354, %v3555 : tensor<64xf32>
    %v4356 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4357 = stablehlo.multiply %v4356, %s1b0bt1v : tensor<64xf32>
    %v4358 = stablehlo.add %v4357, %v4355 : tensor<64xf32>
    %v4359 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4360 = stablehlo.multiply %v4359, %v4358 : tensor<64xf32>
    %v4361 = stablehlo.subtract %s1b0bt1, %v4360 : tensor<64xf32>
    %v4362 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4363 = stablehlo.multiply %v4362, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4364 = stablehlo.add %v4363, %v3561 : tensor<64x64x3x3xf32>
    %v4365 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4366 = stablehlo.multiply %v4365, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4367 = stablehlo.add %v4366, %v4364 : tensor<64x64x3x3xf32>
    %v4368 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4369 = stablehlo.multiply %v4368, %v4367 : tensor<64x64x3x3xf32>
    %v4370 = stablehlo.subtract %s1b0W2, %v4369 : tensor<64x64x3x3xf32>
    %v4371 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4372 = stablehlo.multiply %v4371, %s1b0g2 : tensor<64xf32>
    %v4373 = stablehlo.add %v4372, %v3579 : tensor<64xf32>
    %v4374 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4375 = stablehlo.multiply %v4374, %s1b0g2v : tensor<64xf32>
    %v4376 = stablehlo.add %v4375, %v4373 : tensor<64xf32>
    %v4377 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4378 = stablehlo.multiply %v4377, %v4376 : tensor<64xf32>
    %v4379 = stablehlo.subtract %s1b0g2, %v4378 : tensor<64xf32>
    %v4380 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4381 = stablehlo.multiply %v4380, %s1b0bt2 : tensor<64xf32>
    %v4382 = stablehlo.add %v4381, %v3582 : tensor<64xf32>
    %v4383 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4384 = stablehlo.multiply %v4383, %s1b0bt2v : tensor<64xf32>
    %v4385 = stablehlo.add %v4384, %v4382 : tensor<64xf32>
    %v4386 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4387 = stablehlo.multiply %v4386, %v4385 : tensor<64xf32>
    %v4388 = stablehlo.subtract %s1b0bt2, %v4387 : tensor<64xf32>
    %v4389 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4390 = stablehlo.multiply %v4389, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4391 = stablehlo.add %v4390, %v3394 : tensor<64x64x3x3xf32>
    %v4392 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4393 = stablehlo.multiply %v4392, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4394 = stablehlo.add %v4393, %v4391 : tensor<64x64x3x3xf32>
    %v4395 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4396 = stablehlo.multiply %v4395, %v4394 : tensor<64x64x3x3xf32>
    %v4397 = stablehlo.subtract %s1b1W1, %v4396 : tensor<64x64x3x3xf32>
    %v4398 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4399 = stablehlo.multiply %v4398, %s1b1g1 : tensor<64xf32>
    %v4400 = stablehlo.add %v4399, %v3412 : tensor<64xf32>
    %v4401 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4402 = stablehlo.multiply %v4401, %s1b1g1v : tensor<64xf32>
    %v4403 = stablehlo.add %v4402, %v4400 : tensor<64xf32>
    %v4404 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4405 = stablehlo.multiply %v4404, %v4403 : tensor<64xf32>
    %v4406 = stablehlo.subtract %s1b1g1, %v4405 : tensor<64xf32>
    %v4407 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4408 = stablehlo.multiply %v4407, %s1b1bt1 : tensor<64xf32>
    %v4409 = stablehlo.add %v4408, %v3415 : tensor<64xf32>
    %v4410 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4411 = stablehlo.multiply %v4410, %s1b1bt1v : tensor<64xf32>
    %v4412 = stablehlo.add %v4411, %v4409 : tensor<64xf32>
    %v4413 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4414 = stablehlo.multiply %v4413, %v4412 : tensor<64xf32>
    %v4415 = stablehlo.subtract %s1b1bt1, %v4414 : tensor<64xf32>
    %v4416 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4417 = stablehlo.multiply %v4416, %s1b1W2 : tensor<64x64x3x3xf32>
    %v4418 = stablehlo.add %v4417, %v3421 : tensor<64x64x3x3xf32>
    %v4419 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4420 = stablehlo.multiply %v4419, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4421 = stablehlo.add %v4420, %v4418 : tensor<64x64x3x3xf32>
    %v4422 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4423 = stablehlo.multiply %v4422, %v4421 : tensor<64x64x3x3xf32>
    %v4424 = stablehlo.subtract %s1b1W2, %v4423 : tensor<64x64x3x3xf32>
    %v4425 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4426 = stablehlo.multiply %v4425, %s1b1g2 : tensor<64xf32>
    %v4427 = stablehlo.add %v4426, %v3439 : tensor<64xf32>
    %v4428 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4429 = stablehlo.multiply %v4428, %s1b1g2v : tensor<64xf32>
    %v4430 = stablehlo.add %v4429, %v4427 : tensor<64xf32>
    %v4431 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4432 = stablehlo.multiply %v4431, %v4430 : tensor<64xf32>
    %v4433 = stablehlo.subtract %s1b1g2, %v4432 : tensor<64xf32>
    %v4434 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4435 = stablehlo.multiply %v4434, %s1b1bt2 : tensor<64xf32>
    %v4436 = stablehlo.add %v4435, %v3442 : tensor<64xf32>
    %v4437 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4438 = stablehlo.multiply %v4437, %s1b1bt2v : tensor<64xf32>
    %v4439 = stablehlo.add %v4438, %v4436 : tensor<64xf32>
    %v4440 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4441 = stablehlo.multiply %v4440, %v4439 : tensor<64xf32>
    %v4442 = stablehlo.subtract %s1b1bt2, %v4441 : tensor<64xf32>
    %v4443 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4444 = stablehlo.multiply %v4443, %s1b2W1 : tensor<64x64x3x3xf32>
    %v4445 = stablehlo.add %v4444, %v3254 : tensor<64x64x3x3xf32>
    %v4446 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4447 = stablehlo.multiply %v4446, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4448 = stablehlo.add %v4447, %v4445 : tensor<64x64x3x3xf32>
    %v4449 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4450 = stablehlo.multiply %v4449, %v4448 : tensor<64x64x3x3xf32>
    %v4451 = stablehlo.subtract %s1b2W1, %v4450 : tensor<64x64x3x3xf32>
    %v4452 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4453 = stablehlo.multiply %v4452, %s1b2g1 : tensor<64xf32>
    %v4454 = stablehlo.add %v4453, %v3272 : tensor<64xf32>
    %v4455 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4456 = stablehlo.multiply %v4455, %s1b2g1v : tensor<64xf32>
    %v4457 = stablehlo.add %v4456, %v4454 : tensor<64xf32>
    %v4458 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4459 = stablehlo.multiply %v4458, %v4457 : tensor<64xf32>
    %v4460 = stablehlo.subtract %s1b2g1, %v4459 : tensor<64xf32>
    %v4461 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4462 = stablehlo.multiply %v4461, %s1b2bt1 : tensor<64xf32>
    %v4463 = stablehlo.add %v4462, %v3275 : tensor<64xf32>
    %v4464 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4465 = stablehlo.multiply %v4464, %s1b2bt1v : tensor<64xf32>
    %v4466 = stablehlo.add %v4465, %v4463 : tensor<64xf32>
    %v4467 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4468 = stablehlo.multiply %v4467, %v4466 : tensor<64xf32>
    %v4469 = stablehlo.subtract %s1b2bt1, %v4468 : tensor<64xf32>
    %v4470 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4471 = stablehlo.multiply %v4470, %s1b2W2 : tensor<64x64x3x3xf32>
    %v4472 = stablehlo.add %v4471, %v3281 : tensor<64x64x3x3xf32>
    %v4473 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4474 = stablehlo.multiply %v4473, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4475 = stablehlo.add %v4474, %v4472 : tensor<64x64x3x3xf32>
    %v4476 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4477 = stablehlo.multiply %v4476, %v4475 : tensor<64x64x3x3xf32>
    %v4478 = stablehlo.subtract %s1b2W2, %v4477 : tensor<64x64x3x3xf32>
    %v4479 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4480 = stablehlo.multiply %v4479, %s1b2g2 : tensor<64xf32>
    %v4481 = stablehlo.add %v4480, %v3299 : tensor<64xf32>
    %v4482 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4483 = stablehlo.multiply %v4482, %s1b2g2v : tensor<64xf32>
    %v4484 = stablehlo.add %v4483, %v4481 : tensor<64xf32>
    %v4485 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4486 = stablehlo.multiply %v4485, %v4484 : tensor<64xf32>
    %v4487 = stablehlo.subtract %s1b2g2, %v4486 : tensor<64xf32>
    %v4488 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4489 = stablehlo.multiply %v4488, %s1b2bt2 : tensor<64xf32>
    %v4490 = stablehlo.add %v4489, %v3302 : tensor<64xf32>
    %v4491 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4492 = stablehlo.multiply %v4491, %s1b2bt2v : tensor<64xf32>
    %v4493 = stablehlo.add %v4492, %v4490 : tensor<64xf32>
    %v4494 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4495 = stablehlo.multiply %v4494, %v4493 : tensor<64xf32>
    %v4496 = stablehlo.subtract %s1b2bt2, %v4495 : tensor<64xf32>
    %v4497 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4498 = stablehlo.multiply %v4497, %d2W1 : tensor<128x64x3x3xf32>
    %v4499 = stablehlo.add %v4498, %v3085 : tensor<128x64x3x3xf32>
    %v4500 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4501 = stablehlo.multiply %v4500, %d2W1v : tensor<128x64x3x3xf32>
    %v4502 = stablehlo.add %v4501, %v4499 : tensor<128x64x3x3xf32>
    %v4503 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4504 = stablehlo.multiply %v4503, %v4502 : tensor<128x64x3x3xf32>
    %v4505 = stablehlo.subtract %d2W1, %v4504 : tensor<128x64x3x3xf32>
    %v4506 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4507 = stablehlo.multiply %v4506, %d2g1 : tensor<128xf32>
    %v4508 = stablehlo.add %v4507, %v3103 : tensor<128xf32>
    %v4509 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4510 = stablehlo.multiply %v4509, %d2g1v : tensor<128xf32>
    %v4511 = stablehlo.add %v4510, %v4508 : tensor<128xf32>
    %v4512 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4513 = stablehlo.multiply %v4512, %v4511 : tensor<128xf32>
    %v4514 = stablehlo.subtract %d2g1, %v4513 : tensor<128xf32>
    %v4515 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4516 = stablehlo.multiply %v4515, %d2bt1 : tensor<128xf32>
    %v4517 = stablehlo.add %v4516, %v3106 : tensor<128xf32>
    %v4518 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4519 = stablehlo.multiply %v4518, %d2bt1v : tensor<128xf32>
    %v4520 = stablehlo.add %v4519, %v4517 : tensor<128xf32>
    %v4521 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4522 = stablehlo.multiply %v4521, %v4520 : tensor<128xf32>
    %v4523 = stablehlo.subtract %d2bt1, %v4522 : tensor<128xf32>
    %v4524 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4525 = stablehlo.multiply %v4524, %d2W2 : tensor<128x128x3x3xf32>
    %v4526 = stablehlo.add %v4525, %v3112 : tensor<128x128x3x3xf32>
    %v4527 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4528 = stablehlo.multiply %v4527, %d2W2v : tensor<128x128x3x3xf32>
    %v4529 = stablehlo.add %v4528, %v4526 : tensor<128x128x3x3xf32>
    %v4530 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4531 = stablehlo.multiply %v4530, %v4529 : tensor<128x128x3x3xf32>
    %v4532 = stablehlo.subtract %d2W2, %v4531 : tensor<128x128x3x3xf32>
    %v4533 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4534 = stablehlo.multiply %v4533, %d2g2 : tensor<128xf32>
    %v4535 = stablehlo.add %v4534, %v3130 : tensor<128xf32>
    %v4536 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4537 = stablehlo.multiply %v4536, %d2g2v : tensor<128xf32>
    %v4538 = stablehlo.add %v4537, %v4535 : tensor<128xf32>
    %v4539 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4540 = stablehlo.multiply %v4539, %v4538 : tensor<128xf32>
    %v4541 = stablehlo.subtract %d2g2, %v4540 : tensor<128xf32>
    %v4542 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4543 = stablehlo.multiply %v4542, %d2bt2 : tensor<128xf32>
    %v4544 = stablehlo.add %v4543, %v3133 : tensor<128xf32>
    %v4545 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4546 = stablehlo.multiply %v4545, %d2bt2v : tensor<128xf32>
    %v4547 = stablehlo.add %v4546, %v4544 : tensor<128xf32>
    %v4548 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4549 = stablehlo.multiply %v4548, %v4547 : tensor<128xf32>
    %v4550 = stablehlo.subtract %d2bt2, %v4549 : tensor<128xf32>
    %v4551 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4552 = stablehlo.multiply %v4551, %d2Wp : tensor<128x64x1x1xf32>
    %v4553 = stablehlo.add %v4552, %v3141 : tensor<128x64x1x1xf32>
    %v4554 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4555 = stablehlo.multiply %v4554, %d2Wpv : tensor<128x64x1x1xf32>
    %v4556 = stablehlo.add %v4555, %v4553 : tensor<128x64x1x1xf32>
    %v4557 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4558 = stablehlo.multiply %v4557, %v4556 : tensor<128x64x1x1xf32>
    %v4559 = stablehlo.subtract %d2Wp, %v4558 : tensor<128x64x1x1xf32>
    %v4560 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4561 = stablehlo.multiply %v4560, %d2gp : tensor<128xf32>
    %v4562 = stablehlo.add %v4561, %v3159 : tensor<128xf32>
    %v4563 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4564 = stablehlo.multiply %v4563, %d2gpv : tensor<128xf32>
    %v4565 = stablehlo.add %v4564, %v4562 : tensor<128xf32>
    %v4566 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4567 = stablehlo.multiply %v4566, %v4565 : tensor<128xf32>
    %v4568 = stablehlo.subtract %d2gp, %v4567 : tensor<128xf32>
    %v4569 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4570 = stablehlo.multiply %v4569, %d2btp : tensor<128xf32>
    %v4571 = stablehlo.add %v4570, %v3162 : tensor<128xf32>
    %v4572 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4573 = stablehlo.multiply %v4572, %d2btpv : tensor<128xf32>
    %v4574 = stablehlo.add %v4573, %v4571 : tensor<128xf32>
    %v4575 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4576 = stablehlo.multiply %v4575, %v4574 : tensor<128xf32>
    %v4577 = stablehlo.subtract %d2btp, %v4576 : tensor<128xf32>
    %v4578 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4579 = stablehlo.multiply %v4578, %s2b0W1 : tensor<128x128x3x3xf32>
    %v4580 = stablehlo.add %v4579, %v2904 : tensor<128x128x3x3xf32>
    %v4581 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4582 = stablehlo.multiply %v4581, %s2b0W1v : tensor<128x128x3x3xf32>
    %v4583 = stablehlo.add %v4582, %v4580 : tensor<128x128x3x3xf32>
    %v4584 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4585 = stablehlo.multiply %v4584, %v4583 : tensor<128x128x3x3xf32>
    %v4586 = stablehlo.subtract %s2b0W1, %v4585 : tensor<128x128x3x3xf32>
    %v4587 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4588 = stablehlo.multiply %v4587, %s2b0g1 : tensor<128xf32>
    %v4589 = stablehlo.add %v4588, %v2922 : tensor<128xf32>
    %v4590 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4591 = stablehlo.multiply %v4590, %s2b0g1v : tensor<128xf32>
    %v4592 = stablehlo.add %v4591, %v4589 : tensor<128xf32>
    %v4593 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4594 = stablehlo.multiply %v4593, %v4592 : tensor<128xf32>
    %v4595 = stablehlo.subtract %s2b0g1, %v4594 : tensor<128xf32>
    %v4596 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4597 = stablehlo.multiply %v4596, %s2b0bt1 : tensor<128xf32>
    %v4598 = stablehlo.add %v4597, %v2925 : tensor<128xf32>
    %v4599 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4600 = stablehlo.multiply %v4599, %s2b0bt1v : tensor<128xf32>
    %v4601 = stablehlo.add %v4600, %v4598 : tensor<128xf32>
    %v4602 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4603 = stablehlo.multiply %v4602, %v4601 : tensor<128xf32>
    %v4604 = stablehlo.subtract %s2b0bt1, %v4603 : tensor<128xf32>
    %v4605 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4606 = stablehlo.multiply %v4605, %s2b0W2 : tensor<128x128x3x3xf32>
    %v4607 = stablehlo.add %v4606, %v2931 : tensor<128x128x3x3xf32>
    %v4608 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4609 = stablehlo.multiply %v4608, %s2b0W2v : tensor<128x128x3x3xf32>
    %v4610 = stablehlo.add %v4609, %v4607 : tensor<128x128x3x3xf32>
    %v4611 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4612 = stablehlo.multiply %v4611, %v4610 : tensor<128x128x3x3xf32>
    %v4613 = stablehlo.subtract %s2b0W2, %v4612 : tensor<128x128x3x3xf32>
    %v4614 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4615 = stablehlo.multiply %v4614, %s2b0g2 : tensor<128xf32>
    %v4616 = stablehlo.add %v4615, %v2949 : tensor<128xf32>
    %v4617 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4618 = stablehlo.multiply %v4617, %s2b0g2v : tensor<128xf32>
    %v4619 = stablehlo.add %v4618, %v4616 : tensor<128xf32>
    %v4620 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4621 = stablehlo.multiply %v4620, %v4619 : tensor<128xf32>
    %v4622 = stablehlo.subtract %s2b0g2, %v4621 : tensor<128xf32>
    %v4623 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4624 = stablehlo.multiply %v4623, %s2b0bt2 : tensor<128xf32>
    %v4625 = stablehlo.add %v4624, %v2952 : tensor<128xf32>
    %v4626 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4627 = stablehlo.multiply %v4626, %s2b0bt2v : tensor<128xf32>
    %v4628 = stablehlo.add %v4627, %v4625 : tensor<128xf32>
    %v4629 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4630 = stablehlo.multiply %v4629, %v4628 : tensor<128xf32>
    %v4631 = stablehlo.subtract %s2b0bt2, %v4630 : tensor<128xf32>
    %v4632 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4633 = stablehlo.multiply %v4632, %s2b1W1 : tensor<128x128x3x3xf32>
    %v4634 = stablehlo.add %v4633, %v2764 : tensor<128x128x3x3xf32>
    %v4635 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4636 = stablehlo.multiply %v4635, %s2b1W1v : tensor<128x128x3x3xf32>
    %v4637 = stablehlo.add %v4636, %v4634 : tensor<128x128x3x3xf32>
    %v4638 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4639 = stablehlo.multiply %v4638, %v4637 : tensor<128x128x3x3xf32>
    %v4640 = stablehlo.subtract %s2b1W1, %v4639 : tensor<128x128x3x3xf32>
    %v4641 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4642 = stablehlo.multiply %v4641, %s2b1g1 : tensor<128xf32>
    %v4643 = stablehlo.add %v4642, %v2782 : tensor<128xf32>
    %v4644 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4645 = stablehlo.multiply %v4644, %s2b1g1v : tensor<128xf32>
    %v4646 = stablehlo.add %v4645, %v4643 : tensor<128xf32>
    %v4647 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4648 = stablehlo.multiply %v4647, %v4646 : tensor<128xf32>
    %v4649 = stablehlo.subtract %s2b1g1, %v4648 : tensor<128xf32>
    %v4650 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4651 = stablehlo.multiply %v4650, %s2b1bt1 : tensor<128xf32>
    %v4652 = stablehlo.add %v4651, %v2785 : tensor<128xf32>
    %v4653 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4654 = stablehlo.multiply %v4653, %s2b1bt1v : tensor<128xf32>
    %v4655 = stablehlo.add %v4654, %v4652 : tensor<128xf32>
    %v4656 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4657 = stablehlo.multiply %v4656, %v4655 : tensor<128xf32>
    %v4658 = stablehlo.subtract %s2b1bt1, %v4657 : tensor<128xf32>
    %v4659 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4660 = stablehlo.multiply %v4659, %s2b1W2 : tensor<128x128x3x3xf32>
    %v4661 = stablehlo.add %v4660, %v2791 : tensor<128x128x3x3xf32>
    %v4662 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4663 = stablehlo.multiply %v4662, %s2b1W2v : tensor<128x128x3x3xf32>
    %v4664 = stablehlo.add %v4663, %v4661 : tensor<128x128x3x3xf32>
    %v4665 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4666 = stablehlo.multiply %v4665, %v4664 : tensor<128x128x3x3xf32>
    %v4667 = stablehlo.subtract %s2b1W2, %v4666 : tensor<128x128x3x3xf32>
    %v4668 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4669 = stablehlo.multiply %v4668, %s2b1g2 : tensor<128xf32>
    %v4670 = stablehlo.add %v4669, %v2809 : tensor<128xf32>
    %v4671 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4672 = stablehlo.multiply %v4671, %s2b1g2v : tensor<128xf32>
    %v4673 = stablehlo.add %v4672, %v4670 : tensor<128xf32>
    %v4674 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4675 = stablehlo.multiply %v4674, %v4673 : tensor<128xf32>
    %v4676 = stablehlo.subtract %s2b1g2, %v4675 : tensor<128xf32>
    %v4677 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4678 = stablehlo.multiply %v4677, %s2b1bt2 : tensor<128xf32>
    %v4679 = stablehlo.add %v4678, %v2812 : tensor<128xf32>
    %v4680 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4681 = stablehlo.multiply %v4680, %s2b1bt2v : tensor<128xf32>
    %v4682 = stablehlo.add %v4681, %v4679 : tensor<128xf32>
    %v4683 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4684 = stablehlo.multiply %v4683, %v4682 : tensor<128xf32>
    %v4685 = stablehlo.subtract %s2b1bt2, %v4684 : tensor<128xf32>
    %v4686 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4687 = stablehlo.multiply %v4686, %s2b2W1 : tensor<128x128x3x3xf32>
    %v4688 = stablehlo.add %v4687, %v2624 : tensor<128x128x3x3xf32>
    %v4689 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4690 = stablehlo.multiply %v4689, %s2b2W1v : tensor<128x128x3x3xf32>
    %v4691 = stablehlo.add %v4690, %v4688 : tensor<128x128x3x3xf32>
    %v4692 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4693 = stablehlo.multiply %v4692, %v4691 : tensor<128x128x3x3xf32>
    %v4694 = stablehlo.subtract %s2b2W1, %v4693 : tensor<128x128x3x3xf32>
    %v4695 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4696 = stablehlo.multiply %v4695, %s2b2g1 : tensor<128xf32>
    %v4697 = stablehlo.add %v4696, %v2642 : tensor<128xf32>
    %v4698 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4699 = stablehlo.multiply %v4698, %s2b2g1v : tensor<128xf32>
    %v4700 = stablehlo.add %v4699, %v4697 : tensor<128xf32>
    %v4701 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4702 = stablehlo.multiply %v4701, %v4700 : tensor<128xf32>
    %v4703 = stablehlo.subtract %s2b2g1, %v4702 : tensor<128xf32>
    %v4704 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4705 = stablehlo.multiply %v4704, %s2b2bt1 : tensor<128xf32>
    %v4706 = stablehlo.add %v4705, %v2645 : tensor<128xf32>
    %v4707 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4708 = stablehlo.multiply %v4707, %s2b2bt1v : tensor<128xf32>
    %v4709 = stablehlo.add %v4708, %v4706 : tensor<128xf32>
    %v4710 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4711 = stablehlo.multiply %v4710, %v4709 : tensor<128xf32>
    %v4712 = stablehlo.subtract %s2b2bt1, %v4711 : tensor<128xf32>
    %v4713 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4714 = stablehlo.multiply %v4713, %s2b2W2 : tensor<128x128x3x3xf32>
    %v4715 = stablehlo.add %v4714, %v2651 : tensor<128x128x3x3xf32>
    %v4716 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4717 = stablehlo.multiply %v4716, %s2b2W2v : tensor<128x128x3x3xf32>
    %v4718 = stablehlo.add %v4717, %v4715 : tensor<128x128x3x3xf32>
    %v4719 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4720 = stablehlo.multiply %v4719, %v4718 : tensor<128x128x3x3xf32>
    %v4721 = stablehlo.subtract %s2b2W2, %v4720 : tensor<128x128x3x3xf32>
    %v4722 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4723 = stablehlo.multiply %v4722, %s2b2g2 : tensor<128xf32>
    %v4724 = stablehlo.add %v4723, %v2669 : tensor<128xf32>
    %v4725 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4726 = stablehlo.multiply %v4725, %s2b2g2v : tensor<128xf32>
    %v4727 = stablehlo.add %v4726, %v4724 : tensor<128xf32>
    %v4728 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4729 = stablehlo.multiply %v4728, %v4727 : tensor<128xf32>
    %v4730 = stablehlo.subtract %s2b2g2, %v4729 : tensor<128xf32>
    %v4731 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4732 = stablehlo.multiply %v4731, %s2b2bt2 : tensor<128xf32>
    %v4733 = stablehlo.add %v4732, %v2672 : tensor<128xf32>
    %v4734 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4735 = stablehlo.multiply %v4734, %s2b2bt2v : tensor<128xf32>
    %v4736 = stablehlo.add %v4735, %v4733 : tensor<128xf32>
    %v4737 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4738 = stablehlo.multiply %v4737, %v4736 : tensor<128xf32>
    %v4739 = stablehlo.subtract %s2b2bt2, %v4738 : tensor<128xf32>
    %v4740 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4741 = stablehlo.multiply %v4740, %d3W1 : tensor<256x128x3x3xf32>
    %v4742 = stablehlo.add %v4741, %v2455 : tensor<256x128x3x3xf32>
    %v4743 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4744 = stablehlo.multiply %v4743, %d3W1v : tensor<256x128x3x3xf32>
    %v4745 = stablehlo.add %v4744, %v4742 : tensor<256x128x3x3xf32>
    %v4746 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4747 = stablehlo.multiply %v4746, %v4745 : tensor<256x128x3x3xf32>
    %v4748 = stablehlo.subtract %d3W1, %v4747 : tensor<256x128x3x3xf32>
    %v4749 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4750 = stablehlo.multiply %v4749, %d3g1 : tensor<256xf32>
    %v4751 = stablehlo.add %v4750, %v2473 : tensor<256xf32>
    %v4752 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4753 = stablehlo.multiply %v4752, %d3g1v : tensor<256xf32>
    %v4754 = stablehlo.add %v4753, %v4751 : tensor<256xf32>
    %v4755 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4756 = stablehlo.multiply %v4755, %v4754 : tensor<256xf32>
    %v4757 = stablehlo.subtract %d3g1, %v4756 : tensor<256xf32>
    %v4758 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4759 = stablehlo.multiply %v4758, %d3bt1 : tensor<256xf32>
    %v4760 = stablehlo.add %v4759, %v2476 : tensor<256xf32>
    %v4761 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4762 = stablehlo.multiply %v4761, %d3bt1v : tensor<256xf32>
    %v4763 = stablehlo.add %v4762, %v4760 : tensor<256xf32>
    %v4764 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4765 = stablehlo.multiply %v4764, %v4763 : tensor<256xf32>
    %v4766 = stablehlo.subtract %d3bt1, %v4765 : tensor<256xf32>
    %v4767 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4768 = stablehlo.multiply %v4767, %d3W2 : tensor<256x256x3x3xf32>
    %v4769 = stablehlo.add %v4768, %v2482 : tensor<256x256x3x3xf32>
    %v4770 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4771 = stablehlo.multiply %v4770, %d3W2v : tensor<256x256x3x3xf32>
    %v4772 = stablehlo.add %v4771, %v4769 : tensor<256x256x3x3xf32>
    %v4773 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4774 = stablehlo.multiply %v4773, %v4772 : tensor<256x256x3x3xf32>
    %v4775 = stablehlo.subtract %d3W2, %v4774 : tensor<256x256x3x3xf32>
    %v4776 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4777 = stablehlo.multiply %v4776, %d3g2 : tensor<256xf32>
    %v4778 = stablehlo.add %v4777, %v2500 : tensor<256xf32>
    %v4779 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4780 = stablehlo.multiply %v4779, %d3g2v : tensor<256xf32>
    %v4781 = stablehlo.add %v4780, %v4778 : tensor<256xf32>
    %v4782 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4783 = stablehlo.multiply %v4782, %v4781 : tensor<256xf32>
    %v4784 = stablehlo.subtract %d3g2, %v4783 : tensor<256xf32>
    %v4785 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4786 = stablehlo.multiply %v4785, %d3bt2 : tensor<256xf32>
    %v4787 = stablehlo.add %v4786, %v2503 : tensor<256xf32>
    %v4788 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4789 = stablehlo.multiply %v4788, %d3bt2v : tensor<256xf32>
    %v4790 = stablehlo.add %v4789, %v4787 : tensor<256xf32>
    %v4791 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4792 = stablehlo.multiply %v4791, %v4790 : tensor<256xf32>
    %v4793 = stablehlo.subtract %d3bt2, %v4792 : tensor<256xf32>
    %v4794 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v4795 = stablehlo.multiply %v4794, %d3Wp : tensor<256x128x1x1xf32>
    %v4796 = stablehlo.add %v4795, %v2511 : tensor<256x128x1x1xf32>
    %v4797 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v4798 = stablehlo.multiply %v4797, %d3Wpv : tensor<256x128x1x1xf32>
    %v4799 = stablehlo.add %v4798, %v4796 : tensor<256x128x1x1xf32>
    %v4800 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v4801 = stablehlo.multiply %v4800, %v4799 : tensor<256x128x1x1xf32>
    %v4802 = stablehlo.subtract %d3Wp, %v4801 : tensor<256x128x1x1xf32>
    %v4803 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4804 = stablehlo.multiply %v4803, %d3gp : tensor<256xf32>
    %v4805 = stablehlo.add %v4804, %v2529 : tensor<256xf32>
    %v4806 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4807 = stablehlo.multiply %v4806, %d3gpv : tensor<256xf32>
    %v4808 = stablehlo.add %v4807, %v4805 : tensor<256xf32>
    %v4809 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4810 = stablehlo.multiply %v4809, %v4808 : tensor<256xf32>
    %v4811 = stablehlo.subtract %d3gp, %v4810 : tensor<256xf32>
    %v4812 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4813 = stablehlo.multiply %v4812, %d3btp : tensor<256xf32>
    %v4814 = stablehlo.add %v4813, %v2532 : tensor<256xf32>
    %v4815 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4816 = stablehlo.multiply %v4815, %d3btpv : tensor<256xf32>
    %v4817 = stablehlo.add %v4816, %v4814 : tensor<256xf32>
    %v4818 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4819 = stablehlo.multiply %v4818, %v4817 : tensor<256xf32>
    %v4820 = stablehlo.subtract %d3btp, %v4819 : tensor<256xf32>
    %v4821 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4822 = stablehlo.multiply %v4821, %s3b0W1 : tensor<256x256x3x3xf32>
    %v4823 = stablehlo.add %v4822, %v2274 : tensor<256x256x3x3xf32>
    %v4824 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4825 = stablehlo.multiply %v4824, %s3b0W1v : tensor<256x256x3x3xf32>
    %v4826 = stablehlo.add %v4825, %v4823 : tensor<256x256x3x3xf32>
    %v4827 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4828 = stablehlo.multiply %v4827, %v4826 : tensor<256x256x3x3xf32>
    %v4829 = stablehlo.subtract %s3b0W1, %v4828 : tensor<256x256x3x3xf32>
    %v4830 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4831 = stablehlo.multiply %v4830, %s3b0g1 : tensor<256xf32>
    %v4832 = stablehlo.add %v4831, %v2292 : tensor<256xf32>
    %v4833 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4834 = stablehlo.multiply %v4833, %s3b0g1v : tensor<256xf32>
    %v4835 = stablehlo.add %v4834, %v4832 : tensor<256xf32>
    %v4836 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4837 = stablehlo.multiply %v4836, %v4835 : tensor<256xf32>
    %v4838 = stablehlo.subtract %s3b0g1, %v4837 : tensor<256xf32>
    %v4839 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4840 = stablehlo.multiply %v4839, %s3b0bt1 : tensor<256xf32>
    %v4841 = stablehlo.add %v4840, %v2295 : tensor<256xf32>
    %v4842 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4843 = stablehlo.multiply %v4842, %s3b0bt1v : tensor<256xf32>
    %v4844 = stablehlo.add %v4843, %v4841 : tensor<256xf32>
    %v4845 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4846 = stablehlo.multiply %v4845, %v4844 : tensor<256xf32>
    %v4847 = stablehlo.subtract %s3b0bt1, %v4846 : tensor<256xf32>
    %v4848 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4849 = stablehlo.multiply %v4848, %s3b0W2 : tensor<256x256x3x3xf32>
    %v4850 = stablehlo.add %v4849, %v2301 : tensor<256x256x3x3xf32>
    %v4851 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4852 = stablehlo.multiply %v4851, %s3b0W2v : tensor<256x256x3x3xf32>
    %v4853 = stablehlo.add %v4852, %v4850 : tensor<256x256x3x3xf32>
    %v4854 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4855 = stablehlo.multiply %v4854, %v4853 : tensor<256x256x3x3xf32>
    %v4856 = stablehlo.subtract %s3b0W2, %v4855 : tensor<256x256x3x3xf32>
    %v4857 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4858 = stablehlo.multiply %v4857, %s3b0g2 : tensor<256xf32>
    %v4859 = stablehlo.add %v4858, %v2319 : tensor<256xf32>
    %v4860 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4861 = stablehlo.multiply %v4860, %s3b0g2v : tensor<256xf32>
    %v4862 = stablehlo.add %v4861, %v4859 : tensor<256xf32>
    %v4863 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4864 = stablehlo.multiply %v4863, %v4862 : tensor<256xf32>
    %v4865 = stablehlo.subtract %s3b0g2, %v4864 : tensor<256xf32>
    %v4866 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4867 = stablehlo.multiply %v4866, %s3b0bt2 : tensor<256xf32>
    %v4868 = stablehlo.add %v4867, %v2322 : tensor<256xf32>
    %v4869 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4870 = stablehlo.multiply %v4869, %s3b0bt2v : tensor<256xf32>
    %v4871 = stablehlo.add %v4870, %v4868 : tensor<256xf32>
    %v4872 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4873 = stablehlo.multiply %v4872, %v4871 : tensor<256xf32>
    %v4874 = stablehlo.subtract %s3b0bt2, %v4873 : tensor<256xf32>
    %v4875 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4876 = stablehlo.multiply %v4875, %s3b1W1 : tensor<256x256x3x3xf32>
    %v4877 = stablehlo.add %v4876, %v2134 : tensor<256x256x3x3xf32>
    %v4878 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4879 = stablehlo.multiply %v4878, %s3b1W1v : tensor<256x256x3x3xf32>
    %v4880 = stablehlo.add %v4879, %v4877 : tensor<256x256x3x3xf32>
    %v4881 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4882 = stablehlo.multiply %v4881, %v4880 : tensor<256x256x3x3xf32>
    %v4883 = stablehlo.subtract %s3b1W1, %v4882 : tensor<256x256x3x3xf32>
    %v4884 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4885 = stablehlo.multiply %v4884, %s3b1g1 : tensor<256xf32>
    %v4886 = stablehlo.add %v4885, %v2152 : tensor<256xf32>
    %v4887 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4888 = stablehlo.multiply %v4887, %s3b1g1v : tensor<256xf32>
    %v4889 = stablehlo.add %v4888, %v4886 : tensor<256xf32>
    %v4890 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4891 = stablehlo.multiply %v4890, %v4889 : tensor<256xf32>
    %v4892 = stablehlo.subtract %s3b1g1, %v4891 : tensor<256xf32>
    %v4893 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4894 = stablehlo.multiply %v4893, %s3b1bt1 : tensor<256xf32>
    %v4895 = stablehlo.add %v4894, %v2155 : tensor<256xf32>
    %v4896 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4897 = stablehlo.multiply %v4896, %s3b1bt1v : tensor<256xf32>
    %v4898 = stablehlo.add %v4897, %v4895 : tensor<256xf32>
    %v4899 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4900 = stablehlo.multiply %v4899, %v4898 : tensor<256xf32>
    %v4901 = stablehlo.subtract %s3b1bt1, %v4900 : tensor<256xf32>
    %v4902 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4903 = stablehlo.multiply %v4902, %s3b1W2 : tensor<256x256x3x3xf32>
    %v4904 = stablehlo.add %v4903, %v2161 : tensor<256x256x3x3xf32>
    %v4905 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4906 = stablehlo.multiply %v4905, %s3b1W2v : tensor<256x256x3x3xf32>
    %v4907 = stablehlo.add %v4906, %v4904 : tensor<256x256x3x3xf32>
    %v4908 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4909 = stablehlo.multiply %v4908, %v4907 : tensor<256x256x3x3xf32>
    %v4910 = stablehlo.subtract %s3b1W2, %v4909 : tensor<256x256x3x3xf32>
    %v4911 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4912 = stablehlo.multiply %v4911, %s3b1g2 : tensor<256xf32>
    %v4913 = stablehlo.add %v4912, %v2179 : tensor<256xf32>
    %v4914 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4915 = stablehlo.multiply %v4914, %s3b1g2v : tensor<256xf32>
    %v4916 = stablehlo.add %v4915, %v4913 : tensor<256xf32>
    %v4917 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4918 = stablehlo.multiply %v4917, %v4916 : tensor<256xf32>
    %v4919 = stablehlo.subtract %s3b1g2, %v4918 : tensor<256xf32>
    %v4920 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4921 = stablehlo.multiply %v4920, %s3b1bt2 : tensor<256xf32>
    %v4922 = stablehlo.add %v4921, %v2182 : tensor<256xf32>
    %v4923 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4924 = stablehlo.multiply %v4923, %s3b1bt2v : tensor<256xf32>
    %v4925 = stablehlo.add %v4924, %v4922 : tensor<256xf32>
    %v4926 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4927 = stablehlo.multiply %v4926, %v4925 : tensor<256xf32>
    %v4928 = stablehlo.subtract %s3b1bt2, %v4927 : tensor<256xf32>
    %v4929 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4930 = stablehlo.multiply %v4929, %s3b2W1 : tensor<256x256x3x3xf32>
    %v4931 = stablehlo.add %v4930, %v1994 : tensor<256x256x3x3xf32>
    %v4932 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4933 = stablehlo.multiply %v4932, %s3b2W1v : tensor<256x256x3x3xf32>
    %v4934 = stablehlo.add %v4933, %v4931 : tensor<256x256x3x3xf32>
    %v4935 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4936 = stablehlo.multiply %v4935, %v4934 : tensor<256x256x3x3xf32>
    %v4937 = stablehlo.subtract %s3b2W1, %v4936 : tensor<256x256x3x3xf32>
    %v4938 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4939 = stablehlo.multiply %v4938, %s3b2g1 : tensor<256xf32>
    %v4940 = stablehlo.add %v4939, %v2012 : tensor<256xf32>
    %v4941 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4942 = stablehlo.multiply %v4941, %s3b2g1v : tensor<256xf32>
    %v4943 = stablehlo.add %v4942, %v4940 : tensor<256xf32>
    %v4944 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4945 = stablehlo.multiply %v4944, %v4943 : tensor<256xf32>
    %v4946 = stablehlo.subtract %s3b2g1, %v4945 : tensor<256xf32>
    %v4947 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4948 = stablehlo.multiply %v4947, %s3b2bt1 : tensor<256xf32>
    %v4949 = stablehlo.add %v4948, %v2015 : tensor<256xf32>
    %v4950 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4951 = stablehlo.multiply %v4950, %s3b2bt1v : tensor<256xf32>
    %v4952 = stablehlo.add %v4951, %v4949 : tensor<256xf32>
    %v4953 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4954 = stablehlo.multiply %v4953, %v4952 : tensor<256xf32>
    %v4955 = stablehlo.subtract %s3b2bt1, %v4954 : tensor<256xf32>
    %v4956 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4957 = stablehlo.multiply %v4956, %s3b2W2 : tensor<256x256x3x3xf32>
    %v4958 = stablehlo.add %v4957, %v2021 : tensor<256x256x3x3xf32>
    %v4959 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4960 = stablehlo.multiply %v4959, %s3b2W2v : tensor<256x256x3x3xf32>
    %v4961 = stablehlo.add %v4960, %v4958 : tensor<256x256x3x3xf32>
    %v4962 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4963 = stablehlo.multiply %v4962, %v4961 : tensor<256x256x3x3xf32>
    %v4964 = stablehlo.subtract %s3b2W2, %v4963 : tensor<256x256x3x3xf32>
    %v4965 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4966 = stablehlo.multiply %v4965, %s3b2g2 : tensor<256xf32>
    %v4967 = stablehlo.add %v4966, %v2039 : tensor<256xf32>
    %v4968 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4969 = stablehlo.multiply %v4968, %s3b2g2v : tensor<256xf32>
    %v4970 = stablehlo.add %v4969, %v4967 : tensor<256xf32>
    %v4971 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4972 = stablehlo.multiply %v4971, %v4970 : tensor<256xf32>
    %v4973 = stablehlo.subtract %s3b2g2, %v4972 : tensor<256xf32>
    %v4974 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4975 = stablehlo.multiply %v4974, %s3b2bt2 : tensor<256xf32>
    %v4976 = stablehlo.add %v4975, %v2042 : tensor<256xf32>
    %v4977 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4978 = stablehlo.multiply %v4977, %s3b2bt2v : tensor<256xf32>
    %v4979 = stablehlo.add %v4978, %v4976 : tensor<256xf32>
    %v4980 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4981 = stablehlo.multiply %v4980, %v4979 : tensor<256xf32>
    %v4982 = stablehlo.subtract %s3b2bt2, %v4981 : tensor<256xf32>
    %v4983 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4984 = stablehlo.multiply %v4983, %s3b3W1 : tensor<256x256x3x3xf32>
    %v4985 = stablehlo.add %v4984, %v1854 : tensor<256x256x3x3xf32>
    %v4986 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4987 = stablehlo.multiply %v4986, %s3b3W1v : tensor<256x256x3x3xf32>
    %v4988 = stablehlo.add %v4987, %v4985 : tensor<256x256x3x3xf32>
    %v4989 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4990 = stablehlo.multiply %v4989, %v4988 : tensor<256x256x3x3xf32>
    %v4991 = stablehlo.subtract %s3b3W1, %v4990 : tensor<256x256x3x3xf32>
    %v4992 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4993 = stablehlo.multiply %v4992, %s3b3g1 : tensor<256xf32>
    %v4994 = stablehlo.add %v4993, %v1872 : tensor<256xf32>
    %v4995 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4996 = stablehlo.multiply %v4995, %s3b3g1v : tensor<256xf32>
    %v4997 = stablehlo.add %v4996, %v4994 : tensor<256xf32>
    %v4998 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4999 = stablehlo.multiply %v4998, %v4997 : tensor<256xf32>
    %v5000 = stablehlo.subtract %s3b3g1, %v4999 : tensor<256xf32>
    %v5001 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5002 = stablehlo.multiply %v5001, %s3b3bt1 : tensor<256xf32>
    %v5003 = stablehlo.add %v5002, %v1875 : tensor<256xf32>
    %v5004 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5005 = stablehlo.multiply %v5004, %s3b3bt1v : tensor<256xf32>
    %v5006 = stablehlo.add %v5005, %v5003 : tensor<256xf32>
    %v5007 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5008 = stablehlo.multiply %v5007, %v5006 : tensor<256xf32>
    %v5009 = stablehlo.subtract %s3b3bt1, %v5008 : tensor<256xf32>
    %v5010 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5011 = stablehlo.multiply %v5010, %s3b3W2 : tensor<256x256x3x3xf32>
    %v5012 = stablehlo.add %v5011, %v1881 : tensor<256x256x3x3xf32>
    %v5013 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5014 = stablehlo.multiply %v5013, %s3b3W2v : tensor<256x256x3x3xf32>
    %v5015 = stablehlo.add %v5014, %v5012 : tensor<256x256x3x3xf32>
    %v5016 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5017 = stablehlo.multiply %v5016, %v5015 : tensor<256x256x3x3xf32>
    %v5018 = stablehlo.subtract %s3b3W2, %v5017 : tensor<256x256x3x3xf32>
    %v5019 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5020 = stablehlo.multiply %v5019, %s3b3g2 : tensor<256xf32>
    %v5021 = stablehlo.add %v5020, %v1899 : tensor<256xf32>
    %v5022 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5023 = stablehlo.multiply %v5022, %s3b3g2v : tensor<256xf32>
    %v5024 = stablehlo.add %v5023, %v5021 : tensor<256xf32>
    %v5025 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5026 = stablehlo.multiply %v5025, %v5024 : tensor<256xf32>
    %v5027 = stablehlo.subtract %s3b3g2, %v5026 : tensor<256xf32>
    %v5028 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5029 = stablehlo.multiply %v5028, %s3b3bt2 : tensor<256xf32>
    %v5030 = stablehlo.add %v5029, %v1902 : tensor<256xf32>
    %v5031 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5032 = stablehlo.multiply %v5031, %s3b3bt2v : tensor<256xf32>
    %v5033 = stablehlo.add %v5032, %v5030 : tensor<256xf32>
    %v5034 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5035 = stablehlo.multiply %v5034, %v5033 : tensor<256xf32>
    %v5036 = stablehlo.subtract %s3b3bt2, %v5035 : tensor<256xf32>
    %v5037 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5038 = stablehlo.multiply %v5037, %s3b4W1 : tensor<256x256x3x3xf32>
    %v5039 = stablehlo.add %v5038, %v1714 : tensor<256x256x3x3xf32>
    %v5040 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5041 = stablehlo.multiply %v5040, %s3b4W1v : tensor<256x256x3x3xf32>
    %v5042 = stablehlo.add %v5041, %v5039 : tensor<256x256x3x3xf32>
    %v5043 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5044 = stablehlo.multiply %v5043, %v5042 : tensor<256x256x3x3xf32>
    %v5045 = stablehlo.subtract %s3b4W1, %v5044 : tensor<256x256x3x3xf32>
    %v5046 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5047 = stablehlo.multiply %v5046, %s3b4g1 : tensor<256xf32>
    %v5048 = stablehlo.add %v5047, %v1732 : tensor<256xf32>
    %v5049 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5050 = stablehlo.multiply %v5049, %s3b4g1v : tensor<256xf32>
    %v5051 = stablehlo.add %v5050, %v5048 : tensor<256xf32>
    %v5052 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5053 = stablehlo.multiply %v5052, %v5051 : tensor<256xf32>
    %v5054 = stablehlo.subtract %s3b4g1, %v5053 : tensor<256xf32>
    %v5055 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5056 = stablehlo.multiply %v5055, %s3b4bt1 : tensor<256xf32>
    %v5057 = stablehlo.add %v5056, %v1735 : tensor<256xf32>
    %v5058 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5059 = stablehlo.multiply %v5058, %s3b4bt1v : tensor<256xf32>
    %v5060 = stablehlo.add %v5059, %v5057 : tensor<256xf32>
    %v5061 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5062 = stablehlo.multiply %v5061, %v5060 : tensor<256xf32>
    %v5063 = stablehlo.subtract %s3b4bt1, %v5062 : tensor<256xf32>
    %v5064 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5065 = stablehlo.multiply %v5064, %s3b4W2 : tensor<256x256x3x3xf32>
    %v5066 = stablehlo.add %v5065, %v1741 : tensor<256x256x3x3xf32>
    %v5067 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5068 = stablehlo.multiply %v5067, %s3b4W2v : tensor<256x256x3x3xf32>
    %v5069 = stablehlo.add %v5068, %v5066 : tensor<256x256x3x3xf32>
    %v5070 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5071 = stablehlo.multiply %v5070, %v5069 : tensor<256x256x3x3xf32>
    %v5072 = stablehlo.subtract %s3b4W2, %v5071 : tensor<256x256x3x3xf32>
    %v5073 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5074 = stablehlo.multiply %v5073, %s3b4g2 : tensor<256xf32>
    %v5075 = stablehlo.add %v5074, %v1759 : tensor<256xf32>
    %v5076 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5077 = stablehlo.multiply %v5076, %s3b4g2v : tensor<256xf32>
    %v5078 = stablehlo.add %v5077, %v5075 : tensor<256xf32>
    %v5079 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5080 = stablehlo.multiply %v5079, %v5078 : tensor<256xf32>
    %v5081 = stablehlo.subtract %s3b4g2, %v5080 : tensor<256xf32>
    %v5082 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5083 = stablehlo.multiply %v5082, %s3b4bt2 : tensor<256xf32>
    %v5084 = stablehlo.add %v5083, %v1762 : tensor<256xf32>
    %v5085 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5086 = stablehlo.multiply %v5085, %s3b4bt2v : tensor<256xf32>
    %v5087 = stablehlo.add %v5086, %v5084 : tensor<256xf32>
    %v5088 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5089 = stablehlo.multiply %v5088, %v5087 : tensor<256xf32>
    %v5090 = stablehlo.subtract %s3b4bt2, %v5089 : tensor<256xf32>
    %v5091 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5092 = stablehlo.multiply %v5091, %d4W1 : tensor<512x256x3x3xf32>
    %v5093 = stablehlo.add %v5092, %v1545 : tensor<512x256x3x3xf32>
    %v5094 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5095 = stablehlo.multiply %v5094, %d4W1v : tensor<512x256x3x3xf32>
    %v5096 = stablehlo.add %v5095, %v5093 : tensor<512x256x3x3xf32>
    %v5097 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5098 = stablehlo.multiply %v5097, %v5096 : tensor<512x256x3x3xf32>
    %v5099 = stablehlo.subtract %d4W1, %v5098 : tensor<512x256x3x3xf32>
    %v5100 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5101 = stablehlo.multiply %v5100, %d4g1 : tensor<512xf32>
    %v5102 = stablehlo.add %v5101, %v1563 : tensor<512xf32>
    %v5103 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5104 = stablehlo.multiply %v5103, %d4g1v : tensor<512xf32>
    %v5105 = stablehlo.add %v5104, %v5102 : tensor<512xf32>
    %v5106 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5107 = stablehlo.multiply %v5106, %v5105 : tensor<512xf32>
    %v5108 = stablehlo.subtract %d4g1, %v5107 : tensor<512xf32>
    %v5109 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5110 = stablehlo.multiply %v5109, %d4bt1 : tensor<512xf32>
    %v5111 = stablehlo.add %v5110, %v1566 : tensor<512xf32>
    %v5112 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5113 = stablehlo.multiply %v5112, %d4bt1v : tensor<512xf32>
    %v5114 = stablehlo.add %v5113, %v5111 : tensor<512xf32>
    %v5115 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5116 = stablehlo.multiply %v5115, %v5114 : tensor<512xf32>
    %v5117 = stablehlo.subtract %d4bt1, %v5116 : tensor<512xf32>
    %v5118 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5119 = stablehlo.multiply %v5118, %d4W2 : tensor<512x512x3x3xf32>
    %v5120 = stablehlo.add %v5119, %v1572 : tensor<512x512x3x3xf32>
    %v5121 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5122 = stablehlo.multiply %v5121, %d4W2v : tensor<512x512x3x3xf32>
    %v5123 = stablehlo.add %v5122, %v5120 : tensor<512x512x3x3xf32>
    %v5124 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5125 = stablehlo.multiply %v5124, %v5123 : tensor<512x512x3x3xf32>
    %v5126 = stablehlo.subtract %d4W2, %v5125 : tensor<512x512x3x3xf32>
    %v5127 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5128 = stablehlo.multiply %v5127, %d4g2 : tensor<512xf32>
    %v5129 = stablehlo.add %v5128, %v1590 : tensor<512xf32>
    %v5130 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5131 = stablehlo.multiply %v5130, %d4g2v : tensor<512xf32>
    %v5132 = stablehlo.add %v5131, %v5129 : tensor<512xf32>
    %v5133 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5134 = stablehlo.multiply %v5133, %v5132 : tensor<512xf32>
    %v5135 = stablehlo.subtract %d4g2, %v5134 : tensor<512xf32>
    %v5136 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5137 = stablehlo.multiply %v5136, %d4bt2 : tensor<512xf32>
    %v5138 = stablehlo.add %v5137, %v1593 : tensor<512xf32>
    %v5139 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5140 = stablehlo.multiply %v5139, %d4bt2v : tensor<512xf32>
    %v5141 = stablehlo.add %v5140, %v5138 : tensor<512xf32>
    %v5142 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5143 = stablehlo.multiply %v5142, %v5141 : tensor<512xf32>
    %v5144 = stablehlo.subtract %d4bt2, %v5143 : tensor<512xf32>
    %v5145 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5146 = stablehlo.multiply %v5145, %d4Wp : tensor<512x256x1x1xf32>
    %v5147 = stablehlo.add %v5146, %v1601 : tensor<512x256x1x1xf32>
    %v5148 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5149 = stablehlo.multiply %v5148, %d4Wpv : tensor<512x256x1x1xf32>
    %v5150 = stablehlo.add %v5149, %v5147 : tensor<512x256x1x1xf32>
    %v5151 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5152 = stablehlo.multiply %v5151, %v5150 : tensor<512x256x1x1xf32>
    %v5153 = stablehlo.subtract %d4Wp, %v5152 : tensor<512x256x1x1xf32>
    %v5154 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5155 = stablehlo.multiply %v5154, %d4gp : tensor<512xf32>
    %v5156 = stablehlo.add %v5155, %v1619 : tensor<512xf32>
    %v5157 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5158 = stablehlo.multiply %v5157, %d4gpv : tensor<512xf32>
    %v5159 = stablehlo.add %v5158, %v5156 : tensor<512xf32>
    %v5160 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5161 = stablehlo.multiply %v5160, %v5159 : tensor<512xf32>
    %v5162 = stablehlo.subtract %d4gp, %v5161 : tensor<512xf32>
    %v5163 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5164 = stablehlo.multiply %v5163, %d4btp : tensor<512xf32>
    %v5165 = stablehlo.add %v5164, %v1622 : tensor<512xf32>
    %v5166 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5167 = stablehlo.multiply %v5166, %d4btpv : tensor<512xf32>
    %v5168 = stablehlo.add %v5167, %v5165 : tensor<512xf32>
    %v5169 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5170 = stablehlo.multiply %v5169, %v5168 : tensor<512xf32>
    %v5171 = stablehlo.subtract %d4btp, %v5170 : tensor<512xf32>
    %v5172 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5173 = stablehlo.multiply %v5172, %s4b0W1 : tensor<512x512x3x3xf32>
    %v5174 = stablehlo.add %v5173, %v1364 : tensor<512x512x3x3xf32>
    %v5175 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5176 = stablehlo.multiply %v5175, %s4b0W1v : tensor<512x512x3x3xf32>
    %v5177 = stablehlo.add %v5176, %v5174 : tensor<512x512x3x3xf32>
    %v5178 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5179 = stablehlo.multiply %v5178, %v5177 : tensor<512x512x3x3xf32>
    %v5180 = stablehlo.subtract %s4b0W1, %v5179 : tensor<512x512x3x3xf32>
    %v5181 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5182 = stablehlo.multiply %v5181, %s4b0g1 : tensor<512xf32>
    %v5183 = stablehlo.add %v5182, %v1382 : tensor<512xf32>
    %v5184 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5185 = stablehlo.multiply %v5184, %s4b0g1v : tensor<512xf32>
    %v5186 = stablehlo.add %v5185, %v5183 : tensor<512xf32>
    %v5187 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5188 = stablehlo.multiply %v5187, %v5186 : tensor<512xf32>
    %v5189 = stablehlo.subtract %s4b0g1, %v5188 : tensor<512xf32>
    %v5190 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5191 = stablehlo.multiply %v5190, %s4b0bt1 : tensor<512xf32>
    %v5192 = stablehlo.add %v5191, %v1385 : tensor<512xf32>
    %v5193 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5194 = stablehlo.multiply %v5193, %s4b0bt1v : tensor<512xf32>
    %v5195 = stablehlo.add %v5194, %v5192 : tensor<512xf32>
    %v5196 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5197 = stablehlo.multiply %v5196, %v5195 : tensor<512xf32>
    %v5198 = stablehlo.subtract %s4b0bt1, %v5197 : tensor<512xf32>
    %v5199 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5200 = stablehlo.multiply %v5199, %s4b0W2 : tensor<512x512x3x3xf32>
    %v5201 = stablehlo.add %v5200, %v1391 : tensor<512x512x3x3xf32>
    %v5202 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5203 = stablehlo.multiply %v5202, %s4b0W2v : tensor<512x512x3x3xf32>
    %v5204 = stablehlo.add %v5203, %v5201 : tensor<512x512x3x3xf32>
    %v5205 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5206 = stablehlo.multiply %v5205, %v5204 : tensor<512x512x3x3xf32>
    %v5207 = stablehlo.subtract %s4b0W2, %v5206 : tensor<512x512x3x3xf32>
    %v5208 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5209 = stablehlo.multiply %v5208, %s4b0g2 : tensor<512xf32>
    %v5210 = stablehlo.add %v5209, %v1409 : tensor<512xf32>
    %v5211 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5212 = stablehlo.multiply %v5211, %s4b0g2v : tensor<512xf32>
    %v5213 = stablehlo.add %v5212, %v5210 : tensor<512xf32>
    %v5214 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5215 = stablehlo.multiply %v5214, %v5213 : tensor<512xf32>
    %v5216 = stablehlo.subtract %s4b0g2, %v5215 : tensor<512xf32>
    %v5217 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5218 = stablehlo.multiply %v5217, %s4b0bt2 : tensor<512xf32>
    %v5219 = stablehlo.add %v5218, %v1412 : tensor<512xf32>
    %v5220 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5221 = stablehlo.multiply %v5220, %s4b0bt2v : tensor<512xf32>
    %v5222 = stablehlo.add %v5221, %v5219 : tensor<512xf32>
    %v5223 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5224 = stablehlo.multiply %v5223, %v5222 : tensor<512xf32>
    %v5225 = stablehlo.subtract %s4b0bt2, %v5224 : tensor<512xf32>
    %v5226 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5227 = stablehlo.multiply %v5226, %s4b1W1 : tensor<512x512x3x3xf32>
    %v5228 = stablehlo.add %v5227, %v1224 : tensor<512x512x3x3xf32>
    %v5229 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5230 = stablehlo.multiply %v5229, %s4b1W1v : tensor<512x512x3x3xf32>
    %v5231 = stablehlo.add %v5230, %v5228 : tensor<512x512x3x3xf32>
    %v5232 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5233 = stablehlo.multiply %v5232, %v5231 : tensor<512x512x3x3xf32>
    %v5234 = stablehlo.subtract %s4b1W1, %v5233 : tensor<512x512x3x3xf32>
    %v5235 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5236 = stablehlo.multiply %v5235, %s4b1g1 : tensor<512xf32>
    %v5237 = stablehlo.add %v5236, %v1242 : tensor<512xf32>
    %v5238 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5239 = stablehlo.multiply %v5238, %s4b1g1v : tensor<512xf32>
    %v5240 = stablehlo.add %v5239, %v5237 : tensor<512xf32>
    %v5241 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5242 = stablehlo.multiply %v5241, %v5240 : tensor<512xf32>
    %v5243 = stablehlo.subtract %s4b1g1, %v5242 : tensor<512xf32>
    %v5244 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5245 = stablehlo.multiply %v5244, %s4b1bt1 : tensor<512xf32>
    %v5246 = stablehlo.add %v5245, %v1245 : tensor<512xf32>
    %v5247 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5248 = stablehlo.multiply %v5247, %s4b1bt1v : tensor<512xf32>
    %v5249 = stablehlo.add %v5248, %v5246 : tensor<512xf32>
    %v5250 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5251 = stablehlo.multiply %v5250, %v5249 : tensor<512xf32>
    %v5252 = stablehlo.subtract %s4b1bt1, %v5251 : tensor<512xf32>
    %v5253 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5254 = stablehlo.multiply %v5253, %s4b1W2 : tensor<512x512x3x3xf32>
    %v5255 = stablehlo.add %v5254, %v1251 : tensor<512x512x3x3xf32>
    %v5256 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5257 = stablehlo.multiply %v5256, %s4b1W2v : tensor<512x512x3x3xf32>
    %v5258 = stablehlo.add %v5257, %v5255 : tensor<512x512x3x3xf32>
    %v5259 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5260 = stablehlo.multiply %v5259, %v5258 : tensor<512x512x3x3xf32>
    %v5261 = stablehlo.subtract %s4b1W2, %v5260 : tensor<512x512x3x3xf32>
    %v5262 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5263 = stablehlo.multiply %v5262, %s4b1g2 : tensor<512xf32>
    %v5264 = stablehlo.add %v5263, %v1269 : tensor<512xf32>
    %v5265 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5266 = stablehlo.multiply %v5265, %s4b1g2v : tensor<512xf32>
    %v5267 = stablehlo.add %v5266, %v5264 : tensor<512xf32>
    %v5268 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5269 = stablehlo.multiply %v5268, %v5267 : tensor<512xf32>
    %v5270 = stablehlo.subtract %s4b1g2, %v5269 : tensor<512xf32>
    %v5271 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5272 = stablehlo.multiply %v5271, %s4b1bt2 : tensor<512xf32>
    %v5273 = stablehlo.add %v5272, %v1272 : tensor<512xf32>
    %v5274 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5275 = stablehlo.multiply %v5274, %s4b1bt2v : tensor<512xf32>
    %v5276 = stablehlo.add %v5275, %v5273 : tensor<512xf32>
    %v5277 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5278 = stablehlo.multiply %v5277, %v5276 : tensor<512xf32>
    %v5279 = stablehlo.subtract %s4b1bt2, %v5278 : tensor<512xf32>
    %v5280 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v5281 = stablehlo.multiply %v5280, %Wd : tensor<512x10xf32>
    %v5282 = stablehlo.add %v5281, %v1126 : tensor<512x10xf32>
    %v5283 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v5284 = stablehlo.multiply %v5283, %Wdv : tensor<512x10xf32>
    %v5285 = stablehlo.add %v5284, %v5282 : tensor<512x10xf32>
    %v5286 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v5287 = stablehlo.multiply %v5286, %v5285 : tensor<512x10xf32>
    %v5288 = stablehlo.subtract %Wd, %v5287 : tensor<512x10xf32>
    %v5289 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v5290 = stablehlo.multiply %v5289, %bd : tensor<10xf32>
    %v5291 = stablehlo.add %v5290, %v1128 : tensor<10xf32>
    %v5292 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v5293 = stablehlo.multiply %v5292, %bdv : tensor<10xf32>
    %v5294 = stablehlo.add %v5293, %v5291 : tensor<10xf32>
    %v5295 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v5296 = stablehlo.multiply %v5295, %v5294 : tensor<10xf32>
    %v5297 = stablehlo.subtract %bd, %v5296 : tensor<10xf32>
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
    return %v4316, %v4325, %v4334, %v4343, %v4352, %v4361, %v4370, %v4379, %v4388, %v4397, %v4406, %v4415, %v4424, %v4433, %v4442, %v4451, %v4460, %v4469, %v4478, %v4487, %v4496, %v4505, %v4514, %v4523, %v4532, %v4541, %v4550, %v4559, %v4568, %v4577, %v4586, %v4595, %v4604, %v4613, %v4622, %v4631, %v4640, %v4649, %v4658, %v4667, %v4676, %v4685, %v4694, %v4703, %v4712, %v4721, %v4730, %v4739, %v4748, %v4757, %v4766, %v4775, %v4784, %v4793, %v4802, %v4811, %v4820, %v4829, %v4838, %v4847, %v4856, %v4865, %v4874, %v4883, %v4892, %v4901, %v4910, %v4919, %v4928, %v4937, %v4946, %v4955, %v4964, %v4973, %v4982, %v4991, %v5000, %v5009, %v5018, %v5027, %v5036, %v5045, %v5054, %v5063, %v5072, %v5081, %v5090, %v5099, %v5108, %v5117, %v5126, %v5135, %v5144, %v5153, %v5162, %v5171, %v5180, %v5189, %v5198, %v5207, %v5216, %v5225, %v5234, %v5243, %v5252, %v5261, %v5270, %v5279, %v5288, %v5297, %sWm, %sgm, %sbtm, %s1b0W1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0g2m, %s1b0bt2m, %s1b1W1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1g2m, %s1b1bt2m, %s1b2W1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2g2m, %s1b2bt2m, %d2W1m, %d2g1m, %d2bt1m, %d2W2m, %d2g2m, %d2bt2m, %d2Wpm, %d2gpm, %d2btpm, %s2b0W1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0g2m, %s2b0bt2m, %s2b1W1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1g2m, %s2b1bt2m, %s2b2W1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2g2m, %s2b2bt2m, %d3W1m, %d3g1m, %d3bt1m, %d3W2m, %d3g2m, %d3bt2m, %d3Wpm, %d3gpm, %d3btpm, %s3b0W1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0g2m, %s3b0bt2m, %s3b1W1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1g2m, %s3b1bt2m, %s3b2W1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2g2m, %s3b2bt2m, %s3b3W1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3g2m, %s3b3bt2m, %s3b4W1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4g2m, %s3b4bt2m, %d4W1m, %d4g1m, %d4bt1m, %d4W2m, %d4g2m, %d4bt2m, %d4Wpm, %d4gpm, %d4btpm, %s4b0W1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0g2m, %s4b0bt2m, %s4b1W1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1g2m, %s4b1bt2m, %Wdm, %bdm, %v4313, %v4322, %v4331, %v4340, %v4349, %v4358, %v4367, %v4376, %v4385, %v4394, %v4403, %v4412, %v4421, %v4430, %v4439, %v4448, %v4457, %v4466, %v4475, %v4484, %v4493, %v4502, %v4511, %v4520, %v4529, %v4538, %v4547, %v4556, %v4565, %v4574, %v4583, %v4592, %v4601, %v4610, %v4619, %v4628, %v4637, %v4646, %v4655, %v4664, %v4673, %v4682, %v4691, %v4700, %v4709, %v4718, %v4727, %v4736, %v4745, %v4754, %v4763, %v4772, %v4781, %v4790, %v4799, %v4808, %v4817, %v4826, %v4835, %v4844, %v4853, %v4862, %v4871, %v4880, %v4889, %v4898, %v4907, %v4916, %v4925, %v4934, %v4943, %v4952, %v4961, %v4970, %v4979, %v4988, %v4997, %v5006, %v5015, %v5024, %v5033, %v5042, %v5051, %v5060, %v5069, %v5078, %v5087, %v5096, %v5105, %v5114, %v5123, %v5132, %v5141, %v5150, %v5159, %v5168, %v5177, %v5186, %v5195, %v5204, %v5213, %v5222, %v5231, %v5240, %v5249, %v5258, %v5267, %v5276, %v5285, %v5294, %loss, %bc1, %bc2, %v3736, %v3747, %v3752, %v3763, %v3768, %v3779, %v3784, %v3795, %v3800, %v3811, %v3816, %v3827, %v3832, %v3843, %v3848, %v3859, %v3864, %v3875, %v3880, %v3891, %v3896, %v3907, %v3912, %v3923, %v3928, %v3939, %v3944, %v3955, %v3960, %v3971, %v3976, %v3987, %v3992, %v4003, %v4008, %v4019, %v4024, %v4035, %v4040, %v4051, %v4056, %v4067, %v4072, %v4083, %v4088, %v4099, %v4104, %v4115, %v4120, %v4131, %v4136, %v4147, %v4152, %v4163, %v4168, %v4179, %v4184, %v4195, %v4200, %v4211, %v4216, %v4227, %v4232, %v4243, %v4248, %v4259, %v4264, %v4275, %v4280, %v4291, %v4296, %v4307 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}

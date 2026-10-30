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
    %v1491 = stablehlo.reverse %d4W1, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v1492 = stablehlo.transpose %v1491, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1493 = stablehlo.convolution(%v1490, %v1492)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1494 = stablehlo.reshape %v1493 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1495 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1496 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1497 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1498 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1499 = stablehlo.reduce(%v1495 init: %v1496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1500 = stablehlo.broadcast_in_dim %v1499, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1501 = stablehlo.divide %v1500, %v1497 : tensor<32x512x7x7xf32>
    %v1502 = stablehlo.subtract %v1495, %v1501 : tensor<32x512x7x7xf32>
    %v1503 = stablehlo.multiply %v1502, %v1502 : tensor<32x512x7x7xf32>
    %v1504 = stablehlo.reduce(%v1503 init: %v1496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1505 = stablehlo.broadcast_in_dim %v1504, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1506 = stablehlo.divide %v1505, %v1497 : tensor<32x512x7x7xf32>
    %v1507 = stablehlo.add %v1506, %v1498 : tensor<32x512x7x7xf32>
    %v1508 = stablehlo.rsqrt %v1507 : tensor<32x512x7x7xf32>
    %v1509 = stablehlo.multiply %v1502, %v1508 : tensor<32x512x7x7xf32>
    %v1510 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1511 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1512 = stablehlo.multiply %v1510, %v1511 : tensor<32x512x7x7xf32>
    %v1513 = stablehlo.reduce(%v1512 init: %v1496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1514 = stablehlo.broadcast_in_dim %v1513, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1515 = stablehlo.multiply %v1509, %v1512 : tensor<32x512x7x7xf32>
    %v1516 = stablehlo.reduce(%v1515 init: %v1496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1517 = stablehlo.broadcast_in_dim %v1516, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1518 = stablehlo.multiply %v1512, %v1497 : tensor<32x512x7x7xf32>
    %v1519 = stablehlo.subtract %v1518, %v1514 : tensor<32x512x7x7xf32>
    %v1520 = stablehlo.multiply %v1509, %v1517 : tensor<32x512x7x7xf32>
    %v1521 = stablehlo.subtract %v1519, %v1520 : tensor<32x512x7x7xf32>
    %v1522 = stablehlo.divide %v1508, %v1497 : tensor<32x512x7x7xf32>
    %v1523 = stablehlo.multiply %v1522, %v1521 : tensor<32x512x7x7xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1526 = stablehlo.reverse %d4Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v1527 = stablehlo.transpose %v1526, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v1528 = stablehlo.convolution(%v1525, %v1527)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<256x512x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1530 = stablehlo.reshape %v1494 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1531 = stablehlo.reshape %v1529 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1532 = stablehlo.add %v1530, %v1531 : tensor<32x256x14x14xf32>
    %v1533 = stablehlo.reshape %v1532 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1534 = stablehlo.reshape %v888 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1535 = stablehlo.reshape %v1489 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1536 = stablehlo.transpose %v1534, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1538 = stablehlo.transpose %v1535, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1539 = stablehlo.convolution(%v1536, %v1538)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x7x7xf32>) -> tensor<256x512x3x3xf32>
    %v1540 = stablehlo.transpose %v1539, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1541 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1543 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1544 = stablehlo.reduce(%v1541 init: %v1542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1545 = stablehlo.broadcast_in_dim %v1544, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1546 = stablehlo.divide %v1545, %v1543 : tensor<32x512x7x7xf32>
    %v1547 = stablehlo.subtract %v1541, %v1546 : tensor<32x512x7x7xf32>
    %v1548 = stablehlo.multiply %v1547, %v1547 : tensor<32x512x7x7xf32>
    %v1549 = stablehlo.reduce(%v1548 init: %v1542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1550 = stablehlo.broadcast_in_dim %v1549, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1551 = stablehlo.divide %v1550, %v1543 : tensor<32x512x7x7xf32>
    %v1552 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1553 = stablehlo.add %v1551, %v1552 : tensor<32x512x7x7xf32>
    %v1554 = stablehlo.rsqrt %v1553 : tensor<32x512x7x7xf32>
    %v1555 = stablehlo.multiply %v1547, %v1554 : tensor<32x512x7x7xf32>
    %v1556 = stablehlo.reshape %v1459 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1557 = stablehlo.multiply %v1556, %v1555 : tensor<32x512x7x7xf32>
    %v1558 = stablehlo.reduce(%v1557 init: %v1542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1559 = stablehlo.reshape %v1459 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1560 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1561 = stablehlo.reduce(%v1559 init: %v1560) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1562 = stablehlo.reshape %v917 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1563 = stablehlo.reshape %v1448 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1564 = stablehlo.transpose %v1562, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1565 = stablehlo.transpose %v1563, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1566 = stablehlo.convolution(%v1564, %v1565)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1567 = stablehlo.transpose %v1566, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1568 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1570 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1571 = stablehlo.reduce(%v1568 init: %v1569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1572 = stablehlo.broadcast_in_dim %v1571, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1573 = stablehlo.divide %v1572, %v1570 : tensor<32x512x7x7xf32>
    %v1574 = stablehlo.subtract %v1568, %v1573 : tensor<32x512x7x7xf32>
    %v1575 = stablehlo.multiply %v1574, %v1574 : tensor<32x512x7x7xf32>
    %v1576 = stablehlo.reduce(%v1575 init: %v1569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1577 = stablehlo.broadcast_in_dim %v1576, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1578 = stablehlo.divide %v1577, %v1570 : tensor<32x512x7x7xf32>
    %v1579 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1580 = stablehlo.add %v1578, %v1579 : tensor<32x512x7x7xf32>
    %v1581 = stablehlo.rsqrt %v1580 : tensor<32x512x7x7xf32>
    %v1582 = stablehlo.multiply %v1574, %v1581 : tensor<32x512x7x7xf32>
    %v1583 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1584 = stablehlo.multiply %v1583, %v1582 : tensor<32x512x7x7xf32>
    %v1585 = stablehlo.reduce(%v1584 init: %v1569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1586 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1587 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1588 = stablehlo.reduce(%v1586 init: %v1587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1589 = stablehlo.reshape %v888 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1590 = stablehlo.reshape %v1524 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1591 = stablehlo.transpose %v1589, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1593 = stablehlo.transpose %v1590, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1592 = stablehlo.slice %v1591 [0:256, 0:32, 0:13, 0:13] : (tensor<256x32x14x14xf32>) -> tensor<256x32x13x13xf32>
    %v1594 = stablehlo.convolution(%v1592, %v1593)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x13x13xf32>, tensor<512x32x7x7xf32>) -> tensor<256x512x1x1xf32>
    %v1595 = stablehlo.transpose %v1594, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v1596 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1598 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1599 = stablehlo.reduce(%v1596 init: %v1597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1600 = stablehlo.broadcast_in_dim %v1599, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1601 = stablehlo.divide %v1600, %v1598 : tensor<32x512x7x7xf32>
    %v1602 = stablehlo.subtract %v1596, %v1601 : tensor<32x512x7x7xf32>
    %v1603 = stablehlo.multiply %v1602, %v1602 : tensor<32x512x7x7xf32>
    %v1604 = stablehlo.reduce(%v1603 init: %v1597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1605 = stablehlo.broadcast_in_dim %v1604, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1606 = stablehlo.divide %v1605, %v1598 : tensor<32x512x7x7xf32>
    %v1607 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1608 = stablehlo.add %v1606, %v1607 : tensor<32x512x7x7xf32>
    %v1609 = stablehlo.rsqrt %v1608 : tensor<32x512x7x7xf32>
    %v1610 = stablehlo.multiply %v1602, %v1609 : tensor<32x512x7x7xf32>
    %v1611 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1612 = stablehlo.multiply %v1611, %v1610 : tensor<32x512x7x7xf32>
    %v1613 = stablehlo.reduce(%v1612 init: %v1597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1614 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1616 = stablehlo.reduce(%v1614 init: %v1615) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1617 = stablehlo.reshape %v1533 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1618 = stablehlo.reshape %v884 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1619 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1620 = stablehlo.compare GT, %v1618, %v1619 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1621 = stablehlo.select %v1620, %v1617, %v1619 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1622 = stablehlo.reshape %v1621 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1623 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1625 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1626 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1627 = stablehlo.reduce(%v1623 init: %v1624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1628 = stablehlo.broadcast_in_dim %v1627, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1629 = stablehlo.divide %v1628, %v1625 : tensor<32x256x14x14xf32>
    %v1630 = stablehlo.subtract %v1623, %v1629 : tensor<32x256x14x14xf32>
    %v1631 = stablehlo.multiply %v1630, %v1630 : tensor<32x256x14x14xf32>
    %v1632 = stablehlo.reduce(%v1631 init: %v1624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1633 = stablehlo.broadcast_in_dim %v1632, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1634 = stablehlo.divide %v1633, %v1625 : tensor<32x256x14x14xf32>
    %v1635 = stablehlo.add %v1634, %v1626 : tensor<32x256x14x14xf32>
    %v1636 = stablehlo.rsqrt %v1635 : tensor<32x256x14x14xf32>
    %v1637 = stablehlo.multiply %v1630, %v1636 : tensor<32x256x14x14xf32>
    %v1638 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1639 = stablehlo.reshape %v1622 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1640 = stablehlo.multiply %v1638, %v1639 : tensor<32x256x14x14xf32>
    %v1641 = stablehlo.reduce(%v1640 init: %v1624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1642 = stablehlo.broadcast_in_dim %v1641, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1643 = stablehlo.multiply %v1637, %v1640 : tensor<32x256x14x14xf32>
    %v1644 = stablehlo.reduce(%v1643 init: %v1624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1645 = stablehlo.broadcast_in_dim %v1644, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1646 = stablehlo.multiply %v1640, %v1625 : tensor<32x256x14x14xf32>
    %v1647 = stablehlo.subtract %v1646, %v1642 : tensor<32x256x14x14xf32>
    %v1648 = stablehlo.multiply %v1637, %v1645 : tensor<32x256x14x14xf32>
    %v1649 = stablehlo.subtract %v1647, %v1648 : tensor<32x256x14x14xf32>
    %v1650 = stablehlo.divide %v1636, %v1625 : tensor<32x256x14x14xf32>
    %v1651 = stablehlo.multiply %v1650, %v1649 : tensor<32x256x14x14xf32>
    %v1652 = stablehlo.reshape %v1651 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1653 = stablehlo.reshape %v1652 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1654 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1655 = stablehlo.transpose %v1654, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1656 = stablehlo.convolution(%v1653, %v1655)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1657 = stablehlo.reshape %v1656 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1658 = stablehlo.reshape %v1657 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1659 = stablehlo.reshape %v851 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1660 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1661 = stablehlo.compare GT, %v1659, %v1660 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1662 = stablehlo.select %v1661, %v1658, %v1660 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1664 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1666 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1667 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1668 = stablehlo.reduce(%v1664 init: %v1665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1669 = stablehlo.broadcast_in_dim %v1668, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1670 = stablehlo.divide %v1669, %v1666 : tensor<32x256x14x14xf32>
    %v1671 = stablehlo.subtract %v1664, %v1670 : tensor<32x256x14x14xf32>
    %v1672 = stablehlo.multiply %v1671, %v1671 : tensor<32x256x14x14xf32>
    %v1673 = stablehlo.reduce(%v1672 init: %v1665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1674 = stablehlo.broadcast_in_dim %v1673, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1675 = stablehlo.divide %v1674, %v1666 : tensor<32x256x14x14xf32>
    %v1676 = stablehlo.add %v1675, %v1667 : tensor<32x256x14x14xf32>
    %v1677 = stablehlo.rsqrt %v1676 : tensor<32x256x14x14xf32>
    %v1678 = stablehlo.multiply %v1671, %v1677 : tensor<32x256x14x14xf32>
    %v1679 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1680 = stablehlo.reshape %v1663 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1681 = stablehlo.multiply %v1679, %v1680 : tensor<32x256x14x14xf32>
    %v1682 = stablehlo.reduce(%v1681 init: %v1665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1683 = stablehlo.broadcast_in_dim %v1682, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1684 = stablehlo.multiply %v1678, %v1681 : tensor<32x256x14x14xf32>
    %v1685 = stablehlo.reduce(%v1684 init: %v1665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1686 = stablehlo.broadcast_in_dim %v1685, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1687 = stablehlo.multiply %v1681, %v1666 : tensor<32x256x14x14xf32>
    %v1688 = stablehlo.subtract %v1687, %v1683 : tensor<32x256x14x14xf32>
    %v1689 = stablehlo.multiply %v1678, %v1686 : tensor<32x256x14x14xf32>
    %v1690 = stablehlo.subtract %v1688, %v1689 : tensor<32x256x14x14xf32>
    %v1691 = stablehlo.divide %v1677, %v1666 : tensor<32x256x14x14xf32>
    %v1692 = stablehlo.multiply %v1691, %v1690 : tensor<32x256x14x14xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1694 = stablehlo.reshape %v1693 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1695 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1696 = stablehlo.transpose %v1695, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1697 = stablehlo.convolution(%v1694, %v1696)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1698 = stablehlo.reshape %v1697 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1699 = stablehlo.reshape %v1698 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1700 = stablehlo.reshape %v1622 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1701 = stablehlo.add %v1699, %v1700 : tensor<32x256x14x14xf32>
    %v1702 = stablehlo.reshape %v1701 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1703 = stablehlo.reshape %v826 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1704 = stablehlo.reshape %v1693 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1705 = stablehlo.transpose %v1703, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1706 = stablehlo.transpose %v1704, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1707 = stablehlo.convolution(%v1705, %v1706)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1708 = stablehlo.transpose %v1707, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1709 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1710 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1711 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1712 = stablehlo.reduce(%v1709 init: %v1710) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1713 = stablehlo.broadcast_in_dim %v1712, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1714 = stablehlo.divide %v1713, %v1711 : tensor<32x256x14x14xf32>
    %v1715 = stablehlo.subtract %v1709, %v1714 : tensor<32x256x14x14xf32>
    %v1716 = stablehlo.multiply %v1715, %v1715 : tensor<32x256x14x14xf32>
    %v1717 = stablehlo.reduce(%v1716 init: %v1710) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1718 = stablehlo.broadcast_in_dim %v1717, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1719 = stablehlo.divide %v1718, %v1711 : tensor<32x256x14x14xf32>
    %v1720 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1721 = stablehlo.add %v1719, %v1720 : tensor<32x256x14x14xf32>
    %v1722 = stablehlo.rsqrt %v1721 : tensor<32x256x14x14xf32>
    %v1723 = stablehlo.multiply %v1715, %v1722 : tensor<32x256x14x14xf32>
    %v1724 = stablehlo.reshape %v1663 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1725 = stablehlo.multiply %v1724, %v1723 : tensor<32x256x14x14xf32>
    %v1726 = stablehlo.reduce(%v1725 init: %v1710) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1727 = stablehlo.reshape %v1663 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1729 = stablehlo.reduce(%v1727 init: %v1728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1730 = stablehlo.reshape %v855 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1731 = stablehlo.reshape %v1652 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1732 = stablehlo.transpose %v1730, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1733 = stablehlo.transpose %v1731, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1734 = stablehlo.convolution(%v1732, %v1733)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1735 = stablehlo.transpose %v1734, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1736 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1737 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1738 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1739 = stablehlo.reduce(%v1736 init: %v1737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1740 = stablehlo.broadcast_in_dim %v1739, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1741 = stablehlo.divide %v1740, %v1738 : tensor<32x256x14x14xf32>
    %v1742 = stablehlo.subtract %v1736, %v1741 : tensor<32x256x14x14xf32>
    %v1743 = stablehlo.multiply %v1742, %v1742 : tensor<32x256x14x14xf32>
    %v1744 = stablehlo.reduce(%v1743 init: %v1737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1745 = stablehlo.broadcast_in_dim %v1744, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1746 = stablehlo.divide %v1745, %v1738 : tensor<32x256x14x14xf32>
    %v1747 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1748 = stablehlo.add %v1746, %v1747 : tensor<32x256x14x14xf32>
    %v1749 = stablehlo.rsqrt %v1748 : tensor<32x256x14x14xf32>
    %v1750 = stablehlo.multiply %v1742, %v1749 : tensor<32x256x14x14xf32>
    %v1751 = stablehlo.reshape %v1622 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1752 = stablehlo.multiply %v1751, %v1750 : tensor<32x256x14x14xf32>
    %v1753 = stablehlo.reduce(%v1752 init: %v1737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1754 = stablehlo.reshape %v1622 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1756 = stablehlo.reduce(%v1754 init: %v1755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1757 = stablehlo.reshape %v1702 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1758 = stablehlo.reshape %v822 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1759 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1760 = stablehlo.compare GT, %v1758, %v1759 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1761 = stablehlo.select %v1760, %v1757, %v1759 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1762 = stablehlo.reshape %v1761 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1763 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1764 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1765 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1766 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1767 = stablehlo.reduce(%v1763 init: %v1764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1768 = stablehlo.broadcast_in_dim %v1767, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1769 = stablehlo.divide %v1768, %v1765 : tensor<32x256x14x14xf32>
    %v1770 = stablehlo.subtract %v1763, %v1769 : tensor<32x256x14x14xf32>
    %v1771 = stablehlo.multiply %v1770, %v1770 : tensor<32x256x14x14xf32>
    %v1772 = stablehlo.reduce(%v1771 init: %v1764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1773 = stablehlo.broadcast_in_dim %v1772, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1774 = stablehlo.divide %v1773, %v1765 : tensor<32x256x14x14xf32>
    %v1775 = stablehlo.add %v1774, %v1766 : tensor<32x256x14x14xf32>
    %v1776 = stablehlo.rsqrt %v1775 : tensor<32x256x14x14xf32>
    %v1777 = stablehlo.multiply %v1770, %v1776 : tensor<32x256x14x14xf32>
    %v1778 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1779 = stablehlo.reshape %v1762 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1780 = stablehlo.multiply %v1778, %v1779 : tensor<32x256x14x14xf32>
    %v1781 = stablehlo.reduce(%v1780 init: %v1764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1782 = stablehlo.broadcast_in_dim %v1781, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1783 = stablehlo.multiply %v1777, %v1780 : tensor<32x256x14x14xf32>
    %v1784 = stablehlo.reduce(%v1783 init: %v1764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1785 = stablehlo.broadcast_in_dim %v1784, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1786 = stablehlo.multiply %v1780, %v1765 : tensor<32x256x14x14xf32>
    %v1787 = stablehlo.subtract %v1786, %v1782 : tensor<32x256x14x14xf32>
    %v1788 = stablehlo.multiply %v1777, %v1785 : tensor<32x256x14x14xf32>
    %v1789 = stablehlo.subtract %v1787, %v1788 : tensor<32x256x14x14xf32>
    %v1790 = stablehlo.divide %v1776, %v1765 : tensor<32x256x14x14xf32>
    %v1791 = stablehlo.multiply %v1790, %v1789 : tensor<32x256x14x14xf32>
    %v1792 = stablehlo.reshape %v1791 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1793 = stablehlo.reshape %v1792 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1794 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1795 = stablehlo.transpose %v1794, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1796 = stablehlo.convolution(%v1793, %v1795)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1797 = stablehlo.reshape %v1796 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1798 = stablehlo.reshape %v1797 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1799 = stablehlo.reshape %v789 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1800 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1801 = stablehlo.compare GT, %v1799, %v1800 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1802 = stablehlo.select %v1801, %v1798, %v1800 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1803 = stablehlo.reshape %v1802 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1804 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1805 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1806 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1807 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1808 = stablehlo.reduce(%v1804 init: %v1805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1809 = stablehlo.broadcast_in_dim %v1808, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1810 = stablehlo.divide %v1809, %v1806 : tensor<32x256x14x14xf32>
    %v1811 = stablehlo.subtract %v1804, %v1810 : tensor<32x256x14x14xf32>
    %v1812 = stablehlo.multiply %v1811, %v1811 : tensor<32x256x14x14xf32>
    %v1813 = stablehlo.reduce(%v1812 init: %v1805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1814 = stablehlo.broadcast_in_dim %v1813, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1815 = stablehlo.divide %v1814, %v1806 : tensor<32x256x14x14xf32>
    %v1816 = stablehlo.add %v1815, %v1807 : tensor<32x256x14x14xf32>
    %v1817 = stablehlo.rsqrt %v1816 : tensor<32x256x14x14xf32>
    %v1818 = stablehlo.multiply %v1811, %v1817 : tensor<32x256x14x14xf32>
    %v1819 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1820 = stablehlo.reshape %v1803 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1821 = stablehlo.multiply %v1819, %v1820 : tensor<32x256x14x14xf32>
    %v1822 = stablehlo.reduce(%v1821 init: %v1805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1823 = stablehlo.broadcast_in_dim %v1822, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1824 = stablehlo.multiply %v1818, %v1821 : tensor<32x256x14x14xf32>
    %v1825 = stablehlo.reduce(%v1824 init: %v1805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1826 = stablehlo.broadcast_in_dim %v1825, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1827 = stablehlo.multiply %v1821, %v1806 : tensor<32x256x14x14xf32>
    %v1828 = stablehlo.subtract %v1827, %v1823 : tensor<32x256x14x14xf32>
    %v1829 = stablehlo.multiply %v1818, %v1826 : tensor<32x256x14x14xf32>
    %v1830 = stablehlo.subtract %v1828, %v1829 : tensor<32x256x14x14xf32>
    %v1831 = stablehlo.divide %v1817, %v1806 : tensor<32x256x14x14xf32>
    %v1832 = stablehlo.multiply %v1831, %v1830 : tensor<32x256x14x14xf32>
    %v1833 = stablehlo.reshape %v1832 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1834 = stablehlo.reshape %v1833 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1835 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1836 = stablehlo.transpose %v1835, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1837 = stablehlo.convolution(%v1834, %v1836)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1838 = stablehlo.reshape %v1837 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1839 = stablehlo.reshape %v1838 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1840 = stablehlo.reshape %v1762 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1841 = stablehlo.add %v1839, %v1840 : tensor<32x256x14x14xf32>
    %v1842 = stablehlo.reshape %v1841 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1843 = stablehlo.reshape %v764 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1844 = stablehlo.reshape %v1833 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1845 = stablehlo.transpose %v1843, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1846 = stablehlo.transpose %v1844, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1847 = stablehlo.convolution(%v1845, %v1846)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1848 = stablehlo.transpose %v1847, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1849 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1851 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1852 = stablehlo.reduce(%v1849 init: %v1850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1853 = stablehlo.broadcast_in_dim %v1852, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1854 = stablehlo.divide %v1853, %v1851 : tensor<32x256x14x14xf32>
    %v1855 = stablehlo.subtract %v1849, %v1854 : tensor<32x256x14x14xf32>
    %v1856 = stablehlo.multiply %v1855, %v1855 : tensor<32x256x14x14xf32>
    %v1857 = stablehlo.reduce(%v1856 init: %v1850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1858 = stablehlo.broadcast_in_dim %v1857, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1859 = stablehlo.divide %v1858, %v1851 : tensor<32x256x14x14xf32>
    %v1860 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1861 = stablehlo.add %v1859, %v1860 : tensor<32x256x14x14xf32>
    %v1862 = stablehlo.rsqrt %v1861 : tensor<32x256x14x14xf32>
    %v1863 = stablehlo.multiply %v1855, %v1862 : tensor<32x256x14x14xf32>
    %v1864 = stablehlo.reshape %v1803 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1865 = stablehlo.multiply %v1864, %v1863 : tensor<32x256x14x14xf32>
    %v1866 = stablehlo.reduce(%v1865 init: %v1850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1867 = stablehlo.reshape %v1803 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1868 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1869 = stablehlo.reduce(%v1867 init: %v1868) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1870 = stablehlo.reshape %v793 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1871 = stablehlo.reshape %v1792 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1872 = stablehlo.transpose %v1870, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1873 = stablehlo.transpose %v1871, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1874 = stablehlo.convolution(%v1872, %v1873)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1875 = stablehlo.transpose %v1874, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1876 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1877 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1878 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1879 = stablehlo.reduce(%v1876 init: %v1877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1880 = stablehlo.broadcast_in_dim %v1879, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1881 = stablehlo.divide %v1880, %v1878 : tensor<32x256x14x14xf32>
    %v1882 = stablehlo.subtract %v1876, %v1881 : tensor<32x256x14x14xf32>
    %v1883 = stablehlo.multiply %v1882, %v1882 : tensor<32x256x14x14xf32>
    %v1884 = stablehlo.reduce(%v1883 init: %v1877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1885 = stablehlo.broadcast_in_dim %v1884, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1886 = stablehlo.divide %v1885, %v1878 : tensor<32x256x14x14xf32>
    %v1887 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1888 = stablehlo.add %v1886, %v1887 : tensor<32x256x14x14xf32>
    %v1889 = stablehlo.rsqrt %v1888 : tensor<32x256x14x14xf32>
    %v1890 = stablehlo.multiply %v1882, %v1889 : tensor<32x256x14x14xf32>
    %v1891 = stablehlo.reshape %v1762 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1892 = stablehlo.multiply %v1891, %v1890 : tensor<32x256x14x14xf32>
    %v1893 = stablehlo.reduce(%v1892 init: %v1877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1894 = stablehlo.reshape %v1762 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1896 = stablehlo.reduce(%v1894 init: %v1895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1897 = stablehlo.reshape %v1842 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1898 = stablehlo.reshape %v760 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1899 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1900 = stablehlo.compare GT, %v1898, %v1899 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1901 = stablehlo.select %v1900, %v1897, %v1899 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1902 = stablehlo.reshape %v1901 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1903 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1904 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1905 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1906 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1907 = stablehlo.reduce(%v1903 init: %v1904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1908 = stablehlo.broadcast_in_dim %v1907, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1909 = stablehlo.divide %v1908, %v1905 : tensor<32x256x14x14xf32>
    %v1910 = stablehlo.subtract %v1903, %v1909 : tensor<32x256x14x14xf32>
    %v1911 = stablehlo.multiply %v1910, %v1910 : tensor<32x256x14x14xf32>
    %v1912 = stablehlo.reduce(%v1911 init: %v1904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1913 = stablehlo.broadcast_in_dim %v1912, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1914 = stablehlo.divide %v1913, %v1905 : tensor<32x256x14x14xf32>
    %v1915 = stablehlo.add %v1914, %v1906 : tensor<32x256x14x14xf32>
    %v1916 = stablehlo.rsqrt %v1915 : tensor<32x256x14x14xf32>
    %v1917 = stablehlo.multiply %v1910, %v1916 : tensor<32x256x14x14xf32>
    %v1918 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1919 = stablehlo.reshape %v1902 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1920 = stablehlo.multiply %v1918, %v1919 : tensor<32x256x14x14xf32>
    %v1921 = stablehlo.reduce(%v1920 init: %v1904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1922 = stablehlo.broadcast_in_dim %v1921, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1923 = stablehlo.multiply %v1917, %v1920 : tensor<32x256x14x14xf32>
    %v1924 = stablehlo.reduce(%v1923 init: %v1904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1925 = stablehlo.broadcast_in_dim %v1924, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1926 = stablehlo.multiply %v1920, %v1905 : tensor<32x256x14x14xf32>
    %v1927 = stablehlo.subtract %v1926, %v1922 : tensor<32x256x14x14xf32>
    %v1928 = stablehlo.multiply %v1917, %v1925 : tensor<32x256x14x14xf32>
    %v1929 = stablehlo.subtract %v1927, %v1928 : tensor<32x256x14x14xf32>
    %v1930 = stablehlo.divide %v1916, %v1905 : tensor<32x256x14x14xf32>
    %v1931 = stablehlo.multiply %v1930, %v1929 : tensor<32x256x14x14xf32>
    %v1932 = stablehlo.reshape %v1931 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1933 = stablehlo.reshape %v1932 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1934 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1935 = stablehlo.transpose %v1934, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1936 = stablehlo.convolution(%v1933, %v1935)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1937 = stablehlo.reshape %v1936 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1938 = stablehlo.reshape %v1937 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1939 = stablehlo.reshape %v727 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1940 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1941 = stablehlo.compare GT, %v1939, %v1940 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1942 = stablehlo.select %v1941, %v1938, %v1940 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1943 = stablehlo.reshape %v1942 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1944 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1946 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1947 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1948 = stablehlo.reduce(%v1944 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1949 = stablehlo.broadcast_in_dim %v1948, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1950 = stablehlo.divide %v1949, %v1946 : tensor<32x256x14x14xf32>
    %v1951 = stablehlo.subtract %v1944, %v1950 : tensor<32x256x14x14xf32>
    %v1952 = stablehlo.multiply %v1951, %v1951 : tensor<32x256x14x14xf32>
    %v1953 = stablehlo.reduce(%v1952 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1954 = stablehlo.broadcast_in_dim %v1953, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1955 = stablehlo.divide %v1954, %v1946 : tensor<32x256x14x14xf32>
    %v1956 = stablehlo.add %v1955, %v1947 : tensor<32x256x14x14xf32>
    %v1957 = stablehlo.rsqrt %v1956 : tensor<32x256x14x14xf32>
    %v1958 = stablehlo.multiply %v1951, %v1957 : tensor<32x256x14x14xf32>
    %v1959 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1960 = stablehlo.reshape %v1943 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1961 = stablehlo.multiply %v1959, %v1960 : tensor<32x256x14x14xf32>
    %v1962 = stablehlo.reduce(%v1961 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1963 = stablehlo.broadcast_in_dim %v1962, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1964 = stablehlo.multiply %v1958, %v1961 : tensor<32x256x14x14xf32>
    %v1965 = stablehlo.reduce(%v1964 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1966 = stablehlo.broadcast_in_dim %v1965, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1967 = stablehlo.multiply %v1961, %v1946 : tensor<32x256x14x14xf32>
    %v1968 = stablehlo.subtract %v1967, %v1963 : tensor<32x256x14x14xf32>
    %v1969 = stablehlo.multiply %v1958, %v1966 : tensor<32x256x14x14xf32>
    %v1970 = stablehlo.subtract %v1968, %v1969 : tensor<32x256x14x14xf32>
    %v1971 = stablehlo.divide %v1957, %v1946 : tensor<32x256x14x14xf32>
    %v1972 = stablehlo.multiply %v1971, %v1970 : tensor<32x256x14x14xf32>
    %v1973 = stablehlo.reshape %v1972 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1974 = stablehlo.reshape %v1973 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1975 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1976 = stablehlo.transpose %v1975, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1977 = stablehlo.convolution(%v1974, %v1976)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1978 = stablehlo.reshape %v1977 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1979 = stablehlo.reshape %v1978 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1980 = stablehlo.reshape %v1902 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1981 = stablehlo.add %v1979, %v1980 : tensor<32x256x14x14xf32>
    %v1982 = stablehlo.reshape %v1981 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1983 = stablehlo.reshape %v702 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1984 = stablehlo.reshape %v1973 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1985 = stablehlo.transpose %v1983, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1986 = stablehlo.transpose %v1984, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1987 = stablehlo.convolution(%v1985, %v1986)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1988 = stablehlo.transpose %v1987, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1989 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1991 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1992 = stablehlo.reduce(%v1989 init: %v1990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1993 = stablehlo.broadcast_in_dim %v1992, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1994 = stablehlo.divide %v1993, %v1991 : tensor<32x256x14x14xf32>
    %v1995 = stablehlo.subtract %v1989, %v1994 : tensor<32x256x14x14xf32>
    %v1996 = stablehlo.multiply %v1995, %v1995 : tensor<32x256x14x14xf32>
    %v1997 = stablehlo.reduce(%v1996 init: %v1990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1998 = stablehlo.broadcast_in_dim %v1997, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1999 = stablehlo.divide %v1998, %v1991 : tensor<32x256x14x14xf32>
    %v2000 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2001 = stablehlo.add %v1999, %v2000 : tensor<32x256x14x14xf32>
    %v2002 = stablehlo.rsqrt %v2001 : tensor<32x256x14x14xf32>
    %v2003 = stablehlo.multiply %v1995, %v2002 : tensor<32x256x14x14xf32>
    %v2004 = stablehlo.reshape %v1943 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2005 = stablehlo.multiply %v2004, %v2003 : tensor<32x256x14x14xf32>
    %v2006 = stablehlo.reduce(%v2005 init: %v1990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2007 = stablehlo.reshape %v1943 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2008 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2009 = stablehlo.reduce(%v2007 init: %v2008) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2010 = stablehlo.reshape %v731 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2011 = stablehlo.reshape %v1932 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2012 = stablehlo.transpose %v2010, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2013 = stablehlo.transpose %v2011, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2014 = stablehlo.convolution(%v2012, %v2013)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2015 = stablehlo.transpose %v2014, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2016 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2017 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2018 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2019 = stablehlo.reduce(%v2016 init: %v2017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2020 = stablehlo.broadcast_in_dim %v2019, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2021 = stablehlo.divide %v2020, %v2018 : tensor<32x256x14x14xf32>
    %v2022 = stablehlo.subtract %v2016, %v2021 : tensor<32x256x14x14xf32>
    %v2023 = stablehlo.multiply %v2022, %v2022 : tensor<32x256x14x14xf32>
    %v2024 = stablehlo.reduce(%v2023 init: %v2017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2025 = stablehlo.broadcast_in_dim %v2024, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2026 = stablehlo.divide %v2025, %v2018 : tensor<32x256x14x14xf32>
    %v2027 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2028 = stablehlo.add %v2026, %v2027 : tensor<32x256x14x14xf32>
    %v2029 = stablehlo.rsqrt %v2028 : tensor<32x256x14x14xf32>
    %v2030 = stablehlo.multiply %v2022, %v2029 : tensor<32x256x14x14xf32>
    %v2031 = stablehlo.reshape %v1902 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2032 = stablehlo.multiply %v2031, %v2030 : tensor<32x256x14x14xf32>
    %v2033 = stablehlo.reduce(%v2032 init: %v2017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2034 = stablehlo.reshape %v1902 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2035 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2036 = stablehlo.reduce(%v2034 init: %v2035) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2037 = stablehlo.reshape %v1982 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2038 = stablehlo.reshape %v698 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2039 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2040 = stablehlo.compare GT, %v2038, %v2039 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2041 = stablehlo.select %v2040, %v2037, %v2039 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2042 = stablehlo.reshape %v2041 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2043 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2045 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2046 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2047 = stablehlo.reduce(%v2043 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2048 = stablehlo.broadcast_in_dim %v2047, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2049 = stablehlo.divide %v2048, %v2045 : tensor<32x256x14x14xf32>
    %v2050 = stablehlo.subtract %v2043, %v2049 : tensor<32x256x14x14xf32>
    %v2051 = stablehlo.multiply %v2050, %v2050 : tensor<32x256x14x14xf32>
    %v2052 = stablehlo.reduce(%v2051 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2053 = stablehlo.broadcast_in_dim %v2052, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2054 = stablehlo.divide %v2053, %v2045 : tensor<32x256x14x14xf32>
    %v2055 = stablehlo.add %v2054, %v2046 : tensor<32x256x14x14xf32>
    %v2056 = stablehlo.rsqrt %v2055 : tensor<32x256x14x14xf32>
    %v2057 = stablehlo.multiply %v2050, %v2056 : tensor<32x256x14x14xf32>
    %v2058 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2059 = stablehlo.reshape %v2042 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2060 = stablehlo.multiply %v2058, %v2059 : tensor<32x256x14x14xf32>
    %v2061 = stablehlo.reduce(%v2060 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2062 = stablehlo.broadcast_in_dim %v2061, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2063 = stablehlo.multiply %v2057, %v2060 : tensor<32x256x14x14xf32>
    %v2064 = stablehlo.reduce(%v2063 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2065 = stablehlo.broadcast_in_dim %v2064, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2066 = stablehlo.multiply %v2060, %v2045 : tensor<32x256x14x14xf32>
    %v2067 = stablehlo.subtract %v2066, %v2062 : tensor<32x256x14x14xf32>
    %v2068 = stablehlo.multiply %v2057, %v2065 : tensor<32x256x14x14xf32>
    %v2069 = stablehlo.subtract %v2067, %v2068 : tensor<32x256x14x14xf32>
    %v2070 = stablehlo.divide %v2056, %v2045 : tensor<32x256x14x14xf32>
    %v2071 = stablehlo.multiply %v2070, %v2069 : tensor<32x256x14x14xf32>
    %v2072 = stablehlo.reshape %v2071 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2073 = stablehlo.reshape %v2072 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2074 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2075 = stablehlo.transpose %v2074, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2076 = stablehlo.convolution(%v2073, %v2075)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2077 = stablehlo.reshape %v2076 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2078 = stablehlo.reshape %v2077 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2079 = stablehlo.reshape %v665 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2080 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2081 = stablehlo.compare GT, %v2079, %v2080 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2082 = stablehlo.select %v2081, %v2078, %v2080 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2083 = stablehlo.reshape %v2082 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2084 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2086 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2087 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2088 = stablehlo.reduce(%v2084 init: %v2085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2089 = stablehlo.broadcast_in_dim %v2088, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2090 = stablehlo.divide %v2089, %v2086 : tensor<32x256x14x14xf32>
    %v2091 = stablehlo.subtract %v2084, %v2090 : tensor<32x256x14x14xf32>
    %v2092 = stablehlo.multiply %v2091, %v2091 : tensor<32x256x14x14xf32>
    %v2093 = stablehlo.reduce(%v2092 init: %v2085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2094 = stablehlo.broadcast_in_dim %v2093, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2095 = stablehlo.divide %v2094, %v2086 : tensor<32x256x14x14xf32>
    %v2096 = stablehlo.add %v2095, %v2087 : tensor<32x256x14x14xf32>
    %v2097 = stablehlo.rsqrt %v2096 : tensor<32x256x14x14xf32>
    %v2098 = stablehlo.multiply %v2091, %v2097 : tensor<32x256x14x14xf32>
    %v2099 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2100 = stablehlo.reshape %v2083 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2101 = stablehlo.multiply %v2099, %v2100 : tensor<32x256x14x14xf32>
    %v2102 = stablehlo.reduce(%v2101 init: %v2085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2103 = stablehlo.broadcast_in_dim %v2102, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2104 = stablehlo.multiply %v2098, %v2101 : tensor<32x256x14x14xf32>
    %v2105 = stablehlo.reduce(%v2104 init: %v2085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2106 = stablehlo.broadcast_in_dim %v2105, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2107 = stablehlo.multiply %v2101, %v2086 : tensor<32x256x14x14xf32>
    %v2108 = stablehlo.subtract %v2107, %v2103 : tensor<32x256x14x14xf32>
    %v2109 = stablehlo.multiply %v2098, %v2106 : tensor<32x256x14x14xf32>
    %v2110 = stablehlo.subtract %v2108, %v2109 : tensor<32x256x14x14xf32>
    %v2111 = stablehlo.divide %v2097, %v2086 : tensor<32x256x14x14xf32>
    %v2112 = stablehlo.multiply %v2111, %v2110 : tensor<32x256x14x14xf32>
    %v2113 = stablehlo.reshape %v2112 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2114 = stablehlo.reshape %v2113 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2115 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2116 = stablehlo.transpose %v2115, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2117 = stablehlo.convolution(%v2114, %v2116)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2118 = stablehlo.reshape %v2117 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2120 = stablehlo.reshape %v2042 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2121 = stablehlo.add %v2119, %v2120 : tensor<32x256x14x14xf32>
    %v2122 = stablehlo.reshape %v2121 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2123 = stablehlo.reshape %v640 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2124 = stablehlo.reshape %v2113 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2125 = stablehlo.transpose %v2123, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2126 = stablehlo.transpose %v2124, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2127 = stablehlo.convolution(%v2125, %v2126)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2128 = stablehlo.transpose %v2127, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2129 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2131 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2132 = stablehlo.reduce(%v2129 init: %v2130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2133 = stablehlo.broadcast_in_dim %v2132, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2134 = stablehlo.divide %v2133, %v2131 : tensor<32x256x14x14xf32>
    %v2135 = stablehlo.subtract %v2129, %v2134 : tensor<32x256x14x14xf32>
    %v2136 = stablehlo.multiply %v2135, %v2135 : tensor<32x256x14x14xf32>
    %v2137 = stablehlo.reduce(%v2136 init: %v2130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2138 = stablehlo.broadcast_in_dim %v2137, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2139 = stablehlo.divide %v2138, %v2131 : tensor<32x256x14x14xf32>
    %v2140 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2141 = stablehlo.add %v2139, %v2140 : tensor<32x256x14x14xf32>
    %v2142 = stablehlo.rsqrt %v2141 : tensor<32x256x14x14xf32>
    %v2143 = stablehlo.multiply %v2135, %v2142 : tensor<32x256x14x14xf32>
    %v2144 = stablehlo.reshape %v2083 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2145 = stablehlo.multiply %v2144, %v2143 : tensor<32x256x14x14xf32>
    %v2146 = stablehlo.reduce(%v2145 init: %v2130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2147 = stablehlo.reshape %v2083 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2148 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2149 = stablehlo.reduce(%v2147 init: %v2148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2150 = stablehlo.reshape %v669 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2151 = stablehlo.reshape %v2072 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2152 = stablehlo.transpose %v2150, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2153 = stablehlo.transpose %v2151, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2154 = stablehlo.convolution(%v2152, %v2153)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2155 = stablehlo.transpose %v2154, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2156 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2157 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2158 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2159 = stablehlo.reduce(%v2156 init: %v2157) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2160 = stablehlo.broadcast_in_dim %v2159, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2161 = stablehlo.divide %v2160, %v2158 : tensor<32x256x14x14xf32>
    %v2162 = stablehlo.subtract %v2156, %v2161 : tensor<32x256x14x14xf32>
    %v2163 = stablehlo.multiply %v2162, %v2162 : tensor<32x256x14x14xf32>
    %v2164 = stablehlo.reduce(%v2163 init: %v2157) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2165 = stablehlo.broadcast_in_dim %v2164, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2166 = stablehlo.divide %v2165, %v2158 : tensor<32x256x14x14xf32>
    %v2167 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2168 = stablehlo.add %v2166, %v2167 : tensor<32x256x14x14xf32>
    %v2169 = stablehlo.rsqrt %v2168 : tensor<32x256x14x14xf32>
    %v2170 = stablehlo.multiply %v2162, %v2169 : tensor<32x256x14x14xf32>
    %v2171 = stablehlo.reshape %v2042 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2172 = stablehlo.multiply %v2171, %v2170 : tensor<32x256x14x14xf32>
    %v2173 = stablehlo.reduce(%v2172 init: %v2157) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2174 = stablehlo.reshape %v2042 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2175 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2176 = stablehlo.reduce(%v2174 init: %v2175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2177 = stablehlo.reshape %v2122 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2178 = stablehlo.reshape %v636 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2179 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2180 = stablehlo.compare GT, %v2178, %v2179 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2181 = stablehlo.select %v2180, %v2177, %v2179 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2182 = stablehlo.reshape %v2181 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2183 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2184 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2185 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2186 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2187 = stablehlo.reduce(%v2183 init: %v2184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2188 = stablehlo.broadcast_in_dim %v2187, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2189 = stablehlo.divide %v2188, %v2185 : tensor<32x256x14x14xf32>
    %v2190 = stablehlo.subtract %v2183, %v2189 : tensor<32x256x14x14xf32>
    %v2191 = stablehlo.multiply %v2190, %v2190 : tensor<32x256x14x14xf32>
    %v2192 = stablehlo.reduce(%v2191 init: %v2184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2193 = stablehlo.broadcast_in_dim %v2192, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2194 = stablehlo.divide %v2193, %v2185 : tensor<32x256x14x14xf32>
    %v2195 = stablehlo.add %v2194, %v2186 : tensor<32x256x14x14xf32>
    %v2196 = stablehlo.rsqrt %v2195 : tensor<32x256x14x14xf32>
    %v2197 = stablehlo.multiply %v2190, %v2196 : tensor<32x256x14x14xf32>
    %v2198 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2199 = stablehlo.reshape %v2182 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2200 = stablehlo.multiply %v2198, %v2199 : tensor<32x256x14x14xf32>
    %v2201 = stablehlo.reduce(%v2200 init: %v2184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2202 = stablehlo.broadcast_in_dim %v2201, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2203 = stablehlo.multiply %v2197, %v2200 : tensor<32x256x14x14xf32>
    %v2204 = stablehlo.reduce(%v2203 init: %v2184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2205 = stablehlo.broadcast_in_dim %v2204, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2206 = stablehlo.multiply %v2200, %v2185 : tensor<32x256x14x14xf32>
    %v2207 = stablehlo.subtract %v2206, %v2202 : tensor<32x256x14x14xf32>
    %v2208 = stablehlo.multiply %v2197, %v2205 : tensor<32x256x14x14xf32>
    %v2209 = stablehlo.subtract %v2207, %v2208 : tensor<32x256x14x14xf32>
    %v2210 = stablehlo.divide %v2196, %v2185 : tensor<32x256x14x14xf32>
    %v2211 = stablehlo.multiply %v2210, %v2209 : tensor<32x256x14x14xf32>
    %v2212 = stablehlo.reshape %v2211 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2213 = stablehlo.reshape %v2212 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2214 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2215 = stablehlo.transpose %v2214, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2216 = stablehlo.convolution(%v2213, %v2215)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2217 = stablehlo.reshape %v2216 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2218 = stablehlo.reshape %v2217 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2219 = stablehlo.reshape %v603 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2220 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2221 = stablehlo.compare GT, %v2219, %v2220 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2222 = stablehlo.select %v2221, %v2218, %v2220 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2223 = stablehlo.reshape %v2222 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2224 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2226 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2227 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2228 = stablehlo.reduce(%v2224 init: %v2225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2229 = stablehlo.broadcast_in_dim %v2228, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2230 = stablehlo.divide %v2229, %v2226 : tensor<32x256x14x14xf32>
    %v2231 = stablehlo.subtract %v2224, %v2230 : tensor<32x256x14x14xf32>
    %v2232 = stablehlo.multiply %v2231, %v2231 : tensor<32x256x14x14xf32>
    %v2233 = stablehlo.reduce(%v2232 init: %v2225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2234 = stablehlo.broadcast_in_dim %v2233, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2235 = stablehlo.divide %v2234, %v2226 : tensor<32x256x14x14xf32>
    %v2236 = stablehlo.add %v2235, %v2227 : tensor<32x256x14x14xf32>
    %v2237 = stablehlo.rsqrt %v2236 : tensor<32x256x14x14xf32>
    %v2238 = stablehlo.multiply %v2231, %v2237 : tensor<32x256x14x14xf32>
    %v2239 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2240 = stablehlo.reshape %v2223 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2241 = stablehlo.multiply %v2239, %v2240 : tensor<32x256x14x14xf32>
    %v2242 = stablehlo.reduce(%v2241 init: %v2225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2243 = stablehlo.broadcast_in_dim %v2242, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2244 = stablehlo.multiply %v2238, %v2241 : tensor<32x256x14x14xf32>
    %v2245 = stablehlo.reduce(%v2244 init: %v2225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2246 = stablehlo.broadcast_in_dim %v2245, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2247 = stablehlo.multiply %v2241, %v2226 : tensor<32x256x14x14xf32>
    %v2248 = stablehlo.subtract %v2247, %v2243 : tensor<32x256x14x14xf32>
    %v2249 = stablehlo.multiply %v2238, %v2246 : tensor<32x256x14x14xf32>
    %v2250 = stablehlo.subtract %v2248, %v2249 : tensor<32x256x14x14xf32>
    %v2251 = stablehlo.divide %v2237, %v2226 : tensor<32x256x14x14xf32>
    %v2252 = stablehlo.multiply %v2251, %v2250 : tensor<32x256x14x14xf32>
    %v2253 = stablehlo.reshape %v2252 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2254 = stablehlo.reshape %v2253 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2255 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2256 = stablehlo.transpose %v2255, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2257 = stablehlo.convolution(%v2254, %v2256)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2258 = stablehlo.reshape %v2257 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2260 = stablehlo.reshape %v2182 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2261 = stablehlo.add %v2259, %v2260 : tensor<32x256x14x14xf32>
    %v2262 = stablehlo.reshape %v2261 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2263 = stablehlo.reshape %v578 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2264 = stablehlo.reshape %v2253 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2265 = stablehlo.transpose %v2263, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2266 = stablehlo.transpose %v2264, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2267 = stablehlo.convolution(%v2265, %v2266)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2268 = stablehlo.transpose %v2267, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2269 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2270 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2271 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2272 = stablehlo.reduce(%v2269 init: %v2270) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2273 = stablehlo.broadcast_in_dim %v2272, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2274 = stablehlo.divide %v2273, %v2271 : tensor<32x256x14x14xf32>
    %v2275 = stablehlo.subtract %v2269, %v2274 : tensor<32x256x14x14xf32>
    %v2276 = stablehlo.multiply %v2275, %v2275 : tensor<32x256x14x14xf32>
    %v2277 = stablehlo.reduce(%v2276 init: %v2270) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2278 = stablehlo.broadcast_in_dim %v2277, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2279 = stablehlo.divide %v2278, %v2271 : tensor<32x256x14x14xf32>
    %v2280 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2281 = stablehlo.add %v2279, %v2280 : tensor<32x256x14x14xf32>
    %v2282 = stablehlo.rsqrt %v2281 : tensor<32x256x14x14xf32>
    %v2283 = stablehlo.multiply %v2275, %v2282 : tensor<32x256x14x14xf32>
    %v2284 = stablehlo.reshape %v2223 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2285 = stablehlo.multiply %v2284, %v2283 : tensor<32x256x14x14xf32>
    %v2286 = stablehlo.reduce(%v2285 init: %v2270) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2287 = stablehlo.reshape %v2223 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2289 = stablehlo.reduce(%v2287 init: %v2288) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2290 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2291 = stablehlo.reshape %v2212 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2292 = stablehlo.transpose %v2290, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2293 = stablehlo.transpose %v2291, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2294 = stablehlo.convolution(%v2292, %v2293)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2295 = stablehlo.transpose %v2294, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2296 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2298 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2299 = stablehlo.reduce(%v2296 init: %v2297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2300 = stablehlo.broadcast_in_dim %v2299, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2301 = stablehlo.divide %v2300, %v2298 : tensor<32x256x14x14xf32>
    %v2302 = stablehlo.subtract %v2296, %v2301 : tensor<32x256x14x14xf32>
    %v2303 = stablehlo.multiply %v2302, %v2302 : tensor<32x256x14x14xf32>
    %v2304 = stablehlo.reduce(%v2303 init: %v2297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2305 = stablehlo.broadcast_in_dim %v2304, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2306 = stablehlo.divide %v2305, %v2298 : tensor<32x256x14x14xf32>
    %v2307 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2308 = stablehlo.add %v2306, %v2307 : tensor<32x256x14x14xf32>
    %v2309 = stablehlo.rsqrt %v2308 : tensor<32x256x14x14xf32>
    %v2310 = stablehlo.multiply %v2302, %v2309 : tensor<32x256x14x14xf32>
    %v2311 = stablehlo.reshape %v2182 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2312 = stablehlo.multiply %v2311, %v2310 : tensor<32x256x14x14xf32>
    %v2313 = stablehlo.reduce(%v2312 init: %v2297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2314 = stablehlo.reshape %v2182 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2316 = stablehlo.reduce(%v2314 init: %v2315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2317 = stablehlo.reshape %v2262 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2318 = stablehlo.reshape %v574 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2319 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2320 = stablehlo.compare GT, %v2318, %v2319 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2321 = stablehlo.select %v2320, %v2317, %v2319 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2322 = stablehlo.reshape %v2321 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2323 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2324 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2325 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2326 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2327 = stablehlo.reduce(%v2323 init: %v2324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2328 = stablehlo.broadcast_in_dim %v2327, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2329 = stablehlo.divide %v2328, %v2325 : tensor<32x256x14x14xf32>
    %v2330 = stablehlo.subtract %v2323, %v2329 : tensor<32x256x14x14xf32>
    %v2331 = stablehlo.multiply %v2330, %v2330 : tensor<32x256x14x14xf32>
    %v2332 = stablehlo.reduce(%v2331 init: %v2324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2333 = stablehlo.broadcast_in_dim %v2332, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2334 = stablehlo.divide %v2333, %v2325 : tensor<32x256x14x14xf32>
    %v2335 = stablehlo.add %v2334, %v2326 : tensor<32x256x14x14xf32>
    %v2336 = stablehlo.rsqrt %v2335 : tensor<32x256x14x14xf32>
    %v2337 = stablehlo.multiply %v2330, %v2336 : tensor<32x256x14x14xf32>
    %v2338 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2339 = stablehlo.reshape %v2322 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2340 = stablehlo.multiply %v2338, %v2339 : tensor<32x256x14x14xf32>
    %v2341 = stablehlo.reduce(%v2340 init: %v2324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2342 = stablehlo.broadcast_in_dim %v2341, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2343 = stablehlo.multiply %v2337, %v2340 : tensor<32x256x14x14xf32>
    %v2344 = stablehlo.reduce(%v2343 init: %v2324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2345 = stablehlo.broadcast_in_dim %v2344, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2346 = stablehlo.multiply %v2340, %v2325 : tensor<32x256x14x14xf32>
    %v2347 = stablehlo.subtract %v2346, %v2342 : tensor<32x256x14x14xf32>
    %v2348 = stablehlo.multiply %v2337, %v2345 : tensor<32x256x14x14xf32>
    %v2349 = stablehlo.subtract %v2347, %v2348 : tensor<32x256x14x14xf32>
    %v2350 = stablehlo.divide %v2336, %v2325 : tensor<32x256x14x14xf32>
    %v2351 = stablehlo.multiply %v2350, %v2349 : tensor<32x256x14x14xf32>
    %v2352 = stablehlo.reshape %v2351 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2353 = stablehlo.reshape %v2352 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2354 = stablehlo.reverse %d3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2355 = stablehlo.transpose %v2354, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2356 = stablehlo.convolution(%v2353, %v2355)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2357 = stablehlo.reshape %v2356 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2358 = stablehlo.reshape %v2357 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2359 = stablehlo.reshape %v516 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2360 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2361 = stablehlo.compare GT, %v2359, %v2360 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2362 = stablehlo.select %v2361, %v2358, %v2360 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2363 = stablehlo.reshape %v2362 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2364 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2366 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2367 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2368 = stablehlo.reduce(%v2364 init: %v2365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2369 = stablehlo.broadcast_in_dim %v2368, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2370 = stablehlo.divide %v2369, %v2366 : tensor<32x256x14x14xf32>
    %v2371 = stablehlo.subtract %v2364, %v2370 : tensor<32x256x14x14xf32>
    %v2372 = stablehlo.multiply %v2371, %v2371 : tensor<32x256x14x14xf32>
    %v2373 = stablehlo.reduce(%v2372 init: %v2365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2374 = stablehlo.broadcast_in_dim %v2373, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2375 = stablehlo.divide %v2374, %v2366 : tensor<32x256x14x14xf32>
    %v2376 = stablehlo.add %v2375, %v2367 : tensor<32x256x14x14xf32>
    %v2377 = stablehlo.rsqrt %v2376 : tensor<32x256x14x14xf32>
    %v2378 = stablehlo.multiply %v2371, %v2377 : tensor<32x256x14x14xf32>
    %v2379 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2380 = stablehlo.reshape %v2363 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2381 = stablehlo.multiply %v2379, %v2380 : tensor<32x256x14x14xf32>
    %v2382 = stablehlo.reduce(%v2381 init: %v2365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2383 = stablehlo.broadcast_in_dim %v2382, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2384 = stablehlo.multiply %v2378, %v2381 : tensor<32x256x14x14xf32>
    %v2385 = stablehlo.reduce(%v2384 init: %v2365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2386 = stablehlo.broadcast_in_dim %v2385, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2387 = stablehlo.multiply %v2381, %v2366 : tensor<32x256x14x14xf32>
    %v2388 = stablehlo.subtract %v2387, %v2383 : tensor<32x256x14x14xf32>
    %v2389 = stablehlo.multiply %v2378, %v2386 : tensor<32x256x14x14xf32>
    %v2390 = stablehlo.subtract %v2388, %v2389 : tensor<32x256x14x14xf32>
    %v2391 = stablehlo.divide %v2377, %v2366 : tensor<32x256x14x14xf32>
    %v2392 = stablehlo.multiply %v2391, %v2390 : tensor<32x256x14x14xf32>
    %v2393 = stablehlo.reshape %v2392 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2394 = stablehlo.reshape %v2393 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2395 = stablehlo.reverse %d3W1, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v2396 = stablehlo.transpose %v2395, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2397 = stablehlo.convolution(%v2394, %v2396)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2398 = stablehlo.reshape %v2397 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2399 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2401 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2402 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2403 = stablehlo.reduce(%v2399 init: %v2400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2404 = stablehlo.broadcast_in_dim %v2403, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2405 = stablehlo.divide %v2404, %v2401 : tensor<32x256x14x14xf32>
    %v2406 = stablehlo.subtract %v2399, %v2405 : tensor<32x256x14x14xf32>
    %v2407 = stablehlo.multiply %v2406, %v2406 : tensor<32x256x14x14xf32>
    %v2408 = stablehlo.reduce(%v2407 init: %v2400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2409 = stablehlo.broadcast_in_dim %v2408, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2410 = stablehlo.divide %v2409, %v2401 : tensor<32x256x14x14xf32>
    %v2411 = stablehlo.add %v2410, %v2402 : tensor<32x256x14x14xf32>
    %v2412 = stablehlo.rsqrt %v2411 : tensor<32x256x14x14xf32>
    %v2413 = stablehlo.multiply %v2406, %v2412 : tensor<32x256x14x14xf32>
    %v2414 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2415 = stablehlo.reshape %v2322 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2416 = stablehlo.multiply %v2414, %v2415 : tensor<32x256x14x14xf32>
    %v2417 = stablehlo.reduce(%v2416 init: %v2400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2418 = stablehlo.broadcast_in_dim %v2417, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2419 = stablehlo.multiply %v2413, %v2416 : tensor<32x256x14x14xf32>
    %v2420 = stablehlo.reduce(%v2419 init: %v2400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2421 = stablehlo.broadcast_in_dim %v2420, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2422 = stablehlo.multiply %v2416, %v2401 : tensor<32x256x14x14xf32>
    %v2423 = stablehlo.subtract %v2422, %v2418 : tensor<32x256x14x14xf32>
    %v2424 = stablehlo.multiply %v2413, %v2421 : tensor<32x256x14x14xf32>
    %v2425 = stablehlo.subtract %v2423, %v2424 : tensor<32x256x14x14xf32>
    %v2426 = stablehlo.divide %v2412, %v2401 : tensor<32x256x14x14xf32>
    %v2427 = stablehlo.multiply %v2426, %v2425 : tensor<32x256x14x14xf32>
    %v2428 = stablehlo.reshape %v2427 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2429 = stablehlo.reshape %v2428 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2430 = stablehlo.reverse %d3Wp, dims = [2, 3] : tensor<256x128x1x1xf32>
    %v2431 = stablehlo.transpose %v2430, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v2432 = stablehlo.convolution(%v2429, %v2431)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<128x256x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v2433 = stablehlo.reshape %v2432 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2434 = stablehlo.reshape %v2398 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2435 = stablehlo.reshape %v2433 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2436 = stablehlo.add %v2434, %v2435 : tensor<32x128x28x28xf32>
    %v2437 = stablehlo.reshape %v2436 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2438 = stablehlo.reshape %v491 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2439 = stablehlo.reshape %v2393 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2440 = stablehlo.transpose %v2438, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2442 = stablehlo.transpose %v2439, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2443 = stablehlo.convolution(%v2440, %v2442)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x14x14xf32>) -> tensor<128x256x3x3xf32>
    %v2444 = stablehlo.transpose %v2443, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2445 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2446 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2447 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2448 = stablehlo.reduce(%v2445 init: %v2446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2449 = stablehlo.broadcast_in_dim %v2448, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2450 = stablehlo.divide %v2449, %v2447 : tensor<32x256x14x14xf32>
    %v2451 = stablehlo.subtract %v2445, %v2450 : tensor<32x256x14x14xf32>
    %v2452 = stablehlo.multiply %v2451, %v2451 : tensor<32x256x14x14xf32>
    %v2453 = stablehlo.reduce(%v2452 init: %v2446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2454 = stablehlo.broadcast_in_dim %v2453, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2455 = stablehlo.divide %v2454, %v2447 : tensor<32x256x14x14xf32>
    %v2456 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2457 = stablehlo.add %v2455, %v2456 : tensor<32x256x14x14xf32>
    %v2458 = stablehlo.rsqrt %v2457 : tensor<32x256x14x14xf32>
    %v2459 = stablehlo.multiply %v2451, %v2458 : tensor<32x256x14x14xf32>
    %v2460 = stablehlo.reshape %v2363 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2461 = stablehlo.multiply %v2460, %v2459 : tensor<32x256x14x14xf32>
    %v2462 = stablehlo.reduce(%v2461 init: %v2446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2463 = stablehlo.reshape %v2363 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2464 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2465 = stablehlo.reduce(%v2463 init: %v2464) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2466 = stablehlo.reshape %v520 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2467 = stablehlo.reshape %v2352 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2468 = stablehlo.transpose %v2466, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2469 = stablehlo.transpose %v2467, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2470 = stablehlo.convolution(%v2468, %v2469)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2471 = stablehlo.transpose %v2470, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2472 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2474 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2475 = stablehlo.reduce(%v2472 init: %v2473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2476 = stablehlo.broadcast_in_dim %v2475, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2477 = stablehlo.divide %v2476, %v2474 : tensor<32x256x14x14xf32>
    %v2478 = stablehlo.subtract %v2472, %v2477 : tensor<32x256x14x14xf32>
    %v2479 = stablehlo.multiply %v2478, %v2478 : tensor<32x256x14x14xf32>
    %v2480 = stablehlo.reduce(%v2479 init: %v2473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2481 = stablehlo.broadcast_in_dim %v2480, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2482 = stablehlo.divide %v2481, %v2474 : tensor<32x256x14x14xf32>
    %v2483 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2484 = stablehlo.add %v2482, %v2483 : tensor<32x256x14x14xf32>
    %v2485 = stablehlo.rsqrt %v2484 : tensor<32x256x14x14xf32>
    %v2486 = stablehlo.multiply %v2478, %v2485 : tensor<32x256x14x14xf32>
    %v2487 = stablehlo.reshape %v2322 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2488 = stablehlo.multiply %v2487, %v2486 : tensor<32x256x14x14xf32>
    %v2489 = stablehlo.reduce(%v2488 init: %v2473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2490 = stablehlo.reshape %v2322 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2492 = stablehlo.reduce(%v2490 init: %v2491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2493 = stablehlo.reshape %v491 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2494 = stablehlo.reshape %v2428 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2495 = stablehlo.transpose %v2493, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2497 = stablehlo.transpose %v2494, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2496 = stablehlo.slice %v2495 [0:128, 0:32, 0:27, 0:27] : (tensor<128x32x28x28xf32>) -> tensor<128x32x27x27xf32>
    %v2498 = stablehlo.convolution(%v2496, %v2497)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x27x27xf32>, tensor<256x32x14x14xf32>) -> tensor<128x256x1x1xf32>
    %v2499 = stablehlo.transpose %v2498, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v2500 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2501 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2502 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2503 = stablehlo.reduce(%v2500 init: %v2501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2504 = stablehlo.broadcast_in_dim %v2503, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2505 = stablehlo.divide %v2504, %v2502 : tensor<32x256x14x14xf32>
    %v2506 = stablehlo.subtract %v2500, %v2505 : tensor<32x256x14x14xf32>
    %v2507 = stablehlo.multiply %v2506, %v2506 : tensor<32x256x14x14xf32>
    %v2508 = stablehlo.reduce(%v2507 init: %v2501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2509 = stablehlo.broadcast_in_dim %v2508, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2510 = stablehlo.divide %v2509, %v2502 : tensor<32x256x14x14xf32>
    %v2511 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2512 = stablehlo.add %v2510, %v2511 : tensor<32x256x14x14xf32>
    %v2513 = stablehlo.rsqrt %v2512 : tensor<32x256x14x14xf32>
    %v2514 = stablehlo.multiply %v2506, %v2513 : tensor<32x256x14x14xf32>
    %v2515 = stablehlo.reshape %v2322 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2516 = stablehlo.multiply %v2515, %v2514 : tensor<32x256x14x14xf32>
    %v2517 = stablehlo.reduce(%v2516 init: %v2501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2518 = stablehlo.reshape %v2322 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2519 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2520 = stablehlo.reduce(%v2518 init: %v2519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2521 = stablehlo.reshape %v2437 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2522 = stablehlo.reshape %v487 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2523 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2524 = stablehlo.compare GT, %v2522, %v2523 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2525 = stablehlo.select %v2524, %v2521, %v2523 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2526 = stablehlo.reshape %v2525 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2527 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2529 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2530 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2531 = stablehlo.reduce(%v2527 init: %v2528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2532 = stablehlo.broadcast_in_dim %v2531, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2533 = stablehlo.divide %v2532, %v2529 : tensor<32x128x28x28xf32>
    %v2534 = stablehlo.subtract %v2527, %v2533 : tensor<32x128x28x28xf32>
    %v2535 = stablehlo.multiply %v2534, %v2534 : tensor<32x128x28x28xf32>
    %v2536 = stablehlo.reduce(%v2535 init: %v2528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2537 = stablehlo.broadcast_in_dim %v2536, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2538 = stablehlo.divide %v2537, %v2529 : tensor<32x128x28x28xf32>
    %v2539 = stablehlo.add %v2538, %v2530 : tensor<32x128x28x28xf32>
    %v2540 = stablehlo.rsqrt %v2539 : tensor<32x128x28x28xf32>
    %v2541 = stablehlo.multiply %v2534, %v2540 : tensor<32x128x28x28xf32>
    %v2542 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2543 = stablehlo.reshape %v2526 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2544 = stablehlo.multiply %v2542, %v2543 : tensor<32x128x28x28xf32>
    %v2545 = stablehlo.reduce(%v2544 init: %v2528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2546 = stablehlo.broadcast_in_dim %v2545, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2547 = stablehlo.multiply %v2541, %v2544 : tensor<32x128x28x28xf32>
    %v2548 = stablehlo.reduce(%v2547 init: %v2528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2549 = stablehlo.broadcast_in_dim %v2548, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2550 = stablehlo.multiply %v2544, %v2529 : tensor<32x128x28x28xf32>
    %v2551 = stablehlo.subtract %v2550, %v2546 : tensor<32x128x28x28xf32>
    %v2552 = stablehlo.multiply %v2541, %v2549 : tensor<32x128x28x28xf32>
    %v2553 = stablehlo.subtract %v2551, %v2552 : tensor<32x128x28x28xf32>
    %v2554 = stablehlo.divide %v2540, %v2529 : tensor<32x128x28x28xf32>
    %v2555 = stablehlo.multiply %v2554, %v2553 : tensor<32x128x28x28xf32>
    %v2556 = stablehlo.reshape %v2555 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2557 = stablehlo.reshape %v2556 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2558 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2559 = stablehlo.transpose %v2558, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2560 = stablehlo.convolution(%v2557, %v2559)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2561 = stablehlo.reshape %v2560 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2562 = stablehlo.reshape %v2561 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2563 = stablehlo.reshape %v454 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2564 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2565 = stablehlo.compare GT, %v2563, %v2564 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2566 = stablehlo.select %v2565, %v2562, %v2564 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2567 = stablehlo.reshape %v2566 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2568 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2570 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2571 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2572 = stablehlo.reduce(%v2568 init: %v2569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2573 = stablehlo.broadcast_in_dim %v2572, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2574 = stablehlo.divide %v2573, %v2570 : tensor<32x128x28x28xf32>
    %v2575 = stablehlo.subtract %v2568, %v2574 : tensor<32x128x28x28xf32>
    %v2576 = stablehlo.multiply %v2575, %v2575 : tensor<32x128x28x28xf32>
    %v2577 = stablehlo.reduce(%v2576 init: %v2569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2578 = stablehlo.broadcast_in_dim %v2577, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2579 = stablehlo.divide %v2578, %v2570 : tensor<32x128x28x28xf32>
    %v2580 = stablehlo.add %v2579, %v2571 : tensor<32x128x28x28xf32>
    %v2581 = stablehlo.rsqrt %v2580 : tensor<32x128x28x28xf32>
    %v2582 = stablehlo.multiply %v2575, %v2581 : tensor<32x128x28x28xf32>
    %v2583 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2584 = stablehlo.reshape %v2567 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2585 = stablehlo.multiply %v2583, %v2584 : tensor<32x128x28x28xf32>
    %v2586 = stablehlo.reduce(%v2585 init: %v2569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2587 = stablehlo.broadcast_in_dim %v2586, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2588 = stablehlo.multiply %v2582, %v2585 : tensor<32x128x28x28xf32>
    %v2589 = stablehlo.reduce(%v2588 init: %v2569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2590 = stablehlo.broadcast_in_dim %v2589, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2591 = stablehlo.multiply %v2585, %v2570 : tensor<32x128x28x28xf32>
    %v2592 = stablehlo.subtract %v2591, %v2587 : tensor<32x128x28x28xf32>
    %v2593 = stablehlo.multiply %v2582, %v2590 : tensor<32x128x28x28xf32>
    %v2594 = stablehlo.subtract %v2592, %v2593 : tensor<32x128x28x28xf32>
    %v2595 = stablehlo.divide %v2581, %v2570 : tensor<32x128x28x28xf32>
    %v2596 = stablehlo.multiply %v2595, %v2594 : tensor<32x128x28x28xf32>
    %v2597 = stablehlo.reshape %v2596 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2598 = stablehlo.reshape %v2597 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2599 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2600 = stablehlo.transpose %v2599, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2601 = stablehlo.convolution(%v2598, %v2600)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2602 = stablehlo.reshape %v2601 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2603 = stablehlo.reshape %v2602 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2604 = stablehlo.reshape %v2526 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2605 = stablehlo.add %v2603, %v2604 : tensor<32x128x28x28xf32>
    %v2606 = stablehlo.reshape %v2605 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2607 = stablehlo.reshape %v429 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2608 = stablehlo.reshape %v2597 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2609 = stablehlo.transpose %v2607, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2610 = stablehlo.transpose %v2608, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2611 = stablehlo.convolution(%v2609, %v2610)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2612 = stablehlo.transpose %v2611, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2613 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2615 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2616 = stablehlo.reduce(%v2613 init: %v2614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2617 = stablehlo.broadcast_in_dim %v2616, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2618 = stablehlo.divide %v2617, %v2615 : tensor<32x128x28x28xf32>
    %v2619 = stablehlo.subtract %v2613, %v2618 : tensor<32x128x28x28xf32>
    %v2620 = stablehlo.multiply %v2619, %v2619 : tensor<32x128x28x28xf32>
    %v2621 = stablehlo.reduce(%v2620 init: %v2614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2622 = stablehlo.broadcast_in_dim %v2621, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2623 = stablehlo.divide %v2622, %v2615 : tensor<32x128x28x28xf32>
    %v2624 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2625 = stablehlo.add %v2623, %v2624 : tensor<32x128x28x28xf32>
    %v2626 = stablehlo.rsqrt %v2625 : tensor<32x128x28x28xf32>
    %v2627 = stablehlo.multiply %v2619, %v2626 : tensor<32x128x28x28xf32>
    %v2628 = stablehlo.reshape %v2567 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2629 = stablehlo.multiply %v2628, %v2627 : tensor<32x128x28x28xf32>
    %v2630 = stablehlo.reduce(%v2629 init: %v2614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2631 = stablehlo.reshape %v2567 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2632 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2633 = stablehlo.reduce(%v2631 init: %v2632) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2634 = stablehlo.reshape %v458 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2635 = stablehlo.reshape %v2556 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2636 = stablehlo.transpose %v2634, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2637 = stablehlo.transpose %v2635, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2638 = stablehlo.convolution(%v2636, %v2637)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2639 = stablehlo.transpose %v2638, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2640 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2642 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2643 = stablehlo.reduce(%v2640 init: %v2641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2644 = stablehlo.broadcast_in_dim %v2643, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2645 = stablehlo.divide %v2644, %v2642 : tensor<32x128x28x28xf32>
    %v2646 = stablehlo.subtract %v2640, %v2645 : tensor<32x128x28x28xf32>
    %v2647 = stablehlo.multiply %v2646, %v2646 : tensor<32x128x28x28xf32>
    %v2648 = stablehlo.reduce(%v2647 init: %v2641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2649 = stablehlo.broadcast_in_dim %v2648, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2650 = stablehlo.divide %v2649, %v2642 : tensor<32x128x28x28xf32>
    %v2651 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2652 = stablehlo.add %v2650, %v2651 : tensor<32x128x28x28xf32>
    %v2653 = stablehlo.rsqrt %v2652 : tensor<32x128x28x28xf32>
    %v2654 = stablehlo.multiply %v2646, %v2653 : tensor<32x128x28x28xf32>
    %v2655 = stablehlo.reshape %v2526 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2656 = stablehlo.multiply %v2655, %v2654 : tensor<32x128x28x28xf32>
    %v2657 = stablehlo.reduce(%v2656 init: %v2641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2658 = stablehlo.reshape %v2526 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2660 = stablehlo.reduce(%v2658 init: %v2659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2661 = stablehlo.reshape %v2606 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2662 = stablehlo.reshape %v425 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2663 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2664 = stablehlo.compare GT, %v2662, %v2663 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2665 = stablehlo.select %v2664, %v2661, %v2663 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2666 = stablehlo.reshape %v2665 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2667 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2668 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2669 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2670 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2671 = stablehlo.reduce(%v2667 init: %v2668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2672 = stablehlo.broadcast_in_dim %v2671, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2673 = stablehlo.divide %v2672, %v2669 : tensor<32x128x28x28xf32>
    %v2674 = stablehlo.subtract %v2667, %v2673 : tensor<32x128x28x28xf32>
    %v2675 = stablehlo.multiply %v2674, %v2674 : tensor<32x128x28x28xf32>
    %v2676 = stablehlo.reduce(%v2675 init: %v2668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2677 = stablehlo.broadcast_in_dim %v2676, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2678 = stablehlo.divide %v2677, %v2669 : tensor<32x128x28x28xf32>
    %v2679 = stablehlo.add %v2678, %v2670 : tensor<32x128x28x28xf32>
    %v2680 = stablehlo.rsqrt %v2679 : tensor<32x128x28x28xf32>
    %v2681 = stablehlo.multiply %v2674, %v2680 : tensor<32x128x28x28xf32>
    %v2682 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2683 = stablehlo.reshape %v2666 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2684 = stablehlo.multiply %v2682, %v2683 : tensor<32x128x28x28xf32>
    %v2685 = stablehlo.reduce(%v2684 init: %v2668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2686 = stablehlo.broadcast_in_dim %v2685, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2687 = stablehlo.multiply %v2681, %v2684 : tensor<32x128x28x28xf32>
    %v2688 = stablehlo.reduce(%v2687 init: %v2668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2689 = stablehlo.broadcast_in_dim %v2688, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2690 = stablehlo.multiply %v2684, %v2669 : tensor<32x128x28x28xf32>
    %v2691 = stablehlo.subtract %v2690, %v2686 : tensor<32x128x28x28xf32>
    %v2692 = stablehlo.multiply %v2681, %v2689 : tensor<32x128x28x28xf32>
    %v2693 = stablehlo.subtract %v2691, %v2692 : tensor<32x128x28x28xf32>
    %v2694 = stablehlo.divide %v2680, %v2669 : tensor<32x128x28x28xf32>
    %v2695 = stablehlo.multiply %v2694, %v2693 : tensor<32x128x28x28xf32>
    %v2696 = stablehlo.reshape %v2695 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2697 = stablehlo.reshape %v2696 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2698 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2699 = stablehlo.transpose %v2698, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2700 = stablehlo.convolution(%v2697, %v2699)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2701 = stablehlo.reshape %v2700 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2702 = stablehlo.reshape %v2701 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2703 = stablehlo.reshape %v392 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2704 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2705 = stablehlo.compare GT, %v2703, %v2704 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2706 = stablehlo.select %v2705, %v2702, %v2704 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2707 = stablehlo.reshape %v2706 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2708 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2710 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2711 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2712 = stablehlo.reduce(%v2708 init: %v2709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2713 = stablehlo.broadcast_in_dim %v2712, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2714 = stablehlo.divide %v2713, %v2710 : tensor<32x128x28x28xf32>
    %v2715 = stablehlo.subtract %v2708, %v2714 : tensor<32x128x28x28xf32>
    %v2716 = stablehlo.multiply %v2715, %v2715 : tensor<32x128x28x28xf32>
    %v2717 = stablehlo.reduce(%v2716 init: %v2709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2718 = stablehlo.broadcast_in_dim %v2717, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2719 = stablehlo.divide %v2718, %v2710 : tensor<32x128x28x28xf32>
    %v2720 = stablehlo.add %v2719, %v2711 : tensor<32x128x28x28xf32>
    %v2721 = stablehlo.rsqrt %v2720 : tensor<32x128x28x28xf32>
    %v2722 = stablehlo.multiply %v2715, %v2721 : tensor<32x128x28x28xf32>
    %v2723 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2724 = stablehlo.reshape %v2707 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2725 = stablehlo.multiply %v2723, %v2724 : tensor<32x128x28x28xf32>
    %v2726 = stablehlo.reduce(%v2725 init: %v2709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2727 = stablehlo.broadcast_in_dim %v2726, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2728 = stablehlo.multiply %v2722, %v2725 : tensor<32x128x28x28xf32>
    %v2729 = stablehlo.reduce(%v2728 init: %v2709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2730 = stablehlo.broadcast_in_dim %v2729, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2731 = stablehlo.multiply %v2725, %v2710 : tensor<32x128x28x28xf32>
    %v2732 = stablehlo.subtract %v2731, %v2727 : tensor<32x128x28x28xf32>
    %v2733 = stablehlo.multiply %v2722, %v2730 : tensor<32x128x28x28xf32>
    %v2734 = stablehlo.subtract %v2732, %v2733 : tensor<32x128x28x28xf32>
    %v2735 = stablehlo.divide %v2721, %v2710 : tensor<32x128x28x28xf32>
    %v2736 = stablehlo.multiply %v2735, %v2734 : tensor<32x128x28x28xf32>
    %v2737 = stablehlo.reshape %v2736 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2738 = stablehlo.reshape %v2737 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2739 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2740 = stablehlo.transpose %v2739, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2741 = stablehlo.convolution(%v2738, %v2740)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2742 = stablehlo.reshape %v2741 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2743 = stablehlo.reshape %v2742 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2744 = stablehlo.reshape %v2666 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2745 = stablehlo.add %v2743, %v2744 : tensor<32x128x28x28xf32>
    %v2746 = stablehlo.reshape %v2745 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2747 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2748 = stablehlo.reshape %v2737 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2749 = stablehlo.transpose %v2747, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2750 = stablehlo.transpose %v2748, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2751 = stablehlo.convolution(%v2749, %v2750)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2752 = stablehlo.transpose %v2751, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2753 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2754 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2755 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2756 = stablehlo.reduce(%v2753 init: %v2754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2757 = stablehlo.broadcast_in_dim %v2756, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2758 = stablehlo.divide %v2757, %v2755 : tensor<32x128x28x28xf32>
    %v2759 = stablehlo.subtract %v2753, %v2758 : tensor<32x128x28x28xf32>
    %v2760 = stablehlo.multiply %v2759, %v2759 : tensor<32x128x28x28xf32>
    %v2761 = stablehlo.reduce(%v2760 init: %v2754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2762 = stablehlo.broadcast_in_dim %v2761, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2763 = stablehlo.divide %v2762, %v2755 : tensor<32x128x28x28xf32>
    %v2764 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2765 = stablehlo.add %v2763, %v2764 : tensor<32x128x28x28xf32>
    %v2766 = stablehlo.rsqrt %v2765 : tensor<32x128x28x28xf32>
    %v2767 = stablehlo.multiply %v2759, %v2766 : tensor<32x128x28x28xf32>
    %v2768 = stablehlo.reshape %v2707 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2769 = stablehlo.multiply %v2768, %v2767 : tensor<32x128x28x28xf32>
    %v2770 = stablehlo.reduce(%v2769 init: %v2754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2771 = stablehlo.reshape %v2707 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2772 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2773 = stablehlo.reduce(%v2771 init: %v2772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2774 = stablehlo.reshape %v396 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2775 = stablehlo.reshape %v2696 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2776 = stablehlo.transpose %v2774, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2777 = stablehlo.transpose %v2775, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2778 = stablehlo.convolution(%v2776, %v2777)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2779 = stablehlo.transpose %v2778, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2780 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2782 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2783 = stablehlo.reduce(%v2780 init: %v2781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2784 = stablehlo.broadcast_in_dim %v2783, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2785 = stablehlo.divide %v2784, %v2782 : tensor<32x128x28x28xf32>
    %v2786 = stablehlo.subtract %v2780, %v2785 : tensor<32x128x28x28xf32>
    %v2787 = stablehlo.multiply %v2786, %v2786 : tensor<32x128x28x28xf32>
    %v2788 = stablehlo.reduce(%v2787 init: %v2781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2789 = stablehlo.broadcast_in_dim %v2788, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2790 = stablehlo.divide %v2789, %v2782 : tensor<32x128x28x28xf32>
    %v2791 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2792 = stablehlo.add %v2790, %v2791 : tensor<32x128x28x28xf32>
    %v2793 = stablehlo.rsqrt %v2792 : tensor<32x128x28x28xf32>
    %v2794 = stablehlo.multiply %v2786, %v2793 : tensor<32x128x28x28xf32>
    %v2795 = stablehlo.reshape %v2666 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2796 = stablehlo.multiply %v2795, %v2794 : tensor<32x128x28x28xf32>
    %v2797 = stablehlo.reduce(%v2796 init: %v2781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2798 = stablehlo.reshape %v2666 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2800 = stablehlo.reduce(%v2798 init: %v2799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2801 = stablehlo.reshape %v2746 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2802 = stablehlo.reshape %v363 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2803 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2804 = stablehlo.compare GT, %v2802, %v2803 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2805 = stablehlo.select %v2804, %v2801, %v2803 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2806 = stablehlo.reshape %v2805 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2807 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2808 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2809 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2810 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2811 = stablehlo.reduce(%v2807 init: %v2808) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2812 = stablehlo.broadcast_in_dim %v2811, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2813 = stablehlo.divide %v2812, %v2809 : tensor<32x128x28x28xf32>
    %v2814 = stablehlo.subtract %v2807, %v2813 : tensor<32x128x28x28xf32>
    %v2815 = stablehlo.multiply %v2814, %v2814 : tensor<32x128x28x28xf32>
    %v2816 = stablehlo.reduce(%v2815 init: %v2808) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2817 = stablehlo.broadcast_in_dim %v2816, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2818 = stablehlo.divide %v2817, %v2809 : tensor<32x128x28x28xf32>
    %v2819 = stablehlo.add %v2818, %v2810 : tensor<32x128x28x28xf32>
    %v2820 = stablehlo.rsqrt %v2819 : tensor<32x128x28x28xf32>
    %v2821 = stablehlo.multiply %v2814, %v2820 : tensor<32x128x28x28xf32>
    %v2822 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2823 = stablehlo.reshape %v2806 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2824 = stablehlo.multiply %v2822, %v2823 : tensor<32x128x28x28xf32>
    %v2825 = stablehlo.reduce(%v2824 init: %v2808) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2826 = stablehlo.broadcast_in_dim %v2825, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2827 = stablehlo.multiply %v2821, %v2824 : tensor<32x128x28x28xf32>
    %v2828 = stablehlo.reduce(%v2827 init: %v2808) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2829 = stablehlo.broadcast_in_dim %v2828, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2830 = stablehlo.multiply %v2824, %v2809 : tensor<32x128x28x28xf32>
    %v2831 = stablehlo.subtract %v2830, %v2826 : tensor<32x128x28x28xf32>
    %v2832 = stablehlo.multiply %v2821, %v2829 : tensor<32x128x28x28xf32>
    %v2833 = stablehlo.subtract %v2831, %v2832 : tensor<32x128x28x28xf32>
    %v2834 = stablehlo.divide %v2820, %v2809 : tensor<32x128x28x28xf32>
    %v2835 = stablehlo.multiply %v2834, %v2833 : tensor<32x128x28x28xf32>
    %v2836 = stablehlo.reshape %v2835 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2837 = stablehlo.reshape %v2836 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2838 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2839 = stablehlo.transpose %v2838, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2840 = stablehlo.convolution(%v2837, %v2839)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2841 = stablehlo.reshape %v2840 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2842 = stablehlo.reshape %v2841 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2843 = stablehlo.reshape %v330 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2844 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2845 = stablehlo.compare GT, %v2843, %v2844 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2846 = stablehlo.select %v2845, %v2842, %v2844 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2847 = stablehlo.reshape %v2846 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2848 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2850 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2851 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2852 = stablehlo.reduce(%v2848 init: %v2849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2853 = stablehlo.broadcast_in_dim %v2852, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2854 = stablehlo.divide %v2853, %v2850 : tensor<32x128x28x28xf32>
    %v2855 = stablehlo.subtract %v2848, %v2854 : tensor<32x128x28x28xf32>
    %v2856 = stablehlo.multiply %v2855, %v2855 : tensor<32x128x28x28xf32>
    %v2857 = stablehlo.reduce(%v2856 init: %v2849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2858 = stablehlo.broadcast_in_dim %v2857, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2859 = stablehlo.divide %v2858, %v2850 : tensor<32x128x28x28xf32>
    %v2860 = stablehlo.add %v2859, %v2851 : tensor<32x128x28x28xf32>
    %v2861 = stablehlo.rsqrt %v2860 : tensor<32x128x28x28xf32>
    %v2862 = stablehlo.multiply %v2855, %v2861 : tensor<32x128x28x28xf32>
    %v2863 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2864 = stablehlo.reshape %v2847 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2865 = stablehlo.multiply %v2863, %v2864 : tensor<32x128x28x28xf32>
    %v2866 = stablehlo.reduce(%v2865 init: %v2849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2867 = stablehlo.broadcast_in_dim %v2866, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2868 = stablehlo.multiply %v2862, %v2865 : tensor<32x128x28x28xf32>
    %v2869 = stablehlo.reduce(%v2868 init: %v2849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2870 = stablehlo.broadcast_in_dim %v2869, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2871 = stablehlo.multiply %v2865, %v2850 : tensor<32x128x28x28xf32>
    %v2872 = stablehlo.subtract %v2871, %v2867 : tensor<32x128x28x28xf32>
    %v2873 = stablehlo.multiply %v2862, %v2870 : tensor<32x128x28x28xf32>
    %v2874 = stablehlo.subtract %v2872, %v2873 : tensor<32x128x28x28xf32>
    %v2875 = stablehlo.divide %v2861, %v2850 : tensor<32x128x28x28xf32>
    %v2876 = stablehlo.multiply %v2875, %v2874 : tensor<32x128x28x28xf32>
    %v2877 = stablehlo.reshape %v2876 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2878 = stablehlo.reshape %v2877 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2879 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2880 = stablehlo.transpose %v2879, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2881 = stablehlo.convolution(%v2878, %v2880)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2882 = stablehlo.reshape %v2881 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2883 = stablehlo.reshape %v2882 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2884 = stablehlo.reshape %v2806 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2885 = stablehlo.add %v2883, %v2884 : tensor<32x128x28x28xf32>
    %v2886 = stablehlo.reshape %v2885 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2887 = stablehlo.reshape %v305 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2888 = stablehlo.reshape %v2877 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2889 = stablehlo.transpose %v2887, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2890 = stablehlo.transpose %v2888, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2891 = stablehlo.convolution(%v2889, %v2890)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2892 = stablehlo.transpose %v2891, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2893 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2895 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2896 = stablehlo.reduce(%v2893 init: %v2894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2897 = stablehlo.broadcast_in_dim %v2896, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2898 = stablehlo.divide %v2897, %v2895 : tensor<32x128x28x28xf32>
    %v2899 = stablehlo.subtract %v2893, %v2898 : tensor<32x128x28x28xf32>
    %v2900 = stablehlo.multiply %v2899, %v2899 : tensor<32x128x28x28xf32>
    %v2901 = stablehlo.reduce(%v2900 init: %v2894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2902 = stablehlo.broadcast_in_dim %v2901, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2903 = stablehlo.divide %v2902, %v2895 : tensor<32x128x28x28xf32>
    %v2904 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2905 = stablehlo.add %v2903, %v2904 : tensor<32x128x28x28xf32>
    %v2906 = stablehlo.rsqrt %v2905 : tensor<32x128x28x28xf32>
    %v2907 = stablehlo.multiply %v2899, %v2906 : tensor<32x128x28x28xf32>
    %v2908 = stablehlo.reshape %v2847 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2909 = stablehlo.multiply %v2908, %v2907 : tensor<32x128x28x28xf32>
    %v2910 = stablehlo.reduce(%v2909 init: %v2894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2911 = stablehlo.reshape %v2847 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2912 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2913 = stablehlo.reduce(%v2911 init: %v2912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2914 = stablehlo.reshape %v334 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2915 = stablehlo.reshape %v2836 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2916 = stablehlo.transpose %v2914, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2917 = stablehlo.transpose %v2915, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2918 = stablehlo.convolution(%v2916, %v2917)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2919 = stablehlo.transpose %v2918, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2920 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2922 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2923 = stablehlo.reduce(%v2920 init: %v2921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2924 = stablehlo.broadcast_in_dim %v2923, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2925 = stablehlo.divide %v2924, %v2922 : tensor<32x128x28x28xf32>
    %v2926 = stablehlo.subtract %v2920, %v2925 : tensor<32x128x28x28xf32>
    %v2927 = stablehlo.multiply %v2926, %v2926 : tensor<32x128x28x28xf32>
    %v2928 = stablehlo.reduce(%v2927 init: %v2921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2929 = stablehlo.broadcast_in_dim %v2928, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2930 = stablehlo.divide %v2929, %v2922 : tensor<32x128x28x28xf32>
    %v2931 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2932 = stablehlo.add %v2930, %v2931 : tensor<32x128x28x28xf32>
    %v2933 = stablehlo.rsqrt %v2932 : tensor<32x128x28x28xf32>
    %v2934 = stablehlo.multiply %v2926, %v2933 : tensor<32x128x28x28xf32>
    %v2935 = stablehlo.reshape %v2806 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2936 = stablehlo.multiply %v2935, %v2934 : tensor<32x128x28x28xf32>
    %v2937 = stablehlo.reduce(%v2936 init: %v2921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2938 = stablehlo.reshape %v2806 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2940 = stablehlo.reduce(%v2938 init: %v2939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2941 = stablehlo.reshape %v2886 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2942 = stablehlo.reshape %v301 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2943 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2944 = stablehlo.compare GT, %v2942, %v2943 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2945 = stablehlo.select %v2944, %v2941, %v2943 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2946 = stablehlo.reshape %v2945 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2947 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2948 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2949 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2950 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2951 = stablehlo.reduce(%v2947 init: %v2948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2952 = stablehlo.broadcast_in_dim %v2951, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2953 = stablehlo.divide %v2952, %v2949 : tensor<32x128x28x28xf32>
    %v2954 = stablehlo.subtract %v2947, %v2953 : tensor<32x128x28x28xf32>
    %v2955 = stablehlo.multiply %v2954, %v2954 : tensor<32x128x28x28xf32>
    %v2956 = stablehlo.reduce(%v2955 init: %v2948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2957 = stablehlo.broadcast_in_dim %v2956, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2958 = stablehlo.divide %v2957, %v2949 : tensor<32x128x28x28xf32>
    %v2959 = stablehlo.add %v2958, %v2950 : tensor<32x128x28x28xf32>
    %v2960 = stablehlo.rsqrt %v2959 : tensor<32x128x28x28xf32>
    %v2961 = stablehlo.multiply %v2954, %v2960 : tensor<32x128x28x28xf32>
    %v2962 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2963 = stablehlo.reshape %v2946 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2964 = stablehlo.multiply %v2962, %v2963 : tensor<32x128x28x28xf32>
    %v2965 = stablehlo.reduce(%v2964 init: %v2948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2966 = stablehlo.broadcast_in_dim %v2965, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2967 = stablehlo.multiply %v2961, %v2964 : tensor<32x128x28x28xf32>
    %v2968 = stablehlo.reduce(%v2967 init: %v2948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2969 = stablehlo.broadcast_in_dim %v2968, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2970 = stablehlo.multiply %v2964, %v2949 : tensor<32x128x28x28xf32>
    %v2971 = stablehlo.subtract %v2970, %v2966 : tensor<32x128x28x28xf32>
    %v2972 = stablehlo.multiply %v2961, %v2969 : tensor<32x128x28x28xf32>
    %v2973 = stablehlo.subtract %v2971, %v2972 : tensor<32x128x28x28xf32>
    %v2974 = stablehlo.divide %v2960, %v2949 : tensor<32x128x28x28xf32>
    %v2975 = stablehlo.multiply %v2974, %v2973 : tensor<32x128x28x28xf32>
    %v2976 = stablehlo.reshape %v2975 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2977 = stablehlo.reshape %v2976 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2978 = stablehlo.reverse %d2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2979 = stablehlo.transpose %v2978, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2980 = stablehlo.convolution(%v2977, %v2979)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2981 = stablehlo.reshape %v2980 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2982 = stablehlo.reshape %v2981 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2983 = stablehlo.reshape %v243 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2984 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2985 = stablehlo.compare GT, %v2983, %v2984 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2986 = stablehlo.select %v2985, %v2982, %v2984 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2987 = stablehlo.reshape %v2986 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2988 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2989 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2990 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2991 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2992 = stablehlo.reduce(%v2988 init: %v2989) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2993 = stablehlo.broadcast_in_dim %v2992, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2994 = stablehlo.divide %v2993, %v2990 : tensor<32x128x28x28xf32>
    %v2995 = stablehlo.subtract %v2988, %v2994 : tensor<32x128x28x28xf32>
    %v2996 = stablehlo.multiply %v2995, %v2995 : tensor<32x128x28x28xf32>
    %v2997 = stablehlo.reduce(%v2996 init: %v2989) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2998 = stablehlo.broadcast_in_dim %v2997, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2999 = stablehlo.divide %v2998, %v2990 : tensor<32x128x28x28xf32>
    %v3000 = stablehlo.add %v2999, %v2991 : tensor<32x128x28x28xf32>
    %v3001 = stablehlo.rsqrt %v3000 : tensor<32x128x28x28xf32>
    %v3002 = stablehlo.multiply %v2995, %v3001 : tensor<32x128x28x28xf32>
    %v3003 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3004 = stablehlo.reshape %v2987 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3005 = stablehlo.multiply %v3003, %v3004 : tensor<32x128x28x28xf32>
    %v3006 = stablehlo.reduce(%v3005 init: %v2989) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3007 = stablehlo.broadcast_in_dim %v3006, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3008 = stablehlo.multiply %v3002, %v3005 : tensor<32x128x28x28xf32>
    %v3009 = stablehlo.reduce(%v3008 init: %v2989) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3010 = stablehlo.broadcast_in_dim %v3009, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3011 = stablehlo.multiply %v3005, %v2990 : tensor<32x128x28x28xf32>
    %v3012 = stablehlo.subtract %v3011, %v3007 : tensor<32x128x28x28xf32>
    %v3013 = stablehlo.multiply %v3002, %v3010 : tensor<32x128x28x28xf32>
    %v3014 = stablehlo.subtract %v3012, %v3013 : tensor<32x128x28x28xf32>
    %v3015 = stablehlo.divide %v3001, %v2990 : tensor<32x128x28x28xf32>
    %v3016 = stablehlo.multiply %v3015, %v3014 : tensor<32x128x28x28xf32>
    %v3017 = stablehlo.reshape %v3016 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3018 = stablehlo.reshape %v3017 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3019 = stablehlo.reverse %d2W1, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v3020 = stablehlo.transpose %v3019, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v3021 = stablehlo.convolution(%v3018, %v3020)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3022 = stablehlo.reshape %v3021 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3023 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3024 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3025 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3026 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3027 = stablehlo.reduce(%v3023 init: %v3024) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3028 = stablehlo.broadcast_in_dim %v3027, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3029 = stablehlo.divide %v3028, %v3025 : tensor<32x128x28x28xf32>
    %v3030 = stablehlo.subtract %v3023, %v3029 : tensor<32x128x28x28xf32>
    %v3031 = stablehlo.multiply %v3030, %v3030 : tensor<32x128x28x28xf32>
    %v3032 = stablehlo.reduce(%v3031 init: %v3024) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3033 = stablehlo.broadcast_in_dim %v3032, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3034 = stablehlo.divide %v3033, %v3025 : tensor<32x128x28x28xf32>
    %v3035 = stablehlo.add %v3034, %v3026 : tensor<32x128x28x28xf32>
    %v3036 = stablehlo.rsqrt %v3035 : tensor<32x128x28x28xf32>
    %v3037 = stablehlo.multiply %v3030, %v3036 : tensor<32x128x28x28xf32>
    %v3038 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3039 = stablehlo.reshape %v2946 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3040 = stablehlo.multiply %v3038, %v3039 : tensor<32x128x28x28xf32>
    %v3041 = stablehlo.reduce(%v3040 init: %v3024) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3042 = stablehlo.broadcast_in_dim %v3041, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3043 = stablehlo.multiply %v3037, %v3040 : tensor<32x128x28x28xf32>
    %v3044 = stablehlo.reduce(%v3043 init: %v3024) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3045 = stablehlo.broadcast_in_dim %v3044, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3046 = stablehlo.multiply %v3040, %v3025 : tensor<32x128x28x28xf32>
    %v3047 = stablehlo.subtract %v3046, %v3042 : tensor<32x128x28x28xf32>
    %v3048 = stablehlo.multiply %v3037, %v3045 : tensor<32x128x28x28xf32>
    %v3049 = stablehlo.subtract %v3047, %v3048 : tensor<32x128x28x28xf32>
    %v3050 = stablehlo.divide %v3036, %v3025 : tensor<32x128x28x28xf32>
    %v3051 = stablehlo.multiply %v3050, %v3049 : tensor<32x128x28x28xf32>
    %v3052 = stablehlo.reshape %v3051 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3053 = stablehlo.reshape %v3052 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3054 = stablehlo.reverse %d2Wp, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v3055 = stablehlo.transpose %v3054, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v3056 = stablehlo.convolution(%v3053, %v3055)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<64x128x1x1xf32>) -> tensor<32x64x56x56xf32>
    %v3057 = stablehlo.reshape %v3056 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3058 = stablehlo.reshape %v3022 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3059 = stablehlo.reshape %v3057 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3060 = stablehlo.add %v3058, %v3059 : tensor<32x64x56x56xf32>
    %v3061 = stablehlo.reshape %v3060 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3062 = stablehlo.reshape %v218 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3063 = stablehlo.reshape %v3017 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3064 = stablehlo.transpose %v3062, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3066 = stablehlo.transpose %v3063, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3067 = stablehlo.convolution(%v3064, %v3066)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x28x28xf32>) -> tensor<64x128x3x3xf32>
    %v3068 = stablehlo.transpose %v3067, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v3069 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3070 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3071 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3072 = stablehlo.reduce(%v3069 init: %v3070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3073 = stablehlo.broadcast_in_dim %v3072, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3074 = stablehlo.divide %v3073, %v3071 : tensor<32x128x28x28xf32>
    %v3075 = stablehlo.subtract %v3069, %v3074 : tensor<32x128x28x28xf32>
    %v3076 = stablehlo.multiply %v3075, %v3075 : tensor<32x128x28x28xf32>
    %v3077 = stablehlo.reduce(%v3076 init: %v3070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3078 = stablehlo.broadcast_in_dim %v3077, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3079 = stablehlo.divide %v3078, %v3071 : tensor<32x128x28x28xf32>
    %v3080 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3081 = stablehlo.add %v3079, %v3080 : tensor<32x128x28x28xf32>
    %v3082 = stablehlo.rsqrt %v3081 : tensor<32x128x28x28xf32>
    %v3083 = stablehlo.multiply %v3075, %v3082 : tensor<32x128x28x28xf32>
    %v3084 = stablehlo.reshape %v2987 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3085 = stablehlo.multiply %v3084, %v3083 : tensor<32x128x28x28xf32>
    %v3086 = stablehlo.reduce(%v3085 init: %v3070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3087 = stablehlo.reshape %v2987 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3088 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3089 = stablehlo.reduce(%v3087 init: %v3088) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3090 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3091 = stablehlo.reshape %v2976 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3092 = stablehlo.transpose %v3090, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3093 = stablehlo.transpose %v3091, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3094 = stablehlo.convolution(%v3092, %v3093)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3095 = stablehlo.transpose %v3094, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3096 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3098 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3099 = stablehlo.reduce(%v3096 init: %v3097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3100 = stablehlo.broadcast_in_dim %v3099, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3101 = stablehlo.divide %v3100, %v3098 : tensor<32x128x28x28xf32>
    %v3102 = stablehlo.subtract %v3096, %v3101 : tensor<32x128x28x28xf32>
    %v3103 = stablehlo.multiply %v3102, %v3102 : tensor<32x128x28x28xf32>
    %v3104 = stablehlo.reduce(%v3103 init: %v3097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3105 = stablehlo.broadcast_in_dim %v3104, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3106 = stablehlo.divide %v3105, %v3098 : tensor<32x128x28x28xf32>
    %v3107 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3108 = stablehlo.add %v3106, %v3107 : tensor<32x128x28x28xf32>
    %v3109 = stablehlo.rsqrt %v3108 : tensor<32x128x28x28xf32>
    %v3110 = stablehlo.multiply %v3102, %v3109 : tensor<32x128x28x28xf32>
    %v3111 = stablehlo.reshape %v2946 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3112 = stablehlo.multiply %v3111, %v3110 : tensor<32x128x28x28xf32>
    %v3113 = stablehlo.reduce(%v3112 init: %v3097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3114 = stablehlo.reshape %v2946 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3116 = stablehlo.reduce(%v3114 init: %v3115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3117 = stablehlo.reshape %v218 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3118 = stablehlo.reshape %v3052 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3119 = stablehlo.transpose %v3117, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3121 = stablehlo.transpose %v3118, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3120 = stablehlo.slice %v3119 [0:64, 0:32, 0:55, 0:55] : (tensor<64x32x56x56xf32>) -> tensor<64x32x55x55xf32>
    %v3122 = stablehlo.convolution(%v3120, %v3121)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x55x55xf32>, tensor<128x32x28x28xf32>) -> tensor<64x128x1x1xf32>
    %v3123 = stablehlo.transpose %v3122, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v3124 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3125 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3126 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3127 = stablehlo.reduce(%v3124 init: %v3125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3128 = stablehlo.broadcast_in_dim %v3127, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3129 = stablehlo.divide %v3128, %v3126 : tensor<32x128x28x28xf32>
    %v3130 = stablehlo.subtract %v3124, %v3129 : tensor<32x128x28x28xf32>
    %v3131 = stablehlo.multiply %v3130, %v3130 : tensor<32x128x28x28xf32>
    %v3132 = stablehlo.reduce(%v3131 init: %v3125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3133 = stablehlo.broadcast_in_dim %v3132, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3134 = stablehlo.divide %v3133, %v3126 : tensor<32x128x28x28xf32>
    %v3135 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3136 = stablehlo.add %v3134, %v3135 : tensor<32x128x28x28xf32>
    %v3137 = stablehlo.rsqrt %v3136 : tensor<32x128x28x28xf32>
    %v3138 = stablehlo.multiply %v3130, %v3137 : tensor<32x128x28x28xf32>
    %v3139 = stablehlo.reshape %v2946 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3140 = stablehlo.multiply %v3139, %v3138 : tensor<32x128x28x28xf32>
    %v3141 = stablehlo.reduce(%v3140 init: %v3125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3142 = stablehlo.reshape %v2946 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3144 = stablehlo.reduce(%v3142 init: %v3143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3145 = stablehlo.reshape %v3061 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3146 = stablehlo.reshape %v214 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3147 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3148 = stablehlo.compare GT, %v3146, %v3147 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3149 = stablehlo.select %v3148, %v3145, %v3147 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3150 = stablehlo.reshape %v3149 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3151 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3152 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3153 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3154 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3155 = stablehlo.reduce(%v3151 init: %v3152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3156 = stablehlo.broadcast_in_dim %v3155, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3157 = stablehlo.divide %v3156, %v3153 : tensor<32x64x56x56xf32>
    %v3158 = stablehlo.subtract %v3151, %v3157 : tensor<32x64x56x56xf32>
    %v3159 = stablehlo.multiply %v3158, %v3158 : tensor<32x64x56x56xf32>
    %v3160 = stablehlo.reduce(%v3159 init: %v3152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3161 = stablehlo.broadcast_in_dim %v3160, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3162 = stablehlo.divide %v3161, %v3153 : tensor<32x64x56x56xf32>
    %v3163 = stablehlo.add %v3162, %v3154 : tensor<32x64x56x56xf32>
    %v3164 = stablehlo.rsqrt %v3163 : tensor<32x64x56x56xf32>
    %v3165 = stablehlo.multiply %v3158, %v3164 : tensor<32x64x56x56xf32>
    %v3166 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3167 = stablehlo.reshape %v3150 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3168 = stablehlo.multiply %v3166, %v3167 : tensor<32x64x56x56xf32>
    %v3169 = stablehlo.reduce(%v3168 init: %v3152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3170 = stablehlo.broadcast_in_dim %v3169, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3171 = stablehlo.multiply %v3165, %v3168 : tensor<32x64x56x56xf32>
    %v3172 = stablehlo.reduce(%v3171 init: %v3152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3173 = stablehlo.broadcast_in_dim %v3172, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3174 = stablehlo.multiply %v3168, %v3153 : tensor<32x64x56x56xf32>
    %v3175 = stablehlo.subtract %v3174, %v3170 : tensor<32x64x56x56xf32>
    %v3176 = stablehlo.multiply %v3165, %v3173 : tensor<32x64x56x56xf32>
    %v3177 = stablehlo.subtract %v3175, %v3176 : tensor<32x64x56x56xf32>
    %v3178 = stablehlo.divide %v3164, %v3153 : tensor<32x64x56x56xf32>
    %v3179 = stablehlo.multiply %v3178, %v3177 : tensor<32x64x56x56xf32>
    %v3180 = stablehlo.reshape %v3179 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3181 = stablehlo.reshape %v3180 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3182 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3183 = stablehlo.transpose %v3182, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3184 = stablehlo.convolution(%v3181, %v3183)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3185 = stablehlo.reshape %v3184 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3186 = stablehlo.reshape %v3185 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3187 = stablehlo.reshape %v181 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3188 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3189 = stablehlo.compare GT, %v3187, %v3188 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3190 = stablehlo.select %v3189, %v3186, %v3188 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3191 = stablehlo.reshape %v3190 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3192 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3193 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3194 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3195 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3196 = stablehlo.reduce(%v3192 init: %v3193) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3197 = stablehlo.broadcast_in_dim %v3196, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3198 = stablehlo.divide %v3197, %v3194 : tensor<32x64x56x56xf32>
    %v3199 = stablehlo.subtract %v3192, %v3198 : tensor<32x64x56x56xf32>
    %v3200 = stablehlo.multiply %v3199, %v3199 : tensor<32x64x56x56xf32>
    %v3201 = stablehlo.reduce(%v3200 init: %v3193) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3202 = stablehlo.broadcast_in_dim %v3201, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3203 = stablehlo.divide %v3202, %v3194 : tensor<32x64x56x56xf32>
    %v3204 = stablehlo.add %v3203, %v3195 : tensor<32x64x56x56xf32>
    %v3205 = stablehlo.rsqrt %v3204 : tensor<32x64x56x56xf32>
    %v3206 = stablehlo.multiply %v3199, %v3205 : tensor<32x64x56x56xf32>
    %v3207 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3208 = stablehlo.reshape %v3191 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3209 = stablehlo.multiply %v3207, %v3208 : tensor<32x64x56x56xf32>
    %v3210 = stablehlo.reduce(%v3209 init: %v3193) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3211 = stablehlo.broadcast_in_dim %v3210, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3212 = stablehlo.multiply %v3206, %v3209 : tensor<32x64x56x56xf32>
    %v3213 = stablehlo.reduce(%v3212 init: %v3193) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3214 = stablehlo.broadcast_in_dim %v3213, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3215 = stablehlo.multiply %v3209, %v3194 : tensor<32x64x56x56xf32>
    %v3216 = stablehlo.subtract %v3215, %v3211 : tensor<32x64x56x56xf32>
    %v3217 = stablehlo.multiply %v3206, %v3214 : tensor<32x64x56x56xf32>
    %v3218 = stablehlo.subtract %v3216, %v3217 : tensor<32x64x56x56xf32>
    %v3219 = stablehlo.divide %v3205, %v3194 : tensor<32x64x56x56xf32>
    %v3220 = stablehlo.multiply %v3219, %v3218 : tensor<32x64x56x56xf32>
    %v3221 = stablehlo.reshape %v3220 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3222 = stablehlo.reshape %v3221 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3223 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3224 = stablehlo.transpose %v3223, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3225 = stablehlo.convolution(%v3222, %v3224)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3226 = stablehlo.reshape %v3225 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3227 = stablehlo.reshape %v3226 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3228 = stablehlo.reshape %v3150 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3229 = stablehlo.add %v3227, %v3228 : tensor<32x64x56x56xf32>
    %v3230 = stablehlo.reshape %v3229 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3231 = stablehlo.reshape %v156 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3232 = stablehlo.reshape %v3221 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3233 = stablehlo.transpose %v3231, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3234 = stablehlo.transpose %v3232, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3235 = stablehlo.convolution(%v3233, %v3234)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3236 = stablehlo.transpose %v3235, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3237 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3238 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3239 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3240 = stablehlo.reduce(%v3237 init: %v3238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3241 = stablehlo.broadcast_in_dim %v3240, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3242 = stablehlo.divide %v3241, %v3239 : tensor<32x64x56x56xf32>
    %v3243 = stablehlo.subtract %v3237, %v3242 : tensor<32x64x56x56xf32>
    %v3244 = stablehlo.multiply %v3243, %v3243 : tensor<32x64x56x56xf32>
    %v3245 = stablehlo.reduce(%v3244 init: %v3238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3246 = stablehlo.broadcast_in_dim %v3245, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3247 = stablehlo.divide %v3246, %v3239 : tensor<32x64x56x56xf32>
    %v3248 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3249 = stablehlo.add %v3247, %v3248 : tensor<32x64x56x56xf32>
    %v3250 = stablehlo.rsqrt %v3249 : tensor<32x64x56x56xf32>
    %v3251 = stablehlo.multiply %v3243, %v3250 : tensor<32x64x56x56xf32>
    %v3252 = stablehlo.reshape %v3191 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3253 = stablehlo.multiply %v3252, %v3251 : tensor<32x64x56x56xf32>
    %v3254 = stablehlo.reduce(%v3253 init: %v3238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3255 = stablehlo.reshape %v3191 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3257 = stablehlo.reduce(%v3255 init: %v3256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3258 = stablehlo.reshape %v185 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3259 = stablehlo.reshape %v3180 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3260 = stablehlo.transpose %v3258, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3261 = stablehlo.transpose %v3259, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3262 = stablehlo.convolution(%v3260, %v3261)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3263 = stablehlo.transpose %v3262, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3264 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3266 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3267 = stablehlo.reduce(%v3264 init: %v3265) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3268 = stablehlo.broadcast_in_dim %v3267, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3269 = stablehlo.divide %v3268, %v3266 : tensor<32x64x56x56xf32>
    %v3270 = stablehlo.subtract %v3264, %v3269 : tensor<32x64x56x56xf32>
    %v3271 = stablehlo.multiply %v3270, %v3270 : tensor<32x64x56x56xf32>
    %v3272 = stablehlo.reduce(%v3271 init: %v3265) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3273 = stablehlo.broadcast_in_dim %v3272, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3274 = stablehlo.divide %v3273, %v3266 : tensor<32x64x56x56xf32>
    %v3275 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3276 = stablehlo.add %v3274, %v3275 : tensor<32x64x56x56xf32>
    %v3277 = stablehlo.rsqrt %v3276 : tensor<32x64x56x56xf32>
    %v3278 = stablehlo.multiply %v3270, %v3277 : tensor<32x64x56x56xf32>
    %v3279 = stablehlo.reshape %v3150 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3280 = stablehlo.multiply %v3279, %v3278 : tensor<32x64x56x56xf32>
    %v3281 = stablehlo.reduce(%v3280 init: %v3265) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3282 = stablehlo.reshape %v3150 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3284 = stablehlo.reduce(%v3282 init: %v3283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3285 = stablehlo.reshape %v3230 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3286 = stablehlo.reshape %v152 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3287 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3288 = stablehlo.compare GT, %v3286, %v3287 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3289 = stablehlo.select %v3288, %v3285, %v3287 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3290 = stablehlo.reshape %v3289 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3291 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3292 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3293 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3294 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3295 = stablehlo.reduce(%v3291 init: %v3292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3296 = stablehlo.broadcast_in_dim %v3295, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3297 = stablehlo.divide %v3296, %v3293 : tensor<32x64x56x56xf32>
    %v3298 = stablehlo.subtract %v3291, %v3297 : tensor<32x64x56x56xf32>
    %v3299 = stablehlo.multiply %v3298, %v3298 : tensor<32x64x56x56xf32>
    %v3300 = stablehlo.reduce(%v3299 init: %v3292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3301 = stablehlo.broadcast_in_dim %v3300, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3302 = stablehlo.divide %v3301, %v3293 : tensor<32x64x56x56xf32>
    %v3303 = stablehlo.add %v3302, %v3294 : tensor<32x64x56x56xf32>
    %v3304 = stablehlo.rsqrt %v3303 : tensor<32x64x56x56xf32>
    %v3305 = stablehlo.multiply %v3298, %v3304 : tensor<32x64x56x56xf32>
    %v3306 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3307 = stablehlo.reshape %v3290 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3308 = stablehlo.multiply %v3306, %v3307 : tensor<32x64x56x56xf32>
    %v3309 = stablehlo.reduce(%v3308 init: %v3292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3310 = stablehlo.broadcast_in_dim %v3309, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3311 = stablehlo.multiply %v3305, %v3308 : tensor<32x64x56x56xf32>
    %v3312 = stablehlo.reduce(%v3311 init: %v3292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3313 = stablehlo.broadcast_in_dim %v3312, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3314 = stablehlo.multiply %v3308, %v3293 : tensor<32x64x56x56xf32>
    %v3315 = stablehlo.subtract %v3314, %v3310 : tensor<32x64x56x56xf32>
    %v3316 = stablehlo.multiply %v3305, %v3313 : tensor<32x64x56x56xf32>
    %v3317 = stablehlo.subtract %v3315, %v3316 : tensor<32x64x56x56xf32>
    %v3318 = stablehlo.divide %v3304, %v3293 : tensor<32x64x56x56xf32>
    %v3319 = stablehlo.multiply %v3318, %v3317 : tensor<32x64x56x56xf32>
    %v3320 = stablehlo.reshape %v3319 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3321 = stablehlo.reshape %v3320 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3322 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3323 = stablehlo.transpose %v3322, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3324 = stablehlo.convolution(%v3321, %v3323)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3325 = stablehlo.reshape %v3324 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3326 = stablehlo.reshape %v3325 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3327 = stablehlo.reshape %v119 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3328 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3329 = stablehlo.compare GT, %v3327, %v3328 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3330 = stablehlo.select %v3329, %v3326, %v3328 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3331 = stablehlo.reshape %v3330 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3332 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3333 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3334 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3335 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3336 = stablehlo.reduce(%v3332 init: %v3333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3337 = stablehlo.broadcast_in_dim %v3336, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3338 = stablehlo.divide %v3337, %v3334 : tensor<32x64x56x56xf32>
    %v3339 = stablehlo.subtract %v3332, %v3338 : tensor<32x64x56x56xf32>
    %v3340 = stablehlo.multiply %v3339, %v3339 : tensor<32x64x56x56xf32>
    %v3341 = stablehlo.reduce(%v3340 init: %v3333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3342 = stablehlo.broadcast_in_dim %v3341, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3343 = stablehlo.divide %v3342, %v3334 : tensor<32x64x56x56xf32>
    %v3344 = stablehlo.add %v3343, %v3335 : tensor<32x64x56x56xf32>
    %v3345 = stablehlo.rsqrt %v3344 : tensor<32x64x56x56xf32>
    %v3346 = stablehlo.multiply %v3339, %v3345 : tensor<32x64x56x56xf32>
    %v3347 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3348 = stablehlo.reshape %v3331 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3349 = stablehlo.multiply %v3347, %v3348 : tensor<32x64x56x56xf32>
    %v3350 = stablehlo.reduce(%v3349 init: %v3333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3351 = stablehlo.broadcast_in_dim %v3350, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3352 = stablehlo.multiply %v3346, %v3349 : tensor<32x64x56x56xf32>
    %v3353 = stablehlo.reduce(%v3352 init: %v3333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3354 = stablehlo.broadcast_in_dim %v3353, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3355 = stablehlo.multiply %v3349, %v3334 : tensor<32x64x56x56xf32>
    %v3356 = stablehlo.subtract %v3355, %v3351 : tensor<32x64x56x56xf32>
    %v3357 = stablehlo.multiply %v3346, %v3354 : tensor<32x64x56x56xf32>
    %v3358 = stablehlo.subtract %v3356, %v3357 : tensor<32x64x56x56xf32>
    %v3359 = stablehlo.divide %v3345, %v3334 : tensor<32x64x56x56xf32>
    %v3360 = stablehlo.multiply %v3359, %v3358 : tensor<32x64x56x56xf32>
    %v3361 = stablehlo.reshape %v3360 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3362 = stablehlo.reshape %v3361 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3363 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3364 = stablehlo.transpose %v3363, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3365 = stablehlo.convolution(%v3362, %v3364)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3366 = stablehlo.reshape %v3365 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3367 = stablehlo.reshape %v3366 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3368 = stablehlo.reshape %v3290 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3369 = stablehlo.add %v3367, %v3368 : tensor<32x64x56x56xf32>
    %v3370 = stablehlo.reshape %v3369 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3371 = stablehlo.reshape %v94 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3372 = stablehlo.reshape %v3361 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3373 = stablehlo.transpose %v3371, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3374 = stablehlo.transpose %v3372, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3375 = stablehlo.convolution(%v3373, %v3374)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3376 = stablehlo.transpose %v3375, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3377 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3379 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3380 = stablehlo.reduce(%v3377 init: %v3378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3381 = stablehlo.broadcast_in_dim %v3380, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3382 = stablehlo.divide %v3381, %v3379 : tensor<32x64x56x56xf32>
    %v3383 = stablehlo.subtract %v3377, %v3382 : tensor<32x64x56x56xf32>
    %v3384 = stablehlo.multiply %v3383, %v3383 : tensor<32x64x56x56xf32>
    %v3385 = stablehlo.reduce(%v3384 init: %v3378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3386 = stablehlo.broadcast_in_dim %v3385, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3387 = stablehlo.divide %v3386, %v3379 : tensor<32x64x56x56xf32>
    %v3388 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3389 = stablehlo.add %v3387, %v3388 : tensor<32x64x56x56xf32>
    %v3390 = stablehlo.rsqrt %v3389 : tensor<32x64x56x56xf32>
    %v3391 = stablehlo.multiply %v3383, %v3390 : tensor<32x64x56x56xf32>
    %v3392 = stablehlo.reshape %v3331 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3393 = stablehlo.multiply %v3392, %v3391 : tensor<32x64x56x56xf32>
    %v3394 = stablehlo.reduce(%v3393 init: %v3378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3395 = stablehlo.reshape %v3331 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3397 = stablehlo.reduce(%v3395 init: %v3396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3398 = stablehlo.reshape %v123 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3399 = stablehlo.reshape %v3320 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3400 = stablehlo.transpose %v3398, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3401 = stablehlo.transpose %v3399, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3402 = stablehlo.convolution(%v3400, %v3401)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3403 = stablehlo.transpose %v3402, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3404 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3405 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3406 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3407 = stablehlo.reduce(%v3404 init: %v3405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3408 = stablehlo.broadcast_in_dim %v3407, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3409 = stablehlo.divide %v3408, %v3406 : tensor<32x64x56x56xf32>
    %v3410 = stablehlo.subtract %v3404, %v3409 : tensor<32x64x56x56xf32>
    %v3411 = stablehlo.multiply %v3410, %v3410 : tensor<32x64x56x56xf32>
    %v3412 = stablehlo.reduce(%v3411 init: %v3405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3413 = stablehlo.broadcast_in_dim %v3412, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3414 = stablehlo.divide %v3413, %v3406 : tensor<32x64x56x56xf32>
    %v3415 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3416 = stablehlo.add %v3414, %v3415 : tensor<32x64x56x56xf32>
    %v3417 = stablehlo.rsqrt %v3416 : tensor<32x64x56x56xf32>
    %v3418 = stablehlo.multiply %v3410, %v3417 : tensor<32x64x56x56xf32>
    %v3419 = stablehlo.reshape %v3290 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3420 = stablehlo.multiply %v3419, %v3418 : tensor<32x64x56x56xf32>
    %v3421 = stablehlo.reduce(%v3420 init: %v3405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3422 = stablehlo.reshape %v3290 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3424 = stablehlo.reduce(%v3422 init: %v3423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3425 = stablehlo.reshape %v3370 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3426 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3427 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3428 = stablehlo.compare GT, %v3426, %v3427 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3429 = stablehlo.select %v3428, %v3425, %v3427 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3430 = stablehlo.reshape %v3429 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3431 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3433 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3434 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3435 = stablehlo.reduce(%v3431 init: %v3432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3436 = stablehlo.broadcast_in_dim %v3435, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3437 = stablehlo.divide %v3436, %v3433 : tensor<32x64x56x56xf32>
    %v3438 = stablehlo.subtract %v3431, %v3437 : tensor<32x64x56x56xf32>
    %v3439 = stablehlo.multiply %v3438, %v3438 : tensor<32x64x56x56xf32>
    %v3440 = stablehlo.reduce(%v3439 init: %v3432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3441 = stablehlo.broadcast_in_dim %v3440, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3442 = stablehlo.divide %v3441, %v3433 : tensor<32x64x56x56xf32>
    %v3443 = stablehlo.add %v3442, %v3434 : tensor<32x64x56x56xf32>
    %v3444 = stablehlo.rsqrt %v3443 : tensor<32x64x56x56xf32>
    %v3445 = stablehlo.multiply %v3438, %v3444 : tensor<32x64x56x56xf32>
    %v3446 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3447 = stablehlo.reshape %v3430 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3448 = stablehlo.multiply %v3446, %v3447 : tensor<32x64x56x56xf32>
    %v3449 = stablehlo.reduce(%v3448 init: %v3432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3450 = stablehlo.broadcast_in_dim %v3449, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3451 = stablehlo.multiply %v3445, %v3448 : tensor<32x64x56x56xf32>
    %v3452 = stablehlo.reduce(%v3451 init: %v3432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3453 = stablehlo.broadcast_in_dim %v3452, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3454 = stablehlo.multiply %v3448, %v3433 : tensor<32x64x56x56xf32>
    %v3455 = stablehlo.subtract %v3454, %v3450 : tensor<32x64x56x56xf32>
    %v3456 = stablehlo.multiply %v3445, %v3453 : tensor<32x64x56x56xf32>
    %v3457 = stablehlo.subtract %v3455, %v3456 : tensor<32x64x56x56xf32>
    %v3458 = stablehlo.divide %v3444, %v3433 : tensor<32x64x56x56xf32>
    %v3459 = stablehlo.multiply %v3458, %v3457 : tensor<32x64x56x56xf32>
    %v3460 = stablehlo.reshape %v3459 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3461 = stablehlo.reshape %v3460 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3462 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3463 = stablehlo.transpose %v3462, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3464 = stablehlo.convolution(%v3461, %v3463)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3465 = stablehlo.reshape %v3464 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3466 = stablehlo.reshape %v3465 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3467 = stablehlo.reshape %v57 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3468 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3469 = stablehlo.compare GT, %v3467, %v3468 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3470 = stablehlo.select %v3469, %v3466, %v3468 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3471 = stablehlo.reshape %v3470 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3472 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3474 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3475 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3476 = stablehlo.reduce(%v3472 init: %v3473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3477 = stablehlo.broadcast_in_dim %v3476, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3478 = stablehlo.divide %v3477, %v3474 : tensor<32x64x56x56xf32>
    %v3479 = stablehlo.subtract %v3472, %v3478 : tensor<32x64x56x56xf32>
    %v3480 = stablehlo.multiply %v3479, %v3479 : tensor<32x64x56x56xf32>
    %v3481 = stablehlo.reduce(%v3480 init: %v3473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3482 = stablehlo.broadcast_in_dim %v3481, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3483 = stablehlo.divide %v3482, %v3474 : tensor<32x64x56x56xf32>
    %v3484 = stablehlo.add %v3483, %v3475 : tensor<32x64x56x56xf32>
    %v3485 = stablehlo.rsqrt %v3484 : tensor<32x64x56x56xf32>
    %v3486 = stablehlo.multiply %v3479, %v3485 : tensor<32x64x56x56xf32>
    %v3487 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3488 = stablehlo.reshape %v3471 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3489 = stablehlo.multiply %v3487, %v3488 : tensor<32x64x56x56xf32>
    %v3490 = stablehlo.reduce(%v3489 init: %v3473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3491 = stablehlo.broadcast_in_dim %v3490, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3492 = stablehlo.multiply %v3486, %v3489 : tensor<32x64x56x56xf32>
    %v3493 = stablehlo.reduce(%v3492 init: %v3473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3494 = stablehlo.broadcast_in_dim %v3493, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3495 = stablehlo.multiply %v3489, %v3474 : tensor<32x64x56x56xf32>
    %v3496 = stablehlo.subtract %v3495, %v3491 : tensor<32x64x56x56xf32>
    %v3497 = stablehlo.multiply %v3486, %v3494 : tensor<32x64x56x56xf32>
    %v3498 = stablehlo.subtract %v3496, %v3497 : tensor<32x64x56x56xf32>
    %v3499 = stablehlo.divide %v3485, %v3474 : tensor<32x64x56x56xf32>
    %v3500 = stablehlo.multiply %v3499, %v3498 : tensor<32x64x56x56xf32>
    %v3501 = stablehlo.reshape %v3500 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3502 = stablehlo.reshape %v3501 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3503 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3504 = stablehlo.transpose %v3503, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3505 = stablehlo.convolution(%v3502, %v3504)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3506 = stablehlo.reshape %v3505 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3507 = stablehlo.reshape %v3506 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3508 = stablehlo.reshape %v3430 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3509 = stablehlo.add %v3507, %v3508 : tensor<32x64x56x56xf32>
    %v3510 = stablehlo.reshape %v3509 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3511 = stablehlo.reshape %v32 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3512 = stablehlo.reshape %v3501 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3513 = stablehlo.transpose %v3511, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3514 = stablehlo.transpose %v3512, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3515 = stablehlo.convolution(%v3513, %v3514)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3516 = stablehlo.transpose %v3515, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3517 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3519 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3520 = stablehlo.reduce(%v3517 init: %v3518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3521 = stablehlo.broadcast_in_dim %v3520, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3522 = stablehlo.divide %v3521, %v3519 : tensor<32x64x56x56xf32>
    %v3523 = stablehlo.subtract %v3517, %v3522 : tensor<32x64x56x56xf32>
    %v3524 = stablehlo.multiply %v3523, %v3523 : tensor<32x64x56x56xf32>
    %v3525 = stablehlo.reduce(%v3524 init: %v3518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3526 = stablehlo.broadcast_in_dim %v3525, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3527 = stablehlo.divide %v3526, %v3519 : tensor<32x64x56x56xf32>
    %v3528 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3529 = stablehlo.add %v3527, %v3528 : tensor<32x64x56x56xf32>
    %v3530 = stablehlo.rsqrt %v3529 : tensor<32x64x56x56xf32>
    %v3531 = stablehlo.multiply %v3523, %v3530 : tensor<32x64x56x56xf32>
    %v3532 = stablehlo.reshape %v3471 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3533 = stablehlo.multiply %v3532, %v3531 : tensor<32x64x56x56xf32>
    %v3534 = stablehlo.reduce(%v3533 init: %v3518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3535 = stablehlo.reshape %v3471 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3537 = stablehlo.reduce(%v3535 init: %v3536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3538 = stablehlo.reshape %v61 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3539 = stablehlo.reshape %v3460 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3540 = stablehlo.transpose %v3538, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3541 = stablehlo.transpose %v3539, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3542 = stablehlo.convolution(%v3540, %v3541)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3543 = stablehlo.transpose %v3542, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3544 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3546 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3547 = stablehlo.reduce(%v3544 init: %v3545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3548 = stablehlo.broadcast_in_dim %v3547, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3549 = stablehlo.divide %v3548, %v3546 : tensor<32x64x56x56xf32>
    %v3550 = stablehlo.subtract %v3544, %v3549 : tensor<32x64x56x56xf32>
    %v3551 = stablehlo.multiply %v3550, %v3550 : tensor<32x64x56x56xf32>
    %v3552 = stablehlo.reduce(%v3551 init: %v3545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3553 = stablehlo.broadcast_in_dim %v3552, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3554 = stablehlo.divide %v3553, %v3546 : tensor<32x64x56x56xf32>
    %v3555 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3556 = stablehlo.add %v3554, %v3555 : tensor<32x64x56x56xf32>
    %v3557 = stablehlo.rsqrt %v3556 : tensor<32x64x56x56xf32>
    %v3558 = stablehlo.multiply %v3550, %v3557 : tensor<32x64x56x56xf32>
    %v3559 = stablehlo.reshape %v3430 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3560 = stablehlo.multiply %v3559, %v3558 : tensor<32x64x56x56xf32>
    %v3561 = stablehlo.reduce(%v3560 init: %v3545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3562 = stablehlo.reshape %v3430 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3564 = stablehlo.reduce(%v3562 init: %v3563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3565 = stablehlo.reshape %v28 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3566 = stablehlo.reshape %v3510 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3567 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3568 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3569 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3570 = stablehlo.pad %v3565, %v3567, low = [0, 0, 1, 1], high = [0, 0, 1, 1], interior = [0, 0, 0, 0] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3571 = stablehlo.slice %v3570 [0:32, 0:64, 0:112:2, 0:112:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3572 = stablehlo.slice %v3570 [0:32, 0:64, 0:112:2, 1:113:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3573 = stablehlo.slice %v3570 [0:32, 0:64, 0:112:2, 2:114:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3574 = stablehlo.slice %v3570 [0:32, 0:64, 1:113:2, 0:112:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3575 = stablehlo.slice %v3570 [0:32, 0:64, 1:113:2, 1:113:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3576 = stablehlo.slice %v3570 [0:32, 0:64, 1:113:2, 2:114:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3577 = stablehlo.slice %v3570 [0:32, 0:64, 2:114:2, 0:112:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3578 = stablehlo.slice %v3570 [0:32, 0:64, 2:114:2, 1:113:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3579 = stablehlo.slice %v3570 [0:32, 0:64, 2:114:2, 2:114:2] : (tensor<32x64x114x114xf32>) -> tensor<32x64x56x56xf32>
    %v3580 = stablehlo.maximum %v3571, %v3572 : tensor<32x64x56x56xf32>
    %v3581 = stablehlo.maximum %v3580, %v3573 : tensor<32x64x56x56xf32>
    %v3582 = stablehlo.maximum %v3581, %v3574 : tensor<32x64x56x56xf32>
    %v3583 = stablehlo.maximum %v3582, %v3575 : tensor<32x64x56x56xf32>
    %v3584 = stablehlo.maximum %v3583, %v3576 : tensor<32x64x56x56xf32>
    %v3585 = stablehlo.maximum %v3584, %v3577 : tensor<32x64x56x56xf32>
    %v3586 = stablehlo.maximum %v3585, %v3578 : tensor<32x64x56x56xf32>
    %v3587 = stablehlo.maximum %v3586, %v3579 : tensor<32x64x56x56xf32>
    %v3588 = stablehlo.compare EQ, %v3571, %v3587 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3589 = stablehlo.select %v3588, %v3566, %v3569 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3590 = stablehlo.pad %v3589, %v3568, low = [0, 0, 0, 0], high = [0, 0, 3, 3], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3591 = stablehlo.compare EQ, %v3572, %v3587 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3592 = stablehlo.not %v3588 : tensor<32x64x56x56xi1>
    %v3593 = stablehlo.and %v3591, %v3592 : tensor<32x64x56x56xi1>
    %v3594 = stablehlo.or %v3588, %v3591 : tensor<32x64x56x56xi1>
    %v3595 = stablehlo.select %v3593, %v3566, %v3569 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3596 = stablehlo.pad %v3595, %v3568, low = [0, 0, 0, 1], high = [0, 0, 3, 2], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3597 = stablehlo.add %v3590, %v3596 : tensor<32x64x114x114xf32>
    %v3598 = stablehlo.compare EQ, %v3573, %v3587 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3599 = stablehlo.not %v3594 : tensor<32x64x56x56xi1>
    %v3600 = stablehlo.and %v3598, %v3599 : tensor<32x64x56x56xi1>
    %v3601 = stablehlo.or %v3594, %v3598 : tensor<32x64x56x56xi1>
    %v3602 = stablehlo.select %v3600, %v3566, %v3569 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3603 = stablehlo.pad %v3602, %v3568, low = [0, 0, 0, 2], high = [0, 0, 3, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3604 = stablehlo.add %v3597, %v3603 : tensor<32x64x114x114xf32>
    %v3605 = stablehlo.compare EQ, %v3574, %v3587 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3606 = stablehlo.not %v3601 : tensor<32x64x56x56xi1>
    %v3607 = stablehlo.and %v3605, %v3606 : tensor<32x64x56x56xi1>
    %v3608 = stablehlo.or %v3601, %v3605 : tensor<32x64x56x56xi1>
    %v3609 = stablehlo.select %v3607, %v3566, %v3569 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3610 = stablehlo.pad %v3609, %v3568, low = [0, 0, 1, 0], high = [0, 0, 2, 3], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3611 = stablehlo.add %v3604, %v3610 : tensor<32x64x114x114xf32>
    %v3612 = stablehlo.compare EQ, %v3575, %v3587 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3613 = stablehlo.not %v3608 : tensor<32x64x56x56xi1>
    %v3614 = stablehlo.and %v3612, %v3613 : tensor<32x64x56x56xi1>
    %v3615 = stablehlo.or %v3608, %v3612 : tensor<32x64x56x56xi1>
    %v3616 = stablehlo.select %v3614, %v3566, %v3569 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3617 = stablehlo.pad %v3616, %v3568, low = [0, 0, 1, 1], high = [0, 0, 2, 2], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3618 = stablehlo.add %v3611, %v3617 : tensor<32x64x114x114xf32>
    %v3619 = stablehlo.compare EQ, %v3576, %v3587 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3620 = stablehlo.not %v3615 : tensor<32x64x56x56xi1>
    %v3621 = stablehlo.and %v3619, %v3620 : tensor<32x64x56x56xi1>
    %v3622 = stablehlo.or %v3615, %v3619 : tensor<32x64x56x56xi1>
    %v3623 = stablehlo.select %v3621, %v3566, %v3569 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3624 = stablehlo.pad %v3623, %v3568, low = [0, 0, 1, 2], high = [0, 0, 2, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3625 = stablehlo.add %v3618, %v3624 : tensor<32x64x114x114xf32>
    %v3626 = stablehlo.compare EQ, %v3577, %v3587 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3627 = stablehlo.not %v3622 : tensor<32x64x56x56xi1>
    %v3628 = stablehlo.and %v3626, %v3627 : tensor<32x64x56x56xi1>
    %v3629 = stablehlo.or %v3622, %v3626 : tensor<32x64x56x56xi1>
    %v3630 = stablehlo.select %v3628, %v3566, %v3569 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3631 = stablehlo.pad %v3630, %v3568, low = [0, 0, 2, 0], high = [0, 0, 1, 3], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3632 = stablehlo.add %v3625, %v3631 : tensor<32x64x114x114xf32>
    %v3633 = stablehlo.compare EQ, %v3578, %v3587 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3634 = stablehlo.not %v3629 : tensor<32x64x56x56xi1>
    %v3635 = stablehlo.and %v3633, %v3634 : tensor<32x64x56x56xi1>
    %v3636 = stablehlo.or %v3629, %v3633 : tensor<32x64x56x56xi1>
    %v3637 = stablehlo.select %v3635, %v3566, %v3569 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3638 = stablehlo.pad %v3637, %v3568, low = [0, 0, 2, 1], high = [0, 0, 1, 2], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3639 = stablehlo.add %v3632, %v3638 : tensor<32x64x114x114xf32>
    %v3640 = stablehlo.compare EQ, %v3579, %v3587 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3641 = stablehlo.not %v3636 : tensor<32x64x56x56xi1>
    %v3642 = stablehlo.and %v3640, %v3641 : tensor<32x64x56x56xi1>
    %v3643 = stablehlo.or %v3636, %v3640 : tensor<32x64x56x56xi1>
    %v3644 = stablehlo.select %v3642, %v3566, %v3569 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3645 = stablehlo.pad %v3644, %v3568, low = [0, 0, 2, 2], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x114x114xf32>
    %v3646 = stablehlo.add %v3639, %v3645 : tensor<32x64x114x114xf32>
    %v3647 = stablehlo.slice %v3646 [0:32, 0:64, 1:113, 1:113] : (tensor<32x64x114x114xf32>) -> tensor<32x64x112x112xf32>
    %v3648 = stablehlo.reshape %v3647 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3649 = stablehlo.reshape %v3648 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3650 = stablehlo.reshape %v24 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3651 = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %v3652 = stablehlo.compare GT, %v3650, %v3651 : (tensor<32x64x112x112xf32>, tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xi1>
    %v3653 = stablehlo.select %v3652, %v3649, %v3651 : tensor<32x64x112x112xi1>, tensor<32x64x112x112xf32>
    %v3654 = stablehlo.reshape %v3653 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3655 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3657 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3658 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3659 = stablehlo.reduce(%v3655 init: %v3656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3660 = stablehlo.broadcast_in_dim %v3659, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3661 = stablehlo.divide %v3660, %v3657 : tensor<32x64x112x112xf32>
    %v3662 = stablehlo.subtract %v3655, %v3661 : tensor<32x64x112x112xf32>
    %v3663 = stablehlo.multiply %v3662, %v3662 : tensor<32x64x112x112xf32>
    %v3664 = stablehlo.reduce(%v3663 init: %v3656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3665 = stablehlo.broadcast_in_dim %v3664, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3666 = stablehlo.divide %v3665, %v3657 : tensor<32x64x112x112xf32>
    %v3667 = stablehlo.add %v3666, %v3658 : tensor<32x64x112x112xf32>
    %v3668 = stablehlo.rsqrt %v3667 : tensor<32x64x112x112xf32>
    %v3669 = stablehlo.multiply %v3662, %v3668 : tensor<32x64x112x112xf32>
    %v3670 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3671 = stablehlo.reshape %v3654 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3672 = stablehlo.multiply %v3670, %v3671 : tensor<32x64x112x112xf32>
    %v3673 = stablehlo.reduce(%v3672 init: %v3656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3674 = stablehlo.broadcast_in_dim %v3673, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3675 = stablehlo.multiply %v3669, %v3672 : tensor<32x64x112x112xf32>
    %v3676 = stablehlo.reduce(%v3675 init: %v3656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3677 = stablehlo.broadcast_in_dim %v3676, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3678 = stablehlo.multiply %v3672, %v3657 : tensor<32x64x112x112xf32>
    %v3679 = stablehlo.subtract %v3678, %v3674 : tensor<32x64x112x112xf32>
    %v3680 = stablehlo.multiply %v3669, %v3677 : tensor<32x64x112x112xf32>
    %v3681 = stablehlo.subtract %v3679, %v3680 : tensor<32x64x112x112xf32>
    %v3682 = stablehlo.divide %v3668, %v3657 : tensor<32x64x112x112xf32>
    %v3683 = stablehlo.multiply %v3682, %v3681 : tensor<32x64x112x112xf32>
    %v3684 = stablehlo.reshape %v3683 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3685 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v3686 = stablehlo.reshape %v3684 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3687 = stablehlo.transpose %v3685, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v3689 = stablehlo.transpose %v3686, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %v3690 = stablehlo.convolution(%v3687, %v3689)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<64x32x112x112xf32>) -> tensor<3x64x7x7xf32>
    %v3691 = stablehlo.transpose %v3690, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v3692 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3694 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3695 = stablehlo.reduce(%v3692 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3696 = stablehlo.broadcast_in_dim %v3695, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3697 = stablehlo.divide %v3696, %v3694 : tensor<32x64x112x112xf32>
    %v3698 = stablehlo.subtract %v3692, %v3697 : tensor<32x64x112x112xf32>
    %v3699 = stablehlo.multiply %v3698, %v3698 : tensor<32x64x112x112xf32>
    %v3700 = stablehlo.reduce(%v3699 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3701 = stablehlo.broadcast_in_dim %v3700, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3702 = stablehlo.divide %v3701, %v3694 : tensor<32x64x112x112xf32>
    %v3703 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3704 = stablehlo.add %v3702, %v3703 : tensor<32x64x112x112xf32>
    %v3705 = stablehlo.rsqrt %v3704 : tensor<32x64x112x112xf32>
    %v3706 = stablehlo.multiply %v3698, %v3705 : tensor<32x64x112x112xf32>
    %v3707 = stablehlo.reshape %v3654 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3708 = stablehlo.multiply %v3707, %v3706 : tensor<32x64x112x112xf32>
    %v3709 = stablehlo.reduce(%v3708 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3710 = stablehlo.reshape %v3654 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3712 = stablehlo.reduce(%v3710 init: %v3711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3713 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3715 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3716 = stablehlo.reduce(%v3713 init: %v3714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3717 = stablehlo.divide %v3716, %v3715 : tensor<64xf32>
    %v3718 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3720 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3721 = stablehlo.reduce(%v3718 init: %v3719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3722 = stablehlo.broadcast_in_dim %v3721, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3723 = stablehlo.divide %v3722, %v3720 : tensor<32x64x112x112xf32>
    %v3724 = stablehlo.subtract %v3718, %v3723 : tensor<32x64x112x112xf32>
    %v3725 = stablehlo.multiply %v3724, %v3724 : tensor<32x64x112x112xf32>
    %v3726 = stablehlo.reduce(%v3725 init: %v3719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3727 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3728 = stablehlo.divide %v3726, %v3727 : tensor<64xf32>
    %v3729 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3731 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3732 = stablehlo.reduce(%v3729 init: %v3730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3733 = stablehlo.divide %v3732, %v3731 : tensor<64xf32>
    %v3734 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3736 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3737 = stablehlo.reduce(%v3734 init: %v3735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3738 = stablehlo.broadcast_in_dim %v3737, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3739 = stablehlo.divide %v3738, %v3736 : tensor<32x64x56x56xf32>
    %v3740 = stablehlo.subtract %v3734, %v3739 : tensor<32x64x56x56xf32>
    %v3741 = stablehlo.multiply %v3740, %v3740 : tensor<32x64x56x56xf32>
    %v3742 = stablehlo.reduce(%v3741 init: %v3735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3743 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3744 = stablehlo.divide %v3742, %v3743 : tensor<64xf32>
    %v3745 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3747 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3748 = stablehlo.reduce(%v3745 init: %v3746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3749 = stablehlo.divide %v3748, %v3747 : tensor<64xf32>
    %v3750 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3751 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3752 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3753 = stablehlo.reduce(%v3750 init: %v3751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3754 = stablehlo.broadcast_in_dim %v3753, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3755 = stablehlo.divide %v3754, %v3752 : tensor<32x64x56x56xf32>
    %v3756 = stablehlo.subtract %v3750, %v3755 : tensor<32x64x56x56xf32>
    %v3757 = stablehlo.multiply %v3756, %v3756 : tensor<32x64x56x56xf32>
    %v3758 = stablehlo.reduce(%v3757 init: %v3751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3759 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3760 = stablehlo.divide %v3758, %v3759 : tensor<64xf32>
    %v3761 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3763 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3764 = stablehlo.reduce(%v3761 init: %v3762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3765 = stablehlo.divide %v3764, %v3763 : tensor<64xf32>
    %v3766 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3768 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3769 = stablehlo.reduce(%v3766 init: %v3767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3770 = stablehlo.broadcast_in_dim %v3769, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3771 = stablehlo.divide %v3770, %v3768 : tensor<32x64x56x56xf32>
    %v3772 = stablehlo.subtract %v3766, %v3771 : tensor<32x64x56x56xf32>
    %v3773 = stablehlo.multiply %v3772, %v3772 : tensor<32x64x56x56xf32>
    %v3774 = stablehlo.reduce(%v3773 init: %v3767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3775 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3776 = stablehlo.divide %v3774, %v3775 : tensor<64xf32>
    %v3777 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3779 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3780 = stablehlo.reduce(%v3777 init: %v3778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3781 = stablehlo.divide %v3780, %v3779 : tensor<64xf32>
    %v3782 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3784 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3785 = stablehlo.reduce(%v3782 init: %v3783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3786 = stablehlo.broadcast_in_dim %v3785, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3787 = stablehlo.divide %v3786, %v3784 : tensor<32x64x56x56xf32>
    %v3788 = stablehlo.subtract %v3782, %v3787 : tensor<32x64x56x56xf32>
    %v3789 = stablehlo.multiply %v3788, %v3788 : tensor<32x64x56x56xf32>
    %v3790 = stablehlo.reduce(%v3789 init: %v3783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3791 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3792 = stablehlo.divide %v3790, %v3791 : tensor<64xf32>
    %v3793 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3794 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3795 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3796 = stablehlo.reduce(%v3793 init: %v3794) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3797 = stablehlo.divide %v3796, %v3795 : tensor<64xf32>
    %v3798 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3800 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3801 = stablehlo.reduce(%v3798 init: %v3799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3802 = stablehlo.broadcast_in_dim %v3801, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3803 = stablehlo.divide %v3802, %v3800 : tensor<32x64x56x56xf32>
    %v3804 = stablehlo.subtract %v3798, %v3803 : tensor<32x64x56x56xf32>
    %v3805 = stablehlo.multiply %v3804, %v3804 : tensor<32x64x56x56xf32>
    %v3806 = stablehlo.reduce(%v3805 init: %v3799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3807 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3808 = stablehlo.divide %v3806, %v3807 : tensor<64xf32>
    %v3809 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3811 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3812 = stablehlo.reduce(%v3809 init: %v3810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3813 = stablehlo.divide %v3812, %v3811 : tensor<64xf32>
    %v3814 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3815 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3816 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3817 = stablehlo.reduce(%v3814 init: %v3815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3818 = stablehlo.broadcast_in_dim %v3817, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3819 = stablehlo.divide %v3818, %v3816 : tensor<32x64x56x56xf32>
    %v3820 = stablehlo.subtract %v3814, %v3819 : tensor<32x64x56x56xf32>
    %v3821 = stablehlo.multiply %v3820, %v3820 : tensor<32x64x56x56xf32>
    %v3822 = stablehlo.reduce(%v3821 init: %v3815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3823 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3824 = stablehlo.divide %v3822, %v3823 : tensor<64xf32>
    %v3825 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3827 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3828 = stablehlo.reduce(%v3825 init: %v3826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3829 = stablehlo.divide %v3828, %v3827 : tensor<128xf32>
    %v3830 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3831 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3832 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3833 = stablehlo.reduce(%v3830 init: %v3831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3834 = stablehlo.broadcast_in_dim %v3833, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3835 = stablehlo.divide %v3834, %v3832 : tensor<32x128x28x28xf32>
    %v3836 = stablehlo.subtract %v3830, %v3835 : tensor<32x128x28x28xf32>
    %v3837 = stablehlo.multiply %v3836, %v3836 : tensor<32x128x28x28xf32>
    %v3838 = stablehlo.reduce(%v3837 init: %v3831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3839 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3840 = stablehlo.divide %v3838, %v3839 : tensor<128xf32>
    %v3841 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3843 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3844 = stablehlo.reduce(%v3841 init: %v3842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3845 = stablehlo.divide %v3844, %v3843 : tensor<128xf32>
    %v3846 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3847 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3848 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3849 = stablehlo.reduce(%v3846 init: %v3847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3850 = stablehlo.broadcast_in_dim %v3849, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3851 = stablehlo.divide %v3850, %v3848 : tensor<32x128x28x28xf32>
    %v3852 = stablehlo.subtract %v3846, %v3851 : tensor<32x128x28x28xf32>
    %v3853 = stablehlo.multiply %v3852, %v3852 : tensor<32x128x28x28xf32>
    %v3854 = stablehlo.reduce(%v3853 init: %v3847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3855 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3856 = stablehlo.divide %v3854, %v3855 : tensor<128xf32>
    %v3857 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3858 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3859 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3860 = stablehlo.reduce(%v3857 init: %v3858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3861 = stablehlo.divide %v3860, %v3859 : tensor<128xf32>
    %v3862 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3864 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3865 = stablehlo.reduce(%v3862 init: %v3863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3866 = stablehlo.broadcast_in_dim %v3865, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3867 = stablehlo.divide %v3866, %v3864 : tensor<32x128x28x28xf32>
    %v3868 = stablehlo.subtract %v3862, %v3867 : tensor<32x128x28x28xf32>
    %v3869 = stablehlo.multiply %v3868, %v3868 : tensor<32x128x28x28xf32>
    %v3870 = stablehlo.reduce(%v3869 init: %v3863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3871 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3872 = stablehlo.divide %v3870, %v3871 : tensor<128xf32>
    %v3873 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3875 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3876 = stablehlo.reduce(%v3873 init: %v3874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3877 = stablehlo.divide %v3876, %v3875 : tensor<128xf32>
    %v3878 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3879 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3880 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3881 = stablehlo.reduce(%v3878 init: %v3879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3882 = stablehlo.broadcast_in_dim %v3881, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3883 = stablehlo.divide %v3882, %v3880 : tensor<32x128x28x28xf32>
    %v3884 = stablehlo.subtract %v3878, %v3883 : tensor<32x128x28x28xf32>
    %v3885 = stablehlo.multiply %v3884, %v3884 : tensor<32x128x28x28xf32>
    %v3886 = stablehlo.reduce(%v3885 init: %v3879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3887 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3888 = stablehlo.divide %v3886, %v3887 : tensor<128xf32>
    %v3889 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3891 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3892 = stablehlo.reduce(%v3889 init: %v3890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3893 = stablehlo.divide %v3892, %v3891 : tensor<128xf32>
    %v3894 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3896 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3897 = stablehlo.reduce(%v3894 init: %v3895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3898 = stablehlo.broadcast_in_dim %v3897, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3899 = stablehlo.divide %v3898, %v3896 : tensor<32x128x28x28xf32>
    %v3900 = stablehlo.subtract %v3894, %v3899 : tensor<32x128x28x28xf32>
    %v3901 = stablehlo.multiply %v3900, %v3900 : tensor<32x128x28x28xf32>
    %v3902 = stablehlo.reduce(%v3901 init: %v3895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3903 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3904 = stablehlo.divide %v3902, %v3903 : tensor<128xf32>
    %v3905 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3907 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3908 = stablehlo.reduce(%v3905 init: %v3906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3909 = stablehlo.divide %v3908, %v3907 : tensor<128xf32>
    %v3910 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3912 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3913 = stablehlo.reduce(%v3910 init: %v3911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3914 = stablehlo.broadcast_in_dim %v3913, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3915 = stablehlo.divide %v3914, %v3912 : tensor<32x128x28x28xf32>
    %v3916 = stablehlo.subtract %v3910, %v3915 : tensor<32x128x28x28xf32>
    %v3917 = stablehlo.multiply %v3916, %v3916 : tensor<32x128x28x28xf32>
    %v3918 = stablehlo.reduce(%v3917 init: %v3911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3919 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3920 = stablehlo.divide %v3918, %v3919 : tensor<128xf32>
    %v3921 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3922 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3923 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3924 = stablehlo.reduce(%v3921 init: %v3922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3925 = stablehlo.divide %v3924, %v3923 : tensor<128xf32>
    %v3926 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3928 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3929 = stablehlo.reduce(%v3926 init: %v3927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3930 = stablehlo.broadcast_in_dim %v3929, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3931 = stablehlo.divide %v3930, %v3928 : tensor<32x128x28x28xf32>
    %v3932 = stablehlo.subtract %v3926, %v3931 : tensor<32x128x28x28xf32>
    %v3933 = stablehlo.multiply %v3932, %v3932 : tensor<32x128x28x28xf32>
    %v3934 = stablehlo.reduce(%v3933 init: %v3927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3935 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3936 = stablehlo.divide %v3934, %v3935 : tensor<128xf32>
    %v3937 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3939 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3940 = stablehlo.reduce(%v3937 init: %v3938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3941 = stablehlo.divide %v3940, %v3939 : tensor<128xf32>
    %v3942 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3943 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3944 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3945 = stablehlo.reduce(%v3942 init: %v3943) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3946 = stablehlo.broadcast_in_dim %v3945, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3947 = stablehlo.divide %v3946, %v3944 : tensor<32x128x28x28xf32>
    %v3948 = stablehlo.subtract %v3942, %v3947 : tensor<32x128x28x28xf32>
    %v3949 = stablehlo.multiply %v3948, %v3948 : tensor<32x128x28x28xf32>
    %v3950 = stablehlo.reduce(%v3949 init: %v3943) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3951 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3952 = stablehlo.divide %v3950, %v3951 : tensor<128xf32>
    %v3953 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3955 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3956 = stablehlo.reduce(%v3953 init: %v3954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3957 = stablehlo.divide %v3956, %v3955 : tensor<128xf32>
    %v3958 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3960 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3961 = stablehlo.reduce(%v3958 init: %v3959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3962 = stablehlo.broadcast_in_dim %v3961, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3963 = stablehlo.divide %v3962, %v3960 : tensor<32x128x28x28xf32>
    %v3964 = stablehlo.subtract %v3958, %v3963 : tensor<32x128x28x28xf32>
    %v3965 = stablehlo.multiply %v3964, %v3964 : tensor<32x128x28x28xf32>
    %v3966 = stablehlo.reduce(%v3965 init: %v3959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3967 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3968 = stablehlo.divide %v3966, %v3967 : tensor<128xf32>
    %v3969 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3971 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3972 = stablehlo.reduce(%v3969 init: %v3970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3973 = stablehlo.divide %v3972, %v3971 : tensor<256xf32>
    %v3974 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3976 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3977 = stablehlo.reduce(%v3974 init: %v3975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3978 = stablehlo.broadcast_in_dim %v3977, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3979 = stablehlo.divide %v3978, %v3976 : tensor<32x256x14x14xf32>
    %v3980 = stablehlo.subtract %v3974, %v3979 : tensor<32x256x14x14xf32>
    %v3981 = stablehlo.multiply %v3980, %v3980 : tensor<32x256x14x14xf32>
    %v3982 = stablehlo.reduce(%v3981 init: %v3975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3983 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3984 = stablehlo.divide %v3982, %v3983 : tensor<256xf32>
    %v3985 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3987 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3988 = stablehlo.reduce(%v3985 init: %v3986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3989 = stablehlo.divide %v3988, %v3987 : tensor<256xf32>
    %v3990 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3992 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3993 = stablehlo.reduce(%v3990 init: %v3991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3994 = stablehlo.broadcast_in_dim %v3993, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3995 = stablehlo.divide %v3994, %v3992 : tensor<32x256x14x14xf32>
    %v3996 = stablehlo.subtract %v3990, %v3995 : tensor<32x256x14x14xf32>
    %v3997 = stablehlo.multiply %v3996, %v3996 : tensor<32x256x14x14xf32>
    %v3998 = stablehlo.reduce(%v3997 init: %v3991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3999 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4000 = stablehlo.divide %v3998, %v3999 : tensor<256xf32>
    %v4001 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4003 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4004 = stablehlo.reduce(%v4001 init: %v4002) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4005 = stablehlo.divide %v4004, %v4003 : tensor<256xf32>
    %v4006 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4007 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4008 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4009 = stablehlo.reduce(%v4006 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4010 = stablehlo.broadcast_in_dim %v4009, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4011 = stablehlo.divide %v4010, %v4008 : tensor<32x256x14x14xf32>
    %v4012 = stablehlo.subtract %v4006, %v4011 : tensor<32x256x14x14xf32>
    %v4013 = stablehlo.multiply %v4012, %v4012 : tensor<32x256x14x14xf32>
    %v4014 = stablehlo.reduce(%v4013 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4015 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4016 = stablehlo.divide %v4014, %v4015 : tensor<256xf32>
    %v4017 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4019 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4020 = stablehlo.reduce(%v4017 init: %v4018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4021 = stablehlo.divide %v4020, %v4019 : tensor<256xf32>
    %v4022 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4024 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4025 = stablehlo.reduce(%v4022 init: %v4023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4026 = stablehlo.broadcast_in_dim %v4025, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4027 = stablehlo.divide %v4026, %v4024 : tensor<32x256x14x14xf32>
    %v4028 = stablehlo.subtract %v4022, %v4027 : tensor<32x256x14x14xf32>
    %v4029 = stablehlo.multiply %v4028, %v4028 : tensor<32x256x14x14xf32>
    %v4030 = stablehlo.reduce(%v4029 init: %v4023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4031 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4032 = stablehlo.divide %v4030, %v4031 : tensor<256xf32>
    %v4033 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4034 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4035 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4036 = stablehlo.reduce(%v4033 init: %v4034) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4037 = stablehlo.divide %v4036, %v4035 : tensor<256xf32>
    %v4038 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4040 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4041 = stablehlo.reduce(%v4038 init: %v4039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4042 = stablehlo.broadcast_in_dim %v4041, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4043 = stablehlo.divide %v4042, %v4040 : tensor<32x256x14x14xf32>
    %v4044 = stablehlo.subtract %v4038, %v4043 : tensor<32x256x14x14xf32>
    %v4045 = stablehlo.multiply %v4044, %v4044 : tensor<32x256x14x14xf32>
    %v4046 = stablehlo.reduce(%v4045 init: %v4039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4047 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4048 = stablehlo.divide %v4046, %v4047 : tensor<256xf32>
    %v4049 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4051 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4052 = stablehlo.reduce(%v4049 init: %v4050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4053 = stablehlo.divide %v4052, %v4051 : tensor<256xf32>
    %v4054 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4056 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4057 = stablehlo.reduce(%v4054 init: %v4055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4058 = stablehlo.broadcast_in_dim %v4057, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4059 = stablehlo.divide %v4058, %v4056 : tensor<32x256x14x14xf32>
    %v4060 = stablehlo.subtract %v4054, %v4059 : tensor<32x256x14x14xf32>
    %v4061 = stablehlo.multiply %v4060, %v4060 : tensor<32x256x14x14xf32>
    %v4062 = stablehlo.reduce(%v4061 init: %v4055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4063 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4064 = stablehlo.divide %v4062, %v4063 : tensor<256xf32>
    %v4065 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4066 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4067 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4068 = stablehlo.reduce(%v4065 init: %v4066) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4069 = stablehlo.divide %v4068, %v4067 : tensor<256xf32>
    %v4070 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4071 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4072 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4073 = stablehlo.reduce(%v4070 init: %v4071) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4074 = stablehlo.broadcast_in_dim %v4073, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4075 = stablehlo.divide %v4074, %v4072 : tensor<32x256x14x14xf32>
    %v4076 = stablehlo.subtract %v4070, %v4075 : tensor<32x256x14x14xf32>
    %v4077 = stablehlo.multiply %v4076, %v4076 : tensor<32x256x14x14xf32>
    %v4078 = stablehlo.reduce(%v4077 init: %v4071) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4079 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4080 = stablehlo.divide %v4078, %v4079 : tensor<256xf32>
    %v4081 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4082 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4083 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4084 = stablehlo.reduce(%v4081 init: %v4082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4085 = stablehlo.divide %v4084, %v4083 : tensor<256xf32>
    %v4086 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4088 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4089 = stablehlo.reduce(%v4086 init: %v4087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4090 = stablehlo.broadcast_in_dim %v4089, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4091 = stablehlo.divide %v4090, %v4088 : tensor<32x256x14x14xf32>
    %v4092 = stablehlo.subtract %v4086, %v4091 : tensor<32x256x14x14xf32>
    %v4093 = stablehlo.multiply %v4092, %v4092 : tensor<32x256x14x14xf32>
    %v4094 = stablehlo.reduce(%v4093 init: %v4087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4095 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4096 = stablehlo.divide %v4094, %v4095 : tensor<256xf32>
    %v4097 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4098 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4099 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4100 = stablehlo.reduce(%v4097 init: %v4098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4101 = stablehlo.divide %v4100, %v4099 : tensor<256xf32>
    %v4102 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4104 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4105 = stablehlo.reduce(%v4102 init: %v4103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4106 = stablehlo.broadcast_in_dim %v4105, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4107 = stablehlo.divide %v4106, %v4104 : tensor<32x256x14x14xf32>
    %v4108 = stablehlo.subtract %v4102, %v4107 : tensor<32x256x14x14xf32>
    %v4109 = stablehlo.multiply %v4108, %v4108 : tensor<32x256x14x14xf32>
    %v4110 = stablehlo.reduce(%v4109 init: %v4103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4111 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4112 = stablehlo.divide %v4110, %v4111 : tensor<256xf32>
    %v4113 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4115 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4116 = stablehlo.reduce(%v4113 init: %v4114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4117 = stablehlo.divide %v4116, %v4115 : tensor<256xf32>
    %v4118 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4120 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4121 = stablehlo.reduce(%v4118 init: %v4119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4122 = stablehlo.broadcast_in_dim %v4121, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4123 = stablehlo.divide %v4122, %v4120 : tensor<32x256x14x14xf32>
    %v4124 = stablehlo.subtract %v4118, %v4123 : tensor<32x256x14x14xf32>
    %v4125 = stablehlo.multiply %v4124, %v4124 : tensor<32x256x14x14xf32>
    %v4126 = stablehlo.reduce(%v4125 init: %v4119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4127 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4128 = stablehlo.divide %v4126, %v4127 : tensor<256xf32>
    %v4129 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4131 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4132 = stablehlo.reduce(%v4129 init: %v4130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4133 = stablehlo.divide %v4132, %v4131 : tensor<256xf32>
    %v4134 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4135 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4136 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4137 = stablehlo.reduce(%v4134 init: %v4135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4138 = stablehlo.broadcast_in_dim %v4137, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4139 = stablehlo.divide %v4138, %v4136 : tensor<32x256x14x14xf32>
    %v4140 = stablehlo.subtract %v4134, %v4139 : tensor<32x256x14x14xf32>
    %v4141 = stablehlo.multiply %v4140, %v4140 : tensor<32x256x14x14xf32>
    %v4142 = stablehlo.reduce(%v4141 init: %v4135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4143 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4144 = stablehlo.divide %v4142, %v4143 : tensor<256xf32>
    %v4145 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4147 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4148 = stablehlo.reduce(%v4145 init: %v4146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4149 = stablehlo.divide %v4148, %v4147 : tensor<256xf32>
    %v4150 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4152 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4153 = stablehlo.reduce(%v4150 init: %v4151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4154 = stablehlo.broadcast_in_dim %v4153, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4155 = stablehlo.divide %v4154, %v4152 : tensor<32x256x14x14xf32>
    %v4156 = stablehlo.subtract %v4150, %v4155 : tensor<32x256x14x14xf32>
    %v4157 = stablehlo.multiply %v4156, %v4156 : tensor<32x256x14x14xf32>
    %v4158 = stablehlo.reduce(%v4157 init: %v4151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4159 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4160 = stablehlo.divide %v4158, %v4159 : tensor<256xf32>
    %v4161 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4163 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4164 = stablehlo.reduce(%v4161 init: %v4162) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4165 = stablehlo.divide %v4164, %v4163 : tensor<256xf32>
    %v4166 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4168 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4169 = stablehlo.reduce(%v4166 init: %v4167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4170 = stablehlo.broadcast_in_dim %v4169, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4171 = stablehlo.divide %v4170, %v4168 : tensor<32x256x14x14xf32>
    %v4172 = stablehlo.subtract %v4166, %v4171 : tensor<32x256x14x14xf32>
    %v4173 = stablehlo.multiply %v4172, %v4172 : tensor<32x256x14x14xf32>
    %v4174 = stablehlo.reduce(%v4173 init: %v4167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4175 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4176 = stablehlo.divide %v4174, %v4175 : tensor<256xf32>
    %v4177 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4179 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4180 = stablehlo.reduce(%v4177 init: %v4178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4181 = stablehlo.divide %v4180, %v4179 : tensor<512xf32>
    %v4182 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4183 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4184 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4185 = stablehlo.reduce(%v4182 init: %v4183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4186 = stablehlo.broadcast_in_dim %v4185, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4187 = stablehlo.divide %v4186, %v4184 : tensor<32x512x7x7xf32>
    %v4188 = stablehlo.subtract %v4182, %v4187 : tensor<32x512x7x7xf32>
    %v4189 = stablehlo.multiply %v4188, %v4188 : tensor<32x512x7x7xf32>
    %v4190 = stablehlo.reduce(%v4189 init: %v4183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4191 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4192 = stablehlo.divide %v4190, %v4191 : tensor<512xf32>
    %v4193 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4195 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4196 = stablehlo.reduce(%v4193 init: %v4194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4197 = stablehlo.divide %v4196, %v4195 : tensor<512xf32>
    %v4198 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4200 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4201 = stablehlo.reduce(%v4198 init: %v4199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4202 = stablehlo.broadcast_in_dim %v4201, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4203 = stablehlo.divide %v4202, %v4200 : tensor<32x512x7x7xf32>
    %v4204 = stablehlo.subtract %v4198, %v4203 : tensor<32x512x7x7xf32>
    %v4205 = stablehlo.multiply %v4204, %v4204 : tensor<32x512x7x7xf32>
    %v4206 = stablehlo.reduce(%v4205 init: %v4199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4207 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4208 = stablehlo.divide %v4206, %v4207 : tensor<512xf32>
    %v4209 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4210 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4211 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4212 = stablehlo.reduce(%v4209 init: %v4210) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4213 = stablehlo.divide %v4212, %v4211 : tensor<512xf32>
    %v4214 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4216 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4217 = stablehlo.reduce(%v4214 init: %v4215) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4218 = stablehlo.broadcast_in_dim %v4217, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4219 = stablehlo.divide %v4218, %v4216 : tensor<32x512x7x7xf32>
    %v4220 = stablehlo.subtract %v4214, %v4219 : tensor<32x512x7x7xf32>
    %v4221 = stablehlo.multiply %v4220, %v4220 : tensor<32x512x7x7xf32>
    %v4222 = stablehlo.reduce(%v4221 init: %v4215) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4223 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4224 = stablehlo.divide %v4222, %v4223 : tensor<512xf32>
    %v4225 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4227 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4228 = stablehlo.reduce(%v4225 init: %v4226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4229 = stablehlo.divide %v4228, %v4227 : tensor<512xf32>
    %v4230 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4232 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4233 = stablehlo.reduce(%v4230 init: %v4231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4234 = stablehlo.broadcast_in_dim %v4233, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4235 = stablehlo.divide %v4234, %v4232 : tensor<32x512x7x7xf32>
    %v4236 = stablehlo.subtract %v4230, %v4235 : tensor<32x512x7x7xf32>
    %v4237 = stablehlo.multiply %v4236, %v4236 : tensor<32x512x7x7xf32>
    %v4238 = stablehlo.reduce(%v4237 init: %v4231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4239 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4240 = stablehlo.divide %v4238, %v4239 : tensor<512xf32>
    %v4241 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4243 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4244 = stablehlo.reduce(%v4241 init: %v4242) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4245 = stablehlo.divide %v4244, %v4243 : tensor<512xf32>
    %v4246 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4247 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4248 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4249 = stablehlo.reduce(%v4246 init: %v4247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4250 = stablehlo.broadcast_in_dim %v4249, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4251 = stablehlo.divide %v4250, %v4248 : tensor<32x512x7x7xf32>
    %v4252 = stablehlo.subtract %v4246, %v4251 : tensor<32x512x7x7xf32>
    %v4253 = stablehlo.multiply %v4252, %v4252 : tensor<32x512x7x7xf32>
    %v4254 = stablehlo.reduce(%v4253 init: %v4247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4255 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4256 = stablehlo.divide %v4254, %v4255 : tensor<512xf32>
    %v4257 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4259 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4260 = stablehlo.reduce(%v4257 init: %v4258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4261 = stablehlo.divide %v4260, %v4259 : tensor<512xf32>
    %v4262 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4263 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4264 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4265 = stablehlo.reduce(%v4262 init: %v4263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4266 = stablehlo.broadcast_in_dim %v4265, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4267 = stablehlo.divide %v4266, %v4264 : tensor<32x512x7x7xf32>
    %v4268 = stablehlo.subtract %v4262, %v4267 : tensor<32x512x7x7xf32>
    %v4269 = stablehlo.multiply %v4268, %v4268 : tensor<32x512x7x7xf32>
    %v4270 = stablehlo.reduce(%v4269 init: %v4263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4271 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4272 = stablehlo.divide %v4270, %v4271 : tensor<512xf32>
    %v4273 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4274 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4275 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4276 = stablehlo.reduce(%v4273 init: %v4274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4277 = stablehlo.divide %v4276, %v4275 : tensor<512xf32>
    %v4278 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4280 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4281 = stablehlo.reduce(%v4278 init: %v4279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4282 = stablehlo.broadcast_in_dim %v4281, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4283 = stablehlo.divide %v4282, %v4280 : tensor<32x512x7x7xf32>
    %v4284 = stablehlo.subtract %v4278, %v4283 : tensor<32x512x7x7xf32>
    %v4285 = stablehlo.multiply %v4284, %v4284 : tensor<32x512x7x7xf32>
    %v4286 = stablehlo.reduce(%v4285 init: %v4279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4287 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4288 = stablehlo.divide %v4286, %v4287 : tensor<512xf32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0> : tensor<f32>
    %v4289 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4290 = stablehlo.multiply %v4289, %sW : tensor<64x3x7x7xf32>
    %v4291 = stablehlo.add %v4290, %v3691 : tensor<64x3x7x7xf32>
    %v4292 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4293 = stablehlo.multiply %v4292, %sWv : tensor<64x3x7x7xf32>
    %v4294 = stablehlo.add %v4293, %v4291 : tensor<64x3x7x7xf32>
    %v4295 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4296 = stablehlo.multiply %v4295, %v4294 : tensor<64x3x7x7xf32>
    %v4297 = stablehlo.subtract %sW, %v4296 : tensor<64x3x7x7xf32>
    %v4298 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4299 = stablehlo.multiply %v4298, %sg : tensor<64xf32>
    %v4300 = stablehlo.add %v4299, %v3709 : tensor<64xf32>
    %v4301 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4302 = stablehlo.multiply %v4301, %sgv : tensor<64xf32>
    %v4303 = stablehlo.add %v4302, %v4300 : tensor<64xf32>
    %v4304 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4305 = stablehlo.multiply %v4304, %v4303 : tensor<64xf32>
    %v4306 = stablehlo.subtract %sg, %v4305 : tensor<64xf32>
    %v4307 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4308 = stablehlo.multiply %v4307, %sbt : tensor<64xf32>
    %v4309 = stablehlo.add %v4308, %v3712 : tensor<64xf32>
    %v4310 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4311 = stablehlo.multiply %v4310, %sbtv : tensor<64xf32>
    %v4312 = stablehlo.add %v4311, %v4309 : tensor<64xf32>
    %v4313 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4314 = stablehlo.multiply %v4313, %v4312 : tensor<64xf32>
    %v4315 = stablehlo.subtract %sbt, %v4314 : tensor<64xf32>
    %v4316 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4317 = stablehlo.multiply %v4316, %s1b0W1 : tensor<64x64x3x3xf32>
    %v4318 = stablehlo.add %v4317, %v3516 : tensor<64x64x3x3xf32>
    %v4319 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4320 = stablehlo.multiply %v4319, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4321 = stablehlo.add %v4320, %v4318 : tensor<64x64x3x3xf32>
    %v4322 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4323 = stablehlo.multiply %v4322, %v4321 : tensor<64x64x3x3xf32>
    %v4324 = stablehlo.subtract %s1b0W1, %v4323 : tensor<64x64x3x3xf32>
    %v4325 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4326 = stablehlo.multiply %v4325, %s1b0g1 : tensor<64xf32>
    %v4327 = stablehlo.add %v4326, %v3534 : tensor<64xf32>
    %v4328 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4329 = stablehlo.multiply %v4328, %s1b0g1v : tensor<64xf32>
    %v4330 = stablehlo.add %v4329, %v4327 : tensor<64xf32>
    %v4331 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4332 = stablehlo.multiply %v4331, %v4330 : tensor<64xf32>
    %v4333 = stablehlo.subtract %s1b0g1, %v4332 : tensor<64xf32>
    %v4334 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4335 = stablehlo.multiply %v4334, %s1b0bt1 : tensor<64xf32>
    %v4336 = stablehlo.add %v4335, %v3537 : tensor<64xf32>
    %v4337 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4338 = stablehlo.multiply %v4337, %s1b0bt1v : tensor<64xf32>
    %v4339 = stablehlo.add %v4338, %v4336 : tensor<64xf32>
    %v4340 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4341 = stablehlo.multiply %v4340, %v4339 : tensor<64xf32>
    %v4342 = stablehlo.subtract %s1b0bt1, %v4341 : tensor<64xf32>
    %v4343 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4344 = stablehlo.multiply %v4343, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4345 = stablehlo.add %v4344, %v3543 : tensor<64x64x3x3xf32>
    %v4346 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4347 = stablehlo.multiply %v4346, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4348 = stablehlo.add %v4347, %v4345 : tensor<64x64x3x3xf32>
    %v4349 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4350 = stablehlo.multiply %v4349, %v4348 : tensor<64x64x3x3xf32>
    %v4351 = stablehlo.subtract %s1b0W2, %v4350 : tensor<64x64x3x3xf32>
    %v4352 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4353 = stablehlo.multiply %v4352, %s1b0g2 : tensor<64xf32>
    %v4354 = stablehlo.add %v4353, %v3561 : tensor<64xf32>
    %v4355 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4356 = stablehlo.multiply %v4355, %s1b0g2v : tensor<64xf32>
    %v4357 = stablehlo.add %v4356, %v4354 : tensor<64xf32>
    %v4358 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4359 = stablehlo.multiply %v4358, %v4357 : tensor<64xf32>
    %v4360 = stablehlo.subtract %s1b0g2, %v4359 : tensor<64xf32>
    %v4361 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4362 = stablehlo.multiply %v4361, %s1b0bt2 : tensor<64xf32>
    %v4363 = stablehlo.add %v4362, %v3564 : tensor<64xf32>
    %v4364 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4365 = stablehlo.multiply %v4364, %s1b0bt2v : tensor<64xf32>
    %v4366 = stablehlo.add %v4365, %v4363 : tensor<64xf32>
    %v4367 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4368 = stablehlo.multiply %v4367, %v4366 : tensor<64xf32>
    %v4369 = stablehlo.subtract %s1b0bt2, %v4368 : tensor<64xf32>
    %v4370 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4371 = stablehlo.multiply %v4370, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4372 = stablehlo.add %v4371, %v3376 : tensor<64x64x3x3xf32>
    %v4373 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4374 = stablehlo.multiply %v4373, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4375 = stablehlo.add %v4374, %v4372 : tensor<64x64x3x3xf32>
    %v4376 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4377 = stablehlo.multiply %v4376, %v4375 : tensor<64x64x3x3xf32>
    %v4378 = stablehlo.subtract %s1b1W1, %v4377 : tensor<64x64x3x3xf32>
    %v4379 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4380 = stablehlo.multiply %v4379, %s1b1g1 : tensor<64xf32>
    %v4381 = stablehlo.add %v4380, %v3394 : tensor<64xf32>
    %v4382 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4383 = stablehlo.multiply %v4382, %s1b1g1v : tensor<64xf32>
    %v4384 = stablehlo.add %v4383, %v4381 : tensor<64xf32>
    %v4385 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4386 = stablehlo.multiply %v4385, %v4384 : tensor<64xf32>
    %v4387 = stablehlo.subtract %s1b1g1, %v4386 : tensor<64xf32>
    %v4388 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4389 = stablehlo.multiply %v4388, %s1b1bt1 : tensor<64xf32>
    %v4390 = stablehlo.add %v4389, %v3397 : tensor<64xf32>
    %v4391 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4392 = stablehlo.multiply %v4391, %s1b1bt1v : tensor<64xf32>
    %v4393 = stablehlo.add %v4392, %v4390 : tensor<64xf32>
    %v4394 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4395 = stablehlo.multiply %v4394, %v4393 : tensor<64xf32>
    %v4396 = stablehlo.subtract %s1b1bt1, %v4395 : tensor<64xf32>
    %v4397 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4398 = stablehlo.multiply %v4397, %s1b1W2 : tensor<64x64x3x3xf32>
    %v4399 = stablehlo.add %v4398, %v3403 : tensor<64x64x3x3xf32>
    %v4400 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4401 = stablehlo.multiply %v4400, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4402 = stablehlo.add %v4401, %v4399 : tensor<64x64x3x3xf32>
    %v4403 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4404 = stablehlo.multiply %v4403, %v4402 : tensor<64x64x3x3xf32>
    %v4405 = stablehlo.subtract %s1b1W2, %v4404 : tensor<64x64x3x3xf32>
    %v4406 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4407 = stablehlo.multiply %v4406, %s1b1g2 : tensor<64xf32>
    %v4408 = stablehlo.add %v4407, %v3421 : tensor<64xf32>
    %v4409 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4410 = stablehlo.multiply %v4409, %s1b1g2v : tensor<64xf32>
    %v4411 = stablehlo.add %v4410, %v4408 : tensor<64xf32>
    %v4412 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4413 = stablehlo.multiply %v4412, %v4411 : tensor<64xf32>
    %v4414 = stablehlo.subtract %s1b1g2, %v4413 : tensor<64xf32>
    %v4415 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4416 = stablehlo.multiply %v4415, %s1b1bt2 : tensor<64xf32>
    %v4417 = stablehlo.add %v4416, %v3424 : tensor<64xf32>
    %v4418 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4419 = stablehlo.multiply %v4418, %s1b1bt2v : tensor<64xf32>
    %v4420 = stablehlo.add %v4419, %v4417 : tensor<64xf32>
    %v4421 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4422 = stablehlo.multiply %v4421, %v4420 : tensor<64xf32>
    %v4423 = stablehlo.subtract %s1b1bt2, %v4422 : tensor<64xf32>
    %v4424 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4425 = stablehlo.multiply %v4424, %s1b2W1 : tensor<64x64x3x3xf32>
    %v4426 = stablehlo.add %v4425, %v3236 : tensor<64x64x3x3xf32>
    %v4427 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4428 = stablehlo.multiply %v4427, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4429 = stablehlo.add %v4428, %v4426 : tensor<64x64x3x3xf32>
    %v4430 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4431 = stablehlo.multiply %v4430, %v4429 : tensor<64x64x3x3xf32>
    %v4432 = stablehlo.subtract %s1b2W1, %v4431 : tensor<64x64x3x3xf32>
    %v4433 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4434 = stablehlo.multiply %v4433, %s1b2g1 : tensor<64xf32>
    %v4435 = stablehlo.add %v4434, %v3254 : tensor<64xf32>
    %v4436 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4437 = stablehlo.multiply %v4436, %s1b2g1v : tensor<64xf32>
    %v4438 = stablehlo.add %v4437, %v4435 : tensor<64xf32>
    %v4439 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4440 = stablehlo.multiply %v4439, %v4438 : tensor<64xf32>
    %v4441 = stablehlo.subtract %s1b2g1, %v4440 : tensor<64xf32>
    %v4442 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4443 = stablehlo.multiply %v4442, %s1b2bt1 : tensor<64xf32>
    %v4444 = stablehlo.add %v4443, %v3257 : tensor<64xf32>
    %v4445 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4446 = stablehlo.multiply %v4445, %s1b2bt1v : tensor<64xf32>
    %v4447 = stablehlo.add %v4446, %v4444 : tensor<64xf32>
    %v4448 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4449 = stablehlo.multiply %v4448, %v4447 : tensor<64xf32>
    %v4450 = stablehlo.subtract %s1b2bt1, %v4449 : tensor<64xf32>
    %v4451 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4452 = stablehlo.multiply %v4451, %s1b2W2 : tensor<64x64x3x3xf32>
    %v4453 = stablehlo.add %v4452, %v3263 : tensor<64x64x3x3xf32>
    %v4454 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4455 = stablehlo.multiply %v4454, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4456 = stablehlo.add %v4455, %v4453 : tensor<64x64x3x3xf32>
    %v4457 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4458 = stablehlo.multiply %v4457, %v4456 : tensor<64x64x3x3xf32>
    %v4459 = stablehlo.subtract %s1b2W2, %v4458 : tensor<64x64x3x3xf32>
    %v4460 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4461 = stablehlo.multiply %v4460, %s1b2g2 : tensor<64xf32>
    %v4462 = stablehlo.add %v4461, %v3281 : tensor<64xf32>
    %v4463 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4464 = stablehlo.multiply %v4463, %s1b2g2v : tensor<64xf32>
    %v4465 = stablehlo.add %v4464, %v4462 : tensor<64xf32>
    %v4466 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4467 = stablehlo.multiply %v4466, %v4465 : tensor<64xf32>
    %v4468 = stablehlo.subtract %s1b2g2, %v4467 : tensor<64xf32>
    %v4469 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4470 = stablehlo.multiply %v4469, %s1b2bt2 : tensor<64xf32>
    %v4471 = stablehlo.add %v4470, %v3284 : tensor<64xf32>
    %v4472 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4473 = stablehlo.multiply %v4472, %s1b2bt2v : tensor<64xf32>
    %v4474 = stablehlo.add %v4473, %v4471 : tensor<64xf32>
    %v4475 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4476 = stablehlo.multiply %v4475, %v4474 : tensor<64xf32>
    %v4477 = stablehlo.subtract %s1b2bt2, %v4476 : tensor<64xf32>
    %v4478 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4479 = stablehlo.multiply %v4478, %d2W1 : tensor<128x64x3x3xf32>
    %v4480 = stablehlo.add %v4479, %v3068 : tensor<128x64x3x3xf32>
    %v4481 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4482 = stablehlo.multiply %v4481, %d2W1v : tensor<128x64x3x3xf32>
    %v4483 = stablehlo.add %v4482, %v4480 : tensor<128x64x3x3xf32>
    %v4484 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4485 = stablehlo.multiply %v4484, %v4483 : tensor<128x64x3x3xf32>
    %v4486 = stablehlo.subtract %d2W1, %v4485 : tensor<128x64x3x3xf32>
    %v4487 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4488 = stablehlo.multiply %v4487, %d2g1 : tensor<128xf32>
    %v4489 = stablehlo.add %v4488, %v3086 : tensor<128xf32>
    %v4490 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4491 = stablehlo.multiply %v4490, %d2g1v : tensor<128xf32>
    %v4492 = stablehlo.add %v4491, %v4489 : tensor<128xf32>
    %v4493 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4494 = stablehlo.multiply %v4493, %v4492 : tensor<128xf32>
    %v4495 = stablehlo.subtract %d2g1, %v4494 : tensor<128xf32>
    %v4496 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4497 = stablehlo.multiply %v4496, %d2bt1 : tensor<128xf32>
    %v4498 = stablehlo.add %v4497, %v3089 : tensor<128xf32>
    %v4499 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4500 = stablehlo.multiply %v4499, %d2bt1v : tensor<128xf32>
    %v4501 = stablehlo.add %v4500, %v4498 : tensor<128xf32>
    %v4502 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4503 = stablehlo.multiply %v4502, %v4501 : tensor<128xf32>
    %v4504 = stablehlo.subtract %d2bt1, %v4503 : tensor<128xf32>
    %v4505 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4506 = stablehlo.multiply %v4505, %d2W2 : tensor<128x128x3x3xf32>
    %v4507 = stablehlo.add %v4506, %v3095 : tensor<128x128x3x3xf32>
    %v4508 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4509 = stablehlo.multiply %v4508, %d2W2v : tensor<128x128x3x3xf32>
    %v4510 = stablehlo.add %v4509, %v4507 : tensor<128x128x3x3xf32>
    %v4511 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4512 = stablehlo.multiply %v4511, %v4510 : tensor<128x128x3x3xf32>
    %v4513 = stablehlo.subtract %d2W2, %v4512 : tensor<128x128x3x3xf32>
    %v4514 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4515 = stablehlo.multiply %v4514, %d2g2 : tensor<128xf32>
    %v4516 = stablehlo.add %v4515, %v3113 : tensor<128xf32>
    %v4517 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4518 = stablehlo.multiply %v4517, %d2g2v : tensor<128xf32>
    %v4519 = stablehlo.add %v4518, %v4516 : tensor<128xf32>
    %v4520 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4521 = stablehlo.multiply %v4520, %v4519 : tensor<128xf32>
    %v4522 = stablehlo.subtract %d2g2, %v4521 : tensor<128xf32>
    %v4523 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4524 = stablehlo.multiply %v4523, %d2bt2 : tensor<128xf32>
    %v4525 = stablehlo.add %v4524, %v3116 : tensor<128xf32>
    %v4526 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4527 = stablehlo.multiply %v4526, %d2bt2v : tensor<128xf32>
    %v4528 = stablehlo.add %v4527, %v4525 : tensor<128xf32>
    %v4529 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4530 = stablehlo.multiply %v4529, %v4528 : tensor<128xf32>
    %v4531 = stablehlo.subtract %d2bt2, %v4530 : tensor<128xf32>
    %v4532 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4533 = stablehlo.multiply %v4532, %d2Wp : tensor<128x64x1x1xf32>
    %v4534 = stablehlo.add %v4533, %v3123 : tensor<128x64x1x1xf32>
    %v4535 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4536 = stablehlo.multiply %v4535, %d2Wpv : tensor<128x64x1x1xf32>
    %v4537 = stablehlo.add %v4536, %v4534 : tensor<128x64x1x1xf32>
    %v4538 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4539 = stablehlo.multiply %v4538, %v4537 : tensor<128x64x1x1xf32>
    %v4540 = stablehlo.subtract %d2Wp, %v4539 : tensor<128x64x1x1xf32>
    %v4541 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4542 = stablehlo.multiply %v4541, %d2gp : tensor<128xf32>
    %v4543 = stablehlo.add %v4542, %v3141 : tensor<128xf32>
    %v4544 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4545 = stablehlo.multiply %v4544, %d2gpv : tensor<128xf32>
    %v4546 = stablehlo.add %v4545, %v4543 : tensor<128xf32>
    %v4547 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4548 = stablehlo.multiply %v4547, %v4546 : tensor<128xf32>
    %v4549 = stablehlo.subtract %d2gp, %v4548 : tensor<128xf32>
    %v4550 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4551 = stablehlo.multiply %v4550, %d2btp : tensor<128xf32>
    %v4552 = stablehlo.add %v4551, %v3144 : tensor<128xf32>
    %v4553 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4554 = stablehlo.multiply %v4553, %d2btpv : tensor<128xf32>
    %v4555 = stablehlo.add %v4554, %v4552 : tensor<128xf32>
    %v4556 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4557 = stablehlo.multiply %v4556, %v4555 : tensor<128xf32>
    %v4558 = stablehlo.subtract %d2btp, %v4557 : tensor<128xf32>
    %v4559 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4560 = stablehlo.multiply %v4559, %s2b0W1 : tensor<128x128x3x3xf32>
    %v4561 = stablehlo.add %v4560, %v2892 : tensor<128x128x3x3xf32>
    %v4562 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4563 = stablehlo.multiply %v4562, %s2b0W1v : tensor<128x128x3x3xf32>
    %v4564 = stablehlo.add %v4563, %v4561 : tensor<128x128x3x3xf32>
    %v4565 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4566 = stablehlo.multiply %v4565, %v4564 : tensor<128x128x3x3xf32>
    %v4567 = stablehlo.subtract %s2b0W1, %v4566 : tensor<128x128x3x3xf32>
    %v4568 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4569 = stablehlo.multiply %v4568, %s2b0g1 : tensor<128xf32>
    %v4570 = stablehlo.add %v4569, %v2910 : tensor<128xf32>
    %v4571 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4572 = stablehlo.multiply %v4571, %s2b0g1v : tensor<128xf32>
    %v4573 = stablehlo.add %v4572, %v4570 : tensor<128xf32>
    %v4574 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4575 = stablehlo.multiply %v4574, %v4573 : tensor<128xf32>
    %v4576 = stablehlo.subtract %s2b0g1, %v4575 : tensor<128xf32>
    %v4577 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4578 = stablehlo.multiply %v4577, %s2b0bt1 : tensor<128xf32>
    %v4579 = stablehlo.add %v4578, %v2913 : tensor<128xf32>
    %v4580 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4581 = stablehlo.multiply %v4580, %s2b0bt1v : tensor<128xf32>
    %v4582 = stablehlo.add %v4581, %v4579 : tensor<128xf32>
    %v4583 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4584 = stablehlo.multiply %v4583, %v4582 : tensor<128xf32>
    %v4585 = stablehlo.subtract %s2b0bt1, %v4584 : tensor<128xf32>
    %v4586 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4587 = stablehlo.multiply %v4586, %s2b0W2 : tensor<128x128x3x3xf32>
    %v4588 = stablehlo.add %v4587, %v2919 : tensor<128x128x3x3xf32>
    %v4589 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4590 = stablehlo.multiply %v4589, %s2b0W2v : tensor<128x128x3x3xf32>
    %v4591 = stablehlo.add %v4590, %v4588 : tensor<128x128x3x3xf32>
    %v4592 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4593 = stablehlo.multiply %v4592, %v4591 : tensor<128x128x3x3xf32>
    %v4594 = stablehlo.subtract %s2b0W2, %v4593 : tensor<128x128x3x3xf32>
    %v4595 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4596 = stablehlo.multiply %v4595, %s2b0g2 : tensor<128xf32>
    %v4597 = stablehlo.add %v4596, %v2937 : tensor<128xf32>
    %v4598 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4599 = stablehlo.multiply %v4598, %s2b0g2v : tensor<128xf32>
    %v4600 = stablehlo.add %v4599, %v4597 : tensor<128xf32>
    %v4601 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4602 = stablehlo.multiply %v4601, %v4600 : tensor<128xf32>
    %v4603 = stablehlo.subtract %s2b0g2, %v4602 : tensor<128xf32>
    %v4604 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4605 = stablehlo.multiply %v4604, %s2b0bt2 : tensor<128xf32>
    %v4606 = stablehlo.add %v4605, %v2940 : tensor<128xf32>
    %v4607 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4608 = stablehlo.multiply %v4607, %s2b0bt2v : tensor<128xf32>
    %v4609 = stablehlo.add %v4608, %v4606 : tensor<128xf32>
    %v4610 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4611 = stablehlo.multiply %v4610, %v4609 : tensor<128xf32>
    %v4612 = stablehlo.subtract %s2b0bt2, %v4611 : tensor<128xf32>
    %v4613 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4614 = stablehlo.multiply %v4613, %s2b1W1 : tensor<128x128x3x3xf32>
    %v4615 = stablehlo.add %v4614, %v2752 : tensor<128x128x3x3xf32>
    %v4616 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4617 = stablehlo.multiply %v4616, %s2b1W1v : tensor<128x128x3x3xf32>
    %v4618 = stablehlo.add %v4617, %v4615 : tensor<128x128x3x3xf32>
    %v4619 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4620 = stablehlo.multiply %v4619, %v4618 : tensor<128x128x3x3xf32>
    %v4621 = stablehlo.subtract %s2b1W1, %v4620 : tensor<128x128x3x3xf32>
    %v4622 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4623 = stablehlo.multiply %v4622, %s2b1g1 : tensor<128xf32>
    %v4624 = stablehlo.add %v4623, %v2770 : tensor<128xf32>
    %v4625 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4626 = stablehlo.multiply %v4625, %s2b1g1v : tensor<128xf32>
    %v4627 = stablehlo.add %v4626, %v4624 : tensor<128xf32>
    %v4628 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4629 = stablehlo.multiply %v4628, %v4627 : tensor<128xf32>
    %v4630 = stablehlo.subtract %s2b1g1, %v4629 : tensor<128xf32>
    %v4631 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4632 = stablehlo.multiply %v4631, %s2b1bt1 : tensor<128xf32>
    %v4633 = stablehlo.add %v4632, %v2773 : tensor<128xf32>
    %v4634 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4635 = stablehlo.multiply %v4634, %s2b1bt1v : tensor<128xf32>
    %v4636 = stablehlo.add %v4635, %v4633 : tensor<128xf32>
    %v4637 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4638 = stablehlo.multiply %v4637, %v4636 : tensor<128xf32>
    %v4639 = stablehlo.subtract %s2b1bt1, %v4638 : tensor<128xf32>
    %v4640 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4641 = stablehlo.multiply %v4640, %s2b1W2 : tensor<128x128x3x3xf32>
    %v4642 = stablehlo.add %v4641, %v2779 : tensor<128x128x3x3xf32>
    %v4643 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4644 = stablehlo.multiply %v4643, %s2b1W2v : tensor<128x128x3x3xf32>
    %v4645 = stablehlo.add %v4644, %v4642 : tensor<128x128x3x3xf32>
    %v4646 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4647 = stablehlo.multiply %v4646, %v4645 : tensor<128x128x3x3xf32>
    %v4648 = stablehlo.subtract %s2b1W2, %v4647 : tensor<128x128x3x3xf32>
    %v4649 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4650 = stablehlo.multiply %v4649, %s2b1g2 : tensor<128xf32>
    %v4651 = stablehlo.add %v4650, %v2797 : tensor<128xf32>
    %v4652 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4653 = stablehlo.multiply %v4652, %s2b1g2v : tensor<128xf32>
    %v4654 = stablehlo.add %v4653, %v4651 : tensor<128xf32>
    %v4655 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4656 = stablehlo.multiply %v4655, %v4654 : tensor<128xf32>
    %v4657 = stablehlo.subtract %s2b1g2, %v4656 : tensor<128xf32>
    %v4658 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4659 = stablehlo.multiply %v4658, %s2b1bt2 : tensor<128xf32>
    %v4660 = stablehlo.add %v4659, %v2800 : tensor<128xf32>
    %v4661 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4662 = stablehlo.multiply %v4661, %s2b1bt2v : tensor<128xf32>
    %v4663 = stablehlo.add %v4662, %v4660 : tensor<128xf32>
    %v4664 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4665 = stablehlo.multiply %v4664, %v4663 : tensor<128xf32>
    %v4666 = stablehlo.subtract %s2b1bt2, %v4665 : tensor<128xf32>
    %v4667 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4668 = stablehlo.multiply %v4667, %s2b2W1 : tensor<128x128x3x3xf32>
    %v4669 = stablehlo.add %v4668, %v2612 : tensor<128x128x3x3xf32>
    %v4670 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4671 = stablehlo.multiply %v4670, %s2b2W1v : tensor<128x128x3x3xf32>
    %v4672 = stablehlo.add %v4671, %v4669 : tensor<128x128x3x3xf32>
    %v4673 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4674 = stablehlo.multiply %v4673, %v4672 : tensor<128x128x3x3xf32>
    %v4675 = stablehlo.subtract %s2b2W1, %v4674 : tensor<128x128x3x3xf32>
    %v4676 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4677 = stablehlo.multiply %v4676, %s2b2g1 : tensor<128xf32>
    %v4678 = stablehlo.add %v4677, %v2630 : tensor<128xf32>
    %v4679 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4680 = stablehlo.multiply %v4679, %s2b2g1v : tensor<128xf32>
    %v4681 = stablehlo.add %v4680, %v4678 : tensor<128xf32>
    %v4682 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4683 = stablehlo.multiply %v4682, %v4681 : tensor<128xf32>
    %v4684 = stablehlo.subtract %s2b2g1, %v4683 : tensor<128xf32>
    %v4685 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4686 = stablehlo.multiply %v4685, %s2b2bt1 : tensor<128xf32>
    %v4687 = stablehlo.add %v4686, %v2633 : tensor<128xf32>
    %v4688 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4689 = stablehlo.multiply %v4688, %s2b2bt1v : tensor<128xf32>
    %v4690 = stablehlo.add %v4689, %v4687 : tensor<128xf32>
    %v4691 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4692 = stablehlo.multiply %v4691, %v4690 : tensor<128xf32>
    %v4693 = stablehlo.subtract %s2b2bt1, %v4692 : tensor<128xf32>
    %v4694 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4695 = stablehlo.multiply %v4694, %s2b2W2 : tensor<128x128x3x3xf32>
    %v4696 = stablehlo.add %v4695, %v2639 : tensor<128x128x3x3xf32>
    %v4697 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4698 = stablehlo.multiply %v4697, %s2b2W2v : tensor<128x128x3x3xf32>
    %v4699 = stablehlo.add %v4698, %v4696 : tensor<128x128x3x3xf32>
    %v4700 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4701 = stablehlo.multiply %v4700, %v4699 : tensor<128x128x3x3xf32>
    %v4702 = stablehlo.subtract %s2b2W2, %v4701 : tensor<128x128x3x3xf32>
    %v4703 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4704 = stablehlo.multiply %v4703, %s2b2g2 : tensor<128xf32>
    %v4705 = stablehlo.add %v4704, %v2657 : tensor<128xf32>
    %v4706 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4707 = stablehlo.multiply %v4706, %s2b2g2v : tensor<128xf32>
    %v4708 = stablehlo.add %v4707, %v4705 : tensor<128xf32>
    %v4709 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4710 = stablehlo.multiply %v4709, %v4708 : tensor<128xf32>
    %v4711 = stablehlo.subtract %s2b2g2, %v4710 : tensor<128xf32>
    %v4712 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4713 = stablehlo.multiply %v4712, %s2b2bt2 : tensor<128xf32>
    %v4714 = stablehlo.add %v4713, %v2660 : tensor<128xf32>
    %v4715 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4716 = stablehlo.multiply %v4715, %s2b2bt2v : tensor<128xf32>
    %v4717 = stablehlo.add %v4716, %v4714 : tensor<128xf32>
    %v4718 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4719 = stablehlo.multiply %v4718, %v4717 : tensor<128xf32>
    %v4720 = stablehlo.subtract %s2b2bt2, %v4719 : tensor<128xf32>
    %v4721 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4722 = stablehlo.multiply %v4721, %d3W1 : tensor<256x128x3x3xf32>
    %v4723 = stablehlo.add %v4722, %v2444 : tensor<256x128x3x3xf32>
    %v4724 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4725 = stablehlo.multiply %v4724, %d3W1v : tensor<256x128x3x3xf32>
    %v4726 = stablehlo.add %v4725, %v4723 : tensor<256x128x3x3xf32>
    %v4727 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4728 = stablehlo.multiply %v4727, %v4726 : tensor<256x128x3x3xf32>
    %v4729 = stablehlo.subtract %d3W1, %v4728 : tensor<256x128x3x3xf32>
    %v4730 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4731 = stablehlo.multiply %v4730, %d3g1 : tensor<256xf32>
    %v4732 = stablehlo.add %v4731, %v2462 : tensor<256xf32>
    %v4733 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4734 = stablehlo.multiply %v4733, %d3g1v : tensor<256xf32>
    %v4735 = stablehlo.add %v4734, %v4732 : tensor<256xf32>
    %v4736 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4737 = stablehlo.multiply %v4736, %v4735 : tensor<256xf32>
    %v4738 = stablehlo.subtract %d3g1, %v4737 : tensor<256xf32>
    %v4739 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4740 = stablehlo.multiply %v4739, %d3bt1 : tensor<256xf32>
    %v4741 = stablehlo.add %v4740, %v2465 : tensor<256xf32>
    %v4742 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4743 = stablehlo.multiply %v4742, %d3bt1v : tensor<256xf32>
    %v4744 = stablehlo.add %v4743, %v4741 : tensor<256xf32>
    %v4745 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4746 = stablehlo.multiply %v4745, %v4744 : tensor<256xf32>
    %v4747 = stablehlo.subtract %d3bt1, %v4746 : tensor<256xf32>
    %v4748 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4749 = stablehlo.multiply %v4748, %d3W2 : tensor<256x256x3x3xf32>
    %v4750 = stablehlo.add %v4749, %v2471 : tensor<256x256x3x3xf32>
    %v4751 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4752 = stablehlo.multiply %v4751, %d3W2v : tensor<256x256x3x3xf32>
    %v4753 = stablehlo.add %v4752, %v4750 : tensor<256x256x3x3xf32>
    %v4754 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4755 = stablehlo.multiply %v4754, %v4753 : tensor<256x256x3x3xf32>
    %v4756 = stablehlo.subtract %d3W2, %v4755 : tensor<256x256x3x3xf32>
    %v4757 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4758 = stablehlo.multiply %v4757, %d3g2 : tensor<256xf32>
    %v4759 = stablehlo.add %v4758, %v2489 : tensor<256xf32>
    %v4760 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4761 = stablehlo.multiply %v4760, %d3g2v : tensor<256xf32>
    %v4762 = stablehlo.add %v4761, %v4759 : tensor<256xf32>
    %v4763 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4764 = stablehlo.multiply %v4763, %v4762 : tensor<256xf32>
    %v4765 = stablehlo.subtract %d3g2, %v4764 : tensor<256xf32>
    %v4766 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4767 = stablehlo.multiply %v4766, %d3bt2 : tensor<256xf32>
    %v4768 = stablehlo.add %v4767, %v2492 : tensor<256xf32>
    %v4769 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4770 = stablehlo.multiply %v4769, %d3bt2v : tensor<256xf32>
    %v4771 = stablehlo.add %v4770, %v4768 : tensor<256xf32>
    %v4772 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4773 = stablehlo.multiply %v4772, %v4771 : tensor<256xf32>
    %v4774 = stablehlo.subtract %d3bt2, %v4773 : tensor<256xf32>
    %v4775 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v4776 = stablehlo.multiply %v4775, %d3Wp : tensor<256x128x1x1xf32>
    %v4777 = stablehlo.add %v4776, %v2499 : tensor<256x128x1x1xf32>
    %v4778 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v4779 = stablehlo.multiply %v4778, %d3Wpv : tensor<256x128x1x1xf32>
    %v4780 = stablehlo.add %v4779, %v4777 : tensor<256x128x1x1xf32>
    %v4781 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v4782 = stablehlo.multiply %v4781, %v4780 : tensor<256x128x1x1xf32>
    %v4783 = stablehlo.subtract %d3Wp, %v4782 : tensor<256x128x1x1xf32>
    %v4784 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4785 = stablehlo.multiply %v4784, %d3gp : tensor<256xf32>
    %v4786 = stablehlo.add %v4785, %v2517 : tensor<256xf32>
    %v4787 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4788 = stablehlo.multiply %v4787, %d3gpv : tensor<256xf32>
    %v4789 = stablehlo.add %v4788, %v4786 : tensor<256xf32>
    %v4790 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4791 = stablehlo.multiply %v4790, %v4789 : tensor<256xf32>
    %v4792 = stablehlo.subtract %d3gp, %v4791 : tensor<256xf32>
    %v4793 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4794 = stablehlo.multiply %v4793, %d3btp : tensor<256xf32>
    %v4795 = stablehlo.add %v4794, %v2520 : tensor<256xf32>
    %v4796 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4797 = stablehlo.multiply %v4796, %d3btpv : tensor<256xf32>
    %v4798 = stablehlo.add %v4797, %v4795 : tensor<256xf32>
    %v4799 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4800 = stablehlo.multiply %v4799, %v4798 : tensor<256xf32>
    %v4801 = stablehlo.subtract %d3btp, %v4800 : tensor<256xf32>
    %v4802 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4803 = stablehlo.multiply %v4802, %s3b0W1 : tensor<256x256x3x3xf32>
    %v4804 = stablehlo.add %v4803, %v2268 : tensor<256x256x3x3xf32>
    %v4805 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4806 = stablehlo.multiply %v4805, %s3b0W1v : tensor<256x256x3x3xf32>
    %v4807 = stablehlo.add %v4806, %v4804 : tensor<256x256x3x3xf32>
    %v4808 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4809 = stablehlo.multiply %v4808, %v4807 : tensor<256x256x3x3xf32>
    %v4810 = stablehlo.subtract %s3b0W1, %v4809 : tensor<256x256x3x3xf32>
    %v4811 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4812 = stablehlo.multiply %v4811, %s3b0g1 : tensor<256xf32>
    %v4813 = stablehlo.add %v4812, %v2286 : tensor<256xf32>
    %v4814 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4815 = stablehlo.multiply %v4814, %s3b0g1v : tensor<256xf32>
    %v4816 = stablehlo.add %v4815, %v4813 : tensor<256xf32>
    %v4817 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4818 = stablehlo.multiply %v4817, %v4816 : tensor<256xf32>
    %v4819 = stablehlo.subtract %s3b0g1, %v4818 : tensor<256xf32>
    %v4820 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4821 = stablehlo.multiply %v4820, %s3b0bt1 : tensor<256xf32>
    %v4822 = stablehlo.add %v4821, %v2289 : tensor<256xf32>
    %v4823 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4824 = stablehlo.multiply %v4823, %s3b0bt1v : tensor<256xf32>
    %v4825 = stablehlo.add %v4824, %v4822 : tensor<256xf32>
    %v4826 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4827 = stablehlo.multiply %v4826, %v4825 : tensor<256xf32>
    %v4828 = stablehlo.subtract %s3b0bt1, %v4827 : tensor<256xf32>
    %v4829 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4830 = stablehlo.multiply %v4829, %s3b0W2 : tensor<256x256x3x3xf32>
    %v4831 = stablehlo.add %v4830, %v2295 : tensor<256x256x3x3xf32>
    %v4832 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4833 = stablehlo.multiply %v4832, %s3b0W2v : tensor<256x256x3x3xf32>
    %v4834 = stablehlo.add %v4833, %v4831 : tensor<256x256x3x3xf32>
    %v4835 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4836 = stablehlo.multiply %v4835, %v4834 : tensor<256x256x3x3xf32>
    %v4837 = stablehlo.subtract %s3b0W2, %v4836 : tensor<256x256x3x3xf32>
    %v4838 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4839 = stablehlo.multiply %v4838, %s3b0g2 : tensor<256xf32>
    %v4840 = stablehlo.add %v4839, %v2313 : tensor<256xf32>
    %v4841 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4842 = stablehlo.multiply %v4841, %s3b0g2v : tensor<256xf32>
    %v4843 = stablehlo.add %v4842, %v4840 : tensor<256xf32>
    %v4844 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4845 = stablehlo.multiply %v4844, %v4843 : tensor<256xf32>
    %v4846 = stablehlo.subtract %s3b0g2, %v4845 : tensor<256xf32>
    %v4847 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4848 = stablehlo.multiply %v4847, %s3b0bt2 : tensor<256xf32>
    %v4849 = stablehlo.add %v4848, %v2316 : tensor<256xf32>
    %v4850 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4851 = stablehlo.multiply %v4850, %s3b0bt2v : tensor<256xf32>
    %v4852 = stablehlo.add %v4851, %v4849 : tensor<256xf32>
    %v4853 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4854 = stablehlo.multiply %v4853, %v4852 : tensor<256xf32>
    %v4855 = stablehlo.subtract %s3b0bt2, %v4854 : tensor<256xf32>
    %v4856 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4857 = stablehlo.multiply %v4856, %s3b1W1 : tensor<256x256x3x3xf32>
    %v4858 = stablehlo.add %v4857, %v2128 : tensor<256x256x3x3xf32>
    %v4859 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4860 = stablehlo.multiply %v4859, %s3b1W1v : tensor<256x256x3x3xf32>
    %v4861 = stablehlo.add %v4860, %v4858 : tensor<256x256x3x3xf32>
    %v4862 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4863 = stablehlo.multiply %v4862, %v4861 : tensor<256x256x3x3xf32>
    %v4864 = stablehlo.subtract %s3b1W1, %v4863 : tensor<256x256x3x3xf32>
    %v4865 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4866 = stablehlo.multiply %v4865, %s3b1g1 : tensor<256xf32>
    %v4867 = stablehlo.add %v4866, %v2146 : tensor<256xf32>
    %v4868 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4869 = stablehlo.multiply %v4868, %s3b1g1v : tensor<256xf32>
    %v4870 = stablehlo.add %v4869, %v4867 : tensor<256xf32>
    %v4871 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4872 = stablehlo.multiply %v4871, %v4870 : tensor<256xf32>
    %v4873 = stablehlo.subtract %s3b1g1, %v4872 : tensor<256xf32>
    %v4874 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4875 = stablehlo.multiply %v4874, %s3b1bt1 : tensor<256xf32>
    %v4876 = stablehlo.add %v4875, %v2149 : tensor<256xf32>
    %v4877 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4878 = stablehlo.multiply %v4877, %s3b1bt1v : tensor<256xf32>
    %v4879 = stablehlo.add %v4878, %v4876 : tensor<256xf32>
    %v4880 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4881 = stablehlo.multiply %v4880, %v4879 : tensor<256xf32>
    %v4882 = stablehlo.subtract %s3b1bt1, %v4881 : tensor<256xf32>
    %v4883 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4884 = stablehlo.multiply %v4883, %s3b1W2 : tensor<256x256x3x3xf32>
    %v4885 = stablehlo.add %v4884, %v2155 : tensor<256x256x3x3xf32>
    %v4886 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4887 = stablehlo.multiply %v4886, %s3b1W2v : tensor<256x256x3x3xf32>
    %v4888 = stablehlo.add %v4887, %v4885 : tensor<256x256x3x3xf32>
    %v4889 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4890 = stablehlo.multiply %v4889, %v4888 : tensor<256x256x3x3xf32>
    %v4891 = stablehlo.subtract %s3b1W2, %v4890 : tensor<256x256x3x3xf32>
    %v4892 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4893 = stablehlo.multiply %v4892, %s3b1g2 : tensor<256xf32>
    %v4894 = stablehlo.add %v4893, %v2173 : tensor<256xf32>
    %v4895 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4896 = stablehlo.multiply %v4895, %s3b1g2v : tensor<256xf32>
    %v4897 = stablehlo.add %v4896, %v4894 : tensor<256xf32>
    %v4898 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4899 = stablehlo.multiply %v4898, %v4897 : tensor<256xf32>
    %v4900 = stablehlo.subtract %s3b1g2, %v4899 : tensor<256xf32>
    %v4901 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4902 = stablehlo.multiply %v4901, %s3b1bt2 : tensor<256xf32>
    %v4903 = stablehlo.add %v4902, %v2176 : tensor<256xf32>
    %v4904 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4905 = stablehlo.multiply %v4904, %s3b1bt2v : tensor<256xf32>
    %v4906 = stablehlo.add %v4905, %v4903 : tensor<256xf32>
    %v4907 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4908 = stablehlo.multiply %v4907, %v4906 : tensor<256xf32>
    %v4909 = stablehlo.subtract %s3b1bt2, %v4908 : tensor<256xf32>
    %v4910 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4911 = stablehlo.multiply %v4910, %s3b2W1 : tensor<256x256x3x3xf32>
    %v4912 = stablehlo.add %v4911, %v1988 : tensor<256x256x3x3xf32>
    %v4913 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4914 = stablehlo.multiply %v4913, %s3b2W1v : tensor<256x256x3x3xf32>
    %v4915 = stablehlo.add %v4914, %v4912 : tensor<256x256x3x3xf32>
    %v4916 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4917 = stablehlo.multiply %v4916, %v4915 : tensor<256x256x3x3xf32>
    %v4918 = stablehlo.subtract %s3b2W1, %v4917 : tensor<256x256x3x3xf32>
    %v4919 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4920 = stablehlo.multiply %v4919, %s3b2g1 : tensor<256xf32>
    %v4921 = stablehlo.add %v4920, %v2006 : tensor<256xf32>
    %v4922 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4923 = stablehlo.multiply %v4922, %s3b2g1v : tensor<256xf32>
    %v4924 = stablehlo.add %v4923, %v4921 : tensor<256xf32>
    %v4925 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4926 = stablehlo.multiply %v4925, %v4924 : tensor<256xf32>
    %v4927 = stablehlo.subtract %s3b2g1, %v4926 : tensor<256xf32>
    %v4928 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4929 = stablehlo.multiply %v4928, %s3b2bt1 : tensor<256xf32>
    %v4930 = stablehlo.add %v4929, %v2009 : tensor<256xf32>
    %v4931 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4932 = stablehlo.multiply %v4931, %s3b2bt1v : tensor<256xf32>
    %v4933 = stablehlo.add %v4932, %v4930 : tensor<256xf32>
    %v4934 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4935 = stablehlo.multiply %v4934, %v4933 : tensor<256xf32>
    %v4936 = stablehlo.subtract %s3b2bt1, %v4935 : tensor<256xf32>
    %v4937 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4938 = stablehlo.multiply %v4937, %s3b2W2 : tensor<256x256x3x3xf32>
    %v4939 = stablehlo.add %v4938, %v2015 : tensor<256x256x3x3xf32>
    %v4940 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4941 = stablehlo.multiply %v4940, %s3b2W2v : tensor<256x256x3x3xf32>
    %v4942 = stablehlo.add %v4941, %v4939 : tensor<256x256x3x3xf32>
    %v4943 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4944 = stablehlo.multiply %v4943, %v4942 : tensor<256x256x3x3xf32>
    %v4945 = stablehlo.subtract %s3b2W2, %v4944 : tensor<256x256x3x3xf32>
    %v4946 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4947 = stablehlo.multiply %v4946, %s3b2g2 : tensor<256xf32>
    %v4948 = stablehlo.add %v4947, %v2033 : tensor<256xf32>
    %v4949 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4950 = stablehlo.multiply %v4949, %s3b2g2v : tensor<256xf32>
    %v4951 = stablehlo.add %v4950, %v4948 : tensor<256xf32>
    %v4952 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4953 = stablehlo.multiply %v4952, %v4951 : tensor<256xf32>
    %v4954 = stablehlo.subtract %s3b2g2, %v4953 : tensor<256xf32>
    %v4955 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4956 = stablehlo.multiply %v4955, %s3b2bt2 : tensor<256xf32>
    %v4957 = stablehlo.add %v4956, %v2036 : tensor<256xf32>
    %v4958 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4959 = stablehlo.multiply %v4958, %s3b2bt2v : tensor<256xf32>
    %v4960 = stablehlo.add %v4959, %v4957 : tensor<256xf32>
    %v4961 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4962 = stablehlo.multiply %v4961, %v4960 : tensor<256xf32>
    %v4963 = stablehlo.subtract %s3b2bt2, %v4962 : tensor<256xf32>
    %v4964 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4965 = stablehlo.multiply %v4964, %s3b3W1 : tensor<256x256x3x3xf32>
    %v4966 = stablehlo.add %v4965, %v1848 : tensor<256x256x3x3xf32>
    %v4967 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4968 = stablehlo.multiply %v4967, %s3b3W1v : tensor<256x256x3x3xf32>
    %v4969 = stablehlo.add %v4968, %v4966 : tensor<256x256x3x3xf32>
    %v4970 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4971 = stablehlo.multiply %v4970, %v4969 : tensor<256x256x3x3xf32>
    %v4972 = stablehlo.subtract %s3b3W1, %v4971 : tensor<256x256x3x3xf32>
    %v4973 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4974 = stablehlo.multiply %v4973, %s3b3g1 : tensor<256xf32>
    %v4975 = stablehlo.add %v4974, %v1866 : tensor<256xf32>
    %v4976 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4977 = stablehlo.multiply %v4976, %s3b3g1v : tensor<256xf32>
    %v4978 = stablehlo.add %v4977, %v4975 : tensor<256xf32>
    %v4979 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4980 = stablehlo.multiply %v4979, %v4978 : tensor<256xf32>
    %v4981 = stablehlo.subtract %s3b3g1, %v4980 : tensor<256xf32>
    %v4982 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4983 = stablehlo.multiply %v4982, %s3b3bt1 : tensor<256xf32>
    %v4984 = stablehlo.add %v4983, %v1869 : tensor<256xf32>
    %v4985 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4986 = stablehlo.multiply %v4985, %s3b3bt1v : tensor<256xf32>
    %v4987 = stablehlo.add %v4986, %v4984 : tensor<256xf32>
    %v4988 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4989 = stablehlo.multiply %v4988, %v4987 : tensor<256xf32>
    %v4990 = stablehlo.subtract %s3b3bt1, %v4989 : tensor<256xf32>
    %v4991 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4992 = stablehlo.multiply %v4991, %s3b3W2 : tensor<256x256x3x3xf32>
    %v4993 = stablehlo.add %v4992, %v1875 : tensor<256x256x3x3xf32>
    %v4994 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4995 = stablehlo.multiply %v4994, %s3b3W2v : tensor<256x256x3x3xf32>
    %v4996 = stablehlo.add %v4995, %v4993 : tensor<256x256x3x3xf32>
    %v4997 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4998 = stablehlo.multiply %v4997, %v4996 : tensor<256x256x3x3xf32>
    %v4999 = stablehlo.subtract %s3b3W2, %v4998 : tensor<256x256x3x3xf32>
    %v5000 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5001 = stablehlo.multiply %v5000, %s3b3g2 : tensor<256xf32>
    %v5002 = stablehlo.add %v5001, %v1893 : tensor<256xf32>
    %v5003 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5004 = stablehlo.multiply %v5003, %s3b3g2v : tensor<256xf32>
    %v5005 = stablehlo.add %v5004, %v5002 : tensor<256xf32>
    %v5006 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5007 = stablehlo.multiply %v5006, %v5005 : tensor<256xf32>
    %v5008 = stablehlo.subtract %s3b3g2, %v5007 : tensor<256xf32>
    %v5009 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5010 = stablehlo.multiply %v5009, %s3b3bt2 : tensor<256xf32>
    %v5011 = stablehlo.add %v5010, %v1896 : tensor<256xf32>
    %v5012 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5013 = stablehlo.multiply %v5012, %s3b3bt2v : tensor<256xf32>
    %v5014 = stablehlo.add %v5013, %v5011 : tensor<256xf32>
    %v5015 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5016 = stablehlo.multiply %v5015, %v5014 : tensor<256xf32>
    %v5017 = stablehlo.subtract %s3b3bt2, %v5016 : tensor<256xf32>
    %v5018 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5019 = stablehlo.multiply %v5018, %s3b4W1 : tensor<256x256x3x3xf32>
    %v5020 = stablehlo.add %v5019, %v1708 : tensor<256x256x3x3xf32>
    %v5021 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5022 = stablehlo.multiply %v5021, %s3b4W1v : tensor<256x256x3x3xf32>
    %v5023 = stablehlo.add %v5022, %v5020 : tensor<256x256x3x3xf32>
    %v5024 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5025 = stablehlo.multiply %v5024, %v5023 : tensor<256x256x3x3xf32>
    %v5026 = stablehlo.subtract %s3b4W1, %v5025 : tensor<256x256x3x3xf32>
    %v5027 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5028 = stablehlo.multiply %v5027, %s3b4g1 : tensor<256xf32>
    %v5029 = stablehlo.add %v5028, %v1726 : tensor<256xf32>
    %v5030 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5031 = stablehlo.multiply %v5030, %s3b4g1v : tensor<256xf32>
    %v5032 = stablehlo.add %v5031, %v5029 : tensor<256xf32>
    %v5033 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5034 = stablehlo.multiply %v5033, %v5032 : tensor<256xf32>
    %v5035 = stablehlo.subtract %s3b4g1, %v5034 : tensor<256xf32>
    %v5036 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5037 = stablehlo.multiply %v5036, %s3b4bt1 : tensor<256xf32>
    %v5038 = stablehlo.add %v5037, %v1729 : tensor<256xf32>
    %v5039 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5040 = stablehlo.multiply %v5039, %s3b4bt1v : tensor<256xf32>
    %v5041 = stablehlo.add %v5040, %v5038 : tensor<256xf32>
    %v5042 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5043 = stablehlo.multiply %v5042, %v5041 : tensor<256xf32>
    %v5044 = stablehlo.subtract %s3b4bt1, %v5043 : tensor<256xf32>
    %v5045 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5046 = stablehlo.multiply %v5045, %s3b4W2 : tensor<256x256x3x3xf32>
    %v5047 = stablehlo.add %v5046, %v1735 : tensor<256x256x3x3xf32>
    %v5048 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5049 = stablehlo.multiply %v5048, %s3b4W2v : tensor<256x256x3x3xf32>
    %v5050 = stablehlo.add %v5049, %v5047 : tensor<256x256x3x3xf32>
    %v5051 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5052 = stablehlo.multiply %v5051, %v5050 : tensor<256x256x3x3xf32>
    %v5053 = stablehlo.subtract %s3b4W2, %v5052 : tensor<256x256x3x3xf32>
    %v5054 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5055 = stablehlo.multiply %v5054, %s3b4g2 : tensor<256xf32>
    %v5056 = stablehlo.add %v5055, %v1753 : tensor<256xf32>
    %v5057 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5058 = stablehlo.multiply %v5057, %s3b4g2v : tensor<256xf32>
    %v5059 = stablehlo.add %v5058, %v5056 : tensor<256xf32>
    %v5060 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5061 = stablehlo.multiply %v5060, %v5059 : tensor<256xf32>
    %v5062 = stablehlo.subtract %s3b4g2, %v5061 : tensor<256xf32>
    %v5063 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5064 = stablehlo.multiply %v5063, %s3b4bt2 : tensor<256xf32>
    %v5065 = stablehlo.add %v5064, %v1756 : tensor<256xf32>
    %v5066 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5067 = stablehlo.multiply %v5066, %s3b4bt2v : tensor<256xf32>
    %v5068 = stablehlo.add %v5067, %v5065 : tensor<256xf32>
    %v5069 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5070 = stablehlo.multiply %v5069, %v5068 : tensor<256xf32>
    %v5071 = stablehlo.subtract %s3b4bt2, %v5070 : tensor<256xf32>
    %v5072 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5073 = stablehlo.multiply %v5072, %d4W1 : tensor<512x256x3x3xf32>
    %v5074 = stablehlo.add %v5073, %v1540 : tensor<512x256x3x3xf32>
    %v5075 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5076 = stablehlo.multiply %v5075, %d4W1v : tensor<512x256x3x3xf32>
    %v5077 = stablehlo.add %v5076, %v5074 : tensor<512x256x3x3xf32>
    %v5078 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5079 = stablehlo.multiply %v5078, %v5077 : tensor<512x256x3x3xf32>
    %v5080 = stablehlo.subtract %d4W1, %v5079 : tensor<512x256x3x3xf32>
    %v5081 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5082 = stablehlo.multiply %v5081, %d4g1 : tensor<512xf32>
    %v5083 = stablehlo.add %v5082, %v1558 : tensor<512xf32>
    %v5084 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5085 = stablehlo.multiply %v5084, %d4g1v : tensor<512xf32>
    %v5086 = stablehlo.add %v5085, %v5083 : tensor<512xf32>
    %v5087 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5088 = stablehlo.multiply %v5087, %v5086 : tensor<512xf32>
    %v5089 = stablehlo.subtract %d4g1, %v5088 : tensor<512xf32>
    %v5090 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5091 = stablehlo.multiply %v5090, %d4bt1 : tensor<512xf32>
    %v5092 = stablehlo.add %v5091, %v1561 : tensor<512xf32>
    %v5093 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5094 = stablehlo.multiply %v5093, %d4bt1v : tensor<512xf32>
    %v5095 = stablehlo.add %v5094, %v5092 : tensor<512xf32>
    %v5096 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5097 = stablehlo.multiply %v5096, %v5095 : tensor<512xf32>
    %v5098 = stablehlo.subtract %d4bt1, %v5097 : tensor<512xf32>
    %v5099 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5100 = stablehlo.multiply %v5099, %d4W2 : tensor<512x512x3x3xf32>
    %v5101 = stablehlo.add %v5100, %v1567 : tensor<512x512x3x3xf32>
    %v5102 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5103 = stablehlo.multiply %v5102, %d4W2v : tensor<512x512x3x3xf32>
    %v5104 = stablehlo.add %v5103, %v5101 : tensor<512x512x3x3xf32>
    %v5105 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5106 = stablehlo.multiply %v5105, %v5104 : tensor<512x512x3x3xf32>
    %v5107 = stablehlo.subtract %d4W2, %v5106 : tensor<512x512x3x3xf32>
    %v5108 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5109 = stablehlo.multiply %v5108, %d4g2 : tensor<512xf32>
    %v5110 = stablehlo.add %v5109, %v1585 : tensor<512xf32>
    %v5111 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5112 = stablehlo.multiply %v5111, %d4g2v : tensor<512xf32>
    %v5113 = stablehlo.add %v5112, %v5110 : tensor<512xf32>
    %v5114 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5115 = stablehlo.multiply %v5114, %v5113 : tensor<512xf32>
    %v5116 = stablehlo.subtract %d4g2, %v5115 : tensor<512xf32>
    %v5117 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5118 = stablehlo.multiply %v5117, %d4bt2 : tensor<512xf32>
    %v5119 = stablehlo.add %v5118, %v1588 : tensor<512xf32>
    %v5120 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5121 = stablehlo.multiply %v5120, %d4bt2v : tensor<512xf32>
    %v5122 = stablehlo.add %v5121, %v5119 : tensor<512xf32>
    %v5123 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5124 = stablehlo.multiply %v5123, %v5122 : tensor<512xf32>
    %v5125 = stablehlo.subtract %d4bt2, %v5124 : tensor<512xf32>
    %v5126 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5127 = stablehlo.multiply %v5126, %d4Wp : tensor<512x256x1x1xf32>
    %v5128 = stablehlo.add %v5127, %v1595 : tensor<512x256x1x1xf32>
    %v5129 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5130 = stablehlo.multiply %v5129, %d4Wpv : tensor<512x256x1x1xf32>
    %v5131 = stablehlo.add %v5130, %v5128 : tensor<512x256x1x1xf32>
    %v5132 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5133 = stablehlo.multiply %v5132, %v5131 : tensor<512x256x1x1xf32>
    %v5134 = stablehlo.subtract %d4Wp, %v5133 : tensor<512x256x1x1xf32>
    %v5135 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5136 = stablehlo.multiply %v5135, %d4gp : tensor<512xf32>
    %v5137 = stablehlo.add %v5136, %v1613 : tensor<512xf32>
    %v5138 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5139 = stablehlo.multiply %v5138, %d4gpv : tensor<512xf32>
    %v5140 = stablehlo.add %v5139, %v5137 : tensor<512xf32>
    %v5141 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5142 = stablehlo.multiply %v5141, %v5140 : tensor<512xf32>
    %v5143 = stablehlo.subtract %d4gp, %v5142 : tensor<512xf32>
    %v5144 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5145 = stablehlo.multiply %v5144, %d4btp : tensor<512xf32>
    %v5146 = stablehlo.add %v5145, %v1616 : tensor<512xf32>
    %v5147 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5148 = stablehlo.multiply %v5147, %d4btpv : tensor<512xf32>
    %v5149 = stablehlo.add %v5148, %v5146 : tensor<512xf32>
    %v5150 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5151 = stablehlo.multiply %v5150, %v5149 : tensor<512xf32>
    %v5152 = stablehlo.subtract %d4btp, %v5151 : tensor<512xf32>
    %v5153 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5154 = stablehlo.multiply %v5153, %s4b0W1 : tensor<512x512x3x3xf32>
    %v5155 = stablehlo.add %v5154, %v1364 : tensor<512x512x3x3xf32>
    %v5156 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5157 = stablehlo.multiply %v5156, %s4b0W1v : tensor<512x512x3x3xf32>
    %v5158 = stablehlo.add %v5157, %v5155 : tensor<512x512x3x3xf32>
    %v5159 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5160 = stablehlo.multiply %v5159, %v5158 : tensor<512x512x3x3xf32>
    %v5161 = stablehlo.subtract %s4b0W1, %v5160 : tensor<512x512x3x3xf32>
    %v5162 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5163 = stablehlo.multiply %v5162, %s4b0g1 : tensor<512xf32>
    %v5164 = stablehlo.add %v5163, %v1382 : tensor<512xf32>
    %v5165 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5166 = stablehlo.multiply %v5165, %s4b0g1v : tensor<512xf32>
    %v5167 = stablehlo.add %v5166, %v5164 : tensor<512xf32>
    %v5168 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5169 = stablehlo.multiply %v5168, %v5167 : tensor<512xf32>
    %v5170 = stablehlo.subtract %s4b0g1, %v5169 : tensor<512xf32>
    %v5171 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5172 = stablehlo.multiply %v5171, %s4b0bt1 : tensor<512xf32>
    %v5173 = stablehlo.add %v5172, %v1385 : tensor<512xf32>
    %v5174 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5175 = stablehlo.multiply %v5174, %s4b0bt1v : tensor<512xf32>
    %v5176 = stablehlo.add %v5175, %v5173 : tensor<512xf32>
    %v5177 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5178 = stablehlo.multiply %v5177, %v5176 : tensor<512xf32>
    %v5179 = stablehlo.subtract %s4b0bt1, %v5178 : tensor<512xf32>
    %v5180 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5181 = stablehlo.multiply %v5180, %s4b0W2 : tensor<512x512x3x3xf32>
    %v5182 = stablehlo.add %v5181, %v1391 : tensor<512x512x3x3xf32>
    %v5183 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5184 = stablehlo.multiply %v5183, %s4b0W2v : tensor<512x512x3x3xf32>
    %v5185 = stablehlo.add %v5184, %v5182 : tensor<512x512x3x3xf32>
    %v5186 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5187 = stablehlo.multiply %v5186, %v5185 : tensor<512x512x3x3xf32>
    %v5188 = stablehlo.subtract %s4b0W2, %v5187 : tensor<512x512x3x3xf32>
    %v5189 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5190 = stablehlo.multiply %v5189, %s4b0g2 : tensor<512xf32>
    %v5191 = stablehlo.add %v5190, %v1409 : tensor<512xf32>
    %v5192 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5193 = stablehlo.multiply %v5192, %s4b0g2v : tensor<512xf32>
    %v5194 = stablehlo.add %v5193, %v5191 : tensor<512xf32>
    %v5195 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5196 = stablehlo.multiply %v5195, %v5194 : tensor<512xf32>
    %v5197 = stablehlo.subtract %s4b0g2, %v5196 : tensor<512xf32>
    %v5198 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5199 = stablehlo.multiply %v5198, %s4b0bt2 : tensor<512xf32>
    %v5200 = stablehlo.add %v5199, %v1412 : tensor<512xf32>
    %v5201 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5202 = stablehlo.multiply %v5201, %s4b0bt2v : tensor<512xf32>
    %v5203 = stablehlo.add %v5202, %v5200 : tensor<512xf32>
    %v5204 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5205 = stablehlo.multiply %v5204, %v5203 : tensor<512xf32>
    %v5206 = stablehlo.subtract %s4b0bt2, %v5205 : tensor<512xf32>
    %v5207 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5208 = stablehlo.multiply %v5207, %s4b1W1 : tensor<512x512x3x3xf32>
    %v5209 = stablehlo.add %v5208, %v1224 : tensor<512x512x3x3xf32>
    %v5210 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5211 = stablehlo.multiply %v5210, %s4b1W1v : tensor<512x512x3x3xf32>
    %v5212 = stablehlo.add %v5211, %v5209 : tensor<512x512x3x3xf32>
    %v5213 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5214 = stablehlo.multiply %v5213, %v5212 : tensor<512x512x3x3xf32>
    %v5215 = stablehlo.subtract %s4b1W1, %v5214 : tensor<512x512x3x3xf32>
    %v5216 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5217 = stablehlo.multiply %v5216, %s4b1g1 : tensor<512xf32>
    %v5218 = stablehlo.add %v5217, %v1242 : tensor<512xf32>
    %v5219 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5220 = stablehlo.multiply %v5219, %s4b1g1v : tensor<512xf32>
    %v5221 = stablehlo.add %v5220, %v5218 : tensor<512xf32>
    %v5222 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5223 = stablehlo.multiply %v5222, %v5221 : tensor<512xf32>
    %v5224 = stablehlo.subtract %s4b1g1, %v5223 : tensor<512xf32>
    %v5225 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5226 = stablehlo.multiply %v5225, %s4b1bt1 : tensor<512xf32>
    %v5227 = stablehlo.add %v5226, %v1245 : tensor<512xf32>
    %v5228 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5229 = stablehlo.multiply %v5228, %s4b1bt1v : tensor<512xf32>
    %v5230 = stablehlo.add %v5229, %v5227 : tensor<512xf32>
    %v5231 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5232 = stablehlo.multiply %v5231, %v5230 : tensor<512xf32>
    %v5233 = stablehlo.subtract %s4b1bt1, %v5232 : tensor<512xf32>
    %v5234 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5235 = stablehlo.multiply %v5234, %s4b1W2 : tensor<512x512x3x3xf32>
    %v5236 = stablehlo.add %v5235, %v1251 : tensor<512x512x3x3xf32>
    %v5237 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5238 = stablehlo.multiply %v5237, %s4b1W2v : tensor<512x512x3x3xf32>
    %v5239 = stablehlo.add %v5238, %v5236 : tensor<512x512x3x3xf32>
    %v5240 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5241 = stablehlo.multiply %v5240, %v5239 : tensor<512x512x3x3xf32>
    %v5242 = stablehlo.subtract %s4b1W2, %v5241 : tensor<512x512x3x3xf32>
    %v5243 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5244 = stablehlo.multiply %v5243, %s4b1g2 : tensor<512xf32>
    %v5245 = stablehlo.add %v5244, %v1269 : tensor<512xf32>
    %v5246 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5247 = stablehlo.multiply %v5246, %s4b1g2v : tensor<512xf32>
    %v5248 = stablehlo.add %v5247, %v5245 : tensor<512xf32>
    %v5249 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5250 = stablehlo.multiply %v5249, %v5248 : tensor<512xf32>
    %v5251 = stablehlo.subtract %s4b1g2, %v5250 : tensor<512xf32>
    %v5252 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5253 = stablehlo.multiply %v5252, %s4b1bt2 : tensor<512xf32>
    %v5254 = stablehlo.add %v5253, %v1272 : tensor<512xf32>
    %v5255 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5256 = stablehlo.multiply %v5255, %s4b1bt2v : tensor<512xf32>
    %v5257 = stablehlo.add %v5256, %v5254 : tensor<512xf32>
    %v5258 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5259 = stablehlo.multiply %v5258, %v5257 : tensor<512xf32>
    %v5260 = stablehlo.subtract %s4b1bt2, %v5259 : tensor<512xf32>
    %v5261 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v5262 = stablehlo.multiply %v5261, %Wd : tensor<512x10xf32>
    %v5263 = stablehlo.add %v5262, %v1126 : tensor<512x10xf32>
    %v5264 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v5265 = stablehlo.multiply %v5264, %Wdv : tensor<512x10xf32>
    %v5266 = stablehlo.add %v5265, %v5263 : tensor<512x10xf32>
    %v5267 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v5268 = stablehlo.multiply %v5267, %v5266 : tensor<512x10xf32>
    %v5269 = stablehlo.subtract %Wd, %v5268 : tensor<512x10xf32>
    %v5270 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v5271 = stablehlo.multiply %v5270, %bd : tensor<10xf32>
    %v5272 = stablehlo.add %v5271, %v1128 : tensor<10xf32>
    %v5273 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v5274 = stablehlo.multiply %v5273, %bdv : tensor<10xf32>
    %v5275 = stablehlo.add %v5274, %v5272 : tensor<10xf32>
    %v5276 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v5277 = stablehlo.multiply %v5276, %v5275 : tensor<10xf32>
    %v5278 = stablehlo.subtract %bd, %v5277 : tensor<10xf32>
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
    return %v4297, %v4306, %v4315, %v4324, %v4333, %v4342, %v4351, %v4360, %v4369, %v4378, %v4387, %v4396, %v4405, %v4414, %v4423, %v4432, %v4441, %v4450, %v4459, %v4468, %v4477, %v4486, %v4495, %v4504, %v4513, %v4522, %v4531, %v4540, %v4549, %v4558, %v4567, %v4576, %v4585, %v4594, %v4603, %v4612, %v4621, %v4630, %v4639, %v4648, %v4657, %v4666, %v4675, %v4684, %v4693, %v4702, %v4711, %v4720, %v4729, %v4738, %v4747, %v4756, %v4765, %v4774, %v4783, %v4792, %v4801, %v4810, %v4819, %v4828, %v4837, %v4846, %v4855, %v4864, %v4873, %v4882, %v4891, %v4900, %v4909, %v4918, %v4927, %v4936, %v4945, %v4954, %v4963, %v4972, %v4981, %v4990, %v4999, %v5008, %v5017, %v5026, %v5035, %v5044, %v5053, %v5062, %v5071, %v5080, %v5089, %v5098, %v5107, %v5116, %v5125, %v5134, %v5143, %v5152, %v5161, %v5170, %v5179, %v5188, %v5197, %v5206, %v5215, %v5224, %v5233, %v5242, %v5251, %v5260, %v5269, %v5278, %sWm, %sgm, %sbtm, %s1b0W1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0g2m, %s1b0bt2m, %s1b1W1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1g2m, %s1b1bt2m, %s1b2W1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2g2m, %s1b2bt2m, %d2W1m, %d2g1m, %d2bt1m, %d2W2m, %d2g2m, %d2bt2m, %d2Wpm, %d2gpm, %d2btpm, %s2b0W1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0g2m, %s2b0bt2m, %s2b1W1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1g2m, %s2b1bt2m, %s2b2W1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2g2m, %s2b2bt2m, %d3W1m, %d3g1m, %d3bt1m, %d3W2m, %d3g2m, %d3bt2m, %d3Wpm, %d3gpm, %d3btpm, %s3b0W1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0g2m, %s3b0bt2m, %s3b1W1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1g2m, %s3b1bt2m, %s3b2W1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2g2m, %s3b2bt2m, %s3b3W1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3g2m, %s3b3bt2m, %s3b4W1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4g2m, %s3b4bt2m, %d4W1m, %d4g1m, %d4bt1m, %d4W2m, %d4g2m, %d4bt2m, %d4Wpm, %d4gpm, %d4btpm, %s4b0W1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0g2m, %s4b0bt2m, %s4b1W1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1g2m, %s4b1bt2m, %Wdm, %bdm, %v4294, %v4303, %v4312, %v4321, %v4330, %v4339, %v4348, %v4357, %v4366, %v4375, %v4384, %v4393, %v4402, %v4411, %v4420, %v4429, %v4438, %v4447, %v4456, %v4465, %v4474, %v4483, %v4492, %v4501, %v4510, %v4519, %v4528, %v4537, %v4546, %v4555, %v4564, %v4573, %v4582, %v4591, %v4600, %v4609, %v4618, %v4627, %v4636, %v4645, %v4654, %v4663, %v4672, %v4681, %v4690, %v4699, %v4708, %v4717, %v4726, %v4735, %v4744, %v4753, %v4762, %v4771, %v4780, %v4789, %v4798, %v4807, %v4816, %v4825, %v4834, %v4843, %v4852, %v4861, %v4870, %v4879, %v4888, %v4897, %v4906, %v4915, %v4924, %v4933, %v4942, %v4951, %v4960, %v4969, %v4978, %v4987, %v4996, %v5005, %v5014, %v5023, %v5032, %v5041, %v5050, %v5059, %v5068, %v5077, %v5086, %v5095, %v5104, %v5113, %v5122, %v5131, %v5140, %v5149, %v5158, %v5167, %v5176, %v5185, %v5194, %v5203, %v5212, %v5221, %v5230, %v5239, %v5248, %v5257, %v5266, %v5275, %loss, %bc1, %bc2, %v3717, %v3728, %v3733, %v3744, %v3749, %v3760, %v3765, %v3776, %v3781, %v3792, %v3797, %v3808, %v3813, %v3824, %v3829, %v3840, %v3845, %v3856, %v3861, %v3872, %v3877, %v3888, %v3893, %v3904, %v3909, %v3920, %v3925, %v3936, %v3941, %v3952, %v3957, %v3968, %v3973, %v3984, %v3989, %v4000, %v4005, %v4016, %v4021, %v4032, %v4037, %v4048, %v4053, %v4064, %v4069, %v4080, %v4085, %v4096, %v4101, %v4112, %v4117, %v4128, %v4133, %v4144, %v4149, %v4160, %v4165, %v4176, %v4181, %v4192, %v4197, %v4208, %v4213, %v4224, %v4229, %v4240, %v4245, %v4256, %v4261, %v4272, %v4277, %v4288 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}

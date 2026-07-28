module @m {
  func.func @resnet34_adam_train_step(%x: tensor<32x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sbi: tensor<64xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0b1: tensor<64xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0b2: tensor<64xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1b1: tensor<64xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1b2: tensor<64xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2b1: tensor<64xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2b2: tensor<64xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2b1: tensor<128xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2b2: tensor<128xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x3x3xf32>, %d2bp: tensor<128xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0b1: tensor<128xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0b2: tensor<128xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1b1: tensor<128xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1b2: tensor<128xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2b1: tensor<128xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2b2: tensor<128xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3b1: tensor<256xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3b2: tensor<256xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x3x3xf32>, %d3bp: tensor<256xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0b1: tensor<256xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0b2: tensor<256xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1b1: tensor<256xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1b2: tensor<256xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2b1: tensor<256xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2b2: tensor<256xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3b1: tensor<256xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3b2: tensor<256xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4b1: tensor<256xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4b2: tensor<256xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4b1: tensor<512xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4b2: tensor<512xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x3x3xf32>, %d4bp: tensor<512xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0b1: tensor<512xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0b2: tensor<512xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1b1: tensor<512xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1b2: tensor<512xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %sWm: tensor<64x3x7x7xf32>, %sbim: tensor<64xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x3x3xf32>, %s1b0b1m: tensor<64xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0b2m: tensor<64xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b1W1m: tensor<64x64x3x3xf32>, %s1b1b1m: tensor<64xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1b2m: tensor<64xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b2W1m: tensor<64x64x3x3xf32>, %s1b2b1m: tensor<64xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2b2m: tensor<64xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %d2W1m: tensor<128x64x3x3xf32>, %d2b1m: tensor<128xf32>, %d2g1m: tensor<128xf32>, %d2bt1m: tensor<128xf32>, %d2W2m: tensor<128x128x3x3xf32>, %d2b2m: tensor<128xf32>, %d2g2m: tensor<128xf32>, %d2bt2m: tensor<128xf32>, %d2Wpm: tensor<128x64x3x3xf32>, %d2bpm: tensor<128xf32>, %d2gpm: tensor<128xf32>, %d2btpm: tensor<128xf32>, %s2b0W1m: tensor<128x128x3x3xf32>, %s2b0b1m: tensor<128xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0b2m: tensor<128xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b1W1m: tensor<128x128x3x3xf32>, %s2b1b1m: tensor<128xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1b2m: tensor<128xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b2W1m: tensor<128x128x3x3xf32>, %s2b2b1m: tensor<128xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2b2m: tensor<128xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %d3W1m: tensor<256x128x3x3xf32>, %d3b1m: tensor<256xf32>, %d3g1m: tensor<256xf32>, %d3bt1m: tensor<256xf32>, %d3W2m: tensor<256x256x3x3xf32>, %d3b2m: tensor<256xf32>, %d3g2m: tensor<256xf32>, %d3bt2m: tensor<256xf32>, %d3Wpm: tensor<256x128x3x3xf32>, %d3bpm: tensor<256xf32>, %d3gpm: tensor<256xf32>, %d3btpm: tensor<256xf32>, %s3b0W1m: tensor<256x256x3x3xf32>, %s3b0b1m: tensor<256xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0b2m: tensor<256xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b1W1m: tensor<256x256x3x3xf32>, %s3b1b1m: tensor<256xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1b2m: tensor<256xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b2W1m: tensor<256x256x3x3xf32>, %s3b2b1m: tensor<256xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2b2m: tensor<256xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b3W1m: tensor<256x256x3x3xf32>, %s3b3b1m: tensor<256xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3b2m: tensor<256xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b4W1m: tensor<256x256x3x3xf32>, %s3b4b1m: tensor<256xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4b2m: tensor<256xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %d4W1m: tensor<512x256x3x3xf32>, %d4b1m: tensor<512xf32>, %d4g1m: tensor<512xf32>, %d4bt1m: tensor<512xf32>, %d4W2m: tensor<512x512x3x3xf32>, %d4b2m: tensor<512xf32>, %d4g2m: tensor<512xf32>, %d4bt2m: tensor<512xf32>, %d4Wpm: tensor<512x256x3x3xf32>, %d4bpm: tensor<512xf32>, %d4gpm: tensor<512xf32>, %d4btpm: tensor<512xf32>, %s4b0W1m: tensor<512x512x3x3xf32>, %s4b0b1m: tensor<512xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0b2m: tensor<512xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b1W1m: tensor<512x512x3x3xf32>, %s4b1b1m: tensor<512xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1b2m: tensor<512xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %Wdm: tensor<512x10xf32>, %bdm: tensor<10xf32>, %sWv: tensor<64x3x7x7xf32>, %sbiv: tensor<64xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x3x3xf32>, %s1b0b1v: tensor<64xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0b2v: tensor<64xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b1W1v: tensor<64x64x3x3xf32>, %s1b1b1v: tensor<64xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1b2v: tensor<64xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b2W1v: tensor<64x64x3x3xf32>, %s1b2b1v: tensor<64xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2b2v: tensor<64xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %d2W1v: tensor<128x64x3x3xf32>, %d2b1v: tensor<128xf32>, %d2g1v: tensor<128xf32>, %d2bt1v: tensor<128xf32>, %d2W2v: tensor<128x128x3x3xf32>, %d2b2v: tensor<128xf32>, %d2g2v: tensor<128xf32>, %d2bt2v: tensor<128xf32>, %d2Wpv: tensor<128x64x3x3xf32>, %d2bpv: tensor<128xf32>, %d2gpv: tensor<128xf32>, %d2btpv: tensor<128xf32>, %s2b0W1v: tensor<128x128x3x3xf32>, %s2b0b1v: tensor<128xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0b2v: tensor<128xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b1W1v: tensor<128x128x3x3xf32>, %s2b1b1v: tensor<128xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1b2v: tensor<128xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b2W1v: tensor<128x128x3x3xf32>, %s2b2b1v: tensor<128xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2b2v: tensor<128xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %d3W1v: tensor<256x128x3x3xf32>, %d3b1v: tensor<256xf32>, %d3g1v: tensor<256xf32>, %d3bt1v: tensor<256xf32>, %d3W2v: tensor<256x256x3x3xf32>, %d3b2v: tensor<256xf32>, %d3g2v: tensor<256xf32>, %d3bt2v: tensor<256xf32>, %d3Wpv: tensor<256x128x3x3xf32>, %d3bpv: tensor<256xf32>, %d3gpv: tensor<256xf32>, %d3btpv: tensor<256xf32>, %s3b0W1v: tensor<256x256x3x3xf32>, %s3b0b1v: tensor<256xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0b2v: tensor<256xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b1W1v: tensor<256x256x3x3xf32>, %s3b1b1v: tensor<256xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1b2v: tensor<256xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b2W1v: tensor<256x256x3x3xf32>, %s3b2b1v: tensor<256xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2b2v: tensor<256xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b3W1v: tensor<256x256x3x3xf32>, %s3b3b1v: tensor<256xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3b2v: tensor<256xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b4W1v: tensor<256x256x3x3xf32>, %s3b4b1v: tensor<256xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4b2v: tensor<256xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %d4W1v: tensor<512x256x3x3xf32>, %d4b1v: tensor<512xf32>, %d4g1v: tensor<512xf32>, %d4bt1v: tensor<512xf32>, %d4W2v: tensor<512x512x3x3xf32>, %d4b2v: tensor<512xf32>, %d4g2v: tensor<512xf32>, %d4bt2v: tensor<512xf32>, %d4Wpv: tensor<512x256x3x3xf32>, %d4bpv: tensor<512xf32>, %d4gpv: tensor<512xf32>, %d4btpv: tensor<512xf32>, %s4b0W1v: tensor<512x512x3x3xf32>, %s4b0b1v: tensor<512xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0b2v: tensor<512xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b1W1v: tensor<512x512x3x3xf32>, %s4b1b1v: tensor<512xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1b2v: tensor<512xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %Wdv: tensor<512x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %d2n1mui: tensor<128xf32>, %d2n1vari: tensor<128xf32>, %d2n2mui: tensor<128xf32>, %d2n2vari: tensor<128xf32>, %d2npmui: tensor<128xf32>, %d2npvari: tensor<128xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %d3n1mui: tensor<256xf32>, %d3n1vari: tensor<256xf32>, %d3n2mui: tensor<256xf32>, %d3n2vari: tensor<256xf32>, %d3npmui: tensor<256xf32>, %d3npvari: tensor<256xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %d4n1mui: tensor<512xf32>, %d4n1vari: tensor<512xf32>, %d4n2mui: tensor<512xf32>, %d4n2vari: tensor<512xf32>, %d4npmui: tensor<512xf32>, %d4npvari: tensor<512xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %onehot: tensor<32x10xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) {
    // ── ResNet-34 batch-BN AdamW train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<32x64x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %sbi, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
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
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x56x56xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v32 = stablehlo.convolution(%v31, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v33 = stablehlo.broadcast_in_dim %s1b0b1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
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
    %v60 = stablehlo.broadcast_in_dim %s1b0b2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
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
    %v88 = stablehlo.broadcast_in_dim %s1b1b1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
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
    %v115 = stablehlo.broadcast_in_dim %s1b1b2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
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
    %v143 = stablehlo.broadcast_in_dim %s1b2b1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
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
    %v170 = stablehlo.broadcast_in_dim %s1b2b2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
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
    %v198 = stablehlo.broadcast_in_dim %d2b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v225 = stablehlo.broadcast_in_dim %d2b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v250 = stablehlo.broadcast_in_dim %d2bp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v278 = stablehlo.broadcast_in_dim %s2b0b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v305 = stablehlo.broadcast_in_dim %s2b0b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v333 = stablehlo.broadcast_in_dim %s2b1b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v360 = stablehlo.broadcast_in_dim %s2b1b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v388 = stablehlo.broadcast_in_dim %s2b2b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v415 = stablehlo.broadcast_in_dim %s2b2b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v443 = stablehlo.broadcast_in_dim %d3b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v470 = stablehlo.broadcast_in_dim %d3b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v495 = stablehlo.broadcast_in_dim %d3bp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v523 = stablehlo.broadcast_in_dim %s3b0b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v550 = stablehlo.broadcast_in_dim %s3b0b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v578 = stablehlo.broadcast_in_dim %s3b1b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v605 = stablehlo.broadcast_in_dim %s3b1b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v633 = stablehlo.broadcast_in_dim %s3b2b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v660 = stablehlo.broadcast_in_dim %s3b2b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v688 = stablehlo.broadcast_in_dim %s3b3b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v715 = stablehlo.broadcast_in_dim %s3b3b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v743 = stablehlo.broadcast_in_dim %s3b4b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v770 = stablehlo.broadcast_in_dim %s3b4b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v798 = stablehlo.broadcast_in_dim %d4b1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
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
    %v825 = stablehlo.broadcast_in_dim %d4b2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
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
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v850 = stablehlo.broadcast_in_dim %d4bp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
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
    %v878 = stablehlo.broadcast_in_dim %s4b0b1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
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
    %v905 = stablehlo.broadcast_in_dim %s4b0b2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
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
    %v933 = stablehlo.broadcast_in_dim %s4b1b1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
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
    %v960 = stablehlo.broadcast_in_dim %s4b1b2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
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
    %v1102 = stablehlo.reshape %v1089 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1104 = stablehlo.reduce(%v1102 init: %v1103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1105 = stablehlo.reshape %v935 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1107 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1108 = stablehlo.reduce(%v1105 init: %v1106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1109 = stablehlo.broadcast_in_dim %v1108, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1110 = stablehlo.divide %v1109, %v1107 : tensor<32x512x7x7xf32>
    %v1111 = stablehlo.subtract %v1105, %v1110 : tensor<32x512x7x7xf32>
    %v1112 = stablehlo.multiply %v1111, %v1111 : tensor<32x512x7x7xf32>
    %v1113 = stablehlo.reduce(%v1112 init: %v1106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1114 = stablehlo.broadcast_in_dim %v1113, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1115 = stablehlo.divide %v1114, %v1107 : tensor<32x512x7x7xf32>
    %v1116 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1117 = stablehlo.add %v1115, %v1116 : tensor<32x512x7x7xf32>
    %v1118 = stablehlo.rsqrt %v1117 : tensor<32x512x7x7xf32>
    %v1119 = stablehlo.multiply %v1111, %v1118 : tensor<32x512x7x7xf32>
    %v1120 = stablehlo.reshape %v1059 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1121 = stablehlo.multiply %v1120, %v1119 : tensor<32x512x7x7xf32>
    %v1122 = stablehlo.reduce(%v1121 init: %v1106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1123 = stablehlo.reshape %v1059 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1125 = stablehlo.reduce(%v1123 init: %v1124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1126 = stablehlo.reshape %v957 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1127 = stablehlo.reshape %v1051 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1128 = stablehlo.transpose %v1126, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1129 = stablehlo.transpose %v1127, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1130 = stablehlo.convolution(%v1128, %v1129)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1131 = stablehlo.transpose %v1130, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1132 = stablehlo.reshape %v1051 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1134 = stablehlo.reduce(%v1132 init: %v1133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1135 = stablehlo.reshape %v962 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1137 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1138 = stablehlo.reduce(%v1135 init: %v1136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1139 = stablehlo.broadcast_in_dim %v1138, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1140 = stablehlo.divide %v1139, %v1137 : tensor<32x512x7x7xf32>
    %v1141 = stablehlo.subtract %v1135, %v1140 : tensor<32x512x7x7xf32>
    %v1142 = stablehlo.multiply %v1141, %v1141 : tensor<32x512x7x7xf32>
    %v1143 = stablehlo.reduce(%v1142 init: %v1136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1144 = stablehlo.broadcast_in_dim %v1143, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1145 = stablehlo.divide %v1144, %v1137 : tensor<32x512x7x7xf32>
    %v1146 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1147 = stablehlo.add %v1145, %v1146 : tensor<32x512x7x7xf32>
    %v1148 = stablehlo.rsqrt %v1147 : tensor<32x512x7x7xf32>
    %v1149 = stablehlo.multiply %v1141, %v1148 : tensor<32x512x7x7xf32>
    %v1150 = stablehlo.reshape %v1021 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1151 = stablehlo.multiply %v1150, %v1149 : tensor<32x512x7x7xf32>
    %v1152 = stablehlo.reduce(%v1151 init: %v1136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1153 = stablehlo.reshape %v1021 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1155 = stablehlo.reduce(%v1153 init: %v1154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1156 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1157 = stablehlo.compare GT, %v928, %v1156 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1158 = stablehlo.select %v1157, %v1095, %v1156 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1159 = stablehlo.reshape %v907 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1161 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1162 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1163 = stablehlo.reduce(%v1159 init: %v1160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1164 = stablehlo.broadcast_in_dim %v1163, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1165 = stablehlo.divide %v1164, %v1161 : tensor<32x512x7x7xf32>
    %v1166 = stablehlo.subtract %v1159, %v1165 : tensor<32x512x7x7xf32>
    %v1167 = stablehlo.multiply %v1166, %v1166 : tensor<32x512x7x7xf32>
    %v1168 = stablehlo.reduce(%v1167 init: %v1160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1169 = stablehlo.broadcast_in_dim %v1168, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1170 = stablehlo.divide %v1169, %v1161 : tensor<32x512x7x7xf32>
    %v1171 = stablehlo.add %v1170, %v1162 : tensor<32x512x7x7xf32>
    %v1172 = stablehlo.rsqrt %v1171 : tensor<32x512x7x7xf32>
    %v1173 = stablehlo.multiply %v1166, %v1172 : tensor<32x512x7x7xf32>
    %v1174 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1175 = stablehlo.reshape %v1158 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1176 = stablehlo.multiply %v1174, %v1175 : tensor<32x512x7x7xf32>
    %v1177 = stablehlo.reduce(%v1176 init: %v1160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1178 = stablehlo.broadcast_in_dim %v1177, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1179 = stablehlo.multiply %v1173, %v1176 : tensor<32x512x7x7xf32>
    %v1180 = stablehlo.reduce(%v1179 init: %v1160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1181 = stablehlo.broadcast_in_dim %v1180, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1182 = stablehlo.multiply %v1176, %v1161 : tensor<32x512x7x7xf32>
    %v1183 = stablehlo.subtract %v1182, %v1178 : tensor<32x512x7x7xf32>
    %v1184 = stablehlo.multiply %v1173, %v1181 : tensor<32x512x7x7xf32>
    %v1185 = stablehlo.subtract %v1183, %v1184 : tensor<32x512x7x7xf32>
    %v1186 = stablehlo.divide %v1172, %v1161 : tensor<32x512x7x7xf32>
    %v1187 = stablehlo.multiply %v1186, %v1185 : tensor<32x512x7x7xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1190 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1191 = stablehlo.transpose %v1190, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1192 = stablehlo.convolution(%v1189, %v1191)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1194 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1195 = stablehlo.compare GT, %v900, %v1194 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1196 = stablehlo.select %v1195, %v1193, %v1194 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1197 = stablehlo.reshape %v880 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1199 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1200 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1201 = stablehlo.reduce(%v1197 init: %v1198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1202 = stablehlo.broadcast_in_dim %v1201, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1203 = stablehlo.divide %v1202, %v1199 : tensor<32x512x7x7xf32>
    %v1204 = stablehlo.subtract %v1197, %v1203 : tensor<32x512x7x7xf32>
    %v1205 = stablehlo.multiply %v1204, %v1204 : tensor<32x512x7x7xf32>
    %v1206 = stablehlo.reduce(%v1205 init: %v1198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1207 = stablehlo.broadcast_in_dim %v1206, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1208 = stablehlo.divide %v1207, %v1199 : tensor<32x512x7x7xf32>
    %v1209 = stablehlo.add %v1208, %v1200 : tensor<32x512x7x7xf32>
    %v1210 = stablehlo.rsqrt %v1209 : tensor<32x512x7x7xf32>
    %v1211 = stablehlo.multiply %v1204, %v1210 : tensor<32x512x7x7xf32>
    %v1212 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1213 = stablehlo.reshape %v1196 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1214 = stablehlo.multiply %v1212, %v1213 : tensor<32x512x7x7xf32>
    %v1215 = stablehlo.reduce(%v1214 init: %v1198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1216 = stablehlo.broadcast_in_dim %v1215, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1217 = stablehlo.multiply %v1211, %v1214 : tensor<32x512x7x7xf32>
    %v1218 = stablehlo.reduce(%v1217 init: %v1198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1219 = stablehlo.broadcast_in_dim %v1218, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1220 = stablehlo.multiply %v1214, %v1199 : tensor<32x512x7x7xf32>
    %v1221 = stablehlo.subtract %v1220, %v1216 : tensor<32x512x7x7xf32>
    %v1222 = stablehlo.multiply %v1211, %v1219 : tensor<32x512x7x7xf32>
    %v1223 = stablehlo.subtract %v1221, %v1222 : tensor<32x512x7x7xf32>
    %v1224 = stablehlo.divide %v1210, %v1199 : tensor<32x512x7x7xf32>
    %v1225 = stablehlo.multiply %v1224, %v1223 : tensor<32x512x7x7xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1227 = stablehlo.reshape %v1226 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1228 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1229 = stablehlo.transpose %v1228, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1230 = stablehlo.convolution(%v1227, %v1229)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1232 = stablehlo.add %v1231, %v1158 : tensor<32x25088xf32>
    %v1233 = stablehlo.reshape %v875 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1234 = stablehlo.reshape %v1226 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1235 = stablehlo.transpose %v1233, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1236 = stablehlo.transpose %v1234, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1237 = stablehlo.convolution(%v1235, %v1236)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1238 = stablehlo.transpose %v1237, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1239 = stablehlo.reshape %v1226 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1241 = stablehlo.reduce(%v1239 init: %v1240) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1242 = stablehlo.reshape %v880 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1243 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1244 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1245 = stablehlo.reduce(%v1242 init: %v1243) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1246 = stablehlo.broadcast_in_dim %v1245, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1247 = stablehlo.divide %v1246, %v1244 : tensor<32x512x7x7xf32>
    %v1248 = stablehlo.subtract %v1242, %v1247 : tensor<32x512x7x7xf32>
    %v1249 = stablehlo.multiply %v1248, %v1248 : tensor<32x512x7x7xf32>
    %v1250 = stablehlo.reduce(%v1249 init: %v1243) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1251 = stablehlo.broadcast_in_dim %v1250, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1252 = stablehlo.divide %v1251, %v1244 : tensor<32x512x7x7xf32>
    %v1253 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1254 = stablehlo.add %v1252, %v1253 : tensor<32x512x7x7xf32>
    %v1255 = stablehlo.rsqrt %v1254 : tensor<32x512x7x7xf32>
    %v1256 = stablehlo.multiply %v1248, %v1255 : tensor<32x512x7x7xf32>
    %v1257 = stablehlo.reshape %v1196 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1258 = stablehlo.multiply %v1257, %v1256 : tensor<32x512x7x7xf32>
    %v1259 = stablehlo.reduce(%v1258 init: %v1243) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1260 = stablehlo.reshape %v1196 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1262 = stablehlo.reduce(%v1260 init: %v1261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1263 = stablehlo.reshape %v902 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1264 = stablehlo.reshape %v1188 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1265 = stablehlo.transpose %v1263, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1266 = stablehlo.transpose %v1264, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1267 = stablehlo.convolution(%v1265, %v1266)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1268 = stablehlo.transpose %v1267, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1269 = stablehlo.reshape %v1188 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1270 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1271 = stablehlo.reduce(%v1269 init: %v1270) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1272 = stablehlo.reshape %v907 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1274 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1275 = stablehlo.reduce(%v1272 init: %v1273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1276 = stablehlo.broadcast_in_dim %v1275, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1277 = stablehlo.divide %v1276, %v1274 : tensor<32x512x7x7xf32>
    %v1278 = stablehlo.subtract %v1272, %v1277 : tensor<32x512x7x7xf32>
    %v1279 = stablehlo.multiply %v1278, %v1278 : tensor<32x512x7x7xf32>
    %v1280 = stablehlo.reduce(%v1279 init: %v1273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1281 = stablehlo.broadcast_in_dim %v1280, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1282 = stablehlo.divide %v1281, %v1274 : tensor<32x512x7x7xf32>
    %v1283 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1284 = stablehlo.add %v1282, %v1283 : tensor<32x512x7x7xf32>
    %v1285 = stablehlo.rsqrt %v1284 : tensor<32x512x7x7xf32>
    %v1286 = stablehlo.multiply %v1278, %v1285 : tensor<32x512x7x7xf32>
    %v1287 = stablehlo.reshape %v1158 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1288 = stablehlo.multiply %v1287, %v1286 : tensor<32x512x7x7xf32>
    %v1289 = stablehlo.reduce(%v1288 init: %v1273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1290 = stablehlo.reshape %v1158 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1291 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1292 = stablehlo.reduce(%v1290 init: %v1291) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1293 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1294 = stablehlo.compare GT, %v873, %v1293 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1295 = stablehlo.select %v1294, %v1232, %v1293 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1296 = stablehlo.reshape %v827 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1298 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1299 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1300 = stablehlo.reduce(%v1296 init: %v1297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1301 = stablehlo.broadcast_in_dim %v1300, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1302 = stablehlo.divide %v1301, %v1298 : tensor<32x512x7x7xf32>
    %v1303 = stablehlo.subtract %v1296, %v1302 : tensor<32x512x7x7xf32>
    %v1304 = stablehlo.multiply %v1303, %v1303 : tensor<32x512x7x7xf32>
    %v1305 = stablehlo.reduce(%v1304 init: %v1297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1306 = stablehlo.broadcast_in_dim %v1305, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1307 = stablehlo.divide %v1306, %v1298 : tensor<32x512x7x7xf32>
    %v1308 = stablehlo.add %v1307, %v1299 : tensor<32x512x7x7xf32>
    %v1309 = stablehlo.rsqrt %v1308 : tensor<32x512x7x7xf32>
    %v1310 = stablehlo.multiply %v1303, %v1309 : tensor<32x512x7x7xf32>
    %v1311 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1312 = stablehlo.reshape %v1295 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1313 = stablehlo.multiply %v1311, %v1312 : tensor<32x512x7x7xf32>
    %v1314 = stablehlo.reduce(%v1313 init: %v1297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1315 = stablehlo.broadcast_in_dim %v1314, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1316 = stablehlo.multiply %v1310, %v1313 : tensor<32x512x7x7xf32>
    %v1317 = stablehlo.reduce(%v1316 init: %v1297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1318 = stablehlo.broadcast_in_dim %v1317, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1319 = stablehlo.multiply %v1313, %v1298 : tensor<32x512x7x7xf32>
    %v1320 = stablehlo.subtract %v1319, %v1315 : tensor<32x512x7x7xf32>
    %v1321 = stablehlo.multiply %v1310, %v1318 : tensor<32x512x7x7xf32>
    %v1322 = stablehlo.subtract %v1320, %v1321 : tensor<32x512x7x7xf32>
    %v1323 = stablehlo.divide %v1309, %v1298 : tensor<32x512x7x7xf32>
    %v1324 = stablehlo.multiply %v1323, %v1322 : tensor<32x512x7x7xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1326 = stablehlo.reshape %v1325 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1327 = stablehlo.reverse %d4W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1328 = stablehlo.transpose %v1327, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1329 = stablehlo.convolution(%v1326, %v1328)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1330 = stablehlo.reshape %v1329 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1331 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1332 = stablehlo.compare GT, %v820, %v1331 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1333 = stablehlo.select %v1332, %v1330, %v1331 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1334 = stablehlo.reshape %v800 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1336 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1337 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1338 = stablehlo.reduce(%v1334 init: %v1335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1339 = stablehlo.broadcast_in_dim %v1338, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1340 = stablehlo.divide %v1339, %v1336 : tensor<32x512x7x7xf32>
    %v1341 = stablehlo.subtract %v1334, %v1340 : tensor<32x512x7x7xf32>
    %v1342 = stablehlo.multiply %v1341, %v1341 : tensor<32x512x7x7xf32>
    %v1343 = stablehlo.reduce(%v1342 init: %v1335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1344 = stablehlo.broadcast_in_dim %v1343, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1345 = stablehlo.divide %v1344, %v1336 : tensor<32x512x7x7xf32>
    %v1346 = stablehlo.add %v1345, %v1337 : tensor<32x512x7x7xf32>
    %v1347 = stablehlo.rsqrt %v1346 : tensor<32x512x7x7xf32>
    %v1348 = stablehlo.multiply %v1341, %v1347 : tensor<32x512x7x7xf32>
    %v1349 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1350 = stablehlo.reshape %v1333 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1351 = stablehlo.multiply %v1349, %v1350 : tensor<32x512x7x7xf32>
    %v1352 = stablehlo.reduce(%v1351 init: %v1335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1353 = stablehlo.broadcast_in_dim %v1352, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1354 = stablehlo.multiply %v1348, %v1351 : tensor<32x512x7x7xf32>
    %v1355 = stablehlo.reduce(%v1354 init: %v1335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1356 = stablehlo.broadcast_in_dim %v1355, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1357 = stablehlo.multiply %v1351, %v1336 : tensor<32x512x7x7xf32>
    %v1358 = stablehlo.subtract %v1357, %v1353 : tensor<32x512x7x7xf32>
    %v1359 = stablehlo.multiply %v1348, %v1356 : tensor<32x512x7x7xf32>
    %v1360 = stablehlo.subtract %v1358, %v1359 : tensor<32x512x7x7xf32>
    %v1361 = stablehlo.divide %v1347, %v1336 : tensor<32x512x7x7xf32>
    %v1362 = stablehlo.multiply %v1361, %v1360 : tensor<32x512x7x7xf32>
    %v1363 = stablehlo.reshape %v1362 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1364 = stablehlo.reshape %v1363 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1366 = stablehlo.pad %v1364, %v1365, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1367 = stablehlo.reverse %d4W1, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v1368 = stablehlo.transpose %v1367, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1369 = stablehlo.convolution(%v1366, %v1368)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1370 = stablehlo.reshape %v1369 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1371 = stablehlo.reshape %v852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1372 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1373 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1374 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1375 = stablehlo.reduce(%v1371 init: %v1372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1376 = stablehlo.broadcast_in_dim %v1375, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1377 = stablehlo.divide %v1376, %v1373 : tensor<32x512x7x7xf32>
    %v1378 = stablehlo.subtract %v1371, %v1377 : tensor<32x512x7x7xf32>
    %v1379 = stablehlo.multiply %v1378, %v1378 : tensor<32x512x7x7xf32>
    %v1380 = stablehlo.reduce(%v1379 init: %v1372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1381 = stablehlo.broadcast_in_dim %v1380, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1382 = stablehlo.divide %v1381, %v1373 : tensor<32x512x7x7xf32>
    %v1383 = stablehlo.add %v1382, %v1374 : tensor<32x512x7x7xf32>
    %v1384 = stablehlo.rsqrt %v1383 : tensor<32x512x7x7xf32>
    %v1385 = stablehlo.multiply %v1378, %v1384 : tensor<32x512x7x7xf32>
    %v1386 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1387 = stablehlo.reshape %v1295 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1388 = stablehlo.multiply %v1386, %v1387 : tensor<32x512x7x7xf32>
    %v1389 = stablehlo.reduce(%v1388 init: %v1372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1390 = stablehlo.broadcast_in_dim %v1389, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1391 = stablehlo.multiply %v1385, %v1388 : tensor<32x512x7x7xf32>
    %v1392 = stablehlo.reduce(%v1391 init: %v1372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1393 = stablehlo.broadcast_in_dim %v1392, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1394 = stablehlo.multiply %v1388, %v1373 : tensor<32x512x7x7xf32>
    %v1395 = stablehlo.subtract %v1394, %v1390 : tensor<32x512x7x7xf32>
    %v1396 = stablehlo.multiply %v1385, %v1393 : tensor<32x512x7x7xf32>
    %v1397 = stablehlo.subtract %v1395, %v1396 : tensor<32x512x7x7xf32>
    %v1398 = stablehlo.divide %v1384, %v1373 : tensor<32x512x7x7xf32>
    %v1399 = stablehlo.multiply %v1398, %v1397 : tensor<32x512x7x7xf32>
    %v1400 = stablehlo.reshape %v1399 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1401 = stablehlo.reshape %v1400 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1403 = stablehlo.pad %v1401, %v1402, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1404 = stablehlo.reverse %d4Wp, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v1405 = stablehlo.transpose %v1404, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1406 = stablehlo.convolution(%v1403, %v1405)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1408 = stablehlo.add %v1370, %v1407 : tensor<32x50176xf32>
    %v1409 = stablehlo.reshape %v795 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1410 = stablehlo.reshape %v1363 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1411 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1412 = stablehlo.pad %v1410, %v1411, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1413 = stablehlo.transpose %v1409, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1414 = stablehlo.transpose %v1412, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1415 = stablehlo.convolution(%v1413, %v1414)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x3x3xf32>
    %v1416 = stablehlo.transpose %v1415, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1417 = stablehlo.reshape %v1363 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1418 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1419 = stablehlo.reduce(%v1417 init: %v1418) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1420 = stablehlo.reshape %v800 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1422 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1423 = stablehlo.reduce(%v1420 init: %v1421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1424 = stablehlo.broadcast_in_dim %v1423, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1425 = stablehlo.divide %v1424, %v1422 : tensor<32x512x7x7xf32>
    %v1426 = stablehlo.subtract %v1420, %v1425 : tensor<32x512x7x7xf32>
    %v1427 = stablehlo.multiply %v1426, %v1426 : tensor<32x512x7x7xf32>
    %v1428 = stablehlo.reduce(%v1427 init: %v1421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1429 = stablehlo.broadcast_in_dim %v1428, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1430 = stablehlo.divide %v1429, %v1422 : tensor<32x512x7x7xf32>
    %v1431 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1432 = stablehlo.add %v1430, %v1431 : tensor<32x512x7x7xf32>
    %v1433 = stablehlo.rsqrt %v1432 : tensor<32x512x7x7xf32>
    %v1434 = stablehlo.multiply %v1426, %v1433 : tensor<32x512x7x7xf32>
    %v1435 = stablehlo.reshape %v1333 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1436 = stablehlo.multiply %v1435, %v1434 : tensor<32x512x7x7xf32>
    %v1437 = stablehlo.reduce(%v1436 init: %v1421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1438 = stablehlo.reshape %v1333 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1440 = stablehlo.reduce(%v1438 init: %v1439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1441 = stablehlo.reshape %v822 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1442 = stablehlo.reshape %v1325 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1443 = stablehlo.transpose %v1441, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1444 = stablehlo.transpose %v1442, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1445 = stablehlo.convolution(%v1443, %v1444)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1446 = stablehlo.transpose %v1445, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1447 = stablehlo.reshape %v1325 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1448 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1449 = stablehlo.reduce(%v1447 init: %v1448) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1450 = stablehlo.reshape %v827 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1451 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1452 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1453 = stablehlo.reduce(%v1450 init: %v1451) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1454 = stablehlo.broadcast_in_dim %v1453, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1455 = stablehlo.divide %v1454, %v1452 : tensor<32x512x7x7xf32>
    %v1456 = stablehlo.subtract %v1450, %v1455 : tensor<32x512x7x7xf32>
    %v1457 = stablehlo.multiply %v1456, %v1456 : tensor<32x512x7x7xf32>
    %v1458 = stablehlo.reduce(%v1457 init: %v1451) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1459 = stablehlo.broadcast_in_dim %v1458, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1460 = stablehlo.divide %v1459, %v1452 : tensor<32x512x7x7xf32>
    %v1461 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1462 = stablehlo.add %v1460, %v1461 : tensor<32x512x7x7xf32>
    %v1463 = stablehlo.rsqrt %v1462 : tensor<32x512x7x7xf32>
    %v1464 = stablehlo.multiply %v1456, %v1463 : tensor<32x512x7x7xf32>
    %v1465 = stablehlo.reshape %v1295 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1466 = stablehlo.multiply %v1465, %v1464 : tensor<32x512x7x7xf32>
    %v1467 = stablehlo.reduce(%v1466 init: %v1451) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1468 = stablehlo.reshape %v1295 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1470 = stablehlo.reduce(%v1468 init: %v1469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1471 = stablehlo.reshape %v795 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1472 = stablehlo.reshape %v1400 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1474 = stablehlo.pad %v1472, %v1473, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1475 = stablehlo.transpose %v1471, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1476 = stablehlo.transpose %v1474, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1477 = stablehlo.convolution(%v1475, %v1476)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x3x3xf32>
    %v1478 = stablehlo.transpose %v1477, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1479 = stablehlo.reshape %v1400 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1481 = stablehlo.reduce(%v1479 init: %v1480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1482 = stablehlo.reshape %v852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1483 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1484 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1485 = stablehlo.reduce(%v1482 init: %v1483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1486 = stablehlo.broadcast_in_dim %v1485, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1487 = stablehlo.divide %v1486, %v1484 : tensor<32x512x7x7xf32>
    %v1488 = stablehlo.subtract %v1482, %v1487 : tensor<32x512x7x7xf32>
    %v1489 = stablehlo.multiply %v1488, %v1488 : tensor<32x512x7x7xf32>
    %v1490 = stablehlo.reduce(%v1489 init: %v1483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1491 = stablehlo.broadcast_in_dim %v1490, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1492 = stablehlo.divide %v1491, %v1484 : tensor<32x512x7x7xf32>
    %v1493 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1494 = stablehlo.add %v1492, %v1493 : tensor<32x512x7x7xf32>
    %v1495 = stablehlo.rsqrt %v1494 : tensor<32x512x7x7xf32>
    %v1496 = stablehlo.multiply %v1488, %v1495 : tensor<32x512x7x7xf32>
    %v1497 = stablehlo.reshape %v1295 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1498 = stablehlo.multiply %v1497, %v1496 : tensor<32x512x7x7xf32>
    %v1499 = stablehlo.reduce(%v1498 init: %v1483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1500 = stablehlo.reshape %v1295 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1501 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1502 = stablehlo.reduce(%v1500 init: %v1501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1503 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1504 = stablehlo.compare GT, %v793, %v1503 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1505 = stablehlo.select %v1504, %v1408, %v1503 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1506 = stablehlo.reshape %v772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1507 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1508 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1509 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1510 = stablehlo.reduce(%v1506 init: %v1507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1511 = stablehlo.broadcast_in_dim %v1510, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1512 = stablehlo.divide %v1511, %v1508 : tensor<32x256x14x14xf32>
    %v1513 = stablehlo.subtract %v1506, %v1512 : tensor<32x256x14x14xf32>
    %v1514 = stablehlo.multiply %v1513, %v1513 : tensor<32x256x14x14xf32>
    %v1515 = stablehlo.reduce(%v1514 init: %v1507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1516 = stablehlo.broadcast_in_dim %v1515, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1517 = stablehlo.divide %v1516, %v1508 : tensor<32x256x14x14xf32>
    %v1518 = stablehlo.add %v1517, %v1509 : tensor<32x256x14x14xf32>
    %v1519 = stablehlo.rsqrt %v1518 : tensor<32x256x14x14xf32>
    %v1520 = stablehlo.multiply %v1513, %v1519 : tensor<32x256x14x14xf32>
    %v1521 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1522 = stablehlo.reshape %v1505 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1523 = stablehlo.multiply %v1521, %v1522 : tensor<32x256x14x14xf32>
    %v1524 = stablehlo.reduce(%v1523 init: %v1507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1525 = stablehlo.broadcast_in_dim %v1524, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1526 = stablehlo.multiply %v1520, %v1523 : tensor<32x256x14x14xf32>
    %v1527 = stablehlo.reduce(%v1526 init: %v1507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1528 = stablehlo.broadcast_in_dim %v1527, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1529 = stablehlo.multiply %v1523, %v1508 : tensor<32x256x14x14xf32>
    %v1530 = stablehlo.subtract %v1529, %v1525 : tensor<32x256x14x14xf32>
    %v1531 = stablehlo.multiply %v1520, %v1528 : tensor<32x256x14x14xf32>
    %v1532 = stablehlo.subtract %v1530, %v1531 : tensor<32x256x14x14xf32>
    %v1533 = stablehlo.divide %v1519, %v1508 : tensor<32x256x14x14xf32>
    %v1534 = stablehlo.multiply %v1533, %v1532 : tensor<32x256x14x14xf32>
    %v1535 = stablehlo.reshape %v1534 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1536 = stablehlo.reshape %v1535 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1537 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1538 = stablehlo.transpose %v1537, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1539 = stablehlo.convolution(%v1536, %v1538)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1540 = stablehlo.reshape %v1539 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1541 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1542 = stablehlo.compare GT, %v765, %v1541 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1543 = stablehlo.select %v1542, %v1540, %v1541 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1544 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1546 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1547 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1548 = stablehlo.reduce(%v1544 init: %v1545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1549 = stablehlo.broadcast_in_dim %v1548, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1550 = stablehlo.divide %v1549, %v1546 : tensor<32x256x14x14xf32>
    %v1551 = stablehlo.subtract %v1544, %v1550 : tensor<32x256x14x14xf32>
    %v1552 = stablehlo.multiply %v1551, %v1551 : tensor<32x256x14x14xf32>
    %v1553 = stablehlo.reduce(%v1552 init: %v1545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1554 = stablehlo.broadcast_in_dim %v1553, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1555 = stablehlo.divide %v1554, %v1546 : tensor<32x256x14x14xf32>
    %v1556 = stablehlo.add %v1555, %v1547 : tensor<32x256x14x14xf32>
    %v1557 = stablehlo.rsqrt %v1556 : tensor<32x256x14x14xf32>
    %v1558 = stablehlo.multiply %v1551, %v1557 : tensor<32x256x14x14xf32>
    %v1559 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1560 = stablehlo.reshape %v1543 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1561 = stablehlo.multiply %v1559, %v1560 : tensor<32x256x14x14xf32>
    %v1562 = stablehlo.reduce(%v1561 init: %v1545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1563 = stablehlo.broadcast_in_dim %v1562, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1564 = stablehlo.multiply %v1558, %v1561 : tensor<32x256x14x14xf32>
    %v1565 = stablehlo.reduce(%v1564 init: %v1545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1566 = stablehlo.broadcast_in_dim %v1565, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1567 = stablehlo.multiply %v1561, %v1546 : tensor<32x256x14x14xf32>
    %v1568 = stablehlo.subtract %v1567, %v1563 : tensor<32x256x14x14xf32>
    %v1569 = stablehlo.multiply %v1558, %v1566 : tensor<32x256x14x14xf32>
    %v1570 = stablehlo.subtract %v1568, %v1569 : tensor<32x256x14x14xf32>
    %v1571 = stablehlo.divide %v1557, %v1546 : tensor<32x256x14x14xf32>
    %v1572 = stablehlo.multiply %v1571, %v1570 : tensor<32x256x14x14xf32>
    %v1573 = stablehlo.reshape %v1572 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1574 = stablehlo.reshape %v1573 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1575 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1576 = stablehlo.transpose %v1575, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1577 = stablehlo.convolution(%v1574, %v1576)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1578 = stablehlo.reshape %v1577 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1579 = stablehlo.add %v1578, %v1505 : tensor<32x50176xf32>
    %v1580 = stablehlo.reshape %v740 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1581 = stablehlo.reshape %v1573 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1582 = stablehlo.transpose %v1580, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1583 = stablehlo.transpose %v1581, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1584 = stablehlo.convolution(%v1582, %v1583)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1585 = stablehlo.transpose %v1584, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1586 = stablehlo.reshape %v1573 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1587 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1588 = stablehlo.reduce(%v1586 init: %v1587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1589 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1591 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1592 = stablehlo.reduce(%v1589 init: %v1590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1593 = stablehlo.broadcast_in_dim %v1592, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1594 = stablehlo.divide %v1593, %v1591 : tensor<32x256x14x14xf32>
    %v1595 = stablehlo.subtract %v1589, %v1594 : tensor<32x256x14x14xf32>
    %v1596 = stablehlo.multiply %v1595, %v1595 : tensor<32x256x14x14xf32>
    %v1597 = stablehlo.reduce(%v1596 init: %v1590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1598 = stablehlo.broadcast_in_dim %v1597, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1599 = stablehlo.divide %v1598, %v1591 : tensor<32x256x14x14xf32>
    %v1600 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1601 = stablehlo.add %v1599, %v1600 : tensor<32x256x14x14xf32>
    %v1602 = stablehlo.rsqrt %v1601 : tensor<32x256x14x14xf32>
    %v1603 = stablehlo.multiply %v1595, %v1602 : tensor<32x256x14x14xf32>
    %v1604 = stablehlo.reshape %v1543 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1605 = stablehlo.multiply %v1604, %v1603 : tensor<32x256x14x14xf32>
    %v1606 = stablehlo.reduce(%v1605 init: %v1590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1607 = stablehlo.reshape %v1543 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1609 = stablehlo.reduce(%v1607 init: %v1608) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1610 = stablehlo.reshape %v767 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1611 = stablehlo.reshape %v1535 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1612 = stablehlo.transpose %v1610, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1613 = stablehlo.transpose %v1611, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1614 = stablehlo.convolution(%v1612, %v1613)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1615 = stablehlo.transpose %v1614, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1616 = stablehlo.reshape %v1535 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1618 = stablehlo.reduce(%v1616 init: %v1617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1619 = stablehlo.reshape %v772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1620 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1621 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1622 = stablehlo.reduce(%v1619 init: %v1620) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1623 = stablehlo.broadcast_in_dim %v1622, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1624 = stablehlo.divide %v1623, %v1621 : tensor<32x256x14x14xf32>
    %v1625 = stablehlo.subtract %v1619, %v1624 : tensor<32x256x14x14xf32>
    %v1626 = stablehlo.multiply %v1625, %v1625 : tensor<32x256x14x14xf32>
    %v1627 = stablehlo.reduce(%v1626 init: %v1620) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1628 = stablehlo.broadcast_in_dim %v1627, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1629 = stablehlo.divide %v1628, %v1621 : tensor<32x256x14x14xf32>
    %v1630 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1631 = stablehlo.add %v1629, %v1630 : tensor<32x256x14x14xf32>
    %v1632 = stablehlo.rsqrt %v1631 : tensor<32x256x14x14xf32>
    %v1633 = stablehlo.multiply %v1625, %v1632 : tensor<32x256x14x14xf32>
    %v1634 = stablehlo.reshape %v1505 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1635 = stablehlo.multiply %v1634, %v1633 : tensor<32x256x14x14xf32>
    %v1636 = stablehlo.reduce(%v1635 init: %v1620) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1637 = stablehlo.reshape %v1505 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1638 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1639 = stablehlo.reduce(%v1637 init: %v1638) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1640 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1641 = stablehlo.compare GT, %v738, %v1640 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1642 = stablehlo.select %v1641, %v1579, %v1640 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1643 = stablehlo.reshape %v717 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1644 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1645 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1646 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1647 = stablehlo.reduce(%v1643 init: %v1644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1648 = stablehlo.broadcast_in_dim %v1647, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1649 = stablehlo.divide %v1648, %v1645 : tensor<32x256x14x14xf32>
    %v1650 = stablehlo.subtract %v1643, %v1649 : tensor<32x256x14x14xf32>
    %v1651 = stablehlo.multiply %v1650, %v1650 : tensor<32x256x14x14xf32>
    %v1652 = stablehlo.reduce(%v1651 init: %v1644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1653 = stablehlo.broadcast_in_dim %v1652, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1654 = stablehlo.divide %v1653, %v1645 : tensor<32x256x14x14xf32>
    %v1655 = stablehlo.add %v1654, %v1646 : tensor<32x256x14x14xf32>
    %v1656 = stablehlo.rsqrt %v1655 : tensor<32x256x14x14xf32>
    %v1657 = stablehlo.multiply %v1650, %v1656 : tensor<32x256x14x14xf32>
    %v1658 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1659 = stablehlo.reshape %v1642 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1660 = stablehlo.multiply %v1658, %v1659 : tensor<32x256x14x14xf32>
    %v1661 = stablehlo.reduce(%v1660 init: %v1644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1662 = stablehlo.broadcast_in_dim %v1661, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1663 = stablehlo.multiply %v1657, %v1660 : tensor<32x256x14x14xf32>
    %v1664 = stablehlo.reduce(%v1663 init: %v1644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1665 = stablehlo.broadcast_in_dim %v1664, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1666 = stablehlo.multiply %v1660, %v1645 : tensor<32x256x14x14xf32>
    %v1667 = stablehlo.subtract %v1666, %v1662 : tensor<32x256x14x14xf32>
    %v1668 = stablehlo.multiply %v1657, %v1665 : tensor<32x256x14x14xf32>
    %v1669 = stablehlo.subtract %v1667, %v1668 : tensor<32x256x14x14xf32>
    %v1670 = stablehlo.divide %v1656, %v1645 : tensor<32x256x14x14xf32>
    %v1671 = stablehlo.multiply %v1670, %v1669 : tensor<32x256x14x14xf32>
    %v1672 = stablehlo.reshape %v1671 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1673 = stablehlo.reshape %v1672 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1674 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1675 = stablehlo.transpose %v1674, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1676 = stablehlo.convolution(%v1673, %v1675)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1677 = stablehlo.reshape %v1676 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1678 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1679 = stablehlo.compare GT, %v710, %v1678 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1680 = stablehlo.select %v1679, %v1677, %v1678 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1681 = stablehlo.reshape %v690 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1682 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1683 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1684 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1685 = stablehlo.reduce(%v1681 init: %v1682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1686 = stablehlo.broadcast_in_dim %v1685, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1687 = stablehlo.divide %v1686, %v1683 : tensor<32x256x14x14xf32>
    %v1688 = stablehlo.subtract %v1681, %v1687 : tensor<32x256x14x14xf32>
    %v1689 = stablehlo.multiply %v1688, %v1688 : tensor<32x256x14x14xf32>
    %v1690 = stablehlo.reduce(%v1689 init: %v1682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1691 = stablehlo.broadcast_in_dim %v1690, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1692 = stablehlo.divide %v1691, %v1683 : tensor<32x256x14x14xf32>
    %v1693 = stablehlo.add %v1692, %v1684 : tensor<32x256x14x14xf32>
    %v1694 = stablehlo.rsqrt %v1693 : tensor<32x256x14x14xf32>
    %v1695 = stablehlo.multiply %v1688, %v1694 : tensor<32x256x14x14xf32>
    %v1696 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1697 = stablehlo.reshape %v1680 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1698 = stablehlo.multiply %v1696, %v1697 : tensor<32x256x14x14xf32>
    %v1699 = stablehlo.reduce(%v1698 init: %v1682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1700 = stablehlo.broadcast_in_dim %v1699, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1701 = stablehlo.multiply %v1695, %v1698 : tensor<32x256x14x14xf32>
    %v1702 = stablehlo.reduce(%v1701 init: %v1682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1703 = stablehlo.broadcast_in_dim %v1702, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1704 = stablehlo.multiply %v1698, %v1683 : tensor<32x256x14x14xf32>
    %v1705 = stablehlo.subtract %v1704, %v1700 : tensor<32x256x14x14xf32>
    %v1706 = stablehlo.multiply %v1695, %v1703 : tensor<32x256x14x14xf32>
    %v1707 = stablehlo.subtract %v1705, %v1706 : tensor<32x256x14x14xf32>
    %v1708 = stablehlo.divide %v1694, %v1683 : tensor<32x256x14x14xf32>
    %v1709 = stablehlo.multiply %v1708, %v1707 : tensor<32x256x14x14xf32>
    %v1710 = stablehlo.reshape %v1709 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1711 = stablehlo.reshape %v1710 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1712 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1713 = stablehlo.transpose %v1712, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1714 = stablehlo.convolution(%v1711, %v1713)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1715 = stablehlo.reshape %v1714 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1716 = stablehlo.add %v1715, %v1642 : tensor<32x50176xf32>
    %v1717 = stablehlo.reshape %v685 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1718 = stablehlo.reshape %v1710 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1719 = stablehlo.transpose %v1717, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1720 = stablehlo.transpose %v1718, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1721 = stablehlo.convolution(%v1719, %v1720)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1722 = stablehlo.transpose %v1721, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1723 = stablehlo.reshape %v1710 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1725 = stablehlo.reduce(%v1723 init: %v1724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1726 = stablehlo.reshape %v690 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1727 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1728 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1729 = stablehlo.reduce(%v1726 init: %v1727) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1730 = stablehlo.broadcast_in_dim %v1729, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1731 = stablehlo.divide %v1730, %v1728 : tensor<32x256x14x14xf32>
    %v1732 = stablehlo.subtract %v1726, %v1731 : tensor<32x256x14x14xf32>
    %v1733 = stablehlo.multiply %v1732, %v1732 : tensor<32x256x14x14xf32>
    %v1734 = stablehlo.reduce(%v1733 init: %v1727) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1735 = stablehlo.broadcast_in_dim %v1734, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1736 = stablehlo.divide %v1735, %v1728 : tensor<32x256x14x14xf32>
    %v1737 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1738 = stablehlo.add %v1736, %v1737 : tensor<32x256x14x14xf32>
    %v1739 = stablehlo.rsqrt %v1738 : tensor<32x256x14x14xf32>
    %v1740 = stablehlo.multiply %v1732, %v1739 : tensor<32x256x14x14xf32>
    %v1741 = stablehlo.reshape %v1680 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1742 = stablehlo.multiply %v1741, %v1740 : tensor<32x256x14x14xf32>
    %v1743 = stablehlo.reduce(%v1742 init: %v1727) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1744 = stablehlo.reshape %v1680 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1745 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1746 = stablehlo.reduce(%v1744 init: %v1745) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1747 = stablehlo.reshape %v712 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1748 = stablehlo.reshape %v1672 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1749 = stablehlo.transpose %v1747, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1750 = stablehlo.transpose %v1748, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1751 = stablehlo.convolution(%v1749, %v1750)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1752 = stablehlo.transpose %v1751, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1753 = stablehlo.reshape %v1672 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1754 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1755 = stablehlo.reduce(%v1753 init: %v1754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1756 = stablehlo.reshape %v717 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1758 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1759 = stablehlo.reduce(%v1756 init: %v1757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1760 = stablehlo.broadcast_in_dim %v1759, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1761 = stablehlo.divide %v1760, %v1758 : tensor<32x256x14x14xf32>
    %v1762 = stablehlo.subtract %v1756, %v1761 : tensor<32x256x14x14xf32>
    %v1763 = stablehlo.multiply %v1762, %v1762 : tensor<32x256x14x14xf32>
    %v1764 = stablehlo.reduce(%v1763 init: %v1757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1765 = stablehlo.broadcast_in_dim %v1764, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1766 = stablehlo.divide %v1765, %v1758 : tensor<32x256x14x14xf32>
    %v1767 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1768 = stablehlo.add %v1766, %v1767 : tensor<32x256x14x14xf32>
    %v1769 = stablehlo.rsqrt %v1768 : tensor<32x256x14x14xf32>
    %v1770 = stablehlo.multiply %v1762, %v1769 : tensor<32x256x14x14xf32>
    %v1771 = stablehlo.reshape %v1642 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1772 = stablehlo.multiply %v1771, %v1770 : tensor<32x256x14x14xf32>
    %v1773 = stablehlo.reduce(%v1772 init: %v1757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1774 = stablehlo.reshape %v1642 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1775 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1776 = stablehlo.reduce(%v1774 init: %v1775) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1777 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1778 = stablehlo.compare GT, %v683, %v1777 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1779 = stablehlo.select %v1778, %v1716, %v1777 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1780 = stablehlo.reshape %v662 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1782 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1783 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1784 = stablehlo.reduce(%v1780 init: %v1781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1785 = stablehlo.broadcast_in_dim %v1784, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1786 = stablehlo.divide %v1785, %v1782 : tensor<32x256x14x14xf32>
    %v1787 = stablehlo.subtract %v1780, %v1786 : tensor<32x256x14x14xf32>
    %v1788 = stablehlo.multiply %v1787, %v1787 : tensor<32x256x14x14xf32>
    %v1789 = stablehlo.reduce(%v1788 init: %v1781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1790 = stablehlo.broadcast_in_dim %v1789, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1791 = stablehlo.divide %v1790, %v1782 : tensor<32x256x14x14xf32>
    %v1792 = stablehlo.add %v1791, %v1783 : tensor<32x256x14x14xf32>
    %v1793 = stablehlo.rsqrt %v1792 : tensor<32x256x14x14xf32>
    %v1794 = stablehlo.multiply %v1787, %v1793 : tensor<32x256x14x14xf32>
    %v1795 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1796 = stablehlo.reshape %v1779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1797 = stablehlo.multiply %v1795, %v1796 : tensor<32x256x14x14xf32>
    %v1798 = stablehlo.reduce(%v1797 init: %v1781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1799 = stablehlo.broadcast_in_dim %v1798, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1800 = stablehlo.multiply %v1794, %v1797 : tensor<32x256x14x14xf32>
    %v1801 = stablehlo.reduce(%v1800 init: %v1781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1802 = stablehlo.broadcast_in_dim %v1801, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1803 = stablehlo.multiply %v1797, %v1782 : tensor<32x256x14x14xf32>
    %v1804 = stablehlo.subtract %v1803, %v1799 : tensor<32x256x14x14xf32>
    %v1805 = stablehlo.multiply %v1794, %v1802 : tensor<32x256x14x14xf32>
    %v1806 = stablehlo.subtract %v1804, %v1805 : tensor<32x256x14x14xf32>
    %v1807 = stablehlo.divide %v1793, %v1782 : tensor<32x256x14x14xf32>
    %v1808 = stablehlo.multiply %v1807, %v1806 : tensor<32x256x14x14xf32>
    %v1809 = stablehlo.reshape %v1808 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1810 = stablehlo.reshape %v1809 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1811 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1812 = stablehlo.transpose %v1811, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1813 = stablehlo.convolution(%v1810, %v1812)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1814 = stablehlo.reshape %v1813 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1815 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1816 = stablehlo.compare GT, %v655, %v1815 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1817 = stablehlo.select %v1816, %v1814, %v1815 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1818 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1819 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1820 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1821 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1822 = stablehlo.reduce(%v1818 init: %v1819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1823 = stablehlo.broadcast_in_dim %v1822, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1824 = stablehlo.divide %v1823, %v1820 : tensor<32x256x14x14xf32>
    %v1825 = stablehlo.subtract %v1818, %v1824 : tensor<32x256x14x14xf32>
    %v1826 = stablehlo.multiply %v1825, %v1825 : tensor<32x256x14x14xf32>
    %v1827 = stablehlo.reduce(%v1826 init: %v1819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1828 = stablehlo.broadcast_in_dim %v1827, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1829 = stablehlo.divide %v1828, %v1820 : tensor<32x256x14x14xf32>
    %v1830 = stablehlo.add %v1829, %v1821 : tensor<32x256x14x14xf32>
    %v1831 = stablehlo.rsqrt %v1830 : tensor<32x256x14x14xf32>
    %v1832 = stablehlo.multiply %v1825, %v1831 : tensor<32x256x14x14xf32>
    %v1833 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1834 = stablehlo.reshape %v1817 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1835 = stablehlo.multiply %v1833, %v1834 : tensor<32x256x14x14xf32>
    %v1836 = stablehlo.reduce(%v1835 init: %v1819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1837 = stablehlo.broadcast_in_dim %v1836, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1838 = stablehlo.multiply %v1832, %v1835 : tensor<32x256x14x14xf32>
    %v1839 = stablehlo.reduce(%v1838 init: %v1819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1840 = stablehlo.broadcast_in_dim %v1839, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1841 = stablehlo.multiply %v1835, %v1820 : tensor<32x256x14x14xf32>
    %v1842 = stablehlo.subtract %v1841, %v1837 : tensor<32x256x14x14xf32>
    %v1843 = stablehlo.multiply %v1832, %v1840 : tensor<32x256x14x14xf32>
    %v1844 = stablehlo.subtract %v1842, %v1843 : tensor<32x256x14x14xf32>
    %v1845 = stablehlo.divide %v1831, %v1820 : tensor<32x256x14x14xf32>
    %v1846 = stablehlo.multiply %v1845, %v1844 : tensor<32x256x14x14xf32>
    %v1847 = stablehlo.reshape %v1846 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1848 = stablehlo.reshape %v1847 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1849 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1850 = stablehlo.transpose %v1849, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1851 = stablehlo.convolution(%v1848, %v1850)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1852 = stablehlo.reshape %v1851 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1853 = stablehlo.add %v1852, %v1779 : tensor<32x50176xf32>
    %v1854 = stablehlo.reshape %v630 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1855 = stablehlo.reshape %v1847 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1856 = stablehlo.transpose %v1854, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1857 = stablehlo.transpose %v1855, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1858 = stablehlo.convolution(%v1856, %v1857)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1859 = stablehlo.transpose %v1858, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1860 = stablehlo.reshape %v1847 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1862 = stablehlo.reduce(%v1860 init: %v1861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1863 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1865 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1866 = stablehlo.reduce(%v1863 init: %v1864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1867 = stablehlo.broadcast_in_dim %v1866, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1868 = stablehlo.divide %v1867, %v1865 : tensor<32x256x14x14xf32>
    %v1869 = stablehlo.subtract %v1863, %v1868 : tensor<32x256x14x14xf32>
    %v1870 = stablehlo.multiply %v1869, %v1869 : tensor<32x256x14x14xf32>
    %v1871 = stablehlo.reduce(%v1870 init: %v1864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1872 = stablehlo.broadcast_in_dim %v1871, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1873 = stablehlo.divide %v1872, %v1865 : tensor<32x256x14x14xf32>
    %v1874 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1875 = stablehlo.add %v1873, %v1874 : tensor<32x256x14x14xf32>
    %v1876 = stablehlo.rsqrt %v1875 : tensor<32x256x14x14xf32>
    %v1877 = stablehlo.multiply %v1869, %v1876 : tensor<32x256x14x14xf32>
    %v1878 = stablehlo.reshape %v1817 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1879 = stablehlo.multiply %v1878, %v1877 : tensor<32x256x14x14xf32>
    %v1880 = stablehlo.reduce(%v1879 init: %v1864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1881 = stablehlo.reshape %v1817 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1883 = stablehlo.reduce(%v1881 init: %v1882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1884 = stablehlo.reshape %v657 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1885 = stablehlo.reshape %v1809 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1886 = stablehlo.transpose %v1884, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1887 = stablehlo.transpose %v1885, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1888 = stablehlo.convolution(%v1886, %v1887)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1889 = stablehlo.transpose %v1888, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1890 = stablehlo.reshape %v1809 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1892 = stablehlo.reduce(%v1890 init: %v1891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1893 = stablehlo.reshape %v662 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1895 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1896 = stablehlo.reduce(%v1893 init: %v1894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1897 = stablehlo.broadcast_in_dim %v1896, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1898 = stablehlo.divide %v1897, %v1895 : tensor<32x256x14x14xf32>
    %v1899 = stablehlo.subtract %v1893, %v1898 : tensor<32x256x14x14xf32>
    %v1900 = stablehlo.multiply %v1899, %v1899 : tensor<32x256x14x14xf32>
    %v1901 = stablehlo.reduce(%v1900 init: %v1894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1902 = stablehlo.broadcast_in_dim %v1901, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1903 = stablehlo.divide %v1902, %v1895 : tensor<32x256x14x14xf32>
    %v1904 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1905 = stablehlo.add %v1903, %v1904 : tensor<32x256x14x14xf32>
    %v1906 = stablehlo.rsqrt %v1905 : tensor<32x256x14x14xf32>
    %v1907 = stablehlo.multiply %v1899, %v1906 : tensor<32x256x14x14xf32>
    %v1908 = stablehlo.reshape %v1779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1909 = stablehlo.multiply %v1908, %v1907 : tensor<32x256x14x14xf32>
    %v1910 = stablehlo.reduce(%v1909 init: %v1894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1911 = stablehlo.reshape %v1779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1912 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1913 = stablehlo.reduce(%v1911 init: %v1912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1914 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1915 = stablehlo.compare GT, %v628, %v1914 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1916 = stablehlo.select %v1915, %v1853, %v1914 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1917 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1919 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1920 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1921 = stablehlo.reduce(%v1917 init: %v1918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1922 = stablehlo.broadcast_in_dim %v1921, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1923 = stablehlo.divide %v1922, %v1919 : tensor<32x256x14x14xf32>
    %v1924 = stablehlo.subtract %v1917, %v1923 : tensor<32x256x14x14xf32>
    %v1925 = stablehlo.multiply %v1924, %v1924 : tensor<32x256x14x14xf32>
    %v1926 = stablehlo.reduce(%v1925 init: %v1918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1927 = stablehlo.broadcast_in_dim %v1926, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1928 = stablehlo.divide %v1927, %v1919 : tensor<32x256x14x14xf32>
    %v1929 = stablehlo.add %v1928, %v1920 : tensor<32x256x14x14xf32>
    %v1930 = stablehlo.rsqrt %v1929 : tensor<32x256x14x14xf32>
    %v1931 = stablehlo.multiply %v1924, %v1930 : tensor<32x256x14x14xf32>
    %v1932 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1933 = stablehlo.reshape %v1916 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1934 = stablehlo.multiply %v1932, %v1933 : tensor<32x256x14x14xf32>
    %v1935 = stablehlo.reduce(%v1934 init: %v1918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1936 = stablehlo.broadcast_in_dim %v1935, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1937 = stablehlo.multiply %v1931, %v1934 : tensor<32x256x14x14xf32>
    %v1938 = stablehlo.reduce(%v1937 init: %v1918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1939 = stablehlo.broadcast_in_dim %v1938, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1940 = stablehlo.multiply %v1934, %v1919 : tensor<32x256x14x14xf32>
    %v1941 = stablehlo.subtract %v1940, %v1936 : tensor<32x256x14x14xf32>
    %v1942 = stablehlo.multiply %v1931, %v1939 : tensor<32x256x14x14xf32>
    %v1943 = stablehlo.subtract %v1941, %v1942 : tensor<32x256x14x14xf32>
    %v1944 = stablehlo.divide %v1930, %v1919 : tensor<32x256x14x14xf32>
    %v1945 = stablehlo.multiply %v1944, %v1943 : tensor<32x256x14x14xf32>
    %v1946 = stablehlo.reshape %v1945 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1947 = stablehlo.reshape %v1946 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1948 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1949 = stablehlo.transpose %v1948, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1950 = stablehlo.convolution(%v1947, %v1949)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1951 = stablehlo.reshape %v1950 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1952 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1953 = stablehlo.compare GT, %v600, %v1952 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1954 = stablehlo.select %v1953, %v1951, %v1952 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1955 = stablehlo.reshape %v580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1956 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1957 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1958 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1959 = stablehlo.reduce(%v1955 init: %v1956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1960 = stablehlo.broadcast_in_dim %v1959, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1961 = stablehlo.divide %v1960, %v1957 : tensor<32x256x14x14xf32>
    %v1962 = stablehlo.subtract %v1955, %v1961 : tensor<32x256x14x14xf32>
    %v1963 = stablehlo.multiply %v1962, %v1962 : tensor<32x256x14x14xf32>
    %v1964 = stablehlo.reduce(%v1963 init: %v1956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1965 = stablehlo.broadcast_in_dim %v1964, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1966 = stablehlo.divide %v1965, %v1957 : tensor<32x256x14x14xf32>
    %v1967 = stablehlo.add %v1966, %v1958 : tensor<32x256x14x14xf32>
    %v1968 = stablehlo.rsqrt %v1967 : tensor<32x256x14x14xf32>
    %v1969 = stablehlo.multiply %v1962, %v1968 : tensor<32x256x14x14xf32>
    %v1970 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1971 = stablehlo.reshape %v1954 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1972 = stablehlo.multiply %v1970, %v1971 : tensor<32x256x14x14xf32>
    %v1973 = stablehlo.reduce(%v1972 init: %v1956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1974 = stablehlo.broadcast_in_dim %v1973, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1975 = stablehlo.multiply %v1969, %v1972 : tensor<32x256x14x14xf32>
    %v1976 = stablehlo.reduce(%v1975 init: %v1956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1977 = stablehlo.broadcast_in_dim %v1976, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1978 = stablehlo.multiply %v1972, %v1957 : tensor<32x256x14x14xf32>
    %v1979 = stablehlo.subtract %v1978, %v1974 : tensor<32x256x14x14xf32>
    %v1980 = stablehlo.multiply %v1969, %v1977 : tensor<32x256x14x14xf32>
    %v1981 = stablehlo.subtract %v1979, %v1980 : tensor<32x256x14x14xf32>
    %v1982 = stablehlo.divide %v1968, %v1957 : tensor<32x256x14x14xf32>
    %v1983 = stablehlo.multiply %v1982, %v1981 : tensor<32x256x14x14xf32>
    %v1984 = stablehlo.reshape %v1983 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1985 = stablehlo.reshape %v1984 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1986 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1987 = stablehlo.transpose %v1986, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1988 = stablehlo.convolution(%v1985, %v1987)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1989 = stablehlo.reshape %v1988 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1990 = stablehlo.add %v1989, %v1916 : tensor<32x50176xf32>
    %v1991 = stablehlo.reshape %v575 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1992 = stablehlo.reshape %v1984 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1993 = stablehlo.transpose %v1991, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1994 = stablehlo.transpose %v1992, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1995 = stablehlo.convolution(%v1993, %v1994)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1996 = stablehlo.transpose %v1995, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1997 = stablehlo.reshape %v1984 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1999 = stablehlo.reduce(%v1997 init: %v1998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2000 = stablehlo.reshape %v580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2001 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2002 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2003 = stablehlo.reduce(%v2000 init: %v2001) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2004 = stablehlo.broadcast_in_dim %v2003, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2005 = stablehlo.divide %v2004, %v2002 : tensor<32x256x14x14xf32>
    %v2006 = stablehlo.subtract %v2000, %v2005 : tensor<32x256x14x14xf32>
    %v2007 = stablehlo.multiply %v2006, %v2006 : tensor<32x256x14x14xf32>
    %v2008 = stablehlo.reduce(%v2007 init: %v2001) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2009 = stablehlo.broadcast_in_dim %v2008, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2010 = stablehlo.divide %v2009, %v2002 : tensor<32x256x14x14xf32>
    %v2011 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2012 = stablehlo.add %v2010, %v2011 : tensor<32x256x14x14xf32>
    %v2013 = stablehlo.rsqrt %v2012 : tensor<32x256x14x14xf32>
    %v2014 = stablehlo.multiply %v2006, %v2013 : tensor<32x256x14x14xf32>
    %v2015 = stablehlo.reshape %v1954 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2016 = stablehlo.multiply %v2015, %v2014 : tensor<32x256x14x14xf32>
    %v2017 = stablehlo.reduce(%v2016 init: %v2001) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2018 = stablehlo.reshape %v1954 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2019 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2020 = stablehlo.reduce(%v2018 init: %v2019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2021 = stablehlo.reshape %v602 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2022 = stablehlo.reshape %v1946 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2023 = stablehlo.transpose %v2021, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2024 = stablehlo.transpose %v2022, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2025 = stablehlo.convolution(%v2023, %v2024)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2026 = stablehlo.transpose %v2025, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2027 = stablehlo.reshape %v1946 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2028 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2029 = stablehlo.reduce(%v2027 init: %v2028) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2030 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2031 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2032 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2033 = stablehlo.reduce(%v2030 init: %v2031) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2034 = stablehlo.broadcast_in_dim %v2033, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2035 = stablehlo.divide %v2034, %v2032 : tensor<32x256x14x14xf32>
    %v2036 = stablehlo.subtract %v2030, %v2035 : tensor<32x256x14x14xf32>
    %v2037 = stablehlo.multiply %v2036, %v2036 : tensor<32x256x14x14xf32>
    %v2038 = stablehlo.reduce(%v2037 init: %v2031) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2039 = stablehlo.broadcast_in_dim %v2038, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2040 = stablehlo.divide %v2039, %v2032 : tensor<32x256x14x14xf32>
    %v2041 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2042 = stablehlo.add %v2040, %v2041 : tensor<32x256x14x14xf32>
    %v2043 = stablehlo.rsqrt %v2042 : tensor<32x256x14x14xf32>
    %v2044 = stablehlo.multiply %v2036, %v2043 : tensor<32x256x14x14xf32>
    %v2045 = stablehlo.reshape %v1916 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2046 = stablehlo.multiply %v2045, %v2044 : tensor<32x256x14x14xf32>
    %v2047 = stablehlo.reduce(%v2046 init: %v2031) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2048 = stablehlo.reshape %v1916 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2049 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2050 = stablehlo.reduce(%v2048 init: %v2049) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2051 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2052 = stablehlo.compare GT, %v573, %v2051 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2053 = stablehlo.select %v2052, %v1990, %v2051 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2054 = stablehlo.reshape %v552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2056 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2057 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2058 = stablehlo.reduce(%v2054 init: %v2055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2059 = stablehlo.broadcast_in_dim %v2058, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2060 = stablehlo.divide %v2059, %v2056 : tensor<32x256x14x14xf32>
    %v2061 = stablehlo.subtract %v2054, %v2060 : tensor<32x256x14x14xf32>
    %v2062 = stablehlo.multiply %v2061, %v2061 : tensor<32x256x14x14xf32>
    %v2063 = stablehlo.reduce(%v2062 init: %v2055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2064 = stablehlo.broadcast_in_dim %v2063, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2065 = stablehlo.divide %v2064, %v2056 : tensor<32x256x14x14xf32>
    %v2066 = stablehlo.add %v2065, %v2057 : tensor<32x256x14x14xf32>
    %v2067 = stablehlo.rsqrt %v2066 : tensor<32x256x14x14xf32>
    %v2068 = stablehlo.multiply %v2061, %v2067 : tensor<32x256x14x14xf32>
    %v2069 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2070 = stablehlo.reshape %v2053 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2071 = stablehlo.multiply %v2069, %v2070 : tensor<32x256x14x14xf32>
    %v2072 = stablehlo.reduce(%v2071 init: %v2055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2073 = stablehlo.broadcast_in_dim %v2072, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2074 = stablehlo.multiply %v2068, %v2071 : tensor<32x256x14x14xf32>
    %v2075 = stablehlo.reduce(%v2074 init: %v2055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2076 = stablehlo.broadcast_in_dim %v2075, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2077 = stablehlo.multiply %v2071, %v2056 : tensor<32x256x14x14xf32>
    %v2078 = stablehlo.subtract %v2077, %v2073 : tensor<32x256x14x14xf32>
    %v2079 = stablehlo.multiply %v2068, %v2076 : tensor<32x256x14x14xf32>
    %v2080 = stablehlo.subtract %v2078, %v2079 : tensor<32x256x14x14xf32>
    %v2081 = stablehlo.divide %v2067, %v2056 : tensor<32x256x14x14xf32>
    %v2082 = stablehlo.multiply %v2081, %v2080 : tensor<32x256x14x14xf32>
    %v2083 = stablehlo.reshape %v2082 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2084 = stablehlo.reshape %v2083 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2085 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2086 = stablehlo.transpose %v2085, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2087 = stablehlo.convolution(%v2084, %v2086)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2088 = stablehlo.reshape %v2087 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2089 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2090 = stablehlo.compare GT, %v545, %v2089 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2091 = stablehlo.select %v2090, %v2088, %v2089 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2092 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2094 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2095 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2096 = stablehlo.reduce(%v2092 init: %v2093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2097 = stablehlo.broadcast_in_dim %v2096, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2098 = stablehlo.divide %v2097, %v2094 : tensor<32x256x14x14xf32>
    %v2099 = stablehlo.subtract %v2092, %v2098 : tensor<32x256x14x14xf32>
    %v2100 = stablehlo.multiply %v2099, %v2099 : tensor<32x256x14x14xf32>
    %v2101 = stablehlo.reduce(%v2100 init: %v2093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2102 = stablehlo.broadcast_in_dim %v2101, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2103 = stablehlo.divide %v2102, %v2094 : tensor<32x256x14x14xf32>
    %v2104 = stablehlo.add %v2103, %v2095 : tensor<32x256x14x14xf32>
    %v2105 = stablehlo.rsqrt %v2104 : tensor<32x256x14x14xf32>
    %v2106 = stablehlo.multiply %v2099, %v2105 : tensor<32x256x14x14xf32>
    %v2107 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2108 = stablehlo.reshape %v2091 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2109 = stablehlo.multiply %v2107, %v2108 : tensor<32x256x14x14xf32>
    %v2110 = stablehlo.reduce(%v2109 init: %v2093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2111 = stablehlo.broadcast_in_dim %v2110, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2112 = stablehlo.multiply %v2106, %v2109 : tensor<32x256x14x14xf32>
    %v2113 = stablehlo.reduce(%v2112 init: %v2093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2114 = stablehlo.broadcast_in_dim %v2113, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2115 = stablehlo.multiply %v2109, %v2094 : tensor<32x256x14x14xf32>
    %v2116 = stablehlo.subtract %v2115, %v2111 : tensor<32x256x14x14xf32>
    %v2117 = stablehlo.multiply %v2106, %v2114 : tensor<32x256x14x14xf32>
    %v2118 = stablehlo.subtract %v2116, %v2117 : tensor<32x256x14x14xf32>
    %v2119 = stablehlo.divide %v2105, %v2094 : tensor<32x256x14x14xf32>
    %v2120 = stablehlo.multiply %v2119, %v2118 : tensor<32x256x14x14xf32>
    %v2121 = stablehlo.reshape %v2120 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2122 = stablehlo.reshape %v2121 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2123 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2124 = stablehlo.transpose %v2123, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2125 = stablehlo.convolution(%v2122, %v2124)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2126 = stablehlo.reshape %v2125 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2127 = stablehlo.add %v2126, %v2053 : tensor<32x50176xf32>
    %v2128 = stablehlo.reshape %v520 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2129 = stablehlo.reshape %v2121 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2130 = stablehlo.transpose %v2128, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2131 = stablehlo.transpose %v2129, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2132 = stablehlo.convolution(%v2130, %v2131)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2133 = stablehlo.transpose %v2132, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2134 = stablehlo.reshape %v2121 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2135 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2136 = stablehlo.reduce(%v2134 init: %v2135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2137 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2139 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2140 = stablehlo.reduce(%v2137 init: %v2138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2141 = stablehlo.broadcast_in_dim %v2140, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2142 = stablehlo.divide %v2141, %v2139 : tensor<32x256x14x14xf32>
    %v2143 = stablehlo.subtract %v2137, %v2142 : tensor<32x256x14x14xf32>
    %v2144 = stablehlo.multiply %v2143, %v2143 : tensor<32x256x14x14xf32>
    %v2145 = stablehlo.reduce(%v2144 init: %v2138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2146 = stablehlo.broadcast_in_dim %v2145, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2147 = stablehlo.divide %v2146, %v2139 : tensor<32x256x14x14xf32>
    %v2148 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2149 = stablehlo.add %v2147, %v2148 : tensor<32x256x14x14xf32>
    %v2150 = stablehlo.rsqrt %v2149 : tensor<32x256x14x14xf32>
    %v2151 = stablehlo.multiply %v2143, %v2150 : tensor<32x256x14x14xf32>
    %v2152 = stablehlo.reshape %v2091 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2153 = stablehlo.multiply %v2152, %v2151 : tensor<32x256x14x14xf32>
    %v2154 = stablehlo.reduce(%v2153 init: %v2138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2155 = stablehlo.reshape %v2091 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2157 = stablehlo.reduce(%v2155 init: %v2156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2158 = stablehlo.reshape %v547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2159 = stablehlo.reshape %v2083 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2160 = stablehlo.transpose %v2158, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2161 = stablehlo.transpose %v2159, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2162 = stablehlo.convolution(%v2160, %v2161)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2163 = stablehlo.transpose %v2162, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2164 = stablehlo.reshape %v2083 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2166 = stablehlo.reduce(%v2164 init: %v2165) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2167 = stablehlo.reshape %v552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2168 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2169 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2170 = stablehlo.reduce(%v2167 init: %v2168) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2171 = stablehlo.broadcast_in_dim %v2170, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2172 = stablehlo.divide %v2171, %v2169 : tensor<32x256x14x14xf32>
    %v2173 = stablehlo.subtract %v2167, %v2172 : tensor<32x256x14x14xf32>
    %v2174 = stablehlo.multiply %v2173, %v2173 : tensor<32x256x14x14xf32>
    %v2175 = stablehlo.reduce(%v2174 init: %v2168) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2176 = stablehlo.broadcast_in_dim %v2175, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2177 = stablehlo.divide %v2176, %v2169 : tensor<32x256x14x14xf32>
    %v2178 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2179 = stablehlo.add %v2177, %v2178 : tensor<32x256x14x14xf32>
    %v2180 = stablehlo.rsqrt %v2179 : tensor<32x256x14x14xf32>
    %v2181 = stablehlo.multiply %v2173, %v2180 : tensor<32x256x14x14xf32>
    %v2182 = stablehlo.reshape %v2053 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2183 = stablehlo.multiply %v2182, %v2181 : tensor<32x256x14x14xf32>
    %v2184 = stablehlo.reduce(%v2183 init: %v2168) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2185 = stablehlo.reshape %v2053 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2187 = stablehlo.reduce(%v2185 init: %v2186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2188 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2189 = stablehlo.compare GT, %v518, %v2188 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2190 = stablehlo.select %v2189, %v2127, %v2188 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2191 = stablehlo.reshape %v472 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2193 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2194 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2195 = stablehlo.reduce(%v2191 init: %v2192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2196 = stablehlo.broadcast_in_dim %v2195, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2197 = stablehlo.divide %v2196, %v2193 : tensor<32x256x14x14xf32>
    %v2198 = stablehlo.subtract %v2191, %v2197 : tensor<32x256x14x14xf32>
    %v2199 = stablehlo.multiply %v2198, %v2198 : tensor<32x256x14x14xf32>
    %v2200 = stablehlo.reduce(%v2199 init: %v2192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2201 = stablehlo.broadcast_in_dim %v2200, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2202 = stablehlo.divide %v2201, %v2193 : tensor<32x256x14x14xf32>
    %v2203 = stablehlo.add %v2202, %v2194 : tensor<32x256x14x14xf32>
    %v2204 = stablehlo.rsqrt %v2203 : tensor<32x256x14x14xf32>
    %v2205 = stablehlo.multiply %v2198, %v2204 : tensor<32x256x14x14xf32>
    %v2206 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2207 = stablehlo.reshape %v2190 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2208 = stablehlo.multiply %v2206, %v2207 : tensor<32x256x14x14xf32>
    %v2209 = stablehlo.reduce(%v2208 init: %v2192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2210 = stablehlo.broadcast_in_dim %v2209, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2211 = stablehlo.multiply %v2205, %v2208 : tensor<32x256x14x14xf32>
    %v2212 = stablehlo.reduce(%v2211 init: %v2192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2213 = stablehlo.broadcast_in_dim %v2212, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2214 = stablehlo.multiply %v2208, %v2193 : tensor<32x256x14x14xf32>
    %v2215 = stablehlo.subtract %v2214, %v2210 : tensor<32x256x14x14xf32>
    %v2216 = stablehlo.multiply %v2205, %v2213 : tensor<32x256x14x14xf32>
    %v2217 = stablehlo.subtract %v2215, %v2216 : tensor<32x256x14x14xf32>
    %v2218 = stablehlo.divide %v2204, %v2193 : tensor<32x256x14x14xf32>
    %v2219 = stablehlo.multiply %v2218, %v2217 : tensor<32x256x14x14xf32>
    %v2220 = stablehlo.reshape %v2219 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2221 = stablehlo.reshape %v2220 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2222 = stablehlo.reverse %d3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2223 = stablehlo.transpose %v2222, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2224 = stablehlo.convolution(%v2221, %v2223)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2225 = stablehlo.reshape %v2224 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2226 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2227 = stablehlo.compare GT, %v465, %v2226 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2228 = stablehlo.select %v2227, %v2225, %v2226 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2229 = stablehlo.reshape %v445 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2231 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2232 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2233 = stablehlo.reduce(%v2229 init: %v2230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2234 = stablehlo.broadcast_in_dim %v2233, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2235 = stablehlo.divide %v2234, %v2231 : tensor<32x256x14x14xf32>
    %v2236 = stablehlo.subtract %v2229, %v2235 : tensor<32x256x14x14xf32>
    %v2237 = stablehlo.multiply %v2236, %v2236 : tensor<32x256x14x14xf32>
    %v2238 = stablehlo.reduce(%v2237 init: %v2230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2239 = stablehlo.broadcast_in_dim %v2238, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2240 = stablehlo.divide %v2239, %v2231 : tensor<32x256x14x14xf32>
    %v2241 = stablehlo.add %v2240, %v2232 : tensor<32x256x14x14xf32>
    %v2242 = stablehlo.rsqrt %v2241 : tensor<32x256x14x14xf32>
    %v2243 = stablehlo.multiply %v2236, %v2242 : tensor<32x256x14x14xf32>
    %v2244 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2245 = stablehlo.reshape %v2228 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2246 = stablehlo.multiply %v2244, %v2245 : tensor<32x256x14x14xf32>
    %v2247 = stablehlo.reduce(%v2246 init: %v2230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2248 = stablehlo.broadcast_in_dim %v2247, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2249 = stablehlo.multiply %v2243, %v2246 : tensor<32x256x14x14xf32>
    %v2250 = stablehlo.reduce(%v2249 init: %v2230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2251 = stablehlo.broadcast_in_dim %v2250, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2252 = stablehlo.multiply %v2246, %v2231 : tensor<32x256x14x14xf32>
    %v2253 = stablehlo.subtract %v2252, %v2248 : tensor<32x256x14x14xf32>
    %v2254 = stablehlo.multiply %v2243, %v2251 : tensor<32x256x14x14xf32>
    %v2255 = stablehlo.subtract %v2253, %v2254 : tensor<32x256x14x14xf32>
    %v2256 = stablehlo.divide %v2242, %v2231 : tensor<32x256x14x14xf32>
    %v2257 = stablehlo.multiply %v2256, %v2255 : tensor<32x256x14x14xf32>
    %v2258 = stablehlo.reshape %v2257 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2261 = stablehlo.pad %v2259, %v2260, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2262 = stablehlo.reverse %d3W1, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v2263 = stablehlo.transpose %v2262, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2264 = stablehlo.convolution(%v2261, %v2263)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2265 = stablehlo.reshape %v2264 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2266 = stablehlo.reshape %v497 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2267 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2268 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2269 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2270 = stablehlo.reduce(%v2266 init: %v2267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2271 = stablehlo.broadcast_in_dim %v2270, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2272 = stablehlo.divide %v2271, %v2268 : tensor<32x256x14x14xf32>
    %v2273 = stablehlo.subtract %v2266, %v2272 : tensor<32x256x14x14xf32>
    %v2274 = stablehlo.multiply %v2273, %v2273 : tensor<32x256x14x14xf32>
    %v2275 = stablehlo.reduce(%v2274 init: %v2267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2276 = stablehlo.broadcast_in_dim %v2275, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2277 = stablehlo.divide %v2276, %v2268 : tensor<32x256x14x14xf32>
    %v2278 = stablehlo.add %v2277, %v2269 : tensor<32x256x14x14xf32>
    %v2279 = stablehlo.rsqrt %v2278 : tensor<32x256x14x14xf32>
    %v2280 = stablehlo.multiply %v2273, %v2279 : tensor<32x256x14x14xf32>
    %v2281 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2282 = stablehlo.reshape %v2190 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2283 = stablehlo.multiply %v2281, %v2282 : tensor<32x256x14x14xf32>
    %v2284 = stablehlo.reduce(%v2283 init: %v2267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2285 = stablehlo.broadcast_in_dim %v2284, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2286 = stablehlo.multiply %v2280, %v2283 : tensor<32x256x14x14xf32>
    %v2287 = stablehlo.reduce(%v2286 init: %v2267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2288 = stablehlo.broadcast_in_dim %v2287, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2289 = stablehlo.multiply %v2283, %v2268 : tensor<32x256x14x14xf32>
    %v2290 = stablehlo.subtract %v2289, %v2285 : tensor<32x256x14x14xf32>
    %v2291 = stablehlo.multiply %v2280, %v2288 : tensor<32x256x14x14xf32>
    %v2292 = stablehlo.subtract %v2290, %v2291 : tensor<32x256x14x14xf32>
    %v2293 = stablehlo.divide %v2279, %v2268 : tensor<32x256x14x14xf32>
    %v2294 = stablehlo.multiply %v2293, %v2292 : tensor<32x256x14x14xf32>
    %v2295 = stablehlo.reshape %v2294 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2296 = stablehlo.reshape %v2295 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2298 = stablehlo.pad %v2296, %v2297, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2299 = stablehlo.reverse %d3Wp, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v2300 = stablehlo.transpose %v2299, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2301 = stablehlo.convolution(%v2298, %v2300)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2302 = stablehlo.reshape %v2301 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2303 = stablehlo.add %v2265, %v2302 : tensor<32x100352xf32>
    %v2304 = stablehlo.reshape %v440 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2305 = stablehlo.reshape %v2258 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2307 = stablehlo.pad %v2305, %v2306, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2308 = stablehlo.transpose %v2304, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2309 = stablehlo.transpose %v2307, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2310 = stablehlo.convolution(%v2308, %v2309)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x3x3xf32>
    %v2311 = stablehlo.transpose %v2310, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2312 = stablehlo.reshape %v2258 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2313 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2314 = stablehlo.reduce(%v2312 init: %v2313) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2315 = stablehlo.reshape %v445 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2317 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2318 = stablehlo.reduce(%v2315 init: %v2316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2319 = stablehlo.broadcast_in_dim %v2318, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2320 = stablehlo.divide %v2319, %v2317 : tensor<32x256x14x14xf32>
    %v2321 = stablehlo.subtract %v2315, %v2320 : tensor<32x256x14x14xf32>
    %v2322 = stablehlo.multiply %v2321, %v2321 : tensor<32x256x14x14xf32>
    %v2323 = stablehlo.reduce(%v2322 init: %v2316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2324 = stablehlo.broadcast_in_dim %v2323, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2325 = stablehlo.divide %v2324, %v2317 : tensor<32x256x14x14xf32>
    %v2326 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2327 = stablehlo.add %v2325, %v2326 : tensor<32x256x14x14xf32>
    %v2328 = stablehlo.rsqrt %v2327 : tensor<32x256x14x14xf32>
    %v2329 = stablehlo.multiply %v2321, %v2328 : tensor<32x256x14x14xf32>
    %v2330 = stablehlo.reshape %v2228 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2331 = stablehlo.multiply %v2330, %v2329 : tensor<32x256x14x14xf32>
    %v2332 = stablehlo.reduce(%v2331 init: %v2316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2333 = stablehlo.reshape %v2228 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2335 = stablehlo.reduce(%v2333 init: %v2334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2336 = stablehlo.reshape %v467 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2337 = stablehlo.reshape %v2220 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2338 = stablehlo.transpose %v2336, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2339 = stablehlo.transpose %v2337, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2340 = stablehlo.convolution(%v2338, %v2339)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2341 = stablehlo.transpose %v2340, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2342 = stablehlo.reshape %v2220 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2344 = stablehlo.reduce(%v2342 init: %v2343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2345 = stablehlo.reshape %v472 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2346 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2347 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2348 = stablehlo.reduce(%v2345 init: %v2346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2349 = stablehlo.broadcast_in_dim %v2348, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2350 = stablehlo.divide %v2349, %v2347 : tensor<32x256x14x14xf32>
    %v2351 = stablehlo.subtract %v2345, %v2350 : tensor<32x256x14x14xf32>
    %v2352 = stablehlo.multiply %v2351, %v2351 : tensor<32x256x14x14xf32>
    %v2353 = stablehlo.reduce(%v2352 init: %v2346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2354 = stablehlo.broadcast_in_dim %v2353, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2355 = stablehlo.divide %v2354, %v2347 : tensor<32x256x14x14xf32>
    %v2356 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2357 = stablehlo.add %v2355, %v2356 : tensor<32x256x14x14xf32>
    %v2358 = stablehlo.rsqrt %v2357 : tensor<32x256x14x14xf32>
    %v2359 = stablehlo.multiply %v2351, %v2358 : tensor<32x256x14x14xf32>
    %v2360 = stablehlo.reshape %v2190 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2361 = stablehlo.multiply %v2360, %v2359 : tensor<32x256x14x14xf32>
    %v2362 = stablehlo.reduce(%v2361 init: %v2346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2363 = stablehlo.reshape %v2190 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2365 = stablehlo.reduce(%v2363 init: %v2364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2366 = stablehlo.reshape %v440 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2367 = stablehlo.reshape %v2295 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2369 = stablehlo.pad %v2367, %v2368, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2370 = stablehlo.transpose %v2366, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2371 = stablehlo.transpose %v2369, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2372 = stablehlo.convolution(%v2370, %v2371)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x3x3xf32>
    %v2373 = stablehlo.transpose %v2372, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2374 = stablehlo.reshape %v2295 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2376 = stablehlo.reduce(%v2374 init: %v2375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2377 = stablehlo.reshape %v497 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2379 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2380 = stablehlo.reduce(%v2377 init: %v2378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2381 = stablehlo.broadcast_in_dim %v2380, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2382 = stablehlo.divide %v2381, %v2379 : tensor<32x256x14x14xf32>
    %v2383 = stablehlo.subtract %v2377, %v2382 : tensor<32x256x14x14xf32>
    %v2384 = stablehlo.multiply %v2383, %v2383 : tensor<32x256x14x14xf32>
    %v2385 = stablehlo.reduce(%v2384 init: %v2378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2386 = stablehlo.broadcast_in_dim %v2385, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2387 = stablehlo.divide %v2386, %v2379 : tensor<32x256x14x14xf32>
    %v2388 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2389 = stablehlo.add %v2387, %v2388 : tensor<32x256x14x14xf32>
    %v2390 = stablehlo.rsqrt %v2389 : tensor<32x256x14x14xf32>
    %v2391 = stablehlo.multiply %v2383, %v2390 : tensor<32x256x14x14xf32>
    %v2392 = stablehlo.reshape %v2190 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2393 = stablehlo.multiply %v2392, %v2391 : tensor<32x256x14x14xf32>
    %v2394 = stablehlo.reduce(%v2393 init: %v2378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2395 = stablehlo.reshape %v2190 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2397 = stablehlo.reduce(%v2395 init: %v2396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2398 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2399 = stablehlo.compare GT, %v438, %v2398 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2400 = stablehlo.select %v2399, %v2303, %v2398 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2401 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2403 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2404 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2405 = stablehlo.reduce(%v2401 init: %v2402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2406 = stablehlo.broadcast_in_dim %v2405, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2407 = stablehlo.divide %v2406, %v2403 : tensor<32x128x28x28xf32>
    %v2408 = stablehlo.subtract %v2401, %v2407 : tensor<32x128x28x28xf32>
    %v2409 = stablehlo.multiply %v2408, %v2408 : tensor<32x128x28x28xf32>
    %v2410 = stablehlo.reduce(%v2409 init: %v2402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2411 = stablehlo.broadcast_in_dim %v2410, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2412 = stablehlo.divide %v2411, %v2403 : tensor<32x128x28x28xf32>
    %v2413 = stablehlo.add %v2412, %v2404 : tensor<32x128x28x28xf32>
    %v2414 = stablehlo.rsqrt %v2413 : tensor<32x128x28x28xf32>
    %v2415 = stablehlo.multiply %v2408, %v2414 : tensor<32x128x28x28xf32>
    %v2416 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2417 = stablehlo.reshape %v2400 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2418 = stablehlo.multiply %v2416, %v2417 : tensor<32x128x28x28xf32>
    %v2419 = stablehlo.reduce(%v2418 init: %v2402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2420 = stablehlo.broadcast_in_dim %v2419, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2421 = stablehlo.multiply %v2415, %v2418 : tensor<32x128x28x28xf32>
    %v2422 = stablehlo.reduce(%v2421 init: %v2402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2423 = stablehlo.broadcast_in_dim %v2422, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2424 = stablehlo.multiply %v2418, %v2403 : tensor<32x128x28x28xf32>
    %v2425 = stablehlo.subtract %v2424, %v2420 : tensor<32x128x28x28xf32>
    %v2426 = stablehlo.multiply %v2415, %v2423 : tensor<32x128x28x28xf32>
    %v2427 = stablehlo.subtract %v2425, %v2426 : tensor<32x128x28x28xf32>
    %v2428 = stablehlo.divide %v2414, %v2403 : tensor<32x128x28x28xf32>
    %v2429 = stablehlo.multiply %v2428, %v2427 : tensor<32x128x28x28xf32>
    %v2430 = stablehlo.reshape %v2429 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2431 = stablehlo.reshape %v2430 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2432 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2433 = stablehlo.transpose %v2432, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2434 = stablehlo.convolution(%v2431, %v2433)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2435 = stablehlo.reshape %v2434 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2436 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2437 = stablehlo.compare GT, %v410, %v2436 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2438 = stablehlo.select %v2437, %v2435, %v2436 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2439 = stablehlo.reshape %v390 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2440 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2441 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2442 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2443 = stablehlo.reduce(%v2439 init: %v2440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2444 = stablehlo.broadcast_in_dim %v2443, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2445 = stablehlo.divide %v2444, %v2441 : tensor<32x128x28x28xf32>
    %v2446 = stablehlo.subtract %v2439, %v2445 : tensor<32x128x28x28xf32>
    %v2447 = stablehlo.multiply %v2446, %v2446 : tensor<32x128x28x28xf32>
    %v2448 = stablehlo.reduce(%v2447 init: %v2440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2449 = stablehlo.broadcast_in_dim %v2448, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2450 = stablehlo.divide %v2449, %v2441 : tensor<32x128x28x28xf32>
    %v2451 = stablehlo.add %v2450, %v2442 : tensor<32x128x28x28xf32>
    %v2452 = stablehlo.rsqrt %v2451 : tensor<32x128x28x28xf32>
    %v2453 = stablehlo.multiply %v2446, %v2452 : tensor<32x128x28x28xf32>
    %v2454 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2455 = stablehlo.reshape %v2438 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2456 = stablehlo.multiply %v2454, %v2455 : tensor<32x128x28x28xf32>
    %v2457 = stablehlo.reduce(%v2456 init: %v2440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2458 = stablehlo.broadcast_in_dim %v2457, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2459 = stablehlo.multiply %v2453, %v2456 : tensor<32x128x28x28xf32>
    %v2460 = stablehlo.reduce(%v2459 init: %v2440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2461 = stablehlo.broadcast_in_dim %v2460, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2462 = stablehlo.multiply %v2456, %v2441 : tensor<32x128x28x28xf32>
    %v2463 = stablehlo.subtract %v2462, %v2458 : tensor<32x128x28x28xf32>
    %v2464 = stablehlo.multiply %v2453, %v2461 : tensor<32x128x28x28xf32>
    %v2465 = stablehlo.subtract %v2463, %v2464 : tensor<32x128x28x28xf32>
    %v2466 = stablehlo.divide %v2452, %v2441 : tensor<32x128x28x28xf32>
    %v2467 = stablehlo.multiply %v2466, %v2465 : tensor<32x128x28x28xf32>
    %v2468 = stablehlo.reshape %v2467 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2469 = stablehlo.reshape %v2468 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2470 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2471 = stablehlo.transpose %v2470, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2472 = stablehlo.convolution(%v2469, %v2471)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2473 = stablehlo.reshape %v2472 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2474 = stablehlo.add %v2473, %v2400 : tensor<32x100352xf32>
    %v2475 = stablehlo.reshape %v385 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2476 = stablehlo.reshape %v2468 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2477 = stablehlo.transpose %v2475, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2478 = stablehlo.transpose %v2476, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2479 = stablehlo.convolution(%v2477, %v2478)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2480 = stablehlo.transpose %v2479, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2481 = stablehlo.reshape %v2468 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2482 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2483 = stablehlo.reduce(%v2481 init: %v2482) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2484 = stablehlo.reshape %v390 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2485 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2486 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2487 = stablehlo.reduce(%v2484 init: %v2485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2488 = stablehlo.broadcast_in_dim %v2487, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2489 = stablehlo.divide %v2488, %v2486 : tensor<32x128x28x28xf32>
    %v2490 = stablehlo.subtract %v2484, %v2489 : tensor<32x128x28x28xf32>
    %v2491 = stablehlo.multiply %v2490, %v2490 : tensor<32x128x28x28xf32>
    %v2492 = stablehlo.reduce(%v2491 init: %v2485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2493 = stablehlo.broadcast_in_dim %v2492, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2494 = stablehlo.divide %v2493, %v2486 : tensor<32x128x28x28xf32>
    %v2495 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2496 = stablehlo.add %v2494, %v2495 : tensor<32x128x28x28xf32>
    %v2497 = stablehlo.rsqrt %v2496 : tensor<32x128x28x28xf32>
    %v2498 = stablehlo.multiply %v2490, %v2497 : tensor<32x128x28x28xf32>
    %v2499 = stablehlo.reshape %v2438 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2500 = stablehlo.multiply %v2499, %v2498 : tensor<32x128x28x28xf32>
    %v2501 = stablehlo.reduce(%v2500 init: %v2485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2502 = stablehlo.reshape %v2438 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2504 = stablehlo.reduce(%v2502 init: %v2503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2505 = stablehlo.reshape %v412 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2506 = stablehlo.reshape %v2430 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2507 = stablehlo.transpose %v2505, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2508 = stablehlo.transpose %v2506, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2509 = stablehlo.convolution(%v2507, %v2508)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2510 = stablehlo.transpose %v2509, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2511 = stablehlo.reshape %v2430 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2512 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2513 = stablehlo.reduce(%v2511 init: %v2512) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2514 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2516 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2517 = stablehlo.reduce(%v2514 init: %v2515) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2518 = stablehlo.broadcast_in_dim %v2517, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2519 = stablehlo.divide %v2518, %v2516 : tensor<32x128x28x28xf32>
    %v2520 = stablehlo.subtract %v2514, %v2519 : tensor<32x128x28x28xf32>
    %v2521 = stablehlo.multiply %v2520, %v2520 : tensor<32x128x28x28xf32>
    %v2522 = stablehlo.reduce(%v2521 init: %v2515) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2523 = stablehlo.broadcast_in_dim %v2522, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2524 = stablehlo.divide %v2523, %v2516 : tensor<32x128x28x28xf32>
    %v2525 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2526 = stablehlo.add %v2524, %v2525 : tensor<32x128x28x28xf32>
    %v2527 = stablehlo.rsqrt %v2526 : tensor<32x128x28x28xf32>
    %v2528 = stablehlo.multiply %v2520, %v2527 : tensor<32x128x28x28xf32>
    %v2529 = stablehlo.reshape %v2400 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2530 = stablehlo.multiply %v2529, %v2528 : tensor<32x128x28x28xf32>
    %v2531 = stablehlo.reduce(%v2530 init: %v2515) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2532 = stablehlo.reshape %v2400 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2534 = stablehlo.reduce(%v2532 init: %v2533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2535 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2536 = stablehlo.compare GT, %v383, %v2535 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2537 = stablehlo.select %v2536, %v2474, %v2535 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2538 = stablehlo.reshape %v362 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2539 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2540 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2541 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2542 = stablehlo.reduce(%v2538 init: %v2539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2543 = stablehlo.broadcast_in_dim %v2542, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2544 = stablehlo.divide %v2543, %v2540 : tensor<32x128x28x28xf32>
    %v2545 = stablehlo.subtract %v2538, %v2544 : tensor<32x128x28x28xf32>
    %v2546 = stablehlo.multiply %v2545, %v2545 : tensor<32x128x28x28xf32>
    %v2547 = stablehlo.reduce(%v2546 init: %v2539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2548 = stablehlo.broadcast_in_dim %v2547, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2549 = stablehlo.divide %v2548, %v2540 : tensor<32x128x28x28xf32>
    %v2550 = stablehlo.add %v2549, %v2541 : tensor<32x128x28x28xf32>
    %v2551 = stablehlo.rsqrt %v2550 : tensor<32x128x28x28xf32>
    %v2552 = stablehlo.multiply %v2545, %v2551 : tensor<32x128x28x28xf32>
    %v2553 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2554 = stablehlo.reshape %v2537 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2555 = stablehlo.multiply %v2553, %v2554 : tensor<32x128x28x28xf32>
    %v2556 = stablehlo.reduce(%v2555 init: %v2539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2557 = stablehlo.broadcast_in_dim %v2556, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2558 = stablehlo.multiply %v2552, %v2555 : tensor<32x128x28x28xf32>
    %v2559 = stablehlo.reduce(%v2558 init: %v2539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2560 = stablehlo.broadcast_in_dim %v2559, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2561 = stablehlo.multiply %v2555, %v2540 : tensor<32x128x28x28xf32>
    %v2562 = stablehlo.subtract %v2561, %v2557 : tensor<32x128x28x28xf32>
    %v2563 = stablehlo.multiply %v2552, %v2560 : tensor<32x128x28x28xf32>
    %v2564 = stablehlo.subtract %v2562, %v2563 : tensor<32x128x28x28xf32>
    %v2565 = stablehlo.divide %v2551, %v2540 : tensor<32x128x28x28xf32>
    %v2566 = stablehlo.multiply %v2565, %v2564 : tensor<32x128x28x28xf32>
    %v2567 = stablehlo.reshape %v2566 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2568 = stablehlo.reshape %v2567 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2569 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2570 = stablehlo.transpose %v2569, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2571 = stablehlo.convolution(%v2568, %v2570)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2572 = stablehlo.reshape %v2571 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2573 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2574 = stablehlo.compare GT, %v355, %v2573 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2575 = stablehlo.select %v2574, %v2572, %v2573 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2576 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2577 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2578 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2579 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2580 = stablehlo.reduce(%v2576 init: %v2577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2581 = stablehlo.broadcast_in_dim %v2580, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2582 = stablehlo.divide %v2581, %v2578 : tensor<32x128x28x28xf32>
    %v2583 = stablehlo.subtract %v2576, %v2582 : tensor<32x128x28x28xf32>
    %v2584 = stablehlo.multiply %v2583, %v2583 : tensor<32x128x28x28xf32>
    %v2585 = stablehlo.reduce(%v2584 init: %v2577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2586 = stablehlo.broadcast_in_dim %v2585, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2587 = stablehlo.divide %v2586, %v2578 : tensor<32x128x28x28xf32>
    %v2588 = stablehlo.add %v2587, %v2579 : tensor<32x128x28x28xf32>
    %v2589 = stablehlo.rsqrt %v2588 : tensor<32x128x28x28xf32>
    %v2590 = stablehlo.multiply %v2583, %v2589 : tensor<32x128x28x28xf32>
    %v2591 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2592 = stablehlo.reshape %v2575 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2593 = stablehlo.multiply %v2591, %v2592 : tensor<32x128x28x28xf32>
    %v2594 = stablehlo.reduce(%v2593 init: %v2577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2595 = stablehlo.broadcast_in_dim %v2594, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2596 = stablehlo.multiply %v2590, %v2593 : tensor<32x128x28x28xf32>
    %v2597 = stablehlo.reduce(%v2596 init: %v2577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2598 = stablehlo.broadcast_in_dim %v2597, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2599 = stablehlo.multiply %v2593, %v2578 : tensor<32x128x28x28xf32>
    %v2600 = stablehlo.subtract %v2599, %v2595 : tensor<32x128x28x28xf32>
    %v2601 = stablehlo.multiply %v2590, %v2598 : tensor<32x128x28x28xf32>
    %v2602 = stablehlo.subtract %v2600, %v2601 : tensor<32x128x28x28xf32>
    %v2603 = stablehlo.divide %v2589, %v2578 : tensor<32x128x28x28xf32>
    %v2604 = stablehlo.multiply %v2603, %v2602 : tensor<32x128x28x28xf32>
    %v2605 = stablehlo.reshape %v2604 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2606 = stablehlo.reshape %v2605 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2607 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2608 = stablehlo.transpose %v2607, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2609 = stablehlo.convolution(%v2606, %v2608)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2610 = stablehlo.reshape %v2609 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2611 = stablehlo.add %v2610, %v2537 : tensor<32x100352xf32>
    %v2612 = stablehlo.reshape %v330 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2613 = stablehlo.reshape %v2605 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2614 = stablehlo.transpose %v2612, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2615 = stablehlo.transpose %v2613, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2616 = stablehlo.convolution(%v2614, %v2615)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2617 = stablehlo.transpose %v2616, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2618 = stablehlo.reshape %v2605 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2619 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2620 = stablehlo.reduce(%v2618 init: %v2619) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2621 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2622 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2623 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2624 = stablehlo.reduce(%v2621 init: %v2622) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2625 = stablehlo.broadcast_in_dim %v2624, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2626 = stablehlo.divide %v2625, %v2623 : tensor<32x128x28x28xf32>
    %v2627 = stablehlo.subtract %v2621, %v2626 : tensor<32x128x28x28xf32>
    %v2628 = stablehlo.multiply %v2627, %v2627 : tensor<32x128x28x28xf32>
    %v2629 = stablehlo.reduce(%v2628 init: %v2622) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2630 = stablehlo.broadcast_in_dim %v2629, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2631 = stablehlo.divide %v2630, %v2623 : tensor<32x128x28x28xf32>
    %v2632 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2633 = stablehlo.add %v2631, %v2632 : tensor<32x128x28x28xf32>
    %v2634 = stablehlo.rsqrt %v2633 : tensor<32x128x28x28xf32>
    %v2635 = stablehlo.multiply %v2627, %v2634 : tensor<32x128x28x28xf32>
    %v2636 = stablehlo.reshape %v2575 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2637 = stablehlo.multiply %v2636, %v2635 : tensor<32x128x28x28xf32>
    %v2638 = stablehlo.reduce(%v2637 init: %v2622) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2639 = stablehlo.reshape %v2575 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2641 = stablehlo.reduce(%v2639 init: %v2640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2642 = stablehlo.reshape %v357 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2643 = stablehlo.reshape %v2567 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2644 = stablehlo.transpose %v2642, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2645 = stablehlo.transpose %v2643, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2646 = stablehlo.convolution(%v2644, %v2645)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2647 = stablehlo.transpose %v2646, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2648 = stablehlo.reshape %v2567 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2649 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2650 = stablehlo.reduce(%v2648 init: %v2649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2651 = stablehlo.reshape %v362 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2652 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2653 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2654 = stablehlo.reduce(%v2651 init: %v2652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2655 = stablehlo.broadcast_in_dim %v2654, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2656 = stablehlo.divide %v2655, %v2653 : tensor<32x128x28x28xf32>
    %v2657 = stablehlo.subtract %v2651, %v2656 : tensor<32x128x28x28xf32>
    %v2658 = stablehlo.multiply %v2657, %v2657 : tensor<32x128x28x28xf32>
    %v2659 = stablehlo.reduce(%v2658 init: %v2652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2660 = stablehlo.broadcast_in_dim %v2659, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2661 = stablehlo.divide %v2660, %v2653 : tensor<32x128x28x28xf32>
    %v2662 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2663 = stablehlo.add %v2661, %v2662 : tensor<32x128x28x28xf32>
    %v2664 = stablehlo.rsqrt %v2663 : tensor<32x128x28x28xf32>
    %v2665 = stablehlo.multiply %v2657, %v2664 : tensor<32x128x28x28xf32>
    %v2666 = stablehlo.reshape %v2537 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2667 = stablehlo.multiply %v2666, %v2665 : tensor<32x128x28x28xf32>
    %v2668 = stablehlo.reduce(%v2667 init: %v2652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2669 = stablehlo.reshape %v2537 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2670 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2671 = stablehlo.reduce(%v2669 init: %v2670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2672 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2673 = stablehlo.compare GT, %v328, %v2672 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2674 = stablehlo.select %v2673, %v2611, %v2672 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2675 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2677 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2678 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2679 = stablehlo.reduce(%v2675 init: %v2676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2680 = stablehlo.broadcast_in_dim %v2679, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2681 = stablehlo.divide %v2680, %v2677 : tensor<32x128x28x28xf32>
    %v2682 = stablehlo.subtract %v2675, %v2681 : tensor<32x128x28x28xf32>
    %v2683 = stablehlo.multiply %v2682, %v2682 : tensor<32x128x28x28xf32>
    %v2684 = stablehlo.reduce(%v2683 init: %v2676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2685 = stablehlo.broadcast_in_dim %v2684, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2686 = stablehlo.divide %v2685, %v2677 : tensor<32x128x28x28xf32>
    %v2687 = stablehlo.add %v2686, %v2678 : tensor<32x128x28x28xf32>
    %v2688 = stablehlo.rsqrt %v2687 : tensor<32x128x28x28xf32>
    %v2689 = stablehlo.multiply %v2682, %v2688 : tensor<32x128x28x28xf32>
    %v2690 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2691 = stablehlo.reshape %v2674 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2692 = stablehlo.multiply %v2690, %v2691 : tensor<32x128x28x28xf32>
    %v2693 = stablehlo.reduce(%v2692 init: %v2676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2694 = stablehlo.broadcast_in_dim %v2693, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2695 = stablehlo.multiply %v2689, %v2692 : tensor<32x128x28x28xf32>
    %v2696 = stablehlo.reduce(%v2695 init: %v2676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2697 = stablehlo.broadcast_in_dim %v2696, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2698 = stablehlo.multiply %v2692, %v2677 : tensor<32x128x28x28xf32>
    %v2699 = stablehlo.subtract %v2698, %v2694 : tensor<32x128x28x28xf32>
    %v2700 = stablehlo.multiply %v2689, %v2697 : tensor<32x128x28x28xf32>
    %v2701 = stablehlo.subtract %v2699, %v2700 : tensor<32x128x28x28xf32>
    %v2702 = stablehlo.divide %v2688, %v2677 : tensor<32x128x28x28xf32>
    %v2703 = stablehlo.multiply %v2702, %v2701 : tensor<32x128x28x28xf32>
    %v2704 = stablehlo.reshape %v2703 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2705 = stablehlo.reshape %v2704 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2706 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2707 = stablehlo.transpose %v2706, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2708 = stablehlo.convolution(%v2705, %v2707)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2709 = stablehlo.reshape %v2708 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2710 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2711 = stablehlo.compare GT, %v300, %v2710 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2712 = stablehlo.select %v2711, %v2709, %v2710 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2713 = stablehlo.reshape %v280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2715 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2716 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2717 = stablehlo.reduce(%v2713 init: %v2714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2718 = stablehlo.broadcast_in_dim %v2717, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2719 = stablehlo.divide %v2718, %v2715 : tensor<32x128x28x28xf32>
    %v2720 = stablehlo.subtract %v2713, %v2719 : tensor<32x128x28x28xf32>
    %v2721 = stablehlo.multiply %v2720, %v2720 : tensor<32x128x28x28xf32>
    %v2722 = stablehlo.reduce(%v2721 init: %v2714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2723 = stablehlo.broadcast_in_dim %v2722, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2724 = stablehlo.divide %v2723, %v2715 : tensor<32x128x28x28xf32>
    %v2725 = stablehlo.add %v2724, %v2716 : tensor<32x128x28x28xf32>
    %v2726 = stablehlo.rsqrt %v2725 : tensor<32x128x28x28xf32>
    %v2727 = stablehlo.multiply %v2720, %v2726 : tensor<32x128x28x28xf32>
    %v2728 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2729 = stablehlo.reshape %v2712 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2730 = stablehlo.multiply %v2728, %v2729 : tensor<32x128x28x28xf32>
    %v2731 = stablehlo.reduce(%v2730 init: %v2714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2732 = stablehlo.broadcast_in_dim %v2731, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2733 = stablehlo.multiply %v2727, %v2730 : tensor<32x128x28x28xf32>
    %v2734 = stablehlo.reduce(%v2733 init: %v2714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2735 = stablehlo.broadcast_in_dim %v2734, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2736 = stablehlo.multiply %v2730, %v2715 : tensor<32x128x28x28xf32>
    %v2737 = stablehlo.subtract %v2736, %v2732 : tensor<32x128x28x28xf32>
    %v2738 = stablehlo.multiply %v2727, %v2735 : tensor<32x128x28x28xf32>
    %v2739 = stablehlo.subtract %v2737, %v2738 : tensor<32x128x28x28xf32>
    %v2740 = stablehlo.divide %v2726, %v2715 : tensor<32x128x28x28xf32>
    %v2741 = stablehlo.multiply %v2740, %v2739 : tensor<32x128x28x28xf32>
    %v2742 = stablehlo.reshape %v2741 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2743 = stablehlo.reshape %v2742 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2744 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2745 = stablehlo.transpose %v2744, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2746 = stablehlo.convolution(%v2743, %v2745)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2747 = stablehlo.reshape %v2746 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2748 = stablehlo.add %v2747, %v2674 : tensor<32x100352xf32>
    %v2749 = stablehlo.reshape %v275 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2750 = stablehlo.reshape %v2742 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2751 = stablehlo.transpose %v2749, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2752 = stablehlo.transpose %v2750, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2753 = stablehlo.convolution(%v2751, %v2752)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2754 = stablehlo.transpose %v2753, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2755 = stablehlo.reshape %v2742 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2757 = stablehlo.reduce(%v2755 init: %v2756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2758 = stablehlo.reshape %v280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2760 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2761 = stablehlo.reduce(%v2758 init: %v2759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2762 = stablehlo.broadcast_in_dim %v2761, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2763 = stablehlo.divide %v2762, %v2760 : tensor<32x128x28x28xf32>
    %v2764 = stablehlo.subtract %v2758, %v2763 : tensor<32x128x28x28xf32>
    %v2765 = stablehlo.multiply %v2764, %v2764 : tensor<32x128x28x28xf32>
    %v2766 = stablehlo.reduce(%v2765 init: %v2759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2767 = stablehlo.broadcast_in_dim %v2766, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2768 = stablehlo.divide %v2767, %v2760 : tensor<32x128x28x28xf32>
    %v2769 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2770 = stablehlo.add %v2768, %v2769 : tensor<32x128x28x28xf32>
    %v2771 = stablehlo.rsqrt %v2770 : tensor<32x128x28x28xf32>
    %v2772 = stablehlo.multiply %v2764, %v2771 : tensor<32x128x28x28xf32>
    %v2773 = stablehlo.reshape %v2712 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2774 = stablehlo.multiply %v2773, %v2772 : tensor<32x128x28x28xf32>
    %v2775 = stablehlo.reduce(%v2774 init: %v2759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2776 = stablehlo.reshape %v2712 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2777 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2778 = stablehlo.reduce(%v2776 init: %v2777) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2779 = stablehlo.reshape %v302 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2780 = stablehlo.reshape %v2704 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2781 = stablehlo.transpose %v2779, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2782 = stablehlo.transpose %v2780, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2783 = stablehlo.convolution(%v2781, %v2782)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2784 = stablehlo.transpose %v2783, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2785 = stablehlo.reshape %v2704 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2787 = stablehlo.reduce(%v2785 init: %v2786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2788 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2789 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2790 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2791 = stablehlo.reduce(%v2788 init: %v2789) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2792 = stablehlo.broadcast_in_dim %v2791, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2793 = stablehlo.divide %v2792, %v2790 : tensor<32x128x28x28xf32>
    %v2794 = stablehlo.subtract %v2788, %v2793 : tensor<32x128x28x28xf32>
    %v2795 = stablehlo.multiply %v2794, %v2794 : tensor<32x128x28x28xf32>
    %v2796 = stablehlo.reduce(%v2795 init: %v2789) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2797 = stablehlo.broadcast_in_dim %v2796, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2798 = stablehlo.divide %v2797, %v2790 : tensor<32x128x28x28xf32>
    %v2799 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2800 = stablehlo.add %v2798, %v2799 : tensor<32x128x28x28xf32>
    %v2801 = stablehlo.rsqrt %v2800 : tensor<32x128x28x28xf32>
    %v2802 = stablehlo.multiply %v2794, %v2801 : tensor<32x128x28x28xf32>
    %v2803 = stablehlo.reshape %v2674 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2804 = stablehlo.multiply %v2803, %v2802 : tensor<32x128x28x28xf32>
    %v2805 = stablehlo.reduce(%v2804 init: %v2789) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2806 = stablehlo.reshape %v2674 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2807 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2808 = stablehlo.reduce(%v2806 init: %v2807) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2809 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2810 = stablehlo.compare GT, %v273, %v2809 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2811 = stablehlo.select %v2810, %v2748, %v2809 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2812 = stablehlo.reshape %v227 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2813 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2814 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2815 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2816 = stablehlo.reduce(%v2812 init: %v2813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2817 = stablehlo.broadcast_in_dim %v2816, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2818 = stablehlo.divide %v2817, %v2814 : tensor<32x128x28x28xf32>
    %v2819 = stablehlo.subtract %v2812, %v2818 : tensor<32x128x28x28xf32>
    %v2820 = stablehlo.multiply %v2819, %v2819 : tensor<32x128x28x28xf32>
    %v2821 = stablehlo.reduce(%v2820 init: %v2813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2822 = stablehlo.broadcast_in_dim %v2821, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2823 = stablehlo.divide %v2822, %v2814 : tensor<32x128x28x28xf32>
    %v2824 = stablehlo.add %v2823, %v2815 : tensor<32x128x28x28xf32>
    %v2825 = stablehlo.rsqrt %v2824 : tensor<32x128x28x28xf32>
    %v2826 = stablehlo.multiply %v2819, %v2825 : tensor<32x128x28x28xf32>
    %v2827 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2828 = stablehlo.reshape %v2811 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2829 = stablehlo.multiply %v2827, %v2828 : tensor<32x128x28x28xf32>
    %v2830 = stablehlo.reduce(%v2829 init: %v2813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2831 = stablehlo.broadcast_in_dim %v2830, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2832 = stablehlo.multiply %v2826, %v2829 : tensor<32x128x28x28xf32>
    %v2833 = stablehlo.reduce(%v2832 init: %v2813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2834 = stablehlo.broadcast_in_dim %v2833, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2835 = stablehlo.multiply %v2829, %v2814 : tensor<32x128x28x28xf32>
    %v2836 = stablehlo.subtract %v2835, %v2831 : tensor<32x128x28x28xf32>
    %v2837 = stablehlo.multiply %v2826, %v2834 : tensor<32x128x28x28xf32>
    %v2838 = stablehlo.subtract %v2836, %v2837 : tensor<32x128x28x28xf32>
    %v2839 = stablehlo.divide %v2825, %v2814 : tensor<32x128x28x28xf32>
    %v2840 = stablehlo.multiply %v2839, %v2838 : tensor<32x128x28x28xf32>
    %v2841 = stablehlo.reshape %v2840 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2842 = stablehlo.reshape %v2841 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2843 = stablehlo.reverse %d2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2844 = stablehlo.transpose %v2843, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2845 = stablehlo.convolution(%v2842, %v2844)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2846 = stablehlo.reshape %v2845 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2847 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2848 = stablehlo.compare GT, %v220, %v2847 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2849 = stablehlo.select %v2848, %v2846, %v2847 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2850 = stablehlo.reshape %v200 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2852 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2853 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2854 = stablehlo.reduce(%v2850 init: %v2851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2855 = stablehlo.broadcast_in_dim %v2854, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2856 = stablehlo.divide %v2855, %v2852 : tensor<32x128x28x28xf32>
    %v2857 = stablehlo.subtract %v2850, %v2856 : tensor<32x128x28x28xf32>
    %v2858 = stablehlo.multiply %v2857, %v2857 : tensor<32x128x28x28xf32>
    %v2859 = stablehlo.reduce(%v2858 init: %v2851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2860 = stablehlo.broadcast_in_dim %v2859, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2861 = stablehlo.divide %v2860, %v2852 : tensor<32x128x28x28xf32>
    %v2862 = stablehlo.add %v2861, %v2853 : tensor<32x128x28x28xf32>
    %v2863 = stablehlo.rsqrt %v2862 : tensor<32x128x28x28xf32>
    %v2864 = stablehlo.multiply %v2857, %v2863 : tensor<32x128x28x28xf32>
    %v2865 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2866 = stablehlo.reshape %v2849 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2867 = stablehlo.multiply %v2865, %v2866 : tensor<32x128x28x28xf32>
    %v2868 = stablehlo.reduce(%v2867 init: %v2851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2869 = stablehlo.broadcast_in_dim %v2868, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2870 = stablehlo.multiply %v2864, %v2867 : tensor<32x128x28x28xf32>
    %v2871 = stablehlo.reduce(%v2870 init: %v2851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2872 = stablehlo.broadcast_in_dim %v2871, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2873 = stablehlo.multiply %v2867, %v2852 : tensor<32x128x28x28xf32>
    %v2874 = stablehlo.subtract %v2873, %v2869 : tensor<32x128x28x28xf32>
    %v2875 = stablehlo.multiply %v2864, %v2872 : tensor<32x128x28x28xf32>
    %v2876 = stablehlo.subtract %v2874, %v2875 : tensor<32x128x28x28xf32>
    %v2877 = stablehlo.divide %v2863, %v2852 : tensor<32x128x28x28xf32>
    %v2878 = stablehlo.multiply %v2877, %v2876 : tensor<32x128x28x28xf32>
    %v2879 = stablehlo.reshape %v2878 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2880 = stablehlo.reshape %v2879 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2882 = stablehlo.pad %v2880, %v2881, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v2883 = stablehlo.reverse %d2W1, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v2884 = stablehlo.transpose %v2883, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v2885 = stablehlo.convolution(%v2882, %v2884)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v2886 = stablehlo.reshape %v2885 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v2887 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2888 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2889 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2890 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2891 = stablehlo.reduce(%v2887 init: %v2888) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2892 = stablehlo.broadcast_in_dim %v2891, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2893 = stablehlo.divide %v2892, %v2889 : tensor<32x128x28x28xf32>
    %v2894 = stablehlo.subtract %v2887, %v2893 : tensor<32x128x28x28xf32>
    %v2895 = stablehlo.multiply %v2894, %v2894 : tensor<32x128x28x28xf32>
    %v2896 = stablehlo.reduce(%v2895 init: %v2888) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2897 = stablehlo.broadcast_in_dim %v2896, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2898 = stablehlo.divide %v2897, %v2889 : tensor<32x128x28x28xf32>
    %v2899 = stablehlo.add %v2898, %v2890 : tensor<32x128x28x28xf32>
    %v2900 = stablehlo.rsqrt %v2899 : tensor<32x128x28x28xf32>
    %v2901 = stablehlo.multiply %v2894, %v2900 : tensor<32x128x28x28xf32>
    %v2902 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2903 = stablehlo.reshape %v2811 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2904 = stablehlo.multiply %v2902, %v2903 : tensor<32x128x28x28xf32>
    %v2905 = stablehlo.reduce(%v2904 init: %v2888) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2906 = stablehlo.broadcast_in_dim %v2905, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2907 = stablehlo.multiply %v2901, %v2904 : tensor<32x128x28x28xf32>
    %v2908 = stablehlo.reduce(%v2907 init: %v2888) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2909 = stablehlo.broadcast_in_dim %v2908, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2910 = stablehlo.multiply %v2904, %v2889 : tensor<32x128x28x28xf32>
    %v2911 = stablehlo.subtract %v2910, %v2906 : tensor<32x128x28x28xf32>
    %v2912 = stablehlo.multiply %v2901, %v2909 : tensor<32x128x28x28xf32>
    %v2913 = stablehlo.subtract %v2911, %v2912 : tensor<32x128x28x28xf32>
    %v2914 = stablehlo.divide %v2900, %v2889 : tensor<32x128x28x28xf32>
    %v2915 = stablehlo.multiply %v2914, %v2913 : tensor<32x128x28x28xf32>
    %v2916 = stablehlo.reshape %v2915 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2917 = stablehlo.reshape %v2916 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2919 = stablehlo.pad %v2917, %v2918, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v2920 = stablehlo.reverse %d2Wp, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v2921 = stablehlo.transpose %v2920, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v2922 = stablehlo.convolution(%v2919, %v2921)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v2923 = stablehlo.reshape %v2922 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v2924 = stablehlo.add %v2886, %v2923 : tensor<32x200704xf32>
    %v2925 = stablehlo.reshape %v195 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2926 = stablehlo.reshape %v2879 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2928 = stablehlo.pad %v2926, %v2927, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v2929 = stablehlo.transpose %v2925, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v2930 = stablehlo.transpose %v2928, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v2931 = stablehlo.convolution(%v2929, %v2930)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x3x3xf32>
    %v2932 = stablehlo.transpose %v2931, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v2933 = stablehlo.reshape %v2879 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2934 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2935 = stablehlo.reduce(%v2933 init: %v2934) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2936 = stablehlo.reshape %v200 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2938 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2939 = stablehlo.reduce(%v2936 init: %v2937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2940 = stablehlo.broadcast_in_dim %v2939, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2941 = stablehlo.divide %v2940, %v2938 : tensor<32x128x28x28xf32>
    %v2942 = stablehlo.subtract %v2936, %v2941 : tensor<32x128x28x28xf32>
    %v2943 = stablehlo.multiply %v2942, %v2942 : tensor<32x128x28x28xf32>
    %v2944 = stablehlo.reduce(%v2943 init: %v2937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2945 = stablehlo.broadcast_in_dim %v2944, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2946 = stablehlo.divide %v2945, %v2938 : tensor<32x128x28x28xf32>
    %v2947 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2948 = stablehlo.add %v2946, %v2947 : tensor<32x128x28x28xf32>
    %v2949 = stablehlo.rsqrt %v2948 : tensor<32x128x28x28xf32>
    %v2950 = stablehlo.multiply %v2942, %v2949 : tensor<32x128x28x28xf32>
    %v2951 = stablehlo.reshape %v2849 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2952 = stablehlo.multiply %v2951, %v2950 : tensor<32x128x28x28xf32>
    %v2953 = stablehlo.reduce(%v2952 init: %v2937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2954 = stablehlo.reshape %v2849 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2955 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2956 = stablehlo.reduce(%v2954 init: %v2955) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2957 = stablehlo.reshape %v222 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2958 = stablehlo.reshape %v2841 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2959 = stablehlo.transpose %v2957, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2960 = stablehlo.transpose %v2958, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2961 = stablehlo.convolution(%v2959, %v2960)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2962 = stablehlo.transpose %v2961, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2963 = stablehlo.reshape %v2841 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2965 = stablehlo.reduce(%v2963 init: %v2964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2966 = stablehlo.reshape %v227 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2967 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2968 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2969 = stablehlo.reduce(%v2966 init: %v2967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2970 = stablehlo.broadcast_in_dim %v2969, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2971 = stablehlo.divide %v2970, %v2968 : tensor<32x128x28x28xf32>
    %v2972 = stablehlo.subtract %v2966, %v2971 : tensor<32x128x28x28xf32>
    %v2973 = stablehlo.multiply %v2972, %v2972 : tensor<32x128x28x28xf32>
    %v2974 = stablehlo.reduce(%v2973 init: %v2967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2975 = stablehlo.broadcast_in_dim %v2974, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2976 = stablehlo.divide %v2975, %v2968 : tensor<32x128x28x28xf32>
    %v2977 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2978 = stablehlo.add %v2976, %v2977 : tensor<32x128x28x28xf32>
    %v2979 = stablehlo.rsqrt %v2978 : tensor<32x128x28x28xf32>
    %v2980 = stablehlo.multiply %v2972, %v2979 : tensor<32x128x28x28xf32>
    %v2981 = stablehlo.reshape %v2811 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2982 = stablehlo.multiply %v2981, %v2980 : tensor<32x128x28x28xf32>
    %v2983 = stablehlo.reduce(%v2982 init: %v2967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2984 = stablehlo.reshape %v2811 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2985 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2986 = stablehlo.reduce(%v2984 init: %v2985) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2987 = stablehlo.reshape %v195 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2988 = stablehlo.reshape %v2916 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2989 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2990 = stablehlo.pad %v2988, %v2989, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v2991 = stablehlo.transpose %v2987, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v2992 = stablehlo.transpose %v2990, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v2993 = stablehlo.convolution(%v2991, %v2992)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x3x3xf32>
    %v2994 = stablehlo.transpose %v2993, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v2995 = stablehlo.reshape %v2916 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2996 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2997 = stablehlo.reduce(%v2995 init: %v2996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2998 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2999 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3000 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3001 = stablehlo.reduce(%v2998 init: %v2999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3002 = stablehlo.broadcast_in_dim %v3001, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3003 = stablehlo.divide %v3002, %v3000 : tensor<32x128x28x28xf32>
    %v3004 = stablehlo.subtract %v2998, %v3003 : tensor<32x128x28x28xf32>
    %v3005 = stablehlo.multiply %v3004, %v3004 : tensor<32x128x28x28xf32>
    %v3006 = stablehlo.reduce(%v3005 init: %v2999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3007 = stablehlo.broadcast_in_dim %v3006, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3008 = stablehlo.divide %v3007, %v3000 : tensor<32x128x28x28xf32>
    %v3009 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3010 = stablehlo.add %v3008, %v3009 : tensor<32x128x28x28xf32>
    %v3011 = stablehlo.rsqrt %v3010 : tensor<32x128x28x28xf32>
    %v3012 = stablehlo.multiply %v3004, %v3011 : tensor<32x128x28x28xf32>
    %v3013 = stablehlo.reshape %v2811 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3014 = stablehlo.multiply %v3013, %v3012 : tensor<32x128x28x28xf32>
    %v3015 = stablehlo.reduce(%v3014 init: %v2999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3016 = stablehlo.reshape %v2811 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3017 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3018 = stablehlo.reduce(%v3016 init: %v3017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3019 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3020 = stablehlo.compare GT, %v193, %v3019 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3021 = stablehlo.select %v3020, %v2924, %v3019 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3022 = stablehlo.reshape %v172 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3024 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3025 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3026 = stablehlo.reduce(%v3022 init: %v3023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3027 = stablehlo.broadcast_in_dim %v3026, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3028 = stablehlo.divide %v3027, %v3024 : tensor<32x64x56x56xf32>
    %v3029 = stablehlo.subtract %v3022, %v3028 : tensor<32x64x56x56xf32>
    %v3030 = stablehlo.multiply %v3029, %v3029 : tensor<32x64x56x56xf32>
    %v3031 = stablehlo.reduce(%v3030 init: %v3023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3032 = stablehlo.broadcast_in_dim %v3031, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3033 = stablehlo.divide %v3032, %v3024 : tensor<32x64x56x56xf32>
    %v3034 = stablehlo.add %v3033, %v3025 : tensor<32x64x56x56xf32>
    %v3035 = stablehlo.rsqrt %v3034 : tensor<32x64x56x56xf32>
    %v3036 = stablehlo.multiply %v3029, %v3035 : tensor<32x64x56x56xf32>
    %v3037 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3038 = stablehlo.reshape %v3021 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3039 = stablehlo.multiply %v3037, %v3038 : tensor<32x64x56x56xf32>
    %v3040 = stablehlo.reduce(%v3039 init: %v3023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3041 = stablehlo.broadcast_in_dim %v3040, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3042 = stablehlo.multiply %v3036, %v3039 : tensor<32x64x56x56xf32>
    %v3043 = stablehlo.reduce(%v3042 init: %v3023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3044 = stablehlo.broadcast_in_dim %v3043, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3045 = stablehlo.multiply %v3039, %v3024 : tensor<32x64x56x56xf32>
    %v3046 = stablehlo.subtract %v3045, %v3041 : tensor<32x64x56x56xf32>
    %v3047 = stablehlo.multiply %v3036, %v3044 : tensor<32x64x56x56xf32>
    %v3048 = stablehlo.subtract %v3046, %v3047 : tensor<32x64x56x56xf32>
    %v3049 = stablehlo.divide %v3035, %v3024 : tensor<32x64x56x56xf32>
    %v3050 = stablehlo.multiply %v3049, %v3048 : tensor<32x64x56x56xf32>
    %v3051 = stablehlo.reshape %v3050 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3052 = stablehlo.reshape %v3051 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3053 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3054 = stablehlo.transpose %v3053, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3055 = stablehlo.convolution(%v3052, %v3054)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3056 = stablehlo.reshape %v3055 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3057 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3058 = stablehlo.compare GT, %v165, %v3057 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3059 = stablehlo.select %v3058, %v3056, %v3057 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3060 = stablehlo.reshape %v145 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3062 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3063 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3064 = stablehlo.reduce(%v3060 init: %v3061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3065 = stablehlo.broadcast_in_dim %v3064, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3066 = stablehlo.divide %v3065, %v3062 : tensor<32x64x56x56xf32>
    %v3067 = stablehlo.subtract %v3060, %v3066 : tensor<32x64x56x56xf32>
    %v3068 = stablehlo.multiply %v3067, %v3067 : tensor<32x64x56x56xf32>
    %v3069 = stablehlo.reduce(%v3068 init: %v3061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3070 = stablehlo.broadcast_in_dim %v3069, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3071 = stablehlo.divide %v3070, %v3062 : tensor<32x64x56x56xf32>
    %v3072 = stablehlo.add %v3071, %v3063 : tensor<32x64x56x56xf32>
    %v3073 = stablehlo.rsqrt %v3072 : tensor<32x64x56x56xf32>
    %v3074 = stablehlo.multiply %v3067, %v3073 : tensor<32x64x56x56xf32>
    %v3075 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3076 = stablehlo.reshape %v3059 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3077 = stablehlo.multiply %v3075, %v3076 : tensor<32x64x56x56xf32>
    %v3078 = stablehlo.reduce(%v3077 init: %v3061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3079 = stablehlo.broadcast_in_dim %v3078, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3080 = stablehlo.multiply %v3074, %v3077 : tensor<32x64x56x56xf32>
    %v3081 = stablehlo.reduce(%v3080 init: %v3061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3082 = stablehlo.broadcast_in_dim %v3081, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3083 = stablehlo.multiply %v3077, %v3062 : tensor<32x64x56x56xf32>
    %v3084 = stablehlo.subtract %v3083, %v3079 : tensor<32x64x56x56xf32>
    %v3085 = stablehlo.multiply %v3074, %v3082 : tensor<32x64x56x56xf32>
    %v3086 = stablehlo.subtract %v3084, %v3085 : tensor<32x64x56x56xf32>
    %v3087 = stablehlo.divide %v3073, %v3062 : tensor<32x64x56x56xf32>
    %v3088 = stablehlo.multiply %v3087, %v3086 : tensor<32x64x56x56xf32>
    %v3089 = stablehlo.reshape %v3088 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3090 = stablehlo.reshape %v3089 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3091 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3092 = stablehlo.transpose %v3091, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3093 = stablehlo.convolution(%v3090, %v3092)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3094 = stablehlo.reshape %v3093 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3095 = stablehlo.add %v3094, %v3021 : tensor<32x200704xf32>
    %v3096 = stablehlo.reshape %v140 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3097 = stablehlo.reshape %v3089 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3098 = stablehlo.transpose %v3096, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3099 = stablehlo.transpose %v3097, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3100 = stablehlo.convolution(%v3098, %v3099)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3101 = stablehlo.transpose %v3100, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3102 = stablehlo.reshape %v3089 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3104 = stablehlo.reduce(%v3102 init: %v3103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3105 = stablehlo.reshape %v145 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3107 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3108 = stablehlo.reduce(%v3105 init: %v3106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3109 = stablehlo.broadcast_in_dim %v3108, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3110 = stablehlo.divide %v3109, %v3107 : tensor<32x64x56x56xf32>
    %v3111 = stablehlo.subtract %v3105, %v3110 : tensor<32x64x56x56xf32>
    %v3112 = stablehlo.multiply %v3111, %v3111 : tensor<32x64x56x56xf32>
    %v3113 = stablehlo.reduce(%v3112 init: %v3106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3114 = stablehlo.broadcast_in_dim %v3113, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3115 = stablehlo.divide %v3114, %v3107 : tensor<32x64x56x56xf32>
    %v3116 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3117 = stablehlo.add %v3115, %v3116 : tensor<32x64x56x56xf32>
    %v3118 = stablehlo.rsqrt %v3117 : tensor<32x64x56x56xf32>
    %v3119 = stablehlo.multiply %v3111, %v3118 : tensor<32x64x56x56xf32>
    %v3120 = stablehlo.reshape %v3059 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3121 = stablehlo.multiply %v3120, %v3119 : tensor<32x64x56x56xf32>
    %v3122 = stablehlo.reduce(%v3121 init: %v3106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3123 = stablehlo.reshape %v3059 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3125 = stablehlo.reduce(%v3123 init: %v3124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3126 = stablehlo.reshape %v167 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3127 = stablehlo.reshape %v3051 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3128 = stablehlo.transpose %v3126, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3129 = stablehlo.transpose %v3127, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3130 = stablehlo.convolution(%v3128, %v3129)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3131 = stablehlo.transpose %v3130, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3132 = stablehlo.reshape %v3051 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3134 = stablehlo.reduce(%v3132 init: %v3133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3135 = stablehlo.reshape %v172 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3137 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3138 = stablehlo.reduce(%v3135 init: %v3136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3139 = stablehlo.broadcast_in_dim %v3138, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3140 = stablehlo.divide %v3139, %v3137 : tensor<32x64x56x56xf32>
    %v3141 = stablehlo.subtract %v3135, %v3140 : tensor<32x64x56x56xf32>
    %v3142 = stablehlo.multiply %v3141, %v3141 : tensor<32x64x56x56xf32>
    %v3143 = stablehlo.reduce(%v3142 init: %v3136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3144 = stablehlo.broadcast_in_dim %v3143, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3145 = stablehlo.divide %v3144, %v3137 : tensor<32x64x56x56xf32>
    %v3146 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3147 = stablehlo.add %v3145, %v3146 : tensor<32x64x56x56xf32>
    %v3148 = stablehlo.rsqrt %v3147 : tensor<32x64x56x56xf32>
    %v3149 = stablehlo.multiply %v3141, %v3148 : tensor<32x64x56x56xf32>
    %v3150 = stablehlo.reshape %v3021 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3151 = stablehlo.multiply %v3150, %v3149 : tensor<32x64x56x56xf32>
    %v3152 = stablehlo.reduce(%v3151 init: %v3136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3153 = stablehlo.reshape %v3021 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3155 = stablehlo.reduce(%v3153 init: %v3154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3156 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3157 = stablehlo.compare GT, %v138, %v3156 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3158 = stablehlo.select %v3157, %v3095, %v3156 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3159 = stablehlo.reshape %v117 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3161 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3162 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3163 = stablehlo.reduce(%v3159 init: %v3160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3164 = stablehlo.broadcast_in_dim %v3163, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3165 = stablehlo.divide %v3164, %v3161 : tensor<32x64x56x56xf32>
    %v3166 = stablehlo.subtract %v3159, %v3165 : tensor<32x64x56x56xf32>
    %v3167 = stablehlo.multiply %v3166, %v3166 : tensor<32x64x56x56xf32>
    %v3168 = stablehlo.reduce(%v3167 init: %v3160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3169 = stablehlo.broadcast_in_dim %v3168, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3170 = stablehlo.divide %v3169, %v3161 : tensor<32x64x56x56xf32>
    %v3171 = stablehlo.add %v3170, %v3162 : tensor<32x64x56x56xf32>
    %v3172 = stablehlo.rsqrt %v3171 : tensor<32x64x56x56xf32>
    %v3173 = stablehlo.multiply %v3166, %v3172 : tensor<32x64x56x56xf32>
    %v3174 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3175 = stablehlo.reshape %v3158 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3176 = stablehlo.multiply %v3174, %v3175 : tensor<32x64x56x56xf32>
    %v3177 = stablehlo.reduce(%v3176 init: %v3160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3178 = stablehlo.broadcast_in_dim %v3177, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3179 = stablehlo.multiply %v3173, %v3176 : tensor<32x64x56x56xf32>
    %v3180 = stablehlo.reduce(%v3179 init: %v3160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3181 = stablehlo.broadcast_in_dim %v3180, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3182 = stablehlo.multiply %v3176, %v3161 : tensor<32x64x56x56xf32>
    %v3183 = stablehlo.subtract %v3182, %v3178 : tensor<32x64x56x56xf32>
    %v3184 = stablehlo.multiply %v3173, %v3181 : tensor<32x64x56x56xf32>
    %v3185 = stablehlo.subtract %v3183, %v3184 : tensor<32x64x56x56xf32>
    %v3186 = stablehlo.divide %v3172, %v3161 : tensor<32x64x56x56xf32>
    %v3187 = stablehlo.multiply %v3186, %v3185 : tensor<32x64x56x56xf32>
    %v3188 = stablehlo.reshape %v3187 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3189 = stablehlo.reshape %v3188 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3190 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3191 = stablehlo.transpose %v3190, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3192 = stablehlo.convolution(%v3189, %v3191)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3193 = stablehlo.reshape %v3192 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3194 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3195 = stablehlo.compare GT, %v110, %v3194 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3196 = stablehlo.select %v3195, %v3193, %v3194 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3197 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
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
    %v3212 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
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
    %v3228 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3229 = stablehlo.transpose %v3228, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3230 = stablehlo.convolution(%v3227, %v3229)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3231 = stablehlo.reshape %v3230 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3232 = stablehlo.add %v3231, %v3158 : tensor<32x200704xf32>
    %v3233 = stablehlo.reshape %v85 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3234 = stablehlo.reshape %v3226 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3235 = stablehlo.transpose %v3233, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3236 = stablehlo.transpose %v3234, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3237 = stablehlo.convolution(%v3235, %v3236)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3238 = stablehlo.transpose %v3237, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3239 = stablehlo.reshape %v3226 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3241 = stablehlo.reduce(%v3239 init: %v3240) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3242 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3243 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3244 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3245 = stablehlo.reduce(%v3242 init: %v3243) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3246 = stablehlo.broadcast_in_dim %v3245, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3247 = stablehlo.divide %v3246, %v3244 : tensor<32x64x56x56xf32>
    %v3248 = stablehlo.subtract %v3242, %v3247 : tensor<32x64x56x56xf32>
    %v3249 = stablehlo.multiply %v3248, %v3248 : tensor<32x64x56x56xf32>
    %v3250 = stablehlo.reduce(%v3249 init: %v3243) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3251 = stablehlo.broadcast_in_dim %v3250, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3252 = stablehlo.divide %v3251, %v3244 : tensor<32x64x56x56xf32>
    %v3253 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3254 = stablehlo.add %v3252, %v3253 : tensor<32x64x56x56xf32>
    %v3255 = stablehlo.rsqrt %v3254 : tensor<32x64x56x56xf32>
    %v3256 = stablehlo.multiply %v3248, %v3255 : tensor<32x64x56x56xf32>
    %v3257 = stablehlo.reshape %v3196 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3258 = stablehlo.multiply %v3257, %v3256 : tensor<32x64x56x56xf32>
    %v3259 = stablehlo.reduce(%v3258 init: %v3243) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3260 = stablehlo.reshape %v3196 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3262 = stablehlo.reduce(%v3260 init: %v3261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3263 = stablehlo.reshape %v112 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3264 = stablehlo.reshape %v3188 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3265 = stablehlo.transpose %v3263, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3266 = stablehlo.transpose %v3264, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3267 = stablehlo.convolution(%v3265, %v3266)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3268 = stablehlo.transpose %v3267, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3269 = stablehlo.reshape %v3188 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3270 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3271 = stablehlo.reduce(%v3269 init: %v3270) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3272 = stablehlo.reshape %v117 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3274 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3275 = stablehlo.reduce(%v3272 init: %v3273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3276 = stablehlo.broadcast_in_dim %v3275, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3277 = stablehlo.divide %v3276, %v3274 : tensor<32x64x56x56xf32>
    %v3278 = stablehlo.subtract %v3272, %v3277 : tensor<32x64x56x56xf32>
    %v3279 = stablehlo.multiply %v3278, %v3278 : tensor<32x64x56x56xf32>
    %v3280 = stablehlo.reduce(%v3279 init: %v3273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3281 = stablehlo.broadcast_in_dim %v3280, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3282 = stablehlo.divide %v3281, %v3274 : tensor<32x64x56x56xf32>
    %v3283 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3284 = stablehlo.add %v3282, %v3283 : tensor<32x64x56x56xf32>
    %v3285 = stablehlo.rsqrt %v3284 : tensor<32x64x56x56xf32>
    %v3286 = stablehlo.multiply %v3278, %v3285 : tensor<32x64x56x56xf32>
    %v3287 = stablehlo.reshape %v3158 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3288 = stablehlo.multiply %v3287, %v3286 : tensor<32x64x56x56xf32>
    %v3289 = stablehlo.reduce(%v3288 init: %v3273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3290 = stablehlo.reshape %v3158 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3291 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3292 = stablehlo.reduce(%v3290 init: %v3291) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3293 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3294 = stablehlo.compare GT, %v83, %v3293 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3295 = stablehlo.select %v3294, %v3232, %v3293 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3296 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3298 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3299 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3300 = stablehlo.reduce(%v3296 init: %v3297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3301 = stablehlo.broadcast_in_dim %v3300, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3302 = stablehlo.divide %v3301, %v3298 : tensor<32x64x56x56xf32>
    %v3303 = stablehlo.subtract %v3296, %v3302 : tensor<32x64x56x56xf32>
    %v3304 = stablehlo.multiply %v3303, %v3303 : tensor<32x64x56x56xf32>
    %v3305 = stablehlo.reduce(%v3304 init: %v3297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3306 = stablehlo.broadcast_in_dim %v3305, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3307 = stablehlo.divide %v3306, %v3298 : tensor<32x64x56x56xf32>
    %v3308 = stablehlo.add %v3307, %v3299 : tensor<32x64x56x56xf32>
    %v3309 = stablehlo.rsqrt %v3308 : tensor<32x64x56x56xf32>
    %v3310 = stablehlo.multiply %v3303, %v3309 : tensor<32x64x56x56xf32>
    %v3311 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3312 = stablehlo.reshape %v3295 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3313 = stablehlo.multiply %v3311, %v3312 : tensor<32x64x56x56xf32>
    %v3314 = stablehlo.reduce(%v3313 init: %v3297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3315 = stablehlo.broadcast_in_dim %v3314, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3316 = stablehlo.multiply %v3310, %v3313 : tensor<32x64x56x56xf32>
    %v3317 = stablehlo.reduce(%v3316 init: %v3297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3318 = stablehlo.broadcast_in_dim %v3317, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3319 = stablehlo.multiply %v3313, %v3298 : tensor<32x64x56x56xf32>
    %v3320 = stablehlo.subtract %v3319, %v3315 : tensor<32x64x56x56xf32>
    %v3321 = stablehlo.multiply %v3310, %v3318 : tensor<32x64x56x56xf32>
    %v3322 = stablehlo.subtract %v3320, %v3321 : tensor<32x64x56x56xf32>
    %v3323 = stablehlo.divide %v3309, %v3298 : tensor<32x64x56x56xf32>
    %v3324 = stablehlo.multiply %v3323, %v3322 : tensor<32x64x56x56xf32>
    %v3325 = stablehlo.reshape %v3324 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3326 = stablehlo.reshape %v3325 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3327 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3328 = stablehlo.transpose %v3327, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3329 = stablehlo.convolution(%v3326, %v3328)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3330 = stablehlo.reshape %v3329 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3331 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3332 = stablehlo.compare GT, %v55, %v3331 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3333 = stablehlo.select %v3332, %v3330, %v3331 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3334 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3336 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3337 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3338 = stablehlo.reduce(%v3334 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3339 = stablehlo.broadcast_in_dim %v3338, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3340 = stablehlo.divide %v3339, %v3336 : tensor<32x64x56x56xf32>
    %v3341 = stablehlo.subtract %v3334, %v3340 : tensor<32x64x56x56xf32>
    %v3342 = stablehlo.multiply %v3341, %v3341 : tensor<32x64x56x56xf32>
    %v3343 = stablehlo.reduce(%v3342 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3344 = stablehlo.broadcast_in_dim %v3343, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3345 = stablehlo.divide %v3344, %v3336 : tensor<32x64x56x56xf32>
    %v3346 = stablehlo.add %v3345, %v3337 : tensor<32x64x56x56xf32>
    %v3347 = stablehlo.rsqrt %v3346 : tensor<32x64x56x56xf32>
    %v3348 = stablehlo.multiply %v3341, %v3347 : tensor<32x64x56x56xf32>
    %v3349 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3350 = stablehlo.reshape %v3333 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3351 = stablehlo.multiply %v3349, %v3350 : tensor<32x64x56x56xf32>
    %v3352 = stablehlo.reduce(%v3351 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3353 = stablehlo.broadcast_in_dim %v3352, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3354 = stablehlo.multiply %v3348, %v3351 : tensor<32x64x56x56xf32>
    %v3355 = stablehlo.reduce(%v3354 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3356 = stablehlo.broadcast_in_dim %v3355, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3357 = stablehlo.multiply %v3351, %v3336 : tensor<32x64x56x56xf32>
    %v3358 = stablehlo.subtract %v3357, %v3353 : tensor<32x64x56x56xf32>
    %v3359 = stablehlo.multiply %v3348, %v3356 : tensor<32x64x56x56xf32>
    %v3360 = stablehlo.subtract %v3358, %v3359 : tensor<32x64x56x56xf32>
    %v3361 = stablehlo.divide %v3347, %v3336 : tensor<32x64x56x56xf32>
    %v3362 = stablehlo.multiply %v3361, %v3360 : tensor<32x64x56x56xf32>
    %v3363 = stablehlo.reshape %v3362 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3364 = stablehlo.reshape %v3363 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3365 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3366 = stablehlo.transpose %v3365, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3367 = stablehlo.convolution(%v3364, %v3366)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3368 = stablehlo.reshape %v3367 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3369 = stablehlo.add %v3368, %v3295 : tensor<32x200704xf32>
    %v3370 = stablehlo.reshape %v30 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3371 = stablehlo.reshape %v3363 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3372 = stablehlo.transpose %v3370, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3373 = stablehlo.transpose %v3371, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3374 = stablehlo.convolution(%v3372, %v3373)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3375 = stablehlo.transpose %v3374, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3376 = stablehlo.reshape %v3363 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3378 = stablehlo.reduce(%v3376 init: %v3377) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3379 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3380 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3381 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3382 = stablehlo.reduce(%v3379 init: %v3380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3383 = stablehlo.broadcast_in_dim %v3382, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3384 = stablehlo.divide %v3383, %v3381 : tensor<32x64x56x56xf32>
    %v3385 = stablehlo.subtract %v3379, %v3384 : tensor<32x64x56x56xf32>
    %v3386 = stablehlo.multiply %v3385, %v3385 : tensor<32x64x56x56xf32>
    %v3387 = stablehlo.reduce(%v3386 init: %v3380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3388 = stablehlo.broadcast_in_dim %v3387, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3389 = stablehlo.divide %v3388, %v3381 : tensor<32x64x56x56xf32>
    %v3390 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3391 = stablehlo.add %v3389, %v3390 : tensor<32x64x56x56xf32>
    %v3392 = stablehlo.rsqrt %v3391 : tensor<32x64x56x56xf32>
    %v3393 = stablehlo.multiply %v3385, %v3392 : tensor<32x64x56x56xf32>
    %v3394 = stablehlo.reshape %v3333 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3395 = stablehlo.multiply %v3394, %v3393 : tensor<32x64x56x56xf32>
    %v3396 = stablehlo.reduce(%v3395 init: %v3380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3397 = stablehlo.reshape %v3333 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3399 = stablehlo.reduce(%v3397 init: %v3398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3400 = stablehlo.reshape %v57 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3401 = stablehlo.reshape %v3325 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3402 = stablehlo.transpose %v3400, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3403 = stablehlo.transpose %v3401, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3404 = stablehlo.convolution(%v3402, %v3403)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3405 = stablehlo.transpose %v3404, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3406 = stablehlo.reshape %v3325 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3407 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3408 = stablehlo.reduce(%v3406 init: %v3407) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3409 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3411 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3412 = stablehlo.reduce(%v3409 init: %v3410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3413 = stablehlo.broadcast_in_dim %v3412, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3414 = stablehlo.divide %v3413, %v3411 : tensor<32x64x56x56xf32>
    %v3415 = stablehlo.subtract %v3409, %v3414 : tensor<32x64x56x56xf32>
    %v3416 = stablehlo.multiply %v3415, %v3415 : tensor<32x64x56x56xf32>
    %v3417 = stablehlo.reduce(%v3416 init: %v3410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3418 = stablehlo.broadcast_in_dim %v3417, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3419 = stablehlo.divide %v3418, %v3411 : tensor<32x64x56x56xf32>
    %v3420 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3421 = stablehlo.add %v3419, %v3420 : tensor<32x64x56x56xf32>
    %v3422 = stablehlo.rsqrt %v3421 : tensor<32x64x56x56xf32>
    %v3423 = stablehlo.multiply %v3415, %v3422 : tensor<32x64x56x56xf32>
    %v3424 = stablehlo.reshape %v3295 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3425 = stablehlo.multiply %v3424, %v3423 : tensor<32x64x56x56xf32>
    %v3426 = stablehlo.reduce(%v3425 init: %v3410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3427 = stablehlo.reshape %v3295 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3428 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3429 = stablehlo.reduce(%v3427 init: %v3428) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3430 = stablehlo.reshape %v26 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3431 = stablehlo.reshape %v3369 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3433 = "stablehlo.select_and_scatter"(%v3430, %v3431, %v3432) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<32x64x112x112xf32>, tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v3434 = stablehlo.reshape %v3433 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3435 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v3436 = stablehlo.compare GT, %v24, %v3435 : (tensor<32x802816xf32>, tensor<32x802816xf32>) -> tensor<32x802816xi1>
    %v3437 = stablehlo.select %v3436, %v3434, %v3435 : tensor<32x802816xi1>, tensor<32x802816xf32>
    %v3438 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3440 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3441 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3442 = stablehlo.reduce(%v3438 init: %v3439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3443 = stablehlo.broadcast_in_dim %v3442, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3444 = stablehlo.divide %v3443, %v3440 : tensor<32x64x112x112xf32>
    %v3445 = stablehlo.subtract %v3438, %v3444 : tensor<32x64x112x112xf32>
    %v3446 = stablehlo.multiply %v3445, %v3445 : tensor<32x64x112x112xf32>
    %v3447 = stablehlo.reduce(%v3446 init: %v3439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3448 = stablehlo.broadcast_in_dim %v3447, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3449 = stablehlo.divide %v3448, %v3440 : tensor<32x64x112x112xf32>
    %v3450 = stablehlo.add %v3449, %v3441 : tensor<32x64x112x112xf32>
    %v3451 = stablehlo.rsqrt %v3450 : tensor<32x64x112x112xf32>
    %v3452 = stablehlo.multiply %v3445, %v3451 : tensor<32x64x112x112xf32>
    %v3453 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3454 = stablehlo.reshape %v3437 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3455 = stablehlo.multiply %v3453, %v3454 : tensor<32x64x112x112xf32>
    %v3456 = stablehlo.reduce(%v3455 init: %v3439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3457 = stablehlo.broadcast_in_dim %v3456, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3458 = stablehlo.multiply %v3452, %v3455 : tensor<32x64x112x112xf32>
    %v3459 = stablehlo.reduce(%v3458 init: %v3439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3460 = stablehlo.broadcast_in_dim %v3459, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3461 = stablehlo.multiply %v3455, %v3440 : tensor<32x64x112x112xf32>
    %v3462 = stablehlo.subtract %v3461, %v3457 : tensor<32x64x112x112xf32>
    %v3463 = stablehlo.multiply %v3452, %v3460 : tensor<32x64x112x112xf32>
    %v3464 = stablehlo.subtract %v3462, %v3463 : tensor<32x64x112x112xf32>
    %v3465 = stablehlo.divide %v3451, %v3440 : tensor<32x64x112x112xf32>
    %v3466 = stablehlo.multiply %v3465, %v3464 : tensor<32x64x112x112xf32>
    %v3467 = stablehlo.reshape %v3466 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3468 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v3469 = stablehlo.reshape %v3467 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3471 = stablehlo.pad %v3469, %v3470, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x224x224xf32>
    %v3472 = stablehlo.transpose %v3468, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v3473 = stablehlo.transpose %v3471, dims = [1, 0, 2, 3] : (tensor<32x64x224x224xf32>) -> tensor<64x32x224x224xf32>
    %v3474 = stablehlo.convolution(%v3472, %v3473)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<64x32x224x224xf32>) -> tensor<3x64x7x7xf32>
    %v3475 = stablehlo.transpose %v3474, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v3476 = stablehlo.reshape %v3467 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3477 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3478 = stablehlo.reduce(%v3476 init: %v3477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3479 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3481 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3482 = stablehlo.reduce(%v3479 init: %v3480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3483 = stablehlo.broadcast_in_dim %v3482, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3484 = stablehlo.divide %v3483, %v3481 : tensor<32x64x112x112xf32>
    %v3485 = stablehlo.subtract %v3479, %v3484 : tensor<32x64x112x112xf32>
    %v3486 = stablehlo.multiply %v3485, %v3485 : tensor<32x64x112x112xf32>
    %v3487 = stablehlo.reduce(%v3486 init: %v3480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3488 = stablehlo.broadcast_in_dim %v3487, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3489 = stablehlo.divide %v3488, %v3481 : tensor<32x64x112x112xf32>
    %v3490 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3491 = stablehlo.add %v3489, %v3490 : tensor<32x64x112x112xf32>
    %v3492 = stablehlo.rsqrt %v3491 : tensor<32x64x112x112xf32>
    %v3493 = stablehlo.multiply %v3485, %v3492 : tensor<32x64x112x112xf32>
    %v3494 = stablehlo.reshape %v3437 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3495 = stablehlo.multiply %v3494, %v3493 : tensor<32x64x112x112xf32>
    %v3496 = stablehlo.reduce(%v3495 init: %v3480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3497 = stablehlo.reshape %v3437 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3499 = stablehlo.reduce(%v3497 init: %v3498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3500 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3501 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3502 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3503 = stablehlo.reduce(%v3500 init: %v3501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3504 = stablehlo.divide %v3503, %v3502 : tensor<64xf32>
    %v3505 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3507 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3508 = stablehlo.reduce(%v3505 init: %v3506) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3509 = stablehlo.broadcast_in_dim %v3508, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3510 = stablehlo.divide %v3509, %v3507 : tensor<32x64x112x112xf32>
    %v3511 = stablehlo.subtract %v3505, %v3510 : tensor<32x64x112x112xf32>
    %v3512 = stablehlo.multiply %v3511, %v3511 : tensor<32x64x112x112xf32>
    %v3513 = stablehlo.reduce(%v3512 init: %v3506) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3514 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3515 = stablehlo.divide %v3513, %v3514 : tensor<64xf32>
    %v3516 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3517 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3518 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3519 = stablehlo.reduce(%v3516 init: %v3517) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3520 = stablehlo.divide %v3519, %v3518 : tensor<64xf32>
    %v3521 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3522 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3523 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3524 = stablehlo.reduce(%v3521 init: %v3522) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3525 = stablehlo.broadcast_in_dim %v3524, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3526 = stablehlo.divide %v3525, %v3523 : tensor<32x64x56x56xf32>
    %v3527 = stablehlo.subtract %v3521, %v3526 : tensor<32x64x56x56xf32>
    %v3528 = stablehlo.multiply %v3527, %v3527 : tensor<32x64x56x56xf32>
    %v3529 = stablehlo.reduce(%v3528 init: %v3522) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3530 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3531 = stablehlo.divide %v3529, %v3530 : tensor<64xf32>
    %v3532 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3534 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3535 = stablehlo.reduce(%v3532 init: %v3533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3536 = stablehlo.divide %v3535, %v3534 : tensor<64xf32>
    %v3537 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3539 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3540 = stablehlo.reduce(%v3537 init: %v3538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3541 = stablehlo.broadcast_in_dim %v3540, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3542 = stablehlo.divide %v3541, %v3539 : tensor<32x64x56x56xf32>
    %v3543 = stablehlo.subtract %v3537, %v3542 : tensor<32x64x56x56xf32>
    %v3544 = stablehlo.multiply %v3543, %v3543 : tensor<32x64x56x56xf32>
    %v3545 = stablehlo.reduce(%v3544 init: %v3538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3546 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3547 = stablehlo.divide %v3545, %v3546 : tensor<64xf32>
    %v3548 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3550 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3551 = stablehlo.reduce(%v3548 init: %v3549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3552 = stablehlo.divide %v3551, %v3550 : tensor<64xf32>
    %v3553 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3555 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3556 = stablehlo.reduce(%v3553 init: %v3554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3557 = stablehlo.broadcast_in_dim %v3556, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3558 = stablehlo.divide %v3557, %v3555 : tensor<32x64x56x56xf32>
    %v3559 = stablehlo.subtract %v3553, %v3558 : tensor<32x64x56x56xf32>
    %v3560 = stablehlo.multiply %v3559, %v3559 : tensor<32x64x56x56xf32>
    %v3561 = stablehlo.reduce(%v3560 init: %v3554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3562 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3563 = stablehlo.divide %v3561, %v3562 : tensor<64xf32>
    %v3564 = stablehlo.reshape %v117 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3566 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3567 = stablehlo.reduce(%v3564 init: %v3565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3568 = stablehlo.divide %v3567, %v3566 : tensor<64xf32>
    %v3569 = stablehlo.reshape %v117 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3571 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3572 = stablehlo.reduce(%v3569 init: %v3570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3573 = stablehlo.broadcast_in_dim %v3572, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3574 = stablehlo.divide %v3573, %v3571 : tensor<32x64x56x56xf32>
    %v3575 = stablehlo.subtract %v3569, %v3574 : tensor<32x64x56x56xf32>
    %v3576 = stablehlo.multiply %v3575, %v3575 : tensor<32x64x56x56xf32>
    %v3577 = stablehlo.reduce(%v3576 init: %v3570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3578 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3579 = stablehlo.divide %v3577, %v3578 : tensor<64xf32>
    %v3580 = stablehlo.reshape %v145 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3582 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3583 = stablehlo.reduce(%v3580 init: %v3581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3584 = stablehlo.divide %v3583, %v3582 : tensor<64xf32>
    %v3585 = stablehlo.reshape %v145 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3586 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3587 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3588 = stablehlo.reduce(%v3585 init: %v3586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3589 = stablehlo.broadcast_in_dim %v3588, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3590 = stablehlo.divide %v3589, %v3587 : tensor<32x64x56x56xf32>
    %v3591 = stablehlo.subtract %v3585, %v3590 : tensor<32x64x56x56xf32>
    %v3592 = stablehlo.multiply %v3591, %v3591 : tensor<32x64x56x56xf32>
    %v3593 = stablehlo.reduce(%v3592 init: %v3586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3594 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3595 = stablehlo.divide %v3593, %v3594 : tensor<64xf32>
    %v3596 = stablehlo.reshape %v172 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3598 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3599 = stablehlo.reduce(%v3596 init: %v3597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3600 = stablehlo.divide %v3599, %v3598 : tensor<64xf32>
    %v3601 = stablehlo.reshape %v172 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3603 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3604 = stablehlo.reduce(%v3601 init: %v3602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3605 = stablehlo.broadcast_in_dim %v3604, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3606 = stablehlo.divide %v3605, %v3603 : tensor<32x64x56x56xf32>
    %v3607 = stablehlo.subtract %v3601, %v3606 : tensor<32x64x56x56xf32>
    %v3608 = stablehlo.multiply %v3607, %v3607 : tensor<32x64x56x56xf32>
    %v3609 = stablehlo.reduce(%v3608 init: %v3602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3610 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3611 = stablehlo.divide %v3609, %v3610 : tensor<64xf32>
    %v3612 = stablehlo.reshape %v200 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3614 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3615 = stablehlo.reduce(%v3612 init: %v3613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3616 = stablehlo.divide %v3615, %v3614 : tensor<128xf32>
    %v3617 = stablehlo.reshape %v200 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3619 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3620 = stablehlo.reduce(%v3617 init: %v3618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3621 = stablehlo.broadcast_in_dim %v3620, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3622 = stablehlo.divide %v3621, %v3619 : tensor<32x128x28x28xf32>
    %v3623 = stablehlo.subtract %v3617, %v3622 : tensor<32x128x28x28xf32>
    %v3624 = stablehlo.multiply %v3623, %v3623 : tensor<32x128x28x28xf32>
    %v3625 = stablehlo.reduce(%v3624 init: %v3618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3626 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3627 = stablehlo.divide %v3625, %v3626 : tensor<128xf32>
    %v3628 = stablehlo.reshape %v227 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3630 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3631 = stablehlo.reduce(%v3628 init: %v3629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3632 = stablehlo.divide %v3631, %v3630 : tensor<128xf32>
    %v3633 = stablehlo.reshape %v227 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3634 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3635 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3636 = stablehlo.reduce(%v3633 init: %v3634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3637 = stablehlo.broadcast_in_dim %v3636, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3638 = stablehlo.divide %v3637, %v3635 : tensor<32x128x28x28xf32>
    %v3639 = stablehlo.subtract %v3633, %v3638 : tensor<32x128x28x28xf32>
    %v3640 = stablehlo.multiply %v3639, %v3639 : tensor<32x128x28x28xf32>
    %v3641 = stablehlo.reduce(%v3640 init: %v3634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3642 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3643 = stablehlo.divide %v3641, %v3642 : tensor<128xf32>
    %v3644 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3646 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3647 = stablehlo.reduce(%v3644 init: %v3645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3648 = stablehlo.divide %v3647, %v3646 : tensor<128xf32>
    %v3649 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3650 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3651 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3652 = stablehlo.reduce(%v3649 init: %v3650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3653 = stablehlo.broadcast_in_dim %v3652, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3654 = stablehlo.divide %v3653, %v3651 : tensor<32x128x28x28xf32>
    %v3655 = stablehlo.subtract %v3649, %v3654 : tensor<32x128x28x28xf32>
    %v3656 = stablehlo.multiply %v3655, %v3655 : tensor<32x128x28x28xf32>
    %v3657 = stablehlo.reduce(%v3656 init: %v3650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3658 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3659 = stablehlo.divide %v3657, %v3658 : tensor<128xf32>
    %v3660 = stablehlo.reshape %v280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3661 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3662 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3663 = stablehlo.reduce(%v3660 init: %v3661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3664 = stablehlo.divide %v3663, %v3662 : tensor<128xf32>
    %v3665 = stablehlo.reshape %v280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3666 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3667 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3668 = stablehlo.reduce(%v3665 init: %v3666) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3669 = stablehlo.broadcast_in_dim %v3668, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3670 = stablehlo.divide %v3669, %v3667 : tensor<32x128x28x28xf32>
    %v3671 = stablehlo.subtract %v3665, %v3670 : tensor<32x128x28x28xf32>
    %v3672 = stablehlo.multiply %v3671, %v3671 : tensor<32x128x28x28xf32>
    %v3673 = stablehlo.reduce(%v3672 init: %v3666) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3674 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3675 = stablehlo.divide %v3673, %v3674 : tensor<128xf32>
    %v3676 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3677 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3678 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3679 = stablehlo.reduce(%v3676 init: %v3677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3680 = stablehlo.divide %v3679, %v3678 : tensor<128xf32>
    %v3681 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3682 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3683 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3684 = stablehlo.reduce(%v3681 init: %v3682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3685 = stablehlo.broadcast_in_dim %v3684, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3686 = stablehlo.divide %v3685, %v3683 : tensor<32x128x28x28xf32>
    %v3687 = stablehlo.subtract %v3681, %v3686 : tensor<32x128x28x28xf32>
    %v3688 = stablehlo.multiply %v3687, %v3687 : tensor<32x128x28x28xf32>
    %v3689 = stablehlo.reduce(%v3688 init: %v3682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3690 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3691 = stablehlo.divide %v3689, %v3690 : tensor<128xf32>
    %v3692 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3694 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3695 = stablehlo.reduce(%v3692 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3696 = stablehlo.divide %v3695, %v3694 : tensor<128xf32>
    %v3697 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3699 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3700 = stablehlo.reduce(%v3697 init: %v3698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3701 = stablehlo.broadcast_in_dim %v3700, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3702 = stablehlo.divide %v3701, %v3699 : tensor<32x128x28x28xf32>
    %v3703 = stablehlo.subtract %v3697, %v3702 : tensor<32x128x28x28xf32>
    %v3704 = stablehlo.multiply %v3703, %v3703 : tensor<32x128x28x28xf32>
    %v3705 = stablehlo.reduce(%v3704 init: %v3698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3706 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3707 = stablehlo.divide %v3705, %v3706 : tensor<128xf32>
    %v3708 = stablehlo.reshape %v362 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3710 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3711 = stablehlo.reduce(%v3708 init: %v3709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3712 = stablehlo.divide %v3711, %v3710 : tensor<128xf32>
    %v3713 = stablehlo.reshape %v362 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3715 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3716 = stablehlo.reduce(%v3713 init: %v3714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3717 = stablehlo.broadcast_in_dim %v3716, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3718 = stablehlo.divide %v3717, %v3715 : tensor<32x128x28x28xf32>
    %v3719 = stablehlo.subtract %v3713, %v3718 : tensor<32x128x28x28xf32>
    %v3720 = stablehlo.multiply %v3719, %v3719 : tensor<32x128x28x28xf32>
    %v3721 = stablehlo.reduce(%v3720 init: %v3714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3722 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3723 = stablehlo.divide %v3721, %v3722 : tensor<128xf32>
    %v3724 = stablehlo.reshape %v390 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3726 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3727 = stablehlo.reduce(%v3724 init: %v3725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3728 = stablehlo.divide %v3727, %v3726 : tensor<128xf32>
    %v3729 = stablehlo.reshape %v390 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3731 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3732 = stablehlo.reduce(%v3729 init: %v3730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3733 = stablehlo.broadcast_in_dim %v3732, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3734 = stablehlo.divide %v3733, %v3731 : tensor<32x128x28x28xf32>
    %v3735 = stablehlo.subtract %v3729, %v3734 : tensor<32x128x28x28xf32>
    %v3736 = stablehlo.multiply %v3735, %v3735 : tensor<32x128x28x28xf32>
    %v3737 = stablehlo.reduce(%v3736 init: %v3730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3738 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3739 = stablehlo.divide %v3737, %v3738 : tensor<128xf32>
    %v3740 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3741 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3742 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3743 = stablehlo.reduce(%v3740 init: %v3741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3744 = stablehlo.divide %v3743, %v3742 : tensor<128xf32>
    %v3745 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3747 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3748 = stablehlo.reduce(%v3745 init: %v3746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3749 = stablehlo.broadcast_in_dim %v3748, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3750 = stablehlo.divide %v3749, %v3747 : tensor<32x128x28x28xf32>
    %v3751 = stablehlo.subtract %v3745, %v3750 : tensor<32x128x28x28xf32>
    %v3752 = stablehlo.multiply %v3751, %v3751 : tensor<32x128x28x28xf32>
    %v3753 = stablehlo.reduce(%v3752 init: %v3746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3754 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3755 = stablehlo.divide %v3753, %v3754 : tensor<128xf32>
    %v3756 = stablehlo.reshape %v445 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3758 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3759 = stablehlo.reduce(%v3756 init: %v3757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3760 = stablehlo.divide %v3759, %v3758 : tensor<256xf32>
    %v3761 = stablehlo.reshape %v445 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3763 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3764 = stablehlo.reduce(%v3761 init: %v3762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3765 = stablehlo.broadcast_in_dim %v3764, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3766 = stablehlo.divide %v3765, %v3763 : tensor<32x256x14x14xf32>
    %v3767 = stablehlo.subtract %v3761, %v3766 : tensor<32x256x14x14xf32>
    %v3768 = stablehlo.multiply %v3767, %v3767 : tensor<32x256x14x14xf32>
    %v3769 = stablehlo.reduce(%v3768 init: %v3762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3770 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3771 = stablehlo.divide %v3769, %v3770 : tensor<256xf32>
    %v3772 = stablehlo.reshape %v472 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3773 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3774 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3775 = stablehlo.reduce(%v3772 init: %v3773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3776 = stablehlo.divide %v3775, %v3774 : tensor<256xf32>
    %v3777 = stablehlo.reshape %v472 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3779 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3780 = stablehlo.reduce(%v3777 init: %v3778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3781 = stablehlo.broadcast_in_dim %v3780, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3782 = stablehlo.divide %v3781, %v3779 : tensor<32x256x14x14xf32>
    %v3783 = stablehlo.subtract %v3777, %v3782 : tensor<32x256x14x14xf32>
    %v3784 = stablehlo.multiply %v3783, %v3783 : tensor<32x256x14x14xf32>
    %v3785 = stablehlo.reduce(%v3784 init: %v3778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3786 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3787 = stablehlo.divide %v3785, %v3786 : tensor<256xf32>
    %v3788 = stablehlo.reshape %v497 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3789 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3790 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3791 = stablehlo.reduce(%v3788 init: %v3789) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3792 = stablehlo.divide %v3791, %v3790 : tensor<256xf32>
    %v3793 = stablehlo.reshape %v497 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3794 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3795 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3796 = stablehlo.reduce(%v3793 init: %v3794) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3797 = stablehlo.broadcast_in_dim %v3796, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3798 = stablehlo.divide %v3797, %v3795 : tensor<32x256x14x14xf32>
    %v3799 = stablehlo.subtract %v3793, %v3798 : tensor<32x256x14x14xf32>
    %v3800 = stablehlo.multiply %v3799, %v3799 : tensor<32x256x14x14xf32>
    %v3801 = stablehlo.reduce(%v3800 init: %v3794) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3802 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3803 = stablehlo.divide %v3801, %v3802 : tensor<256xf32>
    %v3804 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3805 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3806 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3807 = stablehlo.reduce(%v3804 init: %v3805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3808 = stablehlo.divide %v3807, %v3806 : tensor<256xf32>
    %v3809 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3811 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3812 = stablehlo.reduce(%v3809 init: %v3810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3813 = stablehlo.broadcast_in_dim %v3812, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3814 = stablehlo.divide %v3813, %v3811 : tensor<32x256x14x14xf32>
    %v3815 = stablehlo.subtract %v3809, %v3814 : tensor<32x256x14x14xf32>
    %v3816 = stablehlo.multiply %v3815, %v3815 : tensor<32x256x14x14xf32>
    %v3817 = stablehlo.reduce(%v3816 init: %v3810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3818 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3819 = stablehlo.divide %v3817, %v3818 : tensor<256xf32>
    %v3820 = stablehlo.reshape %v552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3821 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3822 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3823 = stablehlo.reduce(%v3820 init: %v3821) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3824 = stablehlo.divide %v3823, %v3822 : tensor<256xf32>
    %v3825 = stablehlo.reshape %v552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3827 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3828 = stablehlo.reduce(%v3825 init: %v3826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3829 = stablehlo.broadcast_in_dim %v3828, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3830 = stablehlo.divide %v3829, %v3827 : tensor<32x256x14x14xf32>
    %v3831 = stablehlo.subtract %v3825, %v3830 : tensor<32x256x14x14xf32>
    %v3832 = stablehlo.multiply %v3831, %v3831 : tensor<32x256x14x14xf32>
    %v3833 = stablehlo.reduce(%v3832 init: %v3826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3834 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3835 = stablehlo.divide %v3833, %v3834 : tensor<256xf32>
    %v3836 = stablehlo.reshape %v580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3838 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3839 = stablehlo.reduce(%v3836 init: %v3837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3840 = stablehlo.divide %v3839, %v3838 : tensor<256xf32>
    %v3841 = stablehlo.reshape %v580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3843 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3844 = stablehlo.reduce(%v3841 init: %v3842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3845 = stablehlo.broadcast_in_dim %v3844, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3846 = stablehlo.divide %v3845, %v3843 : tensor<32x256x14x14xf32>
    %v3847 = stablehlo.subtract %v3841, %v3846 : tensor<32x256x14x14xf32>
    %v3848 = stablehlo.multiply %v3847, %v3847 : tensor<32x256x14x14xf32>
    %v3849 = stablehlo.reduce(%v3848 init: %v3842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3850 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3851 = stablehlo.divide %v3849, %v3850 : tensor<256xf32>
    %v3852 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3853 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3854 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3855 = stablehlo.reduce(%v3852 init: %v3853) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3856 = stablehlo.divide %v3855, %v3854 : tensor<256xf32>
    %v3857 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3858 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3859 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3860 = stablehlo.reduce(%v3857 init: %v3858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3861 = stablehlo.broadcast_in_dim %v3860, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3862 = stablehlo.divide %v3861, %v3859 : tensor<32x256x14x14xf32>
    %v3863 = stablehlo.subtract %v3857, %v3862 : tensor<32x256x14x14xf32>
    %v3864 = stablehlo.multiply %v3863, %v3863 : tensor<32x256x14x14xf32>
    %v3865 = stablehlo.reduce(%v3864 init: %v3858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3866 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3867 = stablehlo.divide %v3865, %v3866 : tensor<256xf32>
    %v3868 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3870 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3871 = stablehlo.reduce(%v3868 init: %v3869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3872 = stablehlo.divide %v3871, %v3870 : tensor<256xf32>
    %v3873 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3875 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3876 = stablehlo.reduce(%v3873 init: %v3874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3877 = stablehlo.broadcast_in_dim %v3876, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3878 = stablehlo.divide %v3877, %v3875 : tensor<32x256x14x14xf32>
    %v3879 = stablehlo.subtract %v3873, %v3878 : tensor<32x256x14x14xf32>
    %v3880 = stablehlo.multiply %v3879, %v3879 : tensor<32x256x14x14xf32>
    %v3881 = stablehlo.reduce(%v3880 init: %v3874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3882 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3883 = stablehlo.divide %v3881, %v3882 : tensor<256xf32>
    %v3884 = stablehlo.reshape %v662 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3886 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3887 = stablehlo.reduce(%v3884 init: %v3885) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3888 = stablehlo.divide %v3887, %v3886 : tensor<256xf32>
    %v3889 = stablehlo.reshape %v662 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3891 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3892 = stablehlo.reduce(%v3889 init: %v3890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3893 = stablehlo.broadcast_in_dim %v3892, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3894 = stablehlo.divide %v3893, %v3891 : tensor<32x256x14x14xf32>
    %v3895 = stablehlo.subtract %v3889, %v3894 : tensor<32x256x14x14xf32>
    %v3896 = stablehlo.multiply %v3895, %v3895 : tensor<32x256x14x14xf32>
    %v3897 = stablehlo.reduce(%v3896 init: %v3890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3898 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3899 = stablehlo.divide %v3897, %v3898 : tensor<256xf32>
    %v3900 = stablehlo.reshape %v690 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3901 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3902 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3903 = stablehlo.reduce(%v3900 init: %v3901) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3904 = stablehlo.divide %v3903, %v3902 : tensor<256xf32>
    %v3905 = stablehlo.reshape %v690 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3907 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3908 = stablehlo.reduce(%v3905 init: %v3906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3909 = stablehlo.broadcast_in_dim %v3908, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3910 = stablehlo.divide %v3909, %v3907 : tensor<32x256x14x14xf32>
    %v3911 = stablehlo.subtract %v3905, %v3910 : tensor<32x256x14x14xf32>
    %v3912 = stablehlo.multiply %v3911, %v3911 : tensor<32x256x14x14xf32>
    %v3913 = stablehlo.reduce(%v3912 init: %v3906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3914 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3915 = stablehlo.divide %v3913, %v3914 : tensor<256xf32>
    %v3916 = stablehlo.reshape %v717 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3918 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3919 = stablehlo.reduce(%v3916 init: %v3917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3920 = stablehlo.divide %v3919, %v3918 : tensor<256xf32>
    %v3921 = stablehlo.reshape %v717 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3922 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3923 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3924 = stablehlo.reduce(%v3921 init: %v3922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3925 = stablehlo.broadcast_in_dim %v3924, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3926 = stablehlo.divide %v3925, %v3923 : tensor<32x256x14x14xf32>
    %v3927 = stablehlo.subtract %v3921, %v3926 : tensor<32x256x14x14xf32>
    %v3928 = stablehlo.multiply %v3927, %v3927 : tensor<32x256x14x14xf32>
    %v3929 = stablehlo.reduce(%v3928 init: %v3922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3930 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3931 = stablehlo.divide %v3929, %v3930 : tensor<256xf32>
    %v3932 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3934 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3935 = stablehlo.reduce(%v3932 init: %v3933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3936 = stablehlo.divide %v3935, %v3934 : tensor<256xf32>
    %v3937 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3939 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3940 = stablehlo.reduce(%v3937 init: %v3938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3941 = stablehlo.broadcast_in_dim %v3940, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3942 = stablehlo.divide %v3941, %v3939 : tensor<32x256x14x14xf32>
    %v3943 = stablehlo.subtract %v3937, %v3942 : tensor<32x256x14x14xf32>
    %v3944 = stablehlo.multiply %v3943, %v3943 : tensor<32x256x14x14xf32>
    %v3945 = stablehlo.reduce(%v3944 init: %v3938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3946 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3947 = stablehlo.divide %v3945, %v3946 : tensor<256xf32>
    %v3948 = stablehlo.reshape %v772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3950 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3951 = stablehlo.reduce(%v3948 init: %v3949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3952 = stablehlo.divide %v3951, %v3950 : tensor<256xf32>
    %v3953 = stablehlo.reshape %v772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3955 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3956 = stablehlo.reduce(%v3953 init: %v3954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3957 = stablehlo.broadcast_in_dim %v3956, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3958 = stablehlo.divide %v3957, %v3955 : tensor<32x256x14x14xf32>
    %v3959 = stablehlo.subtract %v3953, %v3958 : tensor<32x256x14x14xf32>
    %v3960 = stablehlo.multiply %v3959, %v3959 : tensor<32x256x14x14xf32>
    %v3961 = stablehlo.reduce(%v3960 init: %v3954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3962 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3963 = stablehlo.divide %v3961, %v3962 : tensor<256xf32>
    %v3964 = stablehlo.reshape %v800 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3965 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3966 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3967 = stablehlo.reduce(%v3964 init: %v3965) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3968 = stablehlo.divide %v3967, %v3966 : tensor<512xf32>
    %v3969 = stablehlo.reshape %v800 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3971 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v3972 = stablehlo.reduce(%v3969 init: %v3970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3973 = stablehlo.broadcast_in_dim %v3972, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v3974 = stablehlo.divide %v3973, %v3971 : tensor<32x512x7x7xf32>
    %v3975 = stablehlo.subtract %v3969, %v3974 : tensor<32x512x7x7xf32>
    %v3976 = stablehlo.multiply %v3975, %v3975 : tensor<32x512x7x7xf32>
    %v3977 = stablehlo.reduce(%v3976 init: %v3970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3978 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3979 = stablehlo.divide %v3977, %v3978 : tensor<512xf32>
    %v3980 = stablehlo.reshape %v827 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3981 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3982 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3983 = stablehlo.reduce(%v3980 init: %v3981) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3984 = stablehlo.divide %v3983, %v3982 : tensor<512xf32>
    %v3985 = stablehlo.reshape %v827 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3987 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v3988 = stablehlo.reduce(%v3985 init: %v3986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3989 = stablehlo.broadcast_in_dim %v3988, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v3990 = stablehlo.divide %v3989, %v3987 : tensor<32x512x7x7xf32>
    %v3991 = stablehlo.subtract %v3985, %v3990 : tensor<32x512x7x7xf32>
    %v3992 = stablehlo.multiply %v3991, %v3991 : tensor<32x512x7x7xf32>
    %v3993 = stablehlo.reduce(%v3992 init: %v3986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3994 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3995 = stablehlo.divide %v3993, %v3994 : tensor<512xf32>
    %v3996 = stablehlo.reshape %v852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v3997 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3998 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v3999 = stablehlo.reduce(%v3996 init: %v3997) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4000 = stablehlo.divide %v3999, %v3998 : tensor<512xf32>
    %v4001 = stablehlo.reshape %v852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4003 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4004 = stablehlo.reduce(%v4001 init: %v4002) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4005 = stablehlo.broadcast_in_dim %v4004, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4006 = stablehlo.divide %v4005, %v4003 : tensor<32x512x7x7xf32>
    %v4007 = stablehlo.subtract %v4001, %v4006 : tensor<32x512x7x7xf32>
    %v4008 = stablehlo.multiply %v4007, %v4007 : tensor<32x512x7x7xf32>
    %v4009 = stablehlo.reduce(%v4008 init: %v4002) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4010 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4011 = stablehlo.divide %v4009, %v4010 : tensor<512xf32>
    %v4012 = stablehlo.reshape %v880 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4014 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4015 = stablehlo.reduce(%v4012 init: %v4013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4016 = stablehlo.divide %v4015, %v4014 : tensor<512xf32>
    %v4017 = stablehlo.reshape %v880 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4019 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4020 = stablehlo.reduce(%v4017 init: %v4018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4021 = stablehlo.broadcast_in_dim %v4020, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4022 = stablehlo.divide %v4021, %v4019 : tensor<32x512x7x7xf32>
    %v4023 = stablehlo.subtract %v4017, %v4022 : tensor<32x512x7x7xf32>
    %v4024 = stablehlo.multiply %v4023, %v4023 : tensor<32x512x7x7xf32>
    %v4025 = stablehlo.reduce(%v4024 init: %v4018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4026 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4027 = stablehlo.divide %v4025, %v4026 : tensor<512xf32>
    %v4028 = stablehlo.reshape %v907 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4029 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4030 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4031 = stablehlo.reduce(%v4028 init: %v4029) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4032 = stablehlo.divide %v4031, %v4030 : tensor<512xf32>
    %v4033 = stablehlo.reshape %v907 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4034 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4035 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4036 = stablehlo.reduce(%v4033 init: %v4034) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4037 = stablehlo.broadcast_in_dim %v4036, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4038 = stablehlo.divide %v4037, %v4035 : tensor<32x512x7x7xf32>
    %v4039 = stablehlo.subtract %v4033, %v4038 : tensor<32x512x7x7xf32>
    %v4040 = stablehlo.multiply %v4039, %v4039 : tensor<32x512x7x7xf32>
    %v4041 = stablehlo.reduce(%v4040 init: %v4034) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4042 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4043 = stablehlo.divide %v4041, %v4042 : tensor<512xf32>
    %v4044 = stablehlo.reshape %v935 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4046 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4047 = stablehlo.reduce(%v4044 init: %v4045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4048 = stablehlo.divide %v4047, %v4046 : tensor<512xf32>
    %v4049 = stablehlo.reshape %v935 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4051 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4052 = stablehlo.reduce(%v4049 init: %v4050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4053 = stablehlo.broadcast_in_dim %v4052, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4054 = stablehlo.divide %v4053, %v4051 : tensor<32x512x7x7xf32>
    %v4055 = stablehlo.subtract %v4049, %v4054 : tensor<32x512x7x7xf32>
    %v4056 = stablehlo.multiply %v4055, %v4055 : tensor<32x512x7x7xf32>
    %v4057 = stablehlo.reduce(%v4056 init: %v4050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4058 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4059 = stablehlo.divide %v4057, %v4058 : tensor<512xf32>
    %v4060 = stablehlo.reshape %v962 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4062 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4063 = stablehlo.reduce(%v4060 init: %v4061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4064 = stablehlo.divide %v4063, %v4062 : tensor<512xf32>
    %v4065 = stablehlo.reshape %v962 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4066 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4067 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4068 = stablehlo.reduce(%v4065 init: %v4066) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4069 = stablehlo.broadcast_in_dim %v4068, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4070 = stablehlo.divide %v4069, %v4067 : tensor<32x512x7x7xf32>
    %v4071 = stablehlo.subtract %v4065, %v4070 : tensor<32x512x7x7xf32>
    %v4072 = stablehlo.multiply %v4071, %v4071 : tensor<32x512x7x7xf32>
    %v4073 = stablehlo.reduce(%v4072 init: %v4066) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4074 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4075 = stablehlo.divide %v4073, %v4074 : tensor<512xf32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %v4076 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4077 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4078 = stablehlo.multiply %v4076, %sWm : tensor<64x3x7x7xf32>
    %v4079 = stablehlo.multiply %v4077, %v3475 : tensor<64x3x7x7xf32>
    %v4080 = stablehlo.add %v4078, %v4079 : tensor<64x3x7x7xf32>
    %v4081 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4082 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4083 = stablehlo.multiply %v4081, %sWv : tensor<64x3x7x7xf32>
    %v4084 = stablehlo.multiply %v3475, %v3475 : tensor<64x3x7x7xf32>
    %v4085 = stablehlo.multiply %v4082, %v4084 : tensor<64x3x7x7xf32>
    %v4086 = stablehlo.add %v4083, %v4085 : tensor<64x3x7x7xf32>
    %v4087 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4088 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4089 = stablehlo.multiply %v4087, %sWm : tensor<64x3x7x7xf32>
    %v4090 = stablehlo.multiply %v4088, %v3475 : tensor<64x3x7x7xf32>
    %v4091 = stablehlo.add %v4089, %v4090 : tensor<64x3x7x7xf32>
    %v4092 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4093 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4094 = stablehlo.multiply %v4092, %sWv : tensor<64x3x7x7xf32>
    %v4095 = stablehlo.multiply %v3475, %v3475 : tensor<64x3x7x7xf32>
    %v4096 = stablehlo.multiply %v4093, %v4095 : tensor<64x3x7x7xf32>
    %v4097 = stablehlo.add %v4094, %v4096 : tensor<64x3x7x7xf32>
    %v4098 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4099 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4100 = stablehlo.divide %v4091, %v4098 : tensor<64x3x7x7xf32>
    %v4101 = stablehlo.divide %v4097, %v4099 : tensor<64x3x7x7xf32>
    %v4102 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4103 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4104 = stablehlo.sqrt %v4101 : tensor<64x3x7x7xf32>
    %v4105 = stablehlo.add %v4104, %v4103 : tensor<64x3x7x7xf32>
    %v4106 = stablehlo.divide %v4100, %v4105 : tensor<64x3x7x7xf32>
    %v4107 = stablehlo.multiply %v4102, %v4106 : tensor<64x3x7x7xf32>
    %v4108 = stablehlo.subtract %sW, %v4107 : tensor<64x3x7x7xf32>
    %v4109 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4110 = stablehlo.multiply %v4109, %v4102 : tensor<64x3x7x7xf32>
    %v4111 = stablehlo.multiply %v4110, %sW : tensor<64x3x7x7xf32>
    %v4112 = stablehlo.subtract %v4108, %v4111 : tensor<64x3x7x7xf32>
    %v4113 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4114 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4115 = stablehlo.multiply %v4113, %sbim : tensor<64xf32>
    %v4116 = stablehlo.multiply %v4114, %v3478 : tensor<64xf32>
    %v4117 = stablehlo.add %v4115, %v4116 : tensor<64xf32>
    %v4118 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4119 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4120 = stablehlo.multiply %v4118, %sbiv : tensor<64xf32>
    %v4121 = stablehlo.multiply %v3478, %v3478 : tensor<64xf32>
    %v4122 = stablehlo.multiply %v4119, %v4121 : tensor<64xf32>
    %v4123 = stablehlo.add %v4120, %v4122 : tensor<64xf32>
    %v4124 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4125 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4126 = stablehlo.multiply %v4124, %sbim : tensor<64xf32>
    %v4127 = stablehlo.multiply %v4125, %v3478 : tensor<64xf32>
    %v4128 = stablehlo.add %v4126, %v4127 : tensor<64xf32>
    %v4129 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4130 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4131 = stablehlo.multiply %v4129, %sbiv : tensor<64xf32>
    %v4132 = stablehlo.multiply %v3478, %v3478 : tensor<64xf32>
    %v4133 = stablehlo.multiply %v4130, %v4132 : tensor<64xf32>
    %v4134 = stablehlo.add %v4131, %v4133 : tensor<64xf32>
    %v4135 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4136 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4137 = stablehlo.divide %v4128, %v4135 : tensor<64xf32>
    %v4138 = stablehlo.divide %v4134, %v4136 : tensor<64xf32>
    %v4139 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4140 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4141 = stablehlo.sqrt %v4138 : tensor<64xf32>
    %v4142 = stablehlo.add %v4141, %v4140 : tensor<64xf32>
    %v4143 = stablehlo.divide %v4137, %v4142 : tensor<64xf32>
    %v4144 = stablehlo.multiply %v4139, %v4143 : tensor<64xf32>
    %v4145 = stablehlo.subtract %sbi, %v4144 : tensor<64xf32>
    %v4146 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4147 = stablehlo.multiply %v4146, %v4139 : tensor<64xf32>
    %v4148 = stablehlo.multiply %v4147, %sbi : tensor<64xf32>
    %v4149 = stablehlo.subtract %v4145, %v4148 : tensor<64xf32>
    %v4150 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4151 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4152 = stablehlo.multiply %v4150, %sgm : tensor<64xf32>
    %v4153 = stablehlo.multiply %v4151, %v3496 : tensor<64xf32>
    %v4154 = stablehlo.add %v4152, %v4153 : tensor<64xf32>
    %v4155 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4156 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4157 = stablehlo.multiply %v4155, %sgv : tensor<64xf32>
    %v4158 = stablehlo.multiply %v3496, %v3496 : tensor<64xf32>
    %v4159 = stablehlo.multiply %v4156, %v4158 : tensor<64xf32>
    %v4160 = stablehlo.add %v4157, %v4159 : tensor<64xf32>
    %v4161 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4162 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4163 = stablehlo.multiply %v4161, %sgm : tensor<64xf32>
    %v4164 = stablehlo.multiply %v4162, %v3496 : tensor<64xf32>
    %v4165 = stablehlo.add %v4163, %v4164 : tensor<64xf32>
    %v4166 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4167 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4168 = stablehlo.multiply %v4166, %sgv : tensor<64xf32>
    %v4169 = stablehlo.multiply %v3496, %v3496 : tensor<64xf32>
    %v4170 = stablehlo.multiply %v4167, %v4169 : tensor<64xf32>
    %v4171 = stablehlo.add %v4168, %v4170 : tensor<64xf32>
    %v4172 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4173 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4174 = stablehlo.divide %v4165, %v4172 : tensor<64xf32>
    %v4175 = stablehlo.divide %v4171, %v4173 : tensor<64xf32>
    %v4176 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4177 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4178 = stablehlo.sqrt %v4175 : tensor<64xf32>
    %v4179 = stablehlo.add %v4178, %v4177 : tensor<64xf32>
    %v4180 = stablehlo.divide %v4174, %v4179 : tensor<64xf32>
    %v4181 = stablehlo.multiply %v4176, %v4180 : tensor<64xf32>
    %v4182 = stablehlo.subtract %sg, %v4181 : tensor<64xf32>
    %v4183 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4184 = stablehlo.multiply %v4183, %v4176 : tensor<64xf32>
    %v4185 = stablehlo.multiply %v4184, %sg : tensor<64xf32>
    %v4186 = stablehlo.subtract %v4182, %v4185 : tensor<64xf32>
    %v4187 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4188 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4189 = stablehlo.multiply %v4187, %sbtm : tensor<64xf32>
    %v4190 = stablehlo.multiply %v4188, %v3499 : tensor<64xf32>
    %v4191 = stablehlo.add %v4189, %v4190 : tensor<64xf32>
    %v4192 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4193 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4194 = stablehlo.multiply %v4192, %sbtv : tensor<64xf32>
    %v4195 = stablehlo.multiply %v3499, %v3499 : tensor<64xf32>
    %v4196 = stablehlo.multiply %v4193, %v4195 : tensor<64xf32>
    %v4197 = stablehlo.add %v4194, %v4196 : tensor<64xf32>
    %v4198 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4199 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4200 = stablehlo.multiply %v4198, %sbtm : tensor<64xf32>
    %v4201 = stablehlo.multiply %v4199, %v3499 : tensor<64xf32>
    %v4202 = stablehlo.add %v4200, %v4201 : tensor<64xf32>
    %v4203 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4204 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4205 = stablehlo.multiply %v4203, %sbtv : tensor<64xf32>
    %v4206 = stablehlo.multiply %v3499, %v3499 : tensor<64xf32>
    %v4207 = stablehlo.multiply %v4204, %v4206 : tensor<64xf32>
    %v4208 = stablehlo.add %v4205, %v4207 : tensor<64xf32>
    %v4209 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4210 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4211 = stablehlo.divide %v4202, %v4209 : tensor<64xf32>
    %v4212 = stablehlo.divide %v4208, %v4210 : tensor<64xf32>
    %v4213 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4214 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4215 = stablehlo.sqrt %v4212 : tensor<64xf32>
    %v4216 = stablehlo.add %v4215, %v4214 : tensor<64xf32>
    %v4217 = stablehlo.divide %v4211, %v4216 : tensor<64xf32>
    %v4218 = stablehlo.multiply %v4213, %v4217 : tensor<64xf32>
    %v4219 = stablehlo.subtract %sbt, %v4218 : tensor<64xf32>
    %v4220 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4221 = stablehlo.multiply %v4220, %v4213 : tensor<64xf32>
    %v4222 = stablehlo.multiply %v4221, %sbt : tensor<64xf32>
    %v4223 = stablehlo.subtract %v4219, %v4222 : tensor<64xf32>
    %v4224 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4225 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4226 = stablehlo.multiply %v4224, %s1b0W1m : tensor<64x64x3x3xf32>
    %v4227 = stablehlo.multiply %v4225, %v3375 : tensor<64x64x3x3xf32>
    %v4228 = stablehlo.add %v4226, %v4227 : tensor<64x64x3x3xf32>
    %v4229 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4230 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4231 = stablehlo.multiply %v4229, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4232 = stablehlo.multiply %v3375, %v3375 : tensor<64x64x3x3xf32>
    %v4233 = stablehlo.multiply %v4230, %v4232 : tensor<64x64x3x3xf32>
    %v4234 = stablehlo.add %v4231, %v4233 : tensor<64x64x3x3xf32>
    %v4235 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4236 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4237 = stablehlo.multiply %v4235, %s1b0W1m : tensor<64x64x3x3xf32>
    %v4238 = stablehlo.multiply %v4236, %v3375 : tensor<64x64x3x3xf32>
    %v4239 = stablehlo.add %v4237, %v4238 : tensor<64x64x3x3xf32>
    %v4240 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4241 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4242 = stablehlo.multiply %v4240, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4243 = stablehlo.multiply %v3375, %v3375 : tensor<64x64x3x3xf32>
    %v4244 = stablehlo.multiply %v4241, %v4243 : tensor<64x64x3x3xf32>
    %v4245 = stablehlo.add %v4242, %v4244 : tensor<64x64x3x3xf32>
    %v4246 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4247 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4248 = stablehlo.divide %v4239, %v4246 : tensor<64x64x3x3xf32>
    %v4249 = stablehlo.divide %v4245, %v4247 : tensor<64x64x3x3xf32>
    %v4250 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4251 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4252 = stablehlo.sqrt %v4249 : tensor<64x64x3x3xf32>
    %v4253 = stablehlo.add %v4252, %v4251 : tensor<64x64x3x3xf32>
    %v4254 = stablehlo.divide %v4248, %v4253 : tensor<64x64x3x3xf32>
    %v4255 = stablehlo.multiply %v4250, %v4254 : tensor<64x64x3x3xf32>
    %v4256 = stablehlo.subtract %s1b0W1, %v4255 : tensor<64x64x3x3xf32>
    %v4257 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4258 = stablehlo.multiply %v4257, %v4250 : tensor<64x64x3x3xf32>
    %v4259 = stablehlo.multiply %v4258, %s1b0W1 : tensor<64x64x3x3xf32>
    %v4260 = stablehlo.subtract %v4256, %v4259 : tensor<64x64x3x3xf32>
    %v4261 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4262 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4263 = stablehlo.multiply %v4261, %s1b0b1m : tensor<64xf32>
    %v4264 = stablehlo.multiply %v4262, %v3378 : tensor<64xf32>
    %v4265 = stablehlo.add %v4263, %v4264 : tensor<64xf32>
    %v4266 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4267 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4268 = stablehlo.multiply %v4266, %s1b0b1v : tensor<64xf32>
    %v4269 = stablehlo.multiply %v3378, %v3378 : tensor<64xf32>
    %v4270 = stablehlo.multiply %v4267, %v4269 : tensor<64xf32>
    %v4271 = stablehlo.add %v4268, %v4270 : tensor<64xf32>
    %v4272 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4273 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4274 = stablehlo.multiply %v4272, %s1b0b1m : tensor<64xf32>
    %v4275 = stablehlo.multiply %v4273, %v3378 : tensor<64xf32>
    %v4276 = stablehlo.add %v4274, %v4275 : tensor<64xf32>
    %v4277 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4278 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4279 = stablehlo.multiply %v4277, %s1b0b1v : tensor<64xf32>
    %v4280 = stablehlo.multiply %v3378, %v3378 : tensor<64xf32>
    %v4281 = stablehlo.multiply %v4278, %v4280 : tensor<64xf32>
    %v4282 = stablehlo.add %v4279, %v4281 : tensor<64xf32>
    %v4283 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4284 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4285 = stablehlo.divide %v4276, %v4283 : tensor<64xf32>
    %v4286 = stablehlo.divide %v4282, %v4284 : tensor<64xf32>
    %v4287 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4288 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4289 = stablehlo.sqrt %v4286 : tensor<64xf32>
    %v4290 = stablehlo.add %v4289, %v4288 : tensor<64xf32>
    %v4291 = stablehlo.divide %v4285, %v4290 : tensor<64xf32>
    %v4292 = stablehlo.multiply %v4287, %v4291 : tensor<64xf32>
    %v4293 = stablehlo.subtract %s1b0b1, %v4292 : tensor<64xf32>
    %v4294 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4295 = stablehlo.multiply %v4294, %v4287 : tensor<64xf32>
    %v4296 = stablehlo.multiply %v4295, %s1b0b1 : tensor<64xf32>
    %v4297 = stablehlo.subtract %v4293, %v4296 : tensor<64xf32>
    %v4298 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4299 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4300 = stablehlo.multiply %v4298, %s1b0g1m : tensor<64xf32>
    %v4301 = stablehlo.multiply %v4299, %v3396 : tensor<64xf32>
    %v4302 = stablehlo.add %v4300, %v4301 : tensor<64xf32>
    %v4303 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4304 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4305 = stablehlo.multiply %v4303, %s1b0g1v : tensor<64xf32>
    %v4306 = stablehlo.multiply %v3396, %v3396 : tensor<64xf32>
    %v4307 = stablehlo.multiply %v4304, %v4306 : tensor<64xf32>
    %v4308 = stablehlo.add %v4305, %v4307 : tensor<64xf32>
    %v4309 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4310 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4311 = stablehlo.multiply %v4309, %s1b0g1m : tensor<64xf32>
    %v4312 = stablehlo.multiply %v4310, %v3396 : tensor<64xf32>
    %v4313 = stablehlo.add %v4311, %v4312 : tensor<64xf32>
    %v4314 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4315 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4316 = stablehlo.multiply %v4314, %s1b0g1v : tensor<64xf32>
    %v4317 = stablehlo.multiply %v3396, %v3396 : tensor<64xf32>
    %v4318 = stablehlo.multiply %v4315, %v4317 : tensor<64xf32>
    %v4319 = stablehlo.add %v4316, %v4318 : tensor<64xf32>
    %v4320 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4321 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4322 = stablehlo.divide %v4313, %v4320 : tensor<64xf32>
    %v4323 = stablehlo.divide %v4319, %v4321 : tensor<64xf32>
    %v4324 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4325 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4326 = stablehlo.sqrt %v4323 : tensor<64xf32>
    %v4327 = stablehlo.add %v4326, %v4325 : tensor<64xf32>
    %v4328 = stablehlo.divide %v4322, %v4327 : tensor<64xf32>
    %v4329 = stablehlo.multiply %v4324, %v4328 : tensor<64xf32>
    %v4330 = stablehlo.subtract %s1b0g1, %v4329 : tensor<64xf32>
    %v4331 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4332 = stablehlo.multiply %v4331, %v4324 : tensor<64xf32>
    %v4333 = stablehlo.multiply %v4332, %s1b0g1 : tensor<64xf32>
    %v4334 = stablehlo.subtract %v4330, %v4333 : tensor<64xf32>
    %v4335 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4336 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4337 = stablehlo.multiply %v4335, %s1b0bt1m : tensor<64xf32>
    %v4338 = stablehlo.multiply %v4336, %v3399 : tensor<64xf32>
    %v4339 = stablehlo.add %v4337, %v4338 : tensor<64xf32>
    %v4340 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4341 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4342 = stablehlo.multiply %v4340, %s1b0bt1v : tensor<64xf32>
    %v4343 = stablehlo.multiply %v3399, %v3399 : tensor<64xf32>
    %v4344 = stablehlo.multiply %v4341, %v4343 : tensor<64xf32>
    %v4345 = stablehlo.add %v4342, %v4344 : tensor<64xf32>
    %v4346 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4347 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4348 = stablehlo.multiply %v4346, %s1b0bt1m : tensor<64xf32>
    %v4349 = stablehlo.multiply %v4347, %v3399 : tensor<64xf32>
    %v4350 = stablehlo.add %v4348, %v4349 : tensor<64xf32>
    %v4351 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4352 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4353 = stablehlo.multiply %v4351, %s1b0bt1v : tensor<64xf32>
    %v4354 = stablehlo.multiply %v3399, %v3399 : tensor<64xf32>
    %v4355 = stablehlo.multiply %v4352, %v4354 : tensor<64xf32>
    %v4356 = stablehlo.add %v4353, %v4355 : tensor<64xf32>
    %v4357 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4358 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4359 = stablehlo.divide %v4350, %v4357 : tensor<64xf32>
    %v4360 = stablehlo.divide %v4356, %v4358 : tensor<64xf32>
    %v4361 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4362 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4363 = stablehlo.sqrt %v4360 : tensor<64xf32>
    %v4364 = stablehlo.add %v4363, %v4362 : tensor<64xf32>
    %v4365 = stablehlo.divide %v4359, %v4364 : tensor<64xf32>
    %v4366 = stablehlo.multiply %v4361, %v4365 : tensor<64xf32>
    %v4367 = stablehlo.subtract %s1b0bt1, %v4366 : tensor<64xf32>
    %v4368 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4369 = stablehlo.multiply %v4368, %v4361 : tensor<64xf32>
    %v4370 = stablehlo.multiply %v4369, %s1b0bt1 : tensor<64xf32>
    %v4371 = stablehlo.subtract %v4367, %v4370 : tensor<64xf32>
    %v4372 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4373 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4374 = stablehlo.multiply %v4372, %s1b0W2m : tensor<64x64x3x3xf32>
    %v4375 = stablehlo.multiply %v4373, %v3405 : tensor<64x64x3x3xf32>
    %v4376 = stablehlo.add %v4374, %v4375 : tensor<64x64x3x3xf32>
    %v4377 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4378 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4379 = stablehlo.multiply %v4377, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4380 = stablehlo.multiply %v3405, %v3405 : tensor<64x64x3x3xf32>
    %v4381 = stablehlo.multiply %v4378, %v4380 : tensor<64x64x3x3xf32>
    %v4382 = stablehlo.add %v4379, %v4381 : tensor<64x64x3x3xf32>
    %v4383 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4384 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4385 = stablehlo.multiply %v4383, %s1b0W2m : tensor<64x64x3x3xf32>
    %v4386 = stablehlo.multiply %v4384, %v3405 : tensor<64x64x3x3xf32>
    %v4387 = stablehlo.add %v4385, %v4386 : tensor<64x64x3x3xf32>
    %v4388 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4389 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4390 = stablehlo.multiply %v4388, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4391 = stablehlo.multiply %v3405, %v3405 : tensor<64x64x3x3xf32>
    %v4392 = stablehlo.multiply %v4389, %v4391 : tensor<64x64x3x3xf32>
    %v4393 = stablehlo.add %v4390, %v4392 : tensor<64x64x3x3xf32>
    %v4394 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4395 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4396 = stablehlo.divide %v4387, %v4394 : tensor<64x64x3x3xf32>
    %v4397 = stablehlo.divide %v4393, %v4395 : tensor<64x64x3x3xf32>
    %v4398 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4399 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4400 = stablehlo.sqrt %v4397 : tensor<64x64x3x3xf32>
    %v4401 = stablehlo.add %v4400, %v4399 : tensor<64x64x3x3xf32>
    %v4402 = stablehlo.divide %v4396, %v4401 : tensor<64x64x3x3xf32>
    %v4403 = stablehlo.multiply %v4398, %v4402 : tensor<64x64x3x3xf32>
    %v4404 = stablehlo.subtract %s1b0W2, %v4403 : tensor<64x64x3x3xf32>
    %v4405 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4406 = stablehlo.multiply %v4405, %v4398 : tensor<64x64x3x3xf32>
    %v4407 = stablehlo.multiply %v4406, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4408 = stablehlo.subtract %v4404, %v4407 : tensor<64x64x3x3xf32>
    %v4409 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4410 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4411 = stablehlo.multiply %v4409, %s1b0b2m : tensor<64xf32>
    %v4412 = stablehlo.multiply %v4410, %v3408 : tensor<64xf32>
    %v4413 = stablehlo.add %v4411, %v4412 : tensor<64xf32>
    %v4414 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4415 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4416 = stablehlo.multiply %v4414, %s1b0b2v : tensor<64xf32>
    %v4417 = stablehlo.multiply %v3408, %v3408 : tensor<64xf32>
    %v4418 = stablehlo.multiply %v4415, %v4417 : tensor<64xf32>
    %v4419 = stablehlo.add %v4416, %v4418 : tensor<64xf32>
    %v4420 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4421 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4422 = stablehlo.multiply %v4420, %s1b0b2m : tensor<64xf32>
    %v4423 = stablehlo.multiply %v4421, %v3408 : tensor<64xf32>
    %v4424 = stablehlo.add %v4422, %v4423 : tensor<64xf32>
    %v4425 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4426 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4427 = stablehlo.multiply %v4425, %s1b0b2v : tensor<64xf32>
    %v4428 = stablehlo.multiply %v3408, %v3408 : tensor<64xf32>
    %v4429 = stablehlo.multiply %v4426, %v4428 : tensor<64xf32>
    %v4430 = stablehlo.add %v4427, %v4429 : tensor<64xf32>
    %v4431 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4432 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4433 = stablehlo.divide %v4424, %v4431 : tensor<64xf32>
    %v4434 = stablehlo.divide %v4430, %v4432 : tensor<64xf32>
    %v4435 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4436 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4437 = stablehlo.sqrt %v4434 : tensor<64xf32>
    %v4438 = stablehlo.add %v4437, %v4436 : tensor<64xf32>
    %v4439 = stablehlo.divide %v4433, %v4438 : tensor<64xf32>
    %v4440 = stablehlo.multiply %v4435, %v4439 : tensor<64xf32>
    %v4441 = stablehlo.subtract %s1b0b2, %v4440 : tensor<64xf32>
    %v4442 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4443 = stablehlo.multiply %v4442, %v4435 : tensor<64xf32>
    %v4444 = stablehlo.multiply %v4443, %s1b0b2 : tensor<64xf32>
    %v4445 = stablehlo.subtract %v4441, %v4444 : tensor<64xf32>
    %v4446 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4447 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4448 = stablehlo.multiply %v4446, %s1b0g2m : tensor<64xf32>
    %v4449 = stablehlo.multiply %v4447, %v3426 : tensor<64xf32>
    %v4450 = stablehlo.add %v4448, %v4449 : tensor<64xf32>
    %v4451 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4452 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4453 = stablehlo.multiply %v4451, %s1b0g2v : tensor<64xf32>
    %v4454 = stablehlo.multiply %v3426, %v3426 : tensor<64xf32>
    %v4455 = stablehlo.multiply %v4452, %v4454 : tensor<64xf32>
    %v4456 = stablehlo.add %v4453, %v4455 : tensor<64xf32>
    %v4457 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4458 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4459 = stablehlo.multiply %v4457, %s1b0g2m : tensor<64xf32>
    %v4460 = stablehlo.multiply %v4458, %v3426 : tensor<64xf32>
    %v4461 = stablehlo.add %v4459, %v4460 : tensor<64xf32>
    %v4462 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4463 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4464 = stablehlo.multiply %v4462, %s1b0g2v : tensor<64xf32>
    %v4465 = stablehlo.multiply %v3426, %v3426 : tensor<64xf32>
    %v4466 = stablehlo.multiply %v4463, %v4465 : tensor<64xf32>
    %v4467 = stablehlo.add %v4464, %v4466 : tensor<64xf32>
    %v4468 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4469 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4470 = stablehlo.divide %v4461, %v4468 : tensor<64xf32>
    %v4471 = stablehlo.divide %v4467, %v4469 : tensor<64xf32>
    %v4472 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4473 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4474 = stablehlo.sqrt %v4471 : tensor<64xf32>
    %v4475 = stablehlo.add %v4474, %v4473 : tensor<64xf32>
    %v4476 = stablehlo.divide %v4470, %v4475 : tensor<64xf32>
    %v4477 = stablehlo.multiply %v4472, %v4476 : tensor<64xf32>
    %v4478 = stablehlo.subtract %s1b0g2, %v4477 : tensor<64xf32>
    %v4479 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4480 = stablehlo.multiply %v4479, %v4472 : tensor<64xf32>
    %v4481 = stablehlo.multiply %v4480, %s1b0g2 : tensor<64xf32>
    %v4482 = stablehlo.subtract %v4478, %v4481 : tensor<64xf32>
    %v4483 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4484 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4485 = stablehlo.multiply %v4483, %s1b0bt2m : tensor<64xf32>
    %v4486 = stablehlo.multiply %v4484, %v3429 : tensor<64xf32>
    %v4487 = stablehlo.add %v4485, %v4486 : tensor<64xf32>
    %v4488 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4489 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4490 = stablehlo.multiply %v4488, %s1b0bt2v : tensor<64xf32>
    %v4491 = stablehlo.multiply %v3429, %v3429 : tensor<64xf32>
    %v4492 = stablehlo.multiply %v4489, %v4491 : tensor<64xf32>
    %v4493 = stablehlo.add %v4490, %v4492 : tensor<64xf32>
    %v4494 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4495 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4496 = stablehlo.multiply %v4494, %s1b0bt2m : tensor<64xf32>
    %v4497 = stablehlo.multiply %v4495, %v3429 : tensor<64xf32>
    %v4498 = stablehlo.add %v4496, %v4497 : tensor<64xf32>
    %v4499 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4500 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4501 = stablehlo.multiply %v4499, %s1b0bt2v : tensor<64xf32>
    %v4502 = stablehlo.multiply %v3429, %v3429 : tensor<64xf32>
    %v4503 = stablehlo.multiply %v4500, %v4502 : tensor<64xf32>
    %v4504 = stablehlo.add %v4501, %v4503 : tensor<64xf32>
    %v4505 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4506 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4507 = stablehlo.divide %v4498, %v4505 : tensor<64xf32>
    %v4508 = stablehlo.divide %v4504, %v4506 : tensor<64xf32>
    %v4509 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4510 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4511 = stablehlo.sqrt %v4508 : tensor<64xf32>
    %v4512 = stablehlo.add %v4511, %v4510 : tensor<64xf32>
    %v4513 = stablehlo.divide %v4507, %v4512 : tensor<64xf32>
    %v4514 = stablehlo.multiply %v4509, %v4513 : tensor<64xf32>
    %v4515 = stablehlo.subtract %s1b0bt2, %v4514 : tensor<64xf32>
    %v4516 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4517 = stablehlo.multiply %v4516, %v4509 : tensor<64xf32>
    %v4518 = stablehlo.multiply %v4517, %s1b0bt2 : tensor<64xf32>
    %v4519 = stablehlo.subtract %v4515, %v4518 : tensor<64xf32>
    %v4520 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4521 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4522 = stablehlo.multiply %v4520, %s1b1W1m : tensor<64x64x3x3xf32>
    %v4523 = stablehlo.multiply %v4521, %v3238 : tensor<64x64x3x3xf32>
    %v4524 = stablehlo.add %v4522, %v4523 : tensor<64x64x3x3xf32>
    %v4525 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4526 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4527 = stablehlo.multiply %v4525, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4528 = stablehlo.multiply %v3238, %v3238 : tensor<64x64x3x3xf32>
    %v4529 = stablehlo.multiply %v4526, %v4528 : tensor<64x64x3x3xf32>
    %v4530 = stablehlo.add %v4527, %v4529 : tensor<64x64x3x3xf32>
    %v4531 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4532 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4533 = stablehlo.multiply %v4531, %s1b1W1m : tensor<64x64x3x3xf32>
    %v4534 = stablehlo.multiply %v4532, %v3238 : tensor<64x64x3x3xf32>
    %v4535 = stablehlo.add %v4533, %v4534 : tensor<64x64x3x3xf32>
    %v4536 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4537 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4538 = stablehlo.multiply %v4536, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4539 = stablehlo.multiply %v3238, %v3238 : tensor<64x64x3x3xf32>
    %v4540 = stablehlo.multiply %v4537, %v4539 : tensor<64x64x3x3xf32>
    %v4541 = stablehlo.add %v4538, %v4540 : tensor<64x64x3x3xf32>
    %v4542 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4543 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4544 = stablehlo.divide %v4535, %v4542 : tensor<64x64x3x3xf32>
    %v4545 = stablehlo.divide %v4541, %v4543 : tensor<64x64x3x3xf32>
    %v4546 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4547 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4548 = stablehlo.sqrt %v4545 : tensor<64x64x3x3xf32>
    %v4549 = stablehlo.add %v4548, %v4547 : tensor<64x64x3x3xf32>
    %v4550 = stablehlo.divide %v4544, %v4549 : tensor<64x64x3x3xf32>
    %v4551 = stablehlo.multiply %v4546, %v4550 : tensor<64x64x3x3xf32>
    %v4552 = stablehlo.subtract %s1b1W1, %v4551 : tensor<64x64x3x3xf32>
    %v4553 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4554 = stablehlo.multiply %v4553, %v4546 : tensor<64x64x3x3xf32>
    %v4555 = stablehlo.multiply %v4554, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4556 = stablehlo.subtract %v4552, %v4555 : tensor<64x64x3x3xf32>
    %v4557 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4558 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4559 = stablehlo.multiply %v4557, %s1b1b1m : tensor<64xf32>
    %v4560 = stablehlo.multiply %v4558, %v3241 : tensor<64xf32>
    %v4561 = stablehlo.add %v4559, %v4560 : tensor<64xf32>
    %v4562 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4563 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4564 = stablehlo.multiply %v4562, %s1b1b1v : tensor<64xf32>
    %v4565 = stablehlo.multiply %v3241, %v3241 : tensor<64xf32>
    %v4566 = stablehlo.multiply %v4563, %v4565 : tensor<64xf32>
    %v4567 = stablehlo.add %v4564, %v4566 : tensor<64xf32>
    %v4568 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4569 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4570 = stablehlo.multiply %v4568, %s1b1b1m : tensor<64xf32>
    %v4571 = stablehlo.multiply %v4569, %v3241 : tensor<64xf32>
    %v4572 = stablehlo.add %v4570, %v4571 : tensor<64xf32>
    %v4573 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4574 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4575 = stablehlo.multiply %v4573, %s1b1b1v : tensor<64xf32>
    %v4576 = stablehlo.multiply %v3241, %v3241 : tensor<64xf32>
    %v4577 = stablehlo.multiply %v4574, %v4576 : tensor<64xf32>
    %v4578 = stablehlo.add %v4575, %v4577 : tensor<64xf32>
    %v4579 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4580 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4581 = stablehlo.divide %v4572, %v4579 : tensor<64xf32>
    %v4582 = stablehlo.divide %v4578, %v4580 : tensor<64xf32>
    %v4583 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4584 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4585 = stablehlo.sqrt %v4582 : tensor<64xf32>
    %v4586 = stablehlo.add %v4585, %v4584 : tensor<64xf32>
    %v4587 = stablehlo.divide %v4581, %v4586 : tensor<64xf32>
    %v4588 = stablehlo.multiply %v4583, %v4587 : tensor<64xf32>
    %v4589 = stablehlo.subtract %s1b1b1, %v4588 : tensor<64xf32>
    %v4590 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4591 = stablehlo.multiply %v4590, %v4583 : tensor<64xf32>
    %v4592 = stablehlo.multiply %v4591, %s1b1b1 : tensor<64xf32>
    %v4593 = stablehlo.subtract %v4589, %v4592 : tensor<64xf32>
    %v4594 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4595 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4596 = stablehlo.multiply %v4594, %s1b1g1m : tensor<64xf32>
    %v4597 = stablehlo.multiply %v4595, %v3259 : tensor<64xf32>
    %v4598 = stablehlo.add %v4596, %v4597 : tensor<64xf32>
    %v4599 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4600 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4601 = stablehlo.multiply %v4599, %s1b1g1v : tensor<64xf32>
    %v4602 = stablehlo.multiply %v3259, %v3259 : tensor<64xf32>
    %v4603 = stablehlo.multiply %v4600, %v4602 : tensor<64xf32>
    %v4604 = stablehlo.add %v4601, %v4603 : tensor<64xf32>
    %v4605 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4606 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4607 = stablehlo.multiply %v4605, %s1b1g1m : tensor<64xf32>
    %v4608 = stablehlo.multiply %v4606, %v3259 : tensor<64xf32>
    %v4609 = stablehlo.add %v4607, %v4608 : tensor<64xf32>
    %v4610 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4611 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4612 = stablehlo.multiply %v4610, %s1b1g1v : tensor<64xf32>
    %v4613 = stablehlo.multiply %v3259, %v3259 : tensor<64xf32>
    %v4614 = stablehlo.multiply %v4611, %v4613 : tensor<64xf32>
    %v4615 = stablehlo.add %v4612, %v4614 : tensor<64xf32>
    %v4616 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4617 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4618 = stablehlo.divide %v4609, %v4616 : tensor<64xf32>
    %v4619 = stablehlo.divide %v4615, %v4617 : tensor<64xf32>
    %v4620 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4621 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4622 = stablehlo.sqrt %v4619 : tensor<64xf32>
    %v4623 = stablehlo.add %v4622, %v4621 : tensor<64xf32>
    %v4624 = stablehlo.divide %v4618, %v4623 : tensor<64xf32>
    %v4625 = stablehlo.multiply %v4620, %v4624 : tensor<64xf32>
    %v4626 = stablehlo.subtract %s1b1g1, %v4625 : tensor<64xf32>
    %v4627 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4628 = stablehlo.multiply %v4627, %v4620 : tensor<64xf32>
    %v4629 = stablehlo.multiply %v4628, %s1b1g1 : tensor<64xf32>
    %v4630 = stablehlo.subtract %v4626, %v4629 : tensor<64xf32>
    %v4631 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4632 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4633 = stablehlo.multiply %v4631, %s1b1bt1m : tensor<64xf32>
    %v4634 = stablehlo.multiply %v4632, %v3262 : tensor<64xf32>
    %v4635 = stablehlo.add %v4633, %v4634 : tensor<64xf32>
    %v4636 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4637 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4638 = stablehlo.multiply %v4636, %s1b1bt1v : tensor<64xf32>
    %v4639 = stablehlo.multiply %v3262, %v3262 : tensor<64xf32>
    %v4640 = stablehlo.multiply %v4637, %v4639 : tensor<64xf32>
    %v4641 = stablehlo.add %v4638, %v4640 : tensor<64xf32>
    %v4642 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4643 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4644 = stablehlo.multiply %v4642, %s1b1bt1m : tensor<64xf32>
    %v4645 = stablehlo.multiply %v4643, %v3262 : tensor<64xf32>
    %v4646 = stablehlo.add %v4644, %v4645 : tensor<64xf32>
    %v4647 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4648 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4649 = stablehlo.multiply %v4647, %s1b1bt1v : tensor<64xf32>
    %v4650 = stablehlo.multiply %v3262, %v3262 : tensor<64xf32>
    %v4651 = stablehlo.multiply %v4648, %v4650 : tensor<64xf32>
    %v4652 = stablehlo.add %v4649, %v4651 : tensor<64xf32>
    %v4653 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4654 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4655 = stablehlo.divide %v4646, %v4653 : tensor<64xf32>
    %v4656 = stablehlo.divide %v4652, %v4654 : tensor<64xf32>
    %v4657 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4658 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4659 = stablehlo.sqrt %v4656 : tensor<64xf32>
    %v4660 = stablehlo.add %v4659, %v4658 : tensor<64xf32>
    %v4661 = stablehlo.divide %v4655, %v4660 : tensor<64xf32>
    %v4662 = stablehlo.multiply %v4657, %v4661 : tensor<64xf32>
    %v4663 = stablehlo.subtract %s1b1bt1, %v4662 : tensor<64xf32>
    %v4664 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4665 = stablehlo.multiply %v4664, %v4657 : tensor<64xf32>
    %v4666 = stablehlo.multiply %v4665, %s1b1bt1 : tensor<64xf32>
    %v4667 = stablehlo.subtract %v4663, %v4666 : tensor<64xf32>
    %v4668 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4669 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4670 = stablehlo.multiply %v4668, %s1b1W2m : tensor<64x64x3x3xf32>
    %v4671 = stablehlo.multiply %v4669, %v3268 : tensor<64x64x3x3xf32>
    %v4672 = stablehlo.add %v4670, %v4671 : tensor<64x64x3x3xf32>
    %v4673 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4674 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4675 = stablehlo.multiply %v4673, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4676 = stablehlo.multiply %v3268, %v3268 : tensor<64x64x3x3xf32>
    %v4677 = stablehlo.multiply %v4674, %v4676 : tensor<64x64x3x3xf32>
    %v4678 = stablehlo.add %v4675, %v4677 : tensor<64x64x3x3xf32>
    %v4679 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4680 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4681 = stablehlo.multiply %v4679, %s1b1W2m : tensor<64x64x3x3xf32>
    %v4682 = stablehlo.multiply %v4680, %v3268 : tensor<64x64x3x3xf32>
    %v4683 = stablehlo.add %v4681, %v4682 : tensor<64x64x3x3xf32>
    %v4684 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4685 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4686 = stablehlo.multiply %v4684, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4687 = stablehlo.multiply %v3268, %v3268 : tensor<64x64x3x3xf32>
    %v4688 = stablehlo.multiply %v4685, %v4687 : tensor<64x64x3x3xf32>
    %v4689 = stablehlo.add %v4686, %v4688 : tensor<64x64x3x3xf32>
    %v4690 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4691 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4692 = stablehlo.divide %v4683, %v4690 : tensor<64x64x3x3xf32>
    %v4693 = stablehlo.divide %v4689, %v4691 : tensor<64x64x3x3xf32>
    %v4694 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4695 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4696 = stablehlo.sqrt %v4693 : tensor<64x64x3x3xf32>
    %v4697 = stablehlo.add %v4696, %v4695 : tensor<64x64x3x3xf32>
    %v4698 = stablehlo.divide %v4692, %v4697 : tensor<64x64x3x3xf32>
    %v4699 = stablehlo.multiply %v4694, %v4698 : tensor<64x64x3x3xf32>
    %v4700 = stablehlo.subtract %s1b1W2, %v4699 : tensor<64x64x3x3xf32>
    %v4701 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4702 = stablehlo.multiply %v4701, %v4694 : tensor<64x64x3x3xf32>
    %v4703 = stablehlo.multiply %v4702, %s1b1W2 : tensor<64x64x3x3xf32>
    %v4704 = stablehlo.subtract %v4700, %v4703 : tensor<64x64x3x3xf32>
    %v4705 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4706 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4707 = stablehlo.multiply %v4705, %s1b1b2m : tensor<64xf32>
    %v4708 = stablehlo.multiply %v4706, %v3271 : tensor<64xf32>
    %v4709 = stablehlo.add %v4707, %v4708 : tensor<64xf32>
    %v4710 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4711 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4712 = stablehlo.multiply %v4710, %s1b1b2v : tensor<64xf32>
    %v4713 = stablehlo.multiply %v3271, %v3271 : tensor<64xf32>
    %v4714 = stablehlo.multiply %v4711, %v4713 : tensor<64xf32>
    %v4715 = stablehlo.add %v4712, %v4714 : tensor<64xf32>
    %v4716 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4717 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4718 = stablehlo.multiply %v4716, %s1b1b2m : tensor<64xf32>
    %v4719 = stablehlo.multiply %v4717, %v3271 : tensor<64xf32>
    %v4720 = stablehlo.add %v4718, %v4719 : tensor<64xf32>
    %v4721 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4722 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4723 = stablehlo.multiply %v4721, %s1b1b2v : tensor<64xf32>
    %v4724 = stablehlo.multiply %v3271, %v3271 : tensor<64xf32>
    %v4725 = stablehlo.multiply %v4722, %v4724 : tensor<64xf32>
    %v4726 = stablehlo.add %v4723, %v4725 : tensor<64xf32>
    %v4727 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4728 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4729 = stablehlo.divide %v4720, %v4727 : tensor<64xf32>
    %v4730 = stablehlo.divide %v4726, %v4728 : tensor<64xf32>
    %v4731 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4732 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4733 = stablehlo.sqrt %v4730 : tensor<64xf32>
    %v4734 = stablehlo.add %v4733, %v4732 : tensor<64xf32>
    %v4735 = stablehlo.divide %v4729, %v4734 : tensor<64xf32>
    %v4736 = stablehlo.multiply %v4731, %v4735 : tensor<64xf32>
    %v4737 = stablehlo.subtract %s1b1b2, %v4736 : tensor<64xf32>
    %v4738 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4739 = stablehlo.multiply %v4738, %v4731 : tensor<64xf32>
    %v4740 = stablehlo.multiply %v4739, %s1b1b2 : tensor<64xf32>
    %v4741 = stablehlo.subtract %v4737, %v4740 : tensor<64xf32>
    %v4742 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4743 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4744 = stablehlo.multiply %v4742, %s1b1g2m : tensor<64xf32>
    %v4745 = stablehlo.multiply %v4743, %v3289 : tensor<64xf32>
    %v4746 = stablehlo.add %v4744, %v4745 : tensor<64xf32>
    %v4747 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4748 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4749 = stablehlo.multiply %v4747, %s1b1g2v : tensor<64xf32>
    %v4750 = stablehlo.multiply %v3289, %v3289 : tensor<64xf32>
    %v4751 = stablehlo.multiply %v4748, %v4750 : tensor<64xf32>
    %v4752 = stablehlo.add %v4749, %v4751 : tensor<64xf32>
    %v4753 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4754 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4755 = stablehlo.multiply %v4753, %s1b1g2m : tensor<64xf32>
    %v4756 = stablehlo.multiply %v4754, %v3289 : tensor<64xf32>
    %v4757 = stablehlo.add %v4755, %v4756 : tensor<64xf32>
    %v4758 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4759 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4760 = stablehlo.multiply %v4758, %s1b1g2v : tensor<64xf32>
    %v4761 = stablehlo.multiply %v3289, %v3289 : tensor<64xf32>
    %v4762 = stablehlo.multiply %v4759, %v4761 : tensor<64xf32>
    %v4763 = stablehlo.add %v4760, %v4762 : tensor<64xf32>
    %v4764 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4765 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4766 = stablehlo.divide %v4757, %v4764 : tensor<64xf32>
    %v4767 = stablehlo.divide %v4763, %v4765 : tensor<64xf32>
    %v4768 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4769 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4770 = stablehlo.sqrt %v4767 : tensor<64xf32>
    %v4771 = stablehlo.add %v4770, %v4769 : tensor<64xf32>
    %v4772 = stablehlo.divide %v4766, %v4771 : tensor<64xf32>
    %v4773 = stablehlo.multiply %v4768, %v4772 : tensor<64xf32>
    %v4774 = stablehlo.subtract %s1b1g2, %v4773 : tensor<64xf32>
    %v4775 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4776 = stablehlo.multiply %v4775, %v4768 : tensor<64xf32>
    %v4777 = stablehlo.multiply %v4776, %s1b1g2 : tensor<64xf32>
    %v4778 = stablehlo.subtract %v4774, %v4777 : tensor<64xf32>
    %v4779 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4780 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4781 = stablehlo.multiply %v4779, %s1b1bt2m : tensor<64xf32>
    %v4782 = stablehlo.multiply %v4780, %v3292 : tensor<64xf32>
    %v4783 = stablehlo.add %v4781, %v4782 : tensor<64xf32>
    %v4784 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4785 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4786 = stablehlo.multiply %v4784, %s1b1bt2v : tensor<64xf32>
    %v4787 = stablehlo.multiply %v3292, %v3292 : tensor<64xf32>
    %v4788 = stablehlo.multiply %v4785, %v4787 : tensor<64xf32>
    %v4789 = stablehlo.add %v4786, %v4788 : tensor<64xf32>
    %v4790 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4791 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4792 = stablehlo.multiply %v4790, %s1b1bt2m : tensor<64xf32>
    %v4793 = stablehlo.multiply %v4791, %v3292 : tensor<64xf32>
    %v4794 = stablehlo.add %v4792, %v4793 : tensor<64xf32>
    %v4795 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4796 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4797 = stablehlo.multiply %v4795, %s1b1bt2v : tensor<64xf32>
    %v4798 = stablehlo.multiply %v3292, %v3292 : tensor<64xf32>
    %v4799 = stablehlo.multiply %v4796, %v4798 : tensor<64xf32>
    %v4800 = stablehlo.add %v4797, %v4799 : tensor<64xf32>
    %v4801 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4802 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4803 = stablehlo.divide %v4794, %v4801 : tensor<64xf32>
    %v4804 = stablehlo.divide %v4800, %v4802 : tensor<64xf32>
    %v4805 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4806 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4807 = stablehlo.sqrt %v4804 : tensor<64xf32>
    %v4808 = stablehlo.add %v4807, %v4806 : tensor<64xf32>
    %v4809 = stablehlo.divide %v4803, %v4808 : tensor<64xf32>
    %v4810 = stablehlo.multiply %v4805, %v4809 : tensor<64xf32>
    %v4811 = stablehlo.subtract %s1b1bt2, %v4810 : tensor<64xf32>
    %v4812 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4813 = stablehlo.multiply %v4812, %v4805 : tensor<64xf32>
    %v4814 = stablehlo.multiply %v4813, %s1b1bt2 : tensor<64xf32>
    %v4815 = stablehlo.subtract %v4811, %v4814 : tensor<64xf32>
    %v4816 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4817 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4818 = stablehlo.multiply %v4816, %s1b2W1m : tensor<64x64x3x3xf32>
    %v4819 = stablehlo.multiply %v4817, %v3101 : tensor<64x64x3x3xf32>
    %v4820 = stablehlo.add %v4818, %v4819 : tensor<64x64x3x3xf32>
    %v4821 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4822 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4823 = stablehlo.multiply %v4821, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4824 = stablehlo.multiply %v3101, %v3101 : tensor<64x64x3x3xf32>
    %v4825 = stablehlo.multiply %v4822, %v4824 : tensor<64x64x3x3xf32>
    %v4826 = stablehlo.add %v4823, %v4825 : tensor<64x64x3x3xf32>
    %v4827 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4828 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4829 = stablehlo.multiply %v4827, %s1b2W1m : tensor<64x64x3x3xf32>
    %v4830 = stablehlo.multiply %v4828, %v3101 : tensor<64x64x3x3xf32>
    %v4831 = stablehlo.add %v4829, %v4830 : tensor<64x64x3x3xf32>
    %v4832 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4833 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4834 = stablehlo.multiply %v4832, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4835 = stablehlo.multiply %v3101, %v3101 : tensor<64x64x3x3xf32>
    %v4836 = stablehlo.multiply %v4833, %v4835 : tensor<64x64x3x3xf32>
    %v4837 = stablehlo.add %v4834, %v4836 : tensor<64x64x3x3xf32>
    %v4838 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4839 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4840 = stablehlo.divide %v4831, %v4838 : tensor<64x64x3x3xf32>
    %v4841 = stablehlo.divide %v4837, %v4839 : tensor<64x64x3x3xf32>
    %v4842 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4843 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4844 = stablehlo.sqrt %v4841 : tensor<64x64x3x3xf32>
    %v4845 = stablehlo.add %v4844, %v4843 : tensor<64x64x3x3xf32>
    %v4846 = stablehlo.divide %v4840, %v4845 : tensor<64x64x3x3xf32>
    %v4847 = stablehlo.multiply %v4842, %v4846 : tensor<64x64x3x3xf32>
    %v4848 = stablehlo.subtract %s1b2W1, %v4847 : tensor<64x64x3x3xf32>
    %v4849 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4850 = stablehlo.multiply %v4849, %v4842 : tensor<64x64x3x3xf32>
    %v4851 = stablehlo.multiply %v4850, %s1b2W1 : tensor<64x64x3x3xf32>
    %v4852 = stablehlo.subtract %v4848, %v4851 : tensor<64x64x3x3xf32>
    %v4853 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4854 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4855 = stablehlo.multiply %v4853, %s1b2b1m : tensor<64xf32>
    %v4856 = stablehlo.multiply %v4854, %v3104 : tensor<64xf32>
    %v4857 = stablehlo.add %v4855, %v4856 : tensor<64xf32>
    %v4858 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4859 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4860 = stablehlo.multiply %v4858, %s1b2b1v : tensor<64xf32>
    %v4861 = stablehlo.multiply %v3104, %v3104 : tensor<64xf32>
    %v4862 = stablehlo.multiply %v4859, %v4861 : tensor<64xf32>
    %v4863 = stablehlo.add %v4860, %v4862 : tensor<64xf32>
    %v4864 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4865 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4866 = stablehlo.multiply %v4864, %s1b2b1m : tensor<64xf32>
    %v4867 = stablehlo.multiply %v4865, %v3104 : tensor<64xf32>
    %v4868 = stablehlo.add %v4866, %v4867 : tensor<64xf32>
    %v4869 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4870 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4871 = stablehlo.multiply %v4869, %s1b2b1v : tensor<64xf32>
    %v4872 = stablehlo.multiply %v3104, %v3104 : tensor<64xf32>
    %v4873 = stablehlo.multiply %v4870, %v4872 : tensor<64xf32>
    %v4874 = stablehlo.add %v4871, %v4873 : tensor<64xf32>
    %v4875 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4876 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4877 = stablehlo.divide %v4868, %v4875 : tensor<64xf32>
    %v4878 = stablehlo.divide %v4874, %v4876 : tensor<64xf32>
    %v4879 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4880 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4881 = stablehlo.sqrt %v4878 : tensor<64xf32>
    %v4882 = stablehlo.add %v4881, %v4880 : tensor<64xf32>
    %v4883 = stablehlo.divide %v4877, %v4882 : tensor<64xf32>
    %v4884 = stablehlo.multiply %v4879, %v4883 : tensor<64xf32>
    %v4885 = stablehlo.subtract %s1b2b1, %v4884 : tensor<64xf32>
    %v4886 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4887 = stablehlo.multiply %v4886, %v4879 : tensor<64xf32>
    %v4888 = stablehlo.multiply %v4887, %s1b2b1 : tensor<64xf32>
    %v4889 = stablehlo.subtract %v4885, %v4888 : tensor<64xf32>
    %v4890 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4891 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4892 = stablehlo.multiply %v4890, %s1b2g1m : tensor<64xf32>
    %v4893 = stablehlo.multiply %v4891, %v3122 : tensor<64xf32>
    %v4894 = stablehlo.add %v4892, %v4893 : tensor<64xf32>
    %v4895 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4896 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4897 = stablehlo.multiply %v4895, %s1b2g1v : tensor<64xf32>
    %v4898 = stablehlo.multiply %v3122, %v3122 : tensor<64xf32>
    %v4899 = stablehlo.multiply %v4896, %v4898 : tensor<64xf32>
    %v4900 = stablehlo.add %v4897, %v4899 : tensor<64xf32>
    %v4901 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4902 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4903 = stablehlo.multiply %v4901, %s1b2g1m : tensor<64xf32>
    %v4904 = stablehlo.multiply %v4902, %v3122 : tensor<64xf32>
    %v4905 = stablehlo.add %v4903, %v4904 : tensor<64xf32>
    %v4906 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4907 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4908 = stablehlo.multiply %v4906, %s1b2g1v : tensor<64xf32>
    %v4909 = stablehlo.multiply %v3122, %v3122 : tensor<64xf32>
    %v4910 = stablehlo.multiply %v4907, %v4909 : tensor<64xf32>
    %v4911 = stablehlo.add %v4908, %v4910 : tensor<64xf32>
    %v4912 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4913 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4914 = stablehlo.divide %v4905, %v4912 : tensor<64xf32>
    %v4915 = stablehlo.divide %v4911, %v4913 : tensor<64xf32>
    %v4916 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4917 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4918 = stablehlo.sqrt %v4915 : tensor<64xf32>
    %v4919 = stablehlo.add %v4918, %v4917 : tensor<64xf32>
    %v4920 = stablehlo.divide %v4914, %v4919 : tensor<64xf32>
    %v4921 = stablehlo.multiply %v4916, %v4920 : tensor<64xf32>
    %v4922 = stablehlo.subtract %s1b2g1, %v4921 : tensor<64xf32>
    %v4923 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4924 = stablehlo.multiply %v4923, %v4916 : tensor<64xf32>
    %v4925 = stablehlo.multiply %v4924, %s1b2g1 : tensor<64xf32>
    %v4926 = stablehlo.subtract %v4922, %v4925 : tensor<64xf32>
    %v4927 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4928 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4929 = stablehlo.multiply %v4927, %s1b2bt1m : tensor<64xf32>
    %v4930 = stablehlo.multiply %v4928, %v3125 : tensor<64xf32>
    %v4931 = stablehlo.add %v4929, %v4930 : tensor<64xf32>
    %v4932 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4933 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4934 = stablehlo.multiply %v4932, %s1b2bt1v : tensor<64xf32>
    %v4935 = stablehlo.multiply %v3125, %v3125 : tensor<64xf32>
    %v4936 = stablehlo.multiply %v4933, %v4935 : tensor<64xf32>
    %v4937 = stablehlo.add %v4934, %v4936 : tensor<64xf32>
    %v4938 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4939 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4940 = stablehlo.multiply %v4938, %s1b2bt1m : tensor<64xf32>
    %v4941 = stablehlo.multiply %v4939, %v3125 : tensor<64xf32>
    %v4942 = stablehlo.add %v4940, %v4941 : tensor<64xf32>
    %v4943 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4944 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4945 = stablehlo.multiply %v4943, %s1b2bt1v : tensor<64xf32>
    %v4946 = stablehlo.multiply %v3125, %v3125 : tensor<64xf32>
    %v4947 = stablehlo.multiply %v4944, %v4946 : tensor<64xf32>
    %v4948 = stablehlo.add %v4945, %v4947 : tensor<64xf32>
    %v4949 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4950 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4951 = stablehlo.divide %v4942, %v4949 : tensor<64xf32>
    %v4952 = stablehlo.divide %v4948, %v4950 : tensor<64xf32>
    %v4953 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4954 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4955 = stablehlo.sqrt %v4952 : tensor<64xf32>
    %v4956 = stablehlo.add %v4955, %v4954 : tensor<64xf32>
    %v4957 = stablehlo.divide %v4951, %v4956 : tensor<64xf32>
    %v4958 = stablehlo.multiply %v4953, %v4957 : tensor<64xf32>
    %v4959 = stablehlo.subtract %s1b2bt1, %v4958 : tensor<64xf32>
    %v4960 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4961 = stablehlo.multiply %v4960, %v4953 : tensor<64xf32>
    %v4962 = stablehlo.multiply %v4961, %s1b2bt1 : tensor<64xf32>
    %v4963 = stablehlo.subtract %v4959, %v4962 : tensor<64xf32>
    %v4964 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4965 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4966 = stablehlo.multiply %v4964, %s1b2W2m : tensor<64x64x3x3xf32>
    %v4967 = stablehlo.multiply %v4965, %v3131 : tensor<64x64x3x3xf32>
    %v4968 = stablehlo.add %v4966, %v4967 : tensor<64x64x3x3xf32>
    %v4969 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4970 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4971 = stablehlo.multiply %v4969, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4972 = stablehlo.multiply %v3131, %v3131 : tensor<64x64x3x3xf32>
    %v4973 = stablehlo.multiply %v4970, %v4972 : tensor<64x64x3x3xf32>
    %v4974 = stablehlo.add %v4971, %v4973 : tensor<64x64x3x3xf32>
    %v4975 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4976 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4977 = stablehlo.multiply %v4975, %s1b2W2m : tensor<64x64x3x3xf32>
    %v4978 = stablehlo.multiply %v4976, %v3131 : tensor<64x64x3x3xf32>
    %v4979 = stablehlo.add %v4977, %v4978 : tensor<64x64x3x3xf32>
    %v4980 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4981 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4982 = stablehlo.multiply %v4980, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4983 = stablehlo.multiply %v3131, %v3131 : tensor<64x64x3x3xf32>
    %v4984 = stablehlo.multiply %v4981, %v4983 : tensor<64x64x3x3xf32>
    %v4985 = stablehlo.add %v4982, %v4984 : tensor<64x64x3x3xf32>
    %v4986 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4987 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4988 = stablehlo.divide %v4979, %v4986 : tensor<64x64x3x3xf32>
    %v4989 = stablehlo.divide %v4985, %v4987 : tensor<64x64x3x3xf32>
    %v4990 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4991 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4992 = stablehlo.sqrt %v4989 : tensor<64x64x3x3xf32>
    %v4993 = stablehlo.add %v4992, %v4991 : tensor<64x64x3x3xf32>
    %v4994 = stablehlo.divide %v4988, %v4993 : tensor<64x64x3x3xf32>
    %v4995 = stablehlo.multiply %v4990, %v4994 : tensor<64x64x3x3xf32>
    %v4996 = stablehlo.subtract %s1b2W2, %v4995 : tensor<64x64x3x3xf32>
    %v4997 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4998 = stablehlo.multiply %v4997, %v4990 : tensor<64x64x3x3xf32>
    %v4999 = stablehlo.multiply %v4998, %s1b2W2 : tensor<64x64x3x3xf32>
    %v5000 = stablehlo.subtract %v4996, %v4999 : tensor<64x64x3x3xf32>
    %v5001 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5002 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5003 = stablehlo.multiply %v5001, %s1b2b2m : tensor<64xf32>
    %v5004 = stablehlo.multiply %v5002, %v3134 : tensor<64xf32>
    %v5005 = stablehlo.add %v5003, %v5004 : tensor<64xf32>
    %v5006 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5007 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5008 = stablehlo.multiply %v5006, %s1b2b2v : tensor<64xf32>
    %v5009 = stablehlo.multiply %v3134, %v3134 : tensor<64xf32>
    %v5010 = stablehlo.multiply %v5007, %v5009 : tensor<64xf32>
    %v5011 = stablehlo.add %v5008, %v5010 : tensor<64xf32>
    %v5012 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5013 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5014 = stablehlo.multiply %v5012, %s1b2b2m : tensor<64xf32>
    %v5015 = stablehlo.multiply %v5013, %v3134 : tensor<64xf32>
    %v5016 = stablehlo.add %v5014, %v5015 : tensor<64xf32>
    %v5017 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5018 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5019 = stablehlo.multiply %v5017, %s1b2b2v : tensor<64xf32>
    %v5020 = stablehlo.multiply %v3134, %v3134 : tensor<64xf32>
    %v5021 = stablehlo.multiply %v5018, %v5020 : tensor<64xf32>
    %v5022 = stablehlo.add %v5019, %v5021 : tensor<64xf32>
    %v5023 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5024 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5025 = stablehlo.divide %v5016, %v5023 : tensor<64xf32>
    %v5026 = stablehlo.divide %v5022, %v5024 : tensor<64xf32>
    %v5027 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5028 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5029 = stablehlo.sqrt %v5026 : tensor<64xf32>
    %v5030 = stablehlo.add %v5029, %v5028 : tensor<64xf32>
    %v5031 = stablehlo.divide %v5025, %v5030 : tensor<64xf32>
    %v5032 = stablehlo.multiply %v5027, %v5031 : tensor<64xf32>
    %v5033 = stablehlo.subtract %s1b2b2, %v5032 : tensor<64xf32>
    %v5034 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5035 = stablehlo.multiply %v5034, %v5027 : tensor<64xf32>
    %v5036 = stablehlo.multiply %v5035, %s1b2b2 : tensor<64xf32>
    %v5037 = stablehlo.subtract %v5033, %v5036 : tensor<64xf32>
    %v5038 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5039 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5040 = stablehlo.multiply %v5038, %s1b2g2m : tensor<64xf32>
    %v5041 = stablehlo.multiply %v5039, %v3152 : tensor<64xf32>
    %v5042 = stablehlo.add %v5040, %v5041 : tensor<64xf32>
    %v5043 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5044 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5045 = stablehlo.multiply %v5043, %s1b2g2v : tensor<64xf32>
    %v5046 = stablehlo.multiply %v3152, %v3152 : tensor<64xf32>
    %v5047 = stablehlo.multiply %v5044, %v5046 : tensor<64xf32>
    %v5048 = stablehlo.add %v5045, %v5047 : tensor<64xf32>
    %v5049 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5050 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5051 = stablehlo.multiply %v5049, %s1b2g2m : tensor<64xf32>
    %v5052 = stablehlo.multiply %v5050, %v3152 : tensor<64xf32>
    %v5053 = stablehlo.add %v5051, %v5052 : tensor<64xf32>
    %v5054 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5055 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5056 = stablehlo.multiply %v5054, %s1b2g2v : tensor<64xf32>
    %v5057 = stablehlo.multiply %v3152, %v3152 : tensor<64xf32>
    %v5058 = stablehlo.multiply %v5055, %v5057 : tensor<64xf32>
    %v5059 = stablehlo.add %v5056, %v5058 : tensor<64xf32>
    %v5060 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5061 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5062 = stablehlo.divide %v5053, %v5060 : tensor<64xf32>
    %v5063 = stablehlo.divide %v5059, %v5061 : tensor<64xf32>
    %v5064 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5065 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5066 = stablehlo.sqrt %v5063 : tensor<64xf32>
    %v5067 = stablehlo.add %v5066, %v5065 : tensor<64xf32>
    %v5068 = stablehlo.divide %v5062, %v5067 : tensor<64xf32>
    %v5069 = stablehlo.multiply %v5064, %v5068 : tensor<64xf32>
    %v5070 = stablehlo.subtract %s1b2g2, %v5069 : tensor<64xf32>
    %v5071 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5072 = stablehlo.multiply %v5071, %v5064 : tensor<64xf32>
    %v5073 = stablehlo.multiply %v5072, %s1b2g2 : tensor<64xf32>
    %v5074 = stablehlo.subtract %v5070, %v5073 : tensor<64xf32>
    %v5075 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5076 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5077 = stablehlo.multiply %v5075, %s1b2bt2m : tensor<64xf32>
    %v5078 = stablehlo.multiply %v5076, %v3155 : tensor<64xf32>
    %v5079 = stablehlo.add %v5077, %v5078 : tensor<64xf32>
    %v5080 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5081 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5082 = stablehlo.multiply %v5080, %s1b2bt2v : tensor<64xf32>
    %v5083 = stablehlo.multiply %v3155, %v3155 : tensor<64xf32>
    %v5084 = stablehlo.multiply %v5081, %v5083 : tensor<64xf32>
    %v5085 = stablehlo.add %v5082, %v5084 : tensor<64xf32>
    %v5086 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5087 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5088 = stablehlo.multiply %v5086, %s1b2bt2m : tensor<64xf32>
    %v5089 = stablehlo.multiply %v5087, %v3155 : tensor<64xf32>
    %v5090 = stablehlo.add %v5088, %v5089 : tensor<64xf32>
    %v5091 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5092 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5093 = stablehlo.multiply %v5091, %s1b2bt2v : tensor<64xf32>
    %v5094 = stablehlo.multiply %v3155, %v3155 : tensor<64xf32>
    %v5095 = stablehlo.multiply %v5092, %v5094 : tensor<64xf32>
    %v5096 = stablehlo.add %v5093, %v5095 : tensor<64xf32>
    %v5097 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5098 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5099 = stablehlo.divide %v5090, %v5097 : tensor<64xf32>
    %v5100 = stablehlo.divide %v5096, %v5098 : tensor<64xf32>
    %v5101 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5102 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5103 = stablehlo.sqrt %v5100 : tensor<64xf32>
    %v5104 = stablehlo.add %v5103, %v5102 : tensor<64xf32>
    %v5105 = stablehlo.divide %v5099, %v5104 : tensor<64xf32>
    %v5106 = stablehlo.multiply %v5101, %v5105 : tensor<64xf32>
    %v5107 = stablehlo.subtract %s1b2bt2, %v5106 : tensor<64xf32>
    %v5108 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5109 = stablehlo.multiply %v5108, %v5101 : tensor<64xf32>
    %v5110 = stablehlo.multiply %v5109, %s1b2bt2 : tensor<64xf32>
    %v5111 = stablehlo.subtract %v5107, %v5110 : tensor<64xf32>
    %v5112 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5113 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5114 = stablehlo.multiply %v5112, %d2W1m : tensor<128x64x3x3xf32>
    %v5115 = stablehlo.multiply %v5113, %v2932 : tensor<128x64x3x3xf32>
    %v5116 = stablehlo.add %v5114, %v5115 : tensor<128x64x3x3xf32>
    %v5117 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5118 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5119 = stablehlo.multiply %v5117, %d2W1v : tensor<128x64x3x3xf32>
    %v5120 = stablehlo.multiply %v2932, %v2932 : tensor<128x64x3x3xf32>
    %v5121 = stablehlo.multiply %v5118, %v5120 : tensor<128x64x3x3xf32>
    %v5122 = stablehlo.add %v5119, %v5121 : tensor<128x64x3x3xf32>
    %v5123 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5124 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5125 = stablehlo.multiply %v5123, %d2W1m : tensor<128x64x3x3xf32>
    %v5126 = stablehlo.multiply %v5124, %v2932 : tensor<128x64x3x3xf32>
    %v5127 = stablehlo.add %v5125, %v5126 : tensor<128x64x3x3xf32>
    %v5128 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5129 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5130 = stablehlo.multiply %v5128, %d2W1v : tensor<128x64x3x3xf32>
    %v5131 = stablehlo.multiply %v2932, %v2932 : tensor<128x64x3x3xf32>
    %v5132 = stablehlo.multiply %v5129, %v5131 : tensor<128x64x3x3xf32>
    %v5133 = stablehlo.add %v5130, %v5132 : tensor<128x64x3x3xf32>
    %v5134 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5135 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5136 = stablehlo.divide %v5127, %v5134 : tensor<128x64x3x3xf32>
    %v5137 = stablehlo.divide %v5133, %v5135 : tensor<128x64x3x3xf32>
    %v5138 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5139 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5140 = stablehlo.sqrt %v5137 : tensor<128x64x3x3xf32>
    %v5141 = stablehlo.add %v5140, %v5139 : tensor<128x64x3x3xf32>
    %v5142 = stablehlo.divide %v5136, %v5141 : tensor<128x64x3x3xf32>
    %v5143 = stablehlo.multiply %v5138, %v5142 : tensor<128x64x3x3xf32>
    %v5144 = stablehlo.subtract %d2W1, %v5143 : tensor<128x64x3x3xf32>
    %v5145 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5146 = stablehlo.multiply %v5145, %v5138 : tensor<128x64x3x3xf32>
    %v5147 = stablehlo.multiply %v5146, %d2W1 : tensor<128x64x3x3xf32>
    %v5148 = stablehlo.subtract %v5144, %v5147 : tensor<128x64x3x3xf32>
    %v5149 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5150 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5151 = stablehlo.multiply %v5149, %d2b1m : tensor<128xf32>
    %v5152 = stablehlo.multiply %v5150, %v2935 : tensor<128xf32>
    %v5153 = stablehlo.add %v5151, %v5152 : tensor<128xf32>
    %v5154 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5155 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5156 = stablehlo.multiply %v5154, %d2b1v : tensor<128xf32>
    %v5157 = stablehlo.multiply %v2935, %v2935 : tensor<128xf32>
    %v5158 = stablehlo.multiply %v5155, %v5157 : tensor<128xf32>
    %v5159 = stablehlo.add %v5156, %v5158 : tensor<128xf32>
    %v5160 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5161 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5162 = stablehlo.multiply %v5160, %d2b1m : tensor<128xf32>
    %v5163 = stablehlo.multiply %v5161, %v2935 : tensor<128xf32>
    %v5164 = stablehlo.add %v5162, %v5163 : tensor<128xf32>
    %v5165 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5166 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5167 = stablehlo.multiply %v5165, %d2b1v : tensor<128xf32>
    %v5168 = stablehlo.multiply %v2935, %v2935 : tensor<128xf32>
    %v5169 = stablehlo.multiply %v5166, %v5168 : tensor<128xf32>
    %v5170 = stablehlo.add %v5167, %v5169 : tensor<128xf32>
    %v5171 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5172 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5173 = stablehlo.divide %v5164, %v5171 : tensor<128xf32>
    %v5174 = stablehlo.divide %v5170, %v5172 : tensor<128xf32>
    %v5175 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5176 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5177 = stablehlo.sqrt %v5174 : tensor<128xf32>
    %v5178 = stablehlo.add %v5177, %v5176 : tensor<128xf32>
    %v5179 = stablehlo.divide %v5173, %v5178 : tensor<128xf32>
    %v5180 = stablehlo.multiply %v5175, %v5179 : tensor<128xf32>
    %v5181 = stablehlo.subtract %d2b1, %v5180 : tensor<128xf32>
    %v5182 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5183 = stablehlo.multiply %v5182, %v5175 : tensor<128xf32>
    %v5184 = stablehlo.multiply %v5183, %d2b1 : tensor<128xf32>
    %v5185 = stablehlo.subtract %v5181, %v5184 : tensor<128xf32>
    %v5186 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5187 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5188 = stablehlo.multiply %v5186, %d2g1m : tensor<128xf32>
    %v5189 = stablehlo.multiply %v5187, %v2953 : tensor<128xf32>
    %v5190 = stablehlo.add %v5188, %v5189 : tensor<128xf32>
    %v5191 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5192 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5193 = stablehlo.multiply %v5191, %d2g1v : tensor<128xf32>
    %v5194 = stablehlo.multiply %v2953, %v2953 : tensor<128xf32>
    %v5195 = stablehlo.multiply %v5192, %v5194 : tensor<128xf32>
    %v5196 = stablehlo.add %v5193, %v5195 : tensor<128xf32>
    %v5197 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5198 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5199 = stablehlo.multiply %v5197, %d2g1m : tensor<128xf32>
    %v5200 = stablehlo.multiply %v5198, %v2953 : tensor<128xf32>
    %v5201 = stablehlo.add %v5199, %v5200 : tensor<128xf32>
    %v5202 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5203 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5204 = stablehlo.multiply %v5202, %d2g1v : tensor<128xf32>
    %v5205 = stablehlo.multiply %v2953, %v2953 : tensor<128xf32>
    %v5206 = stablehlo.multiply %v5203, %v5205 : tensor<128xf32>
    %v5207 = stablehlo.add %v5204, %v5206 : tensor<128xf32>
    %v5208 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5209 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5210 = stablehlo.divide %v5201, %v5208 : tensor<128xf32>
    %v5211 = stablehlo.divide %v5207, %v5209 : tensor<128xf32>
    %v5212 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5213 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5214 = stablehlo.sqrt %v5211 : tensor<128xf32>
    %v5215 = stablehlo.add %v5214, %v5213 : tensor<128xf32>
    %v5216 = stablehlo.divide %v5210, %v5215 : tensor<128xf32>
    %v5217 = stablehlo.multiply %v5212, %v5216 : tensor<128xf32>
    %v5218 = stablehlo.subtract %d2g1, %v5217 : tensor<128xf32>
    %v5219 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5220 = stablehlo.multiply %v5219, %v5212 : tensor<128xf32>
    %v5221 = stablehlo.multiply %v5220, %d2g1 : tensor<128xf32>
    %v5222 = stablehlo.subtract %v5218, %v5221 : tensor<128xf32>
    %v5223 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5224 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5225 = stablehlo.multiply %v5223, %d2bt1m : tensor<128xf32>
    %v5226 = stablehlo.multiply %v5224, %v2956 : tensor<128xf32>
    %v5227 = stablehlo.add %v5225, %v5226 : tensor<128xf32>
    %v5228 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5229 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5230 = stablehlo.multiply %v5228, %d2bt1v : tensor<128xf32>
    %v5231 = stablehlo.multiply %v2956, %v2956 : tensor<128xf32>
    %v5232 = stablehlo.multiply %v5229, %v5231 : tensor<128xf32>
    %v5233 = stablehlo.add %v5230, %v5232 : tensor<128xf32>
    %v5234 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5235 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5236 = stablehlo.multiply %v5234, %d2bt1m : tensor<128xf32>
    %v5237 = stablehlo.multiply %v5235, %v2956 : tensor<128xf32>
    %v5238 = stablehlo.add %v5236, %v5237 : tensor<128xf32>
    %v5239 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5240 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5241 = stablehlo.multiply %v5239, %d2bt1v : tensor<128xf32>
    %v5242 = stablehlo.multiply %v2956, %v2956 : tensor<128xf32>
    %v5243 = stablehlo.multiply %v5240, %v5242 : tensor<128xf32>
    %v5244 = stablehlo.add %v5241, %v5243 : tensor<128xf32>
    %v5245 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5246 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5247 = stablehlo.divide %v5238, %v5245 : tensor<128xf32>
    %v5248 = stablehlo.divide %v5244, %v5246 : tensor<128xf32>
    %v5249 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5250 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5251 = stablehlo.sqrt %v5248 : tensor<128xf32>
    %v5252 = stablehlo.add %v5251, %v5250 : tensor<128xf32>
    %v5253 = stablehlo.divide %v5247, %v5252 : tensor<128xf32>
    %v5254 = stablehlo.multiply %v5249, %v5253 : tensor<128xf32>
    %v5255 = stablehlo.subtract %d2bt1, %v5254 : tensor<128xf32>
    %v5256 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5257 = stablehlo.multiply %v5256, %v5249 : tensor<128xf32>
    %v5258 = stablehlo.multiply %v5257, %d2bt1 : tensor<128xf32>
    %v5259 = stablehlo.subtract %v5255, %v5258 : tensor<128xf32>
    %v5260 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5261 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5262 = stablehlo.multiply %v5260, %d2W2m : tensor<128x128x3x3xf32>
    %v5263 = stablehlo.multiply %v5261, %v2962 : tensor<128x128x3x3xf32>
    %v5264 = stablehlo.add %v5262, %v5263 : tensor<128x128x3x3xf32>
    %v5265 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5266 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5267 = stablehlo.multiply %v5265, %d2W2v : tensor<128x128x3x3xf32>
    %v5268 = stablehlo.multiply %v2962, %v2962 : tensor<128x128x3x3xf32>
    %v5269 = stablehlo.multiply %v5266, %v5268 : tensor<128x128x3x3xf32>
    %v5270 = stablehlo.add %v5267, %v5269 : tensor<128x128x3x3xf32>
    %v5271 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5272 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5273 = stablehlo.multiply %v5271, %d2W2m : tensor<128x128x3x3xf32>
    %v5274 = stablehlo.multiply %v5272, %v2962 : tensor<128x128x3x3xf32>
    %v5275 = stablehlo.add %v5273, %v5274 : tensor<128x128x3x3xf32>
    %v5276 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5277 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5278 = stablehlo.multiply %v5276, %d2W2v : tensor<128x128x3x3xf32>
    %v5279 = stablehlo.multiply %v2962, %v2962 : tensor<128x128x3x3xf32>
    %v5280 = stablehlo.multiply %v5277, %v5279 : tensor<128x128x3x3xf32>
    %v5281 = stablehlo.add %v5278, %v5280 : tensor<128x128x3x3xf32>
    %v5282 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5283 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5284 = stablehlo.divide %v5275, %v5282 : tensor<128x128x3x3xf32>
    %v5285 = stablehlo.divide %v5281, %v5283 : tensor<128x128x3x3xf32>
    %v5286 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5287 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5288 = stablehlo.sqrt %v5285 : tensor<128x128x3x3xf32>
    %v5289 = stablehlo.add %v5288, %v5287 : tensor<128x128x3x3xf32>
    %v5290 = stablehlo.divide %v5284, %v5289 : tensor<128x128x3x3xf32>
    %v5291 = stablehlo.multiply %v5286, %v5290 : tensor<128x128x3x3xf32>
    %v5292 = stablehlo.subtract %d2W2, %v5291 : tensor<128x128x3x3xf32>
    %v5293 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5294 = stablehlo.multiply %v5293, %v5286 : tensor<128x128x3x3xf32>
    %v5295 = stablehlo.multiply %v5294, %d2W2 : tensor<128x128x3x3xf32>
    %v5296 = stablehlo.subtract %v5292, %v5295 : tensor<128x128x3x3xf32>
    %v5297 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5298 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5299 = stablehlo.multiply %v5297, %d2b2m : tensor<128xf32>
    %v5300 = stablehlo.multiply %v5298, %v2965 : tensor<128xf32>
    %v5301 = stablehlo.add %v5299, %v5300 : tensor<128xf32>
    %v5302 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5303 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5304 = stablehlo.multiply %v5302, %d2b2v : tensor<128xf32>
    %v5305 = stablehlo.multiply %v2965, %v2965 : tensor<128xf32>
    %v5306 = stablehlo.multiply %v5303, %v5305 : tensor<128xf32>
    %v5307 = stablehlo.add %v5304, %v5306 : tensor<128xf32>
    %v5308 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5309 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5310 = stablehlo.multiply %v5308, %d2b2m : tensor<128xf32>
    %v5311 = stablehlo.multiply %v5309, %v2965 : tensor<128xf32>
    %v5312 = stablehlo.add %v5310, %v5311 : tensor<128xf32>
    %v5313 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5314 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5315 = stablehlo.multiply %v5313, %d2b2v : tensor<128xf32>
    %v5316 = stablehlo.multiply %v2965, %v2965 : tensor<128xf32>
    %v5317 = stablehlo.multiply %v5314, %v5316 : tensor<128xf32>
    %v5318 = stablehlo.add %v5315, %v5317 : tensor<128xf32>
    %v5319 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5320 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5321 = stablehlo.divide %v5312, %v5319 : tensor<128xf32>
    %v5322 = stablehlo.divide %v5318, %v5320 : tensor<128xf32>
    %v5323 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5324 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5325 = stablehlo.sqrt %v5322 : tensor<128xf32>
    %v5326 = stablehlo.add %v5325, %v5324 : tensor<128xf32>
    %v5327 = stablehlo.divide %v5321, %v5326 : tensor<128xf32>
    %v5328 = stablehlo.multiply %v5323, %v5327 : tensor<128xf32>
    %v5329 = stablehlo.subtract %d2b2, %v5328 : tensor<128xf32>
    %v5330 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5331 = stablehlo.multiply %v5330, %v5323 : tensor<128xf32>
    %v5332 = stablehlo.multiply %v5331, %d2b2 : tensor<128xf32>
    %v5333 = stablehlo.subtract %v5329, %v5332 : tensor<128xf32>
    %v5334 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5335 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5336 = stablehlo.multiply %v5334, %d2g2m : tensor<128xf32>
    %v5337 = stablehlo.multiply %v5335, %v2983 : tensor<128xf32>
    %v5338 = stablehlo.add %v5336, %v5337 : tensor<128xf32>
    %v5339 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5340 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5341 = stablehlo.multiply %v5339, %d2g2v : tensor<128xf32>
    %v5342 = stablehlo.multiply %v2983, %v2983 : tensor<128xf32>
    %v5343 = stablehlo.multiply %v5340, %v5342 : tensor<128xf32>
    %v5344 = stablehlo.add %v5341, %v5343 : tensor<128xf32>
    %v5345 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5346 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5347 = stablehlo.multiply %v5345, %d2g2m : tensor<128xf32>
    %v5348 = stablehlo.multiply %v5346, %v2983 : tensor<128xf32>
    %v5349 = stablehlo.add %v5347, %v5348 : tensor<128xf32>
    %v5350 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5351 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5352 = stablehlo.multiply %v5350, %d2g2v : tensor<128xf32>
    %v5353 = stablehlo.multiply %v2983, %v2983 : tensor<128xf32>
    %v5354 = stablehlo.multiply %v5351, %v5353 : tensor<128xf32>
    %v5355 = stablehlo.add %v5352, %v5354 : tensor<128xf32>
    %v5356 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5357 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5358 = stablehlo.divide %v5349, %v5356 : tensor<128xf32>
    %v5359 = stablehlo.divide %v5355, %v5357 : tensor<128xf32>
    %v5360 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5361 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5362 = stablehlo.sqrt %v5359 : tensor<128xf32>
    %v5363 = stablehlo.add %v5362, %v5361 : tensor<128xf32>
    %v5364 = stablehlo.divide %v5358, %v5363 : tensor<128xf32>
    %v5365 = stablehlo.multiply %v5360, %v5364 : tensor<128xf32>
    %v5366 = stablehlo.subtract %d2g2, %v5365 : tensor<128xf32>
    %v5367 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5368 = stablehlo.multiply %v5367, %v5360 : tensor<128xf32>
    %v5369 = stablehlo.multiply %v5368, %d2g2 : tensor<128xf32>
    %v5370 = stablehlo.subtract %v5366, %v5369 : tensor<128xf32>
    %v5371 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5372 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5373 = stablehlo.multiply %v5371, %d2bt2m : tensor<128xf32>
    %v5374 = stablehlo.multiply %v5372, %v2986 : tensor<128xf32>
    %v5375 = stablehlo.add %v5373, %v5374 : tensor<128xf32>
    %v5376 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5377 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5378 = stablehlo.multiply %v5376, %d2bt2v : tensor<128xf32>
    %v5379 = stablehlo.multiply %v2986, %v2986 : tensor<128xf32>
    %v5380 = stablehlo.multiply %v5377, %v5379 : tensor<128xf32>
    %v5381 = stablehlo.add %v5378, %v5380 : tensor<128xf32>
    %v5382 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5383 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5384 = stablehlo.multiply %v5382, %d2bt2m : tensor<128xf32>
    %v5385 = stablehlo.multiply %v5383, %v2986 : tensor<128xf32>
    %v5386 = stablehlo.add %v5384, %v5385 : tensor<128xf32>
    %v5387 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5388 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5389 = stablehlo.multiply %v5387, %d2bt2v : tensor<128xf32>
    %v5390 = stablehlo.multiply %v2986, %v2986 : tensor<128xf32>
    %v5391 = stablehlo.multiply %v5388, %v5390 : tensor<128xf32>
    %v5392 = stablehlo.add %v5389, %v5391 : tensor<128xf32>
    %v5393 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5394 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5395 = stablehlo.divide %v5386, %v5393 : tensor<128xf32>
    %v5396 = stablehlo.divide %v5392, %v5394 : tensor<128xf32>
    %v5397 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5398 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5399 = stablehlo.sqrt %v5396 : tensor<128xf32>
    %v5400 = stablehlo.add %v5399, %v5398 : tensor<128xf32>
    %v5401 = stablehlo.divide %v5395, %v5400 : tensor<128xf32>
    %v5402 = stablehlo.multiply %v5397, %v5401 : tensor<128xf32>
    %v5403 = stablehlo.subtract %d2bt2, %v5402 : tensor<128xf32>
    %v5404 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5405 = stablehlo.multiply %v5404, %v5397 : tensor<128xf32>
    %v5406 = stablehlo.multiply %v5405, %d2bt2 : tensor<128xf32>
    %v5407 = stablehlo.subtract %v5403, %v5406 : tensor<128xf32>
    %v5408 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5409 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5410 = stablehlo.multiply %v5408, %d2Wpm : tensor<128x64x3x3xf32>
    %v5411 = stablehlo.multiply %v5409, %v2994 : tensor<128x64x3x3xf32>
    %v5412 = stablehlo.add %v5410, %v5411 : tensor<128x64x3x3xf32>
    %v5413 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5414 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5415 = stablehlo.multiply %v5413, %d2Wpv : tensor<128x64x3x3xf32>
    %v5416 = stablehlo.multiply %v2994, %v2994 : tensor<128x64x3x3xf32>
    %v5417 = stablehlo.multiply %v5414, %v5416 : tensor<128x64x3x3xf32>
    %v5418 = stablehlo.add %v5415, %v5417 : tensor<128x64x3x3xf32>
    %v5419 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5420 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5421 = stablehlo.multiply %v5419, %d2Wpm : tensor<128x64x3x3xf32>
    %v5422 = stablehlo.multiply %v5420, %v2994 : tensor<128x64x3x3xf32>
    %v5423 = stablehlo.add %v5421, %v5422 : tensor<128x64x3x3xf32>
    %v5424 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5425 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5426 = stablehlo.multiply %v5424, %d2Wpv : tensor<128x64x3x3xf32>
    %v5427 = stablehlo.multiply %v2994, %v2994 : tensor<128x64x3x3xf32>
    %v5428 = stablehlo.multiply %v5425, %v5427 : tensor<128x64x3x3xf32>
    %v5429 = stablehlo.add %v5426, %v5428 : tensor<128x64x3x3xf32>
    %v5430 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5431 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5432 = stablehlo.divide %v5423, %v5430 : tensor<128x64x3x3xf32>
    %v5433 = stablehlo.divide %v5429, %v5431 : tensor<128x64x3x3xf32>
    %v5434 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5435 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5436 = stablehlo.sqrt %v5433 : tensor<128x64x3x3xf32>
    %v5437 = stablehlo.add %v5436, %v5435 : tensor<128x64x3x3xf32>
    %v5438 = stablehlo.divide %v5432, %v5437 : tensor<128x64x3x3xf32>
    %v5439 = stablehlo.multiply %v5434, %v5438 : tensor<128x64x3x3xf32>
    %v5440 = stablehlo.subtract %d2Wp, %v5439 : tensor<128x64x3x3xf32>
    %v5441 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5442 = stablehlo.multiply %v5441, %v5434 : tensor<128x64x3x3xf32>
    %v5443 = stablehlo.multiply %v5442, %d2Wp : tensor<128x64x3x3xf32>
    %v5444 = stablehlo.subtract %v5440, %v5443 : tensor<128x64x3x3xf32>
    %v5445 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5446 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5447 = stablehlo.multiply %v5445, %d2bpm : tensor<128xf32>
    %v5448 = stablehlo.multiply %v5446, %v2997 : tensor<128xf32>
    %v5449 = stablehlo.add %v5447, %v5448 : tensor<128xf32>
    %v5450 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5451 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5452 = stablehlo.multiply %v5450, %d2bpv : tensor<128xf32>
    %v5453 = stablehlo.multiply %v2997, %v2997 : tensor<128xf32>
    %v5454 = stablehlo.multiply %v5451, %v5453 : tensor<128xf32>
    %v5455 = stablehlo.add %v5452, %v5454 : tensor<128xf32>
    %v5456 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5457 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5458 = stablehlo.multiply %v5456, %d2bpm : tensor<128xf32>
    %v5459 = stablehlo.multiply %v5457, %v2997 : tensor<128xf32>
    %v5460 = stablehlo.add %v5458, %v5459 : tensor<128xf32>
    %v5461 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5462 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5463 = stablehlo.multiply %v5461, %d2bpv : tensor<128xf32>
    %v5464 = stablehlo.multiply %v2997, %v2997 : tensor<128xf32>
    %v5465 = stablehlo.multiply %v5462, %v5464 : tensor<128xf32>
    %v5466 = stablehlo.add %v5463, %v5465 : tensor<128xf32>
    %v5467 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5468 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5469 = stablehlo.divide %v5460, %v5467 : tensor<128xf32>
    %v5470 = stablehlo.divide %v5466, %v5468 : tensor<128xf32>
    %v5471 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5472 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5473 = stablehlo.sqrt %v5470 : tensor<128xf32>
    %v5474 = stablehlo.add %v5473, %v5472 : tensor<128xf32>
    %v5475 = stablehlo.divide %v5469, %v5474 : tensor<128xf32>
    %v5476 = stablehlo.multiply %v5471, %v5475 : tensor<128xf32>
    %v5477 = stablehlo.subtract %d2bp, %v5476 : tensor<128xf32>
    %v5478 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5479 = stablehlo.multiply %v5478, %v5471 : tensor<128xf32>
    %v5480 = stablehlo.multiply %v5479, %d2bp : tensor<128xf32>
    %v5481 = stablehlo.subtract %v5477, %v5480 : tensor<128xf32>
    %v5482 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5483 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5484 = stablehlo.multiply %v5482, %d2gpm : tensor<128xf32>
    %v5485 = stablehlo.multiply %v5483, %v3015 : tensor<128xf32>
    %v5486 = stablehlo.add %v5484, %v5485 : tensor<128xf32>
    %v5487 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5488 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5489 = stablehlo.multiply %v5487, %d2gpv : tensor<128xf32>
    %v5490 = stablehlo.multiply %v3015, %v3015 : tensor<128xf32>
    %v5491 = stablehlo.multiply %v5488, %v5490 : tensor<128xf32>
    %v5492 = stablehlo.add %v5489, %v5491 : tensor<128xf32>
    %v5493 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5494 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5495 = stablehlo.multiply %v5493, %d2gpm : tensor<128xf32>
    %v5496 = stablehlo.multiply %v5494, %v3015 : tensor<128xf32>
    %v5497 = stablehlo.add %v5495, %v5496 : tensor<128xf32>
    %v5498 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5499 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5500 = stablehlo.multiply %v5498, %d2gpv : tensor<128xf32>
    %v5501 = stablehlo.multiply %v3015, %v3015 : tensor<128xf32>
    %v5502 = stablehlo.multiply %v5499, %v5501 : tensor<128xf32>
    %v5503 = stablehlo.add %v5500, %v5502 : tensor<128xf32>
    %v5504 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5505 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5506 = stablehlo.divide %v5497, %v5504 : tensor<128xf32>
    %v5507 = stablehlo.divide %v5503, %v5505 : tensor<128xf32>
    %v5508 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5509 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5510 = stablehlo.sqrt %v5507 : tensor<128xf32>
    %v5511 = stablehlo.add %v5510, %v5509 : tensor<128xf32>
    %v5512 = stablehlo.divide %v5506, %v5511 : tensor<128xf32>
    %v5513 = stablehlo.multiply %v5508, %v5512 : tensor<128xf32>
    %v5514 = stablehlo.subtract %d2gp, %v5513 : tensor<128xf32>
    %v5515 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5516 = stablehlo.multiply %v5515, %v5508 : tensor<128xf32>
    %v5517 = stablehlo.multiply %v5516, %d2gp : tensor<128xf32>
    %v5518 = stablehlo.subtract %v5514, %v5517 : tensor<128xf32>
    %v5519 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5520 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5521 = stablehlo.multiply %v5519, %d2btpm : tensor<128xf32>
    %v5522 = stablehlo.multiply %v5520, %v3018 : tensor<128xf32>
    %v5523 = stablehlo.add %v5521, %v5522 : tensor<128xf32>
    %v5524 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5525 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5526 = stablehlo.multiply %v5524, %d2btpv : tensor<128xf32>
    %v5527 = stablehlo.multiply %v3018, %v3018 : tensor<128xf32>
    %v5528 = stablehlo.multiply %v5525, %v5527 : tensor<128xf32>
    %v5529 = stablehlo.add %v5526, %v5528 : tensor<128xf32>
    %v5530 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5531 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5532 = stablehlo.multiply %v5530, %d2btpm : tensor<128xf32>
    %v5533 = stablehlo.multiply %v5531, %v3018 : tensor<128xf32>
    %v5534 = stablehlo.add %v5532, %v5533 : tensor<128xf32>
    %v5535 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5536 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5537 = stablehlo.multiply %v5535, %d2btpv : tensor<128xf32>
    %v5538 = stablehlo.multiply %v3018, %v3018 : tensor<128xf32>
    %v5539 = stablehlo.multiply %v5536, %v5538 : tensor<128xf32>
    %v5540 = stablehlo.add %v5537, %v5539 : tensor<128xf32>
    %v5541 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5542 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5543 = stablehlo.divide %v5534, %v5541 : tensor<128xf32>
    %v5544 = stablehlo.divide %v5540, %v5542 : tensor<128xf32>
    %v5545 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5546 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5547 = stablehlo.sqrt %v5544 : tensor<128xf32>
    %v5548 = stablehlo.add %v5547, %v5546 : tensor<128xf32>
    %v5549 = stablehlo.divide %v5543, %v5548 : tensor<128xf32>
    %v5550 = stablehlo.multiply %v5545, %v5549 : tensor<128xf32>
    %v5551 = stablehlo.subtract %d2btp, %v5550 : tensor<128xf32>
    %v5552 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5553 = stablehlo.multiply %v5552, %v5545 : tensor<128xf32>
    %v5554 = stablehlo.multiply %v5553, %d2btp : tensor<128xf32>
    %v5555 = stablehlo.subtract %v5551, %v5554 : tensor<128xf32>
    %v5556 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5557 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5558 = stablehlo.multiply %v5556, %s2b0W1m : tensor<128x128x3x3xf32>
    %v5559 = stablehlo.multiply %v5557, %v2754 : tensor<128x128x3x3xf32>
    %v5560 = stablehlo.add %v5558, %v5559 : tensor<128x128x3x3xf32>
    %v5561 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5562 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5563 = stablehlo.multiply %v5561, %s2b0W1v : tensor<128x128x3x3xf32>
    %v5564 = stablehlo.multiply %v2754, %v2754 : tensor<128x128x3x3xf32>
    %v5565 = stablehlo.multiply %v5562, %v5564 : tensor<128x128x3x3xf32>
    %v5566 = stablehlo.add %v5563, %v5565 : tensor<128x128x3x3xf32>
    %v5567 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5568 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5569 = stablehlo.multiply %v5567, %s2b0W1m : tensor<128x128x3x3xf32>
    %v5570 = stablehlo.multiply %v5568, %v2754 : tensor<128x128x3x3xf32>
    %v5571 = stablehlo.add %v5569, %v5570 : tensor<128x128x3x3xf32>
    %v5572 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5573 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5574 = stablehlo.multiply %v5572, %s2b0W1v : tensor<128x128x3x3xf32>
    %v5575 = stablehlo.multiply %v2754, %v2754 : tensor<128x128x3x3xf32>
    %v5576 = stablehlo.multiply %v5573, %v5575 : tensor<128x128x3x3xf32>
    %v5577 = stablehlo.add %v5574, %v5576 : tensor<128x128x3x3xf32>
    %v5578 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5579 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5580 = stablehlo.divide %v5571, %v5578 : tensor<128x128x3x3xf32>
    %v5581 = stablehlo.divide %v5577, %v5579 : tensor<128x128x3x3xf32>
    %v5582 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5583 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5584 = stablehlo.sqrt %v5581 : tensor<128x128x3x3xf32>
    %v5585 = stablehlo.add %v5584, %v5583 : tensor<128x128x3x3xf32>
    %v5586 = stablehlo.divide %v5580, %v5585 : tensor<128x128x3x3xf32>
    %v5587 = stablehlo.multiply %v5582, %v5586 : tensor<128x128x3x3xf32>
    %v5588 = stablehlo.subtract %s2b0W1, %v5587 : tensor<128x128x3x3xf32>
    %v5589 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5590 = stablehlo.multiply %v5589, %v5582 : tensor<128x128x3x3xf32>
    %v5591 = stablehlo.multiply %v5590, %s2b0W1 : tensor<128x128x3x3xf32>
    %v5592 = stablehlo.subtract %v5588, %v5591 : tensor<128x128x3x3xf32>
    %v5593 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5594 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5595 = stablehlo.multiply %v5593, %s2b0b1m : tensor<128xf32>
    %v5596 = stablehlo.multiply %v5594, %v2757 : tensor<128xf32>
    %v5597 = stablehlo.add %v5595, %v5596 : tensor<128xf32>
    %v5598 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5599 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5600 = stablehlo.multiply %v5598, %s2b0b1v : tensor<128xf32>
    %v5601 = stablehlo.multiply %v2757, %v2757 : tensor<128xf32>
    %v5602 = stablehlo.multiply %v5599, %v5601 : tensor<128xf32>
    %v5603 = stablehlo.add %v5600, %v5602 : tensor<128xf32>
    %v5604 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5605 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5606 = stablehlo.multiply %v5604, %s2b0b1m : tensor<128xf32>
    %v5607 = stablehlo.multiply %v5605, %v2757 : tensor<128xf32>
    %v5608 = stablehlo.add %v5606, %v5607 : tensor<128xf32>
    %v5609 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5610 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5611 = stablehlo.multiply %v5609, %s2b0b1v : tensor<128xf32>
    %v5612 = stablehlo.multiply %v2757, %v2757 : tensor<128xf32>
    %v5613 = stablehlo.multiply %v5610, %v5612 : tensor<128xf32>
    %v5614 = stablehlo.add %v5611, %v5613 : tensor<128xf32>
    %v5615 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5616 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5617 = stablehlo.divide %v5608, %v5615 : tensor<128xf32>
    %v5618 = stablehlo.divide %v5614, %v5616 : tensor<128xf32>
    %v5619 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5620 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5621 = stablehlo.sqrt %v5618 : tensor<128xf32>
    %v5622 = stablehlo.add %v5621, %v5620 : tensor<128xf32>
    %v5623 = stablehlo.divide %v5617, %v5622 : tensor<128xf32>
    %v5624 = stablehlo.multiply %v5619, %v5623 : tensor<128xf32>
    %v5625 = stablehlo.subtract %s2b0b1, %v5624 : tensor<128xf32>
    %v5626 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5627 = stablehlo.multiply %v5626, %v5619 : tensor<128xf32>
    %v5628 = stablehlo.multiply %v5627, %s2b0b1 : tensor<128xf32>
    %v5629 = stablehlo.subtract %v5625, %v5628 : tensor<128xf32>
    %v5630 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5631 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5632 = stablehlo.multiply %v5630, %s2b0g1m : tensor<128xf32>
    %v5633 = stablehlo.multiply %v5631, %v2775 : tensor<128xf32>
    %v5634 = stablehlo.add %v5632, %v5633 : tensor<128xf32>
    %v5635 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5636 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5637 = stablehlo.multiply %v5635, %s2b0g1v : tensor<128xf32>
    %v5638 = stablehlo.multiply %v2775, %v2775 : tensor<128xf32>
    %v5639 = stablehlo.multiply %v5636, %v5638 : tensor<128xf32>
    %v5640 = stablehlo.add %v5637, %v5639 : tensor<128xf32>
    %v5641 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5642 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5643 = stablehlo.multiply %v5641, %s2b0g1m : tensor<128xf32>
    %v5644 = stablehlo.multiply %v5642, %v2775 : tensor<128xf32>
    %v5645 = stablehlo.add %v5643, %v5644 : tensor<128xf32>
    %v5646 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5647 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5648 = stablehlo.multiply %v5646, %s2b0g1v : tensor<128xf32>
    %v5649 = stablehlo.multiply %v2775, %v2775 : tensor<128xf32>
    %v5650 = stablehlo.multiply %v5647, %v5649 : tensor<128xf32>
    %v5651 = stablehlo.add %v5648, %v5650 : tensor<128xf32>
    %v5652 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5653 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5654 = stablehlo.divide %v5645, %v5652 : tensor<128xf32>
    %v5655 = stablehlo.divide %v5651, %v5653 : tensor<128xf32>
    %v5656 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5657 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5658 = stablehlo.sqrt %v5655 : tensor<128xf32>
    %v5659 = stablehlo.add %v5658, %v5657 : tensor<128xf32>
    %v5660 = stablehlo.divide %v5654, %v5659 : tensor<128xf32>
    %v5661 = stablehlo.multiply %v5656, %v5660 : tensor<128xf32>
    %v5662 = stablehlo.subtract %s2b0g1, %v5661 : tensor<128xf32>
    %v5663 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5664 = stablehlo.multiply %v5663, %v5656 : tensor<128xf32>
    %v5665 = stablehlo.multiply %v5664, %s2b0g1 : tensor<128xf32>
    %v5666 = stablehlo.subtract %v5662, %v5665 : tensor<128xf32>
    %v5667 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5668 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5669 = stablehlo.multiply %v5667, %s2b0bt1m : tensor<128xf32>
    %v5670 = stablehlo.multiply %v5668, %v2778 : tensor<128xf32>
    %v5671 = stablehlo.add %v5669, %v5670 : tensor<128xf32>
    %v5672 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5673 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5674 = stablehlo.multiply %v5672, %s2b0bt1v : tensor<128xf32>
    %v5675 = stablehlo.multiply %v2778, %v2778 : tensor<128xf32>
    %v5676 = stablehlo.multiply %v5673, %v5675 : tensor<128xf32>
    %v5677 = stablehlo.add %v5674, %v5676 : tensor<128xf32>
    %v5678 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5679 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5680 = stablehlo.multiply %v5678, %s2b0bt1m : tensor<128xf32>
    %v5681 = stablehlo.multiply %v5679, %v2778 : tensor<128xf32>
    %v5682 = stablehlo.add %v5680, %v5681 : tensor<128xf32>
    %v5683 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5684 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5685 = stablehlo.multiply %v5683, %s2b0bt1v : tensor<128xf32>
    %v5686 = stablehlo.multiply %v2778, %v2778 : tensor<128xf32>
    %v5687 = stablehlo.multiply %v5684, %v5686 : tensor<128xf32>
    %v5688 = stablehlo.add %v5685, %v5687 : tensor<128xf32>
    %v5689 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5690 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5691 = stablehlo.divide %v5682, %v5689 : tensor<128xf32>
    %v5692 = stablehlo.divide %v5688, %v5690 : tensor<128xf32>
    %v5693 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5694 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5695 = stablehlo.sqrt %v5692 : tensor<128xf32>
    %v5696 = stablehlo.add %v5695, %v5694 : tensor<128xf32>
    %v5697 = stablehlo.divide %v5691, %v5696 : tensor<128xf32>
    %v5698 = stablehlo.multiply %v5693, %v5697 : tensor<128xf32>
    %v5699 = stablehlo.subtract %s2b0bt1, %v5698 : tensor<128xf32>
    %v5700 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5701 = stablehlo.multiply %v5700, %v5693 : tensor<128xf32>
    %v5702 = stablehlo.multiply %v5701, %s2b0bt1 : tensor<128xf32>
    %v5703 = stablehlo.subtract %v5699, %v5702 : tensor<128xf32>
    %v5704 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5705 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5706 = stablehlo.multiply %v5704, %s2b0W2m : tensor<128x128x3x3xf32>
    %v5707 = stablehlo.multiply %v5705, %v2784 : tensor<128x128x3x3xf32>
    %v5708 = stablehlo.add %v5706, %v5707 : tensor<128x128x3x3xf32>
    %v5709 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5710 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5711 = stablehlo.multiply %v5709, %s2b0W2v : tensor<128x128x3x3xf32>
    %v5712 = stablehlo.multiply %v2784, %v2784 : tensor<128x128x3x3xf32>
    %v5713 = stablehlo.multiply %v5710, %v5712 : tensor<128x128x3x3xf32>
    %v5714 = stablehlo.add %v5711, %v5713 : tensor<128x128x3x3xf32>
    %v5715 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5716 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5717 = stablehlo.multiply %v5715, %s2b0W2m : tensor<128x128x3x3xf32>
    %v5718 = stablehlo.multiply %v5716, %v2784 : tensor<128x128x3x3xf32>
    %v5719 = stablehlo.add %v5717, %v5718 : tensor<128x128x3x3xf32>
    %v5720 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5721 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5722 = stablehlo.multiply %v5720, %s2b0W2v : tensor<128x128x3x3xf32>
    %v5723 = stablehlo.multiply %v2784, %v2784 : tensor<128x128x3x3xf32>
    %v5724 = stablehlo.multiply %v5721, %v5723 : tensor<128x128x3x3xf32>
    %v5725 = stablehlo.add %v5722, %v5724 : tensor<128x128x3x3xf32>
    %v5726 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5727 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5728 = stablehlo.divide %v5719, %v5726 : tensor<128x128x3x3xf32>
    %v5729 = stablehlo.divide %v5725, %v5727 : tensor<128x128x3x3xf32>
    %v5730 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5731 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5732 = stablehlo.sqrt %v5729 : tensor<128x128x3x3xf32>
    %v5733 = stablehlo.add %v5732, %v5731 : tensor<128x128x3x3xf32>
    %v5734 = stablehlo.divide %v5728, %v5733 : tensor<128x128x3x3xf32>
    %v5735 = stablehlo.multiply %v5730, %v5734 : tensor<128x128x3x3xf32>
    %v5736 = stablehlo.subtract %s2b0W2, %v5735 : tensor<128x128x3x3xf32>
    %v5737 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5738 = stablehlo.multiply %v5737, %v5730 : tensor<128x128x3x3xf32>
    %v5739 = stablehlo.multiply %v5738, %s2b0W2 : tensor<128x128x3x3xf32>
    %v5740 = stablehlo.subtract %v5736, %v5739 : tensor<128x128x3x3xf32>
    %v5741 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5742 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5743 = stablehlo.multiply %v5741, %s2b0b2m : tensor<128xf32>
    %v5744 = stablehlo.multiply %v5742, %v2787 : tensor<128xf32>
    %v5745 = stablehlo.add %v5743, %v5744 : tensor<128xf32>
    %v5746 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5747 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5748 = stablehlo.multiply %v5746, %s2b0b2v : tensor<128xf32>
    %v5749 = stablehlo.multiply %v2787, %v2787 : tensor<128xf32>
    %v5750 = stablehlo.multiply %v5747, %v5749 : tensor<128xf32>
    %v5751 = stablehlo.add %v5748, %v5750 : tensor<128xf32>
    %v5752 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5753 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5754 = stablehlo.multiply %v5752, %s2b0b2m : tensor<128xf32>
    %v5755 = stablehlo.multiply %v5753, %v2787 : tensor<128xf32>
    %v5756 = stablehlo.add %v5754, %v5755 : tensor<128xf32>
    %v5757 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5758 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5759 = stablehlo.multiply %v5757, %s2b0b2v : tensor<128xf32>
    %v5760 = stablehlo.multiply %v2787, %v2787 : tensor<128xf32>
    %v5761 = stablehlo.multiply %v5758, %v5760 : tensor<128xf32>
    %v5762 = stablehlo.add %v5759, %v5761 : tensor<128xf32>
    %v5763 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5764 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5765 = stablehlo.divide %v5756, %v5763 : tensor<128xf32>
    %v5766 = stablehlo.divide %v5762, %v5764 : tensor<128xf32>
    %v5767 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5768 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5769 = stablehlo.sqrt %v5766 : tensor<128xf32>
    %v5770 = stablehlo.add %v5769, %v5768 : tensor<128xf32>
    %v5771 = stablehlo.divide %v5765, %v5770 : tensor<128xf32>
    %v5772 = stablehlo.multiply %v5767, %v5771 : tensor<128xf32>
    %v5773 = stablehlo.subtract %s2b0b2, %v5772 : tensor<128xf32>
    %v5774 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5775 = stablehlo.multiply %v5774, %v5767 : tensor<128xf32>
    %v5776 = stablehlo.multiply %v5775, %s2b0b2 : tensor<128xf32>
    %v5777 = stablehlo.subtract %v5773, %v5776 : tensor<128xf32>
    %v5778 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5779 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5780 = stablehlo.multiply %v5778, %s2b0g2m : tensor<128xf32>
    %v5781 = stablehlo.multiply %v5779, %v2805 : tensor<128xf32>
    %v5782 = stablehlo.add %v5780, %v5781 : tensor<128xf32>
    %v5783 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5784 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5785 = stablehlo.multiply %v5783, %s2b0g2v : tensor<128xf32>
    %v5786 = stablehlo.multiply %v2805, %v2805 : tensor<128xf32>
    %v5787 = stablehlo.multiply %v5784, %v5786 : tensor<128xf32>
    %v5788 = stablehlo.add %v5785, %v5787 : tensor<128xf32>
    %v5789 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5790 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5791 = stablehlo.multiply %v5789, %s2b0g2m : tensor<128xf32>
    %v5792 = stablehlo.multiply %v5790, %v2805 : tensor<128xf32>
    %v5793 = stablehlo.add %v5791, %v5792 : tensor<128xf32>
    %v5794 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5795 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5796 = stablehlo.multiply %v5794, %s2b0g2v : tensor<128xf32>
    %v5797 = stablehlo.multiply %v2805, %v2805 : tensor<128xf32>
    %v5798 = stablehlo.multiply %v5795, %v5797 : tensor<128xf32>
    %v5799 = stablehlo.add %v5796, %v5798 : tensor<128xf32>
    %v5800 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5801 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5802 = stablehlo.divide %v5793, %v5800 : tensor<128xf32>
    %v5803 = stablehlo.divide %v5799, %v5801 : tensor<128xf32>
    %v5804 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5805 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5806 = stablehlo.sqrt %v5803 : tensor<128xf32>
    %v5807 = stablehlo.add %v5806, %v5805 : tensor<128xf32>
    %v5808 = stablehlo.divide %v5802, %v5807 : tensor<128xf32>
    %v5809 = stablehlo.multiply %v5804, %v5808 : tensor<128xf32>
    %v5810 = stablehlo.subtract %s2b0g2, %v5809 : tensor<128xf32>
    %v5811 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5812 = stablehlo.multiply %v5811, %v5804 : tensor<128xf32>
    %v5813 = stablehlo.multiply %v5812, %s2b0g2 : tensor<128xf32>
    %v5814 = stablehlo.subtract %v5810, %v5813 : tensor<128xf32>
    %v5815 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5816 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5817 = stablehlo.multiply %v5815, %s2b0bt2m : tensor<128xf32>
    %v5818 = stablehlo.multiply %v5816, %v2808 : tensor<128xf32>
    %v5819 = stablehlo.add %v5817, %v5818 : tensor<128xf32>
    %v5820 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5821 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5822 = stablehlo.multiply %v5820, %s2b0bt2v : tensor<128xf32>
    %v5823 = stablehlo.multiply %v2808, %v2808 : tensor<128xf32>
    %v5824 = stablehlo.multiply %v5821, %v5823 : tensor<128xf32>
    %v5825 = stablehlo.add %v5822, %v5824 : tensor<128xf32>
    %v5826 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5827 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5828 = stablehlo.multiply %v5826, %s2b0bt2m : tensor<128xf32>
    %v5829 = stablehlo.multiply %v5827, %v2808 : tensor<128xf32>
    %v5830 = stablehlo.add %v5828, %v5829 : tensor<128xf32>
    %v5831 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5832 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5833 = stablehlo.multiply %v5831, %s2b0bt2v : tensor<128xf32>
    %v5834 = stablehlo.multiply %v2808, %v2808 : tensor<128xf32>
    %v5835 = stablehlo.multiply %v5832, %v5834 : tensor<128xf32>
    %v5836 = stablehlo.add %v5833, %v5835 : tensor<128xf32>
    %v5837 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5838 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5839 = stablehlo.divide %v5830, %v5837 : tensor<128xf32>
    %v5840 = stablehlo.divide %v5836, %v5838 : tensor<128xf32>
    %v5841 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5842 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5843 = stablehlo.sqrt %v5840 : tensor<128xf32>
    %v5844 = stablehlo.add %v5843, %v5842 : tensor<128xf32>
    %v5845 = stablehlo.divide %v5839, %v5844 : tensor<128xf32>
    %v5846 = stablehlo.multiply %v5841, %v5845 : tensor<128xf32>
    %v5847 = stablehlo.subtract %s2b0bt2, %v5846 : tensor<128xf32>
    %v5848 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5849 = stablehlo.multiply %v5848, %v5841 : tensor<128xf32>
    %v5850 = stablehlo.multiply %v5849, %s2b0bt2 : tensor<128xf32>
    %v5851 = stablehlo.subtract %v5847, %v5850 : tensor<128xf32>
    %v5852 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5853 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5854 = stablehlo.multiply %v5852, %s2b1W1m : tensor<128x128x3x3xf32>
    %v5855 = stablehlo.multiply %v5853, %v2617 : tensor<128x128x3x3xf32>
    %v5856 = stablehlo.add %v5854, %v5855 : tensor<128x128x3x3xf32>
    %v5857 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5858 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5859 = stablehlo.multiply %v5857, %s2b1W1v : tensor<128x128x3x3xf32>
    %v5860 = stablehlo.multiply %v2617, %v2617 : tensor<128x128x3x3xf32>
    %v5861 = stablehlo.multiply %v5858, %v5860 : tensor<128x128x3x3xf32>
    %v5862 = stablehlo.add %v5859, %v5861 : tensor<128x128x3x3xf32>
    %v5863 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5864 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5865 = stablehlo.multiply %v5863, %s2b1W1m : tensor<128x128x3x3xf32>
    %v5866 = stablehlo.multiply %v5864, %v2617 : tensor<128x128x3x3xf32>
    %v5867 = stablehlo.add %v5865, %v5866 : tensor<128x128x3x3xf32>
    %v5868 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5869 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5870 = stablehlo.multiply %v5868, %s2b1W1v : tensor<128x128x3x3xf32>
    %v5871 = stablehlo.multiply %v2617, %v2617 : tensor<128x128x3x3xf32>
    %v5872 = stablehlo.multiply %v5869, %v5871 : tensor<128x128x3x3xf32>
    %v5873 = stablehlo.add %v5870, %v5872 : tensor<128x128x3x3xf32>
    %v5874 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5875 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5876 = stablehlo.divide %v5867, %v5874 : tensor<128x128x3x3xf32>
    %v5877 = stablehlo.divide %v5873, %v5875 : tensor<128x128x3x3xf32>
    %v5878 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5879 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5880 = stablehlo.sqrt %v5877 : tensor<128x128x3x3xf32>
    %v5881 = stablehlo.add %v5880, %v5879 : tensor<128x128x3x3xf32>
    %v5882 = stablehlo.divide %v5876, %v5881 : tensor<128x128x3x3xf32>
    %v5883 = stablehlo.multiply %v5878, %v5882 : tensor<128x128x3x3xf32>
    %v5884 = stablehlo.subtract %s2b1W1, %v5883 : tensor<128x128x3x3xf32>
    %v5885 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5886 = stablehlo.multiply %v5885, %v5878 : tensor<128x128x3x3xf32>
    %v5887 = stablehlo.multiply %v5886, %s2b1W1 : tensor<128x128x3x3xf32>
    %v5888 = stablehlo.subtract %v5884, %v5887 : tensor<128x128x3x3xf32>
    %v5889 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5890 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5891 = stablehlo.multiply %v5889, %s2b1b1m : tensor<128xf32>
    %v5892 = stablehlo.multiply %v5890, %v2620 : tensor<128xf32>
    %v5893 = stablehlo.add %v5891, %v5892 : tensor<128xf32>
    %v5894 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5895 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5896 = stablehlo.multiply %v5894, %s2b1b1v : tensor<128xf32>
    %v5897 = stablehlo.multiply %v2620, %v2620 : tensor<128xf32>
    %v5898 = stablehlo.multiply %v5895, %v5897 : tensor<128xf32>
    %v5899 = stablehlo.add %v5896, %v5898 : tensor<128xf32>
    %v5900 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5901 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5902 = stablehlo.multiply %v5900, %s2b1b1m : tensor<128xf32>
    %v5903 = stablehlo.multiply %v5901, %v2620 : tensor<128xf32>
    %v5904 = stablehlo.add %v5902, %v5903 : tensor<128xf32>
    %v5905 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5906 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5907 = stablehlo.multiply %v5905, %s2b1b1v : tensor<128xf32>
    %v5908 = stablehlo.multiply %v2620, %v2620 : tensor<128xf32>
    %v5909 = stablehlo.multiply %v5906, %v5908 : tensor<128xf32>
    %v5910 = stablehlo.add %v5907, %v5909 : tensor<128xf32>
    %v5911 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5912 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5913 = stablehlo.divide %v5904, %v5911 : tensor<128xf32>
    %v5914 = stablehlo.divide %v5910, %v5912 : tensor<128xf32>
    %v5915 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5916 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5917 = stablehlo.sqrt %v5914 : tensor<128xf32>
    %v5918 = stablehlo.add %v5917, %v5916 : tensor<128xf32>
    %v5919 = stablehlo.divide %v5913, %v5918 : tensor<128xf32>
    %v5920 = stablehlo.multiply %v5915, %v5919 : tensor<128xf32>
    %v5921 = stablehlo.subtract %s2b1b1, %v5920 : tensor<128xf32>
    %v5922 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5923 = stablehlo.multiply %v5922, %v5915 : tensor<128xf32>
    %v5924 = stablehlo.multiply %v5923, %s2b1b1 : tensor<128xf32>
    %v5925 = stablehlo.subtract %v5921, %v5924 : tensor<128xf32>
    %v5926 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5927 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5928 = stablehlo.multiply %v5926, %s2b1g1m : tensor<128xf32>
    %v5929 = stablehlo.multiply %v5927, %v2638 : tensor<128xf32>
    %v5930 = stablehlo.add %v5928, %v5929 : tensor<128xf32>
    %v5931 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5932 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5933 = stablehlo.multiply %v5931, %s2b1g1v : tensor<128xf32>
    %v5934 = stablehlo.multiply %v2638, %v2638 : tensor<128xf32>
    %v5935 = stablehlo.multiply %v5932, %v5934 : tensor<128xf32>
    %v5936 = stablehlo.add %v5933, %v5935 : tensor<128xf32>
    %v5937 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5938 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5939 = stablehlo.multiply %v5937, %s2b1g1m : tensor<128xf32>
    %v5940 = stablehlo.multiply %v5938, %v2638 : tensor<128xf32>
    %v5941 = stablehlo.add %v5939, %v5940 : tensor<128xf32>
    %v5942 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5943 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5944 = stablehlo.multiply %v5942, %s2b1g1v : tensor<128xf32>
    %v5945 = stablehlo.multiply %v2638, %v2638 : tensor<128xf32>
    %v5946 = stablehlo.multiply %v5943, %v5945 : tensor<128xf32>
    %v5947 = stablehlo.add %v5944, %v5946 : tensor<128xf32>
    %v5948 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5949 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5950 = stablehlo.divide %v5941, %v5948 : tensor<128xf32>
    %v5951 = stablehlo.divide %v5947, %v5949 : tensor<128xf32>
    %v5952 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5953 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5954 = stablehlo.sqrt %v5951 : tensor<128xf32>
    %v5955 = stablehlo.add %v5954, %v5953 : tensor<128xf32>
    %v5956 = stablehlo.divide %v5950, %v5955 : tensor<128xf32>
    %v5957 = stablehlo.multiply %v5952, %v5956 : tensor<128xf32>
    %v5958 = stablehlo.subtract %s2b1g1, %v5957 : tensor<128xf32>
    %v5959 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5960 = stablehlo.multiply %v5959, %v5952 : tensor<128xf32>
    %v5961 = stablehlo.multiply %v5960, %s2b1g1 : tensor<128xf32>
    %v5962 = stablehlo.subtract %v5958, %v5961 : tensor<128xf32>
    %v5963 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5964 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5965 = stablehlo.multiply %v5963, %s2b1bt1m : tensor<128xf32>
    %v5966 = stablehlo.multiply %v5964, %v2641 : tensor<128xf32>
    %v5967 = stablehlo.add %v5965, %v5966 : tensor<128xf32>
    %v5968 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5969 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5970 = stablehlo.multiply %v5968, %s2b1bt1v : tensor<128xf32>
    %v5971 = stablehlo.multiply %v2641, %v2641 : tensor<128xf32>
    %v5972 = stablehlo.multiply %v5969, %v5971 : tensor<128xf32>
    %v5973 = stablehlo.add %v5970, %v5972 : tensor<128xf32>
    %v5974 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5975 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5976 = stablehlo.multiply %v5974, %s2b1bt1m : tensor<128xf32>
    %v5977 = stablehlo.multiply %v5975, %v2641 : tensor<128xf32>
    %v5978 = stablehlo.add %v5976, %v5977 : tensor<128xf32>
    %v5979 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5980 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5981 = stablehlo.multiply %v5979, %s2b1bt1v : tensor<128xf32>
    %v5982 = stablehlo.multiply %v2641, %v2641 : tensor<128xf32>
    %v5983 = stablehlo.multiply %v5980, %v5982 : tensor<128xf32>
    %v5984 = stablehlo.add %v5981, %v5983 : tensor<128xf32>
    %v5985 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5986 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5987 = stablehlo.divide %v5978, %v5985 : tensor<128xf32>
    %v5988 = stablehlo.divide %v5984, %v5986 : tensor<128xf32>
    %v5989 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5990 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5991 = stablehlo.sqrt %v5988 : tensor<128xf32>
    %v5992 = stablehlo.add %v5991, %v5990 : tensor<128xf32>
    %v5993 = stablehlo.divide %v5987, %v5992 : tensor<128xf32>
    %v5994 = stablehlo.multiply %v5989, %v5993 : tensor<128xf32>
    %v5995 = stablehlo.subtract %s2b1bt1, %v5994 : tensor<128xf32>
    %v5996 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5997 = stablehlo.multiply %v5996, %v5989 : tensor<128xf32>
    %v5998 = stablehlo.multiply %v5997, %s2b1bt1 : tensor<128xf32>
    %v5999 = stablehlo.subtract %v5995, %v5998 : tensor<128xf32>
    %v6000 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6001 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6002 = stablehlo.multiply %v6000, %s2b1W2m : tensor<128x128x3x3xf32>
    %v6003 = stablehlo.multiply %v6001, %v2647 : tensor<128x128x3x3xf32>
    %v6004 = stablehlo.add %v6002, %v6003 : tensor<128x128x3x3xf32>
    %v6005 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6006 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6007 = stablehlo.multiply %v6005, %s2b1W2v : tensor<128x128x3x3xf32>
    %v6008 = stablehlo.multiply %v2647, %v2647 : tensor<128x128x3x3xf32>
    %v6009 = stablehlo.multiply %v6006, %v6008 : tensor<128x128x3x3xf32>
    %v6010 = stablehlo.add %v6007, %v6009 : tensor<128x128x3x3xf32>
    %v6011 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6012 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6013 = stablehlo.multiply %v6011, %s2b1W2m : tensor<128x128x3x3xf32>
    %v6014 = stablehlo.multiply %v6012, %v2647 : tensor<128x128x3x3xf32>
    %v6015 = stablehlo.add %v6013, %v6014 : tensor<128x128x3x3xf32>
    %v6016 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6017 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6018 = stablehlo.multiply %v6016, %s2b1W2v : tensor<128x128x3x3xf32>
    %v6019 = stablehlo.multiply %v2647, %v2647 : tensor<128x128x3x3xf32>
    %v6020 = stablehlo.multiply %v6017, %v6019 : tensor<128x128x3x3xf32>
    %v6021 = stablehlo.add %v6018, %v6020 : tensor<128x128x3x3xf32>
    %v6022 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6023 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6024 = stablehlo.divide %v6015, %v6022 : tensor<128x128x3x3xf32>
    %v6025 = stablehlo.divide %v6021, %v6023 : tensor<128x128x3x3xf32>
    %v6026 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6027 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6028 = stablehlo.sqrt %v6025 : tensor<128x128x3x3xf32>
    %v6029 = stablehlo.add %v6028, %v6027 : tensor<128x128x3x3xf32>
    %v6030 = stablehlo.divide %v6024, %v6029 : tensor<128x128x3x3xf32>
    %v6031 = stablehlo.multiply %v6026, %v6030 : tensor<128x128x3x3xf32>
    %v6032 = stablehlo.subtract %s2b1W2, %v6031 : tensor<128x128x3x3xf32>
    %v6033 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6034 = stablehlo.multiply %v6033, %v6026 : tensor<128x128x3x3xf32>
    %v6035 = stablehlo.multiply %v6034, %s2b1W2 : tensor<128x128x3x3xf32>
    %v6036 = stablehlo.subtract %v6032, %v6035 : tensor<128x128x3x3xf32>
    %v6037 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6038 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6039 = stablehlo.multiply %v6037, %s2b1b2m : tensor<128xf32>
    %v6040 = stablehlo.multiply %v6038, %v2650 : tensor<128xf32>
    %v6041 = stablehlo.add %v6039, %v6040 : tensor<128xf32>
    %v6042 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6043 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6044 = stablehlo.multiply %v6042, %s2b1b2v : tensor<128xf32>
    %v6045 = stablehlo.multiply %v2650, %v2650 : tensor<128xf32>
    %v6046 = stablehlo.multiply %v6043, %v6045 : tensor<128xf32>
    %v6047 = stablehlo.add %v6044, %v6046 : tensor<128xf32>
    %v6048 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6049 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6050 = stablehlo.multiply %v6048, %s2b1b2m : tensor<128xf32>
    %v6051 = stablehlo.multiply %v6049, %v2650 : tensor<128xf32>
    %v6052 = stablehlo.add %v6050, %v6051 : tensor<128xf32>
    %v6053 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6054 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6055 = stablehlo.multiply %v6053, %s2b1b2v : tensor<128xf32>
    %v6056 = stablehlo.multiply %v2650, %v2650 : tensor<128xf32>
    %v6057 = stablehlo.multiply %v6054, %v6056 : tensor<128xf32>
    %v6058 = stablehlo.add %v6055, %v6057 : tensor<128xf32>
    %v6059 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6060 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6061 = stablehlo.divide %v6052, %v6059 : tensor<128xf32>
    %v6062 = stablehlo.divide %v6058, %v6060 : tensor<128xf32>
    %v6063 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6064 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6065 = stablehlo.sqrt %v6062 : tensor<128xf32>
    %v6066 = stablehlo.add %v6065, %v6064 : tensor<128xf32>
    %v6067 = stablehlo.divide %v6061, %v6066 : tensor<128xf32>
    %v6068 = stablehlo.multiply %v6063, %v6067 : tensor<128xf32>
    %v6069 = stablehlo.subtract %s2b1b2, %v6068 : tensor<128xf32>
    %v6070 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6071 = stablehlo.multiply %v6070, %v6063 : tensor<128xf32>
    %v6072 = stablehlo.multiply %v6071, %s2b1b2 : tensor<128xf32>
    %v6073 = stablehlo.subtract %v6069, %v6072 : tensor<128xf32>
    %v6074 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6075 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6076 = stablehlo.multiply %v6074, %s2b1g2m : tensor<128xf32>
    %v6077 = stablehlo.multiply %v6075, %v2668 : tensor<128xf32>
    %v6078 = stablehlo.add %v6076, %v6077 : tensor<128xf32>
    %v6079 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6080 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6081 = stablehlo.multiply %v6079, %s2b1g2v : tensor<128xf32>
    %v6082 = stablehlo.multiply %v2668, %v2668 : tensor<128xf32>
    %v6083 = stablehlo.multiply %v6080, %v6082 : tensor<128xf32>
    %v6084 = stablehlo.add %v6081, %v6083 : tensor<128xf32>
    %v6085 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6086 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6087 = stablehlo.multiply %v6085, %s2b1g2m : tensor<128xf32>
    %v6088 = stablehlo.multiply %v6086, %v2668 : tensor<128xf32>
    %v6089 = stablehlo.add %v6087, %v6088 : tensor<128xf32>
    %v6090 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6091 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6092 = stablehlo.multiply %v6090, %s2b1g2v : tensor<128xf32>
    %v6093 = stablehlo.multiply %v2668, %v2668 : tensor<128xf32>
    %v6094 = stablehlo.multiply %v6091, %v6093 : tensor<128xf32>
    %v6095 = stablehlo.add %v6092, %v6094 : tensor<128xf32>
    %v6096 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6097 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6098 = stablehlo.divide %v6089, %v6096 : tensor<128xf32>
    %v6099 = stablehlo.divide %v6095, %v6097 : tensor<128xf32>
    %v6100 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6101 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6102 = stablehlo.sqrt %v6099 : tensor<128xf32>
    %v6103 = stablehlo.add %v6102, %v6101 : tensor<128xf32>
    %v6104 = stablehlo.divide %v6098, %v6103 : tensor<128xf32>
    %v6105 = stablehlo.multiply %v6100, %v6104 : tensor<128xf32>
    %v6106 = stablehlo.subtract %s2b1g2, %v6105 : tensor<128xf32>
    %v6107 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6108 = stablehlo.multiply %v6107, %v6100 : tensor<128xf32>
    %v6109 = stablehlo.multiply %v6108, %s2b1g2 : tensor<128xf32>
    %v6110 = stablehlo.subtract %v6106, %v6109 : tensor<128xf32>
    %v6111 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6112 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6113 = stablehlo.multiply %v6111, %s2b1bt2m : tensor<128xf32>
    %v6114 = stablehlo.multiply %v6112, %v2671 : tensor<128xf32>
    %v6115 = stablehlo.add %v6113, %v6114 : tensor<128xf32>
    %v6116 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6117 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6118 = stablehlo.multiply %v6116, %s2b1bt2v : tensor<128xf32>
    %v6119 = stablehlo.multiply %v2671, %v2671 : tensor<128xf32>
    %v6120 = stablehlo.multiply %v6117, %v6119 : tensor<128xf32>
    %v6121 = stablehlo.add %v6118, %v6120 : tensor<128xf32>
    %v6122 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6123 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6124 = stablehlo.multiply %v6122, %s2b1bt2m : tensor<128xf32>
    %v6125 = stablehlo.multiply %v6123, %v2671 : tensor<128xf32>
    %v6126 = stablehlo.add %v6124, %v6125 : tensor<128xf32>
    %v6127 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6128 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6129 = stablehlo.multiply %v6127, %s2b1bt2v : tensor<128xf32>
    %v6130 = stablehlo.multiply %v2671, %v2671 : tensor<128xf32>
    %v6131 = stablehlo.multiply %v6128, %v6130 : tensor<128xf32>
    %v6132 = stablehlo.add %v6129, %v6131 : tensor<128xf32>
    %v6133 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6134 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6135 = stablehlo.divide %v6126, %v6133 : tensor<128xf32>
    %v6136 = stablehlo.divide %v6132, %v6134 : tensor<128xf32>
    %v6137 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6138 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6139 = stablehlo.sqrt %v6136 : tensor<128xf32>
    %v6140 = stablehlo.add %v6139, %v6138 : tensor<128xf32>
    %v6141 = stablehlo.divide %v6135, %v6140 : tensor<128xf32>
    %v6142 = stablehlo.multiply %v6137, %v6141 : tensor<128xf32>
    %v6143 = stablehlo.subtract %s2b1bt2, %v6142 : tensor<128xf32>
    %v6144 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6145 = stablehlo.multiply %v6144, %v6137 : tensor<128xf32>
    %v6146 = stablehlo.multiply %v6145, %s2b1bt2 : tensor<128xf32>
    %v6147 = stablehlo.subtract %v6143, %v6146 : tensor<128xf32>
    %v6148 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6149 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6150 = stablehlo.multiply %v6148, %s2b2W1m : tensor<128x128x3x3xf32>
    %v6151 = stablehlo.multiply %v6149, %v2480 : tensor<128x128x3x3xf32>
    %v6152 = stablehlo.add %v6150, %v6151 : tensor<128x128x3x3xf32>
    %v6153 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6154 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6155 = stablehlo.multiply %v6153, %s2b2W1v : tensor<128x128x3x3xf32>
    %v6156 = stablehlo.multiply %v2480, %v2480 : tensor<128x128x3x3xf32>
    %v6157 = stablehlo.multiply %v6154, %v6156 : tensor<128x128x3x3xf32>
    %v6158 = stablehlo.add %v6155, %v6157 : tensor<128x128x3x3xf32>
    %v6159 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6160 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6161 = stablehlo.multiply %v6159, %s2b2W1m : tensor<128x128x3x3xf32>
    %v6162 = stablehlo.multiply %v6160, %v2480 : tensor<128x128x3x3xf32>
    %v6163 = stablehlo.add %v6161, %v6162 : tensor<128x128x3x3xf32>
    %v6164 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6165 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6166 = stablehlo.multiply %v6164, %s2b2W1v : tensor<128x128x3x3xf32>
    %v6167 = stablehlo.multiply %v2480, %v2480 : tensor<128x128x3x3xf32>
    %v6168 = stablehlo.multiply %v6165, %v6167 : tensor<128x128x3x3xf32>
    %v6169 = stablehlo.add %v6166, %v6168 : tensor<128x128x3x3xf32>
    %v6170 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6171 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6172 = stablehlo.divide %v6163, %v6170 : tensor<128x128x3x3xf32>
    %v6173 = stablehlo.divide %v6169, %v6171 : tensor<128x128x3x3xf32>
    %v6174 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6175 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6176 = stablehlo.sqrt %v6173 : tensor<128x128x3x3xf32>
    %v6177 = stablehlo.add %v6176, %v6175 : tensor<128x128x3x3xf32>
    %v6178 = stablehlo.divide %v6172, %v6177 : tensor<128x128x3x3xf32>
    %v6179 = stablehlo.multiply %v6174, %v6178 : tensor<128x128x3x3xf32>
    %v6180 = stablehlo.subtract %s2b2W1, %v6179 : tensor<128x128x3x3xf32>
    %v6181 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6182 = stablehlo.multiply %v6181, %v6174 : tensor<128x128x3x3xf32>
    %v6183 = stablehlo.multiply %v6182, %s2b2W1 : tensor<128x128x3x3xf32>
    %v6184 = stablehlo.subtract %v6180, %v6183 : tensor<128x128x3x3xf32>
    %v6185 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6186 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6187 = stablehlo.multiply %v6185, %s2b2b1m : tensor<128xf32>
    %v6188 = stablehlo.multiply %v6186, %v2483 : tensor<128xf32>
    %v6189 = stablehlo.add %v6187, %v6188 : tensor<128xf32>
    %v6190 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6191 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6192 = stablehlo.multiply %v6190, %s2b2b1v : tensor<128xf32>
    %v6193 = stablehlo.multiply %v2483, %v2483 : tensor<128xf32>
    %v6194 = stablehlo.multiply %v6191, %v6193 : tensor<128xf32>
    %v6195 = stablehlo.add %v6192, %v6194 : tensor<128xf32>
    %v6196 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6197 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6198 = stablehlo.multiply %v6196, %s2b2b1m : tensor<128xf32>
    %v6199 = stablehlo.multiply %v6197, %v2483 : tensor<128xf32>
    %v6200 = stablehlo.add %v6198, %v6199 : tensor<128xf32>
    %v6201 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6202 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6203 = stablehlo.multiply %v6201, %s2b2b1v : tensor<128xf32>
    %v6204 = stablehlo.multiply %v2483, %v2483 : tensor<128xf32>
    %v6205 = stablehlo.multiply %v6202, %v6204 : tensor<128xf32>
    %v6206 = stablehlo.add %v6203, %v6205 : tensor<128xf32>
    %v6207 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6208 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6209 = stablehlo.divide %v6200, %v6207 : tensor<128xf32>
    %v6210 = stablehlo.divide %v6206, %v6208 : tensor<128xf32>
    %v6211 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6212 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6213 = stablehlo.sqrt %v6210 : tensor<128xf32>
    %v6214 = stablehlo.add %v6213, %v6212 : tensor<128xf32>
    %v6215 = stablehlo.divide %v6209, %v6214 : tensor<128xf32>
    %v6216 = stablehlo.multiply %v6211, %v6215 : tensor<128xf32>
    %v6217 = stablehlo.subtract %s2b2b1, %v6216 : tensor<128xf32>
    %v6218 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6219 = stablehlo.multiply %v6218, %v6211 : tensor<128xf32>
    %v6220 = stablehlo.multiply %v6219, %s2b2b1 : tensor<128xf32>
    %v6221 = stablehlo.subtract %v6217, %v6220 : tensor<128xf32>
    %v6222 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6223 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6224 = stablehlo.multiply %v6222, %s2b2g1m : tensor<128xf32>
    %v6225 = stablehlo.multiply %v6223, %v2501 : tensor<128xf32>
    %v6226 = stablehlo.add %v6224, %v6225 : tensor<128xf32>
    %v6227 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6228 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6229 = stablehlo.multiply %v6227, %s2b2g1v : tensor<128xf32>
    %v6230 = stablehlo.multiply %v2501, %v2501 : tensor<128xf32>
    %v6231 = stablehlo.multiply %v6228, %v6230 : tensor<128xf32>
    %v6232 = stablehlo.add %v6229, %v6231 : tensor<128xf32>
    %v6233 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6234 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6235 = stablehlo.multiply %v6233, %s2b2g1m : tensor<128xf32>
    %v6236 = stablehlo.multiply %v6234, %v2501 : tensor<128xf32>
    %v6237 = stablehlo.add %v6235, %v6236 : tensor<128xf32>
    %v6238 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6239 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6240 = stablehlo.multiply %v6238, %s2b2g1v : tensor<128xf32>
    %v6241 = stablehlo.multiply %v2501, %v2501 : tensor<128xf32>
    %v6242 = stablehlo.multiply %v6239, %v6241 : tensor<128xf32>
    %v6243 = stablehlo.add %v6240, %v6242 : tensor<128xf32>
    %v6244 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6245 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6246 = stablehlo.divide %v6237, %v6244 : tensor<128xf32>
    %v6247 = stablehlo.divide %v6243, %v6245 : tensor<128xf32>
    %v6248 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6249 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6250 = stablehlo.sqrt %v6247 : tensor<128xf32>
    %v6251 = stablehlo.add %v6250, %v6249 : tensor<128xf32>
    %v6252 = stablehlo.divide %v6246, %v6251 : tensor<128xf32>
    %v6253 = stablehlo.multiply %v6248, %v6252 : tensor<128xf32>
    %v6254 = stablehlo.subtract %s2b2g1, %v6253 : tensor<128xf32>
    %v6255 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6256 = stablehlo.multiply %v6255, %v6248 : tensor<128xf32>
    %v6257 = stablehlo.multiply %v6256, %s2b2g1 : tensor<128xf32>
    %v6258 = stablehlo.subtract %v6254, %v6257 : tensor<128xf32>
    %v6259 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6260 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6261 = stablehlo.multiply %v6259, %s2b2bt1m : tensor<128xf32>
    %v6262 = stablehlo.multiply %v6260, %v2504 : tensor<128xf32>
    %v6263 = stablehlo.add %v6261, %v6262 : tensor<128xf32>
    %v6264 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6265 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6266 = stablehlo.multiply %v6264, %s2b2bt1v : tensor<128xf32>
    %v6267 = stablehlo.multiply %v2504, %v2504 : tensor<128xf32>
    %v6268 = stablehlo.multiply %v6265, %v6267 : tensor<128xf32>
    %v6269 = stablehlo.add %v6266, %v6268 : tensor<128xf32>
    %v6270 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6271 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6272 = stablehlo.multiply %v6270, %s2b2bt1m : tensor<128xf32>
    %v6273 = stablehlo.multiply %v6271, %v2504 : tensor<128xf32>
    %v6274 = stablehlo.add %v6272, %v6273 : tensor<128xf32>
    %v6275 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6276 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6277 = stablehlo.multiply %v6275, %s2b2bt1v : tensor<128xf32>
    %v6278 = stablehlo.multiply %v2504, %v2504 : tensor<128xf32>
    %v6279 = stablehlo.multiply %v6276, %v6278 : tensor<128xf32>
    %v6280 = stablehlo.add %v6277, %v6279 : tensor<128xf32>
    %v6281 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6282 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6283 = stablehlo.divide %v6274, %v6281 : tensor<128xf32>
    %v6284 = stablehlo.divide %v6280, %v6282 : tensor<128xf32>
    %v6285 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6286 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6287 = stablehlo.sqrt %v6284 : tensor<128xf32>
    %v6288 = stablehlo.add %v6287, %v6286 : tensor<128xf32>
    %v6289 = stablehlo.divide %v6283, %v6288 : tensor<128xf32>
    %v6290 = stablehlo.multiply %v6285, %v6289 : tensor<128xf32>
    %v6291 = stablehlo.subtract %s2b2bt1, %v6290 : tensor<128xf32>
    %v6292 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6293 = stablehlo.multiply %v6292, %v6285 : tensor<128xf32>
    %v6294 = stablehlo.multiply %v6293, %s2b2bt1 : tensor<128xf32>
    %v6295 = stablehlo.subtract %v6291, %v6294 : tensor<128xf32>
    %v6296 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6297 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6298 = stablehlo.multiply %v6296, %s2b2W2m : tensor<128x128x3x3xf32>
    %v6299 = stablehlo.multiply %v6297, %v2510 : tensor<128x128x3x3xf32>
    %v6300 = stablehlo.add %v6298, %v6299 : tensor<128x128x3x3xf32>
    %v6301 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6302 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6303 = stablehlo.multiply %v6301, %s2b2W2v : tensor<128x128x3x3xf32>
    %v6304 = stablehlo.multiply %v2510, %v2510 : tensor<128x128x3x3xf32>
    %v6305 = stablehlo.multiply %v6302, %v6304 : tensor<128x128x3x3xf32>
    %v6306 = stablehlo.add %v6303, %v6305 : tensor<128x128x3x3xf32>
    %v6307 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6308 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6309 = stablehlo.multiply %v6307, %s2b2W2m : tensor<128x128x3x3xf32>
    %v6310 = stablehlo.multiply %v6308, %v2510 : tensor<128x128x3x3xf32>
    %v6311 = stablehlo.add %v6309, %v6310 : tensor<128x128x3x3xf32>
    %v6312 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6313 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6314 = stablehlo.multiply %v6312, %s2b2W2v : tensor<128x128x3x3xf32>
    %v6315 = stablehlo.multiply %v2510, %v2510 : tensor<128x128x3x3xf32>
    %v6316 = stablehlo.multiply %v6313, %v6315 : tensor<128x128x3x3xf32>
    %v6317 = stablehlo.add %v6314, %v6316 : tensor<128x128x3x3xf32>
    %v6318 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6319 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6320 = stablehlo.divide %v6311, %v6318 : tensor<128x128x3x3xf32>
    %v6321 = stablehlo.divide %v6317, %v6319 : tensor<128x128x3x3xf32>
    %v6322 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6323 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6324 = stablehlo.sqrt %v6321 : tensor<128x128x3x3xf32>
    %v6325 = stablehlo.add %v6324, %v6323 : tensor<128x128x3x3xf32>
    %v6326 = stablehlo.divide %v6320, %v6325 : tensor<128x128x3x3xf32>
    %v6327 = stablehlo.multiply %v6322, %v6326 : tensor<128x128x3x3xf32>
    %v6328 = stablehlo.subtract %s2b2W2, %v6327 : tensor<128x128x3x3xf32>
    %v6329 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6330 = stablehlo.multiply %v6329, %v6322 : tensor<128x128x3x3xf32>
    %v6331 = stablehlo.multiply %v6330, %s2b2W2 : tensor<128x128x3x3xf32>
    %v6332 = stablehlo.subtract %v6328, %v6331 : tensor<128x128x3x3xf32>
    %v6333 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6334 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6335 = stablehlo.multiply %v6333, %s2b2b2m : tensor<128xf32>
    %v6336 = stablehlo.multiply %v6334, %v2513 : tensor<128xf32>
    %v6337 = stablehlo.add %v6335, %v6336 : tensor<128xf32>
    %v6338 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6339 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6340 = stablehlo.multiply %v6338, %s2b2b2v : tensor<128xf32>
    %v6341 = stablehlo.multiply %v2513, %v2513 : tensor<128xf32>
    %v6342 = stablehlo.multiply %v6339, %v6341 : tensor<128xf32>
    %v6343 = stablehlo.add %v6340, %v6342 : tensor<128xf32>
    %v6344 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6345 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6346 = stablehlo.multiply %v6344, %s2b2b2m : tensor<128xf32>
    %v6347 = stablehlo.multiply %v6345, %v2513 : tensor<128xf32>
    %v6348 = stablehlo.add %v6346, %v6347 : tensor<128xf32>
    %v6349 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6350 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6351 = stablehlo.multiply %v6349, %s2b2b2v : tensor<128xf32>
    %v6352 = stablehlo.multiply %v2513, %v2513 : tensor<128xf32>
    %v6353 = stablehlo.multiply %v6350, %v6352 : tensor<128xf32>
    %v6354 = stablehlo.add %v6351, %v6353 : tensor<128xf32>
    %v6355 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6356 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6357 = stablehlo.divide %v6348, %v6355 : tensor<128xf32>
    %v6358 = stablehlo.divide %v6354, %v6356 : tensor<128xf32>
    %v6359 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6360 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6361 = stablehlo.sqrt %v6358 : tensor<128xf32>
    %v6362 = stablehlo.add %v6361, %v6360 : tensor<128xf32>
    %v6363 = stablehlo.divide %v6357, %v6362 : tensor<128xf32>
    %v6364 = stablehlo.multiply %v6359, %v6363 : tensor<128xf32>
    %v6365 = stablehlo.subtract %s2b2b2, %v6364 : tensor<128xf32>
    %v6366 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6367 = stablehlo.multiply %v6366, %v6359 : tensor<128xf32>
    %v6368 = stablehlo.multiply %v6367, %s2b2b2 : tensor<128xf32>
    %v6369 = stablehlo.subtract %v6365, %v6368 : tensor<128xf32>
    %v6370 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6371 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6372 = stablehlo.multiply %v6370, %s2b2g2m : tensor<128xf32>
    %v6373 = stablehlo.multiply %v6371, %v2531 : tensor<128xf32>
    %v6374 = stablehlo.add %v6372, %v6373 : tensor<128xf32>
    %v6375 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6376 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6377 = stablehlo.multiply %v6375, %s2b2g2v : tensor<128xf32>
    %v6378 = stablehlo.multiply %v2531, %v2531 : tensor<128xf32>
    %v6379 = stablehlo.multiply %v6376, %v6378 : tensor<128xf32>
    %v6380 = stablehlo.add %v6377, %v6379 : tensor<128xf32>
    %v6381 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6382 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6383 = stablehlo.multiply %v6381, %s2b2g2m : tensor<128xf32>
    %v6384 = stablehlo.multiply %v6382, %v2531 : tensor<128xf32>
    %v6385 = stablehlo.add %v6383, %v6384 : tensor<128xf32>
    %v6386 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6387 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6388 = stablehlo.multiply %v6386, %s2b2g2v : tensor<128xf32>
    %v6389 = stablehlo.multiply %v2531, %v2531 : tensor<128xf32>
    %v6390 = stablehlo.multiply %v6387, %v6389 : tensor<128xf32>
    %v6391 = stablehlo.add %v6388, %v6390 : tensor<128xf32>
    %v6392 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6393 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6394 = stablehlo.divide %v6385, %v6392 : tensor<128xf32>
    %v6395 = stablehlo.divide %v6391, %v6393 : tensor<128xf32>
    %v6396 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6397 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6398 = stablehlo.sqrt %v6395 : tensor<128xf32>
    %v6399 = stablehlo.add %v6398, %v6397 : tensor<128xf32>
    %v6400 = stablehlo.divide %v6394, %v6399 : tensor<128xf32>
    %v6401 = stablehlo.multiply %v6396, %v6400 : tensor<128xf32>
    %v6402 = stablehlo.subtract %s2b2g2, %v6401 : tensor<128xf32>
    %v6403 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6404 = stablehlo.multiply %v6403, %v6396 : tensor<128xf32>
    %v6405 = stablehlo.multiply %v6404, %s2b2g2 : tensor<128xf32>
    %v6406 = stablehlo.subtract %v6402, %v6405 : tensor<128xf32>
    %v6407 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6408 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6409 = stablehlo.multiply %v6407, %s2b2bt2m : tensor<128xf32>
    %v6410 = stablehlo.multiply %v6408, %v2534 : tensor<128xf32>
    %v6411 = stablehlo.add %v6409, %v6410 : tensor<128xf32>
    %v6412 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6413 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6414 = stablehlo.multiply %v6412, %s2b2bt2v : tensor<128xf32>
    %v6415 = stablehlo.multiply %v2534, %v2534 : tensor<128xf32>
    %v6416 = stablehlo.multiply %v6413, %v6415 : tensor<128xf32>
    %v6417 = stablehlo.add %v6414, %v6416 : tensor<128xf32>
    %v6418 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6419 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6420 = stablehlo.multiply %v6418, %s2b2bt2m : tensor<128xf32>
    %v6421 = stablehlo.multiply %v6419, %v2534 : tensor<128xf32>
    %v6422 = stablehlo.add %v6420, %v6421 : tensor<128xf32>
    %v6423 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6424 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6425 = stablehlo.multiply %v6423, %s2b2bt2v : tensor<128xf32>
    %v6426 = stablehlo.multiply %v2534, %v2534 : tensor<128xf32>
    %v6427 = stablehlo.multiply %v6424, %v6426 : tensor<128xf32>
    %v6428 = stablehlo.add %v6425, %v6427 : tensor<128xf32>
    %v6429 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6430 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6431 = stablehlo.divide %v6422, %v6429 : tensor<128xf32>
    %v6432 = stablehlo.divide %v6428, %v6430 : tensor<128xf32>
    %v6433 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6434 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6435 = stablehlo.sqrt %v6432 : tensor<128xf32>
    %v6436 = stablehlo.add %v6435, %v6434 : tensor<128xf32>
    %v6437 = stablehlo.divide %v6431, %v6436 : tensor<128xf32>
    %v6438 = stablehlo.multiply %v6433, %v6437 : tensor<128xf32>
    %v6439 = stablehlo.subtract %s2b2bt2, %v6438 : tensor<128xf32>
    %v6440 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6441 = stablehlo.multiply %v6440, %v6433 : tensor<128xf32>
    %v6442 = stablehlo.multiply %v6441, %s2b2bt2 : tensor<128xf32>
    %v6443 = stablehlo.subtract %v6439, %v6442 : tensor<128xf32>
    %v6444 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6445 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6446 = stablehlo.multiply %v6444, %d3W1m : tensor<256x128x3x3xf32>
    %v6447 = stablehlo.multiply %v6445, %v2311 : tensor<256x128x3x3xf32>
    %v6448 = stablehlo.add %v6446, %v6447 : tensor<256x128x3x3xf32>
    %v6449 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6450 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6451 = stablehlo.multiply %v6449, %d3W1v : tensor<256x128x3x3xf32>
    %v6452 = stablehlo.multiply %v2311, %v2311 : tensor<256x128x3x3xf32>
    %v6453 = stablehlo.multiply %v6450, %v6452 : tensor<256x128x3x3xf32>
    %v6454 = stablehlo.add %v6451, %v6453 : tensor<256x128x3x3xf32>
    %v6455 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6456 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6457 = stablehlo.multiply %v6455, %d3W1m : tensor<256x128x3x3xf32>
    %v6458 = stablehlo.multiply %v6456, %v2311 : tensor<256x128x3x3xf32>
    %v6459 = stablehlo.add %v6457, %v6458 : tensor<256x128x3x3xf32>
    %v6460 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6461 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6462 = stablehlo.multiply %v6460, %d3W1v : tensor<256x128x3x3xf32>
    %v6463 = stablehlo.multiply %v2311, %v2311 : tensor<256x128x3x3xf32>
    %v6464 = stablehlo.multiply %v6461, %v6463 : tensor<256x128x3x3xf32>
    %v6465 = stablehlo.add %v6462, %v6464 : tensor<256x128x3x3xf32>
    %v6466 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6467 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6468 = stablehlo.divide %v6459, %v6466 : tensor<256x128x3x3xf32>
    %v6469 = stablehlo.divide %v6465, %v6467 : tensor<256x128x3x3xf32>
    %v6470 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6471 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6472 = stablehlo.sqrt %v6469 : tensor<256x128x3x3xf32>
    %v6473 = stablehlo.add %v6472, %v6471 : tensor<256x128x3x3xf32>
    %v6474 = stablehlo.divide %v6468, %v6473 : tensor<256x128x3x3xf32>
    %v6475 = stablehlo.multiply %v6470, %v6474 : tensor<256x128x3x3xf32>
    %v6476 = stablehlo.subtract %d3W1, %v6475 : tensor<256x128x3x3xf32>
    %v6477 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6478 = stablehlo.multiply %v6477, %v6470 : tensor<256x128x3x3xf32>
    %v6479 = stablehlo.multiply %v6478, %d3W1 : tensor<256x128x3x3xf32>
    %v6480 = stablehlo.subtract %v6476, %v6479 : tensor<256x128x3x3xf32>
    %v6481 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6482 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6483 = stablehlo.multiply %v6481, %d3b1m : tensor<256xf32>
    %v6484 = stablehlo.multiply %v6482, %v2314 : tensor<256xf32>
    %v6485 = stablehlo.add %v6483, %v6484 : tensor<256xf32>
    %v6486 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6487 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6488 = stablehlo.multiply %v6486, %d3b1v : tensor<256xf32>
    %v6489 = stablehlo.multiply %v2314, %v2314 : tensor<256xf32>
    %v6490 = stablehlo.multiply %v6487, %v6489 : tensor<256xf32>
    %v6491 = stablehlo.add %v6488, %v6490 : tensor<256xf32>
    %v6492 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6493 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6494 = stablehlo.multiply %v6492, %d3b1m : tensor<256xf32>
    %v6495 = stablehlo.multiply %v6493, %v2314 : tensor<256xf32>
    %v6496 = stablehlo.add %v6494, %v6495 : tensor<256xf32>
    %v6497 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6498 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6499 = stablehlo.multiply %v6497, %d3b1v : tensor<256xf32>
    %v6500 = stablehlo.multiply %v2314, %v2314 : tensor<256xf32>
    %v6501 = stablehlo.multiply %v6498, %v6500 : tensor<256xf32>
    %v6502 = stablehlo.add %v6499, %v6501 : tensor<256xf32>
    %v6503 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6504 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6505 = stablehlo.divide %v6496, %v6503 : tensor<256xf32>
    %v6506 = stablehlo.divide %v6502, %v6504 : tensor<256xf32>
    %v6507 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6508 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6509 = stablehlo.sqrt %v6506 : tensor<256xf32>
    %v6510 = stablehlo.add %v6509, %v6508 : tensor<256xf32>
    %v6511 = stablehlo.divide %v6505, %v6510 : tensor<256xf32>
    %v6512 = stablehlo.multiply %v6507, %v6511 : tensor<256xf32>
    %v6513 = stablehlo.subtract %d3b1, %v6512 : tensor<256xf32>
    %v6514 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6515 = stablehlo.multiply %v6514, %v6507 : tensor<256xf32>
    %v6516 = stablehlo.multiply %v6515, %d3b1 : tensor<256xf32>
    %v6517 = stablehlo.subtract %v6513, %v6516 : tensor<256xf32>
    %v6518 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6519 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6520 = stablehlo.multiply %v6518, %d3g1m : tensor<256xf32>
    %v6521 = stablehlo.multiply %v6519, %v2332 : tensor<256xf32>
    %v6522 = stablehlo.add %v6520, %v6521 : tensor<256xf32>
    %v6523 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6524 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6525 = stablehlo.multiply %v6523, %d3g1v : tensor<256xf32>
    %v6526 = stablehlo.multiply %v2332, %v2332 : tensor<256xf32>
    %v6527 = stablehlo.multiply %v6524, %v6526 : tensor<256xf32>
    %v6528 = stablehlo.add %v6525, %v6527 : tensor<256xf32>
    %v6529 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6530 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6531 = stablehlo.multiply %v6529, %d3g1m : tensor<256xf32>
    %v6532 = stablehlo.multiply %v6530, %v2332 : tensor<256xf32>
    %v6533 = stablehlo.add %v6531, %v6532 : tensor<256xf32>
    %v6534 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6535 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6536 = stablehlo.multiply %v6534, %d3g1v : tensor<256xf32>
    %v6537 = stablehlo.multiply %v2332, %v2332 : tensor<256xf32>
    %v6538 = stablehlo.multiply %v6535, %v6537 : tensor<256xf32>
    %v6539 = stablehlo.add %v6536, %v6538 : tensor<256xf32>
    %v6540 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6541 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6542 = stablehlo.divide %v6533, %v6540 : tensor<256xf32>
    %v6543 = stablehlo.divide %v6539, %v6541 : tensor<256xf32>
    %v6544 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6545 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6546 = stablehlo.sqrt %v6543 : tensor<256xf32>
    %v6547 = stablehlo.add %v6546, %v6545 : tensor<256xf32>
    %v6548 = stablehlo.divide %v6542, %v6547 : tensor<256xf32>
    %v6549 = stablehlo.multiply %v6544, %v6548 : tensor<256xf32>
    %v6550 = stablehlo.subtract %d3g1, %v6549 : tensor<256xf32>
    %v6551 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6552 = stablehlo.multiply %v6551, %v6544 : tensor<256xf32>
    %v6553 = stablehlo.multiply %v6552, %d3g1 : tensor<256xf32>
    %v6554 = stablehlo.subtract %v6550, %v6553 : tensor<256xf32>
    %v6555 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6556 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6557 = stablehlo.multiply %v6555, %d3bt1m : tensor<256xf32>
    %v6558 = stablehlo.multiply %v6556, %v2335 : tensor<256xf32>
    %v6559 = stablehlo.add %v6557, %v6558 : tensor<256xf32>
    %v6560 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6561 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6562 = stablehlo.multiply %v6560, %d3bt1v : tensor<256xf32>
    %v6563 = stablehlo.multiply %v2335, %v2335 : tensor<256xf32>
    %v6564 = stablehlo.multiply %v6561, %v6563 : tensor<256xf32>
    %v6565 = stablehlo.add %v6562, %v6564 : tensor<256xf32>
    %v6566 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6567 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6568 = stablehlo.multiply %v6566, %d3bt1m : tensor<256xf32>
    %v6569 = stablehlo.multiply %v6567, %v2335 : tensor<256xf32>
    %v6570 = stablehlo.add %v6568, %v6569 : tensor<256xf32>
    %v6571 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6572 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6573 = stablehlo.multiply %v6571, %d3bt1v : tensor<256xf32>
    %v6574 = stablehlo.multiply %v2335, %v2335 : tensor<256xf32>
    %v6575 = stablehlo.multiply %v6572, %v6574 : tensor<256xf32>
    %v6576 = stablehlo.add %v6573, %v6575 : tensor<256xf32>
    %v6577 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6578 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6579 = stablehlo.divide %v6570, %v6577 : tensor<256xf32>
    %v6580 = stablehlo.divide %v6576, %v6578 : tensor<256xf32>
    %v6581 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6582 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6583 = stablehlo.sqrt %v6580 : tensor<256xf32>
    %v6584 = stablehlo.add %v6583, %v6582 : tensor<256xf32>
    %v6585 = stablehlo.divide %v6579, %v6584 : tensor<256xf32>
    %v6586 = stablehlo.multiply %v6581, %v6585 : tensor<256xf32>
    %v6587 = stablehlo.subtract %d3bt1, %v6586 : tensor<256xf32>
    %v6588 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6589 = stablehlo.multiply %v6588, %v6581 : tensor<256xf32>
    %v6590 = stablehlo.multiply %v6589, %d3bt1 : tensor<256xf32>
    %v6591 = stablehlo.subtract %v6587, %v6590 : tensor<256xf32>
    %v6592 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6593 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6594 = stablehlo.multiply %v6592, %d3W2m : tensor<256x256x3x3xf32>
    %v6595 = stablehlo.multiply %v6593, %v2341 : tensor<256x256x3x3xf32>
    %v6596 = stablehlo.add %v6594, %v6595 : tensor<256x256x3x3xf32>
    %v6597 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6598 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6599 = stablehlo.multiply %v6597, %d3W2v : tensor<256x256x3x3xf32>
    %v6600 = stablehlo.multiply %v2341, %v2341 : tensor<256x256x3x3xf32>
    %v6601 = stablehlo.multiply %v6598, %v6600 : tensor<256x256x3x3xf32>
    %v6602 = stablehlo.add %v6599, %v6601 : tensor<256x256x3x3xf32>
    %v6603 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6604 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6605 = stablehlo.multiply %v6603, %d3W2m : tensor<256x256x3x3xf32>
    %v6606 = stablehlo.multiply %v6604, %v2341 : tensor<256x256x3x3xf32>
    %v6607 = stablehlo.add %v6605, %v6606 : tensor<256x256x3x3xf32>
    %v6608 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6609 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6610 = stablehlo.multiply %v6608, %d3W2v : tensor<256x256x3x3xf32>
    %v6611 = stablehlo.multiply %v2341, %v2341 : tensor<256x256x3x3xf32>
    %v6612 = stablehlo.multiply %v6609, %v6611 : tensor<256x256x3x3xf32>
    %v6613 = stablehlo.add %v6610, %v6612 : tensor<256x256x3x3xf32>
    %v6614 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6615 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6616 = stablehlo.divide %v6607, %v6614 : tensor<256x256x3x3xf32>
    %v6617 = stablehlo.divide %v6613, %v6615 : tensor<256x256x3x3xf32>
    %v6618 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6619 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6620 = stablehlo.sqrt %v6617 : tensor<256x256x3x3xf32>
    %v6621 = stablehlo.add %v6620, %v6619 : tensor<256x256x3x3xf32>
    %v6622 = stablehlo.divide %v6616, %v6621 : tensor<256x256x3x3xf32>
    %v6623 = stablehlo.multiply %v6618, %v6622 : tensor<256x256x3x3xf32>
    %v6624 = stablehlo.subtract %d3W2, %v6623 : tensor<256x256x3x3xf32>
    %v6625 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6626 = stablehlo.multiply %v6625, %v6618 : tensor<256x256x3x3xf32>
    %v6627 = stablehlo.multiply %v6626, %d3W2 : tensor<256x256x3x3xf32>
    %v6628 = stablehlo.subtract %v6624, %v6627 : tensor<256x256x3x3xf32>
    %v6629 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6630 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6631 = stablehlo.multiply %v6629, %d3b2m : tensor<256xf32>
    %v6632 = stablehlo.multiply %v6630, %v2344 : tensor<256xf32>
    %v6633 = stablehlo.add %v6631, %v6632 : tensor<256xf32>
    %v6634 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6635 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6636 = stablehlo.multiply %v6634, %d3b2v : tensor<256xf32>
    %v6637 = stablehlo.multiply %v2344, %v2344 : tensor<256xf32>
    %v6638 = stablehlo.multiply %v6635, %v6637 : tensor<256xf32>
    %v6639 = stablehlo.add %v6636, %v6638 : tensor<256xf32>
    %v6640 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6641 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6642 = stablehlo.multiply %v6640, %d3b2m : tensor<256xf32>
    %v6643 = stablehlo.multiply %v6641, %v2344 : tensor<256xf32>
    %v6644 = stablehlo.add %v6642, %v6643 : tensor<256xf32>
    %v6645 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6646 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6647 = stablehlo.multiply %v6645, %d3b2v : tensor<256xf32>
    %v6648 = stablehlo.multiply %v2344, %v2344 : tensor<256xf32>
    %v6649 = stablehlo.multiply %v6646, %v6648 : tensor<256xf32>
    %v6650 = stablehlo.add %v6647, %v6649 : tensor<256xf32>
    %v6651 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6652 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6653 = stablehlo.divide %v6644, %v6651 : tensor<256xf32>
    %v6654 = stablehlo.divide %v6650, %v6652 : tensor<256xf32>
    %v6655 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6656 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6657 = stablehlo.sqrt %v6654 : tensor<256xf32>
    %v6658 = stablehlo.add %v6657, %v6656 : tensor<256xf32>
    %v6659 = stablehlo.divide %v6653, %v6658 : tensor<256xf32>
    %v6660 = stablehlo.multiply %v6655, %v6659 : tensor<256xf32>
    %v6661 = stablehlo.subtract %d3b2, %v6660 : tensor<256xf32>
    %v6662 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6663 = stablehlo.multiply %v6662, %v6655 : tensor<256xf32>
    %v6664 = stablehlo.multiply %v6663, %d3b2 : tensor<256xf32>
    %v6665 = stablehlo.subtract %v6661, %v6664 : tensor<256xf32>
    %v6666 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6667 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6668 = stablehlo.multiply %v6666, %d3g2m : tensor<256xf32>
    %v6669 = stablehlo.multiply %v6667, %v2362 : tensor<256xf32>
    %v6670 = stablehlo.add %v6668, %v6669 : tensor<256xf32>
    %v6671 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6672 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6673 = stablehlo.multiply %v6671, %d3g2v : tensor<256xf32>
    %v6674 = stablehlo.multiply %v2362, %v2362 : tensor<256xf32>
    %v6675 = stablehlo.multiply %v6672, %v6674 : tensor<256xf32>
    %v6676 = stablehlo.add %v6673, %v6675 : tensor<256xf32>
    %v6677 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6678 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6679 = stablehlo.multiply %v6677, %d3g2m : tensor<256xf32>
    %v6680 = stablehlo.multiply %v6678, %v2362 : tensor<256xf32>
    %v6681 = stablehlo.add %v6679, %v6680 : tensor<256xf32>
    %v6682 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6683 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6684 = stablehlo.multiply %v6682, %d3g2v : tensor<256xf32>
    %v6685 = stablehlo.multiply %v2362, %v2362 : tensor<256xf32>
    %v6686 = stablehlo.multiply %v6683, %v6685 : tensor<256xf32>
    %v6687 = stablehlo.add %v6684, %v6686 : tensor<256xf32>
    %v6688 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6689 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6690 = stablehlo.divide %v6681, %v6688 : tensor<256xf32>
    %v6691 = stablehlo.divide %v6687, %v6689 : tensor<256xf32>
    %v6692 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6693 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6694 = stablehlo.sqrt %v6691 : tensor<256xf32>
    %v6695 = stablehlo.add %v6694, %v6693 : tensor<256xf32>
    %v6696 = stablehlo.divide %v6690, %v6695 : tensor<256xf32>
    %v6697 = stablehlo.multiply %v6692, %v6696 : tensor<256xf32>
    %v6698 = stablehlo.subtract %d3g2, %v6697 : tensor<256xf32>
    %v6699 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6700 = stablehlo.multiply %v6699, %v6692 : tensor<256xf32>
    %v6701 = stablehlo.multiply %v6700, %d3g2 : tensor<256xf32>
    %v6702 = stablehlo.subtract %v6698, %v6701 : tensor<256xf32>
    %v6703 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6704 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6705 = stablehlo.multiply %v6703, %d3bt2m : tensor<256xf32>
    %v6706 = stablehlo.multiply %v6704, %v2365 : tensor<256xf32>
    %v6707 = stablehlo.add %v6705, %v6706 : tensor<256xf32>
    %v6708 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6709 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6710 = stablehlo.multiply %v6708, %d3bt2v : tensor<256xf32>
    %v6711 = stablehlo.multiply %v2365, %v2365 : tensor<256xf32>
    %v6712 = stablehlo.multiply %v6709, %v6711 : tensor<256xf32>
    %v6713 = stablehlo.add %v6710, %v6712 : tensor<256xf32>
    %v6714 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6715 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6716 = stablehlo.multiply %v6714, %d3bt2m : tensor<256xf32>
    %v6717 = stablehlo.multiply %v6715, %v2365 : tensor<256xf32>
    %v6718 = stablehlo.add %v6716, %v6717 : tensor<256xf32>
    %v6719 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6720 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6721 = stablehlo.multiply %v6719, %d3bt2v : tensor<256xf32>
    %v6722 = stablehlo.multiply %v2365, %v2365 : tensor<256xf32>
    %v6723 = stablehlo.multiply %v6720, %v6722 : tensor<256xf32>
    %v6724 = stablehlo.add %v6721, %v6723 : tensor<256xf32>
    %v6725 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6726 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6727 = stablehlo.divide %v6718, %v6725 : tensor<256xf32>
    %v6728 = stablehlo.divide %v6724, %v6726 : tensor<256xf32>
    %v6729 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6730 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6731 = stablehlo.sqrt %v6728 : tensor<256xf32>
    %v6732 = stablehlo.add %v6731, %v6730 : tensor<256xf32>
    %v6733 = stablehlo.divide %v6727, %v6732 : tensor<256xf32>
    %v6734 = stablehlo.multiply %v6729, %v6733 : tensor<256xf32>
    %v6735 = stablehlo.subtract %d3bt2, %v6734 : tensor<256xf32>
    %v6736 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6737 = stablehlo.multiply %v6736, %v6729 : tensor<256xf32>
    %v6738 = stablehlo.multiply %v6737, %d3bt2 : tensor<256xf32>
    %v6739 = stablehlo.subtract %v6735, %v6738 : tensor<256xf32>
    %v6740 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6741 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6742 = stablehlo.multiply %v6740, %d3Wpm : tensor<256x128x3x3xf32>
    %v6743 = stablehlo.multiply %v6741, %v2373 : tensor<256x128x3x3xf32>
    %v6744 = stablehlo.add %v6742, %v6743 : tensor<256x128x3x3xf32>
    %v6745 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6746 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6747 = stablehlo.multiply %v6745, %d3Wpv : tensor<256x128x3x3xf32>
    %v6748 = stablehlo.multiply %v2373, %v2373 : tensor<256x128x3x3xf32>
    %v6749 = stablehlo.multiply %v6746, %v6748 : tensor<256x128x3x3xf32>
    %v6750 = stablehlo.add %v6747, %v6749 : tensor<256x128x3x3xf32>
    %v6751 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6752 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6753 = stablehlo.multiply %v6751, %d3Wpm : tensor<256x128x3x3xf32>
    %v6754 = stablehlo.multiply %v6752, %v2373 : tensor<256x128x3x3xf32>
    %v6755 = stablehlo.add %v6753, %v6754 : tensor<256x128x3x3xf32>
    %v6756 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6757 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6758 = stablehlo.multiply %v6756, %d3Wpv : tensor<256x128x3x3xf32>
    %v6759 = stablehlo.multiply %v2373, %v2373 : tensor<256x128x3x3xf32>
    %v6760 = stablehlo.multiply %v6757, %v6759 : tensor<256x128x3x3xf32>
    %v6761 = stablehlo.add %v6758, %v6760 : tensor<256x128x3x3xf32>
    %v6762 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6763 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6764 = stablehlo.divide %v6755, %v6762 : tensor<256x128x3x3xf32>
    %v6765 = stablehlo.divide %v6761, %v6763 : tensor<256x128x3x3xf32>
    %v6766 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6767 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6768 = stablehlo.sqrt %v6765 : tensor<256x128x3x3xf32>
    %v6769 = stablehlo.add %v6768, %v6767 : tensor<256x128x3x3xf32>
    %v6770 = stablehlo.divide %v6764, %v6769 : tensor<256x128x3x3xf32>
    %v6771 = stablehlo.multiply %v6766, %v6770 : tensor<256x128x3x3xf32>
    %v6772 = stablehlo.subtract %d3Wp, %v6771 : tensor<256x128x3x3xf32>
    %v6773 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v6774 = stablehlo.multiply %v6773, %v6766 : tensor<256x128x3x3xf32>
    %v6775 = stablehlo.multiply %v6774, %d3Wp : tensor<256x128x3x3xf32>
    %v6776 = stablehlo.subtract %v6772, %v6775 : tensor<256x128x3x3xf32>
    %v6777 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6778 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6779 = stablehlo.multiply %v6777, %d3bpm : tensor<256xf32>
    %v6780 = stablehlo.multiply %v6778, %v2376 : tensor<256xf32>
    %v6781 = stablehlo.add %v6779, %v6780 : tensor<256xf32>
    %v6782 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6783 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6784 = stablehlo.multiply %v6782, %d3bpv : tensor<256xf32>
    %v6785 = stablehlo.multiply %v2376, %v2376 : tensor<256xf32>
    %v6786 = stablehlo.multiply %v6783, %v6785 : tensor<256xf32>
    %v6787 = stablehlo.add %v6784, %v6786 : tensor<256xf32>
    %v6788 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6789 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6790 = stablehlo.multiply %v6788, %d3bpm : tensor<256xf32>
    %v6791 = stablehlo.multiply %v6789, %v2376 : tensor<256xf32>
    %v6792 = stablehlo.add %v6790, %v6791 : tensor<256xf32>
    %v6793 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6794 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6795 = stablehlo.multiply %v6793, %d3bpv : tensor<256xf32>
    %v6796 = stablehlo.multiply %v2376, %v2376 : tensor<256xf32>
    %v6797 = stablehlo.multiply %v6794, %v6796 : tensor<256xf32>
    %v6798 = stablehlo.add %v6795, %v6797 : tensor<256xf32>
    %v6799 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6800 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6801 = stablehlo.divide %v6792, %v6799 : tensor<256xf32>
    %v6802 = stablehlo.divide %v6798, %v6800 : tensor<256xf32>
    %v6803 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6804 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6805 = stablehlo.sqrt %v6802 : tensor<256xf32>
    %v6806 = stablehlo.add %v6805, %v6804 : tensor<256xf32>
    %v6807 = stablehlo.divide %v6801, %v6806 : tensor<256xf32>
    %v6808 = stablehlo.multiply %v6803, %v6807 : tensor<256xf32>
    %v6809 = stablehlo.subtract %d3bp, %v6808 : tensor<256xf32>
    %v6810 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6811 = stablehlo.multiply %v6810, %v6803 : tensor<256xf32>
    %v6812 = stablehlo.multiply %v6811, %d3bp : tensor<256xf32>
    %v6813 = stablehlo.subtract %v6809, %v6812 : tensor<256xf32>
    %v6814 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6815 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6816 = stablehlo.multiply %v6814, %d3gpm : tensor<256xf32>
    %v6817 = stablehlo.multiply %v6815, %v2394 : tensor<256xf32>
    %v6818 = stablehlo.add %v6816, %v6817 : tensor<256xf32>
    %v6819 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6820 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6821 = stablehlo.multiply %v6819, %d3gpv : tensor<256xf32>
    %v6822 = stablehlo.multiply %v2394, %v2394 : tensor<256xf32>
    %v6823 = stablehlo.multiply %v6820, %v6822 : tensor<256xf32>
    %v6824 = stablehlo.add %v6821, %v6823 : tensor<256xf32>
    %v6825 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6826 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6827 = stablehlo.multiply %v6825, %d3gpm : tensor<256xf32>
    %v6828 = stablehlo.multiply %v6826, %v2394 : tensor<256xf32>
    %v6829 = stablehlo.add %v6827, %v6828 : tensor<256xf32>
    %v6830 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6831 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6832 = stablehlo.multiply %v6830, %d3gpv : tensor<256xf32>
    %v6833 = stablehlo.multiply %v2394, %v2394 : tensor<256xf32>
    %v6834 = stablehlo.multiply %v6831, %v6833 : tensor<256xf32>
    %v6835 = stablehlo.add %v6832, %v6834 : tensor<256xf32>
    %v6836 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6837 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6838 = stablehlo.divide %v6829, %v6836 : tensor<256xf32>
    %v6839 = stablehlo.divide %v6835, %v6837 : tensor<256xf32>
    %v6840 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6841 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6842 = stablehlo.sqrt %v6839 : tensor<256xf32>
    %v6843 = stablehlo.add %v6842, %v6841 : tensor<256xf32>
    %v6844 = stablehlo.divide %v6838, %v6843 : tensor<256xf32>
    %v6845 = stablehlo.multiply %v6840, %v6844 : tensor<256xf32>
    %v6846 = stablehlo.subtract %d3gp, %v6845 : tensor<256xf32>
    %v6847 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6848 = stablehlo.multiply %v6847, %v6840 : tensor<256xf32>
    %v6849 = stablehlo.multiply %v6848, %d3gp : tensor<256xf32>
    %v6850 = stablehlo.subtract %v6846, %v6849 : tensor<256xf32>
    %v6851 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6852 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6853 = stablehlo.multiply %v6851, %d3btpm : tensor<256xf32>
    %v6854 = stablehlo.multiply %v6852, %v2397 : tensor<256xf32>
    %v6855 = stablehlo.add %v6853, %v6854 : tensor<256xf32>
    %v6856 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6857 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6858 = stablehlo.multiply %v6856, %d3btpv : tensor<256xf32>
    %v6859 = stablehlo.multiply %v2397, %v2397 : tensor<256xf32>
    %v6860 = stablehlo.multiply %v6857, %v6859 : tensor<256xf32>
    %v6861 = stablehlo.add %v6858, %v6860 : tensor<256xf32>
    %v6862 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6863 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6864 = stablehlo.multiply %v6862, %d3btpm : tensor<256xf32>
    %v6865 = stablehlo.multiply %v6863, %v2397 : tensor<256xf32>
    %v6866 = stablehlo.add %v6864, %v6865 : tensor<256xf32>
    %v6867 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6868 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6869 = stablehlo.multiply %v6867, %d3btpv : tensor<256xf32>
    %v6870 = stablehlo.multiply %v2397, %v2397 : tensor<256xf32>
    %v6871 = stablehlo.multiply %v6868, %v6870 : tensor<256xf32>
    %v6872 = stablehlo.add %v6869, %v6871 : tensor<256xf32>
    %v6873 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6874 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6875 = stablehlo.divide %v6866, %v6873 : tensor<256xf32>
    %v6876 = stablehlo.divide %v6872, %v6874 : tensor<256xf32>
    %v6877 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6878 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6879 = stablehlo.sqrt %v6876 : tensor<256xf32>
    %v6880 = stablehlo.add %v6879, %v6878 : tensor<256xf32>
    %v6881 = stablehlo.divide %v6875, %v6880 : tensor<256xf32>
    %v6882 = stablehlo.multiply %v6877, %v6881 : tensor<256xf32>
    %v6883 = stablehlo.subtract %d3btp, %v6882 : tensor<256xf32>
    %v6884 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6885 = stablehlo.multiply %v6884, %v6877 : tensor<256xf32>
    %v6886 = stablehlo.multiply %v6885, %d3btp : tensor<256xf32>
    %v6887 = stablehlo.subtract %v6883, %v6886 : tensor<256xf32>
    %v6888 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6889 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6890 = stablehlo.multiply %v6888, %s3b0W1m : tensor<256x256x3x3xf32>
    %v6891 = stablehlo.multiply %v6889, %v2133 : tensor<256x256x3x3xf32>
    %v6892 = stablehlo.add %v6890, %v6891 : tensor<256x256x3x3xf32>
    %v6893 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6894 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6895 = stablehlo.multiply %v6893, %s3b0W1v : tensor<256x256x3x3xf32>
    %v6896 = stablehlo.multiply %v2133, %v2133 : tensor<256x256x3x3xf32>
    %v6897 = stablehlo.multiply %v6894, %v6896 : tensor<256x256x3x3xf32>
    %v6898 = stablehlo.add %v6895, %v6897 : tensor<256x256x3x3xf32>
    %v6899 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6900 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6901 = stablehlo.multiply %v6899, %s3b0W1m : tensor<256x256x3x3xf32>
    %v6902 = stablehlo.multiply %v6900, %v2133 : tensor<256x256x3x3xf32>
    %v6903 = stablehlo.add %v6901, %v6902 : tensor<256x256x3x3xf32>
    %v6904 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6905 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6906 = stablehlo.multiply %v6904, %s3b0W1v : tensor<256x256x3x3xf32>
    %v6907 = stablehlo.multiply %v2133, %v2133 : tensor<256x256x3x3xf32>
    %v6908 = stablehlo.multiply %v6905, %v6907 : tensor<256x256x3x3xf32>
    %v6909 = stablehlo.add %v6906, %v6908 : tensor<256x256x3x3xf32>
    %v6910 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6911 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6912 = stablehlo.divide %v6903, %v6910 : tensor<256x256x3x3xf32>
    %v6913 = stablehlo.divide %v6909, %v6911 : tensor<256x256x3x3xf32>
    %v6914 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6915 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6916 = stablehlo.sqrt %v6913 : tensor<256x256x3x3xf32>
    %v6917 = stablehlo.add %v6916, %v6915 : tensor<256x256x3x3xf32>
    %v6918 = stablehlo.divide %v6912, %v6917 : tensor<256x256x3x3xf32>
    %v6919 = stablehlo.multiply %v6914, %v6918 : tensor<256x256x3x3xf32>
    %v6920 = stablehlo.subtract %s3b0W1, %v6919 : tensor<256x256x3x3xf32>
    %v6921 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6922 = stablehlo.multiply %v6921, %v6914 : tensor<256x256x3x3xf32>
    %v6923 = stablehlo.multiply %v6922, %s3b0W1 : tensor<256x256x3x3xf32>
    %v6924 = stablehlo.subtract %v6920, %v6923 : tensor<256x256x3x3xf32>
    %v6925 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6926 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6927 = stablehlo.multiply %v6925, %s3b0b1m : tensor<256xf32>
    %v6928 = stablehlo.multiply %v6926, %v2136 : tensor<256xf32>
    %v6929 = stablehlo.add %v6927, %v6928 : tensor<256xf32>
    %v6930 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6931 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6932 = stablehlo.multiply %v6930, %s3b0b1v : tensor<256xf32>
    %v6933 = stablehlo.multiply %v2136, %v2136 : tensor<256xf32>
    %v6934 = stablehlo.multiply %v6931, %v6933 : tensor<256xf32>
    %v6935 = stablehlo.add %v6932, %v6934 : tensor<256xf32>
    %v6936 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6937 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6938 = stablehlo.multiply %v6936, %s3b0b1m : tensor<256xf32>
    %v6939 = stablehlo.multiply %v6937, %v2136 : tensor<256xf32>
    %v6940 = stablehlo.add %v6938, %v6939 : tensor<256xf32>
    %v6941 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6942 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6943 = stablehlo.multiply %v6941, %s3b0b1v : tensor<256xf32>
    %v6944 = stablehlo.multiply %v2136, %v2136 : tensor<256xf32>
    %v6945 = stablehlo.multiply %v6942, %v6944 : tensor<256xf32>
    %v6946 = stablehlo.add %v6943, %v6945 : tensor<256xf32>
    %v6947 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6948 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6949 = stablehlo.divide %v6940, %v6947 : tensor<256xf32>
    %v6950 = stablehlo.divide %v6946, %v6948 : tensor<256xf32>
    %v6951 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6952 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6953 = stablehlo.sqrt %v6950 : tensor<256xf32>
    %v6954 = stablehlo.add %v6953, %v6952 : tensor<256xf32>
    %v6955 = stablehlo.divide %v6949, %v6954 : tensor<256xf32>
    %v6956 = stablehlo.multiply %v6951, %v6955 : tensor<256xf32>
    %v6957 = stablehlo.subtract %s3b0b1, %v6956 : tensor<256xf32>
    %v6958 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6959 = stablehlo.multiply %v6958, %v6951 : tensor<256xf32>
    %v6960 = stablehlo.multiply %v6959, %s3b0b1 : tensor<256xf32>
    %v6961 = stablehlo.subtract %v6957, %v6960 : tensor<256xf32>
    %v6962 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6963 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6964 = stablehlo.multiply %v6962, %s3b0g1m : tensor<256xf32>
    %v6965 = stablehlo.multiply %v6963, %v2154 : tensor<256xf32>
    %v6966 = stablehlo.add %v6964, %v6965 : tensor<256xf32>
    %v6967 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6968 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6969 = stablehlo.multiply %v6967, %s3b0g1v : tensor<256xf32>
    %v6970 = stablehlo.multiply %v2154, %v2154 : tensor<256xf32>
    %v6971 = stablehlo.multiply %v6968, %v6970 : tensor<256xf32>
    %v6972 = stablehlo.add %v6969, %v6971 : tensor<256xf32>
    %v6973 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6974 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6975 = stablehlo.multiply %v6973, %s3b0g1m : tensor<256xf32>
    %v6976 = stablehlo.multiply %v6974, %v2154 : tensor<256xf32>
    %v6977 = stablehlo.add %v6975, %v6976 : tensor<256xf32>
    %v6978 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6979 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6980 = stablehlo.multiply %v6978, %s3b0g1v : tensor<256xf32>
    %v6981 = stablehlo.multiply %v2154, %v2154 : tensor<256xf32>
    %v6982 = stablehlo.multiply %v6979, %v6981 : tensor<256xf32>
    %v6983 = stablehlo.add %v6980, %v6982 : tensor<256xf32>
    %v6984 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6985 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6986 = stablehlo.divide %v6977, %v6984 : tensor<256xf32>
    %v6987 = stablehlo.divide %v6983, %v6985 : tensor<256xf32>
    %v6988 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6989 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6990 = stablehlo.sqrt %v6987 : tensor<256xf32>
    %v6991 = stablehlo.add %v6990, %v6989 : tensor<256xf32>
    %v6992 = stablehlo.divide %v6986, %v6991 : tensor<256xf32>
    %v6993 = stablehlo.multiply %v6988, %v6992 : tensor<256xf32>
    %v6994 = stablehlo.subtract %s3b0g1, %v6993 : tensor<256xf32>
    %v6995 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6996 = stablehlo.multiply %v6995, %v6988 : tensor<256xf32>
    %v6997 = stablehlo.multiply %v6996, %s3b0g1 : tensor<256xf32>
    %v6998 = stablehlo.subtract %v6994, %v6997 : tensor<256xf32>
    %v6999 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7000 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7001 = stablehlo.multiply %v6999, %s3b0bt1m : tensor<256xf32>
    %v7002 = stablehlo.multiply %v7000, %v2157 : tensor<256xf32>
    %v7003 = stablehlo.add %v7001, %v7002 : tensor<256xf32>
    %v7004 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7005 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7006 = stablehlo.multiply %v7004, %s3b0bt1v : tensor<256xf32>
    %v7007 = stablehlo.multiply %v2157, %v2157 : tensor<256xf32>
    %v7008 = stablehlo.multiply %v7005, %v7007 : tensor<256xf32>
    %v7009 = stablehlo.add %v7006, %v7008 : tensor<256xf32>
    %v7010 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7011 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7012 = stablehlo.multiply %v7010, %s3b0bt1m : tensor<256xf32>
    %v7013 = stablehlo.multiply %v7011, %v2157 : tensor<256xf32>
    %v7014 = stablehlo.add %v7012, %v7013 : tensor<256xf32>
    %v7015 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7016 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7017 = stablehlo.multiply %v7015, %s3b0bt1v : tensor<256xf32>
    %v7018 = stablehlo.multiply %v2157, %v2157 : tensor<256xf32>
    %v7019 = stablehlo.multiply %v7016, %v7018 : tensor<256xf32>
    %v7020 = stablehlo.add %v7017, %v7019 : tensor<256xf32>
    %v7021 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7022 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7023 = stablehlo.divide %v7014, %v7021 : tensor<256xf32>
    %v7024 = stablehlo.divide %v7020, %v7022 : tensor<256xf32>
    %v7025 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7026 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7027 = stablehlo.sqrt %v7024 : tensor<256xf32>
    %v7028 = stablehlo.add %v7027, %v7026 : tensor<256xf32>
    %v7029 = stablehlo.divide %v7023, %v7028 : tensor<256xf32>
    %v7030 = stablehlo.multiply %v7025, %v7029 : tensor<256xf32>
    %v7031 = stablehlo.subtract %s3b0bt1, %v7030 : tensor<256xf32>
    %v7032 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7033 = stablehlo.multiply %v7032, %v7025 : tensor<256xf32>
    %v7034 = stablehlo.multiply %v7033, %s3b0bt1 : tensor<256xf32>
    %v7035 = stablehlo.subtract %v7031, %v7034 : tensor<256xf32>
    %v7036 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7037 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7038 = stablehlo.multiply %v7036, %s3b0W2m : tensor<256x256x3x3xf32>
    %v7039 = stablehlo.multiply %v7037, %v2163 : tensor<256x256x3x3xf32>
    %v7040 = stablehlo.add %v7038, %v7039 : tensor<256x256x3x3xf32>
    %v7041 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7042 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7043 = stablehlo.multiply %v7041, %s3b0W2v : tensor<256x256x3x3xf32>
    %v7044 = stablehlo.multiply %v2163, %v2163 : tensor<256x256x3x3xf32>
    %v7045 = stablehlo.multiply %v7042, %v7044 : tensor<256x256x3x3xf32>
    %v7046 = stablehlo.add %v7043, %v7045 : tensor<256x256x3x3xf32>
    %v7047 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7048 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7049 = stablehlo.multiply %v7047, %s3b0W2m : tensor<256x256x3x3xf32>
    %v7050 = stablehlo.multiply %v7048, %v2163 : tensor<256x256x3x3xf32>
    %v7051 = stablehlo.add %v7049, %v7050 : tensor<256x256x3x3xf32>
    %v7052 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7053 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7054 = stablehlo.multiply %v7052, %s3b0W2v : tensor<256x256x3x3xf32>
    %v7055 = stablehlo.multiply %v2163, %v2163 : tensor<256x256x3x3xf32>
    %v7056 = stablehlo.multiply %v7053, %v7055 : tensor<256x256x3x3xf32>
    %v7057 = stablehlo.add %v7054, %v7056 : tensor<256x256x3x3xf32>
    %v7058 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7059 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7060 = stablehlo.divide %v7051, %v7058 : tensor<256x256x3x3xf32>
    %v7061 = stablehlo.divide %v7057, %v7059 : tensor<256x256x3x3xf32>
    %v7062 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7063 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7064 = stablehlo.sqrt %v7061 : tensor<256x256x3x3xf32>
    %v7065 = stablehlo.add %v7064, %v7063 : tensor<256x256x3x3xf32>
    %v7066 = stablehlo.divide %v7060, %v7065 : tensor<256x256x3x3xf32>
    %v7067 = stablehlo.multiply %v7062, %v7066 : tensor<256x256x3x3xf32>
    %v7068 = stablehlo.subtract %s3b0W2, %v7067 : tensor<256x256x3x3xf32>
    %v7069 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7070 = stablehlo.multiply %v7069, %v7062 : tensor<256x256x3x3xf32>
    %v7071 = stablehlo.multiply %v7070, %s3b0W2 : tensor<256x256x3x3xf32>
    %v7072 = stablehlo.subtract %v7068, %v7071 : tensor<256x256x3x3xf32>
    %v7073 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7074 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7075 = stablehlo.multiply %v7073, %s3b0b2m : tensor<256xf32>
    %v7076 = stablehlo.multiply %v7074, %v2166 : tensor<256xf32>
    %v7077 = stablehlo.add %v7075, %v7076 : tensor<256xf32>
    %v7078 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7079 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7080 = stablehlo.multiply %v7078, %s3b0b2v : tensor<256xf32>
    %v7081 = stablehlo.multiply %v2166, %v2166 : tensor<256xf32>
    %v7082 = stablehlo.multiply %v7079, %v7081 : tensor<256xf32>
    %v7083 = stablehlo.add %v7080, %v7082 : tensor<256xf32>
    %v7084 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7085 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7086 = stablehlo.multiply %v7084, %s3b0b2m : tensor<256xf32>
    %v7087 = stablehlo.multiply %v7085, %v2166 : tensor<256xf32>
    %v7088 = stablehlo.add %v7086, %v7087 : tensor<256xf32>
    %v7089 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7090 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7091 = stablehlo.multiply %v7089, %s3b0b2v : tensor<256xf32>
    %v7092 = stablehlo.multiply %v2166, %v2166 : tensor<256xf32>
    %v7093 = stablehlo.multiply %v7090, %v7092 : tensor<256xf32>
    %v7094 = stablehlo.add %v7091, %v7093 : tensor<256xf32>
    %v7095 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7096 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7097 = stablehlo.divide %v7088, %v7095 : tensor<256xf32>
    %v7098 = stablehlo.divide %v7094, %v7096 : tensor<256xf32>
    %v7099 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7100 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7101 = stablehlo.sqrt %v7098 : tensor<256xf32>
    %v7102 = stablehlo.add %v7101, %v7100 : tensor<256xf32>
    %v7103 = stablehlo.divide %v7097, %v7102 : tensor<256xf32>
    %v7104 = stablehlo.multiply %v7099, %v7103 : tensor<256xf32>
    %v7105 = stablehlo.subtract %s3b0b2, %v7104 : tensor<256xf32>
    %v7106 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7107 = stablehlo.multiply %v7106, %v7099 : tensor<256xf32>
    %v7108 = stablehlo.multiply %v7107, %s3b0b2 : tensor<256xf32>
    %v7109 = stablehlo.subtract %v7105, %v7108 : tensor<256xf32>
    %v7110 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7111 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7112 = stablehlo.multiply %v7110, %s3b0g2m : tensor<256xf32>
    %v7113 = stablehlo.multiply %v7111, %v2184 : tensor<256xf32>
    %v7114 = stablehlo.add %v7112, %v7113 : tensor<256xf32>
    %v7115 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7116 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7117 = stablehlo.multiply %v7115, %s3b0g2v : tensor<256xf32>
    %v7118 = stablehlo.multiply %v2184, %v2184 : tensor<256xf32>
    %v7119 = stablehlo.multiply %v7116, %v7118 : tensor<256xf32>
    %v7120 = stablehlo.add %v7117, %v7119 : tensor<256xf32>
    %v7121 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7122 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7123 = stablehlo.multiply %v7121, %s3b0g2m : tensor<256xf32>
    %v7124 = stablehlo.multiply %v7122, %v2184 : tensor<256xf32>
    %v7125 = stablehlo.add %v7123, %v7124 : tensor<256xf32>
    %v7126 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7127 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7128 = stablehlo.multiply %v7126, %s3b0g2v : tensor<256xf32>
    %v7129 = stablehlo.multiply %v2184, %v2184 : tensor<256xf32>
    %v7130 = stablehlo.multiply %v7127, %v7129 : tensor<256xf32>
    %v7131 = stablehlo.add %v7128, %v7130 : tensor<256xf32>
    %v7132 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7133 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7134 = stablehlo.divide %v7125, %v7132 : tensor<256xf32>
    %v7135 = stablehlo.divide %v7131, %v7133 : tensor<256xf32>
    %v7136 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7137 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7138 = stablehlo.sqrt %v7135 : tensor<256xf32>
    %v7139 = stablehlo.add %v7138, %v7137 : tensor<256xf32>
    %v7140 = stablehlo.divide %v7134, %v7139 : tensor<256xf32>
    %v7141 = stablehlo.multiply %v7136, %v7140 : tensor<256xf32>
    %v7142 = stablehlo.subtract %s3b0g2, %v7141 : tensor<256xf32>
    %v7143 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7144 = stablehlo.multiply %v7143, %v7136 : tensor<256xf32>
    %v7145 = stablehlo.multiply %v7144, %s3b0g2 : tensor<256xf32>
    %v7146 = stablehlo.subtract %v7142, %v7145 : tensor<256xf32>
    %v7147 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7148 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7149 = stablehlo.multiply %v7147, %s3b0bt2m : tensor<256xf32>
    %v7150 = stablehlo.multiply %v7148, %v2187 : tensor<256xf32>
    %v7151 = stablehlo.add %v7149, %v7150 : tensor<256xf32>
    %v7152 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7153 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7154 = stablehlo.multiply %v7152, %s3b0bt2v : tensor<256xf32>
    %v7155 = stablehlo.multiply %v2187, %v2187 : tensor<256xf32>
    %v7156 = stablehlo.multiply %v7153, %v7155 : tensor<256xf32>
    %v7157 = stablehlo.add %v7154, %v7156 : tensor<256xf32>
    %v7158 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7159 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7160 = stablehlo.multiply %v7158, %s3b0bt2m : tensor<256xf32>
    %v7161 = stablehlo.multiply %v7159, %v2187 : tensor<256xf32>
    %v7162 = stablehlo.add %v7160, %v7161 : tensor<256xf32>
    %v7163 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7164 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7165 = stablehlo.multiply %v7163, %s3b0bt2v : tensor<256xf32>
    %v7166 = stablehlo.multiply %v2187, %v2187 : tensor<256xf32>
    %v7167 = stablehlo.multiply %v7164, %v7166 : tensor<256xf32>
    %v7168 = stablehlo.add %v7165, %v7167 : tensor<256xf32>
    %v7169 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7170 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7171 = stablehlo.divide %v7162, %v7169 : tensor<256xf32>
    %v7172 = stablehlo.divide %v7168, %v7170 : tensor<256xf32>
    %v7173 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7174 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7175 = stablehlo.sqrt %v7172 : tensor<256xf32>
    %v7176 = stablehlo.add %v7175, %v7174 : tensor<256xf32>
    %v7177 = stablehlo.divide %v7171, %v7176 : tensor<256xf32>
    %v7178 = stablehlo.multiply %v7173, %v7177 : tensor<256xf32>
    %v7179 = stablehlo.subtract %s3b0bt2, %v7178 : tensor<256xf32>
    %v7180 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7181 = stablehlo.multiply %v7180, %v7173 : tensor<256xf32>
    %v7182 = stablehlo.multiply %v7181, %s3b0bt2 : tensor<256xf32>
    %v7183 = stablehlo.subtract %v7179, %v7182 : tensor<256xf32>
    %v7184 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7185 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7186 = stablehlo.multiply %v7184, %s3b1W1m : tensor<256x256x3x3xf32>
    %v7187 = stablehlo.multiply %v7185, %v1996 : tensor<256x256x3x3xf32>
    %v7188 = stablehlo.add %v7186, %v7187 : tensor<256x256x3x3xf32>
    %v7189 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7190 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7191 = stablehlo.multiply %v7189, %s3b1W1v : tensor<256x256x3x3xf32>
    %v7192 = stablehlo.multiply %v1996, %v1996 : tensor<256x256x3x3xf32>
    %v7193 = stablehlo.multiply %v7190, %v7192 : tensor<256x256x3x3xf32>
    %v7194 = stablehlo.add %v7191, %v7193 : tensor<256x256x3x3xf32>
    %v7195 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7196 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7197 = stablehlo.multiply %v7195, %s3b1W1m : tensor<256x256x3x3xf32>
    %v7198 = stablehlo.multiply %v7196, %v1996 : tensor<256x256x3x3xf32>
    %v7199 = stablehlo.add %v7197, %v7198 : tensor<256x256x3x3xf32>
    %v7200 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7201 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7202 = stablehlo.multiply %v7200, %s3b1W1v : tensor<256x256x3x3xf32>
    %v7203 = stablehlo.multiply %v1996, %v1996 : tensor<256x256x3x3xf32>
    %v7204 = stablehlo.multiply %v7201, %v7203 : tensor<256x256x3x3xf32>
    %v7205 = stablehlo.add %v7202, %v7204 : tensor<256x256x3x3xf32>
    %v7206 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7207 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7208 = stablehlo.divide %v7199, %v7206 : tensor<256x256x3x3xf32>
    %v7209 = stablehlo.divide %v7205, %v7207 : tensor<256x256x3x3xf32>
    %v7210 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7211 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7212 = stablehlo.sqrt %v7209 : tensor<256x256x3x3xf32>
    %v7213 = stablehlo.add %v7212, %v7211 : tensor<256x256x3x3xf32>
    %v7214 = stablehlo.divide %v7208, %v7213 : tensor<256x256x3x3xf32>
    %v7215 = stablehlo.multiply %v7210, %v7214 : tensor<256x256x3x3xf32>
    %v7216 = stablehlo.subtract %s3b1W1, %v7215 : tensor<256x256x3x3xf32>
    %v7217 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7218 = stablehlo.multiply %v7217, %v7210 : tensor<256x256x3x3xf32>
    %v7219 = stablehlo.multiply %v7218, %s3b1W1 : tensor<256x256x3x3xf32>
    %v7220 = stablehlo.subtract %v7216, %v7219 : tensor<256x256x3x3xf32>
    %v7221 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7222 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7223 = stablehlo.multiply %v7221, %s3b1b1m : tensor<256xf32>
    %v7224 = stablehlo.multiply %v7222, %v1999 : tensor<256xf32>
    %v7225 = stablehlo.add %v7223, %v7224 : tensor<256xf32>
    %v7226 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7227 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7228 = stablehlo.multiply %v7226, %s3b1b1v : tensor<256xf32>
    %v7229 = stablehlo.multiply %v1999, %v1999 : tensor<256xf32>
    %v7230 = stablehlo.multiply %v7227, %v7229 : tensor<256xf32>
    %v7231 = stablehlo.add %v7228, %v7230 : tensor<256xf32>
    %v7232 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7233 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7234 = stablehlo.multiply %v7232, %s3b1b1m : tensor<256xf32>
    %v7235 = stablehlo.multiply %v7233, %v1999 : tensor<256xf32>
    %v7236 = stablehlo.add %v7234, %v7235 : tensor<256xf32>
    %v7237 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7238 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7239 = stablehlo.multiply %v7237, %s3b1b1v : tensor<256xf32>
    %v7240 = stablehlo.multiply %v1999, %v1999 : tensor<256xf32>
    %v7241 = stablehlo.multiply %v7238, %v7240 : tensor<256xf32>
    %v7242 = stablehlo.add %v7239, %v7241 : tensor<256xf32>
    %v7243 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7244 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7245 = stablehlo.divide %v7236, %v7243 : tensor<256xf32>
    %v7246 = stablehlo.divide %v7242, %v7244 : tensor<256xf32>
    %v7247 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7248 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7249 = stablehlo.sqrt %v7246 : tensor<256xf32>
    %v7250 = stablehlo.add %v7249, %v7248 : tensor<256xf32>
    %v7251 = stablehlo.divide %v7245, %v7250 : tensor<256xf32>
    %v7252 = stablehlo.multiply %v7247, %v7251 : tensor<256xf32>
    %v7253 = stablehlo.subtract %s3b1b1, %v7252 : tensor<256xf32>
    %v7254 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7255 = stablehlo.multiply %v7254, %v7247 : tensor<256xf32>
    %v7256 = stablehlo.multiply %v7255, %s3b1b1 : tensor<256xf32>
    %v7257 = stablehlo.subtract %v7253, %v7256 : tensor<256xf32>
    %v7258 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7259 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7260 = stablehlo.multiply %v7258, %s3b1g1m : tensor<256xf32>
    %v7261 = stablehlo.multiply %v7259, %v2017 : tensor<256xf32>
    %v7262 = stablehlo.add %v7260, %v7261 : tensor<256xf32>
    %v7263 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7264 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7265 = stablehlo.multiply %v7263, %s3b1g1v : tensor<256xf32>
    %v7266 = stablehlo.multiply %v2017, %v2017 : tensor<256xf32>
    %v7267 = stablehlo.multiply %v7264, %v7266 : tensor<256xf32>
    %v7268 = stablehlo.add %v7265, %v7267 : tensor<256xf32>
    %v7269 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7270 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7271 = stablehlo.multiply %v7269, %s3b1g1m : tensor<256xf32>
    %v7272 = stablehlo.multiply %v7270, %v2017 : tensor<256xf32>
    %v7273 = stablehlo.add %v7271, %v7272 : tensor<256xf32>
    %v7274 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7275 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7276 = stablehlo.multiply %v7274, %s3b1g1v : tensor<256xf32>
    %v7277 = stablehlo.multiply %v2017, %v2017 : tensor<256xf32>
    %v7278 = stablehlo.multiply %v7275, %v7277 : tensor<256xf32>
    %v7279 = stablehlo.add %v7276, %v7278 : tensor<256xf32>
    %v7280 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7281 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7282 = stablehlo.divide %v7273, %v7280 : tensor<256xf32>
    %v7283 = stablehlo.divide %v7279, %v7281 : tensor<256xf32>
    %v7284 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7285 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7286 = stablehlo.sqrt %v7283 : tensor<256xf32>
    %v7287 = stablehlo.add %v7286, %v7285 : tensor<256xf32>
    %v7288 = stablehlo.divide %v7282, %v7287 : tensor<256xf32>
    %v7289 = stablehlo.multiply %v7284, %v7288 : tensor<256xf32>
    %v7290 = stablehlo.subtract %s3b1g1, %v7289 : tensor<256xf32>
    %v7291 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7292 = stablehlo.multiply %v7291, %v7284 : tensor<256xf32>
    %v7293 = stablehlo.multiply %v7292, %s3b1g1 : tensor<256xf32>
    %v7294 = stablehlo.subtract %v7290, %v7293 : tensor<256xf32>
    %v7295 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7296 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7297 = stablehlo.multiply %v7295, %s3b1bt1m : tensor<256xf32>
    %v7298 = stablehlo.multiply %v7296, %v2020 : tensor<256xf32>
    %v7299 = stablehlo.add %v7297, %v7298 : tensor<256xf32>
    %v7300 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7301 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7302 = stablehlo.multiply %v7300, %s3b1bt1v : tensor<256xf32>
    %v7303 = stablehlo.multiply %v2020, %v2020 : tensor<256xf32>
    %v7304 = stablehlo.multiply %v7301, %v7303 : tensor<256xf32>
    %v7305 = stablehlo.add %v7302, %v7304 : tensor<256xf32>
    %v7306 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7307 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7308 = stablehlo.multiply %v7306, %s3b1bt1m : tensor<256xf32>
    %v7309 = stablehlo.multiply %v7307, %v2020 : tensor<256xf32>
    %v7310 = stablehlo.add %v7308, %v7309 : tensor<256xf32>
    %v7311 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7312 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7313 = stablehlo.multiply %v7311, %s3b1bt1v : tensor<256xf32>
    %v7314 = stablehlo.multiply %v2020, %v2020 : tensor<256xf32>
    %v7315 = stablehlo.multiply %v7312, %v7314 : tensor<256xf32>
    %v7316 = stablehlo.add %v7313, %v7315 : tensor<256xf32>
    %v7317 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7318 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7319 = stablehlo.divide %v7310, %v7317 : tensor<256xf32>
    %v7320 = stablehlo.divide %v7316, %v7318 : tensor<256xf32>
    %v7321 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7322 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7323 = stablehlo.sqrt %v7320 : tensor<256xf32>
    %v7324 = stablehlo.add %v7323, %v7322 : tensor<256xf32>
    %v7325 = stablehlo.divide %v7319, %v7324 : tensor<256xf32>
    %v7326 = stablehlo.multiply %v7321, %v7325 : tensor<256xf32>
    %v7327 = stablehlo.subtract %s3b1bt1, %v7326 : tensor<256xf32>
    %v7328 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7329 = stablehlo.multiply %v7328, %v7321 : tensor<256xf32>
    %v7330 = stablehlo.multiply %v7329, %s3b1bt1 : tensor<256xf32>
    %v7331 = stablehlo.subtract %v7327, %v7330 : tensor<256xf32>
    %v7332 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7333 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7334 = stablehlo.multiply %v7332, %s3b1W2m : tensor<256x256x3x3xf32>
    %v7335 = stablehlo.multiply %v7333, %v2026 : tensor<256x256x3x3xf32>
    %v7336 = stablehlo.add %v7334, %v7335 : tensor<256x256x3x3xf32>
    %v7337 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7338 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7339 = stablehlo.multiply %v7337, %s3b1W2v : tensor<256x256x3x3xf32>
    %v7340 = stablehlo.multiply %v2026, %v2026 : tensor<256x256x3x3xf32>
    %v7341 = stablehlo.multiply %v7338, %v7340 : tensor<256x256x3x3xf32>
    %v7342 = stablehlo.add %v7339, %v7341 : tensor<256x256x3x3xf32>
    %v7343 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7344 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7345 = stablehlo.multiply %v7343, %s3b1W2m : tensor<256x256x3x3xf32>
    %v7346 = stablehlo.multiply %v7344, %v2026 : tensor<256x256x3x3xf32>
    %v7347 = stablehlo.add %v7345, %v7346 : tensor<256x256x3x3xf32>
    %v7348 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7349 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7350 = stablehlo.multiply %v7348, %s3b1W2v : tensor<256x256x3x3xf32>
    %v7351 = stablehlo.multiply %v2026, %v2026 : tensor<256x256x3x3xf32>
    %v7352 = stablehlo.multiply %v7349, %v7351 : tensor<256x256x3x3xf32>
    %v7353 = stablehlo.add %v7350, %v7352 : tensor<256x256x3x3xf32>
    %v7354 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7355 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7356 = stablehlo.divide %v7347, %v7354 : tensor<256x256x3x3xf32>
    %v7357 = stablehlo.divide %v7353, %v7355 : tensor<256x256x3x3xf32>
    %v7358 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7359 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7360 = stablehlo.sqrt %v7357 : tensor<256x256x3x3xf32>
    %v7361 = stablehlo.add %v7360, %v7359 : tensor<256x256x3x3xf32>
    %v7362 = stablehlo.divide %v7356, %v7361 : tensor<256x256x3x3xf32>
    %v7363 = stablehlo.multiply %v7358, %v7362 : tensor<256x256x3x3xf32>
    %v7364 = stablehlo.subtract %s3b1W2, %v7363 : tensor<256x256x3x3xf32>
    %v7365 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7366 = stablehlo.multiply %v7365, %v7358 : tensor<256x256x3x3xf32>
    %v7367 = stablehlo.multiply %v7366, %s3b1W2 : tensor<256x256x3x3xf32>
    %v7368 = stablehlo.subtract %v7364, %v7367 : tensor<256x256x3x3xf32>
    %v7369 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7370 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7371 = stablehlo.multiply %v7369, %s3b1b2m : tensor<256xf32>
    %v7372 = stablehlo.multiply %v7370, %v2029 : tensor<256xf32>
    %v7373 = stablehlo.add %v7371, %v7372 : tensor<256xf32>
    %v7374 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7375 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7376 = stablehlo.multiply %v7374, %s3b1b2v : tensor<256xf32>
    %v7377 = stablehlo.multiply %v2029, %v2029 : tensor<256xf32>
    %v7378 = stablehlo.multiply %v7375, %v7377 : tensor<256xf32>
    %v7379 = stablehlo.add %v7376, %v7378 : tensor<256xf32>
    %v7380 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7381 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7382 = stablehlo.multiply %v7380, %s3b1b2m : tensor<256xf32>
    %v7383 = stablehlo.multiply %v7381, %v2029 : tensor<256xf32>
    %v7384 = stablehlo.add %v7382, %v7383 : tensor<256xf32>
    %v7385 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7386 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7387 = stablehlo.multiply %v7385, %s3b1b2v : tensor<256xf32>
    %v7388 = stablehlo.multiply %v2029, %v2029 : tensor<256xf32>
    %v7389 = stablehlo.multiply %v7386, %v7388 : tensor<256xf32>
    %v7390 = stablehlo.add %v7387, %v7389 : tensor<256xf32>
    %v7391 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7392 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7393 = stablehlo.divide %v7384, %v7391 : tensor<256xf32>
    %v7394 = stablehlo.divide %v7390, %v7392 : tensor<256xf32>
    %v7395 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7396 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7397 = stablehlo.sqrt %v7394 : tensor<256xf32>
    %v7398 = stablehlo.add %v7397, %v7396 : tensor<256xf32>
    %v7399 = stablehlo.divide %v7393, %v7398 : tensor<256xf32>
    %v7400 = stablehlo.multiply %v7395, %v7399 : tensor<256xf32>
    %v7401 = stablehlo.subtract %s3b1b2, %v7400 : tensor<256xf32>
    %v7402 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7403 = stablehlo.multiply %v7402, %v7395 : tensor<256xf32>
    %v7404 = stablehlo.multiply %v7403, %s3b1b2 : tensor<256xf32>
    %v7405 = stablehlo.subtract %v7401, %v7404 : tensor<256xf32>
    %v7406 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7407 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7408 = stablehlo.multiply %v7406, %s3b1g2m : tensor<256xf32>
    %v7409 = stablehlo.multiply %v7407, %v2047 : tensor<256xf32>
    %v7410 = stablehlo.add %v7408, %v7409 : tensor<256xf32>
    %v7411 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7412 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7413 = stablehlo.multiply %v7411, %s3b1g2v : tensor<256xf32>
    %v7414 = stablehlo.multiply %v2047, %v2047 : tensor<256xf32>
    %v7415 = stablehlo.multiply %v7412, %v7414 : tensor<256xf32>
    %v7416 = stablehlo.add %v7413, %v7415 : tensor<256xf32>
    %v7417 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7418 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7419 = stablehlo.multiply %v7417, %s3b1g2m : tensor<256xf32>
    %v7420 = stablehlo.multiply %v7418, %v2047 : tensor<256xf32>
    %v7421 = stablehlo.add %v7419, %v7420 : tensor<256xf32>
    %v7422 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7423 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7424 = stablehlo.multiply %v7422, %s3b1g2v : tensor<256xf32>
    %v7425 = stablehlo.multiply %v2047, %v2047 : tensor<256xf32>
    %v7426 = stablehlo.multiply %v7423, %v7425 : tensor<256xf32>
    %v7427 = stablehlo.add %v7424, %v7426 : tensor<256xf32>
    %v7428 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7429 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7430 = stablehlo.divide %v7421, %v7428 : tensor<256xf32>
    %v7431 = stablehlo.divide %v7427, %v7429 : tensor<256xf32>
    %v7432 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7433 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7434 = stablehlo.sqrt %v7431 : tensor<256xf32>
    %v7435 = stablehlo.add %v7434, %v7433 : tensor<256xf32>
    %v7436 = stablehlo.divide %v7430, %v7435 : tensor<256xf32>
    %v7437 = stablehlo.multiply %v7432, %v7436 : tensor<256xf32>
    %v7438 = stablehlo.subtract %s3b1g2, %v7437 : tensor<256xf32>
    %v7439 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7440 = stablehlo.multiply %v7439, %v7432 : tensor<256xf32>
    %v7441 = stablehlo.multiply %v7440, %s3b1g2 : tensor<256xf32>
    %v7442 = stablehlo.subtract %v7438, %v7441 : tensor<256xf32>
    %v7443 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7444 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7445 = stablehlo.multiply %v7443, %s3b1bt2m : tensor<256xf32>
    %v7446 = stablehlo.multiply %v7444, %v2050 : tensor<256xf32>
    %v7447 = stablehlo.add %v7445, %v7446 : tensor<256xf32>
    %v7448 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7449 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7450 = stablehlo.multiply %v7448, %s3b1bt2v : tensor<256xf32>
    %v7451 = stablehlo.multiply %v2050, %v2050 : tensor<256xf32>
    %v7452 = stablehlo.multiply %v7449, %v7451 : tensor<256xf32>
    %v7453 = stablehlo.add %v7450, %v7452 : tensor<256xf32>
    %v7454 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7455 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7456 = stablehlo.multiply %v7454, %s3b1bt2m : tensor<256xf32>
    %v7457 = stablehlo.multiply %v7455, %v2050 : tensor<256xf32>
    %v7458 = stablehlo.add %v7456, %v7457 : tensor<256xf32>
    %v7459 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7460 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7461 = stablehlo.multiply %v7459, %s3b1bt2v : tensor<256xf32>
    %v7462 = stablehlo.multiply %v2050, %v2050 : tensor<256xf32>
    %v7463 = stablehlo.multiply %v7460, %v7462 : tensor<256xf32>
    %v7464 = stablehlo.add %v7461, %v7463 : tensor<256xf32>
    %v7465 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7466 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7467 = stablehlo.divide %v7458, %v7465 : tensor<256xf32>
    %v7468 = stablehlo.divide %v7464, %v7466 : tensor<256xf32>
    %v7469 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7470 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7471 = stablehlo.sqrt %v7468 : tensor<256xf32>
    %v7472 = stablehlo.add %v7471, %v7470 : tensor<256xf32>
    %v7473 = stablehlo.divide %v7467, %v7472 : tensor<256xf32>
    %v7474 = stablehlo.multiply %v7469, %v7473 : tensor<256xf32>
    %v7475 = stablehlo.subtract %s3b1bt2, %v7474 : tensor<256xf32>
    %v7476 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7477 = stablehlo.multiply %v7476, %v7469 : tensor<256xf32>
    %v7478 = stablehlo.multiply %v7477, %s3b1bt2 : tensor<256xf32>
    %v7479 = stablehlo.subtract %v7475, %v7478 : tensor<256xf32>
    %v7480 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7481 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7482 = stablehlo.multiply %v7480, %s3b2W1m : tensor<256x256x3x3xf32>
    %v7483 = stablehlo.multiply %v7481, %v1859 : tensor<256x256x3x3xf32>
    %v7484 = stablehlo.add %v7482, %v7483 : tensor<256x256x3x3xf32>
    %v7485 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7486 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7487 = stablehlo.multiply %v7485, %s3b2W1v : tensor<256x256x3x3xf32>
    %v7488 = stablehlo.multiply %v1859, %v1859 : tensor<256x256x3x3xf32>
    %v7489 = stablehlo.multiply %v7486, %v7488 : tensor<256x256x3x3xf32>
    %v7490 = stablehlo.add %v7487, %v7489 : tensor<256x256x3x3xf32>
    %v7491 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7492 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7493 = stablehlo.multiply %v7491, %s3b2W1m : tensor<256x256x3x3xf32>
    %v7494 = stablehlo.multiply %v7492, %v1859 : tensor<256x256x3x3xf32>
    %v7495 = stablehlo.add %v7493, %v7494 : tensor<256x256x3x3xf32>
    %v7496 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7497 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7498 = stablehlo.multiply %v7496, %s3b2W1v : tensor<256x256x3x3xf32>
    %v7499 = stablehlo.multiply %v1859, %v1859 : tensor<256x256x3x3xf32>
    %v7500 = stablehlo.multiply %v7497, %v7499 : tensor<256x256x3x3xf32>
    %v7501 = stablehlo.add %v7498, %v7500 : tensor<256x256x3x3xf32>
    %v7502 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7503 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7504 = stablehlo.divide %v7495, %v7502 : tensor<256x256x3x3xf32>
    %v7505 = stablehlo.divide %v7501, %v7503 : tensor<256x256x3x3xf32>
    %v7506 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7507 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7508 = stablehlo.sqrt %v7505 : tensor<256x256x3x3xf32>
    %v7509 = stablehlo.add %v7508, %v7507 : tensor<256x256x3x3xf32>
    %v7510 = stablehlo.divide %v7504, %v7509 : tensor<256x256x3x3xf32>
    %v7511 = stablehlo.multiply %v7506, %v7510 : tensor<256x256x3x3xf32>
    %v7512 = stablehlo.subtract %s3b2W1, %v7511 : tensor<256x256x3x3xf32>
    %v7513 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7514 = stablehlo.multiply %v7513, %v7506 : tensor<256x256x3x3xf32>
    %v7515 = stablehlo.multiply %v7514, %s3b2W1 : tensor<256x256x3x3xf32>
    %v7516 = stablehlo.subtract %v7512, %v7515 : tensor<256x256x3x3xf32>
    %v7517 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7518 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7519 = stablehlo.multiply %v7517, %s3b2b1m : tensor<256xf32>
    %v7520 = stablehlo.multiply %v7518, %v1862 : tensor<256xf32>
    %v7521 = stablehlo.add %v7519, %v7520 : tensor<256xf32>
    %v7522 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7523 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7524 = stablehlo.multiply %v7522, %s3b2b1v : tensor<256xf32>
    %v7525 = stablehlo.multiply %v1862, %v1862 : tensor<256xf32>
    %v7526 = stablehlo.multiply %v7523, %v7525 : tensor<256xf32>
    %v7527 = stablehlo.add %v7524, %v7526 : tensor<256xf32>
    %v7528 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7529 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7530 = stablehlo.multiply %v7528, %s3b2b1m : tensor<256xf32>
    %v7531 = stablehlo.multiply %v7529, %v1862 : tensor<256xf32>
    %v7532 = stablehlo.add %v7530, %v7531 : tensor<256xf32>
    %v7533 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7534 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7535 = stablehlo.multiply %v7533, %s3b2b1v : tensor<256xf32>
    %v7536 = stablehlo.multiply %v1862, %v1862 : tensor<256xf32>
    %v7537 = stablehlo.multiply %v7534, %v7536 : tensor<256xf32>
    %v7538 = stablehlo.add %v7535, %v7537 : tensor<256xf32>
    %v7539 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7540 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7541 = stablehlo.divide %v7532, %v7539 : tensor<256xf32>
    %v7542 = stablehlo.divide %v7538, %v7540 : tensor<256xf32>
    %v7543 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7544 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7545 = stablehlo.sqrt %v7542 : tensor<256xf32>
    %v7546 = stablehlo.add %v7545, %v7544 : tensor<256xf32>
    %v7547 = stablehlo.divide %v7541, %v7546 : tensor<256xf32>
    %v7548 = stablehlo.multiply %v7543, %v7547 : tensor<256xf32>
    %v7549 = stablehlo.subtract %s3b2b1, %v7548 : tensor<256xf32>
    %v7550 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7551 = stablehlo.multiply %v7550, %v7543 : tensor<256xf32>
    %v7552 = stablehlo.multiply %v7551, %s3b2b1 : tensor<256xf32>
    %v7553 = stablehlo.subtract %v7549, %v7552 : tensor<256xf32>
    %v7554 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7555 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7556 = stablehlo.multiply %v7554, %s3b2g1m : tensor<256xf32>
    %v7557 = stablehlo.multiply %v7555, %v1880 : tensor<256xf32>
    %v7558 = stablehlo.add %v7556, %v7557 : tensor<256xf32>
    %v7559 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7560 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7561 = stablehlo.multiply %v7559, %s3b2g1v : tensor<256xf32>
    %v7562 = stablehlo.multiply %v1880, %v1880 : tensor<256xf32>
    %v7563 = stablehlo.multiply %v7560, %v7562 : tensor<256xf32>
    %v7564 = stablehlo.add %v7561, %v7563 : tensor<256xf32>
    %v7565 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7566 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7567 = stablehlo.multiply %v7565, %s3b2g1m : tensor<256xf32>
    %v7568 = stablehlo.multiply %v7566, %v1880 : tensor<256xf32>
    %v7569 = stablehlo.add %v7567, %v7568 : tensor<256xf32>
    %v7570 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7571 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7572 = stablehlo.multiply %v7570, %s3b2g1v : tensor<256xf32>
    %v7573 = stablehlo.multiply %v1880, %v1880 : tensor<256xf32>
    %v7574 = stablehlo.multiply %v7571, %v7573 : tensor<256xf32>
    %v7575 = stablehlo.add %v7572, %v7574 : tensor<256xf32>
    %v7576 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7577 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7578 = stablehlo.divide %v7569, %v7576 : tensor<256xf32>
    %v7579 = stablehlo.divide %v7575, %v7577 : tensor<256xf32>
    %v7580 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7581 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7582 = stablehlo.sqrt %v7579 : tensor<256xf32>
    %v7583 = stablehlo.add %v7582, %v7581 : tensor<256xf32>
    %v7584 = stablehlo.divide %v7578, %v7583 : tensor<256xf32>
    %v7585 = stablehlo.multiply %v7580, %v7584 : tensor<256xf32>
    %v7586 = stablehlo.subtract %s3b2g1, %v7585 : tensor<256xf32>
    %v7587 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7588 = stablehlo.multiply %v7587, %v7580 : tensor<256xf32>
    %v7589 = stablehlo.multiply %v7588, %s3b2g1 : tensor<256xf32>
    %v7590 = stablehlo.subtract %v7586, %v7589 : tensor<256xf32>
    %v7591 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7592 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7593 = stablehlo.multiply %v7591, %s3b2bt1m : tensor<256xf32>
    %v7594 = stablehlo.multiply %v7592, %v1883 : tensor<256xf32>
    %v7595 = stablehlo.add %v7593, %v7594 : tensor<256xf32>
    %v7596 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7597 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7598 = stablehlo.multiply %v7596, %s3b2bt1v : tensor<256xf32>
    %v7599 = stablehlo.multiply %v1883, %v1883 : tensor<256xf32>
    %v7600 = stablehlo.multiply %v7597, %v7599 : tensor<256xf32>
    %v7601 = stablehlo.add %v7598, %v7600 : tensor<256xf32>
    %v7602 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7603 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7604 = stablehlo.multiply %v7602, %s3b2bt1m : tensor<256xf32>
    %v7605 = stablehlo.multiply %v7603, %v1883 : tensor<256xf32>
    %v7606 = stablehlo.add %v7604, %v7605 : tensor<256xf32>
    %v7607 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7608 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7609 = stablehlo.multiply %v7607, %s3b2bt1v : tensor<256xf32>
    %v7610 = stablehlo.multiply %v1883, %v1883 : tensor<256xf32>
    %v7611 = stablehlo.multiply %v7608, %v7610 : tensor<256xf32>
    %v7612 = stablehlo.add %v7609, %v7611 : tensor<256xf32>
    %v7613 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7614 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7615 = stablehlo.divide %v7606, %v7613 : tensor<256xf32>
    %v7616 = stablehlo.divide %v7612, %v7614 : tensor<256xf32>
    %v7617 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7618 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7619 = stablehlo.sqrt %v7616 : tensor<256xf32>
    %v7620 = stablehlo.add %v7619, %v7618 : tensor<256xf32>
    %v7621 = stablehlo.divide %v7615, %v7620 : tensor<256xf32>
    %v7622 = stablehlo.multiply %v7617, %v7621 : tensor<256xf32>
    %v7623 = stablehlo.subtract %s3b2bt1, %v7622 : tensor<256xf32>
    %v7624 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7625 = stablehlo.multiply %v7624, %v7617 : tensor<256xf32>
    %v7626 = stablehlo.multiply %v7625, %s3b2bt1 : tensor<256xf32>
    %v7627 = stablehlo.subtract %v7623, %v7626 : tensor<256xf32>
    %v7628 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7629 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7630 = stablehlo.multiply %v7628, %s3b2W2m : tensor<256x256x3x3xf32>
    %v7631 = stablehlo.multiply %v7629, %v1889 : tensor<256x256x3x3xf32>
    %v7632 = stablehlo.add %v7630, %v7631 : tensor<256x256x3x3xf32>
    %v7633 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7634 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7635 = stablehlo.multiply %v7633, %s3b2W2v : tensor<256x256x3x3xf32>
    %v7636 = stablehlo.multiply %v1889, %v1889 : tensor<256x256x3x3xf32>
    %v7637 = stablehlo.multiply %v7634, %v7636 : tensor<256x256x3x3xf32>
    %v7638 = stablehlo.add %v7635, %v7637 : tensor<256x256x3x3xf32>
    %v7639 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7640 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7641 = stablehlo.multiply %v7639, %s3b2W2m : tensor<256x256x3x3xf32>
    %v7642 = stablehlo.multiply %v7640, %v1889 : tensor<256x256x3x3xf32>
    %v7643 = stablehlo.add %v7641, %v7642 : tensor<256x256x3x3xf32>
    %v7644 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7645 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7646 = stablehlo.multiply %v7644, %s3b2W2v : tensor<256x256x3x3xf32>
    %v7647 = stablehlo.multiply %v1889, %v1889 : tensor<256x256x3x3xf32>
    %v7648 = stablehlo.multiply %v7645, %v7647 : tensor<256x256x3x3xf32>
    %v7649 = stablehlo.add %v7646, %v7648 : tensor<256x256x3x3xf32>
    %v7650 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7651 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7652 = stablehlo.divide %v7643, %v7650 : tensor<256x256x3x3xf32>
    %v7653 = stablehlo.divide %v7649, %v7651 : tensor<256x256x3x3xf32>
    %v7654 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7655 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7656 = stablehlo.sqrt %v7653 : tensor<256x256x3x3xf32>
    %v7657 = stablehlo.add %v7656, %v7655 : tensor<256x256x3x3xf32>
    %v7658 = stablehlo.divide %v7652, %v7657 : tensor<256x256x3x3xf32>
    %v7659 = stablehlo.multiply %v7654, %v7658 : tensor<256x256x3x3xf32>
    %v7660 = stablehlo.subtract %s3b2W2, %v7659 : tensor<256x256x3x3xf32>
    %v7661 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7662 = stablehlo.multiply %v7661, %v7654 : tensor<256x256x3x3xf32>
    %v7663 = stablehlo.multiply %v7662, %s3b2W2 : tensor<256x256x3x3xf32>
    %v7664 = stablehlo.subtract %v7660, %v7663 : tensor<256x256x3x3xf32>
    %v7665 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7666 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7667 = stablehlo.multiply %v7665, %s3b2b2m : tensor<256xf32>
    %v7668 = stablehlo.multiply %v7666, %v1892 : tensor<256xf32>
    %v7669 = stablehlo.add %v7667, %v7668 : tensor<256xf32>
    %v7670 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7671 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7672 = stablehlo.multiply %v7670, %s3b2b2v : tensor<256xf32>
    %v7673 = stablehlo.multiply %v1892, %v1892 : tensor<256xf32>
    %v7674 = stablehlo.multiply %v7671, %v7673 : tensor<256xf32>
    %v7675 = stablehlo.add %v7672, %v7674 : tensor<256xf32>
    %v7676 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7677 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7678 = stablehlo.multiply %v7676, %s3b2b2m : tensor<256xf32>
    %v7679 = stablehlo.multiply %v7677, %v1892 : tensor<256xf32>
    %v7680 = stablehlo.add %v7678, %v7679 : tensor<256xf32>
    %v7681 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7682 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7683 = stablehlo.multiply %v7681, %s3b2b2v : tensor<256xf32>
    %v7684 = stablehlo.multiply %v1892, %v1892 : tensor<256xf32>
    %v7685 = stablehlo.multiply %v7682, %v7684 : tensor<256xf32>
    %v7686 = stablehlo.add %v7683, %v7685 : tensor<256xf32>
    %v7687 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7688 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7689 = stablehlo.divide %v7680, %v7687 : tensor<256xf32>
    %v7690 = stablehlo.divide %v7686, %v7688 : tensor<256xf32>
    %v7691 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7692 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7693 = stablehlo.sqrt %v7690 : tensor<256xf32>
    %v7694 = stablehlo.add %v7693, %v7692 : tensor<256xf32>
    %v7695 = stablehlo.divide %v7689, %v7694 : tensor<256xf32>
    %v7696 = stablehlo.multiply %v7691, %v7695 : tensor<256xf32>
    %v7697 = stablehlo.subtract %s3b2b2, %v7696 : tensor<256xf32>
    %v7698 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7699 = stablehlo.multiply %v7698, %v7691 : tensor<256xf32>
    %v7700 = stablehlo.multiply %v7699, %s3b2b2 : tensor<256xf32>
    %v7701 = stablehlo.subtract %v7697, %v7700 : tensor<256xf32>
    %v7702 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7703 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7704 = stablehlo.multiply %v7702, %s3b2g2m : tensor<256xf32>
    %v7705 = stablehlo.multiply %v7703, %v1910 : tensor<256xf32>
    %v7706 = stablehlo.add %v7704, %v7705 : tensor<256xf32>
    %v7707 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7708 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7709 = stablehlo.multiply %v7707, %s3b2g2v : tensor<256xf32>
    %v7710 = stablehlo.multiply %v1910, %v1910 : tensor<256xf32>
    %v7711 = stablehlo.multiply %v7708, %v7710 : tensor<256xf32>
    %v7712 = stablehlo.add %v7709, %v7711 : tensor<256xf32>
    %v7713 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7714 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7715 = stablehlo.multiply %v7713, %s3b2g2m : tensor<256xf32>
    %v7716 = stablehlo.multiply %v7714, %v1910 : tensor<256xf32>
    %v7717 = stablehlo.add %v7715, %v7716 : tensor<256xf32>
    %v7718 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7719 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7720 = stablehlo.multiply %v7718, %s3b2g2v : tensor<256xf32>
    %v7721 = stablehlo.multiply %v1910, %v1910 : tensor<256xf32>
    %v7722 = stablehlo.multiply %v7719, %v7721 : tensor<256xf32>
    %v7723 = stablehlo.add %v7720, %v7722 : tensor<256xf32>
    %v7724 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7725 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7726 = stablehlo.divide %v7717, %v7724 : tensor<256xf32>
    %v7727 = stablehlo.divide %v7723, %v7725 : tensor<256xf32>
    %v7728 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7729 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7730 = stablehlo.sqrt %v7727 : tensor<256xf32>
    %v7731 = stablehlo.add %v7730, %v7729 : tensor<256xf32>
    %v7732 = stablehlo.divide %v7726, %v7731 : tensor<256xf32>
    %v7733 = stablehlo.multiply %v7728, %v7732 : tensor<256xf32>
    %v7734 = stablehlo.subtract %s3b2g2, %v7733 : tensor<256xf32>
    %v7735 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7736 = stablehlo.multiply %v7735, %v7728 : tensor<256xf32>
    %v7737 = stablehlo.multiply %v7736, %s3b2g2 : tensor<256xf32>
    %v7738 = stablehlo.subtract %v7734, %v7737 : tensor<256xf32>
    %v7739 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7740 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7741 = stablehlo.multiply %v7739, %s3b2bt2m : tensor<256xf32>
    %v7742 = stablehlo.multiply %v7740, %v1913 : tensor<256xf32>
    %v7743 = stablehlo.add %v7741, %v7742 : tensor<256xf32>
    %v7744 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7745 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7746 = stablehlo.multiply %v7744, %s3b2bt2v : tensor<256xf32>
    %v7747 = stablehlo.multiply %v1913, %v1913 : tensor<256xf32>
    %v7748 = stablehlo.multiply %v7745, %v7747 : tensor<256xf32>
    %v7749 = stablehlo.add %v7746, %v7748 : tensor<256xf32>
    %v7750 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7751 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7752 = stablehlo.multiply %v7750, %s3b2bt2m : tensor<256xf32>
    %v7753 = stablehlo.multiply %v7751, %v1913 : tensor<256xf32>
    %v7754 = stablehlo.add %v7752, %v7753 : tensor<256xf32>
    %v7755 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7756 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7757 = stablehlo.multiply %v7755, %s3b2bt2v : tensor<256xf32>
    %v7758 = stablehlo.multiply %v1913, %v1913 : tensor<256xf32>
    %v7759 = stablehlo.multiply %v7756, %v7758 : tensor<256xf32>
    %v7760 = stablehlo.add %v7757, %v7759 : tensor<256xf32>
    %v7761 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7762 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7763 = stablehlo.divide %v7754, %v7761 : tensor<256xf32>
    %v7764 = stablehlo.divide %v7760, %v7762 : tensor<256xf32>
    %v7765 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7766 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7767 = stablehlo.sqrt %v7764 : tensor<256xf32>
    %v7768 = stablehlo.add %v7767, %v7766 : tensor<256xf32>
    %v7769 = stablehlo.divide %v7763, %v7768 : tensor<256xf32>
    %v7770 = stablehlo.multiply %v7765, %v7769 : tensor<256xf32>
    %v7771 = stablehlo.subtract %s3b2bt2, %v7770 : tensor<256xf32>
    %v7772 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7773 = stablehlo.multiply %v7772, %v7765 : tensor<256xf32>
    %v7774 = stablehlo.multiply %v7773, %s3b2bt2 : tensor<256xf32>
    %v7775 = stablehlo.subtract %v7771, %v7774 : tensor<256xf32>
    %v7776 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7777 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7778 = stablehlo.multiply %v7776, %s3b3W1m : tensor<256x256x3x3xf32>
    %v7779 = stablehlo.multiply %v7777, %v1722 : tensor<256x256x3x3xf32>
    %v7780 = stablehlo.add %v7778, %v7779 : tensor<256x256x3x3xf32>
    %v7781 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7782 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7783 = stablehlo.multiply %v7781, %s3b3W1v : tensor<256x256x3x3xf32>
    %v7784 = stablehlo.multiply %v1722, %v1722 : tensor<256x256x3x3xf32>
    %v7785 = stablehlo.multiply %v7782, %v7784 : tensor<256x256x3x3xf32>
    %v7786 = stablehlo.add %v7783, %v7785 : tensor<256x256x3x3xf32>
    %v7787 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7788 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7789 = stablehlo.multiply %v7787, %s3b3W1m : tensor<256x256x3x3xf32>
    %v7790 = stablehlo.multiply %v7788, %v1722 : tensor<256x256x3x3xf32>
    %v7791 = stablehlo.add %v7789, %v7790 : tensor<256x256x3x3xf32>
    %v7792 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7793 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7794 = stablehlo.multiply %v7792, %s3b3W1v : tensor<256x256x3x3xf32>
    %v7795 = stablehlo.multiply %v1722, %v1722 : tensor<256x256x3x3xf32>
    %v7796 = stablehlo.multiply %v7793, %v7795 : tensor<256x256x3x3xf32>
    %v7797 = stablehlo.add %v7794, %v7796 : tensor<256x256x3x3xf32>
    %v7798 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7799 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7800 = stablehlo.divide %v7791, %v7798 : tensor<256x256x3x3xf32>
    %v7801 = stablehlo.divide %v7797, %v7799 : tensor<256x256x3x3xf32>
    %v7802 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7803 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7804 = stablehlo.sqrt %v7801 : tensor<256x256x3x3xf32>
    %v7805 = stablehlo.add %v7804, %v7803 : tensor<256x256x3x3xf32>
    %v7806 = stablehlo.divide %v7800, %v7805 : tensor<256x256x3x3xf32>
    %v7807 = stablehlo.multiply %v7802, %v7806 : tensor<256x256x3x3xf32>
    %v7808 = stablehlo.subtract %s3b3W1, %v7807 : tensor<256x256x3x3xf32>
    %v7809 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7810 = stablehlo.multiply %v7809, %v7802 : tensor<256x256x3x3xf32>
    %v7811 = stablehlo.multiply %v7810, %s3b3W1 : tensor<256x256x3x3xf32>
    %v7812 = stablehlo.subtract %v7808, %v7811 : tensor<256x256x3x3xf32>
    %v7813 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7814 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7815 = stablehlo.multiply %v7813, %s3b3b1m : tensor<256xf32>
    %v7816 = stablehlo.multiply %v7814, %v1725 : tensor<256xf32>
    %v7817 = stablehlo.add %v7815, %v7816 : tensor<256xf32>
    %v7818 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7819 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7820 = stablehlo.multiply %v7818, %s3b3b1v : tensor<256xf32>
    %v7821 = stablehlo.multiply %v1725, %v1725 : tensor<256xf32>
    %v7822 = stablehlo.multiply %v7819, %v7821 : tensor<256xf32>
    %v7823 = stablehlo.add %v7820, %v7822 : tensor<256xf32>
    %v7824 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7825 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7826 = stablehlo.multiply %v7824, %s3b3b1m : tensor<256xf32>
    %v7827 = stablehlo.multiply %v7825, %v1725 : tensor<256xf32>
    %v7828 = stablehlo.add %v7826, %v7827 : tensor<256xf32>
    %v7829 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7830 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7831 = stablehlo.multiply %v7829, %s3b3b1v : tensor<256xf32>
    %v7832 = stablehlo.multiply %v1725, %v1725 : tensor<256xf32>
    %v7833 = stablehlo.multiply %v7830, %v7832 : tensor<256xf32>
    %v7834 = stablehlo.add %v7831, %v7833 : tensor<256xf32>
    %v7835 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7836 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7837 = stablehlo.divide %v7828, %v7835 : tensor<256xf32>
    %v7838 = stablehlo.divide %v7834, %v7836 : tensor<256xf32>
    %v7839 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7840 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7841 = stablehlo.sqrt %v7838 : tensor<256xf32>
    %v7842 = stablehlo.add %v7841, %v7840 : tensor<256xf32>
    %v7843 = stablehlo.divide %v7837, %v7842 : tensor<256xf32>
    %v7844 = stablehlo.multiply %v7839, %v7843 : tensor<256xf32>
    %v7845 = stablehlo.subtract %s3b3b1, %v7844 : tensor<256xf32>
    %v7846 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7847 = stablehlo.multiply %v7846, %v7839 : tensor<256xf32>
    %v7848 = stablehlo.multiply %v7847, %s3b3b1 : tensor<256xf32>
    %v7849 = stablehlo.subtract %v7845, %v7848 : tensor<256xf32>
    %v7850 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7851 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7852 = stablehlo.multiply %v7850, %s3b3g1m : tensor<256xf32>
    %v7853 = stablehlo.multiply %v7851, %v1743 : tensor<256xf32>
    %v7854 = stablehlo.add %v7852, %v7853 : tensor<256xf32>
    %v7855 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7856 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7857 = stablehlo.multiply %v7855, %s3b3g1v : tensor<256xf32>
    %v7858 = stablehlo.multiply %v1743, %v1743 : tensor<256xf32>
    %v7859 = stablehlo.multiply %v7856, %v7858 : tensor<256xf32>
    %v7860 = stablehlo.add %v7857, %v7859 : tensor<256xf32>
    %v7861 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7862 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7863 = stablehlo.multiply %v7861, %s3b3g1m : tensor<256xf32>
    %v7864 = stablehlo.multiply %v7862, %v1743 : tensor<256xf32>
    %v7865 = stablehlo.add %v7863, %v7864 : tensor<256xf32>
    %v7866 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7867 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7868 = stablehlo.multiply %v7866, %s3b3g1v : tensor<256xf32>
    %v7869 = stablehlo.multiply %v1743, %v1743 : tensor<256xf32>
    %v7870 = stablehlo.multiply %v7867, %v7869 : tensor<256xf32>
    %v7871 = stablehlo.add %v7868, %v7870 : tensor<256xf32>
    %v7872 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7873 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7874 = stablehlo.divide %v7865, %v7872 : tensor<256xf32>
    %v7875 = stablehlo.divide %v7871, %v7873 : tensor<256xf32>
    %v7876 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7877 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7878 = stablehlo.sqrt %v7875 : tensor<256xf32>
    %v7879 = stablehlo.add %v7878, %v7877 : tensor<256xf32>
    %v7880 = stablehlo.divide %v7874, %v7879 : tensor<256xf32>
    %v7881 = stablehlo.multiply %v7876, %v7880 : tensor<256xf32>
    %v7882 = stablehlo.subtract %s3b3g1, %v7881 : tensor<256xf32>
    %v7883 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7884 = stablehlo.multiply %v7883, %v7876 : tensor<256xf32>
    %v7885 = stablehlo.multiply %v7884, %s3b3g1 : tensor<256xf32>
    %v7886 = stablehlo.subtract %v7882, %v7885 : tensor<256xf32>
    %v7887 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7888 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7889 = stablehlo.multiply %v7887, %s3b3bt1m : tensor<256xf32>
    %v7890 = stablehlo.multiply %v7888, %v1746 : tensor<256xf32>
    %v7891 = stablehlo.add %v7889, %v7890 : tensor<256xf32>
    %v7892 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7893 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7894 = stablehlo.multiply %v7892, %s3b3bt1v : tensor<256xf32>
    %v7895 = stablehlo.multiply %v1746, %v1746 : tensor<256xf32>
    %v7896 = stablehlo.multiply %v7893, %v7895 : tensor<256xf32>
    %v7897 = stablehlo.add %v7894, %v7896 : tensor<256xf32>
    %v7898 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7899 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7900 = stablehlo.multiply %v7898, %s3b3bt1m : tensor<256xf32>
    %v7901 = stablehlo.multiply %v7899, %v1746 : tensor<256xf32>
    %v7902 = stablehlo.add %v7900, %v7901 : tensor<256xf32>
    %v7903 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7904 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7905 = stablehlo.multiply %v7903, %s3b3bt1v : tensor<256xf32>
    %v7906 = stablehlo.multiply %v1746, %v1746 : tensor<256xf32>
    %v7907 = stablehlo.multiply %v7904, %v7906 : tensor<256xf32>
    %v7908 = stablehlo.add %v7905, %v7907 : tensor<256xf32>
    %v7909 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7910 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7911 = stablehlo.divide %v7902, %v7909 : tensor<256xf32>
    %v7912 = stablehlo.divide %v7908, %v7910 : tensor<256xf32>
    %v7913 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7914 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7915 = stablehlo.sqrt %v7912 : tensor<256xf32>
    %v7916 = stablehlo.add %v7915, %v7914 : tensor<256xf32>
    %v7917 = stablehlo.divide %v7911, %v7916 : tensor<256xf32>
    %v7918 = stablehlo.multiply %v7913, %v7917 : tensor<256xf32>
    %v7919 = stablehlo.subtract %s3b3bt1, %v7918 : tensor<256xf32>
    %v7920 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7921 = stablehlo.multiply %v7920, %v7913 : tensor<256xf32>
    %v7922 = stablehlo.multiply %v7921, %s3b3bt1 : tensor<256xf32>
    %v7923 = stablehlo.subtract %v7919, %v7922 : tensor<256xf32>
    %v7924 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7925 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7926 = stablehlo.multiply %v7924, %s3b3W2m : tensor<256x256x3x3xf32>
    %v7927 = stablehlo.multiply %v7925, %v1752 : tensor<256x256x3x3xf32>
    %v7928 = stablehlo.add %v7926, %v7927 : tensor<256x256x3x3xf32>
    %v7929 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7930 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7931 = stablehlo.multiply %v7929, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7932 = stablehlo.multiply %v1752, %v1752 : tensor<256x256x3x3xf32>
    %v7933 = stablehlo.multiply %v7930, %v7932 : tensor<256x256x3x3xf32>
    %v7934 = stablehlo.add %v7931, %v7933 : tensor<256x256x3x3xf32>
    %v7935 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7936 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7937 = stablehlo.multiply %v7935, %s3b3W2m : tensor<256x256x3x3xf32>
    %v7938 = stablehlo.multiply %v7936, %v1752 : tensor<256x256x3x3xf32>
    %v7939 = stablehlo.add %v7937, %v7938 : tensor<256x256x3x3xf32>
    %v7940 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7941 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7942 = stablehlo.multiply %v7940, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7943 = stablehlo.multiply %v1752, %v1752 : tensor<256x256x3x3xf32>
    %v7944 = stablehlo.multiply %v7941, %v7943 : tensor<256x256x3x3xf32>
    %v7945 = stablehlo.add %v7942, %v7944 : tensor<256x256x3x3xf32>
    %v7946 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7947 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7948 = stablehlo.divide %v7939, %v7946 : tensor<256x256x3x3xf32>
    %v7949 = stablehlo.divide %v7945, %v7947 : tensor<256x256x3x3xf32>
    %v7950 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7951 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7952 = stablehlo.sqrt %v7949 : tensor<256x256x3x3xf32>
    %v7953 = stablehlo.add %v7952, %v7951 : tensor<256x256x3x3xf32>
    %v7954 = stablehlo.divide %v7948, %v7953 : tensor<256x256x3x3xf32>
    %v7955 = stablehlo.multiply %v7950, %v7954 : tensor<256x256x3x3xf32>
    %v7956 = stablehlo.subtract %s3b3W2, %v7955 : tensor<256x256x3x3xf32>
    %v7957 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7958 = stablehlo.multiply %v7957, %v7950 : tensor<256x256x3x3xf32>
    %v7959 = stablehlo.multiply %v7958, %s3b3W2 : tensor<256x256x3x3xf32>
    %v7960 = stablehlo.subtract %v7956, %v7959 : tensor<256x256x3x3xf32>
    %v7961 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7962 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7963 = stablehlo.multiply %v7961, %s3b3b2m : tensor<256xf32>
    %v7964 = stablehlo.multiply %v7962, %v1755 : tensor<256xf32>
    %v7965 = stablehlo.add %v7963, %v7964 : tensor<256xf32>
    %v7966 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7967 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7968 = stablehlo.multiply %v7966, %s3b3b2v : tensor<256xf32>
    %v7969 = stablehlo.multiply %v1755, %v1755 : tensor<256xf32>
    %v7970 = stablehlo.multiply %v7967, %v7969 : tensor<256xf32>
    %v7971 = stablehlo.add %v7968, %v7970 : tensor<256xf32>
    %v7972 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7973 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7974 = stablehlo.multiply %v7972, %s3b3b2m : tensor<256xf32>
    %v7975 = stablehlo.multiply %v7973, %v1755 : tensor<256xf32>
    %v7976 = stablehlo.add %v7974, %v7975 : tensor<256xf32>
    %v7977 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7978 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7979 = stablehlo.multiply %v7977, %s3b3b2v : tensor<256xf32>
    %v7980 = stablehlo.multiply %v1755, %v1755 : tensor<256xf32>
    %v7981 = stablehlo.multiply %v7978, %v7980 : tensor<256xf32>
    %v7982 = stablehlo.add %v7979, %v7981 : tensor<256xf32>
    %v7983 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7984 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7985 = stablehlo.divide %v7976, %v7983 : tensor<256xf32>
    %v7986 = stablehlo.divide %v7982, %v7984 : tensor<256xf32>
    %v7987 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7988 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7989 = stablehlo.sqrt %v7986 : tensor<256xf32>
    %v7990 = stablehlo.add %v7989, %v7988 : tensor<256xf32>
    %v7991 = stablehlo.divide %v7985, %v7990 : tensor<256xf32>
    %v7992 = stablehlo.multiply %v7987, %v7991 : tensor<256xf32>
    %v7993 = stablehlo.subtract %s3b3b2, %v7992 : tensor<256xf32>
    %v7994 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7995 = stablehlo.multiply %v7994, %v7987 : tensor<256xf32>
    %v7996 = stablehlo.multiply %v7995, %s3b3b2 : tensor<256xf32>
    %v7997 = stablehlo.subtract %v7993, %v7996 : tensor<256xf32>
    %v7998 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7999 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8000 = stablehlo.multiply %v7998, %s3b3g2m : tensor<256xf32>
    %v8001 = stablehlo.multiply %v7999, %v1773 : tensor<256xf32>
    %v8002 = stablehlo.add %v8000, %v8001 : tensor<256xf32>
    %v8003 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8004 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8005 = stablehlo.multiply %v8003, %s3b3g2v : tensor<256xf32>
    %v8006 = stablehlo.multiply %v1773, %v1773 : tensor<256xf32>
    %v8007 = stablehlo.multiply %v8004, %v8006 : tensor<256xf32>
    %v8008 = stablehlo.add %v8005, %v8007 : tensor<256xf32>
    %v8009 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8010 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8011 = stablehlo.multiply %v8009, %s3b3g2m : tensor<256xf32>
    %v8012 = stablehlo.multiply %v8010, %v1773 : tensor<256xf32>
    %v8013 = stablehlo.add %v8011, %v8012 : tensor<256xf32>
    %v8014 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8015 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8016 = stablehlo.multiply %v8014, %s3b3g2v : tensor<256xf32>
    %v8017 = stablehlo.multiply %v1773, %v1773 : tensor<256xf32>
    %v8018 = stablehlo.multiply %v8015, %v8017 : tensor<256xf32>
    %v8019 = stablehlo.add %v8016, %v8018 : tensor<256xf32>
    %v8020 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8021 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8022 = stablehlo.divide %v8013, %v8020 : tensor<256xf32>
    %v8023 = stablehlo.divide %v8019, %v8021 : tensor<256xf32>
    %v8024 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8025 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8026 = stablehlo.sqrt %v8023 : tensor<256xf32>
    %v8027 = stablehlo.add %v8026, %v8025 : tensor<256xf32>
    %v8028 = stablehlo.divide %v8022, %v8027 : tensor<256xf32>
    %v8029 = stablehlo.multiply %v8024, %v8028 : tensor<256xf32>
    %v8030 = stablehlo.subtract %s3b3g2, %v8029 : tensor<256xf32>
    %v8031 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8032 = stablehlo.multiply %v8031, %v8024 : tensor<256xf32>
    %v8033 = stablehlo.multiply %v8032, %s3b3g2 : tensor<256xf32>
    %v8034 = stablehlo.subtract %v8030, %v8033 : tensor<256xf32>
    %v8035 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8036 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8037 = stablehlo.multiply %v8035, %s3b3bt2m : tensor<256xf32>
    %v8038 = stablehlo.multiply %v8036, %v1776 : tensor<256xf32>
    %v8039 = stablehlo.add %v8037, %v8038 : tensor<256xf32>
    %v8040 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8041 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8042 = stablehlo.multiply %v8040, %s3b3bt2v : tensor<256xf32>
    %v8043 = stablehlo.multiply %v1776, %v1776 : tensor<256xf32>
    %v8044 = stablehlo.multiply %v8041, %v8043 : tensor<256xf32>
    %v8045 = stablehlo.add %v8042, %v8044 : tensor<256xf32>
    %v8046 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8047 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8048 = stablehlo.multiply %v8046, %s3b3bt2m : tensor<256xf32>
    %v8049 = stablehlo.multiply %v8047, %v1776 : tensor<256xf32>
    %v8050 = stablehlo.add %v8048, %v8049 : tensor<256xf32>
    %v8051 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8052 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8053 = stablehlo.multiply %v8051, %s3b3bt2v : tensor<256xf32>
    %v8054 = stablehlo.multiply %v1776, %v1776 : tensor<256xf32>
    %v8055 = stablehlo.multiply %v8052, %v8054 : tensor<256xf32>
    %v8056 = stablehlo.add %v8053, %v8055 : tensor<256xf32>
    %v8057 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8058 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8059 = stablehlo.divide %v8050, %v8057 : tensor<256xf32>
    %v8060 = stablehlo.divide %v8056, %v8058 : tensor<256xf32>
    %v8061 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8062 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8063 = stablehlo.sqrt %v8060 : tensor<256xf32>
    %v8064 = stablehlo.add %v8063, %v8062 : tensor<256xf32>
    %v8065 = stablehlo.divide %v8059, %v8064 : tensor<256xf32>
    %v8066 = stablehlo.multiply %v8061, %v8065 : tensor<256xf32>
    %v8067 = stablehlo.subtract %s3b3bt2, %v8066 : tensor<256xf32>
    %v8068 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8069 = stablehlo.multiply %v8068, %v8061 : tensor<256xf32>
    %v8070 = stablehlo.multiply %v8069, %s3b3bt2 : tensor<256xf32>
    %v8071 = stablehlo.subtract %v8067, %v8070 : tensor<256xf32>
    %v8072 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8073 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8074 = stablehlo.multiply %v8072, %s3b4W1m : tensor<256x256x3x3xf32>
    %v8075 = stablehlo.multiply %v8073, %v1585 : tensor<256x256x3x3xf32>
    %v8076 = stablehlo.add %v8074, %v8075 : tensor<256x256x3x3xf32>
    %v8077 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8078 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8079 = stablehlo.multiply %v8077, %s3b4W1v : tensor<256x256x3x3xf32>
    %v8080 = stablehlo.multiply %v1585, %v1585 : tensor<256x256x3x3xf32>
    %v8081 = stablehlo.multiply %v8078, %v8080 : tensor<256x256x3x3xf32>
    %v8082 = stablehlo.add %v8079, %v8081 : tensor<256x256x3x3xf32>
    %v8083 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8084 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8085 = stablehlo.multiply %v8083, %s3b4W1m : tensor<256x256x3x3xf32>
    %v8086 = stablehlo.multiply %v8084, %v1585 : tensor<256x256x3x3xf32>
    %v8087 = stablehlo.add %v8085, %v8086 : tensor<256x256x3x3xf32>
    %v8088 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8089 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8090 = stablehlo.multiply %v8088, %s3b4W1v : tensor<256x256x3x3xf32>
    %v8091 = stablehlo.multiply %v1585, %v1585 : tensor<256x256x3x3xf32>
    %v8092 = stablehlo.multiply %v8089, %v8091 : tensor<256x256x3x3xf32>
    %v8093 = stablehlo.add %v8090, %v8092 : tensor<256x256x3x3xf32>
    %v8094 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8095 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8096 = stablehlo.divide %v8087, %v8094 : tensor<256x256x3x3xf32>
    %v8097 = stablehlo.divide %v8093, %v8095 : tensor<256x256x3x3xf32>
    %v8098 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8099 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8100 = stablehlo.sqrt %v8097 : tensor<256x256x3x3xf32>
    %v8101 = stablehlo.add %v8100, %v8099 : tensor<256x256x3x3xf32>
    %v8102 = stablehlo.divide %v8096, %v8101 : tensor<256x256x3x3xf32>
    %v8103 = stablehlo.multiply %v8098, %v8102 : tensor<256x256x3x3xf32>
    %v8104 = stablehlo.subtract %s3b4W1, %v8103 : tensor<256x256x3x3xf32>
    %v8105 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8106 = stablehlo.multiply %v8105, %v8098 : tensor<256x256x3x3xf32>
    %v8107 = stablehlo.multiply %v8106, %s3b4W1 : tensor<256x256x3x3xf32>
    %v8108 = stablehlo.subtract %v8104, %v8107 : tensor<256x256x3x3xf32>
    %v8109 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8110 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8111 = stablehlo.multiply %v8109, %s3b4b1m : tensor<256xf32>
    %v8112 = stablehlo.multiply %v8110, %v1588 : tensor<256xf32>
    %v8113 = stablehlo.add %v8111, %v8112 : tensor<256xf32>
    %v8114 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8115 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8116 = stablehlo.multiply %v8114, %s3b4b1v : tensor<256xf32>
    %v8117 = stablehlo.multiply %v1588, %v1588 : tensor<256xf32>
    %v8118 = stablehlo.multiply %v8115, %v8117 : tensor<256xf32>
    %v8119 = stablehlo.add %v8116, %v8118 : tensor<256xf32>
    %v8120 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8121 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8122 = stablehlo.multiply %v8120, %s3b4b1m : tensor<256xf32>
    %v8123 = stablehlo.multiply %v8121, %v1588 : tensor<256xf32>
    %v8124 = stablehlo.add %v8122, %v8123 : tensor<256xf32>
    %v8125 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8126 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8127 = stablehlo.multiply %v8125, %s3b4b1v : tensor<256xf32>
    %v8128 = stablehlo.multiply %v1588, %v1588 : tensor<256xf32>
    %v8129 = stablehlo.multiply %v8126, %v8128 : tensor<256xf32>
    %v8130 = stablehlo.add %v8127, %v8129 : tensor<256xf32>
    %v8131 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8132 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8133 = stablehlo.divide %v8124, %v8131 : tensor<256xf32>
    %v8134 = stablehlo.divide %v8130, %v8132 : tensor<256xf32>
    %v8135 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8136 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8137 = stablehlo.sqrt %v8134 : tensor<256xf32>
    %v8138 = stablehlo.add %v8137, %v8136 : tensor<256xf32>
    %v8139 = stablehlo.divide %v8133, %v8138 : tensor<256xf32>
    %v8140 = stablehlo.multiply %v8135, %v8139 : tensor<256xf32>
    %v8141 = stablehlo.subtract %s3b4b1, %v8140 : tensor<256xf32>
    %v8142 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8143 = stablehlo.multiply %v8142, %v8135 : tensor<256xf32>
    %v8144 = stablehlo.multiply %v8143, %s3b4b1 : tensor<256xf32>
    %v8145 = stablehlo.subtract %v8141, %v8144 : tensor<256xf32>
    %v8146 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8147 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8148 = stablehlo.multiply %v8146, %s3b4g1m : tensor<256xf32>
    %v8149 = stablehlo.multiply %v8147, %v1606 : tensor<256xf32>
    %v8150 = stablehlo.add %v8148, %v8149 : tensor<256xf32>
    %v8151 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8152 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8153 = stablehlo.multiply %v8151, %s3b4g1v : tensor<256xf32>
    %v8154 = stablehlo.multiply %v1606, %v1606 : tensor<256xf32>
    %v8155 = stablehlo.multiply %v8152, %v8154 : tensor<256xf32>
    %v8156 = stablehlo.add %v8153, %v8155 : tensor<256xf32>
    %v8157 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8158 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8159 = stablehlo.multiply %v8157, %s3b4g1m : tensor<256xf32>
    %v8160 = stablehlo.multiply %v8158, %v1606 : tensor<256xf32>
    %v8161 = stablehlo.add %v8159, %v8160 : tensor<256xf32>
    %v8162 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8163 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8164 = stablehlo.multiply %v8162, %s3b4g1v : tensor<256xf32>
    %v8165 = stablehlo.multiply %v1606, %v1606 : tensor<256xf32>
    %v8166 = stablehlo.multiply %v8163, %v8165 : tensor<256xf32>
    %v8167 = stablehlo.add %v8164, %v8166 : tensor<256xf32>
    %v8168 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8169 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8170 = stablehlo.divide %v8161, %v8168 : tensor<256xf32>
    %v8171 = stablehlo.divide %v8167, %v8169 : tensor<256xf32>
    %v8172 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8173 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8174 = stablehlo.sqrt %v8171 : tensor<256xf32>
    %v8175 = stablehlo.add %v8174, %v8173 : tensor<256xf32>
    %v8176 = stablehlo.divide %v8170, %v8175 : tensor<256xf32>
    %v8177 = stablehlo.multiply %v8172, %v8176 : tensor<256xf32>
    %v8178 = stablehlo.subtract %s3b4g1, %v8177 : tensor<256xf32>
    %v8179 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8180 = stablehlo.multiply %v8179, %v8172 : tensor<256xf32>
    %v8181 = stablehlo.multiply %v8180, %s3b4g1 : tensor<256xf32>
    %v8182 = stablehlo.subtract %v8178, %v8181 : tensor<256xf32>
    %v8183 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8184 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8185 = stablehlo.multiply %v8183, %s3b4bt1m : tensor<256xf32>
    %v8186 = stablehlo.multiply %v8184, %v1609 : tensor<256xf32>
    %v8187 = stablehlo.add %v8185, %v8186 : tensor<256xf32>
    %v8188 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8189 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8190 = stablehlo.multiply %v8188, %s3b4bt1v : tensor<256xf32>
    %v8191 = stablehlo.multiply %v1609, %v1609 : tensor<256xf32>
    %v8192 = stablehlo.multiply %v8189, %v8191 : tensor<256xf32>
    %v8193 = stablehlo.add %v8190, %v8192 : tensor<256xf32>
    %v8194 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8195 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8196 = stablehlo.multiply %v8194, %s3b4bt1m : tensor<256xf32>
    %v8197 = stablehlo.multiply %v8195, %v1609 : tensor<256xf32>
    %v8198 = stablehlo.add %v8196, %v8197 : tensor<256xf32>
    %v8199 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8200 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8201 = stablehlo.multiply %v8199, %s3b4bt1v : tensor<256xf32>
    %v8202 = stablehlo.multiply %v1609, %v1609 : tensor<256xf32>
    %v8203 = stablehlo.multiply %v8200, %v8202 : tensor<256xf32>
    %v8204 = stablehlo.add %v8201, %v8203 : tensor<256xf32>
    %v8205 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8206 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8207 = stablehlo.divide %v8198, %v8205 : tensor<256xf32>
    %v8208 = stablehlo.divide %v8204, %v8206 : tensor<256xf32>
    %v8209 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8210 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8211 = stablehlo.sqrt %v8208 : tensor<256xf32>
    %v8212 = stablehlo.add %v8211, %v8210 : tensor<256xf32>
    %v8213 = stablehlo.divide %v8207, %v8212 : tensor<256xf32>
    %v8214 = stablehlo.multiply %v8209, %v8213 : tensor<256xf32>
    %v8215 = stablehlo.subtract %s3b4bt1, %v8214 : tensor<256xf32>
    %v8216 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8217 = stablehlo.multiply %v8216, %v8209 : tensor<256xf32>
    %v8218 = stablehlo.multiply %v8217, %s3b4bt1 : tensor<256xf32>
    %v8219 = stablehlo.subtract %v8215, %v8218 : tensor<256xf32>
    %v8220 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8221 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8222 = stablehlo.multiply %v8220, %s3b4W2m : tensor<256x256x3x3xf32>
    %v8223 = stablehlo.multiply %v8221, %v1615 : tensor<256x256x3x3xf32>
    %v8224 = stablehlo.add %v8222, %v8223 : tensor<256x256x3x3xf32>
    %v8225 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8226 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8227 = stablehlo.multiply %v8225, %s3b4W2v : tensor<256x256x3x3xf32>
    %v8228 = stablehlo.multiply %v1615, %v1615 : tensor<256x256x3x3xf32>
    %v8229 = stablehlo.multiply %v8226, %v8228 : tensor<256x256x3x3xf32>
    %v8230 = stablehlo.add %v8227, %v8229 : tensor<256x256x3x3xf32>
    %v8231 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8232 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8233 = stablehlo.multiply %v8231, %s3b4W2m : tensor<256x256x3x3xf32>
    %v8234 = stablehlo.multiply %v8232, %v1615 : tensor<256x256x3x3xf32>
    %v8235 = stablehlo.add %v8233, %v8234 : tensor<256x256x3x3xf32>
    %v8236 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8237 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8238 = stablehlo.multiply %v8236, %s3b4W2v : tensor<256x256x3x3xf32>
    %v8239 = stablehlo.multiply %v1615, %v1615 : tensor<256x256x3x3xf32>
    %v8240 = stablehlo.multiply %v8237, %v8239 : tensor<256x256x3x3xf32>
    %v8241 = stablehlo.add %v8238, %v8240 : tensor<256x256x3x3xf32>
    %v8242 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8243 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8244 = stablehlo.divide %v8235, %v8242 : tensor<256x256x3x3xf32>
    %v8245 = stablehlo.divide %v8241, %v8243 : tensor<256x256x3x3xf32>
    %v8246 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8247 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8248 = stablehlo.sqrt %v8245 : tensor<256x256x3x3xf32>
    %v8249 = stablehlo.add %v8248, %v8247 : tensor<256x256x3x3xf32>
    %v8250 = stablehlo.divide %v8244, %v8249 : tensor<256x256x3x3xf32>
    %v8251 = stablehlo.multiply %v8246, %v8250 : tensor<256x256x3x3xf32>
    %v8252 = stablehlo.subtract %s3b4W2, %v8251 : tensor<256x256x3x3xf32>
    %v8253 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8254 = stablehlo.multiply %v8253, %v8246 : tensor<256x256x3x3xf32>
    %v8255 = stablehlo.multiply %v8254, %s3b4W2 : tensor<256x256x3x3xf32>
    %v8256 = stablehlo.subtract %v8252, %v8255 : tensor<256x256x3x3xf32>
    %v8257 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8258 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8259 = stablehlo.multiply %v8257, %s3b4b2m : tensor<256xf32>
    %v8260 = stablehlo.multiply %v8258, %v1618 : tensor<256xf32>
    %v8261 = stablehlo.add %v8259, %v8260 : tensor<256xf32>
    %v8262 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8263 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8264 = stablehlo.multiply %v8262, %s3b4b2v : tensor<256xf32>
    %v8265 = stablehlo.multiply %v1618, %v1618 : tensor<256xf32>
    %v8266 = stablehlo.multiply %v8263, %v8265 : tensor<256xf32>
    %v8267 = stablehlo.add %v8264, %v8266 : tensor<256xf32>
    %v8268 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8269 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8270 = stablehlo.multiply %v8268, %s3b4b2m : tensor<256xf32>
    %v8271 = stablehlo.multiply %v8269, %v1618 : tensor<256xf32>
    %v8272 = stablehlo.add %v8270, %v8271 : tensor<256xf32>
    %v8273 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8274 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8275 = stablehlo.multiply %v8273, %s3b4b2v : tensor<256xf32>
    %v8276 = stablehlo.multiply %v1618, %v1618 : tensor<256xf32>
    %v8277 = stablehlo.multiply %v8274, %v8276 : tensor<256xf32>
    %v8278 = stablehlo.add %v8275, %v8277 : tensor<256xf32>
    %v8279 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8280 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8281 = stablehlo.divide %v8272, %v8279 : tensor<256xf32>
    %v8282 = stablehlo.divide %v8278, %v8280 : tensor<256xf32>
    %v8283 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8284 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8285 = stablehlo.sqrt %v8282 : tensor<256xf32>
    %v8286 = stablehlo.add %v8285, %v8284 : tensor<256xf32>
    %v8287 = stablehlo.divide %v8281, %v8286 : tensor<256xf32>
    %v8288 = stablehlo.multiply %v8283, %v8287 : tensor<256xf32>
    %v8289 = stablehlo.subtract %s3b4b2, %v8288 : tensor<256xf32>
    %v8290 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8291 = stablehlo.multiply %v8290, %v8283 : tensor<256xf32>
    %v8292 = stablehlo.multiply %v8291, %s3b4b2 : tensor<256xf32>
    %v8293 = stablehlo.subtract %v8289, %v8292 : tensor<256xf32>
    %v8294 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8295 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8296 = stablehlo.multiply %v8294, %s3b4g2m : tensor<256xf32>
    %v8297 = stablehlo.multiply %v8295, %v1636 : tensor<256xf32>
    %v8298 = stablehlo.add %v8296, %v8297 : tensor<256xf32>
    %v8299 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8300 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8301 = stablehlo.multiply %v8299, %s3b4g2v : tensor<256xf32>
    %v8302 = stablehlo.multiply %v1636, %v1636 : tensor<256xf32>
    %v8303 = stablehlo.multiply %v8300, %v8302 : tensor<256xf32>
    %v8304 = stablehlo.add %v8301, %v8303 : tensor<256xf32>
    %v8305 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8306 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8307 = stablehlo.multiply %v8305, %s3b4g2m : tensor<256xf32>
    %v8308 = stablehlo.multiply %v8306, %v1636 : tensor<256xf32>
    %v8309 = stablehlo.add %v8307, %v8308 : tensor<256xf32>
    %v8310 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8311 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8312 = stablehlo.multiply %v8310, %s3b4g2v : tensor<256xf32>
    %v8313 = stablehlo.multiply %v1636, %v1636 : tensor<256xf32>
    %v8314 = stablehlo.multiply %v8311, %v8313 : tensor<256xf32>
    %v8315 = stablehlo.add %v8312, %v8314 : tensor<256xf32>
    %v8316 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8317 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8318 = stablehlo.divide %v8309, %v8316 : tensor<256xf32>
    %v8319 = stablehlo.divide %v8315, %v8317 : tensor<256xf32>
    %v8320 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8321 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8322 = stablehlo.sqrt %v8319 : tensor<256xf32>
    %v8323 = stablehlo.add %v8322, %v8321 : tensor<256xf32>
    %v8324 = stablehlo.divide %v8318, %v8323 : tensor<256xf32>
    %v8325 = stablehlo.multiply %v8320, %v8324 : tensor<256xf32>
    %v8326 = stablehlo.subtract %s3b4g2, %v8325 : tensor<256xf32>
    %v8327 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8328 = stablehlo.multiply %v8327, %v8320 : tensor<256xf32>
    %v8329 = stablehlo.multiply %v8328, %s3b4g2 : tensor<256xf32>
    %v8330 = stablehlo.subtract %v8326, %v8329 : tensor<256xf32>
    %v8331 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8332 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8333 = stablehlo.multiply %v8331, %s3b4bt2m : tensor<256xf32>
    %v8334 = stablehlo.multiply %v8332, %v1639 : tensor<256xf32>
    %v8335 = stablehlo.add %v8333, %v8334 : tensor<256xf32>
    %v8336 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8337 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8338 = stablehlo.multiply %v8336, %s3b4bt2v : tensor<256xf32>
    %v8339 = stablehlo.multiply %v1639, %v1639 : tensor<256xf32>
    %v8340 = stablehlo.multiply %v8337, %v8339 : tensor<256xf32>
    %v8341 = stablehlo.add %v8338, %v8340 : tensor<256xf32>
    %v8342 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8343 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8344 = stablehlo.multiply %v8342, %s3b4bt2m : tensor<256xf32>
    %v8345 = stablehlo.multiply %v8343, %v1639 : tensor<256xf32>
    %v8346 = stablehlo.add %v8344, %v8345 : tensor<256xf32>
    %v8347 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8348 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8349 = stablehlo.multiply %v8347, %s3b4bt2v : tensor<256xf32>
    %v8350 = stablehlo.multiply %v1639, %v1639 : tensor<256xf32>
    %v8351 = stablehlo.multiply %v8348, %v8350 : tensor<256xf32>
    %v8352 = stablehlo.add %v8349, %v8351 : tensor<256xf32>
    %v8353 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8354 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8355 = stablehlo.divide %v8346, %v8353 : tensor<256xf32>
    %v8356 = stablehlo.divide %v8352, %v8354 : tensor<256xf32>
    %v8357 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8358 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8359 = stablehlo.sqrt %v8356 : tensor<256xf32>
    %v8360 = stablehlo.add %v8359, %v8358 : tensor<256xf32>
    %v8361 = stablehlo.divide %v8355, %v8360 : tensor<256xf32>
    %v8362 = stablehlo.multiply %v8357, %v8361 : tensor<256xf32>
    %v8363 = stablehlo.subtract %s3b4bt2, %v8362 : tensor<256xf32>
    %v8364 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8365 = stablehlo.multiply %v8364, %v8357 : tensor<256xf32>
    %v8366 = stablehlo.multiply %v8365, %s3b4bt2 : tensor<256xf32>
    %v8367 = stablehlo.subtract %v8363, %v8366 : tensor<256xf32>
    %v8368 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8369 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8370 = stablehlo.multiply %v8368, %d4W1m : tensor<512x256x3x3xf32>
    %v8371 = stablehlo.multiply %v8369, %v1416 : tensor<512x256x3x3xf32>
    %v8372 = stablehlo.add %v8370, %v8371 : tensor<512x256x3x3xf32>
    %v8373 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8374 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8375 = stablehlo.multiply %v8373, %d4W1v : tensor<512x256x3x3xf32>
    %v8376 = stablehlo.multiply %v1416, %v1416 : tensor<512x256x3x3xf32>
    %v8377 = stablehlo.multiply %v8374, %v8376 : tensor<512x256x3x3xf32>
    %v8378 = stablehlo.add %v8375, %v8377 : tensor<512x256x3x3xf32>
    %v8379 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8380 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8381 = stablehlo.multiply %v8379, %d4W1m : tensor<512x256x3x3xf32>
    %v8382 = stablehlo.multiply %v8380, %v1416 : tensor<512x256x3x3xf32>
    %v8383 = stablehlo.add %v8381, %v8382 : tensor<512x256x3x3xf32>
    %v8384 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8385 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8386 = stablehlo.multiply %v8384, %d4W1v : tensor<512x256x3x3xf32>
    %v8387 = stablehlo.multiply %v1416, %v1416 : tensor<512x256x3x3xf32>
    %v8388 = stablehlo.multiply %v8385, %v8387 : tensor<512x256x3x3xf32>
    %v8389 = stablehlo.add %v8386, %v8388 : tensor<512x256x3x3xf32>
    %v8390 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8391 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8392 = stablehlo.divide %v8383, %v8390 : tensor<512x256x3x3xf32>
    %v8393 = stablehlo.divide %v8389, %v8391 : tensor<512x256x3x3xf32>
    %v8394 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8395 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8396 = stablehlo.sqrt %v8393 : tensor<512x256x3x3xf32>
    %v8397 = stablehlo.add %v8396, %v8395 : tensor<512x256x3x3xf32>
    %v8398 = stablehlo.divide %v8392, %v8397 : tensor<512x256x3x3xf32>
    %v8399 = stablehlo.multiply %v8394, %v8398 : tensor<512x256x3x3xf32>
    %v8400 = stablehlo.subtract %d4W1, %v8399 : tensor<512x256x3x3xf32>
    %v8401 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8402 = stablehlo.multiply %v8401, %v8394 : tensor<512x256x3x3xf32>
    %v8403 = stablehlo.multiply %v8402, %d4W1 : tensor<512x256x3x3xf32>
    %v8404 = stablehlo.subtract %v8400, %v8403 : tensor<512x256x3x3xf32>
    %v8405 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8406 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8407 = stablehlo.multiply %v8405, %d4b1m : tensor<512xf32>
    %v8408 = stablehlo.multiply %v8406, %v1419 : tensor<512xf32>
    %v8409 = stablehlo.add %v8407, %v8408 : tensor<512xf32>
    %v8410 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8411 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8412 = stablehlo.multiply %v8410, %d4b1v : tensor<512xf32>
    %v8413 = stablehlo.multiply %v1419, %v1419 : tensor<512xf32>
    %v8414 = stablehlo.multiply %v8411, %v8413 : tensor<512xf32>
    %v8415 = stablehlo.add %v8412, %v8414 : tensor<512xf32>
    %v8416 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8417 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8418 = stablehlo.multiply %v8416, %d4b1m : tensor<512xf32>
    %v8419 = stablehlo.multiply %v8417, %v1419 : tensor<512xf32>
    %v8420 = stablehlo.add %v8418, %v8419 : tensor<512xf32>
    %v8421 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8422 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8423 = stablehlo.multiply %v8421, %d4b1v : tensor<512xf32>
    %v8424 = stablehlo.multiply %v1419, %v1419 : tensor<512xf32>
    %v8425 = stablehlo.multiply %v8422, %v8424 : tensor<512xf32>
    %v8426 = stablehlo.add %v8423, %v8425 : tensor<512xf32>
    %v8427 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8428 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8429 = stablehlo.divide %v8420, %v8427 : tensor<512xf32>
    %v8430 = stablehlo.divide %v8426, %v8428 : tensor<512xf32>
    %v8431 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8432 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8433 = stablehlo.sqrt %v8430 : tensor<512xf32>
    %v8434 = stablehlo.add %v8433, %v8432 : tensor<512xf32>
    %v8435 = stablehlo.divide %v8429, %v8434 : tensor<512xf32>
    %v8436 = stablehlo.multiply %v8431, %v8435 : tensor<512xf32>
    %v8437 = stablehlo.subtract %d4b1, %v8436 : tensor<512xf32>
    %v8438 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8439 = stablehlo.multiply %v8438, %v8431 : tensor<512xf32>
    %v8440 = stablehlo.multiply %v8439, %d4b1 : tensor<512xf32>
    %v8441 = stablehlo.subtract %v8437, %v8440 : tensor<512xf32>
    %v8442 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8443 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8444 = stablehlo.multiply %v8442, %d4g1m : tensor<512xf32>
    %v8445 = stablehlo.multiply %v8443, %v1437 : tensor<512xf32>
    %v8446 = stablehlo.add %v8444, %v8445 : tensor<512xf32>
    %v8447 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8448 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8449 = stablehlo.multiply %v8447, %d4g1v : tensor<512xf32>
    %v8450 = stablehlo.multiply %v1437, %v1437 : tensor<512xf32>
    %v8451 = stablehlo.multiply %v8448, %v8450 : tensor<512xf32>
    %v8452 = stablehlo.add %v8449, %v8451 : tensor<512xf32>
    %v8453 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8454 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8455 = stablehlo.multiply %v8453, %d4g1m : tensor<512xf32>
    %v8456 = stablehlo.multiply %v8454, %v1437 : tensor<512xf32>
    %v8457 = stablehlo.add %v8455, %v8456 : tensor<512xf32>
    %v8458 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8459 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8460 = stablehlo.multiply %v8458, %d4g1v : tensor<512xf32>
    %v8461 = stablehlo.multiply %v1437, %v1437 : tensor<512xf32>
    %v8462 = stablehlo.multiply %v8459, %v8461 : tensor<512xf32>
    %v8463 = stablehlo.add %v8460, %v8462 : tensor<512xf32>
    %v8464 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8465 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8466 = stablehlo.divide %v8457, %v8464 : tensor<512xf32>
    %v8467 = stablehlo.divide %v8463, %v8465 : tensor<512xf32>
    %v8468 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8469 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8470 = stablehlo.sqrt %v8467 : tensor<512xf32>
    %v8471 = stablehlo.add %v8470, %v8469 : tensor<512xf32>
    %v8472 = stablehlo.divide %v8466, %v8471 : tensor<512xf32>
    %v8473 = stablehlo.multiply %v8468, %v8472 : tensor<512xf32>
    %v8474 = stablehlo.subtract %d4g1, %v8473 : tensor<512xf32>
    %v8475 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8476 = stablehlo.multiply %v8475, %v8468 : tensor<512xf32>
    %v8477 = stablehlo.multiply %v8476, %d4g1 : tensor<512xf32>
    %v8478 = stablehlo.subtract %v8474, %v8477 : tensor<512xf32>
    %v8479 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8480 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8481 = stablehlo.multiply %v8479, %d4bt1m : tensor<512xf32>
    %v8482 = stablehlo.multiply %v8480, %v1440 : tensor<512xf32>
    %v8483 = stablehlo.add %v8481, %v8482 : tensor<512xf32>
    %v8484 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8485 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8486 = stablehlo.multiply %v8484, %d4bt1v : tensor<512xf32>
    %v8487 = stablehlo.multiply %v1440, %v1440 : tensor<512xf32>
    %v8488 = stablehlo.multiply %v8485, %v8487 : tensor<512xf32>
    %v8489 = stablehlo.add %v8486, %v8488 : tensor<512xf32>
    %v8490 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8491 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8492 = stablehlo.multiply %v8490, %d4bt1m : tensor<512xf32>
    %v8493 = stablehlo.multiply %v8491, %v1440 : tensor<512xf32>
    %v8494 = stablehlo.add %v8492, %v8493 : tensor<512xf32>
    %v8495 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8496 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8497 = stablehlo.multiply %v8495, %d4bt1v : tensor<512xf32>
    %v8498 = stablehlo.multiply %v1440, %v1440 : tensor<512xf32>
    %v8499 = stablehlo.multiply %v8496, %v8498 : tensor<512xf32>
    %v8500 = stablehlo.add %v8497, %v8499 : tensor<512xf32>
    %v8501 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8502 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8503 = stablehlo.divide %v8494, %v8501 : tensor<512xf32>
    %v8504 = stablehlo.divide %v8500, %v8502 : tensor<512xf32>
    %v8505 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8506 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8507 = stablehlo.sqrt %v8504 : tensor<512xf32>
    %v8508 = stablehlo.add %v8507, %v8506 : tensor<512xf32>
    %v8509 = stablehlo.divide %v8503, %v8508 : tensor<512xf32>
    %v8510 = stablehlo.multiply %v8505, %v8509 : tensor<512xf32>
    %v8511 = stablehlo.subtract %d4bt1, %v8510 : tensor<512xf32>
    %v8512 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8513 = stablehlo.multiply %v8512, %v8505 : tensor<512xf32>
    %v8514 = stablehlo.multiply %v8513, %d4bt1 : tensor<512xf32>
    %v8515 = stablehlo.subtract %v8511, %v8514 : tensor<512xf32>
    %v8516 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8517 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8518 = stablehlo.multiply %v8516, %d4W2m : tensor<512x512x3x3xf32>
    %v8519 = stablehlo.multiply %v8517, %v1446 : tensor<512x512x3x3xf32>
    %v8520 = stablehlo.add %v8518, %v8519 : tensor<512x512x3x3xf32>
    %v8521 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8522 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8523 = stablehlo.multiply %v8521, %d4W2v : tensor<512x512x3x3xf32>
    %v8524 = stablehlo.multiply %v1446, %v1446 : tensor<512x512x3x3xf32>
    %v8525 = stablehlo.multiply %v8522, %v8524 : tensor<512x512x3x3xf32>
    %v8526 = stablehlo.add %v8523, %v8525 : tensor<512x512x3x3xf32>
    %v8527 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8528 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8529 = stablehlo.multiply %v8527, %d4W2m : tensor<512x512x3x3xf32>
    %v8530 = stablehlo.multiply %v8528, %v1446 : tensor<512x512x3x3xf32>
    %v8531 = stablehlo.add %v8529, %v8530 : tensor<512x512x3x3xf32>
    %v8532 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8533 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8534 = stablehlo.multiply %v8532, %d4W2v : tensor<512x512x3x3xf32>
    %v8535 = stablehlo.multiply %v1446, %v1446 : tensor<512x512x3x3xf32>
    %v8536 = stablehlo.multiply %v8533, %v8535 : tensor<512x512x3x3xf32>
    %v8537 = stablehlo.add %v8534, %v8536 : tensor<512x512x3x3xf32>
    %v8538 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8539 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8540 = stablehlo.divide %v8531, %v8538 : tensor<512x512x3x3xf32>
    %v8541 = stablehlo.divide %v8537, %v8539 : tensor<512x512x3x3xf32>
    %v8542 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8543 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8544 = stablehlo.sqrt %v8541 : tensor<512x512x3x3xf32>
    %v8545 = stablehlo.add %v8544, %v8543 : tensor<512x512x3x3xf32>
    %v8546 = stablehlo.divide %v8540, %v8545 : tensor<512x512x3x3xf32>
    %v8547 = stablehlo.multiply %v8542, %v8546 : tensor<512x512x3x3xf32>
    %v8548 = stablehlo.subtract %d4W2, %v8547 : tensor<512x512x3x3xf32>
    %v8549 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8550 = stablehlo.multiply %v8549, %v8542 : tensor<512x512x3x3xf32>
    %v8551 = stablehlo.multiply %v8550, %d4W2 : tensor<512x512x3x3xf32>
    %v8552 = stablehlo.subtract %v8548, %v8551 : tensor<512x512x3x3xf32>
    %v8553 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8554 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8555 = stablehlo.multiply %v8553, %d4b2m : tensor<512xf32>
    %v8556 = stablehlo.multiply %v8554, %v1449 : tensor<512xf32>
    %v8557 = stablehlo.add %v8555, %v8556 : tensor<512xf32>
    %v8558 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8559 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8560 = stablehlo.multiply %v8558, %d4b2v : tensor<512xf32>
    %v8561 = stablehlo.multiply %v1449, %v1449 : tensor<512xf32>
    %v8562 = stablehlo.multiply %v8559, %v8561 : tensor<512xf32>
    %v8563 = stablehlo.add %v8560, %v8562 : tensor<512xf32>
    %v8564 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8565 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8566 = stablehlo.multiply %v8564, %d4b2m : tensor<512xf32>
    %v8567 = stablehlo.multiply %v8565, %v1449 : tensor<512xf32>
    %v8568 = stablehlo.add %v8566, %v8567 : tensor<512xf32>
    %v8569 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8570 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8571 = stablehlo.multiply %v8569, %d4b2v : tensor<512xf32>
    %v8572 = stablehlo.multiply %v1449, %v1449 : tensor<512xf32>
    %v8573 = stablehlo.multiply %v8570, %v8572 : tensor<512xf32>
    %v8574 = stablehlo.add %v8571, %v8573 : tensor<512xf32>
    %v8575 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8576 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8577 = stablehlo.divide %v8568, %v8575 : tensor<512xf32>
    %v8578 = stablehlo.divide %v8574, %v8576 : tensor<512xf32>
    %v8579 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8580 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8581 = stablehlo.sqrt %v8578 : tensor<512xf32>
    %v8582 = stablehlo.add %v8581, %v8580 : tensor<512xf32>
    %v8583 = stablehlo.divide %v8577, %v8582 : tensor<512xf32>
    %v8584 = stablehlo.multiply %v8579, %v8583 : tensor<512xf32>
    %v8585 = stablehlo.subtract %d4b2, %v8584 : tensor<512xf32>
    %v8586 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8587 = stablehlo.multiply %v8586, %v8579 : tensor<512xf32>
    %v8588 = stablehlo.multiply %v8587, %d4b2 : tensor<512xf32>
    %v8589 = stablehlo.subtract %v8585, %v8588 : tensor<512xf32>
    %v8590 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8591 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8592 = stablehlo.multiply %v8590, %d4g2m : tensor<512xf32>
    %v8593 = stablehlo.multiply %v8591, %v1467 : tensor<512xf32>
    %v8594 = stablehlo.add %v8592, %v8593 : tensor<512xf32>
    %v8595 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8596 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8597 = stablehlo.multiply %v8595, %d4g2v : tensor<512xf32>
    %v8598 = stablehlo.multiply %v1467, %v1467 : tensor<512xf32>
    %v8599 = stablehlo.multiply %v8596, %v8598 : tensor<512xf32>
    %v8600 = stablehlo.add %v8597, %v8599 : tensor<512xf32>
    %v8601 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8602 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8603 = stablehlo.multiply %v8601, %d4g2m : tensor<512xf32>
    %v8604 = stablehlo.multiply %v8602, %v1467 : tensor<512xf32>
    %v8605 = stablehlo.add %v8603, %v8604 : tensor<512xf32>
    %v8606 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8607 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8608 = stablehlo.multiply %v8606, %d4g2v : tensor<512xf32>
    %v8609 = stablehlo.multiply %v1467, %v1467 : tensor<512xf32>
    %v8610 = stablehlo.multiply %v8607, %v8609 : tensor<512xf32>
    %v8611 = stablehlo.add %v8608, %v8610 : tensor<512xf32>
    %v8612 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8613 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8614 = stablehlo.divide %v8605, %v8612 : tensor<512xf32>
    %v8615 = stablehlo.divide %v8611, %v8613 : tensor<512xf32>
    %v8616 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8617 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8618 = stablehlo.sqrt %v8615 : tensor<512xf32>
    %v8619 = stablehlo.add %v8618, %v8617 : tensor<512xf32>
    %v8620 = stablehlo.divide %v8614, %v8619 : tensor<512xf32>
    %v8621 = stablehlo.multiply %v8616, %v8620 : tensor<512xf32>
    %v8622 = stablehlo.subtract %d4g2, %v8621 : tensor<512xf32>
    %v8623 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8624 = stablehlo.multiply %v8623, %v8616 : tensor<512xf32>
    %v8625 = stablehlo.multiply %v8624, %d4g2 : tensor<512xf32>
    %v8626 = stablehlo.subtract %v8622, %v8625 : tensor<512xf32>
    %v8627 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8628 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8629 = stablehlo.multiply %v8627, %d4bt2m : tensor<512xf32>
    %v8630 = stablehlo.multiply %v8628, %v1470 : tensor<512xf32>
    %v8631 = stablehlo.add %v8629, %v8630 : tensor<512xf32>
    %v8632 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8633 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8634 = stablehlo.multiply %v8632, %d4bt2v : tensor<512xf32>
    %v8635 = stablehlo.multiply %v1470, %v1470 : tensor<512xf32>
    %v8636 = stablehlo.multiply %v8633, %v8635 : tensor<512xf32>
    %v8637 = stablehlo.add %v8634, %v8636 : tensor<512xf32>
    %v8638 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8639 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8640 = stablehlo.multiply %v8638, %d4bt2m : tensor<512xf32>
    %v8641 = stablehlo.multiply %v8639, %v1470 : tensor<512xf32>
    %v8642 = stablehlo.add %v8640, %v8641 : tensor<512xf32>
    %v8643 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8644 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8645 = stablehlo.multiply %v8643, %d4bt2v : tensor<512xf32>
    %v8646 = stablehlo.multiply %v1470, %v1470 : tensor<512xf32>
    %v8647 = stablehlo.multiply %v8644, %v8646 : tensor<512xf32>
    %v8648 = stablehlo.add %v8645, %v8647 : tensor<512xf32>
    %v8649 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8650 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8651 = stablehlo.divide %v8642, %v8649 : tensor<512xf32>
    %v8652 = stablehlo.divide %v8648, %v8650 : tensor<512xf32>
    %v8653 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8654 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8655 = stablehlo.sqrt %v8652 : tensor<512xf32>
    %v8656 = stablehlo.add %v8655, %v8654 : tensor<512xf32>
    %v8657 = stablehlo.divide %v8651, %v8656 : tensor<512xf32>
    %v8658 = stablehlo.multiply %v8653, %v8657 : tensor<512xf32>
    %v8659 = stablehlo.subtract %d4bt2, %v8658 : tensor<512xf32>
    %v8660 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8661 = stablehlo.multiply %v8660, %v8653 : tensor<512xf32>
    %v8662 = stablehlo.multiply %v8661, %d4bt2 : tensor<512xf32>
    %v8663 = stablehlo.subtract %v8659, %v8662 : tensor<512xf32>
    %v8664 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8665 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8666 = stablehlo.multiply %v8664, %d4Wpm : tensor<512x256x3x3xf32>
    %v8667 = stablehlo.multiply %v8665, %v1478 : tensor<512x256x3x3xf32>
    %v8668 = stablehlo.add %v8666, %v8667 : tensor<512x256x3x3xf32>
    %v8669 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8670 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8671 = stablehlo.multiply %v8669, %d4Wpv : tensor<512x256x3x3xf32>
    %v8672 = stablehlo.multiply %v1478, %v1478 : tensor<512x256x3x3xf32>
    %v8673 = stablehlo.multiply %v8670, %v8672 : tensor<512x256x3x3xf32>
    %v8674 = stablehlo.add %v8671, %v8673 : tensor<512x256x3x3xf32>
    %v8675 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8676 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8677 = stablehlo.multiply %v8675, %d4Wpm : tensor<512x256x3x3xf32>
    %v8678 = stablehlo.multiply %v8676, %v1478 : tensor<512x256x3x3xf32>
    %v8679 = stablehlo.add %v8677, %v8678 : tensor<512x256x3x3xf32>
    %v8680 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8681 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8682 = stablehlo.multiply %v8680, %d4Wpv : tensor<512x256x3x3xf32>
    %v8683 = stablehlo.multiply %v1478, %v1478 : tensor<512x256x3x3xf32>
    %v8684 = stablehlo.multiply %v8681, %v8683 : tensor<512x256x3x3xf32>
    %v8685 = stablehlo.add %v8682, %v8684 : tensor<512x256x3x3xf32>
    %v8686 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8687 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8688 = stablehlo.divide %v8679, %v8686 : tensor<512x256x3x3xf32>
    %v8689 = stablehlo.divide %v8685, %v8687 : tensor<512x256x3x3xf32>
    %v8690 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8691 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8692 = stablehlo.sqrt %v8689 : tensor<512x256x3x3xf32>
    %v8693 = stablehlo.add %v8692, %v8691 : tensor<512x256x3x3xf32>
    %v8694 = stablehlo.divide %v8688, %v8693 : tensor<512x256x3x3xf32>
    %v8695 = stablehlo.multiply %v8690, %v8694 : tensor<512x256x3x3xf32>
    %v8696 = stablehlo.subtract %d4Wp, %v8695 : tensor<512x256x3x3xf32>
    %v8697 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v8698 = stablehlo.multiply %v8697, %v8690 : tensor<512x256x3x3xf32>
    %v8699 = stablehlo.multiply %v8698, %d4Wp : tensor<512x256x3x3xf32>
    %v8700 = stablehlo.subtract %v8696, %v8699 : tensor<512x256x3x3xf32>
    %v8701 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8702 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8703 = stablehlo.multiply %v8701, %d4bpm : tensor<512xf32>
    %v8704 = stablehlo.multiply %v8702, %v1481 : tensor<512xf32>
    %v8705 = stablehlo.add %v8703, %v8704 : tensor<512xf32>
    %v8706 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8707 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8708 = stablehlo.multiply %v8706, %d4bpv : tensor<512xf32>
    %v8709 = stablehlo.multiply %v1481, %v1481 : tensor<512xf32>
    %v8710 = stablehlo.multiply %v8707, %v8709 : tensor<512xf32>
    %v8711 = stablehlo.add %v8708, %v8710 : tensor<512xf32>
    %v8712 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8713 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8714 = stablehlo.multiply %v8712, %d4bpm : tensor<512xf32>
    %v8715 = stablehlo.multiply %v8713, %v1481 : tensor<512xf32>
    %v8716 = stablehlo.add %v8714, %v8715 : tensor<512xf32>
    %v8717 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8718 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8719 = stablehlo.multiply %v8717, %d4bpv : tensor<512xf32>
    %v8720 = stablehlo.multiply %v1481, %v1481 : tensor<512xf32>
    %v8721 = stablehlo.multiply %v8718, %v8720 : tensor<512xf32>
    %v8722 = stablehlo.add %v8719, %v8721 : tensor<512xf32>
    %v8723 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8724 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8725 = stablehlo.divide %v8716, %v8723 : tensor<512xf32>
    %v8726 = stablehlo.divide %v8722, %v8724 : tensor<512xf32>
    %v8727 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8728 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8729 = stablehlo.sqrt %v8726 : tensor<512xf32>
    %v8730 = stablehlo.add %v8729, %v8728 : tensor<512xf32>
    %v8731 = stablehlo.divide %v8725, %v8730 : tensor<512xf32>
    %v8732 = stablehlo.multiply %v8727, %v8731 : tensor<512xf32>
    %v8733 = stablehlo.subtract %d4bp, %v8732 : tensor<512xf32>
    %v8734 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8735 = stablehlo.multiply %v8734, %v8727 : tensor<512xf32>
    %v8736 = stablehlo.multiply %v8735, %d4bp : tensor<512xf32>
    %v8737 = stablehlo.subtract %v8733, %v8736 : tensor<512xf32>
    %v8738 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8739 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8740 = stablehlo.multiply %v8738, %d4gpm : tensor<512xf32>
    %v8741 = stablehlo.multiply %v8739, %v1499 : tensor<512xf32>
    %v8742 = stablehlo.add %v8740, %v8741 : tensor<512xf32>
    %v8743 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8744 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8745 = stablehlo.multiply %v8743, %d4gpv : tensor<512xf32>
    %v8746 = stablehlo.multiply %v1499, %v1499 : tensor<512xf32>
    %v8747 = stablehlo.multiply %v8744, %v8746 : tensor<512xf32>
    %v8748 = stablehlo.add %v8745, %v8747 : tensor<512xf32>
    %v8749 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8750 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8751 = stablehlo.multiply %v8749, %d4gpm : tensor<512xf32>
    %v8752 = stablehlo.multiply %v8750, %v1499 : tensor<512xf32>
    %v8753 = stablehlo.add %v8751, %v8752 : tensor<512xf32>
    %v8754 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8755 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8756 = stablehlo.multiply %v8754, %d4gpv : tensor<512xf32>
    %v8757 = stablehlo.multiply %v1499, %v1499 : tensor<512xf32>
    %v8758 = stablehlo.multiply %v8755, %v8757 : tensor<512xf32>
    %v8759 = stablehlo.add %v8756, %v8758 : tensor<512xf32>
    %v8760 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8761 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8762 = stablehlo.divide %v8753, %v8760 : tensor<512xf32>
    %v8763 = stablehlo.divide %v8759, %v8761 : tensor<512xf32>
    %v8764 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8765 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8766 = stablehlo.sqrt %v8763 : tensor<512xf32>
    %v8767 = stablehlo.add %v8766, %v8765 : tensor<512xf32>
    %v8768 = stablehlo.divide %v8762, %v8767 : tensor<512xf32>
    %v8769 = stablehlo.multiply %v8764, %v8768 : tensor<512xf32>
    %v8770 = stablehlo.subtract %d4gp, %v8769 : tensor<512xf32>
    %v8771 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8772 = stablehlo.multiply %v8771, %v8764 : tensor<512xf32>
    %v8773 = stablehlo.multiply %v8772, %d4gp : tensor<512xf32>
    %v8774 = stablehlo.subtract %v8770, %v8773 : tensor<512xf32>
    %v8775 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8776 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8777 = stablehlo.multiply %v8775, %d4btpm : tensor<512xf32>
    %v8778 = stablehlo.multiply %v8776, %v1502 : tensor<512xf32>
    %v8779 = stablehlo.add %v8777, %v8778 : tensor<512xf32>
    %v8780 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8781 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8782 = stablehlo.multiply %v8780, %d4btpv : tensor<512xf32>
    %v8783 = stablehlo.multiply %v1502, %v1502 : tensor<512xf32>
    %v8784 = stablehlo.multiply %v8781, %v8783 : tensor<512xf32>
    %v8785 = stablehlo.add %v8782, %v8784 : tensor<512xf32>
    %v8786 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8787 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8788 = stablehlo.multiply %v8786, %d4btpm : tensor<512xf32>
    %v8789 = stablehlo.multiply %v8787, %v1502 : tensor<512xf32>
    %v8790 = stablehlo.add %v8788, %v8789 : tensor<512xf32>
    %v8791 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8792 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8793 = stablehlo.multiply %v8791, %d4btpv : tensor<512xf32>
    %v8794 = stablehlo.multiply %v1502, %v1502 : tensor<512xf32>
    %v8795 = stablehlo.multiply %v8792, %v8794 : tensor<512xf32>
    %v8796 = stablehlo.add %v8793, %v8795 : tensor<512xf32>
    %v8797 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8798 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8799 = stablehlo.divide %v8790, %v8797 : tensor<512xf32>
    %v8800 = stablehlo.divide %v8796, %v8798 : tensor<512xf32>
    %v8801 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8802 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8803 = stablehlo.sqrt %v8800 : tensor<512xf32>
    %v8804 = stablehlo.add %v8803, %v8802 : tensor<512xf32>
    %v8805 = stablehlo.divide %v8799, %v8804 : tensor<512xf32>
    %v8806 = stablehlo.multiply %v8801, %v8805 : tensor<512xf32>
    %v8807 = stablehlo.subtract %d4btp, %v8806 : tensor<512xf32>
    %v8808 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8809 = stablehlo.multiply %v8808, %v8801 : tensor<512xf32>
    %v8810 = stablehlo.multiply %v8809, %d4btp : tensor<512xf32>
    %v8811 = stablehlo.subtract %v8807, %v8810 : tensor<512xf32>
    %v8812 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8813 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8814 = stablehlo.multiply %v8812, %s4b0W1m : tensor<512x512x3x3xf32>
    %v8815 = stablehlo.multiply %v8813, %v1238 : tensor<512x512x3x3xf32>
    %v8816 = stablehlo.add %v8814, %v8815 : tensor<512x512x3x3xf32>
    %v8817 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8818 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8819 = stablehlo.multiply %v8817, %s4b0W1v : tensor<512x512x3x3xf32>
    %v8820 = stablehlo.multiply %v1238, %v1238 : tensor<512x512x3x3xf32>
    %v8821 = stablehlo.multiply %v8818, %v8820 : tensor<512x512x3x3xf32>
    %v8822 = stablehlo.add %v8819, %v8821 : tensor<512x512x3x3xf32>
    %v8823 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8824 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8825 = stablehlo.multiply %v8823, %s4b0W1m : tensor<512x512x3x3xf32>
    %v8826 = stablehlo.multiply %v8824, %v1238 : tensor<512x512x3x3xf32>
    %v8827 = stablehlo.add %v8825, %v8826 : tensor<512x512x3x3xf32>
    %v8828 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8829 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8830 = stablehlo.multiply %v8828, %s4b0W1v : tensor<512x512x3x3xf32>
    %v8831 = stablehlo.multiply %v1238, %v1238 : tensor<512x512x3x3xf32>
    %v8832 = stablehlo.multiply %v8829, %v8831 : tensor<512x512x3x3xf32>
    %v8833 = stablehlo.add %v8830, %v8832 : tensor<512x512x3x3xf32>
    %v8834 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8835 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8836 = stablehlo.divide %v8827, %v8834 : tensor<512x512x3x3xf32>
    %v8837 = stablehlo.divide %v8833, %v8835 : tensor<512x512x3x3xf32>
    %v8838 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8839 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8840 = stablehlo.sqrt %v8837 : tensor<512x512x3x3xf32>
    %v8841 = stablehlo.add %v8840, %v8839 : tensor<512x512x3x3xf32>
    %v8842 = stablehlo.divide %v8836, %v8841 : tensor<512x512x3x3xf32>
    %v8843 = stablehlo.multiply %v8838, %v8842 : tensor<512x512x3x3xf32>
    %v8844 = stablehlo.subtract %s4b0W1, %v8843 : tensor<512x512x3x3xf32>
    %v8845 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8846 = stablehlo.multiply %v8845, %v8838 : tensor<512x512x3x3xf32>
    %v8847 = stablehlo.multiply %v8846, %s4b0W1 : tensor<512x512x3x3xf32>
    %v8848 = stablehlo.subtract %v8844, %v8847 : tensor<512x512x3x3xf32>
    %v8849 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8850 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8851 = stablehlo.multiply %v8849, %s4b0b1m : tensor<512xf32>
    %v8852 = stablehlo.multiply %v8850, %v1241 : tensor<512xf32>
    %v8853 = stablehlo.add %v8851, %v8852 : tensor<512xf32>
    %v8854 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8855 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8856 = stablehlo.multiply %v8854, %s4b0b1v : tensor<512xf32>
    %v8857 = stablehlo.multiply %v1241, %v1241 : tensor<512xf32>
    %v8858 = stablehlo.multiply %v8855, %v8857 : tensor<512xf32>
    %v8859 = stablehlo.add %v8856, %v8858 : tensor<512xf32>
    %v8860 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8861 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8862 = stablehlo.multiply %v8860, %s4b0b1m : tensor<512xf32>
    %v8863 = stablehlo.multiply %v8861, %v1241 : tensor<512xf32>
    %v8864 = stablehlo.add %v8862, %v8863 : tensor<512xf32>
    %v8865 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8866 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8867 = stablehlo.multiply %v8865, %s4b0b1v : tensor<512xf32>
    %v8868 = stablehlo.multiply %v1241, %v1241 : tensor<512xf32>
    %v8869 = stablehlo.multiply %v8866, %v8868 : tensor<512xf32>
    %v8870 = stablehlo.add %v8867, %v8869 : tensor<512xf32>
    %v8871 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8872 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8873 = stablehlo.divide %v8864, %v8871 : tensor<512xf32>
    %v8874 = stablehlo.divide %v8870, %v8872 : tensor<512xf32>
    %v8875 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8876 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8877 = stablehlo.sqrt %v8874 : tensor<512xf32>
    %v8878 = stablehlo.add %v8877, %v8876 : tensor<512xf32>
    %v8879 = stablehlo.divide %v8873, %v8878 : tensor<512xf32>
    %v8880 = stablehlo.multiply %v8875, %v8879 : tensor<512xf32>
    %v8881 = stablehlo.subtract %s4b0b1, %v8880 : tensor<512xf32>
    %v8882 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8883 = stablehlo.multiply %v8882, %v8875 : tensor<512xf32>
    %v8884 = stablehlo.multiply %v8883, %s4b0b1 : tensor<512xf32>
    %v8885 = stablehlo.subtract %v8881, %v8884 : tensor<512xf32>
    %v8886 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8887 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8888 = stablehlo.multiply %v8886, %s4b0g1m : tensor<512xf32>
    %v8889 = stablehlo.multiply %v8887, %v1259 : tensor<512xf32>
    %v8890 = stablehlo.add %v8888, %v8889 : tensor<512xf32>
    %v8891 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8892 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8893 = stablehlo.multiply %v8891, %s4b0g1v : tensor<512xf32>
    %v8894 = stablehlo.multiply %v1259, %v1259 : tensor<512xf32>
    %v8895 = stablehlo.multiply %v8892, %v8894 : tensor<512xf32>
    %v8896 = stablehlo.add %v8893, %v8895 : tensor<512xf32>
    %v8897 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8898 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8899 = stablehlo.multiply %v8897, %s4b0g1m : tensor<512xf32>
    %v8900 = stablehlo.multiply %v8898, %v1259 : tensor<512xf32>
    %v8901 = stablehlo.add %v8899, %v8900 : tensor<512xf32>
    %v8902 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8903 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8904 = stablehlo.multiply %v8902, %s4b0g1v : tensor<512xf32>
    %v8905 = stablehlo.multiply %v1259, %v1259 : tensor<512xf32>
    %v8906 = stablehlo.multiply %v8903, %v8905 : tensor<512xf32>
    %v8907 = stablehlo.add %v8904, %v8906 : tensor<512xf32>
    %v8908 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8909 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8910 = stablehlo.divide %v8901, %v8908 : tensor<512xf32>
    %v8911 = stablehlo.divide %v8907, %v8909 : tensor<512xf32>
    %v8912 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8913 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8914 = stablehlo.sqrt %v8911 : tensor<512xf32>
    %v8915 = stablehlo.add %v8914, %v8913 : tensor<512xf32>
    %v8916 = stablehlo.divide %v8910, %v8915 : tensor<512xf32>
    %v8917 = stablehlo.multiply %v8912, %v8916 : tensor<512xf32>
    %v8918 = stablehlo.subtract %s4b0g1, %v8917 : tensor<512xf32>
    %v8919 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8920 = stablehlo.multiply %v8919, %v8912 : tensor<512xf32>
    %v8921 = stablehlo.multiply %v8920, %s4b0g1 : tensor<512xf32>
    %v8922 = stablehlo.subtract %v8918, %v8921 : tensor<512xf32>
    %v8923 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8924 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8925 = stablehlo.multiply %v8923, %s4b0bt1m : tensor<512xf32>
    %v8926 = stablehlo.multiply %v8924, %v1262 : tensor<512xf32>
    %v8927 = stablehlo.add %v8925, %v8926 : tensor<512xf32>
    %v8928 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8929 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8930 = stablehlo.multiply %v8928, %s4b0bt1v : tensor<512xf32>
    %v8931 = stablehlo.multiply %v1262, %v1262 : tensor<512xf32>
    %v8932 = stablehlo.multiply %v8929, %v8931 : tensor<512xf32>
    %v8933 = stablehlo.add %v8930, %v8932 : tensor<512xf32>
    %v8934 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8935 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8936 = stablehlo.multiply %v8934, %s4b0bt1m : tensor<512xf32>
    %v8937 = stablehlo.multiply %v8935, %v1262 : tensor<512xf32>
    %v8938 = stablehlo.add %v8936, %v8937 : tensor<512xf32>
    %v8939 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8940 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8941 = stablehlo.multiply %v8939, %s4b0bt1v : tensor<512xf32>
    %v8942 = stablehlo.multiply %v1262, %v1262 : tensor<512xf32>
    %v8943 = stablehlo.multiply %v8940, %v8942 : tensor<512xf32>
    %v8944 = stablehlo.add %v8941, %v8943 : tensor<512xf32>
    %v8945 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8946 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8947 = stablehlo.divide %v8938, %v8945 : tensor<512xf32>
    %v8948 = stablehlo.divide %v8944, %v8946 : tensor<512xf32>
    %v8949 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8950 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8951 = stablehlo.sqrt %v8948 : tensor<512xf32>
    %v8952 = stablehlo.add %v8951, %v8950 : tensor<512xf32>
    %v8953 = stablehlo.divide %v8947, %v8952 : tensor<512xf32>
    %v8954 = stablehlo.multiply %v8949, %v8953 : tensor<512xf32>
    %v8955 = stablehlo.subtract %s4b0bt1, %v8954 : tensor<512xf32>
    %v8956 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8957 = stablehlo.multiply %v8956, %v8949 : tensor<512xf32>
    %v8958 = stablehlo.multiply %v8957, %s4b0bt1 : tensor<512xf32>
    %v8959 = stablehlo.subtract %v8955, %v8958 : tensor<512xf32>
    %v8960 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8961 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8962 = stablehlo.multiply %v8960, %s4b0W2m : tensor<512x512x3x3xf32>
    %v8963 = stablehlo.multiply %v8961, %v1268 : tensor<512x512x3x3xf32>
    %v8964 = stablehlo.add %v8962, %v8963 : tensor<512x512x3x3xf32>
    %v8965 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8966 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8967 = stablehlo.multiply %v8965, %s4b0W2v : tensor<512x512x3x3xf32>
    %v8968 = stablehlo.multiply %v1268, %v1268 : tensor<512x512x3x3xf32>
    %v8969 = stablehlo.multiply %v8966, %v8968 : tensor<512x512x3x3xf32>
    %v8970 = stablehlo.add %v8967, %v8969 : tensor<512x512x3x3xf32>
    %v8971 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8972 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8973 = stablehlo.multiply %v8971, %s4b0W2m : tensor<512x512x3x3xf32>
    %v8974 = stablehlo.multiply %v8972, %v1268 : tensor<512x512x3x3xf32>
    %v8975 = stablehlo.add %v8973, %v8974 : tensor<512x512x3x3xf32>
    %v8976 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8977 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8978 = stablehlo.multiply %v8976, %s4b0W2v : tensor<512x512x3x3xf32>
    %v8979 = stablehlo.multiply %v1268, %v1268 : tensor<512x512x3x3xf32>
    %v8980 = stablehlo.multiply %v8977, %v8979 : tensor<512x512x3x3xf32>
    %v8981 = stablehlo.add %v8978, %v8980 : tensor<512x512x3x3xf32>
    %v8982 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8983 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8984 = stablehlo.divide %v8975, %v8982 : tensor<512x512x3x3xf32>
    %v8985 = stablehlo.divide %v8981, %v8983 : tensor<512x512x3x3xf32>
    %v8986 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8987 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8988 = stablehlo.sqrt %v8985 : tensor<512x512x3x3xf32>
    %v8989 = stablehlo.add %v8988, %v8987 : tensor<512x512x3x3xf32>
    %v8990 = stablehlo.divide %v8984, %v8989 : tensor<512x512x3x3xf32>
    %v8991 = stablehlo.multiply %v8986, %v8990 : tensor<512x512x3x3xf32>
    %v8992 = stablehlo.subtract %s4b0W2, %v8991 : tensor<512x512x3x3xf32>
    %v8993 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8994 = stablehlo.multiply %v8993, %v8986 : tensor<512x512x3x3xf32>
    %v8995 = stablehlo.multiply %v8994, %s4b0W2 : tensor<512x512x3x3xf32>
    %v8996 = stablehlo.subtract %v8992, %v8995 : tensor<512x512x3x3xf32>
    %v8997 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8998 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8999 = stablehlo.multiply %v8997, %s4b0b2m : tensor<512xf32>
    %v9000 = stablehlo.multiply %v8998, %v1271 : tensor<512xf32>
    %v9001 = stablehlo.add %v8999, %v9000 : tensor<512xf32>
    %v9002 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9003 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9004 = stablehlo.multiply %v9002, %s4b0b2v : tensor<512xf32>
    %v9005 = stablehlo.multiply %v1271, %v1271 : tensor<512xf32>
    %v9006 = stablehlo.multiply %v9003, %v9005 : tensor<512xf32>
    %v9007 = stablehlo.add %v9004, %v9006 : tensor<512xf32>
    %v9008 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9009 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9010 = stablehlo.multiply %v9008, %s4b0b2m : tensor<512xf32>
    %v9011 = stablehlo.multiply %v9009, %v1271 : tensor<512xf32>
    %v9012 = stablehlo.add %v9010, %v9011 : tensor<512xf32>
    %v9013 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9014 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9015 = stablehlo.multiply %v9013, %s4b0b2v : tensor<512xf32>
    %v9016 = stablehlo.multiply %v1271, %v1271 : tensor<512xf32>
    %v9017 = stablehlo.multiply %v9014, %v9016 : tensor<512xf32>
    %v9018 = stablehlo.add %v9015, %v9017 : tensor<512xf32>
    %v9019 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9020 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9021 = stablehlo.divide %v9012, %v9019 : tensor<512xf32>
    %v9022 = stablehlo.divide %v9018, %v9020 : tensor<512xf32>
    %v9023 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9024 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9025 = stablehlo.sqrt %v9022 : tensor<512xf32>
    %v9026 = stablehlo.add %v9025, %v9024 : tensor<512xf32>
    %v9027 = stablehlo.divide %v9021, %v9026 : tensor<512xf32>
    %v9028 = stablehlo.multiply %v9023, %v9027 : tensor<512xf32>
    %v9029 = stablehlo.subtract %s4b0b2, %v9028 : tensor<512xf32>
    %v9030 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9031 = stablehlo.multiply %v9030, %v9023 : tensor<512xf32>
    %v9032 = stablehlo.multiply %v9031, %s4b0b2 : tensor<512xf32>
    %v9033 = stablehlo.subtract %v9029, %v9032 : tensor<512xf32>
    %v9034 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9035 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9036 = stablehlo.multiply %v9034, %s4b0g2m : tensor<512xf32>
    %v9037 = stablehlo.multiply %v9035, %v1289 : tensor<512xf32>
    %v9038 = stablehlo.add %v9036, %v9037 : tensor<512xf32>
    %v9039 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9040 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9041 = stablehlo.multiply %v9039, %s4b0g2v : tensor<512xf32>
    %v9042 = stablehlo.multiply %v1289, %v1289 : tensor<512xf32>
    %v9043 = stablehlo.multiply %v9040, %v9042 : tensor<512xf32>
    %v9044 = stablehlo.add %v9041, %v9043 : tensor<512xf32>
    %v9045 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9046 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9047 = stablehlo.multiply %v9045, %s4b0g2m : tensor<512xf32>
    %v9048 = stablehlo.multiply %v9046, %v1289 : tensor<512xf32>
    %v9049 = stablehlo.add %v9047, %v9048 : tensor<512xf32>
    %v9050 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9051 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9052 = stablehlo.multiply %v9050, %s4b0g2v : tensor<512xf32>
    %v9053 = stablehlo.multiply %v1289, %v1289 : tensor<512xf32>
    %v9054 = stablehlo.multiply %v9051, %v9053 : tensor<512xf32>
    %v9055 = stablehlo.add %v9052, %v9054 : tensor<512xf32>
    %v9056 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9057 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9058 = stablehlo.divide %v9049, %v9056 : tensor<512xf32>
    %v9059 = stablehlo.divide %v9055, %v9057 : tensor<512xf32>
    %v9060 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9061 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9062 = stablehlo.sqrt %v9059 : tensor<512xf32>
    %v9063 = stablehlo.add %v9062, %v9061 : tensor<512xf32>
    %v9064 = stablehlo.divide %v9058, %v9063 : tensor<512xf32>
    %v9065 = stablehlo.multiply %v9060, %v9064 : tensor<512xf32>
    %v9066 = stablehlo.subtract %s4b0g2, %v9065 : tensor<512xf32>
    %v9067 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9068 = stablehlo.multiply %v9067, %v9060 : tensor<512xf32>
    %v9069 = stablehlo.multiply %v9068, %s4b0g2 : tensor<512xf32>
    %v9070 = stablehlo.subtract %v9066, %v9069 : tensor<512xf32>
    %v9071 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9072 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9073 = stablehlo.multiply %v9071, %s4b0bt2m : tensor<512xf32>
    %v9074 = stablehlo.multiply %v9072, %v1292 : tensor<512xf32>
    %v9075 = stablehlo.add %v9073, %v9074 : tensor<512xf32>
    %v9076 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9077 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9078 = stablehlo.multiply %v9076, %s4b0bt2v : tensor<512xf32>
    %v9079 = stablehlo.multiply %v1292, %v1292 : tensor<512xf32>
    %v9080 = stablehlo.multiply %v9077, %v9079 : tensor<512xf32>
    %v9081 = stablehlo.add %v9078, %v9080 : tensor<512xf32>
    %v9082 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9083 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9084 = stablehlo.multiply %v9082, %s4b0bt2m : tensor<512xf32>
    %v9085 = stablehlo.multiply %v9083, %v1292 : tensor<512xf32>
    %v9086 = stablehlo.add %v9084, %v9085 : tensor<512xf32>
    %v9087 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9088 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9089 = stablehlo.multiply %v9087, %s4b0bt2v : tensor<512xf32>
    %v9090 = stablehlo.multiply %v1292, %v1292 : tensor<512xf32>
    %v9091 = stablehlo.multiply %v9088, %v9090 : tensor<512xf32>
    %v9092 = stablehlo.add %v9089, %v9091 : tensor<512xf32>
    %v9093 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9094 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9095 = stablehlo.divide %v9086, %v9093 : tensor<512xf32>
    %v9096 = stablehlo.divide %v9092, %v9094 : tensor<512xf32>
    %v9097 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9098 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9099 = stablehlo.sqrt %v9096 : tensor<512xf32>
    %v9100 = stablehlo.add %v9099, %v9098 : tensor<512xf32>
    %v9101 = stablehlo.divide %v9095, %v9100 : tensor<512xf32>
    %v9102 = stablehlo.multiply %v9097, %v9101 : tensor<512xf32>
    %v9103 = stablehlo.subtract %s4b0bt2, %v9102 : tensor<512xf32>
    %v9104 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9105 = stablehlo.multiply %v9104, %v9097 : tensor<512xf32>
    %v9106 = stablehlo.multiply %v9105, %s4b0bt2 : tensor<512xf32>
    %v9107 = stablehlo.subtract %v9103, %v9106 : tensor<512xf32>
    %v9108 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9109 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9110 = stablehlo.multiply %v9108, %s4b1W1m : tensor<512x512x3x3xf32>
    %v9111 = stablehlo.multiply %v9109, %v1101 : tensor<512x512x3x3xf32>
    %v9112 = stablehlo.add %v9110, %v9111 : tensor<512x512x3x3xf32>
    %v9113 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9114 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9115 = stablehlo.multiply %v9113, %s4b1W1v : tensor<512x512x3x3xf32>
    %v9116 = stablehlo.multiply %v1101, %v1101 : tensor<512x512x3x3xf32>
    %v9117 = stablehlo.multiply %v9114, %v9116 : tensor<512x512x3x3xf32>
    %v9118 = stablehlo.add %v9115, %v9117 : tensor<512x512x3x3xf32>
    %v9119 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9120 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9121 = stablehlo.multiply %v9119, %s4b1W1m : tensor<512x512x3x3xf32>
    %v9122 = stablehlo.multiply %v9120, %v1101 : tensor<512x512x3x3xf32>
    %v9123 = stablehlo.add %v9121, %v9122 : tensor<512x512x3x3xf32>
    %v9124 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9125 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9126 = stablehlo.multiply %v9124, %s4b1W1v : tensor<512x512x3x3xf32>
    %v9127 = stablehlo.multiply %v1101, %v1101 : tensor<512x512x3x3xf32>
    %v9128 = stablehlo.multiply %v9125, %v9127 : tensor<512x512x3x3xf32>
    %v9129 = stablehlo.add %v9126, %v9128 : tensor<512x512x3x3xf32>
    %v9130 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9131 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9132 = stablehlo.divide %v9123, %v9130 : tensor<512x512x3x3xf32>
    %v9133 = stablehlo.divide %v9129, %v9131 : tensor<512x512x3x3xf32>
    %v9134 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9135 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9136 = stablehlo.sqrt %v9133 : tensor<512x512x3x3xf32>
    %v9137 = stablehlo.add %v9136, %v9135 : tensor<512x512x3x3xf32>
    %v9138 = stablehlo.divide %v9132, %v9137 : tensor<512x512x3x3xf32>
    %v9139 = stablehlo.multiply %v9134, %v9138 : tensor<512x512x3x3xf32>
    %v9140 = stablehlo.subtract %s4b1W1, %v9139 : tensor<512x512x3x3xf32>
    %v9141 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9142 = stablehlo.multiply %v9141, %v9134 : tensor<512x512x3x3xf32>
    %v9143 = stablehlo.multiply %v9142, %s4b1W1 : tensor<512x512x3x3xf32>
    %v9144 = stablehlo.subtract %v9140, %v9143 : tensor<512x512x3x3xf32>
    %v9145 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9146 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9147 = stablehlo.multiply %v9145, %s4b1b1m : tensor<512xf32>
    %v9148 = stablehlo.multiply %v9146, %v1104 : tensor<512xf32>
    %v9149 = stablehlo.add %v9147, %v9148 : tensor<512xf32>
    %v9150 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9151 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9152 = stablehlo.multiply %v9150, %s4b1b1v : tensor<512xf32>
    %v9153 = stablehlo.multiply %v1104, %v1104 : tensor<512xf32>
    %v9154 = stablehlo.multiply %v9151, %v9153 : tensor<512xf32>
    %v9155 = stablehlo.add %v9152, %v9154 : tensor<512xf32>
    %v9156 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9157 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9158 = stablehlo.multiply %v9156, %s4b1b1m : tensor<512xf32>
    %v9159 = stablehlo.multiply %v9157, %v1104 : tensor<512xf32>
    %v9160 = stablehlo.add %v9158, %v9159 : tensor<512xf32>
    %v9161 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9162 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9163 = stablehlo.multiply %v9161, %s4b1b1v : tensor<512xf32>
    %v9164 = stablehlo.multiply %v1104, %v1104 : tensor<512xf32>
    %v9165 = stablehlo.multiply %v9162, %v9164 : tensor<512xf32>
    %v9166 = stablehlo.add %v9163, %v9165 : tensor<512xf32>
    %v9167 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9168 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9169 = stablehlo.divide %v9160, %v9167 : tensor<512xf32>
    %v9170 = stablehlo.divide %v9166, %v9168 : tensor<512xf32>
    %v9171 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9172 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9173 = stablehlo.sqrt %v9170 : tensor<512xf32>
    %v9174 = stablehlo.add %v9173, %v9172 : tensor<512xf32>
    %v9175 = stablehlo.divide %v9169, %v9174 : tensor<512xf32>
    %v9176 = stablehlo.multiply %v9171, %v9175 : tensor<512xf32>
    %v9177 = stablehlo.subtract %s4b1b1, %v9176 : tensor<512xf32>
    %v9178 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9179 = stablehlo.multiply %v9178, %v9171 : tensor<512xf32>
    %v9180 = stablehlo.multiply %v9179, %s4b1b1 : tensor<512xf32>
    %v9181 = stablehlo.subtract %v9177, %v9180 : tensor<512xf32>
    %v9182 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9183 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9184 = stablehlo.multiply %v9182, %s4b1g1m : tensor<512xf32>
    %v9185 = stablehlo.multiply %v9183, %v1122 : tensor<512xf32>
    %v9186 = stablehlo.add %v9184, %v9185 : tensor<512xf32>
    %v9187 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9188 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9189 = stablehlo.multiply %v9187, %s4b1g1v : tensor<512xf32>
    %v9190 = stablehlo.multiply %v1122, %v1122 : tensor<512xf32>
    %v9191 = stablehlo.multiply %v9188, %v9190 : tensor<512xf32>
    %v9192 = stablehlo.add %v9189, %v9191 : tensor<512xf32>
    %v9193 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9194 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9195 = stablehlo.multiply %v9193, %s4b1g1m : tensor<512xf32>
    %v9196 = stablehlo.multiply %v9194, %v1122 : tensor<512xf32>
    %v9197 = stablehlo.add %v9195, %v9196 : tensor<512xf32>
    %v9198 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9199 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9200 = stablehlo.multiply %v9198, %s4b1g1v : tensor<512xf32>
    %v9201 = stablehlo.multiply %v1122, %v1122 : tensor<512xf32>
    %v9202 = stablehlo.multiply %v9199, %v9201 : tensor<512xf32>
    %v9203 = stablehlo.add %v9200, %v9202 : tensor<512xf32>
    %v9204 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9205 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9206 = stablehlo.divide %v9197, %v9204 : tensor<512xf32>
    %v9207 = stablehlo.divide %v9203, %v9205 : tensor<512xf32>
    %v9208 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9209 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9210 = stablehlo.sqrt %v9207 : tensor<512xf32>
    %v9211 = stablehlo.add %v9210, %v9209 : tensor<512xf32>
    %v9212 = stablehlo.divide %v9206, %v9211 : tensor<512xf32>
    %v9213 = stablehlo.multiply %v9208, %v9212 : tensor<512xf32>
    %v9214 = stablehlo.subtract %s4b1g1, %v9213 : tensor<512xf32>
    %v9215 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9216 = stablehlo.multiply %v9215, %v9208 : tensor<512xf32>
    %v9217 = stablehlo.multiply %v9216, %s4b1g1 : tensor<512xf32>
    %v9218 = stablehlo.subtract %v9214, %v9217 : tensor<512xf32>
    %v9219 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9220 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9221 = stablehlo.multiply %v9219, %s4b1bt1m : tensor<512xf32>
    %v9222 = stablehlo.multiply %v9220, %v1125 : tensor<512xf32>
    %v9223 = stablehlo.add %v9221, %v9222 : tensor<512xf32>
    %v9224 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9225 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9226 = stablehlo.multiply %v9224, %s4b1bt1v : tensor<512xf32>
    %v9227 = stablehlo.multiply %v1125, %v1125 : tensor<512xf32>
    %v9228 = stablehlo.multiply %v9225, %v9227 : tensor<512xf32>
    %v9229 = stablehlo.add %v9226, %v9228 : tensor<512xf32>
    %v9230 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9231 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9232 = stablehlo.multiply %v9230, %s4b1bt1m : tensor<512xf32>
    %v9233 = stablehlo.multiply %v9231, %v1125 : tensor<512xf32>
    %v9234 = stablehlo.add %v9232, %v9233 : tensor<512xf32>
    %v9235 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9236 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9237 = stablehlo.multiply %v9235, %s4b1bt1v : tensor<512xf32>
    %v9238 = stablehlo.multiply %v1125, %v1125 : tensor<512xf32>
    %v9239 = stablehlo.multiply %v9236, %v9238 : tensor<512xf32>
    %v9240 = stablehlo.add %v9237, %v9239 : tensor<512xf32>
    %v9241 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9242 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9243 = stablehlo.divide %v9234, %v9241 : tensor<512xf32>
    %v9244 = stablehlo.divide %v9240, %v9242 : tensor<512xf32>
    %v9245 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9246 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9247 = stablehlo.sqrt %v9244 : tensor<512xf32>
    %v9248 = stablehlo.add %v9247, %v9246 : tensor<512xf32>
    %v9249 = stablehlo.divide %v9243, %v9248 : tensor<512xf32>
    %v9250 = stablehlo.multiply %v9245, %v9249 : tensor<512xf32>
    %v9251 = stablehlo.subtract %s4b1bt1, %v9250 : tensor<512xf32>
    %v9252 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9253 = stablehlo.multiply %v9252, %v9245 : tensor<512xf32>
    %v9254 = stablehlo.multiply %v9253, %s4b1bt1 : tensor<512xf32>
    %v9255 = stablehlo.subtract %v9251, %v9254 : tensor<512xf32>
    %v9256 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9257 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9258 = stablehlo.multiply %v9256, %s4b1W2m : tensor<512x512x3x3xf32>
    %v9259 = stablehlo.multiply %v9257, %v1131 : tensor<512x512x3x3xf32>
    %v9260 = stablehlo.add %v9258, %v9259 : tensor<512x512x3x3xf32>
    %v9261 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9262 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9263 = stablehlo.multiply %v9261, %s4b1W2v : tensor<512x512x3x3xf32>
    %v9264 = stablehlo.multiply %v1131, %v1131 : tensor<512x512x3x3xf32>
    %v9265 = stablehlo.multiply %v9262, %v9264 : tensor<512x512x3x3xf32>
    %v9266 = stablehlo.add %v9263, %v9265 : tensor<512x512x3x3xf32>
    %v9267 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9268 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9269 = stablehlo.multiply %v9267, %s4b1W2m : tensor<512x512x3x3xf32>
    %v9270 = stablehlo.multiply %v9268, %v1131 : tensor<512x512x3x3xf32>
    %v9271 = stablehlo.add %v9269, %v9270 : tensor<512x512x3x3xf32>
    %v9272 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9273 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9274 = stablehlo.multiply %v9272, %s4b1W2v : tensor<512x512x3x3xf32>
    %v9275 = stablehlo.multiply %v1131, %v1131 : tensor<512x512x3x3xf32>
    %v9276 = stablehlo.multiply %v9273, %v9275 : tensor<512x512x3x3xf32>
    %v9277 = stablehlo.add %v9274, %v9276 : tensor<512x512x3x3xf32>
    %v9278 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9279 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9280 = stablehlo.divide %v9271, %v9278 : tensor<512x512x3x3xf32>
    %v9281 = stablehlo.divide %v9277, %v9279 : tensor<512x512x3x3xf32>
    %v9282 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9283 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9284 = stablehlo.sqrt %v9281 : tensor<512x512x3x3xf32>
    %v9285 = stablehlo.add %v9284, %v9283 : tensor<512x512x3x3xf32>
    %v9286 = stablehlo.divide %v9280, %v9285 : tensor<512x512x3x3xf32>
    %v9287 = stablehlo.multiply %v9282, %v9286 : tensor<512x512x3x3xf32>
    %v9288 = stablehlo.subtract %s4b1W2, %v9287 : tensor<512x512x3x3xf32>
    %v9289 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v9290 = stablehlo.multiply %v9289, %v9282 : tensor<512x512x3x3xf32>
    %v9291 = stablehlo.multiply %v9290, %s4b1W2 : tensor<512x512x3x3xf32>
    %v9292 = stablehlo.subtract %v9288, %v9291 : tensor<512x512x3x3xf32>
    %v9293 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9294 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9295 = stablehlo.multiply %v9293, %s4b1b2m : tensor<512xf32>
    %v9296 = stablehlo.multiply %v9294, %v1134 : tensor<512xf32>
    %v9297 = stablehlo.add %v9295, %v9296 : tensor<512xf32>
    %v9298 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9299 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9300 = stablehlo.multiply %v9298, %s4b1b2v : tensor<512xf32>
    %v9301 = stablehlo.multiply %v1134, %v1134 : tensor<512xf32>
    %v9302 = stablehlo.multiply %v9299, %v9301 : tensor<512xf32>
    %v9303 = stablehlo.add %v9300, %v9302 : tensor<512xf32>
    %v9304 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9305 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9306 = stablehlo.multiply %v9304, %s4b1b2m : tensor<512xf32>
    %v9307 = stablehlo.multiply %v9305, %v1134 : tensor<512xf32>
    %v9308 = stablehlo.add %v9306, %v9307 : tensor<512xf32>
    %v9309 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9310 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9311 = stablehlo.multiply %v9309, %s4b1b2v : tensor<512xf32>
    %v9312 = stablehlo.multiply %v1134, %v1134 : tensor<512xf32>
    %v9313 = stablehlo.multiply %v9310, %v9312 : tensor<512xf32>
    %v9314 = stablehlo.add %v9311, %v9313 : tensor<512xf32>
    %v9315 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9316 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9317 = stablehlo.divide %v9308, %v9315 : tensor<512xf32>
    %v9318 = stablehlo.divide %v9314, %v9316 : tensor<512xf32>
    %v9319 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9320 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9321 = stablehlo.sqrt %v9318 : tensor<512xf32>
    %v9322 = stablehlo.add %v9321, %v9320 : tensor<512xf32>
    %v9323 = stablehlo.divide %v9317, %v9322 : tensor<512xf32>
    %v9324 = stablehlo.multiply %v9319, %v9323 : tensor<512xf32>
    %v9325 = stablehlo.subtract %s4b1b2, %v9324 : tensor<512xf32>
    %v9326 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9327 = stablehlo.multiply %v9326, %v9319 : tensor<512xf32>
    %v9328 = stablehlo.multiply %v9327, %s4b1b2 : tensor<512xf32>
    %v9329 = stablehlo.subtract %v9325, %v9328 : tensor<512xf32>
    %v9330 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9331 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9332 = stablehlo.multiply %v9330, %s4b1g2m : tensor<512xf32>
    %v9333 = stablehlo.multiply %v9331, %v1152 : tensor<512xf32>
    %v9334 = stablehlo.add %v9332, %v9333 : tensor<512xf32>
    %v9335 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9336 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9337 = stablehlo.multiply %v9335, %s4b1g2v : tensor<512xf32>
    %v9338 = stablehlo.multiply %v1152, %v1152 : tensor<512xf32>
    %v9339 = stablehlo.multiply %v9336, %v9338 : tensor<512xf32>
    %v9340 = stablehlo.add %v9337, %v9339 : tensor<512xf32>
    %v9341 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9342 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9343 = stablehlo.multiply %v9341, %s4b1g2m : tensor<512xf32>
    %v9344 = stablehlo.multiply %v9342, %v1152 : tensor<512xf32>
    %v9345 = stablehlo.add %v9343, %v9344 : tensor<512xf32>
    %v9346 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9347 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9348 = stablehlo.multiply %v9346, %s4b1g2v : tensor<512xf32>
    %v9349 = stablehlo.multiply %v1152, %v1152 : tensor<512xf32>
    %v9350 = stablehlo.multiply %v9347, %v9349 : tensor<512xf32>
    %v9351 = stablehlo.add %v9348, %v9350 : tensor<512xf32>
    %v9352 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9353 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9354 = stablehlo.divide %v9345, %v9352 : tensor<512xf32>
    %v9355 = stablehlo.divide %v9351, %v9353 : tensor<512xf32>
    %v9356 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9357 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9358 = stablehlo.sqrt %v9355 : tensor<512xf32>
    %v9359 = stablehlo.add %v9358, %v9357 : tensor<512xf32>
    %v9360 = stablehlo.divide %v9354, %v9359 : tensor<512xf32>
    %v9361 = stablehlo.multiply %v9356, %v9360 : tensor<512xf32>
    %v9362 = stablehlo.subtract %s4b1g2, %v9361 : tensor<512xf32>
    %v9363 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9364 = stablehlo.multiply %v9363, %v9356 : tensor<512xf32>
    %v9365 = stablehlo.multiply %v9364, %s4b1g2 : tensor<512xf32>
    %v9366 = stablehlo.subtract %v9362, %v9365 : tensor<512xf32>
    %v9367 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9368 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9369 = stablehlo.multiply %v9367, %s4b1bt2m : tensor<512xf32>
    %v9370 = stablehlo.multiply %v9368, %v1155 : tensor<512xf32>
    %v9371 = stablehlo.add %v9369, %v9370 : tensor<512xf32>
    %v9372 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9373 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9374 = stablehlo.multiply %v9372, %s4b1bt2v : tensor<512xf32>
    %v9375 = stablehlo.multiply %v1155, %v1155 : tensor<512xf32>
    %v9376 = stablehlo.multiply %v9373, %v9375 : tensor<512xf32>
    %v9377 = stablehlo.add %v9374, %v9376 : tensor<512xf32>
    %v9378 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9379 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9380 = stablehlo.multiply %v9378, %s4b1bt2m : tensor<512xf32>
    %v9381 = stablehlo.multiply %v9379, %v1155 : tensor<512xf32>
    %v9382 = stablehlo.add %v9380, %v9381 : tensor<512xf32>
    %v9383 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9384 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9385 = stablehlo.multiply %v9383, %s4b1bt2v : tensor<512xf32>
    %v9386 = stablehlo.multiply %v1155, %v1155 : tensor<512xf32>
    %v9387 = stablehlo.multiply %v9384, %v9386 : tensor<512xf32>
    %v9388 = stablehlo.add %v9385, %v9387 : tensor<512xf32>
    %v9389 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9390 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9391 = stablehlo.divide %v9382, %v9389 : tensor<512xf32>
    %v9392 = stablehlo.divide %v9388, %v9390 : tensor<512xf32>
    %v9393 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9394 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9395 = stablehlo.sqrt %v9392 : tensor<512xf32>
    %v9396 = stablehlo.add %v9395, %v9394 : tensor<512xf32>
    %v9397 = stablehlo.divide %v9391, %v9396 : tensor<512xf32>
    %v9398 = stablehlo.multiply %v9393, %v9397 : tensor<512xf32>
    %v9399 = stablehlo.subtract %s4b1bt2, %v9398 : tensor<512xf32>
    %v9400 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v9401 = stablehlo.multiply %v9400, %v9393 : tensor<512xf32>
    %v9402 = stablehlo.multiply %v9401, %s4b1bt2 : tensor<512xf32>
    %v9403 = stablehlo.subtract %v9399, %v9402 : tensor<512xf32>
    %v9404 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v9405 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v9406 = stablehlo.multiply %v9404, %Wdm : tensor<512x10xf32>
    %v9407 = stablehlo.multiply %v9405, %v1012 : tensor<512x10xf32>
    %v9408 = stablehlo.add %v9406, %v9407 : tensor<512x10xf32>
    %v9409 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v9410 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v9411 = stablehlo.multiply %v9409, %Wdv : tensor<512x10xf32>
    %v9412 = stablehlo.multiply %v1012, %v1012 : tensor<512x10xf32>
    %v9413 = stablehlo.multiply %v9410, %v9412 : tensor<512x10xf32>
    %v9414 = stablehlo.add %v9411, %v9413 : tensor<512x10xf32>
    %v9415 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v9416 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v9417 = stablehlo.multiply %v9415, %Wdm : tensor<512x10xf32>
    %v9418 = stablehlo.multiply %v9416, %v1012 : tensor<512x10xf32>
    %v9419 = stablehlo.add %v9417, %v9418 : tensor<512x10xf32>
    %v9420 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v9421 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v9422 = stablehlo.multiply %v9420, %Wdv : tensor<512x10xf32>
    %v9423 = stablehlo.multiply %v1012, %v1012 : tensor<512x10xf32>
    %v9424 = stablehlo.multiply %v9421, %v9423 : tensor<512x10xf32>
    %v9425 = stablehlo.add %v9422, %v9424 : tensor<512x10xf32>
    %v9426 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v9427 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v9428 = stablehlo.divide %v9419, %v9426 : tensor<512x10xf32>
    %v9429 = stablehlo.divide %v9425, %v9427 : tensor<512x10xf32>
    %v9430 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v9431 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v9432 = stablehlo.sqrt %v9429 : tensor<512x10xf32>
    %v9433 = stablehlo.add %v9432, %v9431 : tensor<512x10xf32>
    %v9434 = stablehlo.divide %v9428, %v9433 : tensor<512x10xf32>
    %v9435 = stablehlo.multiply %v9430, %v9434 : tensor<512x10xf32>
    %v9436 = stablehlo.subtract %Wd, %v9435 : tensor<512x10xf32>
    %v9437 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v9438 = stablehlo.multiply %v9437, %v9430 : tensor<512x10xf32>
    %v9439 = stablehlo.multiply %v9438, %Wd : tensor<512x10xf32>
    %v9440 = stablehlo.subtract %v9436, %v9439 : tensor<512x10xf32>
    %v9441 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9442 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9443 = stablehlo.multiply %v9441, %bdm : tensor<10xf32>
    %v9444 = stablehlo.multiply %v9442, %v1014 : tensor<10xf32>
    %v9445 = stablehlo.add %v9443, %v9444 : tensor<10xf32>
    %v9446 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9447 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9448 = stablehlo.multiply %v9446, %bdv : tensor<10xf32>
    %v9449 = stablehlo.multiply %v1014, %v1014 : tensor<10xf32>
    %v9450 = stablehlo.multiply %v9447, %v9449 : tensor<10xf32>
    %v9451 = stablehlo.add %v9448, %v9450 : tensor<10xf32>
    %v9452 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9453 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9454 = stablehlo.multiply %v9452, %bdm : tensor<10xf32>
    %v9455 = stablehlo.multiply %v9453, %v1014 : tensor<10xf32>
    %v9456 = stablehlo.add %v9454, %v9455 : tensor<10xf32>
    %v9457 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9458 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9459 = stablehlo.multiply %v9457, %bdv : tensor<10xf32>
    %v9460 = stablehlo.multiply %v1014, %v1014 : tensor<10xf32>
    %v9461 = stablehlo.multiply %v9458, %v9460 : tensor<10xf32>
    %v9462 = stablehlo.add %v9459, %v9461 : tensor<10xf32>
    %v9463 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9464 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9465 = stablehlo.divide %v9456, %v9463 : tensor<10xf32>
    %v9466 = stablehlo.divide %v9462, %v9464 : tensor<10xf32>
    %v9467 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9468 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9469 = stablehlo.sqrt %v9466 : tensor<10xf32>
    %v9470 = stablehlo.add %v9469, %v9468 : tensor<10xf32>
    %v9471 = stablehlo.divide %v9465, %v9470 : tensor<10xf32>
    %v9472 = stablehlo.multiply %v9467, %v9471 : tensor<10xf32>
    %v9473 = stablehlo.subtract %bd, %v9472 : tensor<10xf32>
    %v9474 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9475 = stablehlo.multiply %v9474, %v9467 : tensor<10xf32>
    %v9476 = stablehlo.multiply %v9475, %bd : tensor<10xf32>
    %v9477 = stablehlo.subtract %v9473, %v9476 : tensor<10xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1000 : tensor<32x10xf32>
    %lohll = stablehlo.multiply %onehot, %llog : tensor<32x10xf32>
    %lsum2 = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<f32>
    %lbn = stablehlo.constant dense<32.0> : tensor<f32>
    %lmean = stablehlo.divide %lsum2, %lbn : tensor<f32>
    %loss = stablehlo.negate %lmean : tensor<f32>
    return %v4112, %v4149, %v4186, %v4223, %v4260, %v4297, %v4334, %v4371, %v4408, %v4445, %v4482, %v4519, %v4556, %v4593, %v4630, %v4667, %v4704, %v4741, %v4778, %v4815, %v4852, %v4889, %v4926, %v4963, %v5000, %v5037, %v5074, %v5111, %v5148, %v5185, %v5222, %v5259, %v5296, %v5333, %v5370, %v5407, %v5444, %v5481, %v5518, %v5555, %v5592, %v5629, %v5666, %v5703, %v5740, %v5777, %v5814, %v5851, %v5888, %v5925, %v5962, %v5999, %v6036, %v6073, %v6110, %v6147, %v6184, %v6221, %v6258, %v6295, %v6332, %v6369, %v6406, %v6443, %v6480, %v6517, %v6554, %v6591, %v6628, %v6665, %v6702, %v6739, %v6776, %v6813, %v6850, %v6887, %v6924, %v6961, %v6998, %v7035, %v7072, %v7109, %v7146, %v7183, %v7220, %v7257, %v7294, %v7331, %v7368, %v7405, %v7442, %v7479, %v7516, %v7553, %v7590, %v7627, %v7664, %v7701, %v7738, %v7775, %v7812, %v7849, %v7886, %v7923, %v7960, %v7997, %v8034, %v8071, %v8108, %v8145, %v8182, %v8219, %v8256, %v8293, %v8330, %v8367, %v8404, %v8441, %v8478, %v8515, %v8552, %v8589, %v8626, %v8663, %v8700, %v8737, %v8774, %v8811, %v8848, %v8885, %v8922, %v8959, %v8996, %v9033, %v9070, %v9107, %v9144, %v9181, %v9218, %v9255, %v9292, %v9329, %v9366, %v9403, %v9440, %v9477, %v4080, %v4117, %v4154, %v4191, %v4228, %v4265, %v4302, %v4339, %v4376, %v4413, %v4450, %v4487, %v4524, %v4561, %v4598, %v4635, %v4672, %v4709, %v4746, %v4783, %v4820, %v4857, %v4894, %v4931, %v4968, %v5005, %v5042, %v5079, %v5116, %v5153, %v5190, %v5227, %v5264, %v5301, %v5338, %v5375, %v5412, %v5449, %v5486, %v5523, %v5560, %v5597, %v5634, %v5671, %v5708, %v5745, %v5782, %v5819, %v5856, %v5893, %v5930, %v5967, %v6004, %v6041, %v6078, %v6115, %v6152, %v6189, %v6226, %v6263, %v6300, %v6337, %v6374, %v6411, %v6448, %v6485, %v6522, %v6559, %v6596, %v6633, %v6670, %v6707, %v6744, %v6781, %v6818, %v6855, %v6892, %v6929, %v6966, %v7003, %v7040, %v7077, %v7114, %v7151, %v7188, %v7225, %v7262, %v7299, %v7336, %v7373, %v7410, %v7447, %v7484, %v7521, %v7558, %v7595, %v7632, %v7669, %v7706, %v7743, %v7780, %v7817, %v7854, %v7891, %v7928, %v7965, %v8002, %v8039, %v8076, %v8113, %v8150, %v8187, %v8224, %v8261, %v8298, %v8335, %v8372, %v8409, %v8446, %v8483, %v8520, %v8557, %v8594, %v8631, %v8668, %v8705, %v8742, %v8779, %v8816, %v8853, %v8890, %v8927, %v8964, %v9001, %v9038, %v9075, %v9112, %v9149, %v9186, %v9223, %v9260, %v9297, %v9334, %v9371, %v9408, %v9445, %v4086, %v4123, %v4160, %v4197, %v4234, %v4271, %v4308, %v4345, %v4382, %v4419, %v4456, %v4493, %v4530, %v4567, %v4604, %v4641, %v4678, %v4715, %v4752, %v4789, %v4826, %v4863, %v4900, %v4937, %v4974, %v5011, %v5048, %v5085, %v5122, %v5159, %v5196, %v5233, %v5270, %v5307, %v5344, %v5381, %v5418, %v5455, %v5492, %v5529, %v5566, %v5603, %v5640, %v5677, %v5714, %v5751, %v5788, %v5825, %v5862, %v5899, %v5936, %v5973, %v6010, %v6047, %v6084, %v6121, %v6158, %v6195, %v6232, %v6269, %v6306, %v6343, %v6380, %v6417, %v6454, %v6491, %v6528, %v6565, %v6602, %v6639, %v6676, %v6713, %v6750, %v6787, %v6824, %v6861, %v6898, %v6935, %v6972, %v7009, %v7046, %v7083, %v7120, %v7157, %v7194, %v7231, %v7268, %v7305, %v7342, %v7379, %v7416, %v7453, %v7490, %v7527, %v7564, %v7601, %v7638, %v7675, %v7712, %v7749, %v7786, %v7823, %v7860, %v7897, %v7934, %v7971, %v8008, %v8045, %v8082, %v8119, %v8156, %v8193, %v8230, %v8267, %v8304, %v8341, %v8378, %v8415, %v8452, %v8489, %v8526, %v8563, %v8600, %v8637, %v8674, %v8711, %v8748, %v8785, %v8822, %v8859, %v8896, %v8933, %v8970, %v9007, %v9044, %v9081, %v9118, %v9155, %v9192, %v9229, %v9266, %v9303, %v9340, %v9377, %v9414, %v9451, %loss, %bc1, %bc2, %v3504, %v3515, %v3520, %v3531, %v3536, %v3547, %v3552, %v3563, %v3568, %v3579, %v3584, %v3595, %v3600, %v3611, %v3616, %v3627, %v3632, %v3643, %v3648, %v3659, %v3664, %v3675, %v3680, %v3691, %v3696, %v3707, %v3712, %v3723, %v3728, %v3739, %v3744, %v3755, %v3760, %v3771, %v3776, %v3787, %v3792, %v3803, %v3808, %v3819, %v3824, %v3835, %v3840, %v3851, %v3856, %v3867, %v3872, %v3883, %v3888, %v3899, %v3904, %v3915, %v3920, %v3931, %v3936, %v3947, %v3952, %v3963, %v3968, %v3979, %v3984, %v3995, %v4000, %v4011, %v4016, %v4027, %v4032, %v4043, %v4048, %v4059, %v4064, %v4075 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}

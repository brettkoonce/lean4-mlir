module @m {
  func.func @vit_adam_train_step(%x: tensor<32x150528xf32>, %wConv: tensor<192x3x16x16xf32>, %bConv: tensor<192xf32>, %cls: tensor<192xf32>, %pos: tensor<197x192xf32>, %g1_0: tensor<192xf32>, %b1_0: tensor<192xf32>, %Wq_0: tensor<192x192xf32>, %bq_0: tensor<192xf32>, %Wk_0: tensor<192x192xf32>, %bk_0: tensor<192xf32>, %Wv_0: tensor<192x192xf32>, %bv_0: tensor<192xf32>, %Wo_0: tensor<192x192xf32>, %bo_0: tensor<192xf32>, %g2_0: tensor<192xf32>, %b2_0: tensor<192xf32>, %Wfc1_0: tensor<192x768xf32>, %bfc1_0: tensor<768xf32>, %Wfc2_0: tensor<768x192xf32>, %bfc2_0: tensor<192xf32>, %g1_1: tensor<192xf32>, %b1_1: tensor<192xf32>, %Wq_1: tensor<192x192xf32>, %bq_1: tensor<192xf32>, %Wk_1: tensor<192x192xf32>, %bk_1: tensor<192xf32>, %Wv_1: tensor<192x192xf32>, %bv_1: tensor<192xf32>, %Wo_1: tensor<192x192xf32>, %bo_1: tensor<192xf32>, %g2_1: tensor<192xf32>, %b2_1: tensor<192xf32>, %Wfc1_1: tensor<192x768xf32>, %bfc1_1: tensor<768xf32>, %Wfc2_1: tensor<768x192xf32>, %bfc2_1: tensor<192xf32>, %g1_2: tensor<192xf32>, %b1_2: tensor<192xf32>, %Wq_2: tensor<192x192xf32>, %bq_2: tensor<192xf32>, %Wk_2: tensor<192x192xf32>, %bk_2: tensor<192xf32>, %Wv_2: tensor<192x192xf32>, %bv_2: tensor<192xf32>, %Wo_2: tensor<192x192xf32>, %bo_2: tensor<192xf32>, %g2_2: tensor<192xf32>, %b2_2: tensor<192xf32>, %Wfc1_2: tensor<192x768xf32>, %bfc1_2: tensor<768xf32>, %Wfc2_2: tensor<768x192xf32>, %bfc2_2: tensor<192xf32>, %g1_3: tensor<192xf32>, %b1_3: tensor<192xf32>, %Wq_3: tensor<192x192xf32>, %bq_3: tensor<192xf32>, %Wk_3: tensor<192x192xf32>, %bk_3: tensor<192xf32>, %Wv_3: tensor<192x192xf32>, %bv_3: tensor<192xf32>, %Wo_3: tensor<192x192xf32>, %bo_3: tensor<192xf32>, %g2_3: tensor<192xf32>, %b2_3: tensor<192xf32>, %Wfc1_3: tensor<192x768xf32>, %bfc1_3: tensor<768xf32>, %Wfc2_3: tensor<768x192xf32>, %bfc2_3: tensor<192xf32>, %g1_4: tensor<192xf32>, %b1_4: tensor<192xf32>, %Wq_4: tensor<192x192xf32>, %bq_4: tensor<192xf32>, %Wk_4: tensor<192x192xf32>, %bk_4: tensor<192xf32>, %Wv_4: tensor<192x192xf32>, %bv_4: tensor<192xf32>, %Wo_4: tensor<192x192xf32>, %bo_4: tensor<192xf32>, %g2_4: tensor<192xf32>, %b2_4: tensor<192xf32>, %Wfc1_4: tensor<192x768xf32>, %bfc1_4: tensor<768xf32>, %Wfc2_4: tensor<768x192xf32>, %bfc2_4: tensor<192xf32>, %g1_5: tensor<192xf32>, %b1_5: tensor<192xf32>, %Wq_5: tensor<192x192xf32>, %bq_5: tensor<192xf32>, %Wk_5: tensor<192x192xf32>, %bk_5: tensor<192xf32>, %Wv_5: tensor<192x192xf32>, %bv_5: tensor<192xf32>, %Wo_5: tensor<192x192xf32>, %bo_5: tensor<192xf32>, %g2_5: tensor<192xf32>, %b2_5: tensor<192xf32>, %Wfc1_5: tensor<192x768xf32>, %bfc1_5: tensor<768xf32>, %Wfc2_5: tensor<768x192xf32>, %bfc2_5: tensor<192xf32>, %g1_6: tensor<192xf32>, %b1_6: tensor<192xf32>, %Wq_6: tensor<192x192xf32>, %bq_6: tensor<192xf32>, %Wk_6: tensor<192x192xf32>, %bk_6: tensor<192xf32>, %Wv_6: tensor<192x192xf32>, %bv_6: tensor<192xf32>, %Wo_6: tensor<192x192xf32>, %bo_6: tensor<192xf32>, %g2_6: tensor<192xf32>, %b2_6: tensor<192xf32>, %Wfc1_6: tensor<192x768xf32>, %bfc1_6: tensor<768xf32>, %Wfc2_6: tensor<768x192xf32>, %bfc2_6: tensor<192xf32>, %g1_7: tensor<192xf32>, %b1_7: tensor<192xf32>, %Wq_7: tensor<192x192xf32>, %bq_7: tensor<192xf32>, %Wk_7: tensor<192x192xf32>, %bk_7: tensor<192xf32>, %Wv_7: tensor<192x192xf32>, %bv_7: tensor<192xf32>, %Wo_7: tensor<192x192xf32>, %bo_7: tensor<192xf32>, %g2_7: tensor<192xf32>, %b2_7: tensor<192xf32>, %Wfc1_7: tensor<192x768xf32>, %bfc1_7: tensor<768xf32>, %Wfc2_7: tensor<768x192xf32>, %bfc2_7: tensor<192xf32>, %g1_8: tensor<192xf32>, %b1_8: tensor<192xf32>, %Wq_8: tensor<192x192xf32>, %bq_8: tensor<192xf32>, %Wk_8: tensor<192x192xf32>, %bk_8: tensor<192xf32>, %Wv_8: tensor<192x192xf32>, %bv_8: tensor<192xf32>, %Wo_8: tensor<192x192xf32>, %bo_8: tensor<192xf32>, %g2_8: tensor<192xf32>, %b2_8: tensor<192xf32>, %Wfc1_8: tensor<192x768xf32>, %bfc1_8: tensor<768xf32>, %Wfc2_8: tensor<768x192xf32>, %bfc2_8: tensor<192xf32>, %g1_9: tensor<192xf32>, %b1_9: tensor<192xf32>, %Wq_9: tensor<192x192xf32>, %bq_9: tensor<192xf32>, %Wk_9: tensor<192x192xf32>, %bk_9: tensor<192xf32>, %Wv_9: tensor<192x192xf32>, %bv_9: tensor<192xf32>, %Wo_9: tensor<192x192xf32>, %bo_9: tensor<192xf32>, %g2_9: tensor<192xf32>, %b2_9: tensor<192xf32>, %Wfc1_9: tensor<192x768xf32>, %bfc1_9: tensor<768xf32>, %Wfc2_9: tensor<768x192xf32>, %bfc2_9: tensor<192xf32>, %g1_10: tensor<192xf32>, %b1_10: tensor<192xf32>, %Wq_10: tensor<192x192xf32>, %bq_10: tensor<192xf32>, %Wk_10: tensor<192x192xf32>, %bk_10: tensor<192xf32>, %Wv_10: tensor<192x192xf32>, %bv_10: tensor<192xf32>, %Wo_10: tensor<192x192xf32>, %bo_10: tensor<192xf32>, %g2_10: tensor<192xf32>, %b2_10: tensor<192xf32>, %Wfc1_10: tensor<192x768xf32>, %bfc1_10: tensor<768xf32>, %Wfc2_10: tensor<768x192xf32>, %bfc2_10: tensor<192xf32>, %g1_11: tensor<192xf32>, %b1_11: tensor<192xf32>, %Wq_11: tensor<192x192xf32>, %bq_11: tensor<192xf32>, %Wk_11: tensor<192x192xf32>, %bk_11: tensor<192xf32>, %Wv_11: tensor<192x192xf32>, %bv_11: tensor<192xf32>, %Wo_11: tensor<192x192xf32>, %bo_11: tensor<192xf32>, %g2_11: tensor<192xf32>, %b2_11: tensor<192xf32>, %Wfc1_11: tensor<192x768xf32>, %bfc1_11: tensor<768xf32>, %Wfc2_11: tensor<768x192xf32>, %bfc2_11: tensor<192xf32>, %gF: tensor<192xf32>, %bF: tensor<192xf32>, %Wc: tensor<192x10xf32>, %bc: tensor<10xf32>, %wConvm: tensor<192x3x16x16xf32>, %bConvm: tensor<192xf32>, %clsm: tensor<192xf32>, %posm: tensor<197x192xf32>, %g1_0m: tensor<192xf32>, %b1_0m: tensor<192xf32>, %Wq_0m: tensor<192x192xf32>, %bq_0m: tensor<192xf32>, %Wk_0m: tensor<192x192xf32>, %bk_0m: tensor<192xf32>, %Wv_0m: tensor<192x192xf32>, %bv_0m: tensor<192xf32>, %Wo_0m: tensor<192x192xf32>, %bo_0m: tensor<192xf32>, %g2_0m: tensor<192xf32>, %b2_0m: tensor<192xf32>, %Wfc1_0m: tensor<192x768xf32>, %bfc1_0m: tensor<768xf32>, %Wfc2_0m: tensor<768x192xf32>, %bfc2_0m: tensor<192xf32>, %g1_1m: tensor<192xf32>, %b1_1m: tensor<192xf32>, %Wq_1m: tensor<192x192xf32>, %bq_1m: tensor<192xf32>, %Wk_1m: tensor<192x192xf32>, %bk_1m: tensor<192xf32>, %Wv_1m: tensor<192x192xf32>, %bv_1m: tensor<192xf32>, %Wo_1m: tensor<192x192xf32>, %bo_1m: tensor<192xf32>, %g2_1m: tensor<192xf32>, %b2_1m: tensor<192xf32>, %Wfc1_1m: tensor<192x768xf32>, %bfc1_1m: tensor<768xf32>, %Wfc2_1m: tensor<768x192xf32>, %bfc2_1m: tensor<192xf32>, %g1_2m: tensor<192xf32>, %b1_2m: tensor<192xf32>, %Wq_2m: tensor<192x192xf32>, %bq_2m: tensor<192xf32>, %Wk_2m: tensor<192x192xf32>, %bk_2m: tensor<192xf32>, %Wv_2m: tensor<192x192xf32>, %bv_2m: tensor<192xf32>, %Wo_2m: tensor<192x192xf32>, %bo_2m: tensor<192xf32>, %g2_2m: tensor<192xf32>, %b2_2m: tensor<192xf32>, %Wfc1_2m: tensor<192x768xf32>, %bfc1_2m: tensor<768xf32>, %Wfc2_2m: tensor<768x192xf32>, %bfc2_2m: tensor<192xf32>, %g1_3m: tensor<192xf32>, %b1_3m: tensor<192xf32>, %Wq_3m: tensor<192x192xf32>, %bq_3m: tensor<192xf32>, %Wk_3m: tensor<192x192xf32>, %bk_3m: tensor<192xf32>, %Wv_3m: tensor<192x192xf32>, %bv_3m: tensor<192xf32>, %Wo_3m: tensor<192x192xf32>, %bo_3m: tensor<192xf32>, %g2_3m: tensor<192xf32>, %b2_3m: tensor<192xf32>, %Wfc1_3m: tensor<192x768xf32>, %bfc1_3m: tensor<768xf32>, %Wfc2_3m: tensor<768x192xf32>, %bfc2_3m: tensor<192xf32>, %g1_4m: tensor<192xf32>, %b1_4m: tensor<192xf32>, %Wq_4m: tensor<192x192xf32>, %bq_4m: tensor<192xf32>, %Wk_4m: tensor<192x192xf32>, %bk_4m: tensor<192xf32>, %Wv_4m: tensor<192x192xf32>, %bv_4m: tensor<192xf32>, %Wo_4m: tensor<192x192xf32>, %bo_4m: tensor<192xf32>, %g2_4m: tensor<192xf32>, %b2_4m: tensor<192xf32>, %Wfc1_4m: tensor<192x768xf32>, %bfc1_4m: tensor<768xf32>, %Wfc2_4m: tensor<768x192xf32>, %bfc2_4m: tensor<192xf32>, %g1_5m: tensor<192xf32>, %b1_5m: tensor<192xf32>, %Wq_5m: tensor<192x192xf32>, %bq_5m: tensor<192xf32>, %Wk_5m: tensor<192x192xf32>, %bk_5m: tensor<192xf32>, %Wv_5m: tensor<192x192xf32>, %bv_5m: tensor<192xf32>, %Wo_5m: tensor<192x192xf32>, %bo_5m: tensor<192xf32>, %g2_5m: tensor<192xf32>, %b2_5m: tensor<192xf32>, %Wfc1_5m: tensor<192x768xf32>, %bfc1_5m: tensor<768xf32>, %Wfc2_5m: tensor<768x192xf32>, %bfc2_5m: tensor<192xf32>, %g1_6m: tensor<192xf32>, %b1_6m: tensor<192xf32>, %Wq_6m: tensor<192x192xf32>, %bq_6m: tensor<192xf32>, %Wk_6m: tensor<192x192xf32>, %bk_6m: tensor<192xf32>, %Wv_6m: tensor<192x192xf32>, %bv_6m: tensor<192xf32>, %Wo_6m: tensor<192x192xf32>, %bo_6m: tensor<192xf32>, %g2_6m: tensor<192xf32>, %b2_6m: tensor<192xf32>, %Wfc1_6m: tensor<192x768xf32>, %bfc1_6m: tensor<768xf32>, %Wfc2_6m: tensor<768x192xf32>, %bfc2_6m: tensor<192xf32>, %g1_7m: tensor<192xf32>, %b1_7m: tensor<192xf32>, %Wq_7m: tensor<192x192xf32>, %bq_7m: tensor<192xf32>, %Wk_7m: tensor<192x192xf32>, %bk_7m: tensor<192xf32>, %Wv_7m: tensor<192x192xf32>, %bv_7m: tensor<192xf32>, %Wo_7m: tensor<192x192xf32>, %bo_7m: tensor<192xf32>, %g2_7m: tensor<192xf32>, %b2_7m: tensor<192xf32>, %Wfc1_7m: tensor<192x768xf32>, %bfc1_7m: tensor<768xf32>, %Wfc2_7m: tensor<768x192xf32>, %bfc2_7m: tensor<192xf32>, %g1_8m: tensor<192xf32>, %b1_8m: tensor<192xf32>, %Wq_8m: tensor<192x192xf32>, %bq_8m: tensor<192xf32>, %Wk_8m: tensor<192x192xf32>, %bk_8m: tensor<192xf32>, %Wv_8m: tensor<192x192xf32>, %bv_8m: tensor<192xf32>, %Wo_8m: tensor<192x192xf32>, %bo_8m: tensor<192xf32>, %g2_8m: tensor<192xf32>, %b2_8m: tensor<192xf32>, %Wfc1_8m: tensor<192x768xf32>, %bfc1_8m: tensor<768xf32>, %Wfc2_8m: tensor<768x192xf32>, %bfc2_8m: tensor<192xf32>, %g1_9m: tensor<192xf32>, %b1_9m: tensor<192xf32>, %Wq_9m: tensor<192x192xf32>, %bq_9m: tensor<192xf32>, %Wk_9m: tensor<192x192xf32>, %bk_9m: tensor<192xf32>, %Wv_9m: tensor<192x192xf32>, %bv_9m: tensor<192xf32>, %Wo_9m: tensor<192x192xf32>, %bo_9m: tensor<192xf32>, %g2_9m: tensor<192xf32>, %b2_9m: tensor<192xf32>, %Wfc1_9m: tensor<192x768xf32>, %bfc1_9m: tensor<768xf32>, %Wfc2_9m: tensor<768x192xf32>, %bfc2_9m: tensor<192xf32>, %g1_10m: tensor<192xf32>, %b1_10m: tensor<192xf32>, %Wq_10m: tensor<192x192xf32>, %bq_10m: tensor<192xf32>, %Wk_10m: tensor<192x192xf32>, %bk_10m: tensor<192xf32>, %Wv_10m: tensor<192x192xf32>, %bv_10m: tensor<192xf32>, %Wo_10m: tensor<192x192xf32>, %bo_10m: tensor<192xf32>, %g2_10m: tensor<192xf32>, %b2_10m: tensor<192xf32>, %Wfc1_10m: tensor<192x768xf32>, %bfc1_10m: tensor<768xf32>, %Wfc2_10m: tensor<768x192xf32>, %bfc2_10m: tensor<192xf32>, %g1_11m: tensor<192xf32>, %b1_11m: tensor<192xf32>, %Wq_11m: tensor<192x192xf32>, %bq_11m: tensor<192xf32>, %Wk_11m: tensor<192x192xf32>, %bk_11m: tensor<192xf32>, %Wv_11m: tensor<192x192xf32>, %bv_11m: tensor<192xf32>, %Wo_11m: tensor<192x192xf32>, %bo_11m: tensor<192xf32>, %g2_11m: tensor<192xf32>, %b2_11m: tensor<192xf32>, %Wfc1_11m: tensor<192x768xf32>, %bfc1_11m: tensor<768xf32>, %Wfc2_11m: tensor<768x192xf32>, %bfc2_11m: tensor<192xf32>, %gFm: tensor<192xf32>, %bFm: tensor<192xf32>, %Wcm: tensor<192x10xf32>, %bcm: tensor<10xf32>, %wConvv: tensor<192x3x16x16xf32>, %bConvv: tensor<192xf32>, %clsv: tensor<192xf32>, %posv: tensor<197x192xf32>, %g1_0v: tensor<192xf32>, %b1_0v: tensor<192xf32>, %Wq_0v: tensor<192x192xf32>, %bq_0v: tensor<192xf32>, %Wk_0v: tensor<192x192xf32>, %bk_0v: tensor<192xf32>, %Wv_0v: tensor<192x192xf32>, %bv_0v: tensor<192xf32>, %Wo_0v: tensor<192x192xf32>, %bo_0v: tensor<192xf32>, %g2_0v: tensor<192xf32>, %b2_0v: tensor<192xf32>, %Wfc1_0v: tensor<192x768xf32>, %bfc1_0v: tensor<768xf32>, %Wfc2_0v: tensor<768x192xf32>, %bfc2_0v: tensor<192xf32>, %g1_1v: tensor<192xf32>, %b1_1v: tensor<192xf32>, %Wq_1v: tensor<192x192xf32>, %bq_1v: tensor<192xf32>, %Wk_1v: tensor<192x192xf32>, %bk_1v: tensor<192xf32>, %Wv_1v: tensor<192x192xf32>, %bv_1v: tensor<192xf32>, %Wo_1v: tensor<192x192xf32>, %bo_1v: tensor<192xf32>, %g2_1v: tensor<192xf32>, %b2_1v: tensor<192xf32>, %Wfc1_1v: tensor<192x768xf32>, %bfc1_1v: tensor<768xf32>, %Wfc2_1v: tensor<768x192xf32>, %bfc2_1v: tensor<192xf32>, %g1_2v: tensor<192xf32>, %b1_2v: tensor<192xf32>, %Wq_2v: tensor<192x192xf32>, %bq_2v: tensor<192xf32>, %Wk_2v: tensor<192x192xf32>, %bk_2v: tensor<192xf32>, %Wv_2v: tensor<192x192xf32>, %bv_2v: tensor<192xf32>, %Wo_2v: tensor<192x192xf32>, %bo_2v: tensor<192xf32>, %g2_2v: tensor<192xf32>, %b2_2v: tensor<192xf32>, %Wfc1_2v: tensor<192x768xf32>, %bfc1_2v: tensor<768xf32>, %Wfc2_2v: tensor<768x192xf32>, %bfc2_2v: tensor<192xf32>, %g1_3v: tensor<192xf32>, %b1_3v: tensor<192xf32>, %Wq_3v: tensor<192x192xf32>, %bq_3v: tensor<192xf32>, %Wk_3v: tensor<192x192xf32>, %bk_3v: tensor<192xf32>, %Wv_3v: tensor<192x192xf32>, %bv_3v: tensor<192xf32>, %Wo_3v: tensor<192x192xf32>, %bo_3v: tensor<192xf32>, %g2_3v: tensor<192xf32>, %b2_3v: tensor<192xf32>, %Wfc1_3v: tensor<192x768xf32>, %bfc1_3v: tensor<768xf32>, %Wfc2_3v: tensor<768x192xf32>, %bfc2_3v: tensor<192xf32>, %g1_4v: tensor<192xf32>, %b1_4v: tensor<192xf32>, %Wq_4v: tensor<192x192xf32>, %bq_4v: tensor<192xf32>, %Wk_4v: tensor<192x192xf32>, %bk_4v: tensor<192xf32>, %Wv_4v: tensor<192x192xf32>, %bv_4v: tensor<192xf32>, %Wo_4v: tensor<192x192xf32>, %bo_4v: tensor<192xf32>, %g2_4v: tensor<192xf32>, %b2_4v: tensor<192xf32>, %Wfc1_4v: tensor<192x768xf32>, %bfc1_4v: tensor<768xf32>, %Wfc2_4v: tensor<768x192xf32>, %bfc2_4v: tensor<192xf32>, %g1_5v: tensor<192xf32>, %b1_5v: tensor<192xf32>, %Wq_5v: tensor<192x192xf32>, %bq_5v: tensor<192xf32>, %Wk_5v: tensor<192x192xf32>, %bk_5v: tensor<192xf32>, %Wv_5v: tensor<192x192xf32>, %bv_5v: tensor<192xf32>, %Wo_5v: tensor<192x192xf32>, %bo_5v: tensor<192xf32>, %g2_5v: tensor<192xf32>, %b2_5v: tensor<192xf32>, %Wfc1_5v: tensor<192x768xf32>, %bfc1_5v: tensor<768xf32>, %Wfc2_5v: tensor<768x192xf32>, %bfc2_5v: tensor<192xf32>, %g1_6v: tensor<192xf32>, %b1_6v: tensor<192xf32>, %Wq_6v: tensor<192x192xf32>, %bq_6v: tensor<192xf32>, %Wk_6v: tensor<192x192xf32>, %bk_6v: tensor<192xf32>, %Wv_6v: tensor<192x192xf32>, %bv_6v: tensor<192xf32>, %Wo_6v: tensor<192x192xf32>, %bo_6v: tensor<192xf32>, %g2_6v: tensor<192xf32>, %b2_6v: tensor<192xf32>, %Wfc1_6v: tensor<192x768xf32>, %bfc1_6v: tensor<768xf32>, %Wfc2_6v: tensor<768x192xf32>, %bfc2_6v: tensor<192xf32>, %g1_7v: tensor<192xf32>, %b1_7v: tensor<192xf32>, %Wq_7v: tensor<192x192xf32>, %bq_7v: tensor<192xf32>, %Wk_7v: tensor<192x192xf32>, %bk_7v: tensor<192xf32>, %Wv_7v: tensor<192x192xf32>, %bv_7v: tensor<192xf32>, %Wo_7v: tensor<192x192xf32>, %bo_7v: tensor<192xf32>, %g2_7v: tensor<192xf32>, %b2_7v: tensor<192xf32>, %Wfc1_7v: tensor<192x768xf32>, %bfc1_7v: tensor<768xf32>, %Wfc2_7v: tensor<768x192xf32>, %bfc2_7v: tensor<192xf32>, %g1_8v: tensor<192xf32>, %b1_8v: tensor<192xf32>, %Wq_8v: tensor<192x192xf32>, %bq_8v: tensor<192xf32>, %Wk_8v: tensor<192x192xf32>, %bk_8v: tensor<192xf32>, %Wv_8v: tensor<192x192xf32>, %bv_8v: tensor<192xf32>, %Wo_8v: tensor<192x192xf32>, %bo_8v: tensor<192xf32>, %g2_8v: tensor<192xf32>, %b2_8v: tensor<192xf32>, %Wfc1_8v: tensor<192x768xf32>, %bfc1_8v: tensor<768xf32>, %Wfc2_8v: tensor<768x192xf32>, %bfc2_8v: tensor<192xf32>, %g1_9v: tensor<192xf32>, %b1_9v: tensor<192xf32>, %Wq_9v: tensor<192x192xf32>, %bq_9v: tensor<192xf32>, %Wk_9v: tensor<192x192xf32>, %bk_9v: tensor<192xf32>, %Wv_9v: tensor<192x192xf32>, %bv_9v: tensor<192xf32>, %Wo_9v: tensor<192x192xf32>, %bo_9v: tensor<192xf32>, %g2_9v: tensor<192xf32>, %b2_9v: tensor<192xf32>, %Wfc1_9v: tensor<192x768xf32>, %bfc1_9v: tensor<768xf32>, %Wfc2_9v: tensor<768x192xf32>, %bfc2_9v: tensor<192xf32>, %g1_10v: tensor<192xf32>, %b1_10v: tensor<192xf32>, %Wq_10v: tensor<192x192xf32>, %bq_10v: tensor<192xf32>, %Wk_10v: tensor<192x192xf32>, %bk_10v: tensor<192xf32>, %Wv_10v: tensor<192x192xf32>, %bv_10v: tensor<192xf32>, %Wo_10v: tensor<192x192xf32>, %bo_10v: tensor<192xf32>, %g2_10v: tensor<192xf32>, %b2_10v: tensor<192xf32>, %Wfc1_10v: tensor<192x768xf32>, %bfc1_10v: tensor<768xf32>, %Wfc2_10v: tensor<768x192xf32>, %bfc2_10v: tensor<192xf32>, %g1_11v: tensor<192xf32>, %b1_11v: tensor<192xf32>, %Wq_11v: tensor<192x192xf32>, %bq_11v: tensor<192xf32>, %Wk_11v: tensor<192x192xf32>, %bk_11v: tensor<192xf32>, %Wv_11v: tensor<192x192xf32>, %bv_11v: tensor<192xf32>, %Wo_11v: tensor<192x192xf32>, %bo_11v: tensor<192xf32>, %g2_11v: tensor<192xf32>, %b2_11v: tensor<192xf32>, %Wfc1_11v: tensor<192x768xf32>, %bfc1_11v: tensor<768xf32>, %Wfc2_11v: tensor<768x192xf32>, %bfc2_11v: tensor<192xf32>, %gFv: tensor<192xf32>, %bFv: tensor<192xf32>, %Wcv: tensor<192x10xf32>, %bcv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<32x10xf32>) -> (tensor<192x3x16x16xf32>, tensor<192xf32>, tensor<192xf32>, tensor<197x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x10xf32>, tensor<10xf32>, tensor<192x3x16x16xf32>, tensor<192xf32>, tensor<192xf32>, tensor<197x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x10xf32>, tensor<10xf32>, tensor<192x3x16x16xf32>, tensor<192xf32>, tensor<192xf32>, tensor<197x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %xr = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %vitpec = stablehlo.convolution(%xr, %wConv)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [16, 16], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<192x3x16x16xf32>) -> tensor<32x192x14x14xf32>
    %vitpecbb = stablehlo.broadcast_in_dim %bConv, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %vitpepe = stablehlo.add %vitpec, %vitpecbb : tensor<32x192x14x14xf32>
    %vitpept = stablehlo.transpose %vitpepe, dims = [0, 2, 3, 1] : (tensor<32x192x14x14xf32>) -> tensor<32x14x14x192xf32>
    %vitpetok = stablehlo.reshape %vitpept : (tensor<32x14x14x192xf32>) -> tensor<32x196x192xf32>
    %vitcpclsb = stablehlo.broadcast_in_dim %cls, dims = [2] : (tensor<192xf32>) -> tensor<32x1x192xf32>
    %vitcpcat = stablehlo.concatenate %vitcpclsb, %vitpetok, dim = 1 : (tensor<32x1x192xf32>, tensor<32x196x192xf32>) -> tensor<32x197x192xf32>
    %vitcpposb = stablehlo.broadcast_in_dim %pos, dims = [1, 2] : (tensor<197x192xf32>) -> tensor<32x197x192xf32>
    %vitcpz = stablehlo.add %vitcpcat, %vitcpposb : tensor<32x197x192xf32>
    %vitb0_1sum = stablehlo.reduce(%vitcpz init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb0_1mu = stablehlo.divide %vitb0_1sum, %vitb0_1nf : tensor<32x197xf32>
    %vitb0_1mub = stablehlo.broadcast_in_dim %vitb0_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_1xc = stablehlo.subtract %vitcpz, %vitb0_1mub : tensor<32x197x192xf32>
    %vitb0_1sq = stablehlo.multiply %vitb0_1xc, %vitb0_1xc : tensor<32x197x192xf32>
    %vitb0_1vsum = stablehlo.reduce(%vitb0_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_1var = stablehlo.divide %vitb0_1vsum, %vitb0_1nf : tensor<32x197xf32>
    %vitb0_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb0_1ve = stablehlo.add %vitb0_1var, %vitb0_1eps : tensor<32x197xf32>
    %vitb0_1istd = stablehlo.rsqrt %vitb0_1ve : tensor<32x197xf32>
    %vitb0_1istdb = stablehlo.broadcast_in_dim %vitb0_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_1xhat = stablehlo.multiply %vitb0_1xc, %vitb0_1istdb : tensor<32x197x192xf32>
    %vitb0_1gb = stablehlo.broadcast_in_dim %g1_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_1bbc = stablehlo.broadcast_in_dim %b1_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_1gx = stablehlo.multiply %vitb0_1xhat, %vitb0_1gb : tensor<32x197x192xf32>
    %vitb0_1y = stablehlo.add %vitb0_1gx, %vitb0_1bbc : tensor<32x197x192xf32>
    %vitb0_mQd = stablehlo.dot_general %vitb0_1y, %Wq_0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mQbb = stablehlo.broadcast_in_dim %bq_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mQ = stablehlo.add %vitb0_mQd, %vitb0_mQbb : tensor<32x197x192xf32>
    %vitb0_mKd = stablehlo.dot_general %vitb0_1y, %Wk_0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mKbb = stablehlo.broadcast_in_dim %bk_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mK = stablehlo.add %vitb0_mKd, %vitb0_mKbb : tensor<32x197x192xf32>
    %vitb0_mVd = stablehlo.dot_general %vitb0_1y, %Wv_0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mVbb = stablehlo.broadcast_in_dim %bv_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mV = stablehlo.add %vitb0_mVd, %vitb0_mVbb : tensor<32x197x192xf32>
    %vitb0_mQhr = stablehlo.reshape %vitb0_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mQh = stablehlo.transpose %vitb0_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mKhr = stablehlo.reshape %vitb0_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mKh = stablehlo.transpose %vitb0_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mVhr = stablehlo.reshape %vitb0_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mVh = stablehlo.transpose %vitb0_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mS = stablehlo.dot_general %vitb0_mQh, %vitb0_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb0_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb0_mSs = stablehlo.multiply %vitb0_mS, %vitb0_mscl : tensor<32x3x197x197xf32>
    %vitb0_mse = stablehlo.exponential %vitb0_mSs : tensor<32x3x197x197xf32>
    %vitb0_msum = stablehlo.reduce(%vitb0_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb0_msumb = stablehlo.broadcast_in_dim %vitb0_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb0_mW = stablehlo.divide %vitb0_mse, %vitb0_msumb : tensor<32x3x197x197xf32>
    %vitb0_mA = stablehlo.dot_general %vitb0_mW, %vitb0_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mAT = stablehlo.transpose %vitb0_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mP = stablehlo.reshape %vitb0_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb0_mod = stablehlo.dot_general %vitb0_mP, %Wo_0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mobb = stablehlo.broadcast_in_dim %bo_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mO = stablehlo.add %vitb0_mod, %vitb0_mobb : tensor<32x197x192xf32>
    %vitb0_r1 = stablehlo.add %vitcpz, %vitb0_mO : tensor<32x197x192xf32>
    %vitb0_2sum = stablehlo.reduce(%vitb0_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb0_2mu = stablehlo.divide %vitb0_2sum, %vitb0_2nf : tensor<32x197xf32>
    %vitb0_2mub = stablehlo.broadcast_in_dim %vitb0_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_2xc = stablehlo.subtract %vitb0_r1, %vitb0_2mub : tensor<32x197x192xf32>
    %vitb0_2sq = stablehlo.multiply %vitb0_2xc, %vitb0_2xc : tensor<32x197x192xf32>
    %vitb0_2vsum = stablehlo.reduce(%vitb0_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_2var = stablehlo.divide %vitb0_2vsum, %vitb0_2nf : tensor<32x197xf32>
    %vitb0_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb0_2ve = stablehlo.add %vitb0_2var, %vitb0_2eps : tensor<32x197xf32>
    %vitb0_2istd = stablehlo.rsqrt %vitb0_2ve : tensor<32x197xf32>
    %vitb0_2istdb = stablehlo.broadcast_in_dim %vitb0_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_2xhat = stablehlo.multiply %vitb0_2xc, %vitb0_2istdb : tensor<32x197x192xf32>
    %vitb0_2gb = stablehlo.broadcast_in_dim %g2_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_2bbc = stablehlo.broadcast_in_dim %b2_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_2gx = stablehlo.multiply %vitb0_2xhat, %vitb0_2gb : tensor<32x197x192xf32>
    %vitb0_2y = stablehlo.add %vitb0_2gx, %vitb0_2bbc : tensor<32x197x192xf32>
    %vitb0_ph1d = stablehlo.dot_general %vitb0_2y, %Wfc1_0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb0_ph1bb = stablehlo.broadcast_in_dim %bfc1_0, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb0_ph1 = stablehlo.add %vitb0_ph1d, %vitb0_ph1bb : tensor<32x197x768xf32>
    %vitb0_pgx2 = stablehlo.multiply %vitb0_ph1, %vitb0_ph1 : tensor<32x197x768xf32>
    %vitb0_pgx3 = stablehlo.multiply %vitb0_pgx2, %vitb0_ph1 : tensor<32x197x768xf32>
    %vitb0_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb0_pgkx3 = stablehlo.multiply %vitb0_pgck, %vitb0_pgx3 : tensor<32x197x768xf32>
    %vitb0_pginn = stablehlo.add %vitb0_ph1, %vitb0_pgkx3 : tensor<32x197x768xf32>
    %vitb0_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb0_pgu = stablehlo.multiply %vitb0_pgcsqrt, %vitb0_pginn : tensor<32x197x768xf32>
    %vitb0_pgt = stablehlo.tanh %vitb0_pgu : tensor<32x197x768xf32>
    %vitb0_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb0_pgopt = stablehlo.add %vitb0_pgone, %vitb0_pgt : tensor<32x197x768xf32>
    %vitb0_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb0_pghx = stablehlo.multiply %vitb0_pgchalf, %vitb0_ph1 : tensor<32x197x768xf32>
    %vitb0_pga = stablehlo.multiply %vitb0_pghx, %vitb0_pgopt : tensor<32x197x768xf32>
    %vitb0_py2d = stablehlo.dot_general %vitb0_pga, %Wfc2_0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_py2bb = stablehlo.broadcast_in_dim %bfc2_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_py = stablehlo.add %vitb0_py2d, %vitb0_py2bb : tensor<32x197x192xf32>
    %vitb0_out = stablehlo.add %vitb0_r1, %vitb0_py : tensor<32x197x192xf32>
    %vitb1_1sum = stablehlo.reduce(%vitb0_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb1_1mu = stablehlo.divide %vitb1_1sum, %vitb1_1nf : tensor<32x197xf32>
    %vitb1_1mub = stablehlo.broadcast_in_dim %vitb1_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_1xc = stablehlo.subtract %vitb0_out, %vitb1_1mub : tensor<32x197x192xf32>
    %vitb1_1sq = stablehlo.multiply %vitb1_1xc, %vitb1_1xc : tensor<32x197x192xf32>
    %vitb1_1vsum = stablehlo.reduce(%vitb1_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_1var = stablehlo.divide %vitb1_1vsum, %vitb1_1nf : tensor<32x197xf32>
    %vitb1_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb1_1ve = stablehlo.add %vitb1_1var, %vitb1_1eps : tensor<32x197xf32>
    %vitb1_1istd = stablehlo.rsqrt %vitb1_1ve : tensor<32x197xf32>
    %vitb1_1istdb = stablehlo.broadcast_in_dim %vitb1_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_1xhat = stablehlo.multiply %vitb1_1xc, %vitb1_1istdb : tensor<32x197x192xf32>
    %vitb1_1gb = stablehlo.broadcast_in_dim %g1_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_1bbc = stablehlo.broadcast_in_dim %b1_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_1gx = stablehlo.multiply %vitb1_1xhat, %vitb1_1gb : tensor<32x197x192xf32>
    %vitb1_1y = stablehlo.add %vitb1_1gx, %vitb1_1bbc : tensor<32x197x192xf32>
    %vitb1_mQd = stablehlo.dot_general %vitb1_1y, %Wq_1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mQbb = stablehlo.broadcast_in_dim %bq_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mQ = stablehlo.add %vitb1_mQd, %vitb1_mQbb : tensor<32x197x192xf32>
    %vitb1_mKd = stablehlo.dot_general %vitb1_1y, %Wk_1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mKbb = stablehlo.broadcast_in_dim %bk_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mK = stablehlo.add %vitb1_mKd, %vitb1_mKbb : tensor<32x197x192xf32>
    %vitb1_mVd = stablehlo.dot_general %vitb1_1y, %Wv_1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mVbb = stablehlo.broadcast_in_dim %bv_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mV = stablehlo.add %vitb1_mVd, %vitb1_mVbb : tensor<32x197x192xf32>
    %vitb1_mQhr = stablehlo.reshape %vitb1_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mQh = stablehlo.transpose %vitb1_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mKhr = stablehlo.reshape %vitb1_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mKh = stablehlo.transpose %vitb1_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mVhr = stablehlo.reshape %vitb1_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mVh = stablehlo.transpose %vitb1_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mS = stablehlo.dot_general %vitb1_mQh, %vitb1_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb1_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb1_mSs = stablehlo.multiply %vitb1_mS, %vitb1_mscl : tensor<32x3x197x197xf32>
    %vitb1_mse = stablehlo.exponential %vitb1_mSs : tensor<32x3x197x197xf32>
    %vitb1_msum = stablehlo.reduce(%vitb1_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb1_msumb = stablehlo.broadcast_in_dim %vitb1_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb1_mW = stablehlo.divide %vitb1_mse, %vitb1_msumb : tensor<32x3x197x197xf32>
    %vitb1_mA = stablehlo.dot_general %vitb1_mW, %vitb1_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mAT = stablehlo.transpose %vitb1_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mP = stablehlo.reshape %vitb1_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb1_mod = stablehlo.dot_general %vitb1_mP, %Wo_1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mobb = stablehlo.broadcast_in_dim %bo_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mO = stablehlo.add %vitb1_mod, %vitb1_mobb : tensor<32x197x192xf32>
    %vitb1_r1 = stablehlo.add %vitb0_out, %vitb1_mO : tensor<32x197x192xf32>
    %vitb1_2sum = stablehlo.reduce(%vitb1_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb1_2mu = stablehlo.divide %vitb1_2sum, %vitb1_2nf : tensor<32x197xf32>
    %vitb1_2mub = stablehlo.broadcast_in_dim %vitb1_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_2xc = stablehlo.subtract %vitb1_r1, %vitb1_2mub : tensor<32x197x192xf32>
    %vitb1_2sq = stablehlo.multiply %vitb1_2xc, %vitb1_2xc : tensor<32x197x192xf32>
    %vitb1_2vsum = stablehlo.reduce(%vitb1_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_2var = stablehlo.divide %vitb1_2vsum, %vitb1_2nf : tensor<32x197xf32>
    %vitb1_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb1_2ve = stablehlo.add %vitb1_2var, %vitb1_2eps : tensor<32x197xf32>
    %vitb1_2istd = stablehlo.rsqrt %vitb1_2ve : tensor<32x197xf32>
    %vitb1_2istdb = stablehlo.broadcast_in_dim %vitb1_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_2xhat = stablehlo.multiply %vitb1_2xc, %vitb1_2istdb : tensor<32x197x192xf32>
    %vitb1_2gb = stablehlo.broadcast_in_dim %g2_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_2bbc = stablehlo.broadcast_in_dim %b2_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_2gx = stablehlo.multiply %vitb1_2xhat, %vitb1_2gb : tensor<32x197x192xf32>
    %vitb1_2y = stablehlo.add %vitb1_2gx, %vitb1_2bbc : tensor<32x197x192xf32>
    %vitb1_ph1d = stablehlo.dot_general %vitb1_2y, %Wfc1_1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb1_ph1bb = stablehlo.broadcast_in_dim %bfc1_1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb1_ph1 = stablehlo.add %vitb1_ph1d, %vitb1_ph1bb : tensor<32x197x768xf32>
    %vitb1_pgx2 = stablehlo.multiply %vitb1_ph1, %vitb1_ph1 : tensor<32x197x768xf32>
    %vitb1_pgx3 = stablehlo.multiply %vitb1_pgx2, %vitb1_ph1 : tensor<32x197x768xf32>
    %vitb1_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb1_pgkx3 = stablehlo.multiply %vitb1_pgck, %vitb1_pgx3 : tensor<32x197x768xf32>
    %vitb1_pginn = stablehlo.add %vitb1_ph1, %vitb1_pgkx3 : tensor<32x197x768xf32>
    %vitb1_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb1_pgu = stablehlo.multiply %vitb1_pgcsqrt, %vitb1_pginn : tensor<32x197x768xf32>
    %vitb1_pgt = stablehlo.tanh %vitb1_pgu : tensor<32x197x768xf32>
    %vitb1_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb1_pgopt = stablehlo.add %vitb1_pgone, %vitb1_pgt : tensor<32x197x768xf32>
    %vitb1_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb1_pghx = stablehlo.multiply %vitb1_pgchalf, %vitb1_ph1 : tensor<32x197x768xf32>
    %vitb1_pga = stablehlo.multiply %vitb1_pghx, %vitb1_pgopt : tensor<32x197x768xf32>
    %vitb1_py2d = stablehlo.dot_general %vitb1_pga, %Wfc2_1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_py2bb = stablehlo.broadcast_in_dim %bfc2_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_py = stablehlo.add %vitb1_py2d, %vitb1_py2bb : tensor<32x197x192xf32>
    %vitb1_out = stablehlo.add %vitb1_r1, %vitb1_py : tensor<32x197x192xf32>
    %vitb2_1sum = stablehlo.reduce(%vitb1_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb2_1mu = stablehlo.divide %vitb2_1sum, %vitb2_1nf : tensor<32x197xf32>
    %vitb2_1mub = stablehlo.broadcast_in_dim %vitb2_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_1xc = stablehlo.subtract %vitb1_out, %vitb2_1mub : tensor<32x197x192xf32>
    %vitb2_1sq = stablehlo.multiply %vitb2_1xc, %vitb2_1xc : tensor<32x197x192xf32>
    %vitb2_1vsum = stablehlo.reduce(%vitb2_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_1var = stablehlo.divide %vitb2_1vsum, %vitb2_1nf : tensor<32x197xf32>
    %vitb2_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb2_1ve = stablehlo.add %vitb2_1var, %vitb2_1eps : tensor<32x197xf32>
    %vitb2_1istd = stablehlo.rsqrt %vitb2_1ve : tensor<32x197xf32>
    %vitb2_1istdb = stablehlo.broadcast_in_dim %vitb2_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_1xhat = stablehlo.multiply %vitb2_1xc, %vitb2_1istdb : tensor<32x197x192xf32>
    %vitb2_1gb = stablehlo.broadcast_in_dim %g1_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_1bbc = stablehlo.broadcast_in_dim %b1_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_1gx = stablehlo.multiply %vitb2_1xhat, %vitb2_1gb : tensor<32x197x192xf32>
    %vitb2_1y = stablehlo.add %vitb2_1gx, %vitb2_1bbc : tensor<32x197x192xf32>
    %vitb2_mQd = stablehlo.dot_general %vitb2_1y, %Wq_2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mQbb = stablehlo.broadcast_in_dim %bq_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mQ = stablehlo.add %vitb2_mQd, %vitb2_mQbb : tensor<32x197x192xf32>
    %vitb2_mKd = stablehlo.dot_general %vitb2_1y, %Wk_2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mKbb = stablehlo.broadcast_in_dim %bk_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mK = stablehlo.add %vitb2_mKd, %vitb2_mKbb : tensor<32x197x192xf32>
    %vitb2_mVd = stablehlo.dot_general %vitb2_1y, %Wv_2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mVbb = stablehlo.broadcast_in_dim %bv_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mV = stablehlo.add %vitb2_mVd, %vitb2_mVbb : tensor<32x197x192xf32>
    %vitb2_mQhr = stablehlo.reshape %vitb2_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mQh = stablehlo.transpose %vitb2_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mKhr = stablehlo.reshape %vitb2_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mKh = stablehlo.transpose %vitb2_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mVhr = stablehlo.reshape %vitb2_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mVh = stablehlo.transpose %vitb2_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mS = stablehlo.dot_general %vitb2_mQh, %vitb2_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb2_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb2_mSs = stablehlo.multiply %vitb2_mS, %vitb2_mscl : tensor<32x3x197x197xf32>
    %vitb2_mse = stablehlo.exponential %vitb2_mSs : tensor<32x3x197x197xf32>
    %vitb2_msum = stablehlo.reduce(%vitb2_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb2_msumb = stablehlo.broadcast_in_dim %vitb2_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb2_mW = stablehlo.divide %vitb2_mse, %vitb2_msumb : tensor<32x3x197x197xf32>
    %vitb2_mA = stablehlo.dot_general %vitb2_mW, %vitb2_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mAT = stablehlo.transpose %vitb2_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mP = stablehlo.reshape %vitb2_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb2_mod = stablehlo.dot_general %vitb2_mP, %Wo_2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mobb = stablehlo.broadcast_in_dim %bo_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mO = stablehlo.add %vitb2_mod, %vitb2_mobb : tensor<32x197x192xf32>
    %vitb2_r1 = stablehlo.add %vitb1_out, %vitb2_mO : tensor<32x197x192xf32>
    %vitb2_2sum = stablehlo.reduce(%vitb2_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb2_2mu = stablehlo.divide %vitb2_2sum, %vitb2_2nf : tensor<32x197xf32>
    %vitb2_2mub = stablehlo.broadcast_in_dim %vitb2_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_2xc = stablehlo.subtract %vitb2_r1, %vitb2_2mub : tensor<32x197x192xf32>
    %vitb2_2sq = stablehlo.multiply %vitb2_2xc, %vitb2_2xc : tensor<32x197x192xf32>
    %vitb2_2vsum = stablehlo.reduce(%vitb2_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_2var = stablehlo.divide %vitb2_2vsum, %vitb2_2nf : tensor<32x197xf32>
    %vitb2_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb2_2ve = stablehlo.add %vitb2_2var, %vitb2_2eps : tensor<32x197xf32>
    %vitb2_2istd = stablehlo.rsqrt %vitb2_2ve : tensor<32x197xf32>
    %vitb2_2istdb = stablehlo.broadcast_in_dim %vitb2_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_2xhat = stablehlo.multiply %vitb2_2xc, %vitb2_2istdb : tensor<32x197x192xf32>
    %vitb2_2gb = stablehlo.broadcast_in_dim %g2_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_2bbc = stablehlo.broadcast_in_dim %b2_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_2gx = stablehlo.multiply %vitb2_2xhat, %vitb2_2gb : tensor<32x197x192xf32>
    %vitb2_2y = stablehlo.add %vitb2_2gx, %vitb2_2bbc : tensor<32x197x192xf32>
    %vitb2_ph1d = stablehlo.dot_general %vitb2_2y, %Wfc1_2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb2_ph1bb = stablehlo.broadcast_in_dim %bfc1_2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb2_ph1 = stablehlo.add %vitb2_ph1d, %vitb2_ph1bb : tensor<32x197x768xf32>
    %vitb2_pgx2 = stablehlo.multiply %vitb2_ph1, %vitb2_ph1 : tensor<32x197x768xf32>
    %vitb2_pgx3 = stablehlo.multiply %vitb2_pgx2, %vitb2_ph1 : tensor<32x197x768xf32>
    %vitb2_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb2_pgkx3 = stablehlo.multiply %vitb2_pgck, %vitb2_pgx3 : tensor<32x197x768xf32>
    %vitb2_pginn = stablehlo.add %vitb2_ph1, %vitb2_pgkx3 : tensor<32x197x768xf32>
    %vitb2_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb2_pgu = stablehlo.multiply %vitb2_pgcsqrt, %vitb2_pginn : tensor<32x197x768xf32>
    %vitb2_pgt = stablehlo.tanh %vitb2_pgu : tensor<32x197x768xf32>
    %vitb2_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb2_pgopt = stablehlo.add %vitb2_pgone, %vitb2_pgt : tensor<32x197x768xf32>
    %vitb2_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb2_pghx = stablehlo.multiply %vitb2_pgchalf, %vitb2_ph1 : tensor<32x197x768xf32>
    %vitb2_pga = stablehlo.multiply %vitb2_pghx, %vitb2_pgopt : tensor<32x197x768xf32>
    %vitb2_py2d = stablehlo.dot_general %vitb2_pga, %Wfc2_2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_py2bb = stablehlo.broadcast_in_dim %bfc2_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_py = stablehlo.add %vitb2_py2d, %vitb2_py2bb : tensor<32x197x192xf32>
    %vitb2_out = stablehlo.add %vitb2_r1, %vitb2_py : tensor<32x197x192xf32>
    %vitb3_1sum = stablehlo.reduce(%vitb2_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb3_1mu = stablehlo.divide %vitb3_1sum, %vitb3_1nf : tensor<32x197xf32>
    %vitb3_1mub = stablehlo.broadcast_in_dim %vitb3_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_1xc = stablehlo.subtract %vitb2_out, %vitb3_1mub : tensor<32x197x192xf32>
    %vitb3_1sq = stablehlo.multiply %vitb3_1xc, %vitb3_1xc : tensor<32x197x192xf32>
    %vitb3_1vsum = stablehlo.reduce(%vitb3_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_1var = stablehlo.divide %vitb3_1vsum, %vitb3_1nf : tensor<32x197xf32>
    %vitb3_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb3_1ve = stablehlo.add %vitb3_1var, %vitb3_1eps : tensor<32x197xf32>
    %vitb3_1istd = stablehlo.rsqrt %vitb3_1ve : tensor<32x197xf32>
    %vitb3_1istdb = stablehlo.broadcast_in_dim %vitb3_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_1xhat = stablehlo.multiply %vitb3_1xc, %vitb3_1istdb : tensor<32x197x192xf32>
    %vitb3_1gb = stablehlo.broadcast_in_dim %g1_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_1bbc = stablehlo.broadcast_in_dim %b1_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_1gx = stablehlo.multiply %vitb3_1xhat, %vitb3_1gb : tensor<32x197x192xf32>
    %vitb3_1y = stablehlo.add %vitb3_1gx, %vitb3_1bbc : tensor<32x197x192xf32>
    %vitb3_mQd = stablehlo.dot_general %vitb3_1y, %Wq_3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mQbb = stablehlo.broadcast_in_dim %bq_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mQ = stablehlo.add %vitb3_mQd, %vitb3_mQbb : tensor<32x197x192xf32>
    %vitb3_mKd = stablehlo.dot_general %vitb3_1y, %Wk_3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mKbb = stablehlo.broadcast_in_dim %bk_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mK = stablehlo.add %vitb3_mKd, %vitb3_mKbb : tensor<32x197x192xf32>
    %vitb3_mVd = stablehlo.dot_general %vitb3_1y, %Wv_3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mVbb = stablehlo.broadcast_in_dim %bv_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mV = stablehlo.add %vitb3_mVd, %vitb3_mVbb : tensor<32x197x192xf32>
    %vitb3_mQhr = stablehlo.reshape %vitb3_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mQh = stablehlo.transpose %vitb3_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mKhr = stablehlo.reshape %vitb3_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mKh = stablehlo.transpose %vitb3_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mVhr = stablehlo.reshape %vitb3_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mVh = stablehlo.transpose %vitb3_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mS = stablehlo.dot_general %vitb3_mQh, %vitb3_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb3_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb3_mSs = stablehlo.multiply %vitb3_mS, %vitb3_mscl : tensor<32x3x197x197xf32>
    %vitb3_mse = stablehlo.exponential %vitb3_mSs : tensor<32x3x197x197xf32>
    %vitb3_msum = stablehlo.reduce(%vitb3_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb3_msumb = stablehlo.broadcast_in_dim %vitb3_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb3_mW = stablehlo.divide %vitb3_mse, %vitb3_msumb : tensor<32x3x197x197xf32>
    %vitb3_mA = stablehlo.dot_general %vitb3_mW, %vitb3_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mAT = stablehlo.transpose %vitb3_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mP = stablehlo.reshape %vitb3_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb3_mod = stablehlo.dot_general %vitb3_mP, %Wo_3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mobb = stablehlo.broadcast_in_dim %bo_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mO = stablehlo.add %vitb3_mod, %vitb3_mobb : tensor<32x197x192xf32>
    %vitb3_r1 = stablehlo.add %vitb2_out, %vitb3_mO : tensor<32x197x192xf32>
    %vitb3_2sum = stablehlo.reduce(%vitb3_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb3_2mu = stablehlo.divide %vitb3_2sum, %vitb3_2nf : tensor<32x197xf32>
    %vitb3_2mub = stablehlo.broadcast_in_dim %vitb3_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_2xc = stablehlo.subtract %vitb3_r1, %vitb3_2mub : tensor<32x197x192xf32>
    %vitb3_2sq = stablehlo.multiply %vitb3_2xc, %vitb3_2xc : tensor<32x197x192xf32>
    %vitb3_2vsum = stablehlo.reduce(%vitb3_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_2var = stablehlo.divide %vitb3_2vsum, %vitb3_2nf : tensor<32x197xf32>
    %vitb3_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb3_2ve = stablehlo.add %vitb3_2var, %vitb3_2eps : tensor<32x197xf32>
    %vitb3_2istd = stablehlo.rsqrt %vitb3_2ve : tensor<32x197xf32>
    %vitb3_2istdb = stablehlo.broadcast_in_dim %vitb3_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_2xhat = stablehlo.multiply %vitb3_2xc, %vitb3_2istdb : tensor<32x197x192xf32>
    %vitb3_2gb = stablehlo.broadcast_in_dim %g2_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_2bbc = stablehlo.broadcast_in_dim %b2_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_2gx = stablehlo.multiply %vitb3_2xhat, %vitb3_2gb : tensor<32x197x192xf32>
    %vitb3_2y = stablehlo.add %vitb3_2gx, %vitb3_2bbc : tensor<32x197x192xf32>
    %vitb3_ph1d = stablehlo.dot_general %vitb3_2y, %Wfc1_3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb3_ph1bb = stablehlo.broadcast_in_dim %bfc1_3, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb3_ph1 = stablehlo.add %vitb3_ph1d, %vitb3_ph1bb : tensor<32x197x768xf32>
    %vitb3_pgx2 = stablehlo.multiply %vitb3_ph1, %vitb3_ph1 : tensor<32x197x768xf32>
    %vitb3_pgx3 = stablehlo.multiply %vitb3_pgx2, %vitb3_ph1 : tensor<32x197x768xf32>
    %vitb3_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb3_pgkx3 = stablehlo.multiply %vitb3_pgck, %vitb3_pgx3 : tensor<32x197x768xf32>
    %vitb3_pginn = stablehlo.add %vitb3_ph1, %vitb3_pgkx3 : tensor<32x197x768xf32>
    %vitb3_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb3_pgu = stablehlo.multiply %vitb3_pgcsqrt, %vitb3_pginn : tensor<32x197x768xf32>
    %vitb3_pgt = stablehlo.tanh %vitb3_pgu : tensor<32x197x768xf32>
    %vitb3_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb3_pgopt = stablehlo.add %vitb3_pgone, %vitb3_pgt : tensor<32x197x768xf32>
    %vitb3_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb3_pghx = stablehlo.multiply %vitb3_pgchalf, %vitb3_ph1 : tensor<32x197x768xf32>
    %vitb3_pga = stablehlo.multiply %vitb3_pghx, %vitb3_pgopt : tensor<32x197x768xf32>
    %vitb3_py2d = stablehlo.dot_general %vitb3_pga, %Wfc2_3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_py2bb = stablehlo.broadcast_in_dim %bfc2_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_py = stablehlo.add %vitb3_py2d, %vitb3_py2bb : tensor<32x197x192xf32>
    %vitb3_out = stablehlo.add %vitb3_r1, %vitb3_py : tensor<32x197x192xf32>
    %vitb4_1sum = stablehlo.reduce(%vitb3_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb4_1mu = stablehlo.divide %vitb4_1sum, %vitb4_1nf : tensor<32x197xf32>
    %vitb4_1mub = stablehlo.broadcast_in_dim %vitb4_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_1xc = stablehlo.subtract %vitb3_out, %vitb4_1mub : tensor<32x197x192xf32>
    %vitb4_1sq = stablehlo.multiply %vitb4_1xc, %vitb4_1xc : tensor<32x197x192xf32>
    %vitb4_1vsum = stablehlo.reduce(%vitb4_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_1var = stablehlo.divide %vitb4_1vsum, %vitb4_1nf : tensor<32x197xf32>
    %vitb4_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb4_1ve = stablehlo.add %vitb4_1var, %vitb4_1eps : tensor<32x197xf32>
    %vitb4_1istd = stablehlo.rsqrt %vitb4_1ve : tensor<32x197xf32>
    %vitb4_1istdb = stablehlo.broadcast_in_dim %vitb4_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_1xhat = stablehlo.multiply %vitb4_1xc, %vitb4_1istdb : tensor<32x197x192xf32>
    %vitb4_1gb = stablehlo.broadcast_in_dim %g1_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_1bbc = stablehlo.broadcast_in_dim %b1_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_1gx = stablehlo.multiply %vitb4_1xhat, %vitb4_1gb : tensor<32x197x192xf32>
    %vitb4_1y = stablehlo.add %vitb4_1gx, %vitb4_1bbc : tensor<32x197x192xf32>
    %vitb4_mQd = stablehlo.dot_general %vitb4_1y, %Wq_4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mQbb = stablehlo.broadcast_in_dim %bq_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mQ = stablehlo.add %vitb4_mQd, %vitb4_mQbb : tensor<32x197x192xf32>
    %vitb4_mKd = stablehlo.dot_general %vitb4_1y, %Wk_4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mKbb = stablehlo.broadcast_in_dim %bk_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mK = stablehlo.add %vitb4_mKd, %vitb4_mKbb : tensor<32x197x192xf32>
    %vitb4_mVd = stablehlo.dot_general %vitb4_1y, %Wv_4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mVbb = stablehlo.broadcast_in_dim %bv_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mV = stablehlo.add %vitb4_mVd, %vitb4_mVbb : tensor<32x197x192xf32>
    %vitb4_mQhr = stablehlo.reshape %vitb4_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mQh = stablehlo.transpose %vitb4_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mKhr = stablehlo.reshape %vitb4_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mKh = stablehlo.transpose %vitb4_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mVhr = stablehlo.reshape %vitb4_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mVh = stablehlo.transpose %vitb4_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mS = stablehlo.dot_general %vitb4_mQh, %vitb4_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb4_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb4_mSs = stablehlo.multiply %vitb4_mS, %vitb4_mscl : tensor<32x3x197x197xf32>
    %vitb4_mse = stablehlo.exponential %vitb4_mSs : tensor<32x3x197x197xf32>
    %vitb4_msum = stablehlo.reduce(%vitb4_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb4_msumb = stablehlo.broadcast_in_dim %vitb4_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb4_mW = stablehlo.divide %vitb4_mse, %vitb4_msumb : tensor<32x3x197x197xf32>
    %vitb4_mA = stablehlo.dot_general %vitb4_mW, %vitb4_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mAT = stablehlo.transpose %vitb4_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mP = stablehlo.reshape %vitb4_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb4_mod = stablehlo.dot_general %vitb4_mP, %Wo_4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mobb = stablehlo.broadcast_in_dim %bo_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mO = stablehlo.add %vitb4_mod, %vitb4_mobb : tensor<32x197x192xf32>
    %vitb4_r1 = stablehlo.add %vitb3_out, %vitb4_mO : tensor<32x197x192xf32>
    %vitb4_2sum = stablehlo.reduce(%vitb4_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb4_2mu = stablehlo.divide %vitb4_2sum, %vitb4_2nf : tensor<32x197xf32>
    %vitb4_2mub = stablehlo.broadcast_in_dim %vitb4_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_2xc = stablehlo.subtract %vitb4_r1, %vitb4_2mub : tensor<32x197x192xf32>
    %vitb4_2sq = stablehlo.multiply %vitb4_2xc, %vitb4_2xc : tensor<32x197x192xf32>
    %vitb4_2vsum = stablehlo.reduce(%vitb4_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_2var = stablehlo.divide %vitb4_2vsum, %vitb4_2nf : tensor<32x197xf32>
    %vitb4_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb4_2ve = stablehlo.add %vitb4_2var, %vitb4_2eps : tensor<32x197xf32>
    %vitb4_2istd = stablehlo.rsqrt %vitb4_2ve : tensor<32x197xf32>
    %vitb4_2istdb = stablehlo.broadcast_in_dim %vitb4_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_2xhat = stablehlo.multiply %vitb4_2xc, %vitb4_2istdb : tensor<32x197x192xf32>
    %vitb4_2gb = stablehlo.broadcast_in_dim %g2_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_2bbc = stablehlo.broadcast_in_dim %b2_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_2gx = stablehlo.multiply %vitb4_2xhat, %vitb4_2gb : tensor<32x197x192xf32>
    %vitb4_2y = stablehlo.add %vitb4_2gx, %vitb4_2bbc : tensor<32x197x192xf32>
    %vitb4_ph1d = stablehlo.dot_general %vitb4_2y, %Wfc1_4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb4_ph1bb = stablehlo.broadcast_in_dim %bfc1_4, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb4_ph1 = stablehlo.add %vitb4_ph1d, %vitb4_ph1bb : tensor<32x197x768xf32>
    %vitb4_pgx2 = stablehlo.multiply %vitb4_ph1, %vitb4_ph1 : tensor<32x197x768xf32>
    %vitb4_pgx3 = stablehlo.multiply %vitb4_pgx2, %vitb4_ph1 : tensor<32x197x768xf32>
    %vitb4_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb4_pgkx3 = stablehlo.multiply %vitb4_pgck, %vitb4_pgx3 : tensor<32x197x768xf32>
    %vitb4_pginn = stablehlo.add %vitb4_ph1, %vitb4_pgkx3 : tensor<32x197x768xf32>
    %vitb4_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb4_pgu = stablehlo.multiply %vitb4_pgcsqrt, %vitb4_pginn : tensor<32x197x768xf32>
    %vitb4_pgt = stablehlo.tanh %vitb4_pgu : tensor<32x197x768xf32>
    %vitb4_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb4_pgopt = stablehlo.add %vitb4_pgone, %vitb4_pgt : tensor<32x197x768xf32>
    %vitb4_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb4_pghx = stablehlo.multiply %vitb4_pgchalf, %vitb4_ph1 : tensor<32x197x768xf32>
    %vitb4_pga = stablehlo.multiply %vitb4_pghx, %vitb4_pgopt : tensor<32x197x768xf32>
    %vitb4_py2d = stablehlo.dot_general %vitb4_pga, %Wfc2_4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_py2bb = stablehlo.broadcast_in_dim %bfc2_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_py = stablehlo.add %vitb4_py2d, %vitb4_py2bb : tensor<32x197x192xf32>
    %vitb4_out = stablehlo.add %vitb4_r1, %vitb4_py : tensor<32x197x192xf32>
    %vitb5_1sum = stablehlo.reduce(%vitb4_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb5_1mu = stablehlo.divide %vitb5_1sum, %vitb5_1nf : tensor<32x197xf32>
    %vitb5_1mub = stablehlo.broadcast_in_dim %vitb5_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_1xc = stablehlo.subtract %vitb4_out, %vitb5_1mub : tensor<32x197x192xf32>
    %vitb5_1sq = stablehlo.multiply %vitb5_1xc, %vitb5_1xc : tensor<32x197x192xf32>
    %vitb5_1vsum = stablehlo.reduce(%vitb5_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_1var = stablehlo.divide %vitb5_1vsum, %vitb5_1nf : tensor<32x197xf32>
    %vitb5_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb5_1ve = stablehlo.add %vitb5_1var, %vitb5_1eps : tensor<32x197xf32>
    %vitb5_1istd = stablehlo.rsqrt %vitb5_1ve : tensor<32x197xf32>
    %vitb5_1istdb = stablehlo.broadcast_in_dim %vitb5_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_1xhat = stablehlo.multiply %vitb5_1xc, %vitb5_1istdb : tensor<32x197x192xf32>
    %vitb5_1gb = stablehlo.broadcast_in_dim %g1_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_1bbc = stablehlo.broadcast_in_dim %b1_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_1gx = stablehlo.multiply %vitb5_1xhat, %vitb5_1gb : tensor<32x197x192xf32>
    %vitb5_1y = stablehlo.add %vitb5_1gx, %vitb5_1bbc : tensor<32x197x192xf32>
    %vitb5_mQd = stablehlo.dot_general %vitb5_1y, %Wq_5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mQbb = stablehlo.broadcast_in_dim %bq_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mQ = stablehlo.add %vitb5_mQd, %vitb5_mQbb : tensor<32x197x192xf32>
    %vitb5_mKd = stablehlo.dot_general %vitb5_1y, %Wk_5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mKbb = stablehlo.broadcast_in_dim %bk_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mK = stablehlo.add %vitb5_mKd, %vitb5_mKbb : tensor<32x197x192xf32>
    %vitb5_mVd = stablehlo.dot_general %vitb5_1y, %Wv_5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mVbb = stablehlo.broadcast_in_dim %bv_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mV = stablehlo.add %vitb5_mVd, %vitb5_mVbb : tensor<32x197x192xf32>
    %vitb5_mQhr = stablehlo.reshape %vitb5_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mQh = stablehlo.transpose %vitb5_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mKhr = stablehlo.reshape %vitb5_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mKh = stablehlo.transpose %vitb5_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mVhr = stablehlo.reshape %vitb5_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mVh = stablehlo.transpose %vitb5_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mS = stablehlo.dot_general %vitb5_mQh, %vitb5_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb5_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb5_mSs = stablehlo.multiply %vitb5_mS, %vitb5_mscl : tensor<32x3x197x197xf32>
    %vitb5_mse = stablehlo.exponential %vitb5_mSs : tensor<32x3x197x197xf32>
    %vitb5_msum = stablehlo.reduce(%vitb5_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb5_msumb = stablehlo.broadcast_in_dim %vitb5_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb5_mW = stablehlo.divide %vitb5_mse, %vitb5_msumb : tensor<32x3x197x197xf32>
    %vitb5_mA = stablehlo.dot_general %vitb5_mW, %vitb5_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mAT = stablehlo.transpose %vitb5_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mP = stablehlo.reshape %vitb5_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb5_mod = stablehlo.dot_general %vitb5_mP, %Wo_5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mobb = stablehlo.broadcast_in_dim %bo_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mO = stablehlo.add %vitb5_mod, %vitb5_mobb : tensor<32x197x192xf32>
    %vitb5_r1 = stablehlo.add %vitb4_out, %vitb5_mO : tensor<32x197x192xf32>
    %vitb5_2sum = stablehlo.reduce(%vitb5_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb5_2mu = stablehlo.divide %vitb5_2sum, %vitb5_2nf : tensor<32x197xf32>
    %vitb5_2mub = stablehlo.broadcast_in_dim %vitb5_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_2xc = stablehlo.subtract %vitb5_r1, %vitb5_2mub : tensor<32x197x192xf32>
    %vitb5_2sq = stablehlo.multiply %vitb5_2xc, %vitb5_2xc : tensor<32x197x192xf32>
    %vitb5_2vsum = stablehlo.reduce(%vitb5_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_2var = stablehlo.divide %vitb5_2vsum, %vitb5_2nf : tensor<32x197xf32>
    %vitb5_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb5_2ve = stablehlo.add %vitb5_2var, %vitb5_2eps : tensor<32x197xf32>
    %vitb5_2istd = stablehlo.rsqrt %vitb5_2ve : tensor<32x197xf32>
    %vitb5_2istdb = stablehlo.broadcast_in_dim %vitb5_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_2xhat = stablehlo.multiply %vitb5_2xc, %vitb5_2istdb : tensor<32x197x192xf32>
    %vitb5_2gb = stablehlo.broadcast_in_dim %g2_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_2bbc = stablehlo.broadcast_in_dim %b2_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_2gx = stablehlo.multiply %vitb5_2xhat, %vitb5_2gb : tensor<32x197x192xf32>
    %vitb5_2y = stablehlo.add %vitb5_2gx, %vitb5_2bbc : tensor<32x197x192xf32>
    %vitb5_ph1d = stablehlo.dot_general %vitb5_2y, %Wfc1_5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb5_ph1bb = stablehlo.broadcast_in_dim %bfc1_5, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb5_ph1 = stablehlo.add %vitb5_ph1d, %vitb5_ph1bb : tensor<32x197x768xf32>
    %vitb5_pgx2 = stablehlo.multiply %vitb5_ph1, %vitb5_ph1 : tensor<32x197x768xf32>
    %vitb5_pgx3 = stablehlo.multiply %vitb5_pgx2, %vitb5_ph1 : tensor<32x197x768xf32>
    %vitb5_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb5_pgkx3 = stablehlo.multiply %vitb5_pgck, %vitb5_pgx3 : tensor<32x197x768xf32>
    %vitb5_pginn = stablehlo.add %vitb5_ph1, %vitb5_pgkx3 : tensor<32x197x768xf32>
    %vitb5_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb5_pgu = stablehlo.multiply %vitb5_pgcsqrt, %vitb5_pginn : tensor<32x197x768xf32>
    %vitb5_pgt = stablehlo.tanh %vitb5_pgu : tensor<32x197x768xf32>
    %vitb5_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb5_pgopt = stablehlo.add %vitb5_pgone, %vitb5_pgt : tensor<32x197x768xf32>
    %vitb5_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb5_pghx = stablehlo.multiply %vitb5_pgchalf, %vitb5_ph1 : tensor<32x197x768xf32>
    %vitb5_pga = stablehlo.multiply %vitb5_pghx, %vitb5_pgopt : tensor<32x197x768xf32>
    %vitb5_py2d = stablehlo.dot_general %vitb5_pga, %Wfc2_5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_py2bb = stablehlo.broadcast_in_dim %bfc2_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_py = stablehlo.add %vitb5_py2d, %vitb5_py2bb : tensor<32x197x192xf32>
    %vitb5_out = stablehlo.add %vitb5_r1, %vitb5_py : tensor<32x197x192xf32>
    %vitb6_1sum = stablehlo.reduce(%vitb5_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb6_1mu = stablehlo.divide %vitb6_1sum, %vitb6_1nf : tensor<32x197xf32>
    %vitb6_1mub = stablehlo.broadcast_in_dim %vitb6_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_1xc = stablehlo.subtract %vitb5_out, %vitb6_1mub : tensor<32x197x192xf32>
    %vitb6_1sq = stablehlo.multiply %vitb6_1xc, %vitb6_1xc : tensor<32x197x192xf32>
    %vitb6_1vsum = stablehlo.reduce(%vitb6_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_1var = stablehlo.divide %vitb6_1vsum, %vitb6_1nf : tensor<32x197xf32>
    %vitb6_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb6_1ve = stablehlo.add %vitb6_1var, %vitb6_1eps : tensor<32x197xf32>
    %vitb6_1istd = stablehlo.rsqrt %vitb6_1ve : tensor<32x197xf32>
    %vitb6_1istdb = stablehlo.broadcast_in_dim %vitb6_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_1xhat = stablehlo.multiply %vitb6_1xc, %vitb6_1istdb : tensor<32x197x192xf32>
    %vitb6_1gb = stablehlo.broadcast_in_dim %g1_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_1bbc = stablehlo.broadcast_in_dim %b1_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_1gx = stablehlo.multiply %vitb6_1xhat, %vitb6_1gb : tensor<32x197x192xf32>
    %vitb6_1y = stablehlo.add %vitb6_1gx, %vitb6_1bbc : tensor<32x197x192xf32>
    %vitb6_mQd = stablehlo.dot_general %vitb6_1y, %Wq_6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mQbb = stablehlo.broadcast_in_dim %bq_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mQ = stablehlo.add %vitb6_mQd, %vitb6_mQbb : tensor<32x197x192xf32>
    %vitb6_mKd = stablehlo.dot_general %vitb6_1y, %Wk_6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mKbb = stablehlo.broadcast_in_dim %bk_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mK = stablehlo.add %vitb6_mKd, %vitb6_mKbb : tensor<32x197x192xf32>
    %vitb6_mVd = stablehlo.dot_general %vitb6_1y, %Wv_6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mVbb = stablehlo.broadcast_in_dim %bv_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mV = stablehlo.add %vitb6_mVd, %vitb6_mVbb : tensor<32x197x192xf32>
    %vitb6_mQhr = stablehlo.reshape %vitb6_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mQh = stablehlo.transpose %vitb6_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mKhr = stablehlo.reshape %vitb6_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mKh = stablehlo.transpose %vitb6_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mVhr = stablehlo.reshape %vitb6_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mVh = stablehlo.transpose %vitb6_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mS = stablehlo.dot_general %vitb6_mQh, %vitb6_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb6_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb6_mSs = stablehlo.multiply %vitb6_mS, %vitb6_mscl : tensor<32x3x197x197xf32>
    %vitb6_mse = stablehlo.exponential %vitb6_mSs : tensor<32x3x197x197xf32>
    %vitb6_msum = stablehlo.reduce(%vitb6_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb6_msumb = stablehlo.broadcast_in_dim %vitb6_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb6_mW = stablehlo.divide %vitb6_mse, %vitb6_msumb : tensor<32x3x197x197xf32>
    %vitb6_mA = stablehlo.dot_general %vitb6_mW, %vitb6_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mAT = stablehlo.transpose %vitb6_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mP = stablehlo.reshape %vitb6_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb6_mod = stablehlo.dot_general %vitb6_mP, %Wo_6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mobb = stablehlo.broadcast_in_dim %bo_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mO = stablehlo.add %vitb6_mod, %vitb6_mobb : tensor<32x197x192xf32>
    %vitb6_r1 = stablehlo.add %vitb5_out, %vitb6_mO : tensor<32x197x192xf32>
    %vitb6_2sum = stablehlo.reduce(%vitb6_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb6_2mu = stablehlo.divide %vitb6_2sum, %vitb6_2nf : tensor<32x197xf32>
    %vitb6_2mub = stablehlo.broadcast_in_dim %vitb6_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_2xc = stablehlo.subtract %vitb6_r1, %vitb6_2mub : tensor<32x197x192xf32>
    %vitb6_2sq = stablehlo.multiply %vitb6_2xc, %vitb6_2xc : tensor<32x197x192xf32>
    %vitb6_2vsum = stablehlo.reduce(%vitb6_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_2var = stablehlo.divide %vitb6_2vsum, %vitb6_2nf : tensor<32x197xf32>
    %vitb6_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb6_2ve = stablehlo.add %vitb6_2var, %vitb6_2eps : tensor<32x197xf32>
    %vitb6_2istd = stablehlo.rsqrt %vitb6_2ve : tensor<32x197xf32>
    %vitb6_2istdb = stablehlo.broadcast_in_dim %vitb6_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_2xhat = stablehlo.multiply %vitb6_2xc, %vitb6_2istdb : tensor<32x197x192xf32>
    %vitb6_2gb = stablehlo.broadcast_in_dim %g2_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_2bbc = stablehlo.broadcast_in_dim %b2_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_2gx = stablehlo.multiply %vitb6_2xhat, %vitb6_2gb : tensor<32x197x192xf32>
    %vitb6_2y = stablehlo.add %vitb6_2gx, %vitb6_2bbc : tensor<32x197x192xf32>
    %vitb6_ph1d = stablehlo.dot_general %vitb6_2y, %Wfc1_6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb6_ph1bb = stablehlo.broadcast_in_dim %bfc1_6, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb6_ph1 = stablehlo.add %vitb6_ph1d, %vitb6_ph1bb : tensor<32x197x768xf32>
    %vitb6_pgx2 = stablehlo.multiply %vitb6_ph1, %vitb6_ph1 : tensor<32x197x768xf32>
    %vitb6_pgx3 = stablehlo.multiply %vitb6_pgx2, %vitb6_ph1 : tensor<32x197x768xf32>
    %vitb6_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb6_pgkx3 = stablehlo.multiply %vitb6_pgck, %vitb6_pgx3 : tensor<32x197x768xf32>
    %vitb6_pginn = stablehlo.add %vitb6_ph1, %vitb6_pgkx3 : tensor<32x197x768xf32>
    %vitb6_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb6_pgu = stablehlo.multiply %vitb6_pgcsqrt, %vitb6_pginn : tensor<32x197x768xf32>
    %vitb6_pgt = stablehlo.tanh %vitb6_pgu : tensor<32x197x768xf32>
    %vitb6_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb6_pgopt = stablehlo.add %vitb6_pgone, %vitb6_pgt : tensor<32x197x768xf32>
    %vitb6_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb6_pghx = stablehlo.multiply %vitb6_pgchalf, %vitb6_ph1 : tensor<32x197x768xf32>
    %vitb6_pga = stablehlo.multiply %vitb6_pghx, %vitb6_pgopt : tensor<32x197x768xf32>
    %vitb6_py2d = stablehlo.dot_general %vitb6_pga, %Wfc2_6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_py2bb = stablehlo.broadcast_in_dim %bfc2_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_py = stablehlo.add %vitb6_py2d, %vitb6_py2bb : tensor<32x197x192xf32>
    %vitb6_out = stablehlo.add %vitb6_r1, %vitb6_py : tensor<32x197x192xf32>
    %vitb7_1sum = stablehlo.reduce(%vitb6_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb7_1mu = stablehlo.divide %vitb7_1sum, %vitb7_1nf : tensor<32x197xf32>
    %vitb7_1mub = stablehlo.broadcast_in_dim %vitb7_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_1xc = stablehlo.subtract %vitb6_out, %vitb7_1mub : tensor<32x197x192xf32>
    %vitb7_1sq = stablehlo.multiply %vitb7_1xc, %vitb7_1xc : tensor<32x197x192xf32>
    %vitb7_1vsum = stablehlo.reduce(%vitb7_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_1var = stablehlo.divide %vitb7_1vsum, %vitb7_1nf : tensor<32x197xf32>
    %vitb7_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb7_1ve = stablehlo.add %vitb7_1var, %vitb7_1eps : tensor<32x197xf32>
    %vitb7_1istd = stablehlo.rsqrt %vitb7_1ve : tensor<32x197xf32>
    %vitb7_1istdb = stablehlo.broadcast_in_dim %vitb7_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_1xhat = stablehlo.multiply %vitb7_1xc, %vitb7_1istdb : tensor<32x197x192xf32>
    %vitb7_1gb = stablehlo.broadcast_in_dim %g1_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_1bbc = stablehlo.broadcast_in_dim %b1_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_1gx = stablehlo.multiply %vitb7_1xhat, %vitb7_1gb : tensor<32x197x192xf32>
    %vitb7_1y = stablehlo.add %vitb7_1gx, %vitb7_1bbc : tensor<32x197x192xf32>
    %vitb7_mQd = stablehlo.dot_general %vitb7_1y, %Wq_7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mQbb = stablehlo.broadcast_in_dim %bq_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mQ = stablehlo.add %vitb7_mQd, %vitb7_mQbb : tensor<32x197x192xf32>
    %vitb7_mKd = stablehlo.dot_general %vitb7_1y, %Wk_7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mKbb = stablehlo.broadcast_in_dim %bk_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mK = stablehlo.add %vitb7_mKd, %vitb7_mKbb : tensor<32x197x192xf32>
    %vitb7_mVd = stablehlo.dot_general %vitb7_1y, %Wv_7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mVbb = stablehlo.broadcast_in_dim %bv_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mV = stablehlo.add %vitb7_mVd, %vitb7_mVbb : tensor<32x197x192xf32>
    %vitb7_mQhr = stablehlo.reshape %vitb7_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mQh = stablehlo.transpose %vitb7_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mKhr = stablehlo.reshape %vitb7_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mKh = stablehlo.transpose %vitb7_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mVhr = stablehlo.reshape %vitb7_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mVh = stablehlo.transpose %vitb7_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mS = stablehlo.dot_general %vitb7_mQh, %vitb7_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb7_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb7_mSs = stablehlo.multiply %vitb7_mS, %vitb7_mscl : tensor<32x3x197x197xf32>
    %vitb7_mse = stablehlo.exponential %vitb7_mSs : tensor<32x3x197x197xf32>
    %vitb7_msum = stablehlo.reduce(%vitb7_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb7_msumb = stablehlo.broadcast_in_dim %vitb7_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb7_mW = stablehlo.divide %vitb7_mse, %vitb7_msumb : tensor<32x3x197x197xf32>
    %vitb7_mA = stablehlo.dot_general %vitb7_mW, %vitb7_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mAT = stablehlo.transpose %vitb7_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mP = stablehlo.reshape %vitb7_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb7_mod = stablehlo.dot_general %vitb7_mP, %Wo_7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mobb = stablehlo.broadcast_in_dim %bo_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mO = stablehlo.add %vitb7_mod, %vitb7_mobb : tensor<32x197x192xf32>
    %vitb7_r1 = stablehlo.add %vitb6_out, %vitb7_mO : tensor<32x197x192xf32>
    %vitb7_2sum = stablehlo.reduce(%vitb7_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb7_2mu = stablehlo.divide %vitb7_2sum, %vitb7_2nf : tensor<32x197xf32>
    %vitb7_2mub = stablehlo.broadcast_in_dim %vitb7_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_2xc = stablehlo.subtract %vitb7_r1, %vitb7_2mub : tensor<32x197x192xf32>
    %vitb7_2sq = stablehlo.multiply %vitb7_2xc, %vitb7_2xc : tensor<32x197x192xf32>
    %vitb7_2vsum = stablehlo.reduce(%vitb7_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_2var = stablehlo.divide %vitb7_2vsum, %vitb7_2nf : tensor<32x197xf32>
    %vitb7_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb7_2ve = stablehlo.add %vitb7_2var, %vitb7_2eps : tensor<32x197xf32>
    %vitb7_2istd = stablehlo.rsqrt %vitb7_2ve : tensor<32x197xf32>
    %vitb7_2istdb = stablehlo.broadcast_in_dim %vitb7_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_2xhat = stablehlo.multiply %vitb7_2xc, %vitb7_2istdb : tensor<32x197x192xf32>
    %vitb7_2gb = stablehlo.broadcast_in_dim %g2_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_2bbc = stablehlo.broadcast_in_dim %b2_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_2gx = stablehlo.multiply %vitb7_2xhat, %vitb7_2gb : tensor<32x197x192xf32>
    %vitb7_2y = stablehlo.add %vitb7_2gx, %vitb7_2bbc : tensor<32x197x192xf32>
    %vitb7_ph1d = stablehlo.dot_general %vitb7_2y, %Wfc1_7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb7_ph1bb = stablehlo.broadcast_in_dim %bfc1_7, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb7_ph1 = stablehlo.add %vitb7_ph1d, %vitb7_ph1bb : tensor<32x197x768xf32>
    %vitb7_pgx2 = stablehlo.multiply %vitb7_ph1, %vitb7_ph1 : tensor<32x197x768xf32>
    %vitb7_pgx3 = stablehlo.multiply %vitb7_pgx2, %vitb7_ph1 : tensor<32x197x768xf32>
    %vitb7_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb7_pgkx3 = stablehlo.multiply %vitb7_pgck, %vitb7_pgx3 : tensor<32x197x768xf32>
    %vitb7_pginn = stablehlo.add %vitb7_ph1, %vitb7_pgkx3 : tensor<32x197x768xf32>
    %vitb7_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb7_pgu = stablehlo.multiply %vitb7_pgcsqrt, %vitb7_pginn : tensor<32x197x768xf32>
    %vitb7_pgt = stablehlo.tanh %vitb7_pgu : tensor<32x197x768xf32>
    %vitb7_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb7_pgopt = stablehlo.add %vitb7_pgone, %vitb7_pgt : tensor<32x197x768xf32>
    %vitb7_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb7_pghx = stablehlo.multiply %vitb7_pgchalf, %vitb7_ph1 : tensor<32x197x768xf32>
    %vitb7_pga = stablehlo.multiply %vitb7_pghx, %vitb7_pgopt : tensor<32x197x768xf32>
    %vitb7_py2d = stablehlo.dot_general %vitb7_pga, %Wfc2_7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_py2bb = stablehlo.broadcast_in_dim %bfc2_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_py = stablehlo.add %vitb7_py2d, %vitb7_py2bb : tensor<32x197x192xf32>
    %vitb7_out = stablehlo.add %vitb7_r1, %vitb7_py : tensor<32x197x192xf32>
    %vitb8_1sum = stablehlo.reduce(%vitb7_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb8_1mu = stablehlo.divide %vitb8_1sum, %vitb8_1nf : tensor<32x197xf32>
    %vitb8_1mub = stablehlo.broadcast_in_dim %vitb8_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_1xc = stablehlo.subtract %vitb7_out, %vitb8_1mub : tensor<32x197x192xf32>
    %vitb8_1sq = stablehlo.multiply %vitb8_1xc, %vitb8_1xc : tensor<32x197x192xf32>
    %vitb8_1vsum = stablehlo.reduce(%vitb8_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_1var = stablehlo.divide %vitb8_1vsum, %vitb8_1nf : tensor<32x197xf32>
    %vitb8_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb8_1ve = stablehlo.add %vitb8_1var, %vitb8_1eps : tensor<32x197xf32>
    %vitb8_1istd = stablehlo.rsqrt %vitb8_1ve : tensor<32x197xf32>
    %vitb8_1istdb = stablehlo.broadcast_in_dim %vitb8_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_1xhat = stablehlo.multiply %vitb8_1xc, %vitb8_1istdb : tensor<32x197x192xf32>
    %vitb8_1gb = stablehlo.broadcast_in_dim %g1_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_1bbc = stablehlo.broadcast_in_dim %b1_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_1gx = stablehlo.multiply %vitb8_1xhat, %vitb8_1gb : tensor<32x197x192xf32>
    %vitb8_1y = stablehlo.add %vitb8_1gx, %vitb8_1bbc : tensor<32x197x192xf32>
    %vitb8_mQd = stablehlo.dot_general %vitb8_1y, %Wq_8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mQbb = stablehlo.broadcast_in_dim %bq_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mQ = stablehlo.add %vitb8_mQd, %vitb8_mQbb : tensor<32x197x192xf32>
    %vitb8_mKd = stablehlo.dot_general %vitb8_1y, %Wk_8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mKbb = stablehlo.broadcast_in_dim %bk_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mK = stablehlo.add %vitb8_mKd, %vitb8_mKbb : tensor<32x197x192xf32>
    %vitb8_mVd = stablehlo.dot_general %vitb8_1y, %Wv_8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mVbb = stablehlo.broadcast_in_dim %bv_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mV = stablehlo.add %vitb8_mVd, %vitb8_mVbb : tensor<32x197x192xf32>
    %vitb8_mQhr = stablehlo.reshape %vitb8_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mQh = stablehlo.transpose %vitb8_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mKhr = stablehlo.reshape %vitb8_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mKh = stablehlo.transpose %vitb8_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mVhr = stablehlo.reshape %vitb8_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mVh = stablehlo.transpose %vitb8_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mS = stablehlo.dot_general %vitb8_mQh, %vitb8_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb8_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb8_mSs = stablehlo.multiply %vitb8_mS, %vitb8_mscl : tensor<32x3x197x197xf32>
    %vitb8_mse = stablehlo.exponential %vitb8_mSs : tensor<32x3x197x197xf32>
    %vitb8_msum = stablehlo.reduce(%vitb8_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb8_msumb = stablehlo.broadcast_in_dim %vitb8_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb8_mW = stablehlo.divide %vitb8_mse, %vitb8_msumb : tensor<32x3x197x197xf32>
    %vitb8_mA = stablehlo.dot_general %vitb8_mW, %vitb8_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mAT = stablehlo.transpose %vitb8_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mP = stablehlo.reshape %vitb8_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb8_mod = stablehlo.dot_general %vitb8_mP, %Wo_8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mobb = stablehlo.broadcast_in_dim %bo_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mO = stablehlo.add %vitb8_mod, %vitb8_mobb : tensor<32x197x192xf32>
    %vitb8_r1 = stablehlo.add %vitb7_out, %vitb8_mO : tensor<32x197x192xf32>
    %vitb8_2sum = stablehlo.reduce(%vitb8_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb8_2mu = stablehlo.divide %vitb8_2sum, %vitb8_2nf : tensor<32x197xf32>
    %vitb8_2mub = stablehlo.broadcast_in_dim %vitb8_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_2xc = stablehlo.subtract %vitb8_r1, %vitb8_2mub : tensor<32x197x192xf32>
    %vitb8_2sq = stablehlo.multiply %vitb8_2xc, %vitb8_2xc : tensor<32x197x192xf32>
    %vitb8_2vsum = stablehlo.reduce(%vitb8_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_2var = stablehlo.divide %vitb8_2vsum, %vitb8_2nf : tensor<32x197xf32>
    %vitb8_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb8_2ve = stablehlo.add %vitb8_2var, %vitb8_2eps : tensor<32x197xf32>
    %vitb8_2istd = stablehlo.rsqrt %vitb8_2ve : tensor<32x197xf32>
    %vitb8_2istdb = stablehlo.broadcast_in_dim %vitb8_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_2xhat = stablehlo.multiply %vitb8_2xc, %vitb8_2istdb : tensor<32x197x192xf32>
    %vitb8_2gb = stablehlo.broadcast_in_dim %g2_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_2bbc = stablehlo.broadcast_in_dim %b2_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_2gx = stablehlo.multiply %vitb8_2xhat, %vitb8_2gb : tensor<32x197x192xf32>
    %vitb8_2y = stablehlo.add %vitb8_2gx, %vitb8_2bbc : tensor<32x197x192xf32>
    %vitb8_ph1d = stablehlo.dot_general %vitb8_2y, %Wfc1_8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb8_ph1bb = stablehlo.broadcast_in_dim %bfc1_8, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb8_ph1 = stablehlo.add %vitb8_ph1d, %vitb8_ph1bb : tensor<32x197x768xf32>
    %vitb8_pgx2 = stablehlo.multiply %vitb8_ph1, %vitb8_ph1 : tensor<32x197x768xf32>
    %vitb8_pgx3 = stablehlo.multiply %vitb8_pgx2, %vitb8_ph1 : tensor<32x197x768xf32>
    %vitb8_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb8_pgkx3 = stablehlo.multiply %vitb8_pgck, %vitb8_pgx3 : tensor<32x197x768xf32>
    %vitb8_pginn = stablehlo.add %vitb8_ph1, %vitb8_pgkx3 : tensor<32x197x768xf32>
    %vitb8_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb8_pgu = stablehlo.multiply %vitb8_pgcsqrt, %vitb8_pginn : tensor<32x197x768xf32>
    %vitb8_pgt = stablehlo.tanh %vitb8_pgu : tensor<32x197x768xf32>
    %vitb8_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb8_pgopt = stablehlo.add %vitb8_pgone, %vitb8_pgt : tensor<32x197x768xf32>
    %vitb8_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb8_pghx = stablehlo.multiply %vitb8_pgchalf, %vitb8_ph1 : tensor<32x197x768xf32>
    %vitb8_pga = stablehlo.multiply %vitb8_pghx, %vitb8_pgopt : tensor<32x197x768xf32>
    %vitb8_py2d = stablehlo.dot_general %vitb8_pga, %Wfc2_8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_py2bb = stablehlo.broadcast_in_dim %bfc2_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_py = stablehlo.add %vitb8_py2d, %vitb8_py2bb : tensor<32x197x192xf32>
    %vitb8_out = stablehlo.add %vitb8_r1, %vitb8_py : tensor<32x197x192xf32>
    %vitb9_1sum = stablehlo.reduce(%vitb8_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb9_1mu = stablehlo.divide %vitb9_1sum, %vitb9_1nf : tensor<32x197xf32>
    %vitb9_1mub = stablehlo.broadcast_in_dim %vitb9_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_1xc = stablehlo.subtract %vitb8_out, %vitb9_1mub : tensor<32x197x192xf32>
    %vitb9_1sq = stablehlo.multiply %vitb9_1xc, %vitb9_1xc : tensor<32x197x192xf32>
    %vitb9_1vsum = stablehlo.reduce(%vitb9_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_1var = stablehlo.divide %vitb9_1vsum, %vitb9_1nf : tensor<32x197xf32>
    %vitb9_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb9_1ve = stablehlo.add %vitb9_1var, %vitb9_1eps : tensor<32x197xf32>
    %vitb9_1istd = stablehlo.rsqrt %vitb9_1ve : tensor<32x197xf32>
    %vitb9_1istdb = stablehlo.broadcast_in_dim %vitb9_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_1xhat = stablehlo.multiply %vitb9_1xc, %vitb9_1istdb : tensor<32x197x192xf32>
    %vitb9_1gb = stablehlo.broadcast_in_dim %g1_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_1bbc = stablehlo.broadcast_in_dim %b1_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_1gx = stablehlo.multiply %vitb9_1xhat, %vitb9_1gb : tensor<32x197x192xf32>
    %vitb9_1y = stablehlo.add %vitb9_1gx, %vitb9_1bbc : tensor<32x197x192xf32>
    %vitb9_mQd = stablehlo.dot_general %vitb9_1y, %Wq_9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mQbb = stablehlo.broadcast_in_dim %bq_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mQ = stablehlo.add %vitb9_mQd, %vitb9_mQbb : tensor<32x197x192xf32>
    %vitb9_mKd = stablehlo.dot_general %vitb9_1y, %Wk_9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mKbb = stablehlo.broadcast_in_dim %bk_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mK = stablehlo.add %vitb9_mKd, %vitb9_mKbb : tensor<32x197x192xf32>
    %vitb9_mVd = stablehlo.dot_general %vitb9_1y, %Wv_9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mVbb = stablehlo.broadcast_in_dim %bv_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mV = stablehlo.add %vitb9_mVd, %vitb9_mVbb : tensor<32x197x192xf32>
    %vitb9_mQhr = stablehlo.reshape %vitb9_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mQh = stablehlo.transpose %vitb9_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mKhr = stablehlo.reshape %vitb9_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mKh = stablehlo.transpose %vitb9_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mVhr = stablehlo.reshape %vitb9_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mVh = stablehlo.transpose %vitb9_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mS = stablehlo.dot_general %vitb9_mQh, %vitb9_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb9_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb9_mSs = stablehlo.multiply %vitb9_mS, %vitb9_mscl : tensor<32x3x197x197xf32>
    %vitb9_mse = stablehlo.exponential %vitb9_mSs : tensor<32x3x197x197xf32>
    %vitb9_msum = stablehlo.reduce(%vitb9_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb9_msumb = stablehlo.broadcast_in_dim %vitb9_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb9_mW = stablehlo.divide %vitb9_mse, %vitb9_msumb : tensor<32x3x197x197xf32>
    %vitb9_mA = stablehlo.dot_general %vitb9_mW, %vitb9_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mAT = stablehlo.transpose %vitb9_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mP = stablehlo.reshape %vitb9_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb9_mod = stablehlo.dot_general %vitb9_mP, %Wo_9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mobb = stablehlo.broadcast_in_dim %bo_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mO = stablehlo.add %vitb9_mod, %vitb9_mobb : tensor<32x197x192xf32>
    %vitb9_r1 = stablehlo.add %vitb8_out, %vitb9_mO : tensor<32x197x192xf32>
    %vitb9_2sum = stablehlo.reduce(%vitb9_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb9_2mu = stablehlo.divide %vitb9_2sum, %vitb9_2nf : tensor<32x197xf32>
    %vitb9_2mub = stablehlo.broadcast_in_dim %vitb9_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_2xc = stablehlo.subtract %vitb9_r1, %vitb9_2mub : tensor<32x197x192xf32>
    %vitb9_2sq = stablehlo.multiply %vitb9_2xc, %vitb9_2xc : tensor<32x197x192xf32>
    %vitb9_2vsum = stablehlo.reduce(%vitb9_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_2var = stablehlo.divide %vitb9_2vsum, %vitb9_2nf : tensor<32x197xf32>
    %vitb9_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb9_2ve = stablehlo.add %vitb9_2var, %vitb9_2eps : tensor<32x197xf32>
    %vitb9_2istd = stablehlo.rsqrt %vitb9_2ve : tensor<32x197xf32>
    %vitb9_2istdb = stablehlo.broadcast_in_dim %vitb9_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_2xhat = stablehlo.multiply %vitb9_2xc, %vitb9_2istdb : tensor<32x197x192xf32>
    %vitb9_2gb = stablehlo.broadcast_in_dim %g2_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_2bbc = stablehlo.broadcast_in_dim %b2_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_2gx = stablehlo.multiply %vitb9_2xhat, %vitb9_2gb : tensor<32x197x192xf32>
    %vitb9_2y = stablehlo.add %vitb9_2gx, %vitb9_2bbc : tensor<32x197x192xf32>
    %vitb9_ph1d = stablehlo.dot_general %vitb9_2y, %Wfc1_9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb9_ph1bb = stablehlo.broadcast_in_dim %bfc1_9, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb9_ph1 = stablehlo.add %vitb9_ph1d, %vitb9_ph1bb : tensor<32x197x768xf32>
    %vitb9_pgx2 = stablehlo.multiply %vitb9_ph1, %vitb9_ph1 : tensor<32x197x768xf32>
    %vitb9_pgx3 = stablehlo.multiply %vitb9_pgx2, %vitb9_ph1 : tensor<32x197x768xf32>
    %vitb9_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb9_pgkx3 = stablehlo.multiply %vitb9_pgck, %vitb9_pgx3 : tensor<32x197x768xf32>
    %vitb9_pginn = stablehlo.add %vitb9_ph1, %vitb9_pgkx3 : tensor<32x197x768xf32>
    %vitb9_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb9_pgu = stablehlo.multiply %vitb9_pgcsqrt, %vitb9_pginn : tensor<32x197x768xf32>
    %vitb9_pgt = stablehlo.tanh %vitb9_pgu : tensor<32x197x768xf32>
    %vitb9_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb9_pgopt = stablehlo.add %vitb9_pgone, %vitb9_pgt : tensor<32x197x768xf32>
    %vitb9_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb9_pghx = stablehlo.multiply %vitb9_pgchalf, %vitb9_ph1 : tensor<32x197x768xf32>
    %vitb9_pga = stablehlo.multiply %vitb9_pghx, %vitb9_pgopt : tensor<32x197x768xf32>
    %vitb9_py2d = stablehlo.dot_general %vitb9_pga, %Wfc2_9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_py2bb = stablehlo.broadcast_in_dim %bfc2_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_py = stablehlo.add %vitb9_py2d, %vitb9_py2bb : tensor<32x197x192xf32>
    %vitb9_out = stablehlo.add %vitb9_r1, %vitb9_py : tensor<32x197x192xf32>
    %vitb10_1sum = stablehlo.reduce(%vitb9_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb10_1mu = stablehlo.divide %vitb10_1sum, %vitb10_1nf : tensor<32x197xf32>
    %vitb10_1mub = stablehlo.broadcast_in_dim %vitb10_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_1xc = stablehlo.subtract %vitb9_out, %vitb10_1mub : tensor<32x197x192xf32>
    %vitb10_1sq = stablehlo.multiply %vitb10_1xc, %vitb10_1xc : tensor<32x197x192xf32>
    %vitb10_1vsum = stablehlo.reduce(%vitb10_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_1var = stablehlo.divide %vitb10_1vsum, %vitb10_1nf : tensor<32x197xf32>
    %vitb10_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb10_1ve = stablehlo.add %vitb10_1var, %vitb10_1eps : tensor<32x197xf32>
    %vitb10_1istd = stablehlo.rsqrt %vitb10_1ve : tensor<32x197xf32>
    %vitb10_1istdb = stablehlo.broadcast_in_dim %vitb10_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_1xhat = stablehlo.multiply %vitb10_1xc, %vitb10_1istdb : tensor<32x197x192xf32>
    %vitb10_1gb = stablehlo.broadcast_in_dim %g1_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_1bbc = stablehlo.broadcast_in_dim %b1_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_1gx = stablehlo.multiply %vitb10_1xhat, %vitb10_1gb : tensor<32x197x192xf32>
    %vitb10_1y = stablehlo.add %vitb10_1gx, %vitb10_1bbc : tensor<32x197x192xf32>
    %vitb10_mQd = stablehlo.dot_general %vitb10_1y, %Wq_10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mQbb = stablehlo.broadcast_in_dim %bq_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mQ = stablehlo.add %vitb10_mQd, %vitb10_mQbb : tensor<32x197x192xf32>
    %vitb10_mKd = stablehlo.dot_general %vitb10_1y, %Wk_10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mKbb = stablehlo.broadcast_in_dim %bk_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mK = stablehlo.add %vitb10_mKd, %vitb10_mKbb : tensor<32x197x192xf32>
    %vitb10_mVd = stablehlo.dot_general %vitb10_1y, %Wv_10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mVbb = stablehlo.broadcast_in_dim %bv_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mV = stablehlo.add %vitb10_mVd, %vitb10_mVbb : tensor<32x197x192xf32>
    %vitb10_mQhr = stablehlo.reshape %vitb10_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mQh = stablehlo.transpose %vitb10_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mKhr = stablehlo.reshape %vitb10_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mKh = stablehlo.transpose %vitb10_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mVhr = stablehlo.reshape %vitb10_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mVh = stablehlo.transpose %vitb10_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mS = stablehlo.dot_general %vitb10_mQh, %vitb10_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb10_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb10_mSs = stablehlo.multiply %vitb10_mS, %vitb10_mscl : tensor<32x3x197x197xf32>
    %vitb10_mse = stablehlo.exponential %vitb10_mSs : tensor<32x3x197x197xf32>
    %vitb10_msum = stablehlo.reduce(%vitb10_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb10_msumb = stablehlo.broadcast_in_dim %vitb10_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb10_mW = stablehlo.divide %vitb10_mse, %vitb10_msumb : tensor<32x3x197x197xf32>
    %vitb10_mA = stablehlo.dot_general %vitb10_mW, %vitb10_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mAT = stablehlo.transpose %vitb10_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mP = stablehlo.reshape %vitb10_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb10_mod = stablehlo.dot_general %vitb10_mP, %Wo_10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mobb = stablehlo.broadcast_in_dim %bo_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mO = stablehlo.add %vitb10_mod, %vitb10_mobb : tensor<32x197x192xf32>
    %vitb10_r1 = stablehlo.add %vitb9_out, %vitb10_mO : tensor<32x197x192xf32>
    %vitb10_2sum = stablehlo.reduce(%vitb10_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb10_2mu = stablehlo.divide %vitb10_2sum, %vitb10_2nf : tensor<32x197xf32>
    %vitb10_2mub = stablehlo.broadcast_in_dim %vitb10_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_2xc = stablehlo.subtract %vitb10_r1, %vitb10_2mub : tensor<32x197x192xf32>
    %vitb10_2sq = stablehlo.multiply %vitb10_2xc, %vitb10_2xc : tensor<32x197x192xf32>
    %vitb10_2vsum = stablehlo.reduce(%vitb10_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_2var = stablehlo.divide %vitb10_2vsum, %vitb10_2nf : tensor<32x197xf32>
    %vitb10_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb10_2ve = stablehlo.add %vitb10_2var, %vitb10_2eps : tensor<32x197xf32>
    %vitb10_2istd = stablehlo.rsqrt %vitb10_2ve : tensor<32x197xf32>
    %vitb10_2istdb = stablehlo.broadcast_in_dim %vitb10_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_2xhat = stablehlo.multiply %vitb10_2xc, %vitb10_2istdb : tensor<32x197x192xf32>
    %vitb10_2gb = stablehlo.broadcast_in_dim %g2_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_2bbc = stablehlo.broadcast_in_dim %b2_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_2gx = stablehlo.multiply %vitb10_2xhat, %vitb10_2gb : tensor<32x197x192xf32>
    %vitb10_2y = stablehlo.add %vitb10_2gx, %vitb10_2bbc : tensor<32x197x192xf32>
    %vitb10_ph1d = stablehlo.dot_general %vitb10_2y, %Wfc1_10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb10_ph1bb = stablehlo.broadcast_in_dim %bfc1_10, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb10_ph1 = stablehlo.add %vitb10_ph1d, %vitb10_ph1bb : tensor<32x197x768xf32>
    %vitb10_pgx2 = stablehlo.multiply %vitb10_ph1, %vitb10_ph1 : tensor<32x197x768xf32>
    %vitb10_pgx3 = stablehlo.multiply %vitb10_pgx2, %vitb10_ph1 : tensor<32x197x768xf32>
    %vitb10_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb10_pgkx3 = stablehlo.multiply %vitb10_pgck, %vitb10_pgx3 : tensor<32x197x768xf32>
    %vitb10_pginn = stablehlo.add %vitb10_ph1, %vitb10_pgkx3 : tensor<32x197x768xf32>
    %vitb10_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb10_pgu = stablehlo.multiply %vitb10_pgcsqrt, %vitb10_pginn : tensor<32x197x768xf32>
    %vitb10_pgt = stablehlo.tanh %vitb10_pgu : tensor<32x197x768xf32>
    %vitb10_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb10_pgopt = stablehlo.add %vitb10_pgone, %vitb10_pgt : tensor<32x197x768xf32>
    %vitb10_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb10_pghx = stablehlo.multiply %vitb10_pgchalf, %vitb10_ph1 : tensor<32x197x768xf32>
    %vitb10_pga = stablehlo.multiply %vitb10_pghx, %vitb10_pgopt : tensor<32x197x768xf32>
    %vitb10_py2d = stablehlo.dot_general %vitb10_pga, %Wfc2_10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_py2bb = stablehlo.broadcast_in_dim %bfc2_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_py = stablehlo.add %vitb10_py2d, %vitb10_py2bb : tensor<32x197x192xf32>
    %vitb10_out = stablehlo.add %vitb10_r1, %vitb10_py : tensor<32x197x192xf32>
    %vitb11_1sum = stablehlo.reduce(%vitb10_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb11_1mu = stablehlo.divide %vitb11_1sum, %vitb11_1nf : tensor<32x197xf32>
    %vitb11_1mub = stablehlo.broadcast_in_dim %vitb11_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_1xc = stablehlo.subtract %vitb10_out, %vitb11_1mub : tensor<32x197x192xf32>
    %vitb11_1sq = stablehlo.multiply %vitb11_1xc, %vitb11_1xc : tensor<32x197x192xf32>
    %vitb11_1vsum = stablehlo.reduce(%vitb11_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_1var = stablehlo.divide %vitb11_1vsum, %vitb11_1nf : tensor<32x197xf32>
    %vitb11_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb11_1ve = stablehlo.add %vitb11_1var, %vitb11_1eps : tensor<32x197xf32>
    %vitb11_1istd = stablehlo.rsqrt %vitb11_1ve : tensor<32x197xf32>
    %vitb11_1istdb = stablehlo.broadcast_in_dim %vitb11_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_1xhat = stablehlo.multiply %vitb11_1xc, %vitb11_1istdb : tensor<32x197x192xf32>
    %vitb11_1gb = stablehlo.broadcast_in_dim %g1_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_1bbc = stablehlo.broadcast_in_dim %b1_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_1gx = stablehlo.multiply %vitb11_1xhat, %vitb11_1gb : tensor<32x197x192xf32>
    %vitb11_1y = stablehlo.add %vitb11_1gx, %vitb11_1bbc : tensor<32x197x192xf32>
    %vitb11_mQd = stablehlo.dot_general %vitb11_1y, %Wq_11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mQbb = stablehlo.broadcast_in_dim %bq_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mQ = stablehlo.add %vitb11_mQd, %vitb11_mQbb : tensor<32x197x192xf32>
    %vitb11_mKd = stablehlo.dot_general %vitb11_1y, %Wk_11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mKbb = stablehlo.broadcast_in_dim %bk_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mK = stablehlo.add %vitb11_mKd, %vitb11_mKbb : tensor<32x197x192xf32>
    %vitb11_mVd = stablehlo.dot_general %vitb11_1y, %Wv_11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mVbb = stablehlo.broadcast_in_dim %bv_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mV = stablehlo.add %vitb11_mVd, %vitb11_mVbb : tensor<32x197x192xf32>
    %vitb11_mQhr = stablehlo.reshape %vitb11_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mQh = stablehlo.transpose %vitb11_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mKhr = stablehlo.reshape %vitb11_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mKh = stablehlo.transpose %vitb11_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mVhr = stablehlo.reshape %vitb11_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mVh = stablehlo.transpose %vitb11_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mS = stablehlo.dot_general %vitb11_mQh, %vitb11_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb11_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb11_mSs = stablehlo.multiply %vitb11_mS, %vitb11_mscl : tensor<32x3x197x197xf32>
    %vitb11_mse = stablehlo.exponential %vitb11_mSs : tensor<32x3x197x197xf32>
    %vitb11_msum = stablehlo.reduce(%vitb11_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb11_msumb = stablehlo.broadcast_in_dim %vitb11_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb11_mW = stablehlo.divide %vitb11_mse, %vitb11_msumb : tensor<32x3x197x197xf32>
    %vitb11_mA = stablehlo.dot_general %vitb11_mW, %vitb11_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mAT = stablehlo.transpose %vitb11_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mP = stablehlo.reshape %vitb11_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb11_mod = stablehlo.dot_general %vitb11_mP, %Wo_11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mobb = stablehlo.broadcast_in_dim %bo_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mO = stablehlo.add %vitb11_mod, %vitb11_mobb : tensor<32x197x192xf32>
    %vitb11_r1 = stablehlo.add %vitb10_out, %vitb11_mO : tensor<32x197x192xf32>
    %vitb11_2sum = stablehlo.reduce(%vitb11_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb11_2mu = stablehlo.divide %vitb11_2sum, %vitb11_2nf : tensor<32x197xf32>
    %vitb11_2mub = stablehlo.broadcast_in_dim %vitb11_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_2xc = stablehlo.subtract %vitb11_r1, %vitb11_2mub : tensor<32x197x192xf32>
    %vitb11_2sq = stablehlo.multiply %vitb11_2xc, %vitb11_2xc : tensor<32x197x192xf32>
    %vitb11_2vsum = stablehlo.reduce(%vitb11_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_2var = stablehlo.divide %vitb11_2vsum, %vitb11_2nf : tensor<32x197xf32>
    %vitb11_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb11_2ve = stablehlo.add %vitb11_2var, %vitb11_2eps : tensor<32x197xf32>
    %vitb11_2istd = stablehlo.rsqrt %vitb11_2ve : tensor<32x197xf32>
    %vitb11_2istdb = stablehlo.broadcast_in_dim %vitb11_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_2xhat = stablehlo.multiply %vitb11_2xc, %vitb11_2istdb : tensor<32x197x192xf32>
    %vitb11_2gb = stablehlo.broadcast_in_dim %g2_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_2bbc = stablehlo.broadcast_in_dim %b2_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_2gx = stablehlo.multiply %vitb11_2xhat, %vitb11_2gb : tensor<32x197x192xf32>
    %vitb11_2y = stablehlo.add %vitb11_2gx, %vitb11_2bbc : tensor<32x197x192xf32>
    %vitb11_ph1d = stablehlo.dot_general %vitb11_2y, %Wfc1_11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb11_ph1bb = stablehlo.broadcast_in_dim %bfc1_11, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb11_ph1 = stablehlo.add %vitb11_ph1d, %vitb11_ph1bb : tensor<32x197x768xf32>
    %vitb11_pgx2 = stablehlo.multiply %vitb11_ph1, %vitb11_ph1 : tensor<32x197x768xf32>
    %vitb11_pgx3 = stablehlo.multiply %vitb11_pgx2, %vitb11_ph1 : tensor<32x197x768xf32>
    %vitb11_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb11_pgkx3 = stablehlo.multiply %vitb11_pgck, %vitb11_pgx3 : tensor<32x197x768xf32>
    %vitb11_pginn = stablehlo.add %vitb11_ph1, %vitb11_pgkx3 : tensor<32x197x768xf32>
    %vitb11_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb11_pgu = stablehlo.multiply %vitb11_pgcsqrt, %vitb11_pginn : tensor<32x197x768xf32>
    %vitb11_pgt = stablehlo.tanh %vitb11_pgu : tensor<32x197x768xf32>
    %vitb11_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb11_pgopt = stablehlo.add %vitb11_pgone, %vitb11_pgt : tensor<32x197x768xf32>
    %vitb11_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb11_pghx = stablehlo.multiply %vitb11_pgchalf, %vitb11_ph1 : tensor<32x197x768xf32>
    %vitb11_pga = stablehlo.multiply %vitb11_pghx, %vitb11_pgopt : tensor<32x197x768xf32>
    %vitb11_py2d = stablehlo.dot_general %vitb11_pga, %Wfc2_11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_py2bb = stablehlo.broadcast_in_dim %bfc2_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_py = stablehlo.add %vitb11_py2d, %vitb11_py2bb : tensor<32x197x192xf32>
    %vitb11_out = stablehlo.add %vitb11_r1, %vitb11_py : tensor<32x197x192xf32>
    %vitflnsum = stablehlo.reduce(%vitb11_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitflnnf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitflnmu = stablehlo.divide %vitflnsum, %vitflnnf : tensor<32x197xf32>
    %vitflnmub = stablehlo.broadcast_in_dim %vitflnmu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitflnxc = stablehlo.subtract %vitb11_out, %vitflnmub : tensor<32x197x192xf32>
    %vitflnsq = stablehlo.multiply %vitflnxc, %vitflnxc : tensor<32x197x192xf32>
    %vitflnvsum = stablehlo.reduce(%vitflnsq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitflnvar = stablehlo.divide %vitflnvsum, %vitflnnf : tensor<32x197xf32>
    %vitflneps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitflnve = stablehlo.add %vitflnvar, %vitflneps : tensor<32x197xf32>
    %vitflnistd = stablehlo.rsqrt %vitflnve : tensor<32x197xf32>
    %vitflnistdb = stablehlo.broadcast_in_dim %vitflnistd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitflnxhat = stablehlo.multiply %vitflnxc, %vitflnistdb : tensor<32x197x192xf32>
    %vitflngb = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitflnbbc = stablehlo.broadcast_in_dim %bF, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitflngx = stablehlo.multiply %vitflnxhat, %vitflngb : tensor<32x197x192xf32>
    %vitflny = stablehlo.add %vitflngx, %vitflnbbc : tensor<32x197x192xf32>
    %vithdcls = stablehlo.slice %vitflny [0:32, 0:1, 0:192] : (tensor<32x197x192xf32>) -> tensor<32x1x192xf32>
    %vithdclsv = stablehlo.reshape %vithdcls : (tensor<32x1x192xf32>) -> tensor<32x192xf32>
    %vithdhd = stablehlo.dot_general %vithdclsv, %Wc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x192xf32>, tensor<192x10xf32>) -> tensor<32x10xf32>
    %vithdhbb = stablehlo.broadcast_in_dim %bc, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %vithdlogits = stablehlo.add %vithdhd, %vithdhbb : tensor<32x10xf32>
    %le = stablehlo.exponential %vithdlogits : tensor<32x10xf32>
    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %lsm = stablehlo.divide %le, %lsb : tensor<32x10xf32>
    %dyr0 = stablehlo.subtract %lsm, %onehot : tensor<32x10xf32>
    %lsa = stablehlo.constant dense<0.100000> : tensor<32x10xf32>
    %lsaoh = stablehlo.multiply %lsa, %onehot : tensor<32x10xf32>
    %dyr1 = stablehlo.add %dyr0, %lsaoh : tensor<32x10xf32>
    %lsaik = stablehlo.constant dense<0.010000> : tensor<32x10xf32>
    %dyr = stablehlo.subtract %dyr1, %lsaik : tensor<32x10xf32>
    %bnc = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %dy = stablehlo.divide %dyr, %bnc : tensor<32x10xf32>
    %llog = stablehlo.log %lsm : tensor<32x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<32x10xf32>
    %t1s = stablehlo.reduce(%ohll init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %lls = stablehlo.reduce(%llog init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %omac = stablehlo.constant dense<0.900000> : tensor<32xf32>
    %aKc = stablehlo.constant dense<0.010000> : tensor<32xf32>
    %lt1 = stablehlo.multiply %omac, %t1s : tensor<32xf32>
    %lt2 = stablehlo.multiply %aKc, %lls : tensor<32xf32>
    %lpe = stablehlo.add %lt1, %lt2 : tensor<32xf32>
    %lsum2 = stablehlo.reduce(%lpe init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %lbfc = stablehlo.constant dense<32.0> : tensor<f32>
    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>
    %loss = stablehlo.negate %lossm : tensor<f32>
    %vithddWc = stablehlo.dot_general %vithdclsv, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x192xf32>, tensor<32x10xf32>) -> tensor<192x10xf32>
    %vithddbc = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %vithddclsv = stablehlo.dot_general %dy, %Wc, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<192x10xf32>) -> tensor<32x192xf32>
    %vithddclsr = stablehlo.reshape %vithddclsv : (tensor<32x192xf32>) -> tensor<32x1x192xf32>
    %vithddz = stablehlo.pad %vithddclsr, %sc, low = [0, 0, 0], high = [0, 196, 0], interior = [0, 0, 0] : (tensor<32x1x192xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %vitflngbk = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitflndxhat = stablehlo.multiply %vithddz, %vitflngbk : tensor<32x197x192xf32>
    %vitflndgpre = stablehlo.multiply %vithddz, %vitflnxhat : tensor<32x197x192xf32>
    %vitflndg = stablehlo.reduce(%vitflndgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitflndb = stablehlo.reduce(%vithddz init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitflnm1s = stablehlo.reduce(%vitflndxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitflnm1 = stablehlo.divide %vitflnm1s, %vitflnnf : tensor<32x197xf32>
    %vitflndxxh = stablehlo.multiply %vitflndxhat, %vitflnxhat : tensor<32x197x192xf32>
    %vitflnm2s = stablehlo.reduce(%vitflndxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitflnm2 = stablehlo.divide %vitflnm2s, %vitflnnf : tensor<32x197xf32>
    %vitflnm1b = stablehlo.broadcast_in_dim %vitflnm1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitflnm2b = stablehlo.broadcast_in_dim %vitflnm2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitflnt1 = stablehlo.subtract %vitflndxhat, %vitflnm1b : tensor<32x197x192xf32>
    %vitflnxm2 = stablehlo.multiply %vitflnxhat, %vitflnm2b : tensor<32x197x192xf32>
    %vitflnt2 = stablehlo.subtract %vitflnt1, %vitflnxm2 : tensor<32x197x192xf32>
    %vitflndx = stablehlo.multiply %vitflnistdb, %vitflnt2 : tensor<32x197x192xf32>
    %vitb11_pda1 = stablehlo.dot_general %vitflndx, %Wfc2_11, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb11_pdWfc2 = stablehlo.dot_general %vitb11_pga, %vitflndx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb11_pdbfc2 = stablehlo.reduce(%vitflndx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_pgbbx2 = stablehlo.multiply %vitb11_ph1, %vitb11_ph1 : tensor<32x197x768xf32>
    %vitb11_pgbbx3 = stablehlo.multiply %vitb11_pgbbx2, %vitb11_ph1 : tensor<32x197x768xf32>
    %vitb11_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb11_pgbbkx3 = stablehlo.multiply %vitb11_pgbbck, %vitb11_pgbbx3 : tensor<32x197x768xf32>
    %vitb11_pgbbinn = stablehlo.add %vitb11_ph1, %vitb11_pgbbkx3 : tensor<32x197x768xf32>
    %vitb11_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb11_pgbbu = stablehlo.multiply %vitb11_pgbbcsqrt, %vitb11_pgbbinn : tensor<32x197x768xf32>
    %vitb11_pgbbt = stablehlo.tanh %vitb11_pgbbu : tensor<32x197x768xf32>
    %vitb11_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb11_pgbbopt = stablehlo.add %vitb11_pgbbone, %vitb11_pgbbt : tensor<32x197x768xf32>
    %vitb11_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb11_pgbbterm1 = stablehlo.multiply %vitb11_pgbbchalf, %vitb11_pgbbopt : tensor<32x197x768xf32>
    %vitb11_pgbbt2 = stablehlo.multiply %vitb11_pgbbt, %vitb11_pgbbt : tensor<32x197x768xf32>
    %vitb11_pgbbomt2 = stablehlo.subtract %vitb11_pgbbone, %vitb11_pgbbt2 : tensor<32x197x768xf32>
    %vitb11_pgbbhx = stablehlo.multiply %vitb11_pgbbchalf, %vitb11_ph1 : tensor<32x197x768xf32>
    %vitb11_pgbbhxo = stablehlo.multiply %vitb11_pgbbhx, %vitb11_pgbbomt2 : tensor<32x197x768xf32>
    %vitb11_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb11_pgbba3x2 = stablehlo.multiply %vitb11_pgbbc3b, %vitb11_pgbbx2 : tensor<32x197x768xf32>
    %vitb11_pgbbin2 = stablehlo.add %vitb11_pgbbone, %vitb11_pgbba3x2 : tensor<32x197x768xf32>
    %vitb11_pgbbup = stablehlo.multiply %vitb11_pgbbcsqrt, %vitb11_pgbbin2 : tensor<32x197x768xf32>
    %vitb11_pgbbterm2 = stablehlo.multiply %vitb11_pgbbhxo, %vitb11_pgbbup : tensor<32x197x768xf32>
    %vitb11_pgbbgp = stablehlo.add %vitb11_pgbbterm1, %vitb11_pgbbterm2 : tensor<32x197x768xf32>
    %vitb11_pgbdx = stablehlo.multiply %vitb11_pda1, %vitb11_pgbbgp : tensor<32x197x768xf32>
    %vitb11_pdx = stablehlo.dot_general %vitb11_pgbdx, %Wfc1_11, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb11_pdWfc1 = stablehlo.dot_general %vitb11_2y, %vitb11_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb11_pdbfc1 = stablehlo.reduce(%vitb11_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb11_2gbk = stablehlo.broadcast_in_dim %g2_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_2dxhat = stablehlo.multiply %vitb11_pdx, %vitb11_2gbk : tensor<32x197x192xf32>
    %vitb11_2dgpre = stablehlo.multiply %vitb11_pdx, %vitb11_2xhat : tensor<32x197x192xf32>
    %vitb11_2dg = stablehlo.reduce(%vitb11_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_2db = stablehlo.reduce(%vitb11_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_2m1s = stablehlo.reduce(%vitb11_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_2m1 = stablehlo.divide %vitb11_2m1s, %vitb11_2nf : tensor<32x197xf32>
    %vitb11_2dxxh = stablehlo.multiply %vitb11_2dxhat, %vitb11_2xhat : tensor<32x197x192xf32>
    %vitb11_2m2s = stablehlo.reduce(%vitb11_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_2m2 = stablehlo.divide %vitb11_2m2s, %vitb11_2nf : tensor<32x197xf32>
    %vitb11_2m1b = stablehlo.broadcast_in_dim %vitb11_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_2m2b = stablehlo.broadcast_in_dim %vitb11_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_2t1 = stablehlo.subtract %vitb11_2dxhat, %vitb11_2m1b : tensor<32x197x192xf32>
    %vitb11_2xm2 = stablehlo.multiply %vitb11_2xhat, %vitb11_2m2b : tensor<32x197x192xf32>
    %vitb11_2t2 = stablehlo.subtract %vitb11_2t1, %vitb11_2xm2 : tensor<32x197x192xf32>
    %vitb11_2dx = stablehlo.multiply %vitb11_2istdb, %vitb11_2t2 : tensor<32x197x192xf32>
    %vitb11_dr1 = stablehlo.add %vitflndx, %vitb11_2dx : tensor<32x197x192xf32>
    %vitb11_mdP = stablehlo.dot_general %vitb11_dr1, %Wo_11, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mdWo = stablehlo.dot_general %vitb11_mP, %vitb11_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb11_mdbo = stablehlo.reduce(%vitb11_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_mdPr = stablehlo.reshape %vitb11_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mdA = stablehlo.transpose %vitb11_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mdW = stablehlo.dot_general %vitb11_mdA, %vitb11_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb11_mdVh = stablehlo.dot_general %vitb11_mW, %vitb11_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mpdw = stablehlo.multiply %vitb11_mW, %vitb11_mdW : tensor<32x3x197x197xf32>
    %vitb11_msrow = stablehlo.reduce(%vitb11_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb11_msrowb = stablehlo.broadcast_in_dim %vitb11_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb11_mdiff = stablehlo.subtract %vitb11_mdW, %vitb11_msrowb : tensor<32x3x197x197xf32>
    %vitb11_mdSs = stablehlo.multiply %vitb11_mW, %vitb11_mdiff : tensor<32x3x197x197xf32>
    %vitb11_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb11_mdS = stablehlo.multiply %vitb11_mdSs, %vitb11_msclb : tensor<32x3x197x197xf32>
    %vitb11_mdQh = stablehlo.dot_general %vitb11_mdS, %vitb11_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mdKh = stablehlo.dot_general %vitb11_mdS, %vitb11_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mdQT = stablehlo.transpose %vitb11_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mdQ = stablehlo.reshape %vitb11_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb11_mdKT = stablehlo.transpose %vitb11_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mdK = stablehlo.reshape %vitb11_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb11_mdVT = stablehlo.transpose %vitb11_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mdV = stablehlo.reshape %vitb11_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb11_mdxQ = stablehlo.dot_general %vitb11_mdQ, %Wq_11, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mdWQ = stablehlo.dot_general %vitb11_1y, %vitb11_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb11_mdbQ = stablehlo.reduce(%vitb11_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_mdxK = stablehlo.dot_general %vitb11_mdK, %Wk_11, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mdWK = stablehlo.dot_general %vitb11_1y, %vitb11_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb11_mdbK = stablehlo.reduce(%vitb11_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_mdxV = stablehlo.dot_general %vitb11_mdV, %Wv_11, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mdWV = stablehlo.dot_general %vitb11_1y, %vitb11_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb11_mdbV = stablehlo.reduce(%vitb11_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_mdxa = stablehlo.add %vitb11_mdxQ, %vitb11_mdxK : tensor<32x197x192xf32>
    %vitb11_mdx = stablehlo.add %vitb11_mdxa, %vitb11_mdxV : tensor<32x197x192xf32>
    %vitb11_1gbk = stablehlo.broadcast_in_dim %g1_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_1dxhat = stablehlo.multiply %vitb11_mdx, %vitb11_1gbk : tensor<32x197x192xf32>
    %vitb11_1dgpre = stablehlo.multiply %vitb11_mdx, %vitb11_1xhat : tensor<32x197x192xf32>
    %vitb11_1dg = stablehlo.reduce(%vitb11_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_1db = stablehlo.reduce(%vitb11_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_1m1s = stablehlo.reduce(%vitb11_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_1m1 = stablehlo.divide %vitb11_1m1s, %vitb11_1nf : tensor<32x197xf32>
    %vitb11_1dxxh = stablehlo.multiply %vitb11_1dxhat, %vitb11_1xhat : tensor<32x197x192xf32>
    %vitb11_1m2s = stablehlo.reduce(%vitb11_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_1m2 = stablehlo.divide %vitb11_1m2s, %vitb11_1nf : tensor<32x197xf32>
    %vitb11_1m1b = stablehlo.broadcast_in_dim %vitb11_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_1m2b = stablehlo.broadcast_in_dim %vitb11_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_1t1 = stablehlo.subtract %vitb11_1dxhat, %vitb11_1m1b : tensor<32x197x192xf32>
    %vitb11_1xm2 = stablehlo.multiply %vitb11_1xhat, %vitb11_1m2b : tensor<32x197x192xf32>
    %vitb11_1t2 = stablehlo.subtract %vitb11_1t1, %vitb11_1xm2 : tensor<32x197x192xf32>
    %vitb11_1dx = stablehlo.multiply %vitb11_1istdb, %vitb11_1t2 : tensor<32x197x192xf32>
    %vitb11_dx = stablehlo.add %vitb11_dr1, %vitb11_1dx : tensor<32x197x192xf32>
    %vitb10_pda1 = stablehlo.dot_general %vitb11_dx, %Wfc2_10, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb10_pdWfc2 = stablehlo.dot_general %vitb10_pga, %vitb11_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb10_pdbfc2 = stablehlo.reduce(%vitb11_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_pgbbx2 = stablehlo.multiply %vitb10_ph1, %vitb10_ph1 : tensor<32x197x768xf32>
    %vitb10_pgbbx3 = stablehlo.multiply %vitb10_pgbbx2, %vitb10_ph1 : tensor<32x197x768xf32>
    %vitb10_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb10_pgbbkx3 = stablehlo.multiply %vitb10_pgbbck, %vitb10_pgbbx3 : tensor<32x197x768xf32>
    %vitb10_pgbbinn = stablehlo.add %vitb10_ph1, %vitb10_pgbbkx3 : tensor<32x197x768xf32>
    %vitb10_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb10_pgbbu = stablehlo.multiply %vitb10_pgbbcsqrt, %vitb10_pgbbinn : tensor<32x197x768xf32>
    %vitb10_pgbbt = stablehlo.tanh %vitb10_pgbbu : tensor<32x197x768xf32>
    %vitb10_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb10_pgbbopt = stablehlo.add %vitb10_pgbbone, %vitb10_pgbbt : tensor<32x197x768xf32>
    %vitb10_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb10_pgbbterm1 = stablehlo.multiply %vitb10_pgbbchalf, %vitb10_pgbbopt : tensor<32x197x768xf32>
    %vitb10_pgbbt2 = stablehlo.multiply %vitb10_pgbbt, %vitb10_pgbbt : tensor<32x197x768xf32>
    %vitb10_pgbbomt2 = stablehlo.subtract %vitb10_pgbbone, %vitb10_pgbbt2 : tensor<32x197x768xf32>
    %vitb10_pgbbhx = stablehlo.multiply %vitb10_pgbbchalf, %vitb10_ph1 : tensor<32x197x768xf32>
    %vitb10_pgbbhxo = stablehlo.multiply %vitb10_pgbbhx, %vitb10_pgbbomt2 : tensor<32x197x768xf32>
    %vitb10_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb10_pgbba3x2 = stablehlo.multiply %vitb10_pgbbc3b, %vitb10_pgbbx2 : tensor<32x197x768xf32>
    %vitb10_pgbbin2 = stablehlo.add %vitb10_pgbbone, %vitb10_pgbba3x2 : tensor<32x197x768xf32>
    %vitb10_pgbbup = stablehlo.multiply %vitb10_pgbbcsqrt, %vitb10_pgbbin2 : tensor<32x197x768xf32>
    %vitb10_pgbbterm2 = stablehlo.multiply %vitb10_pgbbhxo, %vitb10_pgbbup : tensor<32x197x768xf32>
    %vitb10_pgbbgp = stablehlo.add %vitb10_pgbbterm1, %vitb10_pgbbterm2 : tensor<32x197x768xf32>
    %vitb10_pgbdx = stablehlo.multiply %vitb10_pda1, %vitb10_pgbbgp : tensor<32x197x768xf32>
    %vitb10_pdx = stablehlo.dot_general %vitb10_pgbdx, %Wfc1_10, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb10_pdWfc1 = stablehlo.dot_general %vitb10_2y, %vitb10_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb10_pdbfc1 = stablehlo.reduce(%vitb10_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb10_2gbk = stablehlo.broadcast_in_dim %g2_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_2dxhat = stablehlo.multiply %vitb10_pdx, %vitb10_2gbk : tensor<32x197x192xf32>
    %vitb10_2dgpre = stablehlo.multiply %vitb10_pdx, %vitb10_2xhat : tensor<32x197x192xf32>
    %vitb10_2dg = stablehlo.reduce(%vitb10_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_2db = stablehlo.reduce(%vitb10_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_2m1s = stablehlo.reduce(%vitb10_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_2m1 = stablehlo.divide %vitb10_2m1s, %vitb10_2nf : tensor<32x197xf32>
    %vitb10_2dxxh = stablehlo.multiply %vitb10_2dxhat, %vitb10_2xhat : tensor<32x197x192xf32>
    %vitb10_2m2s = stablehlo.reduce(%vitb10_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_2m2 = stablehlo.divide %vitb10_2m2s, %vitb10_2nf : tensor<32x197xf32>
    %vitb10_2m1b = stablehlo.broadcast_in_dim %vitb10_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_2m2b = stablehlo.broadcast_in_dim %vitb10_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_2t1 = stablehlo.subtract %vitb10_2dxhat, %vitb10_2m1b : tensor<32x197x192xf32>
    %vitb10_2xm2 = stablehlo.multiply %vitb10_2xhat, %vitb10_2m2b : tensor<32x197x192xf32>
    %vitb10_2t2 = stablehlo.subtract %vitb10_2t1, %vitb10_2xm2 : tensor<32x197x192xf32>
    %vitb10_2dx = stablehlo.multiply %vitb10_2istdb, %vitb10_2t2 : tensor<32x197x192xf32>
    %vitb10_dr1 = stablehlo.add %vitb11_dx, %vitb10_2dx : tensor<32x197x192xf32>
    %vitb10_mdP = stablehlo.dot_general %vitb10_dr1, %Wo_10, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mdWo = stablehlo.dot_general %vitb10_mP, %vitb10_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb10_mdbo = stablehlo.reduce(%vitb10_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_mdPr = stablehlo.reshape %vitb10_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mdA = stablehlo.transpose %vitb10_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mdW = stablehlo.dot_general %vitb10_mdA, %vitb10_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb10_mdVh = stablehlo.dot_general %vitb10_mW, %vitb10_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mpdw = stablehlo.multiply %vitb10_mW, %vitb10_mdW : tensor<32x3x197x197xf32>
    %vitb10_msrow = stablehlo.reduce(%vitb10_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb10_msrowb = stablehlo.broadcast_in_dim %vitb10_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb10_mdiff = stablehlo.subtract %vitb10_mdW, %vitb10_msrowb : tensor<32x3x197x197xf32>
    %vitb10_mdSs = stablehlo.multiply %vitb10_mW, %vitb10_mdiff : tensor<32x3x197x197xf32>
    %vitb10_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb10_mdS = stablehlo.multiply %vitb10_mdSs, %vitb10_msclb : tensor<32x3x197x197xf32>
    %vitb10_mdQh = stablehlo.dot_general %vitb10_mdS, %vitb10_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mdKh = stablehlo.dot_general %vitb10_mdS, %vitb10_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mdQT = stablehlo.transpose %vitb10_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mdQ = stablehlo.reshape %vitb10_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb10_mdKT = stablehlo.transpose %vitb10_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mdK = stablehlo.reshape %vitb10_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb10_mdVT = stablehlo.transpose %vitb10_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mdV = stablehlo.reshape %vitb10_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb10_mdxQ = stablehlo.dot_general %vitb10_mdQ, %Wq_10, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mdWQ = stablehlo.dot_general %vitb10_1y, %vitb10_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb10_mdbQ = stablehlo.reduce(%vitb10_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_mdxK = stablehlo.dot_general %vitb10_mdK, %Wk_10, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mdWK = stablehlo.dot_general %vitb10_1y, %vitb10_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb10_mdbK = stablehlo.reduce(%vitb10_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_mdxV = stablehlo.dot_general %vitb10_mdV, %Wv_10, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mdWV = stablehlo.dot_general %vitb10_1y, %vitb10_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb10_mdbV = stablehlo.reduce(%vitb10_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_mdxa = stablehlo.add %vitb10_mdxQ, %vitb10_mdxK : tensor<32x197x192xf32>
    %vitb10_mdx = stablehlo.add %vitb10_mdxa, %vitb10_mdxV : tensor<32x197x192xf32>
    %vitb10_1gbk = stablehlo.broadcast_in_dim %g1_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_1dxhat = stablehlo.multiply %vitb10_mdx, %vitb10_1gbk : tensor<32x197x192xf32>
    %vitb10_1dgpre = stablehlo.multiply %vitb10_mdx, %vitb10_1xhat : tensor<32x197x192xf32>
    %vitb10_1dg = stablehlo.reduce(%vitb10_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_1db = stablehlo.reduce(%vitb10_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_1m1s = stablehlo.reduce(%vitb10_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_1m1 = stablehlo.divide %vitb10_1m1s, %vitb10_1nf : tensor<32x197xf32>
    %vitb10_1dxxh = stablehlo.multiply %vitb10_1dxhat, %vitb10_1xhat : tensor<32x197x192xf32>
    %vitb10_1m2s = stablehlo.reduce(%vitb10_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_1m2 = stablehlo.divide %vitb10_1m2s, %vitb10_1nf : tensor<32x197xf32>
    %vitb10_1m1b = stablehlo.broadcast_in_dim %vitb10_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_1m2b = stablehlo.broadcast_in_dim %vitb10_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_1t1 = stablehlo.subtract %vitb10_1dxhat, %vitb10_1m1b : tensor<32x197x192xf32>
    %vitb10_1xm2 = stablehlo.multiply %vitb10_1xhat, %vitb10_1m2b : tensor<32x197x192xf32>
    %vitb10_1t2 = stablehlo.subtract %vitb10_1t1, %vitb10_1xm2 : tensor<32x197x192xf32>
    %vitb10_1dx = stablehlo.multiply %vitb10_1istdb, %vitb10_1t2 : tensor<32x197x192xf32>
    %vitb10_dx = stablehlo.add %vitb10_dr1, %vitb10_1dx : tensor<32x197x192xf32>
    %vitb9_pda1 = stablehlo.dot_general %vitb10_dx, %Wfc2_9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb9_pdWfc2 = stablehlo.dot_general %vitb9_pga, %vitb10_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb9_pdbfc2 = stablehlo.reduce(%vitb10_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_pgbbx2 = stablehlo.multiply %vitb9_ph1, %vitb9_ph1 : tensor<32x197x768xf32>
    %vitb9_pgbbx3 = stablehlo.multiply %vitb9_pgbbx2, %vitb9_ph1 : tensor<32x197x768xf32>
    %vitb9_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb9_pgbbkx3 = stablehlo.multiply %vitb9_pgbbck, %vitb9_pgbbx3 : tensor<32x197x768xf32>
    %vitb9_pgbbinn = stablehlo.add %vitb9_ph1, %vitb9_pgbbkx3 : tensor<32x197x768xf32>
    %vitb9_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb9_pgbbu = stablehlo.multiply %vitb9_pgbbcsqrt, %vitb9_pgbbinn : tensor<32x197x768xf32>
    %vitb9_pgbbt = stablehlo.tanh %vitb9_pgbbu : tensor<32x197x768xf32>
    %vitb9_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb9_pgbbopt = stablehlo.add %vitb9_pgbbone, %vitb9_pgbbt : tensor<32x197x768xf32>
    %vitb9_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb9_pgbbterm1 = stablehlo.multiply %vitb9_pgbbchalf, %vitb9_pgbbopt : tensor<32x197x768xf32>
    %vitb9_pgbbt2 = stablehlo.multiply %vitb9_pgbbt, %vitb9_pgbbt : tensor<32x197x768xf32>
    %vitb9_pgbbomt2 = stablehlo.subtract %vitb9_pgbbone, %vitb9_pgbbt2 : tensor<32x197x768xf32>
    %vitb9_pgbbhx = stablehlo.multiply %vitb9_pgbbchalf, %vitb9_ph1 : tensor<32x197x768xf32>
    %vitb9_pgbbhxo = stablehlo.multiply %vitb9_pgbbhx, %vitb9_pgbbomt2 : tensor<32x197x768xf32>
    %vitb9_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb9_pgbba3x2 = stablehlo.multiply %vitb9_pgbbc3b, %vitb9_pgbbx2 : tensor<32x197x768xf32>
    %vitb9_pgbbin2 = stablehlo.add %vitb9_pgbbone, %vitb9_pgbba3x2 : tensor<32x197x768xf32>
    %vitb9_pgbbup = stablehlo.multiply %vitb9_pgbbcsqrt, %vitb9_pgbbin2 : tensor<32x197x768xf32>
    %vitb9_pgbbterm2 = stablehlo.multiply %vitb9_pgbbhxo, %vitb9_pgbbup : tensor<32x197x768xf32>
    %vitb9_pgbbgp = stablehlo.add %vitb9_pgbbterm1, %vitb9_pgbbterm2 : tensor<32x197x768xf32>
    %vitb9_pgbdx = stablehlo.multiply %vitb9_pda1, %vitb9_pgbbgp : tensor<32x197x768xf32>
    %vitb9_pdx = stablehlo.dot_general %vitb9_pgbdx, %Wfc1_9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb9_pdWfc1 = stablehlo.dot_general %vitb9_2y, %vitb9_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb9_pdbfc1 = stablehlo.reduce(%vitb9_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb9_2gbk = stablehlo.broadcast_in_dim %g2_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_2dxhat = stablehlo.multiply %vitb9_pdx, %vitb9_2gbk : tensor<32x197x192xf32>
    %vitb9_2dgpre = stablehlo.multiply %vitb9_pdx, %vitb9_2xhat : tensor<32x197x192xf32>
    %vitb9_2dg = stablehlo.reduce(%vitb9_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_2db = stablehlo.reduce(%vitb9_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_2m1s = stablehlo.reduce(%vitb9_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_2m1 = stablehlo.divide %vitb9_2m1s, %vitb9_2nf : tensor<32x197xf32>
    %vitb9_2dxxh = stablehlo.multiply %vitb9_2dxhat, %vitb9_2xhat : tensor<32x197x192xf32>
    %vitb9_2m2s = stablehlo.reduce(%vitb9_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_2m2 = stablehlo.divide %vitb9_2m2s, %vitb9_2nf : tensor<32x197xf32>
    %vitb9_2m1b = stablehlo.broadcast_in_dim %vitb9_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_2m2b = stablehlo.broadcast_in_dim %vitb9_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_2t1 = stablehlo.subtract %vitb9_2dxhat, %vitb9_2m1b : tensor<32x197x192xf32>
    %vitb9_2xm2 = stablehlo.multiply %vitb9_2xhat, %vitb9_2m2b : tensor<32x197x192xf32>
    %vitb9_2t2 = stablehlo.subtract %vitb9_2t1, %vitb9_2xm2 : tensor<32x197x192xf32>
    %vitb9_2dx = stablehlo.multiply %vitb9_2istdb, %vitb9_2t2 : tensor<32x197x192xf32>
    %vitb9_dr1 = stablehlo.add %vitb10_dx, %vitb9_2dx : tensor<32x197x192xf32>
    %vitb9_mdP = stablehlo.dot_general %vitb9_dr1, %Wo_9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mdWo = stablehlo.dot_general %vitb9_mP, %vitb9_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb9_mdbo = stablehlo.reduce(%vitb9_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_mdPr = stablehlo.reshape %vitb9_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mdA = stablehlo.transpose %vitb9_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mdW = stablehlo.dot_general %vitb9_mdA, %vitb9_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb9_mdVh = stablehlo.dot_general %vitb9_mW, %vitb9_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mpdw = stablehlo.multiply %vitb9_mW, %vitb9_mdW : tensor<32x3x197x197xf32>
    %vitb9_msrow = stablehlo.reduce(%vitb9_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb9_msrowb = stablehlo.broadcast_in_dim %vitb9_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb9_mdiff = stablehlo.subtract %vitb9_mdW, %vitb9_msrowb : tensor<32x3x197x197xf32>
    %vitb9_mdSs = stablehlo.multiply %vitb9_mW, %vitb9_mdiff : tensor<32x3x197x197xf32>
    %vitb9_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb9_mdS = stablehlo.multiply %vitb9_mdSs, %vitb9_msclb : tensor<32x3x197x197xf32>
    %vitb9_mdQh = stablehlo.dot_general %vitb9_mdS, %vitb9_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mdKh = stablehlo.dot_general %vitb9_mdS, %vitb9_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mdQT = stablehlo.transpose %vitb9_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mdQ = stablehlo.reshape %vitb9_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb9_mdKT = stablehlo.transpose %vitb9_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mdK = stablehlo.reshape %vitb9_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb9_mdVT = stablehlo.transpose %vitb9_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mdV = stablehlo.reshape %vitb9_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb9_mdxQ = stablehlo.dot_general %vitb9_mdQ, %Wq_9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mdWQ = stablehlo.dot_general %vitb9_1y, %vitb9_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb9_mdbQ = stablehlo.reduce(%vitb9_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_mdxK = stablehlo.dot_general %vitb9_mdK, %Wk_9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mdWK = stablehlo.dot_general %vitb9_1y, %vitb9_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb9_mdbK = stablehlo.reduce(%vitb9_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_mdxV = stablehlo.dot_general %vitb9_mdV, %Wv_9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mdWV = stablehlo.dot_general %vitb9_1y, %vitb9_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb9_mdbV = stablehlo.reduce(%vitb9_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_mdxa = stablehlo.add %vitb9_mdxQ, %vitb9_mdxK : tensor<32x197x192xf32>
    %vitb9_mdx = stablehlo.add %vitb9_mdxa, %vitb9_mdxV : tensor<32x197x192xf32>
    %vitb9_1gbk = stablehlo.broadcast_in_dim %g1_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_1dxhat = stablehlo.multiply %vitb9_mdx, %vitb9_1gbk : tensor<32x197x192xf32>
    %vitb9_1dgpre = stablehlo.multiply %vitb9_mdx, %vitb9_1xhat : tensor<32x197x192xf32>
    %vitb9_1dg = stablehlo.reduce(%vitb9_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_1db = stablehlo.reduce(%vitb9_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_1m1s = stablehlo.reduce(%vitb9_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_1m1 = stablehlo.divide %vitb9_1m1s, %vitb9_1nf : tensor<32x197xf32>
    %vitb9_1dxxh = stablehlo.multiply %vitb9_1dxhat, %vitb9_1xhat : tensor<32x197x192xf32>
    %vitb9_1m2s = stablehlo.reduce(%vitb9_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_1m2 = stablehlo.divide %vitb9_1m2s, %vitb9_1nf : tensor<32x197xf32>
    %vitb9_1m1b = stablehlo.broadcast_in_dim %vitb9_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_1m2b = stablehlo.broadcast_in_dim %vitb9_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_1t1 = stablehlo.subtract %vitb9_1dxhat, %vitb9_1m1b : tensor<32x197x192xf32>
    %vitb9_1xm2 = stablehlo.multiply %vitb9_1xhat, %vitb9_1m2b : tensor<32x197x192xf32>
    %vitb9_1t2 = stablehlo.subtract %vitb9_1t1, %vitb9_1xm2 : tensor<32x197x192xf32>
    %vitb9_1dx = stablehlo.multiply %vitb9_1istdb, %vitb9_1t2 : tensor<32x197x192xf32>
    %vitb9_dx = stablehlo.add %vitb9_dr1, %vitb9_1dx : tensor<32x197x192xf32>
    %vitb8_pda1 = stablehlo.dot_general %vitb9_dx, %Wfc2_8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb8_pdWfc2 = stablehlo.dot_general %vitb8_pga, %vitb9_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb8_pdbfc2 = stablehlo.reduce(%vitb9_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_pgbbx2 = stablehlo.multiply %vitb8_ph1, %vitb8_ph1 : tensor<32x197x768xf32>
    %vitb8_pgbbx3 = stablehlo.multiply %vitb8_pgbbx2, %vitb8_ph1 : tensor<32x197x768xf32>
    %vitb8_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb8_pgbbkx3 = stablehlo.multiply %vitb8_pgbbck, %vitb8_pgbbx3 : tensor<32x197x768xf32>
    %vitb8_pgbbinn = stablehlo.add %vitb8_ph1, %vitb8_pgbbkx3 : tensor<32x197x768xf32>
    %vitb8_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb8_pgbbu = stablehlo.multiply %vitb8_pgbbcsqrt, %vitb8_pgbbinn : tensor<32x197x768xf32>
    %vitb8_pgbbt = stablehlo.tanh %vitb8_pgbbu : tensor<32x197x768xf32>
    %vitb8_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb8_pgbbopt = stablehlo.add %vitb8_pgbbone, %vitb8_pgbbt : tensor<32x197x768xf32>
    %vitb8_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb8_pgbbterm1 = stablehlo.multiply %vitb8_pgbbchalf, %vitb8_pgbbopt : tensor<32x197x768xf32>
    %vitb8_pgbbt2 = stablehlo.multiply %vitb8_pgbbt, %vitb8_pgbbt : tensor<32x197x768xf32>
    %vitb8_pgbbomt2 = stablehlo.subtract %vitb8_pgbbone, %vitb8_pgbbt2 : tensor<32x197x768xf32>
    %vitb8_pgbbhx = stablehlo.multiply %vitb8_pgbbchalf, %vitb8_ph1 : tensor<32x197x768xf32>
    %vitb8_pgbbhxo = stablehlo.multiply %vitb8_pgbbhx, %vitb8_pgbbomt2 : tensor<32x197x768xf32>
    %vitb8_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb8_pgbba3x2 = stablehlo.multiply %vitb8_pgbbc3b, %vitb8_pgbbx2 : tensor<32x197x768xf32>
    %vitb8_pgbbin2 = stablehlo.add %vitb8_pgbbone, %vitb8_pgbba3x2 : tensor<32x197x768xf32>
    %vitb8_pgbbup = stablehlo.multiply %vitb8_pgbbcsqrt, %vitb8_pgbbin2 : tensor<32x197x768xf32>
    %vitb8_pgbbterm2 = stablehlo.multiply %vitb8_pgbbhxo, %vitb8_pgbbup : tensor<32x197x768xf32>
    %vitb8_pgbbgp = stablehlo.add %vitb8_pgbbterm1, %vitb8_pgbbterm2 : tensor<32x197x768xf32>
    %vitb8_pgbdx = stablehlo.multiply %vitb8_pda1, %vitb8_pgbbgp : tensor<32x197x768xf32>
    %vitb8_pdx = stablehlo.dot_general %vitb8_pgbdx, %Wfc1_8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb8_pdWfc1 = stablehlo.dot_general %vitb8_2y, %vitb8_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb8_pdbfc1 = stablehlo.reduce(%vitb8_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb8_2gbk = stablehlo.broadcast_in_dim %g2_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_2dxhat = stablehlo.multiply %vitb8_pdx, %vitb8_2gbk : tensor<32x197x192xf32>
    %vitb8_2dgpre = stablehlo.multiply %vitb8_pdx, %vitb8_2xhat : tensor<32x197x192xf32>
    %vitb8_2dg = stablehlo.reduce(%vitb8_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_2db = stablehlo.reduce(%vitb8_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_2m1s = stablehlo.reduce(%vitb8_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_2m1 = stablehlo.divide %vitb8_2m1s, %vitb8_2nf : tensor<32x197xf32>
    %vitb8_2dxxh = stablehlo.multiply %vitb8_2dxhat, %vitb8_2xhat : tensor<32x197x192xf32>
    %vitb8_2m2s = stablehlo.reduce(%vitb8_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_2m2 = stablehlo.divide %vitb8_2m2s, %vitb8_2nf : tensor<32x197xf32>
    %vitb8_2m1b = stablehlo.broadcast_in_dim %vitb8_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_2m2b = stablehlo.broadcast_in_dim %vitb8_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_2t1 = stablehlo.subtract %vitb8_2dxhat, %vitb8_2m1b : tensor<32x197x192xf32>
    %vitb8_2xm2 = stablehlo.multiply %vitb8_2xhat, %vitb8_2m2b : tensor<32x197x192xf32>
    %vitb8_2t2 = stablehlo.subtract %vitb8_2t1, %vitb8_2xm2 : tensor<32x197x192xf32>
    %vitb8_2dx = stablehlo.multiply %vitb8_2istdb, %vitb8_2t2 : tensor<32x197x192xf32>
    %vitb8_dr1 = stablehlo.add %vitb9_dx, %vitb8_2dx : tensor<32x197x192xf32>
    %vitb8_mdP = stablehlo.dot_general %vitb8_dr1, %Wo_8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mdWo = stablehlo.dot_general %vitb8_mP, %vitb8_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb8_mdbo = stablehlo.reduce(%vitb8_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_mdPr = stablehlo.reshape %vitb8_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mdA = stablehlo.transpose %vitb8_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mdW = stablehlo.dot_general %vitb8_mdA, %vitb8_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb8_mdVh = stablehlo.dot_general %vitb8_mW, %vitb8_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mpdw = stablehlo.multiply %vitb8_mW, %vitb8_mdW : tensor<32x3x197x197xf32>
    %vitb8_msrow = stablehlo.reduce(%vitb8_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb8_msrowb = stablehlo.broadcast_in_dim %vitb8_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb8_mdiff = stablehlo.subtract %vitb8_mdW, %vitb8_msrowb : tensor<32x3x197x197xf32>
    %vitb8_mdSs = stablehlo.multiply %vitb8_mW, %vitb8_mdiff : tensor<32x3x197x197xf32>
    %vitb8_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb8_mdS = stablehlo.multiply %vitb8_mdSs, %vitb8_msclb : tensor<32x3x197x197xf32>
    %vitb8_mdQh = stablehlo.dot_general %vitb8_mdS, %vitb8_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mdKh = stablehlo.dot_general %vitb8_mdS, %vitb8_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mdQT = stablehlo.transpose %vitb8_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mdQ = stablehlo.reshape %vitb8_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb8_mdKT = stablehlo.transpose %vitb8_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mdK = stablehlo.reshape %vitb8_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb8_mdVT = stablehlo.transpose %vitb8_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mdV = stablehlo.reshape %vitb8_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb8_mdxQ = stablehlo.dot_general %vitb8_mdQ, %Wq_8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mdWQ = stablehlo.dot_general %vitb8_1y, %vitb8_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb8_mdbQ = stablehlo.reduce(%vitb8_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_mdxK = stablehlo.dot_general %vitb8_mdK, %Wk_8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mdWK = stablehlo.dot_general %vitb8_1y, %vitb8_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb8_mdbK = stablehlo.reduce(%vitb8_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_mdxV = stablehlo.dot_general %vitb8_mdV, %Wv_8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mdWV = stablehlo.dot_general %vitb8_1y, %vitb8_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb8_mdbV = stablehlo.reduce(%vitb8_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_mdxa = stablehlo.add %vitb8_mdxQ, %vitb8_mdxK : tensor<32x197x192xf32>
    %vitb8_mdx = stablehlo.add %vitb8_mdxa, %vitb8_mdxV : tensor<32x197x192xf32>
    %vitb8_1gbk = stablehlo.broadcast_in_dim %g1_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_1dxhat = stablehlo.multiply %vitb8_mdx, %vitb8_1gbk : tensor<32x197x192xf32>
    %vitb8_1dgpre = stablehlo.multiply %vitb8_mdx, %vitb8_1xhat : tensor<32x197x192xf32>
    %vitb8_1dg = stablehlo.reduce(%vitb8_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_1db = stablehlo.reduce(%vitb8_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_1m1s = stablehlo.reduce(%vitb8_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_1m1 = stablehlo.divide %vitb8_1m1s, %vitb8_1nf : tensor<32x197xf32>
    %vitb8_1dxxh = stablehlo.multiply %vitb8_1dxhat, %vitb8_1xhat : tensor<32x197x192xf32>
    %vitb8_1m2s = stablehlo.reduce(%vitb8_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_1m2 = stablehlo.divide %vitb8_1m2s, %vitb8_1nf : tensor<32x197xf32>
    %vitb8_1m1b = stablehlo.broadcast_in_dim %vitb8_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_1m2b = stablehlo.broadcast_in_dim %vitb8_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_1t1 = stablehlo.subtract %vitb8_1dxhat, %vitb8_1m1b : tensor<32x197x192xf32>
    %vitb8_1xm2 = stablehlo.multiply %vitb8_1xhat, %vitb8_1m2b : tensor<32x197x192xf32>
    %vitb8_1t2 = stablehlo.subtract %vitb8_1t1, %vitb8_1xm2 : tensor<32x197x192xf32>
    %vitb8_1dx = stablehlo.multiply %vitb8_1istdb, %vitb8_1t2 : tensor<32x197x192xf32>
    %vitb8_dx = stablehlo.add %vitb8_dr1, %vitb8_1dx : tensor<32x197x192xf32>
    %vitb7_pda1 = stablehlo.dot_general %vitb8_dx, %Wfc2_7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb7_pdWfc2 = stablehlo.dot_general %vitb7_pga, %vitb8_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb7_pdbfc2 = stablehlo.reduce(%vitb8_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_pgbbx2 = stablehlo.multiply %vitb7_ph1, %vitb7_ph1 : tensor<32x197x768xf32>
    %vitb7_pgbbx3 = stablehlo.multiply %vitb7_pgbbx2, %vitb7_ph1 : tensor<32x197x768xf32>
    %vitb7_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb7_pgbbkx3 = stablehlo.multiply %vitb7_pgbbck, %vitb7_pgbbx3 : tensor<32x197x768xf32>
    %vitb7_pgbbinn = stablehlo.add %vitb7_ph1, %vitb7_pgbbkx3 : tensor<32x197x768xf32>
    %vitb7_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb7_pgbbu = stablehlo.multiply %vitb7_pgbbcsqrt, %vitb7_pgbbinn : tensor<32x197x768xf32>
    %vitb7_pgbbt = stablehlo.tanh %vitb7_pgbbu : tensor<32x197x768xf32>
    %vitb7_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb7_pgbbopt = stablehlo.add %vitb7_pgbbone, %vitb7_pgbbt : tensor<32x197x768xf32>
    %vitb7_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb7_pgbbterm1 = stablehlo.multiply %vitb7_pgbbchalf, %vitb7_pgbbopt : tensor<32x197x768xf32>
    %vitb7_pgbbt2 = stablehlo.multiply %vitb7_pgbbt, %vitb7_pgbbt : tensor<32x197x768xf32>
    %vitb7_pgbbomt2 = stablehlo.subtract %vitb7_pgbbone, %vitb7_pgbbt2 : tensor<32x197x768xf32>
    %vitb7_pgbbhx = stablehlo.multiply %vitb7_pgbbchalf, %vitb7_ph1 : tensor<32x197x768xf32>
    %vitb7_pgbbhxo = stablehlo.multiply %vitb7_pgbbhx, %vitb7_pgbbomt2 : tensor<32x197x768xf32>
    %vitb7_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb7_pgbba3x2 = stablehlo.multiply %vitb7_pgbbc3b, %vitb7_pgbbx2 : tensor<32x197x768xf32>
    %vitb7_pgbbin2 = stablehlo.add %vitb7_pgbbone, %vitb7_pgbba3x2 : tensor<32x197x768xf32>
    %vitb7_pgbbup = stablehlo.multiply %vitb7_pgbbcsqrt, %vitb7_pgbbin2 : tensor<32x197x768xf32>
    %vitb7_pgbbterm2 = stablehlo.multiply %vitb7_pgbbhxo, %vitb7_pgbbup : tensor<32x197x768xf32>
    %vitb7_pgbbgp = stablehlo.add %vitb7_pgbbterm1, %vitb7_pgbbterm2 : tensor<32x197x768xf32>
    %vitb7_pgbdx = stablehlo.multiply %vitb7_pda1, %vitb7_pgbbgp : tensor<32x197x768xf32>
    %vitb7_pdx = stablehlo.dot_general %vitb7_pgbdx, %Wfc1_7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb7_pdWfc1 = stablehlo.dot_general %vitb7_2y, %vitb7_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb7_pdbfc1 = stablehlo.reduce(%vitb7_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb7_2gbk = stablehlo.broadcast_in_dim %g2_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_2dxhat = stablehlo.multiply %vitb7_pdx, %vitb7_2gbk : tensor<32x197x192xf32>
    %vitb7_2dgpre = stablehlo.multiply %vitb7_pdx, %vitb7_2xhat : tensor<32x197x192xf32>
    %vitb7_2dg = stablehlo.reduce(%vitb7_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_2db = stablehlo.reduce(%vitb7_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_2m1s = stablehlo.reduce(%vitb7_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_2m1 = stablehlo.divide %vitb7_2m1s, %vitb7_2nf : tensor<32x197xf32>
    %vitb7_2dxxh = stablehlo.multiply %vitb7_2dxhat, %vitb7_2xhat : tensor<32x197x192xf32>
    %vitb7_2m2s = stablehlo.reduce(%vitb7_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_2m2 = stablehlo.divide %vitb7_2m2s, %vitb7_2nf : tensor<32x197xf32>
    %vitb7_2m1b = stablehlo.broadcast_in_dim %vitb7_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_2m2b = stablehlo.broadcast_in_dim %vitb7_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_2t1 = stablehlo.subtract %vitb7_2dxhat, %vitb7_2m1b : tensor<32x197x192xf32>
    %vitb7_2xm2 = stablehlo.multiply %vitb7_2xhat, %vitb7_2m2b : tensor<32x197x192xf32>
    %vitb7_2t2 = stablehlo.subtract %vitb7_2t1, %vitb7_2xm2 : tensor<32x197x192xf32>
    %vitb7_2dx = stablehlo.multiply %vitb7_2istdb, %vitb7_2t2 : tensor<32x197x192xf32>
    %vitb7_dr1 = stablehlo.add %vitb8_dx, %vitb7_2dx : tensor<32x197x192xf32>
    %vitb7_mdP = stablehlo.dot_general %vitb7_dr1, %Wo_7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mdWo = stablehlo.dot_general %vitb7_mP, %vitb7_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb7_mdbo = stablehlo.reduce(%vitb7_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_mdPr = stablehlo.reshape %vitb7_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mdA = stablehlo.transpose %vitb7_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mdW = stablehlo.dot_general %vitb7_mdA, %vitb7_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb7_mdVh = stablehlo.dot_general %vitb7_mW, %vitb7_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mpdw = stablehlo.multiply %vitb7_mW, %vitb7_mdW : tensor<32x3x197x197xf32>
    %vitb7_msrow = stablehlo.reduce(%vitb7_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb7_msrowb = stablehlo.broadcast_in_dim %vitb7_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb7_mdiff = stablehlo.subtract %vitb7_mdW, %vitb7_msrowb : tensor<32x3x197x197xf32>
    %vitb7_mdSs = stablehlo.multiply %vitb7_mW, %vitb7_mdiff : tensor<32x3x197x197xf32>
    %vitb7_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb7_mdS = stablehlo.multiply %vitb7_mdSs, %vitb7_msclb : tensor<32x3x197x197xf32>
    %vitb7_mdQh = stablehlo.dot_general %vitb7_mdS, %vitb7_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mdKh = stablehlo.dot_general %vitb7_mdS, %vitb7_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mdQT = stablehlo.transpose %vitb7_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mdQ = stablehlo.reshape %vitb7_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb7_mdKT = stablehlo.transpose %vitb7_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mdK = stablehlo.reshape %vitb7_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb7_mdVT = stablehlo.transpose %vitb7_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mdV = stablehlo.reshape %vitb7_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb7_mdxQ = stablehlo.dot_general %vitb7_mdQ, %Wq_7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mdWQ = stablehlo.dot_general %vitb7_1y, %vitb7_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb7_mdbQ = stablehlo.reduce(%vitb7_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_mdxK = stablehlo.dot_general %vitb7_mdK, %Wk_7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mdWK = stablehlo.dot_general %vitb7_1y, %vitb7_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb7_mdbK = stablehlo.reduce(%vitb7_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_mdxV = stablehlo.dot_general %vitb7_mdV, %Wv_7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mdWV = stablehlo.dot_general %vitb7_1y, %vitb7_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb7_mdbV = stablehlo.reduce(%vitb7_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_mdxa = stablehlo.add %vitb7_mdxQ, %vitb7_mdxK : tensor<32x197x192xf32>
    %vitb7_mdx = stablehlo.add %vitb7_mdxa, %vitb7_mdxV : tensor<32x197x192xf32>
    %vitb7_1gbk = stablehlo.broadcast_in_dim %g1_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_1dxhat = stablehlo.multiply %vitb7_mdx, %vitb7_1gbk : tensor<32x197x192xf32>
    %vitb7_1dgpre = stablehlo.multiply %vitb7_mdx, %vitb7_1xhat : tensor<32x197x192xf32>
    %vitb7_1dg = stablehlo.reduce(%vitb7_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_1db = stablehlo.reduce(%vitb7_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_1m1s = stablehlo.reduce(%vitb7_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_1m1 = stablehlo.divide %vitb7_1m1s, %vitb7_1nf : tensor<32x197xf32>
    %vitb7_1dxxh = stablehlo.multiply %vitb7_1dxhat, %vitb7_1xhat : tensor<32x197x192xf32>
    %vitb7_1m2s = stablehlo.reduce(%vitb7_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_1m2 = stablehlo.divide %vitb7_1m2s, %vitb7_1nf : tensor<32x197xf32>
    %vitb7_1m1b = stablehlo.broadcast_in_dim %vitb7_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_1m2b = stablehlo.broadcast_in_dim %vitb7_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_1t1 = stablehlo.subtract %vitb7_1dxhat, %vitb7_1m1b : tensor<32x197x192xf32>
    %vitb7_1xm2 = stablehlo.multiply %vitb7_1xhat, %vitb7_1m2b : tensor<32x197x192xf32>
    %vitb7_1t2 = stablehlo.subtract %vitb7_1t1, %vitb7_1xm2 : tensor<32x197x192xf32>
    %vitb7_1dx = stablehlo.multiply %vitb7_1istdb, %vitb7_1t2 : tensor<32x197x192xf32>
    %vitb7_dx = stablehlo.add %vitb7_dr1, %vitb7_1dx : tensor<32x197x192xf32>
    %vitb6_pda1 = stablehlo.dot_general %vitb7_dx, %Wfc2_6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb6_pdWfc2 = stablehlo.dot_general %vitb6_pga, %vitb7_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb6_pdbfc2 = stablehlo.reduce(%vitb7_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_pgbbx2 = stablehlo.multiply %vitb6_ph1, %vitb6_ph1 : tensor<32x197x768xf32>
    %vitb6_pgbbx3 = stablehlo.multiply %vitb6_pgbbx2, %vitb6_ph1 : tensor<32x197x768xf32>
    %vitb6_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb6_pgbbkx3 = stablehlo.multiply %vitb6_pgbbck, %vitb6_pgbbx3 : tensor<32x197x768xf32>
    %vitb6_pgbbinn = stablehlo.add %vitb6_ph1, %vitb6_pgbbkx3 : tensor<32x197x768xf32>
    %vitb6_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb6_pgbbu = stablehlo.multiply %vitb6_pgbbcsqrt, %vitb6_pgbbinn : tensor<32x197x768xf32>
    %vitb6_pgbbt = stablehlo.tanh %vitb6_pgbbu : tensor<32x197x768xf32>
    %vitb6_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb6_pgbbopt = stablehlo.add %vitb6_pgbbone, %vitb6_pgbbt : tensor<32x197x768xf32>
    %vitb6_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb6_pgbbterm1 = stablehlo.multiply %vitb6_pgbbchalf, %vitb6_pgbbopt : tensor<32x197x768xf32>
    %vitb6_pgbbt2 = stablehlo.multiply %vitb6_pgbbt, %vitb6_pgbbt : tensor<32x197x768xf32>
    %vitb6_pgbbomt2 = stablehlo.subtract %vitb6_pgbbone, %vitb6_pgbbt2 : tensor<32x197x768xf32>
    %vitb6_pgbbhx = stablehlo.multiply %vitb6_pgbbchalf, %vitb6_ph1 : tensor<32x197x768xf32>
    %vitb6_pgbbhxo = stablehlo.multiply %vitb6_pgbbhx, %vitb6_pgbbomt2 : tensor<32x197x768xf32>
    %vitb6_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb6_pgbba3x2 = stablehlo.multiply %vitb6_pgbbc3b, %vitb6_pgbbx2 : tensor<32x197x768xf32>
    %vitb6_pgbbin2 = stablehlo.add %vitb6_pgbbone, %vitb6_pgbba3x2 : tensor<32x197x768xf32>
    %vitb6_pgbbup = stablehlo.multiply %vitb6_pgbbcsqrt, %vitb6_pgbbin2 : tensor<32x197x768xf32>
    %vitb6_pgbbterm2 = stablehlo.multiply %vitb6_pgbbhxo, %vitb6_pgbbup : tensor<32x197x768xf32>
    %vitb6_pgbbgp = stablehlo.add %vitb6_pgbbterm1, %vitb6_pgbbterm2 : tensor<32x197x768xf32>
    %vitb6_pgbdx = stablehlo.multiply %vitb6_pda1, %vitb6_pgbbgp : tensor<32x197x768xf32>
    %vitb6_pdx = stablehlo.dot_general %vitb6_pgbdx, %Wfc1_6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb6_pdWfc1 = stablehlo.dot_general %vitb6_2y, %vitb6_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb6_pdbfc1 = stablehlo.reduce(%vitb6_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb6_2gbk = stablehlo.broadcast_in_dim %g2_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_2dxhat = stablehlo.multiply %vitb6_pdx, %vitb6_2gbk : tensor<32x197x192xf32>
    %vitb6_2dgpre = stablehlo.multiply %vitb6_pdx, %vitb6_2xhat : tensor<32x197x192xf32>
    %vitb6_2dg = stablehlo.reduce(%vitb6_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_2db = stablehlo.reduce(%vitb6_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_2m1s = stablehlo.reduce(%vitb6_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_2m1 = stablehlo.divide %vitb6_2m1s, %vitb6_2nf : tensor<32x197xf32>
    %vitb6_2dxxh = stablehlo.multiply %vitb6_2dxhat, %vitb6_2xhat : tensor<32x197x192xf32>
    %vitb6_2m2s = stablehlo.reduce(%vitb6_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_2m2 = stablehlo.divide %vitb6_2m2s, %vitb6_2nf : tensor<32x197xf32>
    %vitb6_2m1b = stablehlo.broadcast_in_dim %vitb6_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_2m2b = stablehlo.broadcast_in_dim %vitb6_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_2t1 = stablehlo.subtract %vitb6_2dxhat, %vitb6_2m1b : tensor<32x197x192xf32>
    %vitb6_2xm2 = stablehlo.multiply %vitb6_2xhat, %vitb6_2m2b : tensor<32x197x192xf32>
    %vitb6_2t2 = stablehlo.subtract %vitb6_2t1, %vitb6_2xm2 : tensor<32x197x192xf32>
    %vitb6_2dx = stablehlo.multiply %vitb6_2istdb, %vitb6_2t2 : tensor<32x197x192xf32>
    %vitb6_dr1 = stablehlo.add %vitb7_dx, %vitb6_2dx : tensor<32x197x192xf32>
    %vitb6_mdP = stablehlo.dot_general %vitb6_dr1, %Wo_6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mdWo = stablehlo.dot_general %vitb6_mP, %vitb6_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb6_mdbo = stablehlo.reduce(%vitb6_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_mdPr = stablehlo.reshape %vitb6_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mdA = stablehlo.transpose %vitb6_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mdW = stablehlo.dot_general %vitb6_mdA, %vitb6_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb6_mdVh = stablehlo.dot_general %vitb6_mW, %vitb6_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mpdw = stablehlo.multiply %vitb6_mW, %vitb6_mdW : tensor<32x3x197x197xf32>
    %vitb6_msrow = stablehlo.reduce(%vitb6_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb6_msrowb = stablehlo.broadcast_in_dim %vitb6_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb6_mdiff = stablehlo.subtract %vitb6_mdW, %vitb6_msrowb : tensor<32x3x197x197xf32>
    %vitb6_mdSs = stablehlo.multiply %vitb6_mW, %vitb6_mdiff : tensor<32x3x197x197xf32>
    %vitb6_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb6_mdS = stablehlo.multiply %vitb6_mdSs, %vitb6_msclb : tensor<32x3x197x197xf32>
    %vitb6_mdQh = stablehlo.dot_general %vitb6_mdS, %vitb6_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mdKh = stablehlo.dot_general %vitb6_mdS, %vitb6_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mdQT = stablehlo.transpose %vitb6_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mdQ = stablehlo.reshape %vitb6_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb6_mdKT = stablehlo.transpose %vitb6_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mdK = stablehlo.reshape %vitb6_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb6_mdVT = stablehlo.transpose %vitb6_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mdV = stablehlo.reshape %vitb6_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb6_mdxQ = stablehlo.dot_general %vitb6_mdQ, %Wq_6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mdWQ = stablehlo.dot_general %vitb6_1y, %vitb6_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb6_mdbQ = stablehlo.reduce(%vitb6_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_mdxK = stablehlo.dot_general %vitb6_mdK, %Wk_6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mdWK = stablehlo.dot_general %vitb6_1y, %vitb6_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb6_mdbK = stablehlo.reduce(%vitb6_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_mdxV = stablehlo.dot_general %vitb6_mdV, %Wv_6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mdWV = stablehlo.dot_general %vitb6_1y, %vitb6_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb6_mdbV = stablehlo.reduce(%vitb6_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_mdxa = stablehlo.add %vitb6_mdxQ, %vitb6_mdxK : tensor<32x197x192xf32>
    %vitb6_mdx = stablehlo.add %vitb6_mdxa, %vitb6_mdxV : tensor<32x197x192xf32>
    %vitb6_1gbk = stablehlo.broadcast_in_dim %g1_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_1dxhat = stablehlo.multiply %vitb6_mdx, %vitb6_1gbk : tensor<32x197x192xf32>
    %vitb6_1dgpre = stablehlo.multiply %vitb6_mdx, %vitb6_1xhat : tensor<32x197x192xf32>
    %vitb6_1dg = stablehlo.reduce(%vitb6_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_1db = stablehlo.reduce(%vitb6_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_1m1s = stablehlo.reduce(%vitb6_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_1m1 = stablehlo.divide %vitb6_1m1s, %vitb6_1nf : tensor<32x197xf32>
    %vitb6_1dxxh = stablehlo.multiply %vitb6_1dxhat, %vitb6_1xhat : tensor<32x197x192xf32>
    %vitb6_1m2s = stablehlo.reduce(%vitb6_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_1m2 = stablehlo.divide %vitb6_1m2s, %vitb6_1nf : tensor<32x197xf32>
    %vitb6_1m1b = stablehlo.broadcast_in_dim %vitb6_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_1m2b = stablehlo.broadcast_in_dim %vitb6_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_1t1 = stablehlo.subtract %vitb6_1dxhat, %vitb6_1m1b : tensor<32x197x192xf32>
    %vitb6_1xm2 = stablehlo.multiply %vitb6_1xhat, %vitb6_1m2b : tensor<32x197x192xf32>
    %vitb6_1t2 = stablehlo.subtract %vitb6_1t1, %vitb6_1xm2 : tensor<32x197x192xf32>
    %vitb6_1dx = stablehlo.multiply %vitb6_1istdb, %vitb6_1t2 : tensor<32x197x192xf32>
    %vitb6_dx = stablehlo.add %vitb6_dr1, %vitb6_1dx : tensor<32x197x192xf32>
    %vitb5_pda1 = stablehlo.dot_general %vitb6_dx, %Wfc2_5, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb5_pdWfc2 = stablehlo.dot_general %vitb5_pga, %vitb6_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb5_pdbfc2 = stablehlo.reduce(%vitb6_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_pgbbx2 = stablehlo.multiply %vitb5_ph1, %vitb5_ph1 : tensor<32x197x768xf32>
    %vitb5_pgbbx3 = stablehlo.multiply %vitb5_pgbbx2, %vitb5_ph1 : tensor<32x197x768xf32>
    %vitb5_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb5_pgbbkx3 = stablehlo.multiply %vitb5_pgbbck, %vitb5_pgbbx3 : tensor<32x197x768xf32>
    %vitb5_pgbbinn = stablehlo.add %vitb5_ph1, %vitb5_pgbbkx3 : tensor<32x197x768xf32>
    %vitb5_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb5_pgbbu = stablehlo.multiply %vitb5_pgbbcsqrt, %vitb5_pgbbinn : tensor<32x197x768xf32>
    %vitb5_pgbbt = stablehlo.tanh %vitb5_pgbbu : tensor<32x197x768xf32>
    %vitb5_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb5_pgbbopt = stablehlo.add %vitb5_pgbbone, %vitb5_pgbbt : tensor<32x197x768xf32>
    %vitb5_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb5_pgbbterm1 = stablehlo.multiply %vitb5_pgbbchalf, %vitb5_pgbbopt : tensor<32x197x768xf32>
    %vitb5_pgbbt2 = stablehlo.multiply %vitb5_pgbbt, %vitb5_pgbbt : tensor<32x197x768xf32>
    %vitb5_pgbbomt2 = stablehlo.subtract %vitb5_pgbbone, %vitb5_pgbbt2 : tensor<32x197x768xf32>
    %vitb5_pgbbhx = stablehlo.multiply %vitb5_pgbbchalf, %vitb5_ph1 : tensor<32x197x768xf32>
    %vitb5_pgbbhxo = stablehlo.multiply %vitb5_pgbbhx, %vitb5_pgbbomt2 : tensor<32x197x768xf32>
    %vitb5_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb5_pgbba3x2 = stablehlo.multiply %vitb5_pgbbc3b, %vitb5_pgbbx2 : tensor<32x197x768xf32>
    %vitb5_pgbbin2 = stablehlo.add %vitb5_pgbbone, %vitb5_pgbba3x2 : tensor<32x197x768xf32>
    %vitb5_pgbbup = stablehlo.multiply %vitb5_pgbbcsqrt, %vitb5_pgbbin2 : tensor<32x197x768xf32>
    %vitb5_pgbbterm2 = stablehlo.multiply %vitb5_pgbbhxo, %vitb5_pgbbup : tensor<32x197x768xf32>
    %vitb5_pgbbgp = stablehlo.add %vitb5_pgbbterm1, %vitb5_pgbbterm2 : tensor<32x197x768xf32>
    %vitb5_pgbdx = stablehlo.multiply %vitb5_pda1, %vitb5_pgbbgp : tensor<32x197x768xf32>
    %vitb5_pdx = stablehlo.dot_general %vitb5_pgbdx, %Wfc1_5, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb5_pdWfc1 = stablehlo.dot_general %vitb5_2y, %vitb5_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb5_pdbfc1 = stablehlo.reduce(%vitb5_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb5_2gbk = stablehlo.broadcast_in_dim %g2_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_2dxhat = stablehlo.multiply %vitb5_pdx, %vitb5_2gbk : tensor<32x197x192xf32>
    %vitb5_2dgpre = stablehlo.multiply %vitb5_pdx, %vitb5_2xhat : tensor<32x197x192xf32>
    %vitb5_2dg = stablehlo.reduce(%vitb5_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_2db = stablehlo.reduce(%vitb5_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_2m1s = stablehlo.reduce(%vitb5_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_2m1 = stablehlo.divide %vitb5_2m1s, %vitb5_2nf : tensor<32x197xf32>
    %vitb5_2dxxh = stablehlo.multiply %vitb5_2dxhat, %vitb5_2xhat : tensor<32x197x192xf32>
    %vitb5_2m2s = stablehlo.reduce(%vitb5_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_2m2 = stablehlo.divide %vitb5_2m2s, %vitb5_2nf : tensor<32x197xf32>
    %vitb5_2m1b = stablehlo.broadcast_in_dim %vitb5_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_2m2b = stablehlo.broadcast_in_dim %vitb5_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_2t1 = stablehlo.subtract %vitb5_2dxhat, %vitb5_2m1b : tensor<32x197x192xf32>
    %vitb5_2xm2 = stablehlo.multiply %vitb5_2xhat, %vitb5_2m2b : tensor<32x197x192xf32>
    %vitb5_2t2 = stablehlo.subtract %vitb5_2t1, %vitb5_2xm2 : tensor<32x197x192xf32>
    %vitb5_2dx = stablehlo.multiply %vitb5_2istdb, %vitb5_2t2 : tensor<32x197x192xf32>
    %vitb5_dr1 = stablehlo.add %vitb6_dx, %vitb5_2dx : tensor<32x197x192xf32>
    %vitb5_mdP = stablehlo.dot_general %vitb5_dr1, %Wo_5, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mdWo = stablehlo.dot_general %vitb5_mP, %vitb5_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb5_mdbo = stablehlo.reduce(%vitb5_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_mdPr = stablehlo.reshape %vitb5_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mdA = stablehlo.transpose %vitb5_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mdW = stablehlo.dot_general %vitb5_mdA, %vitb5_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb5_mdVh = stablehlo.dot_general %vitb5_mW, %vitb5_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mpdw = stablehlo.multiply %vitb5_mW, %vitb5_mdW : tensor<32x3x197x197xf32>
    %vitb5_msrow = stablehlo.reduce(%vitb5_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb5_msrowb = stablehlo.broadcast_in_dim %vitb5_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb5_mdiff = stablehlo.subtract %vitb5_mdW, %vitb5_msrowb : tensor<32x3x197x197xf32>
    %vitb5_mdSs = stablehlo.multiply %vitb5_mW, %vitb5_mdiff : tensor<32x3x197x197xf32>
    %vitb5_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb5_mdS = stablehlo.multiply %vitb5_mdSs, %vitb5_msclb : tensor<32x3x197x197xf32>
    %vitb5_mdQh = stablehlo.dot_general %vitb5_mdS, %vitb5_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mdKh = stablehlo.dot_general %vitb5_mdS, %vitb5_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mdQT = stablehlo.transpose %vitb5_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mdQ = stablehlo.reshape %vitb5_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb5_mdKT = stablehlo.transpose %vitb5_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mdK = stablehlo.reshape %vitb5_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb5_mdVT = stablehlo.transpose %vitb5_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mdV = stablehlo.reshape %vitb5_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb5_mdxQ = stablehlo.dot_general %vitb5_mdQ, %Wq_5, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mdWQ = stablehlo.dot_general %vitb5_1y, %vitb5_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb5_mdbQ = stablehlo.reduce(%vitb5_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_mdxK = stablehlo.dot_general %vitb5_mdK, %Wk_5, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mdWK = stablehlo.dot_general %vitb5_1y, %vitb5_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb5_mdbK = stablehlo.reduce(%vitb5_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_mdxV = stablehlo.dot_general %vitb5_mdV, %Wv_5, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mdWV = stablehlo.dot_general %vitb5_1y, %vitb5_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb5_mdbV = stablehlo.reduce(%vitb5_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_mdxa = stablehlo.add %vitb5_mdxQ, %vitb5_mdxK : tensor<32x197x192xf32>
    %vitb5_mdx = stablehlo.add %vitb5_mdxa, %vitb5_mdxV : tensor<32x197x192xf32>
    %vitb5_1gbk = stablehlo.broadcast_in_dim %g1_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_1dxhat = stablehlo.multiply %vitb5_mdx, %vitb5_1gbk : tensor<32x197x192xf32>
    %vitb5_1dgpre = stablehlo.multiply %vitb5_mdx, %vitb5_1xhat : tensor<32x197x192xf32>
    %vitb5_1dg = stablehlo.reduce(%vitb5_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_1db = stablehlo.reduce(%vitb5_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_1m1s = stablehlo.reduce(%vitb5_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_1m1 = stablehlo.divide %vitb5_1m1s, %vitb5_1nf : tensor<32x197xf32>
    %vitb5_1dxxh = stablehlo.multiply %vitb5_1dxhat, %vitb5_1xhat : tensor<32x197x192xf32>
    %vitb5_1m2s = stablehlo.reduce(%vitb5_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_1m2 = stablehlo.divide %vitb5_1m2s, %vitb5_1nf : tensor<32x197xf32>
    %vitb5_1m1b = stablehlo.broadcast_in_dim %vitb5_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_1m2b = stablehlo.broadcast_in_dim %vitb5_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_1t1 = stablehlo.subtract %vitb5_1dxhat, %vitb5_1m1b : tensor<32x197x192xf32>
    %vitb5_1xm2 = stablehlo.multiply %vitb5_1xhat, %vitb5_1m2b : tensor<32x197x192xf32>
    %vitb5_1t2 = stablehlo.subtract %vitb5_1t1, %vitb5_1xm2 : tensor<32x197x192xf32>
    %vitb5_1dx = stablehlo.multiply %vitb5_1istdb, %vitb5_1t2 : tensor<32x197x192xf32>
    %vitb5_dx = stablehlo.add %vitb5_dr1, %vitb5_1dx : tensor<32x197x192xf32>
    %vitb4_pda1 = stablehlo.dot_general %vitb5_dx, %Wfc2_4, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb4_pdWfc2 = stablehlo.dot_general %vitb4_pga, %vitb5_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb4_pdbfc2 = stablehlo.reduce(%vitb5_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_pgbbx2 = stablehlo.multiply %vitb4_ph1, %vitb4_ph1 : tensor<32x197x768xf32>
    %vitb4_pgbbx3 = stablehlo.multiply %vitb4_pgbbx2, %vitb4_ph1 : tensor<32x197x768xf32>
    %vitb4_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb4_pgbbkx3 = stablehlo.multiply %vitb4_pgbbck, %vitb4_pgbbx3 : tensor<32x197x768xf32>
    %vitb4_pgbbinn = stablehlo.add %vitb4_ph1, %vitb4_pgbbkx3 : tensor<32x197x768xf32>
    %vitb4_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb4_pgbbu = stablehlo.multiply %vitb4_pgbbcsqrt, %vitb4_pgbbinn : tensor<32x197x768xf32>
    %vitb4_pgbbt = stablehlo.tanh %vitb4_pgbbu : tensor<32x197x768xf32>
    %vitb4_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb4_pgbbopt = stablehlo.add %vitb4_pgbbone, %vitb4_pgbbt : tensor<32x197x768xf32>
    %vitb4_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb4_pgbbterm1 = stablehlo.multiply %vitb4_pgbbchalf, %vitb4_pgbbopt : tensor<32x197x768xf32>
    %vitb4_pgbbt2 = stablehlo.multiply %vitb4_pgbbt, %vitb4_pgbbt : tensor<32x197x768xf32>
    %vitb4_pgbbomt2 = stablehlo.subtract %vitb4_pgbbone, %vitb4_pgbbt2 : tensor<32x197x768xf32>
    %vitb4_pgbbhx = stablehlo.multiply %vitb4_pgbbchalf, %vitb4_ph1 : tensor<32x197x768xf32>
    %vitb4_pgbbhxo = stablehlo.multiply %vitb4_pgbbhx, %vitb4_pgbbomt2 : tensor<32x197x768xf32>
    %vitb4_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb4_pgbba3x2 = stablehlo.multiply %vitb4_pgbbc3b, %vitb4_pgbbx2 : tensor<32x197x768xf32>
    %vitb4_pgbbin2 = stablehlo.add %vitb4_pgbbone, %vitb4_pgbba3x2 : tensor<32x197x768xf32>
    %vitb4_pgbbup = stablehlo.multiply %vitb4_pgbbcsqrt, %vitb4_pgbbin2 : tensor<32x197x768xf32>
    %vitb4_pgbbterm2 = stablehlo.multiply %vitb4_pgbbhxo, %vitb4_pgbbup : tensor<32x197x768xf32>
    %vitb4_pgbbgp = stablehlo.add %vitb4_pgbbterm1, %vitb4_pgbbterm2 : tensor<32x197x768xf32>
    %vitb4_pgbdx = stablehlo.multiply %vitb4_pda1, %vitb4_pgbbgp : tensor<32x197x768xf32>
    %vitb4_pdx = stablehlo.dot_general %vitb4_pgbdx, %Wfc1_4, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb4_pdWfc1 = stablehlo.dot_general %vitb4_2y, %vitb4_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb4_pdbfc1 = stablehlo.reduce(%vitb4_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb4_2gbk = stablehlo.broadcast_in_dim %g2_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_2dxhat = stablehlo.multiply %vitb4_pdx, %vitb4_2gbk : tensor<32x197x192xf32>
    %vitb4_2dgpre = stablehlo.multiply %vitb4_pdx, %vitb4_2xhat : tensor<32x197x192xf32>
    %vitb4_2dg = stablehlo.reduce(%vitb4_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_2db = stablehlo.reduce(%vitb4_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_2m1s = stablehlo.reduce(%vitb4_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_2m1 = stablehlo.divide %vitb4_2m1s, %vitb4_2nf : tensor<32x197xf32>
    %vitb4_2dxxh = stablehlo.multiply %vitb4_2dxhat, %vitb4_2xhat : tensor<32x197x192xf32>
    %vitb4_2m2s = stablehlo.reduce(%vitb4_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_2m2 = stablehlo.divide %vitb4_2m2s, %vitb4_2nf : tensor<32x197xf32>
    %vitb4_2m1b = stablehlo.broadcast_in_dim %vitb4_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_2m2b = stablehlo.broadcast_in_dim %vitb4_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_2t1 = stablehlo.subtract %vitb4_2dxhat, %vitb4_2m1b : tensor<32x197x192xf32>
    %vitb4_2xm2 = stablehlo.multiply %vitb4_2xhat, %vitb4_2m2b : tensor<32x197x192xf32>
    %vitb4_2t2 = stablehlo.subtract %vitb4_2t1, %vitb4_2xm2 : tensor<32x197x192xf32>
    %vitb4_2dx = stablehlo.multiply %vitb4_2istdb, %vitb4_2t2 : tensor<32x197x192xf32>
    %vitb4_dr1 = stablehlo.add %vitb5_dx, %vitb4_2dx : tensor<32x197x192xf32>
    %vitb4_mdP = stablehlo.dot_general %vitb4_dr1, %Wo_4, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mdWo = stablehlo.dot_general %vitb4_mP, %vitb4_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb4_mdbo = stablehlo.reduce(%vitb4_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_mdPr = stablehlo.reshape %vitb4_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mdA = stablehlo.transpose %vitb4_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mdW = stablehlo.dot_general %vitb4_mdA, %vitb4_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb4_mdVh = stablehlo.dot_general %vitb4_mW, %vitb4_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mpdw = stablehlo.multiply %vitb4_mW, %vitb4_mdW : tensor<32x3x197x197xf32>
    %vitb4_msrow = stablehlo.reduce(%vitb4_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb4_msrowb = stablehlo.broadcast_in_dim %vitb4_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb4_mdiff = stablehlo.subtract %vitb4_mdW, %vitb4_msrowb : tensor<32x3x197x197xf32>
    %vitb4_mdSs = stablehlo.multiply %vitb4_mW, %vitb4_mdiff : tensor<32x3x197x197xf32>
    %vitb4_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb4_mdS = stablehlo.multiply %vitb4_mdSs, %vitb4_msclb : tensor<32x3x197x197xf32>
    %vitb4_mdQh = stablehlo.dot_general %vitb4_mdS, %vitb4_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mdKh = stablehlo.dot_general %vitb4_mdS, %vitb4_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mdQT = stablehlo.transpose %vitb4_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mdQ = stablehlo.reshape %vitb4_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb4_mdKT = stablehlo.transpose %vitb4_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mdK = stablehlo.reshape %vitb4_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb4_mdVT = stablehlo.transpose %vitb4_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mdV = stablehlo.reshape %vitb4_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb4_mdxQ = stablehlo.dot_general %vitb4_mdQ, %Wq_4, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mdWQ = stablehlo.dot_general %vitb4_1y, %vitb4_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb4_mdbQ = stablehlo.reduce(%vitb4_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_mdxK = stablehlo.dot_general %vitb4_mdK, %Wk_4, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mdWK = stablehlo.dot_general %vitb4_1y, %vitb4_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb4_mdbK = stablehlo.reduce(%vitb4_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_mdxV = stablehlo.dot_general %vitb4_mdV, %Wv_4, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mdWV = stablehlo.dot_general %vitb4_1y, %vitb4_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb4_mdbV = stablehlo.reduce(%vitb4_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_mdxa = stablehlo.add %vitb4_mdxQ, %vitb4_mdxK : tensor<32x197x192xf32>
    %vitb4_mdx = stablehlo.add %vitb4_mdxa, %vitb4_mdxV : tensor<32x197x192xf32>
    %vitb4_1gbk = stablehlo.broadcast_in_dim %g1_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_1dxhat = stablehlo.multiply %vitb4_mdx, %vitb4_1gbk : tensor<32x197x192xf32>
    %vitb4_1dgpre = stablehlo.multiply %vitb4_mdx, %vitb4_1xhat : tensor<32x197x192xf32>
    %vitb4_1dg = stablehlo.reduce(%vitb4_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_1db = stablehlo.reduce(%vitb4_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_1m1s = stablehlo.reduce(%vitb4_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_1m1 = stablehlo.divide %vitb4_1m1s, %vitb4_1nf : tensor<32x197xf32>
    %vitb4_1dxxh = stablehlo.multiply %vitb4_1dxhat, %vitb4_1xhat : tensor<32x197x192xf32>
    %vitb4_1m2s = stablehlo.reduce(%vitb4_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_1m2 = stablehlo.divide %vitb4_1m2s, %vitb4_1nf : tensor<32x197xf32>
    %vitb4_1m1b = stablehlo.broadcast_in_dim %vitb4_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_1m2b = stablehlo.broadcast_in_dim %vitb4_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_1t1 = stablehlo.subtract %vitb4_1dxhat, %vitb4_1m1b : tensor<32x197x192xf32>
    %vitb4_1xm2 = stablehlo.multiply %vitb4_1xhat, %vitb4_1m2b : tensor<32x197x192xf32>
    %vitb4_1t2 = stablehlo.subtract %vitb4_1t1, %vitb4_1xm2 : tensor<32x197x192xf32>
    %vitb4_1dx = stablehlo.multiply %vitb4_1istdb, %vitb4_1t2 : tensor<32x197x192xf32>
    %vitb4_dx = stablehlo.add %vitb4_dr1, %vitb4_1dx : tensor<32x197x192xf32>
    %vitb3_pda1 = stablehlo.dot_general %vitb4_dx, %Wfc2_3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb3_pdWfc2 = stablehlo.dot_general %vitb3_pga, %vitb4_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb3_pdbfc2 = stablehlo.reduce(%vitb4_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_pgbbx2 = stablehlo.multiply %vitb3_ph1, %vitb3_ph1 : tensor<32x197x768xf32>
    %vitb3_pgbbx3 = stablehlo.multiply %vitb3_pgbbx2, %vitb3_ph1 : tensor<32x197x768xf32>
    %vitb3_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb3_pgbbkx3 = stablehlo.multiply %vitb3_pgbbck, %vitb3_pgbbx3 : tensor<32x197x768xf32>
    %vitb3_pgbbinn = stablehlo.add %vitb3_ph1, %vitb3_pgbbkx3 : tensor<32x197x768xf32>
    %vitb3_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb3_pgbbu = stablehlo.multiply %vitb3_pgbbcsqrt, %vitb3_pgbbinn : tensor<32x197x768xf32>
    %vitb3_pgbbt = stablehlo.tanh %vitb3_pgbbu : tensor<32x197x768xf32>
    %vitb3_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb3_pgbbopt = stablehlo.add %vitb3_pgbbone, %vitb3_pgbbt : tensor<32x197x768xf32>
    %vitb3_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb3_pgbbterm1 = stablehlo.multiply %vitb3_pgbbchalf, %vitb3_pgbbopt : tensor<32x197x768xf32>
    %vitb3_pgbbt2 = stablehlo.multiply %vitb3_pgbbt, %vitb3_pgbbt : tensor<32x197x768xf32>
    %vitb3_pgbbomt2 = stablehlo.subtract %vitb3_pgbbone, %vitb3_pgbbt2 : tensor<32x197x768xf32>
    %vitb3_pgbbhx = stablehlo.multiply %vitb3_pgbbchalf, %vitb3_ph1 : tensor<32x197x768xf32>
    %vitb3_pgbbhxo = stablehlo.multiply %vitb3_pgbbhx, %vitb3_pgbbomt2 : tensor<32x197x768xf32>
    %vitb3_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb3_pgbba3x2 = stablehlo.multiply %vitb3_pgbbc3b, %vitb3_pgbbx2 : tensor<32x197x768xf32>
    %vitb3_pgbbin2 = stablehlo.add %vitb3_pgbbone, %vitb3_pgbba3x2 : tensor<32x197x768xf32>
    %vitb3_pgbbup = stablehlo.multiply %vitb3_pgbbcsqrt, %vitb3_pgbbin2 : tensor<32x197x768xf32>
    %vitb3_pgbbterm2 = stablehlo.multiply %vitb3_pgbbhxo, %vitb3_pgbbup : tensor<32x197x768xf32>
    %vitb3_pgbbgp = stablehlo.add %vitb3_pgbbterm1, %vitb3_pgbbterm2 : tensor<32x197x768xf32>
    %vitb3_pgbdx = stablehlo.multiply %vitb3_pda1, %vitb3_pgbbgp : tensor<32x197x768xf32>
    %vitb3_pdx = stablehlo.dot_general %vitb3_pgbdx, %Wfc1_3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb3_pdWfc1 = stablehlo.dot_general %vitb3_2y, %vitb3_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb3_pdbfc1 = stablehlo.reduce(%vitb3_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb3_2gbk = stablehlo.broadcast_in_dim %g2_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_2dxhat = stablehlo.multiply %vitb3_pdx, %vitb3_2gbk : tensor<32x197x192xf32>
    %vitb3_2dgpre = stablehlo.multiply %vitb3_pdx, %vitb3_2xhat : tensor<32x197x192xf32>
    %vitb3_2dg = stablehlo.reduce(%vitb3_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_2db = stablehlo.reduce(%vitb3_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_2m1s = stablehlo.reduce(%vitb3_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_2m1 = stablehlo.divide %vitb3_2m1s, %vitb3_2nf : tensor<32x197xf32>
    %vitb3_2dxxh = stablehlo.multiply %vitb3_2dxhat, %vitb3_2xhat : tensor<32x197x192xf32>
    %vitb3_2m2s = stablehlo.reduce(%vitb3_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_2m2 = stablehlo.divide %vitb3_2m2s, %vitb3_2nf : tensor<32x197xf32>
    %vitb3_2m1b = stablehlo.broadcast_in_dim %vitb3_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_2m2b = stablehlo.broadcast_in_dim %vitb3_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_2t1 = stablehlo.subtract %vitb3_2dxhat, %vitb3_2m1b : tensor<32x197x192xf32>
    %vitb3_2xm2 = stablehlo.multiply %vitb3_2xhat, %vitb3_2m2b : tensor<32x197x192xf32>
    %vitb3_2t2 = stablehlo.subtract %vitb3_2t1, %vitb3_2xm2 : tensor<32x197x192xf32>
    %vitb3_2dx = stablehlo.multiply %vitb3_2istdb, %vitb3_2t2 : tensor<32x197x192xf32>
    %vitb3_dr1 = stablehlo.add %vitb4_dx, %vitb3_2dx : tensor<32x197x192xf32>
    %vitb3_mdP = stablehlo.dot_general %vitb3_dr1, %Wo_3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mdWo = stablehlo.dot_general %vitb3_mP, %vitb3_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb3_mdbo = stablehlo.reduce(%vitb3_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_mdPr = stablehlo.reshape %vitb3_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mdA = stablehlo.transpose %vitb3_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mdW = stablehlo.dot_general %vitb3_mdA, %vitb3_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb3_mdVh = stablehlo.dot_general %vitb3_mW, %vitb3_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mpdw = stablehlo.multiply %vitb3_mW, %vitb3_mdW : tensor<32x3x197x197xf32>
    %vitb3_msrow = stablehlo.reduce(%vitb3_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb3_msrowb = stablehlo.broadcast_in_dim %vitb3_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb3_mdiff = stablehlo.subtract %vitb3_mdW, %vitb3_msrowb : tensor<32x3x197x197xf32>
    %vitb3_mdSs = stablehlo.multiply %vitb3_mW, %vitb3_mdiff : tensor<32x3x197x197xf32>
    %vitb3_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb3_mdS = stablehlo.multiply %vitb3_mdSs, %vitb3_msclb : tensor<32x3x197x197xf32>
    %vitb3_mdQh = stablehlo.dot_general %vitb3_mdS, %vitb3_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mdKh = stablehlo.dot_general %vitb3_mdS, %vitb3_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mdQT = stablehlo.transpose %vitb3_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mdQ = stablehlo.reshape %vitb3_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb3_mdKT = stablehlo.transpose %vitb3_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mdK = stablehlo.reshape %vitb3_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb3_mdVT = stablehlo.transpose %vitb3_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mdV = stablehlo.reshape %vitb3_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb3_mdxQ = stablehlo.dot_general %vitb3_mdQ, %Wq_3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mdWQ = stablehlo.dot_general %vitb3_1y, %vitb3_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb3_mdbQ = stablehlo.reduce(%vitb3_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_mdxK = stablehlo.dot_general %vitb3_mdK, %Wk_3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mdWK = stablehlo.dot_general %vitb3_1y, %vitb3_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb3_mdbK = stablehlo.reduce(%vitb3_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_mdxV = stablehlo.dot_general %vitb3_mdV, %Wv_3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mdWV = stablehlo.dot_general %vitb3_1y, %vitb3_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb3_mdbV = stablehlo.reduce(%vitb3_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_mdxa = stablehlo.add %vitb3_mdxQ, %vitb3_mdxK : tensor<32x197x192xf32>
    %vitb3_mdx = stablehlo.add %vitb3_mdxa, %vitb3_mdxV : tensor<32x197x192xf32>
    %vitb3_1gbk = stablehlo.broadcast_in_dim %g1_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_1dxhat = stablehlo.multiply %vitb3_mdx, %vitb3_1gbk : tensor<32x197x192xf32>
    %vitb3_1dgpre = stablehlo.multiply %vitb3_mdx, %vitb3_1xhat : tensor<32x197x192xf32>
    %vitb3_1dg = stablehlo.reduce(%vitb3_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_1db = stablehlo.reduce(%vitb3_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_1m1s = stablehlo.reduce(%vitb3_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_1m1 = stablehlo.divide %vitb3_1m1s, %vitb3_1nf : tensor<32x197xf32>
    %vitb3_1dxxh = stablehlo.multiply %vitb3_1dxhat, %vitb3_1xhat : tensor<32x197x192xf32>
    %vitb3_1m2s = stablehlo.reduce(%vitb3_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_1m2 = stablehlo.divide %vitb3_1m2s, %vitb3_1nf : tensor<32x197xf32>
    %vitb3_1m1b = stablehlo.broadcast_in_dim %vitb3_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_1m2b = stablehlo.broadcast_in_dim %vitb3_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_1t1 = stablehlo.subtract %vitb3_1dxhat, %vitb3_1m1b : tensor<32x197x192xf32>
    %vitb3_1xm2 = stablehlo.multiply %vitb3_1xhat, %vitb3_1m2b : tensor<32x197x192xf32>
    %vitb3_1t2 = stablehlo.subtract %vitb3_1t1, %vitb3_1xm2 : tensor<32x197x192xf32>
    %vitb3_1dx = stablehlo.multiply %vitb3_1istdb, %vitb3_1t2 : tensor<32x197x192xf32>
    %vitb3_dx = stablehlo.add %vitb3_dr1, %vitb3_1dx : tensor<32x197x192xf32>
    %vitb2_pda1 = stablehlo.dot_general %vitb3_dx, %Wfc2_2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb2_pdWfc2 = stablehlo.dot_general %vitb2_pga, %vitb3_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb2_pdbfc2 = stablehlo.reduce(%vitb3_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_pgbbx2 = stablehlo.multiply %vitb2_ph1, %vitb2_ph1 : tensor<32x197x768xf32>
    %vitb2_pgbbx3 = stablehlo.multiply %vitb2_pgbbx2, %vitb2_ph1 : tensor<32x197x768xf32>
    %vitb2_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb2_pgbbkx3 = stablehlo.multiply %vitb2_pgbbck, %vitb2_pgbbx3 : tensor<32x197x768xf32>
    %vitb2_pgbbinn = stablehlo.add %vitb2_ph1, %vitb2_pgbbkx3 : tensor<32x197x768xf32>
    %vitb2_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb2_pgbbu = stablehlo.multiply %vitb2_pgbbcsqrt, %vitb2_pgbbinn : tensor<32x197x768xf32>
    %vitb2_pgbbt = stablehlo.tanh %vitb2_pgbbu : tensor<32x197x768xf32>
    %vitb2_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb2_pgbbopt = stablehlo.add %vitb2_pgbbone, %vitb2_pgbbt : tensor<32x197x768xf32>
    %vitb2_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb2_pgbbterm1 = stablehlo.multiply %vitb2_pgbbchalf, %vitb2_pgbbopt : tensor<32x197x768xf32>
    %vitb2_pgbbt2 = stablehlo.multiply %vitb2_pgbbt, %vitb2_pgbbt : tensor<32x197x768xf32>
    %vitb2_pgbbomt2 = stablehlo.subtract %vitb2_pgbbone, %vitb2_pgbbt2 : tensor<32x197x768xf32>
    %vitb2_pgbbhx = stablehlo.multiply %vitb2_pgbbchalf, %vitb2_ph1 : tensor<32x197x768xf32>
    %vitb2_pgbbhxo = stablehlo.multiply %vitb2_pgbbhx, %vitb2_pgbbomt2 : tensor<32x197x768xf32>
    %vitb2_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb2_pgbba3x2 = stablehlo.multiply %vitb2_pgbbc3b, %vitb2_pgbbx2 : tensor<32x197x768xf32>
    %vitb2_pgbbin2 = stablehlo.add %vitb2_pgbbone, %vitb2_pgbba3x2 : tensor<32x197x768xf32>
    %vitb2_pgbbup = stablehlo.multiply %vitb2_pgbbcsqrt, %vitb2_pgbbin2 : tensor<32x197x768xf32>
    %vitb2_pgbbterm2 = stablehlo.multiply %vitb2_pgbbhxo, %vitb2_pgbbup : tensor<32x197x768xf32>
    %vitb2_pgbbgp = stablehlo.add %vitb2_pgbbterm1, %vitb2_pgbbterm2 : tensor<32x197x768xf32>
    %vitb2_pgbdx = stablehlo.multiply %vitb2_pda1, %vitb2_pgbbgp : tensor<32x197x768xf32>
    %vitb2_pdx = stablehlo.dot_general %vitb2_pgbdx, %Wfc1_2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb2_pdWfc1 = stablehlo.dot_general %vitb2_2y, %vitb2_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb2_pdbfc1 = stablehlo.reduce(%vitb2_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb2_2gbk = stablehlo.broadcast_in_dim %g2_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_2dxhat = stablehlo.multiply %vitb2_pdx, %vitb2_2gbk : tensor<32x197x192xf32>
    %vitb2_2dgpre = stablehlo.multiply %vitb2_pdx, %vitb2_2xhat : tensor<32x197x192xf32>
    %vitb2_2dg = stablehlo.reduce(%vitb2_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_2db = stablehlo.reduce(%vitb2_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_2m1s = stablehlo.reduce(%vitb2_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_2m1 = stablehlo.divide %vitb2_2m1s, %vitb2_2nf : tensor<32x197xf32>
    %vitb2_2dxxh = stablehlo.multiply %vitb2_2dxhat, %vitb2_2xhat : tensor<32x197x192xf32>
    %vitb2_2m2s = stablehlo.reduce(%vitb2_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_2m2 = stablehlo.divide %vitb2_2m2s, %vitb2_2nf : tensor<32x197xf32>
    %vitb2_2m1b = stablehlo.broadcast_in_dim %vitb2_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_2m2b = stablehlo.broadcast_in_dim %vitb2_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_2t1 = stablehlo.subtract %vitb2_2dxhat, %vitb2_2m1b : tensor<32x197x192xf32>
    %vitb2_2xm2 = stablehlo.multiply %vitb2_2xhat, %vitb2_2m2b : tensor<32x197x192xf32>
    %vitb2_2t2 = stablehlo.subtract %vitb2_2t1, %vitb2_2xm2 : tensor<32x197x192xf32>
    %vitb2_2dx = stablehlo.multiply %vitb2_2istdb, %vitb2_2t2 : tensor<32x197x192xf32>
    %vitb2_dr1 = stablehlo.add %vitb3_dx, %vitb2_2dx : tensor<32x197x192xf32>
    %vitb2_mdP = stablehlo.dot_general %vitb2_dr1, %Wo_2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mdWo = stablehlo.dot_general %vitb2_mP, %vitb2_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb2_mdbo = stablehlo.reduce(%vitb2_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_mdPr = stablehlo.reshape %vitb2_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mdA = stablehlo.transpose %vitb2_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mdW = stablehlo.dot_general %vitb2_mdA, %vitb2_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb2_mdVh = stablehlo.dot_general %vitb2_mW, %vitb2_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mpdw = stablehlo.multiply %vitb2_mW, %vitb2_mdW : tensor<32x3x197x197xf32>
    %vitb2_msrow = stablehlo.reduce(%vitb2_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb2_msrowb = stablehlo.broadcast_in_dim %vitb2_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb2_mdiff = stablehlo.subtract %vitb2_mdW, %vitb2_msrowb : tensor<32x3x197x197xf32>
    %vitb2_mdSs = stablehlo.multiply %vitb2_mW, %vitb2_mdiff : tensor<32x3x197x197xf32>
    %vitb2_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb2_mdS = stablehlo.multiply %vitb2_mdSs, %vitb2_msclb : tensor<32x3x197x197xf32>
    %vitb2_mdQh = stablehlo.dot_general %vitb2_mdS, %vitb2_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mdKh = stablehlo.dot_general %vitb2_mdS, %vitb2_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mdQT = stablehlo.transpose %vitb2_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mdQ = stablehlo.reshape %vitb2_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb2_mdKT = stablehlo.transpose %vitb2_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mdK = stablehlo.reshape %vitb2_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb2_mdVT = stablehlo.transpose %vitb2_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mdV = stablehlo.reshape %vitb2_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb2_mdxQ = stablehlo.dot_general %vitb2_mdQ, %Wq_2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mdWQ = stablehlo.dot_general %vitb2_1y, %vitb2_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb2_mdbQ = stablehlo.reduce(%vitb2_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_mdxK = stablehlo.dot_general %vitb2_mdK, %Wk_2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mdWK = stablehlo.dot_general %vitb2_1y, %vitb2_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb2_mdbK = stablehlo.reduce(%vitb2_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_mdxV = stablehlo.dot_general %vitb2_mdV, %Wv_2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mdWV = stablehlo.dot_general %vitb2_1y, %vitb2_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb2_mdbV = stablehlo.reduce(%vitb2_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_mdxa = stablehlo.add %vitb2_mdxQ, %vitb2_mdxK : tensor<32x197x192xf32>
    %vitb2_mdx = stablehlo.add %vitb2_mdxa, %vitb2_mdxV : tensor<32x197x192xf32>
    %vitb2_1gbk = stablehlo.broadcast_in_dim %g1_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_1dxhat = stablehlo.multiply %vitb2_mdx, %vitb2_1gbk : tensor<32x197x192xf32>
    %vitb2_1dgpre = stablehlo.multiply %vitb2_mdx, %vitb2_1xhat : tensor<32x197x192xf32>
    %vitb2_1dg = stablehlo.reduce(%vitb2_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_1db = stablehlo.reduce(%vitb2_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_1m1s = stablehlo.reduce(%vitb2_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_1m1 = stablehlo.divide %vitb2_1m1s, %vitb2_1nf : tensor<32x197xf32>
    %vitb2_1dxxh = stablehlo.multiply %vitb2_1dxhat, %vitb2_1xhat : tensor<32x197x192xf32>
    %vitb2_1m2s = stablehlo.reduce(%vitb2_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_1m2 = stablehlo.divide %vitb2_1m2s, %vitb2_1nf : tensor<32x197xf32>
    %vitb2_1m1b = stablehlo.broadcast_in_dim %vitb2_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_1m2b = stablehlo.broadcast_in_dim %vitb2_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_1t1 = stablehlo.subtract %vitb2_1dxhat, %vitb2_1m1b : tensor<32x197x192xf32>
    %vitb2_1xm2 = stablehlo.multiply %vitb2_1xhat, %vitb2_1m2b : tensor<32x197x192xf32>
    %vitb2_1t2 = stablehlo.subtract %vitb2_1t1, %vitb2_1xm2 : tensor<32x197x192xf32>
    %vitb2_1dx = stablehlo.multiply %vitb2_1istdb, %vitb2_1t2 : tensor<32x197x192xf32>
    %vitb2_dx = stablehlo.add %vitb2_dr1, %vitb2_1dx : tensor<32x197x192xf32>
    %vitb1_pda1 = stablehlo.dot_general %vitb2_dx, %Wfc2_1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb1_pdWfc2 = stablehlo.dot_general %vitb1_pga, %vitb2_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb1_pdbfc2 = stablehlo.reduce(%vitb2_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_pgbbx2 = stablehlo.multiply %vitb1_ph1, %vitb1_ph1 : tensor<32x197x768xf32>
    %vitb1_pgbbx3 = stablehlo.multiply %vitb1_pgbbx2, %vitb1_ph1 : tensor<32x197x768xf32>
    %vitb1_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb1_pgbbkx3 = stablehlo.multiply %vitb1_pgbbck, %vitb1_pgbbx3 : tensor<32x197x768xf32>
    %vitb1_pgbbinn = stablehlo.add %vitb1_ph1, %vitb1_pgbbkx3 : tensor<32x197x768xf32>
    %vitb1_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb1_pgbbu = stablehlo.multiply %vitb1_pgbbcsqrt, %vitb1_pgbbinn : tensor<32x197x768xf32>
    %vitb1_pgbbt = stablehlo.tanh %vitb1_pgbbu : tensor<32x197x768xf32>
    %vitb1_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb1_pgbbopt = stablehlo.add %vitb1_pgbbone, %vitb1_pgbbt : tensor<32x197x768xf32>
    %vitb1_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb1_pgbbterm1 = stablehlo.multiply %vitb1_pgbbchalf, %vitb1_pgbbopt : tensor<32x197x768xf32>
    %vitb1_pgbbt2 = stablehlo.multiply %vitb1_pgbbt, %vitb1_pgbbt : tensor<32x197x768xf32>
    %vitb1_pgbbomt2 = stablehlo.subtract %vitb1_pgbbone, %vitb1_pgbbt2 : tensor<32x197x768xf32>
    %vitb1_pgbbhx = stablehlo.multiply %vitb1_pgbbchalf, %vitb1_ph1 : tensor<32x197x768xf32>
    %vitb1_pgbbhxo = stablehlo.multiply %vitb1_pgbbhx, %vitb1_pgbbomt2 : tensor<32x197x768xf32>
    %vitb1_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb1_pgbba3x2 = stablehlo.multiply %vitb1_pgbbc3b, %vitb1_pgbbx2 : tensor<32x197x768xf32>
    %vitb1_pgbbin2 = stablehlo.add %vitb1_pgbbone, %vitb1_pgbba3x2 : tensor<32x197x768xf32>
    %vitb1_pgbbup = stablehlo.multiply %vitb1_pgbbcsqrt, %vitb1_pgbbin2 : tensor<32x197x768xf32>
    %vitb1_pgbbterm2 = stablehlo.multiply %vitb1_pgbbhxo, %vitb1_pgbbup : tensor<32x197x768xf32>
    %vitb1_pgbbgp = stablehlo.add %vitb1_pgbbterm1, %vitb1_pgbbterm2 : tensor<32x197x768xf32>
    %vitb1_pgbdx = stablehlo.multiply %vitb1_pda1, %vitb1_pgbbgp : tensor<32x197x768xf32>
    %vitb1_pdx = stablehlo.dot_general %vitb1_pgbdx, %Wfc1_1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb1_pdWfc1 = stablehlo.dot_general %vitb1_2y, %vitb1_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb1_pdbfc1 = stablehlo.reduce(%vitb1_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb1_2gbk = stablehlo.broadcast_in_dim %g2_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_2dxhat = stablehlo.multiply %vitb1_pdx, %vitb1_2gbk : tensor<32x197x192xf32>
    %vitb1_2dgpre = stablehlo.multiply %vitb1_pdx, %vitb1_2xhat : tensor<32x197x192xf32>
    %vitb1_2dg = stablehlo.reduce(%vitb1_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_2db = stablehlo.reduce(%vitb1_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_2m1s = stablehlo.reduce(%vitb1_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_2m1 = stablehlo.divide %vitb1_2m1s, %vitb1_2nf : tensor<32x197xf32>
    %vitb1_2dxxh = stablehlo.multiply %vitb1_2dxhat, %vitb1_2xhat : tensor<32x197x192xf32>
    %vitb1_2m2s = stablehlo.reduce(%vitb1_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_2m2 = stablehlo.divide %vitb1_2m2s, %vitb1_2nf : tensor<32x197xf32>
    %vitb1_2m1b = stablehlo.broadcast_in_dim %vitb1_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_2m2b = stablehlo.broadcast_in_dim %vitb1_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_2t1 = stablehlo.subtract %vitb1_2dxhat, %vitb1_2m1b : tensor<32x197x192xf32>
    %vitb1_2xm2 = stablehlo.multiply %vitb1_2xhat, %vitb1_2m2b : tensor<32x197x192xf32>
    %vitb1_2t2 = stablehlo.subtract %vitb1_2t1, %vitb1_2xm2 : tensor<32x197x192xf32>
    %vitb1_2dx = stablehlo.multiply %vitb1_2istdb, %vitb1_2t2 : tensor<32x197x192xf32>
    %vitb1_dr1 = stablehlo.add %vitb2_dx, %vitb1_2dx : tensor<32x197x192xf32>
    %vitb1_mdP = stablehlo.dot_general %vitb1_dr1, %Wo_1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mdWo = stablehlo.dot_general %vitb1_mP, %vitb1_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb1_mdbo = stablehlo.reduce(%vitb1_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_mdPr = stablehlo.reshape %vitb1_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mdA = stablehlo.transpose %vitb1_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mdW = stablehlo.dot_general %vitb1_mdA, %vitb1_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb1_mdVh = stablehlo.dot_general %vitb1_mW, %vitb1_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mpdw = stablehlo.multiply %vitb1_mW, %vitb1_mdW : tensor<32x3x197x197xf32>
    %vitb1_msrow = stablehlo.reduce(%vitb1_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb1_msrowb = stablehlo.broadcast_in_dim %vitb1_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb1_mdiff = stablehlo.subtract %vitb1_mdW, %vitb1_msrowb : tensor<32x3x197x197xf32>
    %vitb1_mdSs = stablehlo.multiply %vitb1_mW, %vitb1_mdiff : tensor<32x3x197x197xf32>
    %vitb1_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb1_mdS = stablehlo.multiply %vitb1_mdSs, %vitb1_msclb : tensor<32x3x197x197xf32>
    %vitb1_mdQh = stablehlo.dot_general %vitb1_mdS, %vitb1_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mdKh = stablehlo.dot_general %vitb1_mdS, %vitb1_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mdQT = stablehlo.transpose %vitb1_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mdQ = stablehlo.reshape %vitb1_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb1_mdKT = stablehlo.transpose %vitb1_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mdK = stablehlo.reshape %vitb1_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb1_mdVT = stablehlo.transpose %vitb1_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mdV = stablehlo.reshape %vitb1_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb1_mdxQ = stablehlo.dot_general %vitb1_mdQ, %Wq_1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mdWQ = stablehlo.dot_general %vitb1_1y, %vitb1_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb1_mdbQ = stablehlo.reduce(%vitb1_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_mdxK = stablehlo.dot_general %vitb1_mdK, %Wk_1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mdWK = stablehlo.dot_general %vitb1_1y, %vitb1_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb1_mdbK = stablehlo.reduce(%vitb1_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_mdxV = stablehlo.dot_general %vitb1_mdV, %Wv_1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mdWV = stablehlo.dot_general %vitb1_1y, %vitb1_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb1_mdbV = stablehlo.reduce(%vitb1_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_mdxa = stablehlo.add %vitb1_mdxQ, %vitb1_mdxK : tensor<32x197x192xf32>
    %vitb1_mdx = stablehlo.add %vitb1_mdxa, %vitb1_mdxV : tensor<32x197x192xf32>
    %vitb1_1gbk = stablehlo.broadcast_in_dim %g1_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_1dxhat = stablehlo.multiply %vitb1_mdx, %vitb1_1gbk : tensor<32x197x192xf32>
    %vitb1_1dgpre = stablehlo.multiply %vitb1_mdx, %vitb1_1xhat : tensor<32x197x192xf32>
    %vitb1_1dg = stablehlo.reduce(%vitb1_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_1db = stablehlo.reduce(%vitb1_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_1m1s = stablehlo.reduce(%vitb1_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_1m1 = stablehlo.divide %vitb1_1m1s, %vitb1_1nf : tensor<32x197xf32>
    %vitb1_1dxxh = stablehlo.multiply %vitb1_1dxhat, %vitb1_1xhat : tensor<32x197x192xf32>
    %vitb1_1m2s = stablehlo.reduce(%vitb1_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_1m2 = stablehlo.divide %vitb1_1m2s, %vitb1_1nf : tensor<32x197xf32>
    %vitb1_1m1b = stablehlo.broadcast_in_dim %vitb1_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_1m2b = stablehlo.broadcast_in_dim %vitb1_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_1t1 = stablehlo.subtract %vitb1_1dxhat, %vitb1_1m1b : tensor<32x197x192xf32>
    %vitb1_1xm2 = stablehlo.multiply %vitb1_1xhat, %vitb1_1m2b : tensor<32x197x192xf32>
    %vitb1_1t2 = stablehlo.subtract %vitb1_1t1, %vitb1_1xm2 : tensor<32x197x192xf32>
    %vitb1_1dx = stablehlo.multiply %vitb1_1istdb, %vitb1_1t2 : tensor<32x197x192xf32>
    %vitb1_dx = stablehlo.add %vitb1_dr1, %vitb1_1dx : tensor<32x197x192xf32>
    %vitb0_pda1 = stablehlo.dot_general %vitb1_dx, %Wfc2_0, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb0_pdWfc2 = stablehlo.dot_general %vitb0_pga, %vitb1_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb0_pdbfc2 = stablehlo.reduce(%vitb1_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_pgbbx2 = stablehlo.multiply %vitb0_ph1, %vitb0_ph1 : tensor<32x197x768xf32>
    %vitb0_pgbbx3 = stablehlo.multiply %vitb0_pgbbx2, %vitb0_ph1 : tensor<32x197x768xf32>
    %vitb0_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb0_pgbbkx3 = stablehlo.multiply %vitb0_pgbbck, %vitb0_pgbbx3 : tensor<32x197x768xf32>
    %vitb0_pgbbinn = stablehlo.add %vitb0_ph1, %vitb0_pgbbkx3 : tensor<32x197x768xf32>
    %vitb0_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb0_pgbbu = stablehlo.multiply %vitb0_pgbbcsqrt, %vitb0_pgbbinn : tensor<32x197x768xf32>
    %vitb0_pgbbt = stablehlo.tanh %vitb0_pgbbu : tensor<32x197x768xf32>
    %vitb0_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb0_pgbbopt = stablehlo.add %vitb0_pgbbone, %vitb0_pgbbt : tensor<32x197x768xf32>
    %vitb0_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb0_pgbbterm1 = stablehlo.multiply %vitb0_pgbbchalf, %vitb0_pgbbopt : tensor<32x197x768xf32>
    %vitb0_pgbbt2 = stablehlo.multiply %vitb0_pgbbt, %vitb0_pgbbt : tensor<32x197x768xf32>
    %vitb0_pgbbomt2 = stablehlo.subtract %vitb0_pgbbone, %vitb0_pgbbt2 : tensor<32x197x768xf32>
    %vitb0_pgbbhx = stablehlo.multiply %vitb0_pgbbchalf, %vitb0_ph1 : tensor<32x197x768xf32>
    %vitb0_pgbbhxo = stablehlo.multiply %vitb0_pgbbhx, %vitb0_pgbbomt2 : tensor<32x197x768xf32>
    %vitb0_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb0_pgbba3x2 = stablehlo.multiply %vitb0_pgbbc3b, %vitb0_pgbbx2 : tensor<32x197x768xf32>
    %vitb0_pgbbin2 = stablehlo.add %vitb0_pgbbone, %vitb0_pgbba3x2 : tensor<32x197x768xf32>
    %vitb0_pgbbup = stablehlo.multiply %vitb0_pgbbcsqrt, %vitb0_pgbbin2 : tensor<32x197x768xf32>
    %vitb0_pgbbterm2 = stablehlo.multiply %vitb0_pgbbhxo, %vitb0_pgbbup : tensor<32x197x768xf32>
    %vitb0_pgbbgp = stablehlo.add %vitb0_pgbbterm1, %vitb0_pgbbterm2 : tensor<32x197x768xf32>
    %vitb0_pgbdx = stablehlo.multiply %vitb0_pda1, %vitb0_pgbbgp : tensor<32x197x768xf32>
    %vitb0_pdx = stablehlo.dot_general %vitb0_pgbdx, %Wfc1_0, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb0_pdWfc1 = stablehlo.dot_general %vitb0_2y, %vitb0_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb0_pdbfc1 = stablehlo.reduce(%vitb0_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb0_2gbk = stablehlo.broadcast_in_dim %g2_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_2dxhat = stablehlo.multiply %vitb0_pdx, %vitb0_2gbk : tensor<32x197x192xf32>
    %vitb0_2dgpre = stablehlo.multiply %vitb0_pdx, %vitb0_2xhat : tensor<32x197x192xf32>
    %vitb0_2dg = stablehlo.reduce(%vitb0_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_2db = stablehlo.reduce(%vitb0_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_2m1s = stablehlo.reduce(%vitb0_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_2m1 = stablehlo.divide %vitb0_2m1s, %vitb0_2nf : tensor<32x197xf32>
    %vitb0_2dxxh = stablehlo.multiply %vitb0_2dxhat, %vitb0_2xhat : tensor<32x197x192xf32>
    %vitb0_2m2s = stablehlo.reduce(%vitb0_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_2m2 = stablehlo.divide %vitb0_2m2s, %vitb0_2nf : tensor<32x197xf32>
    %vitb0_2m1b = stablehlo.broadcast_in_dim %vitb0_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_2m2b = stablehlo.broadcast_in_dim %vitb0_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_2t1 = stablehlo.subtract %vitb0_2dxhat, %vitb0_2m1b : tensor<32x197x192xf32>
    %vitb0_2xm2 = stablehlo.multiply %vitb0_2xhat, %vitb0_2m2b : tensor<32x197x192xf32>
    %vitb0_2t2 = stablehlo.subtract %vitb0_2t1, %vitb0_2xm2 : tensor<32x197x192xf32>
    %vitb0_2dx = stablehlo.multiply %vitb0_2istdb, %vitb0_2t2 : tensor<32x197x192xf32>
    %vitb0_dr1 = stablehlo.add %vitb1_dx, %vitb0_2dx : tensor<32x197x192xf32>
    %vitb0_mdP = stablehlo.dot_general %vitb0_dr1, %Wo_0, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mdWo = stablehlo.dot_general %vitb0_mP, %vitb0_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb0_mdbo = stablehlo.reduce(%vitb0_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_mdPr = stablehlo.reshape %vitb0_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mdA = stablehlo.transpose %vitb0_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mdW = stablehlo.dot_general %vitb0_mdA, %vitb0_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb0_mdVh = stablehlo.dot_general %vitb0_mW, %vitb0_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mpdw = stablehlo.multiply %vitb0_mW, %vitb0_mdW : tensor<32x3x197x197xf32>
    %vitb0_msrow = stablehlo.reduce(%vitb0_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb0_msrowb = stablehlo.broadcast_in_dim %vitb0_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb0_mdiff = stablehlo.subtract %vitb0_mdW, %vitb0_msrowb : tensor<32x3x197x197xf32>
    %vitb0_mdSs = stablehlo.multiply %vitb0_mW, %vitb0_mdiff : tensor<32x3x197x197xf32>
    %vitb0_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb0_mdS = stablehlo.multiply %vitb0_mdSs, %vitb0_msclb : tensor<32x3x197x197xf32>
    %vitb0_mdQh = stablehlo.dot_general %vitb0_mdS, %vitb0_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mdKh = stablehlo.dot_general %vitb0_mdS, %vitb0_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mdQT = stablehlo.transpose %vitb0_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mdQ = stablehlo.reshape %vitb0_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb0_mdKT = stablehlo.transpose %vitb0_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mdK = stablehlo.reshape %vitb0_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb0_mdVT = stablehlo.transpose %vitb0_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mdV = stablehlo.reshape %vitb0_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb0_mdxQ = stablehlo.dot_general %vitb0_mdQ, %Wq_0, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mdWQ = stablehlo.dot_general %vitb0_1y, %vitb0_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb0_mdbQ = stablehlo.reduce(%vitb0_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_mdxK = stablehlo.dot_general %vitb0_mdK, %Wk_0, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mdWK = stablehlo.dot_general %vitb0_1y, %vitb0_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb0_mdbK = stablehlo.reduce(%vitb0_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_mdxV = stablehlo.dot_general %vitb0_mdV, %Wv_0, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mdWV = stablehlo.dot_general %vitb0_1y, %vitb0_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb0_mdbV = stablehlo.reduce(%vitb0_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_mdxa = stablehlo.add %vitb0_mdxQ, %vitb0_mdxK : tensor<32x197x192xf32>
    %vitb0_mdx = stablehlo.add %vitb0_mdxa, %vitb0_mdxV : tensor<32x197x192xf32>
    %vitb0_1gbk = stablehlo.broadcast_in_dim %g1_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_1dxhat = stablehlo.multiply %vitb0_mdx, %vitb0_1gbk : tensor<32x197x192xf32>
    %vitb0_1dgpre = stablehlo.multiply %vitb0_mdx, %vitb0_1xhat : tensor<32x197x192xf32>
    %vitb0_1dg = stablehlo.reduce(%vitb0_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_1db = stablehlo.reduce(%vitb0_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_1m1s = stablehlo.reduce(%vitb0_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_1m1 = stablehlo.divide %vitb0_1m1s, %vitb0_1nf : tensor<32x197xf32>
    %vitb0_1dxxh = stablehlo.multiply %vitb0_1dxhat, %vitb0_1xhat : tensor<32x197x192xf32>
    %vitb0_1m2s = stablehlo.reduce(%vitb0_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_1m2 = stablehlo.divide %vitb0_1m2s, %vitb0_1nf : tensor<32x197xf32>
    %vitb0_1m1b = stablehlo.broadcast_in_dim %vitb0_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_1m2b = stablehlo.broadcast_in_dim %vitb0_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_1t1 = stablehlo.subtract %vitb0_1dxhat, %vitb0_1m1b : tensor<32x197x192xf32>
    %vitb0_1xm2 = stablehlo.multiply %vitb0_1xhat, %vitb0_1m2b : tensor<32x197x192xf32>
    %vitb0_1t2 = stablehlo.subtract %vitb0_1t1, %vitb0_1xm2 : tensor<32x197x192xf32>
    %vitb0_1dx = stablehlo.multiply %vitb0_1istdb, %vitb0_1t2 : tensor<32x197x192xf32>
    %vitb0_dx = stablehlo.add %vitb0_dr1, %vitb0_1dx : tensor<32x197x192xf32>
    %vitcpdpos = stablehlo.reduce(%vitb0_dx init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<197x192xf32>
    %vitcpcslc = stablehlo.slice %vitb0_dx [0:32, 0:1, 0:192] : (tensor<32x197x192xf32>) -> tensor<32x1x192xf32>
    %vitcpcr = stablehlo.reshape %vitcpcslc : (tensor<32x1x192xf32>) -> tensor<32x192xf32>
    %vitcpdcls = stablehlo.reduce(%vitcpcr init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitcpdtok = stablehlo.slice %vitb0_dx [0:32, 1:197, 0:192] : (tensor<32x197x192xf32>) -> tensor<32x196x192xf32>
    %vitpedtr = stablehlo.reshape %vitcpdtok : (tensor<32x196x192xf32>) -> tensor<32x14x14x192xf32>
    %vitpedy = stablehlo.transpose %vitpedtr, dims = [0, 3, 1, 2] : (tensor<32x14x14x192xf32>) -> tensor<32x192x14x14xf32>
    %vitpedb = stablehlo.reduce(%vitpedy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %vitpeu = stablehlo.pad %vitpedy, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 15, 15] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192x209x209xf32>
    %vitpext = stablehlo.transpose %xr, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %vitpedt = stablehlo.transpose %vitpeu, dims = [1, 0, 2, 3] : (tensor<32x192x209x209xf32>) -> tensor<192x32x209x209xf32>
    %vitperaw = stablehlo.convolution(%vitpext, %vitpedt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<192x32x209x209xf32>) -> tensor<3x192x16x16xf32>
    %vitpedw = stablehlo.transpose %vitperaw, dims = [1, 0, 2, 3] : (tensor<3x192x16x16xf32>) -> tensor<192x3x16x16xf32>
    %adb1wConv = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x3x16x16xf32>
    %adob1wConv = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x3x16x16xf32>
    %admswConv = stablehlo.multiply %adb1wConv, %wConvm : tensor<192x3x16x16xf32>
    %admgwConv = stablehlo.multiply %adob1wConv, %vitpedw : tensor<192x3x16x16xf32>
    %admnwConv = stablehlo.add %admswConv, %admgwConv : tensor<192x3x16x16xf32>
    %adb2wConv = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x3x16x16xf32>
    %adob2wConv = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x3x16x16xf32>
    %advswConv = stablehlo.multiply %adb2wConv, %wConvv : tensor<192x3x16x16xf32>
    %adg2wConv = stablehlo.multiply %vitpedw, %vitpedw : tensor<192x3x16x16xf32>
    %advgwConv = stablehlo.multiply %adob2wConv, %adg2wConv : tensor<192x3x16x16xf32>
    %advnwConv = stablehlo.add %advswConv, %advgwConv : tensor<192x3x16x16xf32>
    %adbc1wConv = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x3x16x16xf32>
    %adbc2wConv = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x3x16x16xf32>
    %admhwConv = stablehlo.divide %admnwConv, %adbc1wConv : tensor<192x3x16x16xf32>
    %advhwConv = stablehlo.divide %advnwConv, %adbc2wConv : tensor<192x3x16x16xf32>
    %adlrwConv = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x3x16x16xf32>
    %adepswConv = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x3x16x16xf32>
    %adsqwConv = stablehlo.sqrt %advhwConv : tensor<192x3x16x16xf32>
    %addenwConv = stablehlo.add %adsqwConv, %adepswConv : tensor<192x3x16x16xf32>
    %adratwConv = stablehlo.divide %admhwConv, %addenwConv : tensor<192x3x16x16xf32>
    %adstwConv = stablehlo.multiply %adlrwConv, %adratwConv : tensor<192x3x16x16xf32>
    %adsubwConv = stablehlo.subtract %wConv, %adstwConv : tensor<192x3x16x16xf32>
    %adwdwConv = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x3x16x16xf32>
    %adwdlrwConv = stablehlo.multiply %adwdwConv, %adlrwConv : tensor<192x3x16x16xf32>
    %adwdpwConv = stablehlo.multiply %adwdlrwConv, %wConv : tensor<192x3x16x16xf32>
    %adnewwConv = stablehlo.subtract %adsubwConv, %adwdpwConv : tensor<192x3x16x16xf32>
    %adb1bConv = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bConv = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbConv = stablehlo.multiply %adb1bConv, %bConvm : tensor<192xf32>
    %admgbConv = stablehlo.multiply %adob1bConv, %vitpedb : tensor<192xf32>
    %admnbConv = stablehlo.add %admsbConv, %admgbConv : tensor<192xf32>
    %adb2bConv = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bConv = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbConv = stablehlo.multiply %adb2bConv, %bConvv : tensor<192xf32>
    %adg2bConv = stablehlo.multiply %vitpedb, %vitpedb : tensor<192xf32>
    %advgbConv = stablehlo.multiply %adob2bConv, %adg2bConv : tensor<192xf32>
    %advnbConv = stablehlo.add %advsbConv, %advgbConv : tensor<192xf32>
    %adbc1bConv = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bConv = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbConv = stablehlo.divide %admnbConv, %adbc1bConv : tensor<192xf32>
    %advhbConv = stablehlo.divide %advnbConv, %adbc2bConv : tensor<192xf32>
    %adlrbConv = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbConv = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbConv = stablehlo.sqrt %advhbConv : tensor<192xf32>
    %addenbConv = stablehlo.add %adsqbConv, %adepsbConv : tensor<192xf32>
    %adratbConv = stablehlo.divide %admhbConv, %addenbConv : tensor<192xf32>
    %adstbConv = stablehlo.multiply %adlrbConv, %adratbConv : tensor<192xf32>
    %adsubbConv = stablehlo.subtract %bConv, %adstbConv : tensor<192xf32>
    %adwdbConv = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbConv = stablehlo.multiply %adwdbConv, %adlrbConv : tensor<192xf32>
    %adwdpbConv = stablehlo.multiply %adwdlrbConv, %bConv : tensor<192xf32>
    %adnewbConv = stablehlo.subtract %adsubbConv, %adwdpbConv : tensor<192xf32>
    %adb1cls = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1cls = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admscls = stablehlo.multiply %adb1cls, %clsm : tensor<192xf32>
    %admgcls = stablehlo.multiply %adob1cls, %vitcpdcls : tensor<192xf32>
    %admncls = stablehlo.add %admscls, %admgcls : tensor<192xf32>
    %adb2cls = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2cls = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advscls = stablehlo.multiply %adb2cls, %clsv : tensor<192xf32>
    %adg2cls = stablehlo.multiply %vitcpdcls, %vitcpdcls : tensor<192xf32>
    %advgcls = stablehlo.multiply %adob2cls, %adg2cls : tensor<192xf32>
    %advncls = stablehlo.add %advscls, %advgcls : tensor<192xf32>
    %adbc1cls = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2cls = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhcls = stablehlo.divide %admncls, %adbc1cls : tensor<192xf32>
    %advhcls = stablehlo.divide %advncls, %adbc2cls : tensor<192xf32>
    %adlrcls = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepscls = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqcls = stablehlo.sqrt %advhcls : tensor<192xf32>
    %addencls = stablehlo.add %adsqcls, %adepscls : tensor<192xf32>
    %adratcls = stablehlo.divide %admhcls, %addencls : tensor<192xf32>
    %adstcls = stablehlo.multiply %adlrcls, %adratcls : tensor<192xf32>
    %adsubcls = stablehlo.subtract %cls, %adstcls : tensor<192xf32>
    %adwdcls = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrcls = stablehlo.multiply %adwdcls, %adlrcls : tensor<192xf32>
    %adwdpcls = stablehlo.multiply %adwdlrcls, %cls : tensor<192xf32>
    %adnewcls = stablehlo.subtract %adsubcls, %adwdpcls : tensor<192xf32>
    %adb1pos = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<197x192xf32>
    %adob1pos = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<197x192xf32>
    %admspos = stablehlo.multiply %adb1pos, %posm : tensor<197x192xf32>
    %admgpos = stablehlo.multiply %adob1pos, %vitcpdpos : tensor<197x192xf32>
    %admnpos = stablehlo.add %admspos, %admgpos : tensor<197x192xf32>
    %adb2pos = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<197x192xf32>
    %adob2pos = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<197x192xf32>
    %advspos = stablehlo.multiply %adb2pos, %posv : tensor<197x192xf32>
    %adg2pos = stablehlo.multiply %vitcpdpos, %vitcpdpos : tensor<197x192xf32>
    %advgpos = stablehlo.multiply %adob2pos, %adg2pos : tensor<197x192xf32>
    %advnpos = stablehlo.add %advspos, %advgpos : tensor<197x192xf32>
    %adbc1pos = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<197x192xf32>
    %adbc2pos = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<197x192xf32>
    %admhpos = stablehlo.divide %admnpos, %adbc1pos : tensor<197x192xf32>
    %advhpos = stablehlo.divide %advnpos, %adbc2pos : tensor<197x192xf32>
    %adlrpos = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<197x192xf32>
    %adepspos = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<197x192xf32>
    %adsqpos = stablehlo.sqrt %advhpos : tensor<197x192xf32>
    %addenpos = stablehlo.add %adsqpos, %adepspos : tensor<197x192xf32>
    %adratpos = stablehlo.divide %admhpos, %addenpos : tensor<197x192xf32>
    %adstpos = stablehlo.multiply %adlrpos, %adratpos : tensor<197x192xf32>
    %adsubpos = stablehlo.subtract %pos, %adstpos : tensor<197x192xf32>
    %adwdpos = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<197x192xf32>
    %adwdlrpos = stablehlo.multiply %adwdpos, %adlrpos : tensor<197x192xf32>
    %adwdppos = stablehlo.multiply %adwdlrpos, %pos : tensor<197x192xf32>
    %adnewpos = stablehlo.subtract %adsubpos, %adwdppos : tensor<197x192xf32>
    %adb1g1_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g1_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg1_0 = stablehlo.multiply %adb1g1_0, %g1_0m : tensor<192xf32>
    %admgg1_0 = stablehlo.multiply %adob1g1_0, %vitb0_1dg : tensor<192xf32>
    %admng1_0 = stablehlo.add %admsg1_0, %admgg1_0 : tensor<192xf32>
    %adb2g1_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g1_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg1_0 = stablehlo.multiply %adb2g1_0, %g1_0v : tensor<192xf32>
    %adg2g1_0 = stablehlo.multiply %vitb0_1dg, %vitb0_1dg : tensor<192xf32>
    %advgg1_0 = stablehlo.multiply %adob2g1_0, %adg2g1_0 : tensor<192xf32>
    %advng1_0 = stablehlo.add %advsg1_0, %advgg1_0 : tensor<192xf32>
    %adbc1g1_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g1_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg1_0 = stablehlo.divide %admng1_0, %adbc1g1_0 : tensor<192xf32>
    %advhg1_0 = stablehlo.divide %advng1_0, %adbc2g1_0 : tensor<192xf32>
    %adlrg1_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg1_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg1_0 = stablehlo.sqrt %advhg1_0 : tensor<192xf32>
    %addeng1_0 = stablehlo.add %adsqg1_0, %adepsg1_0 : tensor<192xf32>
    %adratg1_0 = stablehlo.divide %admhg1_0, %addeng1_0 : tensor<192xf32>
    %adstg1_0 = stablehlo.multiply %adlrg1_0, %adratg1_0 : tensor<192xf32>
    %adsubg1_0 = stablehlo.subtract %g1_0, %adstg1_0 : tensor<192xf32>
    %adwdg1_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg1_0 = stablehlo.multiply %adwdg1_0, %adlrg1_0 : tensor<192xf32>
    %adwdpg1_0 = stablehlo.multiply %adwdlrg1_0, %g1_0 : tensor<192xf32>
    %adnewg1_0 = stablehlo.subtract %adsubg1_0, %adwdpg1_0 : tensor<192xf32>
    %adb1b1_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b1_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb1_0 = stablehlo.multiply %adb1b1_0, %b1_0m : tensor<192xf32>
    %admgb1_0 = stablehlo.multiply %adob1b1_0, %vitb0_1db : tensor<192xf32>
    %admnb1_0 = stablehlo.add %admsb1_0, %admgb1_0 : tensor<192xf32>
    %adb2b1_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b1_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb1_0 = stablehlo.multiply %adb2b1_0, %b1_0v : tensor<192xf32>
    %adg2b1_0 = stablehlo.multiply %vitb0_1db, %vitb0_1db : tensor<192xf32>
    %advgb1_0 = stablehlo.multiply %adob2b1_0, %adg2b1_0 : tensor<192xf32>
    %advnb1_0 = stablehlo.add %advsb1_0, %advgb1_0 : tensor<192xf32>
    %adbc1b1_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b1_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb1_0 = stablehlo.divide %admnb1_0, %adbc1b1_0 : tensor<192xf32>
    %advhb1_0 = stablehlo.divide %advnb1_0, %adbc2b1_0 : tensor<192xf32>
    %adlrb1_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb1_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb1_0 = stablehlo.sqrt %advhb1_0 : tensor<192xf32>
    %addenb1_0 = stablehlo.add %adsqb1_0, %adepsb1_0 : tensor<192xf32>
    %adratb1_0 = stablehlo.divide %admhb1_0, %addenb1_0 : tensor<192xf32>
    %adstb1_0 = stablehlo.multiply %adlrb1_0, %adratb1_0 : tensor<192xf32>
    %adsubb1_0 = stablehlo.subtract %b1_0, %adstb1_0 : tensor<192xf32>
    %adwdb1_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb1_0 = stablehlo.multiply %adwdb1_0, %adlrb1_0 : tensor<192xf32>
    %adwdpb1_0 = stablehlo.multiply %adwdlrb1_0, %b1_0 : tensor<192xf32>
    %adnewb1_0 = stablehlo.subtract %adsubb1_0, %adwdpb1_0 : tensor<192xf32>
    %adb1Wq_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wq_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWq_0 = stablehlo.multiply %adb1Wq_0, %Wq_0m : tensor<192x192xf32>
    %admgWq_0 = stablehlo.multiply %adob1Wq_0, %vitb0_mdWQ : tensor<192x192xf32>
    %admnWq_0 = stablehlo.add %admsWq_0, %admgWq_0 : tensor<192x192xf32>
    %adb2Wq_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wq_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWq_0 = stablehlo.multiply %adb2Wq_0, %Wq_0v : tensor<192x192xf32>
    %adg2Wq_0 = stablehlo.multiply %vitb0_mdWQ, %vitb0_mdWQ : tensor<192x192xf32>
    %advgWq_0 = stablehlo.multiply %adob2Wq_0, %adg2Wq_0 : tensor<192x192xf32>
    %advnWq_0 = stablehlo.add %advsWq_0, %advgWq_0 : tensor<192x192xf32>
    %adbc1Wq_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wq_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWq_0 = stablehlo.divide %admnWq_0, %adbc1Wq_0 : tensor<192x192xf32>
    %advhWq_0 = stablehlo.divide %advnWq_0, %adbc2Wq_0 : tensor<192x192xf32>
    %adlrWq_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWq_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWq_0 = stablehlo.sqrt %advhWq_0 : tensor<192x192xf32>
    %addenWq_0 = stablehlo.add %adsqWq_0, %adepsWq_0 : tensor<192x192xf32>
    %adratWq_0 = stablehlo.divide %admhWq_0, %addenWq_0 : tensor<192x192xf32>
    %adstWq_0 = stablehlo.multiply %adlrWq_0, %adratWq_0 : tensor<192x192xf32>
    %adsubWq_0 = stablehlo.subtract %Wq_0, %adstWq_0 : tensor<192x192xf32>
    %adwdWq_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWq_0 = stablehlo.multiply %adwdWq_0, %adlrWq_0 : tensor<192x192xf32>
    %adwdpWq_0 = stablehlo.multiply %adwdlrWq_0, %Wq_0 : tensor<192x192xf32>
    %adnewWq_0 = stablehlo.subtract %adsubWq_0, %adwdpWq_0 : tensor<192x192xf32>
    %adb1bq_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bq_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbq_0 = stablehlo.multiply %adb1bq_0, %bq_0m : tensor<192xf32>
    %admgbq_0 = stablehlo.multiply %adob1bq_0, %vitb0_mdbQ : tensor<192xf32>
    %admnbq_0 = stablehlo.add %admsbq_0, %admgbq_0 : tensor<192xf32>
    %adb2bq_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bq_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbq_0 = stablehlo.multiply %adb2bq_0, %bq_0v : tensor<192xf32>
    %adg2bq_0 = stablehlo.multiply %vitb0_mdbQ, %vitb0_mdbQ : tensor<192xf32>
    %advgbq_0 = stablehlo.multiply %adob2bq_0, %adg2bq_0 : tensor<192xf32>
    %advnbq_0 = stablehlo.add %advsbq_0, %advgbq_0 : tensor<192xf32>
    %adbc1bq_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bq_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbq_0 = stablehlo.divide %admnbq_0, %adbc1bq_0 : tensor<192xf32>
    %advhbq_0 = stablehlo.divide %advnbq_0, %adbc2bq_0 : tensor<192xf32>
    %adlrbq_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbq_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbq_0 = stablehlo.sqrt %advhbq_0 : tensor<192xf32>
    %addenbq_0 = stablehlo.add %adsqbq_0, %adepsbq_0 : tensor<192xf32>
    %adratbq_0 = stablehlo.divide %admhbq_0, %addenbq_0 : tensor<192xf32>
    %adstbq_0 = stablehlo.multiply %adlrbq_0, %adratbq_0 : tensor<192xf32>
    %adsubbq_0 = stablehlo.subtract %bq_0, %adstbq_0 : tensor<192xf32>
    %adwdbq_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbq_0 = stablehlo.multiply %adwdbq_0, %adlrbq_0 : tensor<192xf32>
    %adwdpbq_0 = stablehlo.multiply %adwdlrbq_0, %bq_0 : tensor<192xf32>
    %adnewbq_0 = stablehlo.subtract %adsubbq_0, %adwdpbq_0 : tensor<192xf32>
    %adb1Wk_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wk_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWk_0 = stablehlo.multiply %adb1Wk_0, %Wk_0m : tensor<192x192xf32>
    %admgWk_0 = stablehlo.multiply %adob1Wk_0, %vitb0_mdWK : tensor<192x192xf32>
    %admnWk_0 = stablehlo.add %admsWk_0, %admgWk_0 : tensor<192x192xf32>
    %adb2Wk_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wk_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWk_0 = stablehlo.multiply %adb2Wk_0, %Wk_0v : tensor<192x192xf32>
    %adg2Wk_0 = stablehlo.multiply %vitb0_mdWK, %vitb0_mdWK : tensor<192x192xf32>
    %advgWk_0 = stablehlo.multiply %adob2Wk_0, %adg2Wk_0 : tensor<192x192xf32>
    %advnWk_0 = stablehlo.add %advsWk_0, %advgWk_0 : tensor<192x192xf32>
    %adbc1Wk_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wk_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWk_0 = stablehlo.divide %admnWk_0, %adbc1Wk_0 : tensor<192x192xf32>
    %advhWk_0 = stablehlo.divide %advnWk_0, %adbc2Wk_0 : tensor<192x192xf32>
    %adlrWk_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWk_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWk_0 = stablehlo.sqrt %advhWk_0 : tensor<192x192xf32>
    %addenWk_0 = stablehlo.add %adsqWk_0, %adepsWk_0 : tensor<192x192xf32>
    %adratWk_0 = stablehlo.divide %admhWk_0, %addenWk_0 : tensor<192x192xf32>
    %adstWk_0 = stablehlo.multiply %adlrWk_0, %adratWk_0 : tensor<192x192xf32>
    %adsubWk_0 = stablehlo.subtract %Wk_0, %adstWk_0 : tensor<192x192xf32>
    %adwdWk_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWk_0 = stablehlo.multiply %adwdWk_0, %adlrWk_0 : tensor<192x192xf32>
    %adwdpWk_0 = stablehlo.multiply %adwdlrWk_0, %Wk_0 : tensor<192x192xf32>
    %adnewWk_0 = stablehlo.subtract %adsubWk_0, %adwdpWk_0 : tensor<192x192xf32>
    %adb1bk_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bk_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbk_0 = stablehlo.multiply %adb1bk_0, %bk_0m : tensor<192xf32>
    %admgbk_0 = stablehlo.multiply %adob1bk_0, %vitb0_mdbK : tensor<192xf32>
    %admnbk_0 = stablehlo.add %admsbk_0, %admgbk_0 : tensor<192xf32>
    %adb2bk_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bk_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbk_0 = stablehlo.multiply %adb2bk_0, %bk_0v : tensor<192xf32>
    %adg2bk_0 = stablehlo.multiply %vitb0_mdbK, %vitb0_mdbK : tensor<192xf32>
    %advgbk_0 = stablehlo.multiply %adob2bk_0, %adg2bk_0 : tensor<192xf32>
    %advnbk_0 = stablehlo.add %advsbk_0, %advgbk_0 : tensor<192xf32>
    %adbc1bk_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bk_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbk_0 = stablehlo.divide %admnbk_0, %adbc1bk_0 : tensor<192xf32>
    %advhbk_0 = stablehlo.divide %advnbk_0, %adbc2bk_0 : tensor<192xf32>
    %adlrbk_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbk_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbk_0 = stablehlo.sqrt %advhbk_0 : tensor<192xf32>
    %addenbk_0 = stablehlo.add %adsqbk_0, %adepsbk_0 : tensor<192xf32>
    %adratbk_0 = stablehlo.divide %admhbk_0, %addenbk_0 : tensor<192xf32>
    %adstbk_0 = stablehlo.multiply %adlrbk_0, %adratbk_0 : tensor<192xf32>
    %adsubbk_0 = stablehlo.subtract %bk_0, %adstbk_0 : tensor<192xf32>
    %adwdbk_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbk_0 = stablehlo.multiply %adwdbk_0, %adlrbk_0 : tensor<192xf32>
    %adwdpbk_0 = stablehlo.multiply %adwdlrbk_0, %bk_0 : tensor<192xf32>
    %adnewbk_0 = stablehlo.subtract %adsubbk_0, %adwdpbk_0 : tensor<192xf32>
    %adb1Wv_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wv_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWv_0 = stablehlo.multiply %adb1Wv_0, %Wv_0m : tensor<192x192xf32>
    %admgWv_0 = stablehlo.multiply %adob1Wv_0, %vitb0_mdWV : tensor<192x192xf32>
    %admnWv_0 = stablehlo.add %admsWv_0, %admgWv_0 : tensor<192x192xf32>
    %adb2Wv_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wv_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWv_0 = stablehlo.multiply %adb2Wv_0, %Wv_0v : tensor<192x192xf32>
    %adg2Wv_0 = stablehlo.multiply %vitb0_mdWV, %vitb0_mdWV : tensor<192x192xf32>
    %advgWv_0 = stablehlo.multiply %adob2Wv_0, %adg2Wv_0 : tensor<192x192xf32>
    %advnWv_0 = stablehlo.add %advsWv_0, %advgWv_0 : tensor<192x192xf32>
    %adbc1Wv_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wv_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWv_0 = stablehlo.divide %admnWv_0, %adbc1Wv_0 : tensor<192x192xf32>
    %advhWv_0 = stablehlo.divide %advnWv_0, %adbc2Wv_0 : tensor<192x192xf32>
    %adlrWv_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWv_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWv_0 = stablehlo.sqrt %advhWv_0 : tensor<192x192xf32>
    %addenWv_0 = stablehlo.add %adsqWv_0, %adepsWv_0 : tensor<192x192xf32>
    %adratWv_0 = stablehlo.divide %admhWv_0, %addenWv_0 : tensor<192x192xf32>
    %adstWv_0 = stablehlo.multiply %adlrWv_0, %adratWv_0 : tensor<192x192xf32>
    %adsubWv_0 = stablehlo.subtract %Wv_0, %adstWv_0 : tensor<192x192xf32>
    %adwdWv_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWv_0 = stablehlo.multiply %adwdWv_0, %adlrWv_0 : tensor<192x192xf32>
    %adwdpWv_0 = stablehlo.multiply %adwdlrWv_0, %Wv_0 : tensor<192x192xf32>
    %adnewWv_0 = stablehlo.subtract %adsubWv_0, %adwdpWv_0 : tensor<192x192xf32>
    %adb1bv_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bv_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbv_0 = stablehlo.multiply %adb1bv_0, %bv_0m : tensor<192xf32>
    %admgbv_0 = stablehlo.multiply %adob1bv_0, %vitb0_mdbV : tensor<192xf32>
    %admnbv_0 = stablehlo.add %admsbv_0, %admgbv_0 : tensor<192xf32>
    %adb2bv_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bv_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbv_0 = stablehlo.multiply %adb2bv_0, %bv_0v : tensor<192xf32>
    %adg2bv_0 = stablehlo.multiply %vitb0_mdbV, %vitb0_mdbV : tensor<192xf32>
    %advgbv_0 = stablehlo.multiply %adob2bv_0, %adg2bv_0 : tensor<192xf32>
    %advnbv_0 = stablehlo.add %advsbv_0, %advgbv_0 : tensor<192xf32>
    %adbc1bv_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bv_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbv_0 = stablehlo.divide %admnbv_0, %adbc1bv_0 : tensor<192xf32>
    %advhbv_0 = stablehlo.divide %advnbv_0, %adbc2bv_0 : tensor<192xf32>
    %adlrbv_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbv_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbv_0 = stablehlo.sqrt %advhbv_0 : tensor<192xf32>
    %addenbv_0 = stablehlo.add %adsqbv_0, %adepsbv_0 : tensor<192xf32>
    %adratbv_0 = stablehlo.divide %admhbv_0, %addenbv_0 : tensor<192xf32>
    %adstbv_0 = stablehlo.multiply %adlrbv_0, %adratbv_0 : tensor<192xf32>
    %adsubbv_0 = stablehlo.subtract %bv_0, %adstbv_0 : tensor<192xf32>
    %adwdbv_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbv_0 = stablehlo.multiply %adwdbv_0, %adlrbv_0 : tensor<192xf32>
    %adwdpbv_0 = stablehlo.multiply %adwdlrbv_0, %bv_0 : tensor<192xf32>
    %adnewbv_0 = stablehlo.subtract %adsubbv_0, %adwdpbv_0 : tensor<192xf32>
    %adb1Wo_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wo_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWo_0 = stablehlo.multiply %adb1Wo_0, %Wo_0m : tensor<192x192xf32>
    %admgWo_0 = stablehlo.multiply %adob1Wo_0, %vitb0_mdWo : tensor<192x192xf32>
    %admnWo_0 = stablehlo.add %admsWo_0, %admgWo_0 : tensor<192x192xf32>
    %adb2Wo_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wo_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWo_0 = stablehlo.multiply %adb2Wo_0, %Wo_0v : tensor<192x192xf32>
    %adg2Wo_0 = stablehlo.multiply %vitb0_mdWo, %vitb0_mdWo : tensor<192x192xf32>
    %advgWo_0 = stablehlo.multiply %adob2Wo_0, %adg2Wo_0 : tensor<192x192xf32>
    %advnWo_0 = stablehlo.add %advsWo_0, %advgWo_0 : tensor<192x192xf32>
    %adbc1Wo_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wo_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWo_0 = stablehlo.divide %admnWo_0, %adbc1Wo_0 : tensor<192x192xf32>
    %advhWo_0 = stablehlo.divide %advnWo_0, %adbc2Wo_0 : tensor<192x192xf32>
    %adlrWo_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWo_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWo_0 = stablehlo.sqrt %advhWo_0 : tensor<192x192xf32>
    %addenWo_0 = stablehlo.add %adsqWo_0, %adepsWo_0 : tensor<192x192xf32>
    %adratWo_0 = stablehlo.divide %admhWo_0, %addenWo_0 : tensor<192x192xf32>
    %adstWo_0 = stablehlo.multiply %adlrWo_0, %adratWo_0 : tensor<192x192xf32>
    %adsubWo_0 = stablehlo.subtract %Wo_0, %adstWo_0 : tensor<192x192xf32>
    %adwdWo_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWo_0 = stablehlo.multiply %adwdWo_0, %adlrWo_0 : tensor<192x192xf32>
    %adwdpWo_0 = stablehlo.multiply %adwdlrWo_0, %Wo_0 : tensor<192x192xf32>
    %adnewWo_0 = stablehlo.subtract %adsubWo_0, %adwdpWo_0 : tensor<192x192xf32>
    %adb1bo_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bo_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbo_0 = stablehlo.multiply %adb1bo_0, %bo_0m : tensor<192xf32>
    %admgbo_0 = stablehlo.multiply %adob1bo_0, %vitb0_mdbo : tensor<192xf32>
    %admnbo_0 = stablehlo.add %admsbo_0, %admgbo_0 : tensor<192xf32>
    %adb2bo_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bo_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbo_0 = stablehlo.multiply %adb2bo_0, %bo_0v : tensor<192xf32>
    %adg2bo_0 = stablehlo.multiply %vitb0_mdbo, %vitb0_mdbo : tensor<192xf32>
    %advgbo_0 = stablehlo.multiply %adob2bo_0, %adg2bo_0 : tensor<192xf32>
    %advnbo_0 = stablehlo.add %advsbo_0, %advgbo_0 : tensor<192xf32>
    %adbc1bo_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bo_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbo_0 = stablehlo.divide %admnbo_0, %adbc1bo_0 : tensor<192xf32>
    %advhbo_0 = stablehlo.divide %advnbo_0, %adbc2bo_0 : tensor<192xf32>
    %adlrbo_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbo_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbo_0 = stablehlo.sqrt %advhbo_0 : tensor<192xf32>
    %addenbo_0 = stablehlo.add %adsqbo_0, %adepsbo_0 : tensor<192xf32>
    %adratbo_0 = stablehlo.divide %admhbo_0, %addenbo_0 : tensor<192xf32>
    %adstbo_0 = stablehlo.multiply %adlrbo_0, %adratbo_0 : tensor<192xf32>
    %adsubbo_0 = stablehlo.subtract %bo_0, %adstbo_0 : tensor<192xf32>
    %adwdbo_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbo_0 = stablehlo.multiply %adwdbo_0, %adlrbo_0 : tensor<192xf32>
    %adwdpbo_0 = stablehlo.multiply %adwdlrbo_0, %bo_0 : tensor<192xf32>
    %adnewbo_0 = stablehlo.subtract %adsubbo_0, %adwdpbo_0 : tensor<192xf32>
    %adb1g2_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g2_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg2_0 = stablehlo.multiply %adb1g2_0, %g2_0m : tensor<192xf32>
    %admgg2_0 = stablehlo.multiply %adob1g2_0, %vitb0_2dg : tensor<192xf32>
    %admng2_0 = stablehlo.add %admsg2_0, %admgg2_0 : tensor<192xf32>
    %adb2g2_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g2_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg2_0 = stablehlo.multiply %adb2g2_0, %g2_0v : tensor<192xf32>
    %adg2g2_0 = stablehlo.multiply %vitb0_2dg, %vitb0_2dg : tensor<192xf32>
    %advgg2_0 = stablehlo.multiply %adob2g2_0, %adg2g2_0 : tensor<192xf32>
    %advng2_0 = stablehlo.add %advsg2_0, %advgg2_0 : tensor<192xf32>
    %adbc1g2_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g2_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg2_0 = stablehlo.divide %admng2_0, %adbc1g2_0 : tensor<192xf32>
    %advhg2_0 = stablehlo.divide %advng2_0, %adbc2g2_0 : tensor<192xf32>
    %adlrg2_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg2_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg2_0 = stablehlo.sqrt %advhg2_0 : tensor<192xf32>
    %addeng2_0 = stablehlo.add %adsqg2_0, %adepsg2_0 : tensor<192xf32>
    %adratg2_0 = stablehlo.divide %admhg2_0, %addeng2_0 : tensor<192xf32>
    %adstg2_0 = stablehlo.multiply %adlrg2_0, %adratg2_0 : tensor<192xf32>
    %adsubg2_0 = stablehlo.subtract %g2_0, %adstg2_0 : tensor<192xf32>
    %adwdg2_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg2_0 = stablehlo.multiply %adwdg2_0, %adlrg2_0 : tensor<192xf32>
    %adwdpg2_0 = stablehlo.multiply %adwdlrg2_0, %g2_0 : tensor<192xf32>
    %adnewg2_0 = stablehlo.subtract %adsubg2_0, %adwdpg2_0 : tensor<192xf32>
    %adb1b2_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b2_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb2_0 = stablehlo.multiply %adb1b2_0, %b2_0m : tensor<192xf32>
    %admgb2_0 = stablehlo.multiply %adob1b2_0, %vitb0_2db : tensor<192xf32>
    %admnb2_0 = stablehlo.add %admsb2_0, %admgb2_0 : tensor<192xf32>
    %adb2b2_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b2_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb2_0 = stablehlo.multiply %adb2b2_0, %b2_0v : tensor<192xf32>
    %adg2b2_0 = stablehlo.multiply %vitb0_2db, %vitb0_2db : tensor<192xf32>
    %advgb2_0 = stablehlo.multiply %adob2b2_0, %adg2b2_0 : tensor<192xf32>
    %advnb2_0 = stablehlo.add %advsb2_0, %advgb2_0 : tensor<192xf32>
    %adbc1b2_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b2_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb2_0 = stablehlo.divide %admnb2_0, %adbc1b2_0 : tensor<192xf32>
    %advhb2_0 = stablehlo.divide %advnb2_0, %adbc2b2_0 : tensor<192xf32>
    %adlrb2_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb2_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb2_0 = stablehlo.sqrt %advhb2_0 : tensor<192xf32>
    %addenb2_0 = stablehlo.add %adsqb2_0, %adepsb2_0 : tensor<192xf32>
    %adratb2_0 = stablehlo.divide %admhb2_0, %addenb2_0 : tensor<192xf32>
    %adstb2_0 = stablehlo.multiply %adlrb2_0, %adratb2_0 : tensor<192xf32>
    %adsubb2_0 = stablehlo.subtract %b2_0, %adstb2_0 : tensor<192xf32>
    %adwdb2_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb2_0 = stablehlo.multiply %adwdb2_0, %adlrb2_0 : tensor<192xf32>
    %adwdpb2_0 = stablehlo.multiply %adwdlrb2_0, %b2_0 : tensor<192xf32>
    %adnewb2_0 = stablehlo.subtract %adsubb2_0, %adwdpb2_0 : tensor<192xf32>
    %adb1Wfc1_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob1Wfc1_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admsWfc1_0 = stablehlo.multiply %adb1Wfc1_0, %Wfc1_0m : tensor<192x768xf32>
    %admgWfc1_0 = stablehlo.multiply %adob1Wfc1_0, %vitb0_pdWfc1 : tensor<192x768xf32>
    %admnWfc1_0 = stablehlo.add %admsWfc1_0, %admgWfc1_0 : tensor<192x768xf32>
    %adb2Wfc1_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob2Wfc1_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %advsWfc1_0 = stablehlo.multiply %adb2Wfc1_0, %Wfc1_0v : tensor<192x768xf32>
    %adg2Wfc1_0 = stablehlo.multiply %vitb0_pdWfc1, %vitb0_pdWfc1 : tensor<192x768xf32>
    %advgWfc1_0 = stablehlo.multiply %adob2Wfc1_0, %adg2Wfc1_0 : tensor<192x768xf32>
    %advnWfc1_0 = stablehlo.add %advsWfc1_0, %advgWfc1_0 : tensor<192x768xf32>
    %adbc1Wfc1_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adbc2Wfc1_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admhWfc1_0 = stablehlo.divide %admnWfc1_0, %adbc1Wfc1_0 : tensor<192x768xf32>
    %advhWfc1_0 = stablehlo.divide %advnWfc1_0, %adbc2Wfc1_0 : tensor<192x768xf32>
    %adlrWfc1_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adepsWfc1_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adsqWfc1_0 = stablehlo.sqrt %advhWfc1_0 : tensor<192x768xf32>
    %addenWfc1_0 = stablehlo.add %adsqWfc1_0, %adepsWfc1_0 : tensor<192x768xf32>
    %adratWfc1_0 = stablehlo.divide %admhWfc1_0, %addenWfc1_0 : tensor<192x768xf32>
    %adstWfc1_0 = stablehlo.multiply %adlrWfc1_0, %adratWfc1_0 : tensor<192x768xf32>
    %adsubWfc1_0 = stablehlo.subtract %Wfc1_0, %adstWfc1_0 : tensor<192x768xf32>
    %adwdWfc1_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adwdlrWfc1_0 = stablehlo.multiply %adwdWfc1_0, %adlrWfc1_0 : tensor<192x768xf32>
    %adwdpWfc1_0 = stablehlo.multiply %adwdlrWfc1_0, %Wfc1_0 : tensor<192x768xf32>
    %adnewWfc1_0 = stablehlo.subtract %adsubWfc1_0, %adwdpWfc1_0 : tensor<192x768xf32>
    %adb1bfc1_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1bfc1_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admsbfc1_0 = stablehlo.multiply %adb1bfc1_0, %bfc1_0m : tensor<768xf32>
    %admgbfc1_0 = stablehlo.multiply %adob1bfc1_0, %vitb0_pdbfc1 : tensor<768xf32>
    %admnbfc1_0 = stablehlo.add %admsbfc1_0, %admgbfc1_0 : tensor<768xf32>
    %adb2bfc1_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2bfc1_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advsbfc1_0 = stablehlo.multiply %adb2bfc1_0, %bfc1_0v : tensor<768xf32>
    %adg2bfc1_0 = stablehlo.multiply %vitb0_pdbfc1, %vitb0_pdbfc1 : tensor<768xf32>
    %advgbfc1_0 = stablehlo.multiply %adob2bfc1_0, %adg2bfc1_0 : tensor<768xf32>
    %advnbfc1_0 = stablehlo.add %advsbfc1_0, %advgbfc1_0 : tensor<768xf32>
    %adbc1bfc1_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2bfc1_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhbfc1_0 = stablehlo.divide %admnbfc1_0, %adbc1bfc1_0 : tensor<768xf32>
    %advhbfc1_0 = stablehlo.divide %advnbfc1_0, %adbc2bfc1_0 : tensor<768xf32>
    %adlrbfc1_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepsbfc1_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqbfc1_0 = stablehlo.sqrt %advhbfc1_0 : tensor<768xf32>
    %addenbfc1_0 = stablehlo.add %adsqbfc1_0, %adepsbfc1_0 : tensor<768xf32>
    %adratbfc1_0 = stablehlo.divide %admhbfc1_0, %addenbfc1_0 : tensor<768xf32>
    %adstbfc1_0 = stablehlo.multiply %adlrbfc1_0, %adratbfc1_0 : tensor<768xf32>
    %adsubbfc1_0 = stablehlo.subtract %bfc1_0, %adstbfc1_0 : tensor<768xf32>
    %adwdbfc1_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrbfc1_0 = stablehlo.multiply %adwdbfc1_0, %adlrbfc1_0 : tensor<768xf32>
    %adwdpbfc1_0 = stablehlo.multiply %adwdlrbfc1_0, %bfc1_0 : tensor<768xf32>
    %adnewbfc1_0 = stablehlo.subtract %adsubbfc1_0, %adwdpbfc1_0 : tensor<768xf32>
    %adb1Wfc2_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob1Wfc2_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admsWfc2_0 = stablehlo.multiply %adb1Wfc2_0, %Wfc2_0m : tensor<768x192xf32>
    %admgWfc2_0 = stablehlo.multiply %adob1Wfc2_0, %vitb0_pdWfc2 : tensor<768x192xf32>
    %admnWfc2_0 = stablehlo.add %admsWfc2_0, %admgWfc2_0 : tensor<768x192xf32>
    %adb2Wfc2_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob2Wfc2_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %advsWfc2_0 = stablehlo.multiply %adb2Wfc2_0, %Wfc2_0v : tensor<768x192xf32>
    %adg2Wfc2_0 = stablehlo.multiply %vitb0_pdWfc2, %vitb0_pdWfc2 : tensor<768x192xf32>
    %advgWfc2_0 = stablehlo.multiply %adob2Wfc2_0, %adg2Wfc2_0 : tensor<768x192xf32>
    %advnWfc2_0 = stablehlo.add %advsWfc2_0, %advgWfc2_0 : tensor<768x192xf32>
    %adbc1Wfc2_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adbc2Wfc2_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admhWfc2_0 = stablehlo.divide %admnWfc2_0, %adbc1Wfc2_0 : tensor<768x192xf32>
    %advhWfc2_0 = stablehlo.divide %advnWfc2_0, %adbc2Wfc2_0 : tensor<768x192xf32>
    %adlrWfc2_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adepsWfc2_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adsqWfc2_0 = stablehlo.sqrt %advhWfc2_0 : tensor<768x192xf32>
    %addenWfc2_0 = stablehlo.add %adsqWfc2_0, %adepsWfc2_0 : tensor<768x192xf32>
    %adratWfc2_0 = stablehlo.divide %admhWfc2_0, %addenWfc2_0 : tensor<768x192xf32>
    %adstWfc2_0 = stablehlo.multiply %adlrWfc2_0, %adratWfc2_0 : tensor<768x192xf32>
    %adsubWfc2_0 = stablehlo.subtract %Wfc2_0, %adstWfc2_0 : tensor<768x192xf32>
    %adwdWfc2_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adwdlrWfc2_0 = stablehlo.multiply %adwdWfc2_0, %adlrWfc2_0 : tensor<768x192xf32>
    %adwdpWfc2_0 = stablehlo.multiply %adwdlrWfc2_0, %Wfc2_0 : tensor<768x192xf32>
    %adnewWfc2_0 = stablehlo.subtract %adsubWfc2_0, %adwdpWfc2_0 : tensor<768x192xf32>
    %adb1bfc2_0 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bfc2_0 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbfc2_0 = stablehlo.multiply %adb1bfc2_0, %bfc2_0m : tensor<192xf32>
    %admgbfc2_0 = stablehlo.multiply %adob1bfc2_0, %vitb0_pdbfc2 : tensor<192xf32>
    %admnbfc2_0 = stablehlo.add %admsbfc2_0, %admgbfc2_0 : tensor<192xf32>
    %adb2bfc2_0 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bfc2_0 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbfc2_0 = stablehlo.multiply %adb2bfc2_0, %bfc2_0v : tensor<192xf32>
    %adg2bfc2_0 = stablehlo.multiply %vitb0_pdbfc2, %vitb0_pdbfc2 : tensor<192xf32>
    %advgbfc2_0 = stablehlo.multiply %adob2bfc2_0, %adg2bfc2_0 : tensor<192xf32>
    %advnbfc2_0 = stablehlo.add %advsbfc2_0, %advgbfc2_0 : tensor<192xf32>
    %adbc1bfc2_0 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bfc2_0 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbfc2_0 = stablehlo.divide %admnbfc2_0, %adbc1bfc2_0 : tensor<192xf32>
    %advhbfc2_0 = stablehlo.divide %advnbfc2_0, %adbc2bfc2_0 : tensor<192xf32>
    %adlrbfc2_0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbfc2_0 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbfc2_0 = stablehlo.sqrt %advhbfc2_0 : tensor<192xf32>
    %addenbfc2_0 = stablehlo.add %adsqbfc2_0, %adepsbfc2_0 : tensor<192xf32>
    %adratbfc2_0 = stablehlo.divide %admhbfc2_0, %addenbfc2_0 : tensor<192xf32>
    %adstbfc2_0 = stablehlo.multiply %adlrbfc2_0, %adratbfc2_0 : tensor<192xf32>
    %adsubbfc2_0 = stablehlo.subtract %bfc2_0, %adstbfc2_0 : tensor<192xf32>
    %adwdbfc2_0 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbfc2_0 = stablehlo.multiply %adwdbfc2_0, %adlrbfc2_0 : tensor<192xf32>
    %adwdpbfc2_0 = stablehlo.multiply %adwdlrbfc2_0, %bfc2_0 : tensor<192xf32>
    %adnewbfc2_0 = stablehlo.subtract %adsubbfc2_0, %adwdpbfc2_0 : tensor<192xf32>
    %adb1g1_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g1_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg1_1 = stablehlo.multiply %adb1g1_1, %g1_1m : tensor<192xf32>
    %admgg1_1 = stablehlo.multiply %adob1g1_1, %vitb1_1dg : tensor<192xf32>
    %admng1_1 = stablehlo.add %admsg1_1, %admgg1_1 : tensor<192xf32>
    %adb2g1_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g1_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg1_1 = stablehlo.multiply %adb2g1_1, %g1_1v : tensor<192xf32>
    %adg2g1_1 = stablehlo.multiply %vitb1_1dg, %vitb1_1dg : tensor<192xf32>
    %advgg1_1 = stablehlo.multiply %adob2g1_1, %adg2g1_1 : tensor<192xf32>
    %advng1_1 = stablehlo.add %advsg1_1, %advgg1_1 : tensor<192xf32>
    %adbc1g1_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g1_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg1_1 = stablehlo.divide %admng1_1, %adbc1g1_1 : tensor<192xf32>
    %advhg1_1 = stablehlo.divide %advng1_1, %adbc2g1_1 : tensor<192xf32>
    %adlrg1_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg1_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg1_1 = stablehlo.sqrt %advhg1_1 : tensor<192xf32>
    %addeng1_1 = stablehlo.add %adsqg1_1, %adepsg1_1 : tensor<192xf32>
    %adratg1_1 = stablehlo.divide %admhg1_1, %addeng1_1 : tensor<192xf32>
    %adstg1_1 = stablehlo.multiply %adlrg1_1, %adratg1_1 : tensor<192xf32>
    %adsubg1_1 = stablehlo.subtract %g1_1, %adstg1_1 : tensor<192xf32>
    %adwdg1_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg1_1 = stablehlo.multiply %adwdg1_1, %adlrg1_1 : tensor<192xf32>
    %adwdpg1_1 = stablehlo.multiply %adwdlrg1_1, %g1_1 : tensor<192xf32>
    %adnewg1_1 = stablehlo.subtract %adsubg1_1, %adwdpg1_1 : tensor<192xf32>
    %adb1b1_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b1_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb1_1 = stablehlo.multiply %adb1b1_1, %b1_1m : tensor<192xf32>
    %admgb1_1 = stablehlo.multiply %adob1b1_1, %vitb1_1db : tensor<192xf32>
    %admnb1_1 = stablehlo.add %admsb1_1, %admgb1_1 : tensor<192xf32>
    %adb2b1_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b1_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb1_1 = stablehlo.multiply %adb2b1_1, %b1_1v : tensor<192xf32>
    %adg2b1_1 = stablehlo.multiply %vitb1_1db, %vitb1_1db : tensor<192xf32>
    %advgb1_1 = stablehlo.multiply %adob2b1_1, %adg2b1_1 : tensor<192xf32>
    %advnb1_1 = stablehlo.add %advsb1_1, %advgb1_1 : tensor<192xf32>
    %adbc1b1_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b1_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb1_1 = stablehlo.divide %admnb1_1, %adbc1b1_1 : tensor<192xf32>
    %advhb1_1 = stablehlo.divide %advnb1_1, %adbc2b1_1 : tensor<192xf32>
    %adlrb1_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb1_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb1_1 = stablehlo.sqrt %advhb1_1 : tensor<192xf32>
    %addenb1_1 = stablehlo.add %adsqb1_1, %adepsb1_1 : tensor<192xf32>
    %adratb1_1 = stablehlo.divide %admhb1_1, %addenb1_1 : tensor<192xf32>
    %adstb1_1 = stablehlo.multiply %adlrb1_1, %adratb1_1 : tensor<192xf32>
    %adsubb1_1 = stablehlo.subtract %b1_1, %adstb1_1 : tensor<192xf32>
    %adwdb1_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb1_1 = stablehlo.multiply %adwdb1_1, %adlrb1_1 : tensor<192xf32>
    %adwdpb1_1 = stablehlo.multiply %adwdlrb1_1, %b1_1 : tensor<192xf32>
    %adnewb1_1 = stablehlo.subtract %adsubb1_1, %adwdpb1_1 : tensor<192xf32>
    %adb1Wq_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wq_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWq_1 = stablehlo.multiply %adb1Wq_1, %Wq_1m : tensor<192x192xf32>
    %admgWq_1 = stablehlo.multiply %adob1Wq_1, %vitb1_mdWQ : tensor<192x192xf32>
    %admnWq_1 = stablehlo.add %admsWq_1, %admgWq_1 : tensor<192x192xf32>
    %adb2Wq_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wq_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWq_1 = stablehlo.multiply %adb2Wq_1, %Wq_1v : tensor<192x192xf32>
    %adg2Wq_1 = stablehlo.multiply %vitb1_mdWQ, %vitb1_mdWQ : tensor<192x192xf32>
    %advgWq_1 = stablehlo.multiply %adob2Wq_1, %adg2Wq_1 : tensor<192x192xf32>
    %advnWq_1 = stablehlo.add %advsWq_1, %advgWq_1 : tensor<192x192xf32>
    %adbc1Wq_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wq_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWq_1 = stablehlo.divide %admnWq_1, %adbc1Wq_1 : tensor<192x192xf32>
    %advhWq_1 = stablehlo.divide %advnWq_1, %adbc2Wq_1 : tensor<192x192xf32>
    %adlrWq_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWq_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWq_1 = stablehlo.sqrt %advhWq_1 : tensor<192x192xf32>
    %addenWq_1 = stablehlo.add %adsqWq_1, %adepsWq_1 : tensor<192x192xf32>
    %adratWq_1 = stablehlo.divide %admhWq_1, %addenWq_1 : tensor<192x192xf32>
    %adstWq_1 = stablehlo.multiply %adlrWq_1, %adratWq_1 : tensor<192x192xf32>
    %adsubWq_1 = stablehlo.subtract %Wq_1, %adstWq_1 : tensor<192x192xf32>
    %adwdWq_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWq_1 = stablehlo.multiply %adwdWq_1, %adlrWq_1 : tensor<192x192xf32>
    %adwdpWq_1 = stablehlo.multiply %adwdlrWq_1, %Wq_1 : tensor<192x192xf32>
    %adnewWq_1 = stablehlo.subtract %adsubWq_1, %adwdpWq_1 : tensor<192x192xf32>
    %adb1bq_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bq_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbq_1 = stablehlo.multiply %adb1bq_1, %bq_1m : tensor<192xf32>
    %admgbq_1 = stablehlo.multiply %adob1bq_1, %vitb1_mdbQ : tensor<192xf32>
    %admnbq_1 = stablehlo.add %admsbq_1, %admgbq_1 : tensor<192xf32>
    %adb2bq_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bq_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbq_1 = stablehlo.multiply %adb2bq_1, %bq_1v : tensor<192xf32>
    %adg2bq_1 = stablehlo.multiply %vitb1_mdbQ, %vitb1_mdbQ : tensor<192xf32>
    %advgbq_1 = stablehlo.multiply %adob2bq_1, %adg2bq_1 : tensor<192xf32>
    %advnbq_1 = stablehlo.add %advsbq_1, %advgbq_1 : tensor<192xf32>
    %adbc1bq_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bq_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbq_1 = stablehlo.divide %admnbq_1, %adbc1bq_1 : tensor<192xf32>
    %advhbq_1 = stablehlo.divide %advnbq_1, %adbc2bq_1 : tensor<192xf32>
    %adlrbq_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbq_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbq_1 = stablehlo.sqrt %advhbq_1 : tensor<192xf32>
    %addenbq_1 = stablehlo.add %adsqbq_1, %adepsbq_1 : tensor<192xf32>
    %adratbq_1 = stablehlo.divide %admhbq_1, %addenbq_1 : tensor<192xf32>
    %adstbq_1 = stablehlo.multiply %adlrbq_1, %adratbq_1 : tensor<192xf32>
    %adsubbq_1 = stablehlo.subtract %bq_1, %adstbq_1 : tensor<192xf32>
    %adwdbq_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbq_1 = stablehlo.multiply %adwdbq_1, %adlrbq_1 : tensor<192xf32>
    %adwdpbq_1 = stablehlo.multiply %adwdlrbq_1, %bq_1 : tensor<192xf32>
    %adnewbq_1 = stablehlo.subtract %adsubbq_1, %adwdpbq_1 : tensor<192xf32>
    %adb1Wk_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wk_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWk_1 = stablehlo.multiply %adb1Wk_1, %Wk_1m : tensor<192x192xf32>
    %admgWk_1 = stablehlo.multiply %adob1Wk_1, %vitb1_mdWK : tensor<192x192xf32>
    %admnWk_1 = stablehlo.add %admsWk_1, %admgWk_1 : tensor<192x192xf32>
    %adb2Wk_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wk_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWk_1 = stablehlo.multiply %adb2Wk_1, %Wk_1v : tensor<192x192xf32>
    %adg2Wk_1 = stablehlo.multiply %vitb1_mdWK, %vitb1_mdWK : tensor<192x192xf32>
    %advgWk_1 = stablehlo.multiply %adob2Wk_1, %adg2Wk_1 : tensor<192x192xf32>
    %advnWk_1 = stablehlo.add %advsWk_1, %advgWk_1 : tensor<192x192xf32>
    %adbc1Wk_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wk_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWk_1 = stablehlo.divide %admnWk_1, %adbc1Wk_1 : tensor<192x192xf32>
    %advhWk_1 = stablehlo.divide %advnWk_1, %adbc2Wk_1 : tensor<192x192xf32>
    %adlrWk_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWk_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWk_1 = stablehlo.sqrt %advhWk_1 : tensor<192x192xf32>
    %addenWk_1 = stablehlo.add %adsqWk_1, %adepsWk_1 : tensor<192x192xf32>
    %adratWk_1 = stablehlo.divide %admhWk_1, %addenWk_1 : tensor<192x192xf32>
    %adstWk_1 = stablehlo.multiply %adlrWk_1, %adratWk_1 : tensor<192x192xf32>
    %adsubWk_1 = stablehlo.subtract %Wk_1, %adstWk_1 : tensor<192x192xf32>
    %adwdWk_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWk_1 = stablehlo.multiply %adwdWk_1, %adlrWk_1 : tensor<192x192xf32>
    %adwdpWk_1 = stablehlo.multiply %adwdlrWk_1, %Wk_1 : tensor<192x192xf32>
    %adnewWk_1 = stablehlo.subtract %adsubWk_1, %adwdpWk_1 : tensor<192x192xf32>
    %adb1bk_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bk_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbk_1 = stablehlo.multiply %adb1bk_1, %bk_1m : tensor<192xf32>
    %admgbk_1 = stablehlo.multiply %adob1bk_1, %vitb1_mdbK : tensor<192xf32>
    %admnbk_1 = stablehlo.add %admsbk_1, %admgbk_1 : tensor<192xf32>
    %adb2bk_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bk_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbk_1 = stablehlo.multiply %adb2bk_1, %bk_1v : tensor<192xf32>
    %adg2bk_1 = stablehlo.multiply %vitb1_mdbK, %vitb1_mdbK : tensor<192xf32>
    %advgbk_1 = stablehlo.multiply %adob2bk_1, %adg2bk_1 : tensor<192xf32>
    %advnbk_1 = stablehlo.add %advsbk_1, %advgbk_1 : tensor<192xf32>
    %adbc1bk_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bk_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbk_1 = stablehlo.divide %admnbk_1, %adbc1bk_1 : tensor<192xf32>
    %advhbk_1 = stablehlo.divide %advnbk_1, %adbc2bk_1 : tensor<192xf32>
    %adlrbk_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbk_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbk_1 = stablehlo.sqrt %advhbk_1 : tensor<192xf32>
    %addenbk_1 = stablehlo.add %adsqbk_1, %adepsbk_1 : tensor<192xf32>
    %adratbk_1 = stablehlo.divide %admhbk_1, %addenbk_1 : tensor<192xf32>
    %adstbk_1 = stablehlo.multiply %adlrbk_1, %adratbk_1 : tensor<192xf32>
    %adsubbk_1 = stablehlo.subtract %bk_1, %adstbk_1 : tensor<192xf32>
    %adwdbk_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbk_1 = stablehlo.multiply %adwdbk_1, %adlrbk_1 : tensor<192xf32>
    %adwdpbk_1 = stablehlo.multiply %adwdlrbk_1, %bk_1 : tensor<192xf32>
    %adnewbk_1 = stablehlo.subtract %adsubbk_1, %adwdpbk_1 : tensor<192xf32>
    %adb1Wv_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wv_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWv_1 = stablehlo.multiply %adb1Wv_1, %Wv_1m : tensor<192x192xf32>
    %admgWv_1 = stablehlo.multiply %adob1Wv_1, %vitb1_mdWV : tensor<192x192xf32>
    %admnWv_1 = stablehlo.add %admsWv_1, %admgWv_1 : tensor<192x192xf32>
    %adb2Wv_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wv_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWv_1 = stablehlo.multiply %adb2Wv_1, %Wv_1v : tensor<192x192xf32>
    %adg2Wv_1 = stablehlo.multiply %vitb1_mdWV, %vitb1_mdWV : tensor<192x192xf32>
    %advgWv_1 = stablehlo.multiply %adob2Wv_1, %adg2Wv_1 : tensor<192x192xf32>
    %advnWv_1 = stablehlo.add %advsWv_1, %advgWv_1 : tensor<192x192xf32>
    %adbc1Wv_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wv_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWv_1 = stablehlo.divide %admnWv_1, %adbc1Wv_1 : tensor<192x192xf32>
    %advhWv_1 = stablehlo.divide %advnWv_1, %adbc2Wv_1 : tensor<192x192xf32>
    %adlrWv_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWv_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWv_1 = stablehlo.sqrt %advhWv_1 : tensor<192x192xf32>
    %addenWv_1 = stablehlo.add %adsqWv_1, %adepsWv_1 : tensor<192x192xf32>
    %adratWv_1 = stablehlo.divide %admhWv_1, %addenWv_1 : tensor<192x192xf32>
    %adstWv_1 = stablehlo.multiply %adlrWv_1, %adratWv_1 : tensor<192x192xf32>
    %adsubWv_1 = stablehlo.subtract %Wv_1, %adstWv_1 : tensor<192x192xf32>
    %adwdWv_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWv_1 = stablehlo.multiply %adwdWv_1, %adlrWv_1 : tensor<192x192xf32>
    %adwdpWv_1 = stablehlo.multiply %adwdlrWv_1, %Wv_1 : tensor<192x192xf32>
    %adnewWv_1 = stablehlo.subtract %adsubWv_1, %adwdpWv_1 : tensor<192x192xf32>
    %adb1bv_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bv_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbv_1 = stablehlo.multiply %adb1bv_1, %bv_1m : tensor<192xf32>
    %admgbv_1 = stablehlo.multiply %adob1bv_1, %vitb1_mdbV : tensor<192xf32>
    %admnbv_1 = stablehlo.add %admsbv_1, %admgbv_1 : tensor<192xf32>
    %adb2bv_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bv_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbv_1 = stablehlo.multiply %adb2bv_1, %bv_1v : tensor<192xf32>
    %adg2bv_1 = stablehlo.multiply %vitb1_mdbV, %vitb1_mdbV : tensor<192xf32>
    %advgbv_1 = stablehlo.multiply %adob2bv_1, %adg2bv_1 : tensor<192xf32>
    %advnbv_1 = stablehlo.add %advsbv_1, %advgbv_1 : tensor<192xf32>
    %adbc1bv_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bv_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbv_1 = stablehlo.divide %admnbv_1, %adbc1bv_1 : tensor<192xf32>
    %advhbv_1 = stablehlo.divide %advnbv_1, %adbc2bv_1 : tensor<192xf32>
    %adlrbv_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbv_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbv_1 = stablehlo.sqrt %advhbv_1 : tensor<192xf32>
    %addenbv_1 = stablehlo.add %adsqbv_1, %adepsbv_1 : tensor<192xf32>
    %adratbv_1 = stablehlo.divide %admhbv_1, %addenbv_1 : tensor<192xf32>
    %adstbv_1 = stablehlo.multiply %adlrbv_1, %adratbv_1 : tensor<192xf32>
    %adsubbv_1 = stablehlo.subtract %bv_1, %adstbv_1 : tensor<192xf32>
    %adwdbv_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbv_1 = stablehlo.multiply %adwdbv_1, %adlrbv_1 : tensor<192xf32>
    %adwdpbv_1 = stablehlo.multiply %adwdlrbv_1, %bv_1 : tensor<192xf32>
    %adnewbv_1 = stablehlo.subtract %adsubbv_1, %adwdpbv_1 : tensor<192xf32>
    %adb1Wo_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wo_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWo_1 = stablehlo.multiply %adb1Wo_1, %Wo_1m : tensor<192x192xf32>
    %admgWo_1 = stablehlo.multiply %adob1Wo_1, %vitb1_mdWo : tensor<192x192xf32>
    %admnWo_1 = stablehlo.add %admsWo_1, %admgWo_1 : tensor<192x192xf32>
    %adb2Wo_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wo_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWo_1 = stablehlo.multiply %adb2Wo_1, %Wo_1v : tensor<192x192xf32>
    %adg2Wo_1 = stablehlo.multiply %vitb1_mdWo, %vitb1_mdWo : tensor<192x192xf32>
    %advgWo_1 = stablehlo.multiply %adob2Wo_1, %adg2Wo_1 : tensor<192x192xf32>
    %advnWo_1 = stablehlo.add %advsWo_1, %advgWo_1 : tensor<192x192xf32>
    %adbc1Wo_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wo_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWo_1 = stablehlo.divide %admnWo_1, %adbc1Wo_1 : tensor<192x192xf32>
    %advhWo_1 = stablehlo.divide %advnWo_1, %adbc2Wo_1 : tensor<192x192xf32>
    %adlrWo_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWo_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWo_1 = stablehlo.sqrt %advhWo_1 : tensor<192x192xf32>
    %addenWo_1 = stablehlo.add %adsqWo_1, %adepsWo_1 : tensor<192x192xf32>
    %adratWo_1 = stablehlo.divide %admhWo_1, %addenWo_1 : tensor<192x192xf32>
    %adstWo_1 = stablehlo.multiply %adlrWo_1, %adratWo_1 : tensor<192x192xf32>
    %adsubWo_1 = stablehlo.subtract %Wo_1, %adstWo_1 : tensor<192x192xf32>
    %adwdWo_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWo_1 = stablehlo.multiply %adwdWo_1, %adlrWo_1 : tensor<192x192xf32>
    %adwdpWo_1 = stablehlo.multiply %adwdlrWo_1, %Wo_1 : tensor<192x192xf32>
    %adnewWo_1 = stablehlo.subtract %adsubWo_1, %adwdpWo_1 : tensor<192x192xf32>
    %adb1bo_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bo_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbo_1 = stablehlo.multiply %adb1bo_1, %bo_1m : tensor<192xf32>
    %admgbo_1 = stablehlo.multiply %adob1bo_1, %vitb1_mdbo : tensor<192xf32>
    %admnbo_1 = stablehlo.add %admsbo_1, %admgbo_1 : tensor<192xf32>
    %adb2bo_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bo_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbo_1 = stablehlo.multiply %adb2bo_1, %bo_1v : tensor<192xf32>
    %adg2bo_1 = stablehlo.multiply %vitb1_mdbo, %vitb1_mdbo : tensor<192xf32>
    %advgbo_1 = stablehlo.multiply %adob2bo_1, %adg2bo_1 : tensor<192xf32>
    %advnbo_1 = stablehlo.add %advsbo_1, %advgbo_1 : tensor<192xf32>
    %adbc1bo_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bo_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbo_1 = stablehlo.divide %admnbo_1, %adbc1bo_1 : tensor<192xf32>
    %advhbo_1 = stablehlo.divide %advnbo_1, %adbc2bo_1 : tensor<192xf32>
    %adlrbo_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbo_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbo_1 = stablehlo.sqrt %advhbo_1 : tensor<192xf32>
    %addenbo_1 = stablehlo.add %adsqbo_1, %adepsbo_1 : tensor<192xf32>
    %adratbo_1 = stablehlo.divide %admhbo_1, %addenbo_1 : tensor<192xf32>
    %adstbo_1 = stablehlo.multiply %adlrbo_1, %adratbo_1 : tensor<192xf32>
    %adsubbo_1 = stablehlo.subtract %bo_1, %adstbo_1 : tensor<192xf32>
    %adwdbo_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbo_1 = stablehlo.multiply %adwdbo_1, %adlrbo_1 : tensor<192xf32>
    %adwdpbo_1 = stablehlo.multiply %adwdlrbo_1, %bo_1 : tensor<192xf32>
    %adnewbo_1 = stablehlo.subtract %adsubbo_1, %adwdpbo_1 : tensor<192xf32>
    %adb1g2_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g2_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg2_1 = stablehlo.multiply %adb1g2_1, %g2_1m : tensor<192xf32>
    %admgg2_1 = stablehlo.multiply %adob1g2_1, %vitb1_2dg : tensor<192xf32>
    %admng2_1 = stablehlo.add %admsg2_1, %admgg2_1 : tensor<192xf32>
    %adb2g2_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g2_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg2_1 = stablehlo.multiply %adb2g2_1, %g2_1v : tensor<192xf32>
    %adg2g2_1 = stablehlo.multiply %vitb1_2dg, %vitb1_2dg : tensor<192xf32>
    %advgg2_1 = stablehlo.multiply %adob2g2_1, %adg2g2_1 : tensor<192xf32>
    %advng2_1 = stablehlo.add %advsg2_1, %advgg2_1 : tensor<192xf32>
    %adbc1g2_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g2_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg2_1 = stablehlo.divide %admng2_1, %adbc1g2_1 : tensor<192xf32>
    %advhg2_1 = stablehlo.divide %advng2_1, %adbc2g2_1 : tensor<192xf32>
    %adlrg2_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg2_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg2_1 = stablehlo.sqrt %advhg2_1 : tensor<192xf32>
    %addeng2_1 = stablehlo.add %adsqg2_1, %adepsg2_1 : tensor<192xf32>
    %adratg2_1 = stablehlo.divide %admhg2_1, %addeng2_1 : tensor<192xf32>
    %adstg2_1 = stablehlo.multiply %adlrg2_1, %adratg2_1 : tensor<192xf32>
    %adsubg2_1 = stablehlo.subtract %g2_1, %adstg2_1 : tensor<192xf32>
    %adwdg2_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg2_1 = stablehlo.multiply %adwdg2_1, %adlrg2_1 : tensor<192xf32>
    %adwdpg2_1 = stablehlo.multiply %adwdlrg2_1, %g2_1 : tensor<192xf32>
    %adnewg2_1 = stablehlo.subtract %adsubg2_1, %adwdpg2_1 : tensor<192xf32>
    %adb1b2_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b2_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb2_1 = stablehlo.multiply %adb1b2_1, %b2_1m : tensor<192xf32>
    %admgb2_1 = stablehlo.multiply %adob1b2_1, %vitb1_2db : tensor<192xf32>
    %admnb2_1 = stablehlo.add %admsb2_1, %admgb2_1 : tensor<192xf32>
    %adb2b2_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b2_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb2_1 = stablehlo.multiply %adb2b2_1, %b2_1v : tensor<192xf32>
    %adg2b2_1 = stablehlo.multiply %vitb1_2db, %vitb1_2db : tensor<192xf32>
    %advgb2_1 = stablehlo.multiply %adob2b2_1, %adg2b2_1 : tensor<192xf32>
    %advnb2_1 = stablehlo.add %advsb2_1, %advgb2_1 : tensor<192xf32>
    %adbc1b2_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b2_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb2_1 = stablehlo.divide %admnb2_1, %adbc1b2_1 : tensor<192xf32>
    %advhb2_1 = stablehlo.divide %advnb2_1, %adbc2b2_1 : tensor<192xf32>
    %adlrb2_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb2_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb2_1 = stablehlo.sqrt %advhb2_1 : tensor<192xf32>
    %addenb2_1 = stablehlo.add %adsqb2_1, %adepsb2_1 : tensor<192xf32>
    %adratb2_1 = stablehlo.divide %admhb2_1, %addenb2_1 : tensor<192xf32>
    %adstb2_1 = stablehlo.multiply %adlrb2_1, %adratb2_1 : tensor<192xf32>
    %adsubb2_1 = stablehlo.subtract %b2_1, %adstb2_1 : tensor<192xf32>
    %adwdb2_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb2_1 = stablehlo.multiply %adwdb2_1, %adlrb2_1 : tensor<192xf32>
    %adwdpb2_1 = stablehlo.multiply %adwdlrb2_1, %b2_1 : tensor<192xf32>
    %adnewb2_1 = stablehlo.subtract %adsubb2_1, %adwdpb2_1 : tensor<192xf32>
    %adb1Wfc1_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob1Wfc1_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admsWfc1_1 = stablehlo.multiply %adb1Wfc1_1, %Wfc1_1m : tensor<192x768xf32>
    %admgWfc1_1 = stablehlo.multiply %adob1Wfc1_1, %vitb1_pdWfc1 : tensor<192x768xf32>
    %admnWfc1_1 = stablehlo.add %admsWfc1_1, %admgWfc1_1 : tensor<192x768xf32>
    %adb2Wfc1_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob2Wfc1_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %advsWfc1_1 = stablehlo.multiply %adb2Wfc1_1, %Wfc1_1v : tensor<192x768xf32>
    %adg2Wfc1_1 = stablehlo.multiply %vitb1_pdWfc1, %vitb1_pdWfc1 : tensor<192x768xf32>
    %advgWfc1_1 = stablehlo.multiply %adob2Wfc1_1, %adg2Wfc1_1 : tensor<192x768xf32>
    %advnWfc1_1 = stablehlo.add %advsWfc1_1, %advgWfc1_1 : tensor<192x768xf32>
    %adbc1Wfc1_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adbc2Wfc1_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admhWfc1_1 = stablehlo.divide %admnWfc1_1, %adbc1Wfc1_1 : tensor<192x768xf32>
    %advhWfc1_1 = stablehlo.divide %advnWfc1_1, %adbc2Wfc1_1 : tensor<192x768xf32>
    %adlrWfc1_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adepsWfc1_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adsqWfc1_1 = stablehlo.sqrt %advhWfc1_1 : tensor<192x768xf32>
    %addenWfc1_1 = stablehlo.add %adsqWfc1_1, %adepsWfc1_1 : tensor<192x768xf32>
    %adratWfc1_1 = stablehlo.divide %admhWfc1_1, %addenWfc1_1 : tensor<192x768xf32>
    %adstWfc1_1 = stablehlo.multiply %adlrWfc1_1, %adratWfc1_1 : tensor<192x768xf32>
    %adsubWfc1_1 = stablehlo.subtract %Wfc1_1, %adstWfc1_1 : tensor<192x768xf32>
    %adwdWfc1_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adwdlrWfc1_1 = stablehlo.multiply %adwdWfc1_1, %adlrWfc1_1 : tensor<192x768xf32>
    %adwdpWfc1_1 = stablehlo.multiply %adwdlrWfc1_1, %Wfc1_1 : tensor<192x768xf32>
    %adnewWfc1_1 = stablehlo.subtract %adsubWfc1_1, %adwdpWfc1_1 : tensor<192x768xf32>
    %adb1bfc1_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1bfc1_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admsbfc1_1 = stablehlo.multiply %adb1bfc1_1, %bfc1_1m : tensor<768xf32>
    %admgbfc1_1 = stablehlo.multiply %adob1bfc1_1, %vitb1_pdbfc1 : tensor<768xf32>
    %admnbfc1_1 = stablehlo.add %admsbfc1_1, %admgbfc1_1 : tensor<768xf32>
    %adb2bfc1_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2bfc1_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advsbfc1_1 = stablehlo.multiply %adb2bfc1_1, %bfc1_1v : tensor<768xf32>
    %adg2bfc1_1 = stablehlo.multiply %vitb1_pdbfc1, %vitb1_pdbfc1 : tensor<768xf32>
    %advgbfc1_1 = stablehlo.multiply %adob2bfc1_1, %adg2bfc1_1 : tensor<768xf32>
    %advnbfc1_1 = stablehlo.add %advsbfc1_1, %advgbfc1_1 : tensor<768xf32>
    %adbc1bfc1_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2bfc1_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhbfc1_1 = stablehlo.divide %admnbfc1_1, %adbc1bfc1_1 : tensor<768xf32>
    %advhbfc1_1 = stablehlo.divide %advnbfc1_1, %adbc2bfc1_1 : tensor<768xf32>
    %adlrbfc1_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepsbfc1_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqbfc1_1 = stablehlo.sqrt %advhbfc1_1 : tensor<768xf32>
    %addenbfc1_1 = stablehlo.add %adsqbfc1_1, %adepsbfc1_1 : tensor<768xf32>
    %adratbfc1_1 = stablehlo.divide %admhbfc1_1, %addenbfc1_1 : tensor<768xf32>
    %adstbfc1_1 = stablehlo.multiply %adlrbfc1_1, %adratbfc1_1 : tensor<768xf32>
    %adsubbfc1_1 = stablehlo.subtract %bfc1_1, %adstbfc1_1 : tensor<768xf32>
    %adwdbfc1_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrbfc1_1 = stablehlo.multiply %adwdbfc1_1, %adlrbfc1_1 : tensor<768xf32>
    %adwdpbfc1_1 = stablehlo.multiply %adwdlrbfc1_1, %bfc1_1 : tensor<768xf32>
    %adnewbfc1_1 = stablehlo.subtract %adsubbfc1_1, %adwdpbfc1_1 : tensor<768xf32>
    %adb1Wfc2_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob1Wfc2_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admsWfc2_1 = stablehlo.multiply %adb1Wfc2_1, %Wfc2_1m : tensor<768x192xf32>
    %admgWfc2_1 = stablehlo.multiply %adob1Wfc2_1, %vitb1_pdWfc2 : tensor<768x192xf32>
    %admnWfc2_1 = stablehlo.add %admsWfc2_1, %admgWfc2_1 : tensor<768x192xf32>
    %adb2Wfc2_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob2Wfc2_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %advsWfc2_1 = stablehlo.multiply %adb2Wfc2_1, %Wfc2_1v : tensor<768x192xf32>
    %adg2Wfc2_1 = stablehlo.multiply %vitb1_pdWfc2, %vitb1_pdWfc2 : tensor<768x192xf32>
    %advgWfc2_1 = stablehlo.multiply %adob2Wfc2_1, %adg2Wfc2_1 : tensor<768x192xf32>
    %advnWfc2_1 = stablehlo.add %advsWfc2_1, %advgWfc2_1 : tensor<768x192xf32>
    %adbc1Wfc2_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adbc2Wfc2_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admhWfc2_1 = stablehlo.divide %admnWfc2_1, %adbc1Wfc2_1 : tensor<768x192xf32>
    %advhWfc2_1 = stablehlo.divide %advnWfc2_1, %adbc2Wfc2_1 : tensor<768x192xf32>
    %adlrWfc2_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adepsWfc2_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adsqWfc2_1 = stablehlo.sqrt %advhWfc2_1 : tensor<768x192xf32>
    %addenWfc2_1 = stablehlo.add %adsqWfc2_1, %adepsWfc2_1 : tensor<768x192xf32>
    %adratWfc2_1 = stablehlo.divide %admhWfc2_1, %addenWfc2_1 : tensor<768x192xf32>
    %adstWfc2_1 = stablehlo.multiply %adlrWfc2_1, %adratWfc2_1 : tensor<768x192xf32>
    %adsubWfc2_1 = stablehlo.subtract %Wfc2_1, %adstWfc2_1 : tensor<768x192xf32>
    %adwdWfc2_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adwdlrWfc2_1 = stablehlo.multiply %adwdWfc2_1, %adlrWfc2_1 : tensor<768x192xf32>
    %adwdpWfc2_1 = stablehlo.multiply %adwdlrWfc2_1, %Wfc2_1 : tensor<768x192xf32>
    %adnewWfc2_1 = stablehlo.subtract %adsubWfc2_1, %adwdpWfc2_1 : tensor<768x192xf32>
    %adb1bfc2_1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bfc2_1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbfc2_1 = stablehlo.multiply %adb1bfc2_1, %bfc2_1m : tensor<192xf32>
    %admgbfc2_1 = stablehlo.multiply %adob1bfc2_1, %vitb1_pdbfc2 : tensor<192xf32>
    %admnbfc2_1 = stablehlo.add %admsbfc2_1, %admgbfc2_1 : tensor<192xf32>
    %adb2bfc2_1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bfc2_1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbfc2_1 = stablehlo.multiply %adb2bfc2_1, %bfc2_1v : tensor<192xf32>
    %adg2bfc2_1 = stablehlo.multiply %vitb1_pdbfc2, %vitb1_pdbfc2 : tensor<192xf32>
    %advgbfc2_1 = stablehlo.multiply %adob2bfc2_1, %adg2bfc2_1 : tensor<192xf32>
    %advnbfc2_1 = stablehlo.add %advsbfc2_1, %advgbfc2_1 : tensor<192xf32>
    %adbc1bfc2_1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bfc2_1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbfc2_1 = stablehlo.divide %admnbfc2_1, %adbc1bfc2_1 : tensor<192xf32>
    %advhbfc2_1 = stablehlo.divide %advnbfc2_1, %adbc2bfc2_1 : tensor<192xf32>
    %adlrbfc2_1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbfc2_1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbfc2_1 = stablehlo.sqrt %advhbfc2_1 : tensor<192xf32>
    %addenbfc2_1 = stablehlo.add %adsqbfc2_1, %adepsbfc2_1 : tensor<192xf32>
    %adratbfc2_1 = stablehlo.divide %admhbfc2_1, %addenbfc2_1 : tensor<192xf32>
    %adstbfc2_1 = stablehlo.multiply %adlrbfc2_1, %adratbfc2_1 : tensor<192xf32>
    %adsubbfc2_1 = stablehlo.subtract %bfc2_1, %adstbfc2_1 : tensor<192xf32>
    %adwdbfc2_1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbfc2_1 = stablehlo.multiply %adwdbfc2_1, %adlrbfc2_1 : tensor<192xf32>
    %adwdpbfc2_1 = stablehlo.multiply %adwdlrbfc2_1, %bfc2_1 : tensor<192xf32>
    %adnewbfc2_1 = stablehlo.subtract %adsubbfc2_1, %adwdpbfc2_1 : tensor<192xf32>
    %adb1g1_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g1_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg1_2 = stablehlo.multiply %adb1g1_2, %g1_2m : tensor<192xf32>
    %admgg1_2 = stablehlo.multiply %adob1g1_2, %vitb2_1dg : tensor<192xf32>
    %admng1_2 = stablehlo.add %admsg1_2, %admgg1_2 : tensor<192xf32>
    %adb2g1_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g1_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg1_2 = stablehlo.multiply %adb2g1_2, %g1_2v : tensor<192xf32>
    %adg2g1_2 = stablehlo.multiply %vitb2_1dg, %vitb2_1dg : tensor<192xf32>
    %advgg1_2 = stablehlo.multiply %adob2g1_2, %adg2g1_2 : tensor<192xf32>
    %advng1_2 = stablehlo.add %advsg1_2, %advgg1_2 : tensor<192xf32>
    %adbc1g1_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g1_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg1_2 = stablehlo.divide %admng1_2, %adbc1g1_2 : tensor<192xf32>
    %advhg1_2 = stablehlo.divide %advng1_2, %adbc2g1_2 : tensor<192xf32>
    %adlrg1_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg1_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg1_2 = stablehlo.sqrt %advhg1_2 : tensor<192xf32>
    %addeng1_2 = stablehlo.add %adsqg1_2, %adepsg1_2 : tensor<192xf32>
    %adratg1_2 = stablehlo.divide %admhg1_2, %addeng1_2 : tensor<192xf32>
    %adstg1_2 = stablehlo.multiply %adlrg1_2, %adratg1_2 : tensor<192xf32>
    %adsubg1_2 = stablehlo.subtract %g1_2, %adstg1_2 : tensor<192xf32>
    %adwdg1_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg1_2 = stablehlo.multiply %adwdg1_2, %adlrg1_2 : tensor<192xf32>
    %adwdpg1_2 = stablehlo.multiply %adwdlrg1_2, %g1_2 : tensor<192xf32>
    %adnewg1_2 = stablehlo.subtract %adsubg1_2, %adwdpg1_2 : tensor<192xf32>
    %adb1b1_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b1_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb1_2 = stablehlo.multiply %adb1b1_2, %b1_2m : tensor<192xf32>
    %admgb1_2 = stablehlo.multiply %adob1b1_2, %vitb2_1db : tensor<192xf32>
    %admnb1_2 = stablehlo.add %admsb1_2, %admgb1_2 : tensor<192xf32>
    %adb2b1_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b1_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb1_2 = stablehlo.multiply %adb2b1_2, %b1_2v : tensor<192xf32>
    %adg2b1_2 = stablehlo.multiply %vitb2_1db, %vitb2_1db : tensor<192xf32>
    %advgb1_2 = stablehlo.multiply %adob2b1_2, %adg2b1_2 : tensor<192xf32>
    %advnb1_2 = stablehlo.add %advsb1_2, %advgb1_2 : tensor<192xf32>
    %adbc1b1_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b1_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb1_2 = stablehlo.divide %admnb1_2, %adbc1b1_2 : tensor<192xf32>
    %advhb1_2 = stablehlo.divide %advnb1_2, %adbc2b1_2 : tensor<192xf32>
    %adlrb1_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb1_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb1_2 = stablehlo.sqrt %advhb1_2 : tensor<192xf32>
    %addenb1_2 = stablehlo.add %adsqb1_2, %adepsb1_2 : tensor<192xf32>
    %adratb1_2 = stablehlo.divide %admhb1_2, %addenb1_2 : tensor<192xf32>
    %adstb1_2 = stablehlo.multiply %adlrb1_2, %adratb1_2 : tensor<192xf32>
    %adsubb1_2 = stablehlo.subtract %b1_2, %adstb1_2 : tensor<192xf32>
    %adwdb1_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb1_2 = stablehlo.multiply %adwdb1_2, %adlrb1_2 : tensor<192xf32>
    %adwdpb1_2 = stablehlo.multiply %adwdlrb1_2, %b1_2 : tensor<192xf32>
    %adnewb1_2 = stablehlo.subtract %adsubb1_2, %adwdpb1_2 : tensor<192xf32>
    %adb1Wq_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wq_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWq_2 = stablehlo.multiply %adb1Wq_2, %Wq_2m : tensor<192x192xf32>
    %admgWq_2 = stablehlo.multiply %adob1Wq_2, %vitb2_mdWQ : tensor<192x192xf32>
    %admnWq_2 = stablehlo.add %admsWq_2, %admgWq_2 : tensor<192x192xf32>
    %adb2Wq_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wq_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWq_2 = stablehlo.multiply %adb2Wq_2, %Wq_2v : tensor<192x192xf32>
    %adg2Wq_2 = stablehlo.multiply %vitb2_mdWQ, %vitb2_mdWQ : tensor<192x192xf32>
    %advgWq_2 = stablehlo.multiply %adob2Wq_2, %adg2Wq_2 : tensor<192x192xf32>
    %advnWq_2 = stablehlo.add %advsWq_2, %advgWq_2 : tensor<192x192xf32>
    %adbc1Wq_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wq_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWq_2 = stablehlo.divide %admnWq_2, %adbc1Wq_2 : tensor<192x192xf32>
    %advhWq_2 = stablehlo.divide %advnWq_2, %adbc2Wq_2 : tensor<192x192xf32>
    %adlrWq_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWq_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWq_2 = stablehlo.sqrt %advhWq_2 : tensor<192x192xf32>
    %addenWq_2 = stablehlo.add %adsqWq_2, %adepsWq_2 : tensor<192x192xf32>
    %adratWq_2 = stablehlo.divide %admhWq_2, %addenWq_2 : tensor<192x192xf32>
    %adstWq_2 = stablehlo.multiply %adlrWq_2, %adratWq_2 : tensor<192x192xf32>
    %adsubWq_2 = stablehlo.subtract %Wq_2, %adstWq_2 : tensor<192x192xf32>
    %adwdWq_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWq_2 = stablehlo.multiply %adwdWq_2, %adlrWq_2 : tensor<192x192xf32>
    %adwdpWq_2 = stablehlo.multiply %adwdlrWq_2, %Wq_2 : tensor<192x192xf32>
    %adnewWq_2 = stablehlo.subtract %adsubWq_2, %adwdpWq_2 : tensor<192x192xf32>
    %adb1bq_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bq_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbq_2 = stablehlo.multiply %adb1bq_2, %bq_2m : tensor<192xf32>
    %admgbq_2 = stablehlo.multiply %adob1bq_2, %vitb2_mdbQ : tensor<192xf32>
    %admnbq_2 = stablehlo.add %admsbq_2, %admgbq_2 : tensor<192xf32>
    %adb2bq_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bq_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbq_2 = stablehlo.multiply %adb2bq_2, %bq_2v : tensor<192xf32>
    %adg2bq_2 = stablehlo.multiply %vitb2_mdbQ, %vitb2_mdbQ : tensor<192xf32>
    %advgbq_2 = stablehlo.multiply %adob2bq_2, %adg2bq_2 : tensor<192xf32>
    %advnbq_2 = stablehlo.add %advsbq_2, %advgbq_2 : tensor<192xf32>
    %adbc1bq_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bq_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbq_2 = stablehlo.divide %admnbq_2, %adbc1bq_2 : tensor<192xf32>
    %advhbq_2 = stablehlo.divide %advnbq_2, %adbc2bq_2 : tensor<192xf32>
    %adlrbq_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbq_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbq_2 = stablehlo.sqrt %advhbq_2 : tensor<192xf32>
    %addenbq_2 = stablehlo.add %adsqbq_2, %adepsbq_2 : tensor<192xf32>
    %adratbq_2 = stablehlo.divide %admhbq_2, %addenbq_2 : tensor<192xf32>
    %adstbq_2 = stablehlo.multiply %adlrbq_2, %adratbq_2 : tensor<192xf32>
    %adsubbq_2 = stablehlo.subtract %bq_2, %adstbq_2 : tensor<192xf32>
    %adwdbq_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbq_2 = stablehlo.multiply %adwdbq_2, %adlrbq_2 : tensor<192xf32>
    %adwdpbq_2 = stablehlo.multiply %adwdlrbq_2, %bq_2 : tensor<192xf32>
    %adnewbq_2 = stablehlo.subtract %adsubbq_2, %adwdpbq_2 : tensor<192xf32>
    %adb1Wk_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wk_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWk_2 = stablehlo.multiply %adb1Wk_2, %Wk_2m : tensor<192x192xf32>
    %admgWk_2 = stablehlo.multiply %adob1Wk_2, %vitb2_mdWK : tensor<192x192xf32>
    %admnWk_2 = stablehlo.add %admsWk_2, %admgWk_2 : tensor<192x192xf32>
    %adb2Wk_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wk_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWk_2 = stablehlo.multiply %adb2Wk_2, %Wk_2v : tensor<192x192xf32>
    %adg2Wk_2 = stablehlo.multiply %vitb2_mdWK, %vitb2_mdWK : tensor<192x192xf32>
    %advgWk_2 = stablehlo.multiply %adob2Wk_2, %adg2Wk_2 : tensor<192x192xf32>
    %advnWk_2 = stablehlo.add %advsWk_2, %advgWk_2 : tensor<192x192xf32>
    %adbc1Wk_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wk_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWk_2 = stablehlo.divide %admnWk_2, %adbc1Wk_2 : tensor<192x192xf32>
    %advhWk_2 = stablehlo.divide %advnWk_2, %adbc2Wk_2 : tensor<192x192xf32>
    %adlrWk_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWk_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWk_2 = stablehlo.sqrt %advhWk_2 : tensor<192x192xf32>
    %addenWk_2 = stablehlo.add %adsqWk_2, %adepsWk_2 : tensor<192x192xf32>
    %adratWk_2 = stablehlo.divide %admhWk_2, %addenWk_2 : tensor<192x192xf32>
    %adstWk_2 = stablehlo.multiply %adlrWk_2, %adratWk_2 : tensor<192x192xf32>
    %adsubWk_2 = stablehlo.subtract %Wk_2, %adstWk_2 : tensor<192x192xf32>
    %adwdWk_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWk_2 = stablehlo.multiply %adwdWk_2, %adlrWk_2 : tensor<192x192xf32>
    %adwdpWk_2 = stablehlo.multiply %adwdlrWk_2, %Wk_2 : tensor<192x192xf32>
    %adnewWk_2 = stablehlo.subtract %adsubWk_2, %adwdpWk_2 : tensor<192x192xf32>
    %adb1bk_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bk_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbk_2 = stablehlo.multiply %adb1bk_2, %bk_2m : tensor<192xf32>
    %admgbk_2 = stablehlo.multiply %adob1bk_2, %vitb2_mdbK : tensor<192xf32>
    %admnbk_2 = stablehlo.add %admsbk_2, %admgbk_2 : tensor<192xf32>
    %adb2bk_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bk_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbk_2 = stablehlo.multiply %adb2bk_2, %bk_2v : tensor<192xf32>
    %adg2bk_2 = stablehlo.multiply %vitb2_mdbK, %vitb2_mdbK : tensor<192xf32>
    %advgbk_2 = stablehlo.multiply %adob2bk_2, %adg2bk_2 : tensor<192xf32>
    %advnbk_2 = stablehlo.add %advsbk_2, %advgbk_2 : tensor<192xf32>
    %adbc1bk_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bk_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbk_2 = stablehlo.divide %admnbk_2, %adbc1bk_2 : tensor<192xf32>
    %advhbk_2 = stablehlo.divide %advnbk_2, %adbc2bk_2 : tensor<192xf32>
    %adlrbk_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbk_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbk_2 = stablehlo.sqrt %advhbk_2 : tensor<192xf32>
    %addenbk_2 = stablehlo.add %adsqbk_2, %adepsbk_2 : tensor<192xf32>
    %adratbk_2 = stablehlo.divide %admhbk_2, %addenbk_2 : tensor<192xf32>
    %adstbk_2 = stablehlo.multiply %adlrbk_2, %adratbk_2 : tensor<192xf32>
    %adsubbk_2 = stablehlo.subtract %bk_2, %adstbk_2 : tensor<192xf32>
    %adwdbk_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbk_2 = stablehlo.multiply %adwdbk_2, %adlrbk_2 : tensor<192xf32>
    %adwdpbk_2 = stablehlo.multiply %adwdlrbk_2, %bk_2 : tensor<192xf32>
    %adnewbk_2 = stablehlo.subtract %adsubbk_2, %adwdpbk_2 : tensor<192xf32>
    %adb1Wv_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wv_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWv_2 = stablehlo.multiply %adb1Wv_2, %Wv_2m : tensor<192x192xf32>
    %admgWv_2 = stablehlo.multiply %adob1Wv_2, %vitb2_mdWV : tensor<192x192xf32>
    %admnWv_2 = stablehlo.add %admsWv_2, %admgWv_2 : tensor<192x192xf32>
    %adb2Wv_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wv_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWv_2 = stablehlo.multiply %adb2Wv_2, %Wv_2v : tensor<192x192xf32>
    %adg2Wv_2 = stablehlo.multiply %vitb2_mdWV, %vitb2_mdWV : tensor<192x192xf32>
    %advgWv_2 = stablehlo.multiply %adob2Wv_2, %adg2Wv_2 : tensor<192x192xf32>
    %advnWv_2 = stablehlo.add %advsWv_2, %advgWv_2 : tensor<192x192xf32>
    %adbc1Wv_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wv_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWv_2 = stablehlo.divide %admnWv_2, %adbc1Wv_2 : tensor<192x192xf32>
    %advhWv_2 = stablehlo.divide %advnWv_2, %adbc2Wv_2 : tensor<192x192xf32>
    %adlrWv_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWv_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWv_2 = stablehlo.sqrt %advhWv_2 : tensor<192x192xf32>
    %addenWv_2 = stablehlo.add %adsqWv_2, %adepsWv_2 : tensor<192x192xf32>
    %adratWv_2 = stablehlo.divide %admhWv_2, %addenWv_2 : tensor<192x192xf32>
    %adstWv_2 = stablehlo.multiply %adlrWv_2, %adratWv_2 : tensor<192x192xf32>
    %adsubWv_2 = stablehlo.subtract %Wv_2, %adstWv_2 : tensor<192x192xf32>
    %adwdWv_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWv_2 = stablehlo.multiply %adwdWv_2, %adlrWv_2 : tensor<192x192xf32>
    %adwdpWv_2 = stablehlo.multiply %adwdlrWv_2, %Wv_2 : tensor<192x192xf32>
    %adnewWv_2 = stablehlo.subtract %adsubWv_2, %adwdpWv_2 : tensor<192x192xf32>
    %adb1bv_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bv_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbv_2 = stablehlo.multiply %adb1bv_2, %bv_2m : tensor<192xf32>
    %admgbv_2 = stablehlo.multiply %adob1bv_2, %vitb2_mdbV : tensor<192xf32>
    %admnbv_2 = stablehlo.add %admsbv_2, %admgbv_2 : tensor<192xf32>
    %adb2bv_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bv_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbv_2 = stablehlo.multiply %adb2bv_2, %bv_2v : tensor<192xf32>
    %adg2bv_2 = stablehlo.multiply %vitb2_mdbV, %vitb2_mdbV : tensor<192xf32>
    %advgbv_2 = stablehlo.multiply %adob2bv_2, %adg2bv_2 : tensor<192xf32>
    %advnbv_2 = stablehlo.add %advsbv_2, %advgbv_2 : tensor<192xf32>
    %adbc1bv_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bv_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbv_2 = stablehlo.divide %admnbv_2, %adbc1bv_2 : tensor<192xf32>
    %advhbv_2 = stablehlo.divide %advnbv_2, %adbc2bv_2 : tensor<192xf32>
    %adlrbv_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbv_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbv_2 = stablehlo.sqrt %advhbv_2 : tensor<192xf32>
    %addenbv_2 = stablehlo.add %adsqbv_2, %adepsbv_2 : tensor<192xf32>
    %adratbv_2 = stablehlo.divide %admhbv_2, %addenbv_2 : tensor<192xf32>
    %adstbv_2 = stablehlo.multiply %adlrbv_2, %adratbv_2 : tensor<192xf32>
    %adsubbv_2 = stablehlo.subtract %bv_2, %adstbv_2 : tensor<192xf32>
    %adwdbv_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbv_2 = stablehlo.multiply %adwdbv_2, %adlrbv_2 : tensor<192xf32>
    %adwdpbv_2 = stablehlo.multiply %adwdlrbv_2, %bv_2 : tensor<192xf32>
    %adnewbv_2 = stablehlo.subtract %adsubbv_2, %adwdpbv_2 : tensor<192xf32>
    %adb1Wo_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wo_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWo_2 = stablehlo.multiply %adb1Wo_2, %Wo_2m : tensor<192x192xf32>
    %admgWo_2 = stablehlo.multiply %adob1Wo_2, %vitb2_mdWo : tensor<192x192xf32>
    %admnWo_2 = stablehlo.add %admsWo_2, %admgWo_2 : tensor<192x192xf32>
    %adb2Wo_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wo_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWo_2 = stablehlo.multiply %adb2Wo_2, %Wo_2v : tensor<192x192xf32>
    %adg2Wo_2 = stablehlo.multiply %vitb2_mdWo, %vitb2_mdWo : tensor<192x192xf32>
    %advgWo_2 = stablehlo.multiply %adob2Wo_2, %adg2Wo_2 : tensor<192x192xf32>
    %advnWo_2 = stablehlo.add %advsWo_2, %advgWo_2 : tensor<192x192xf32>
    %adbc1Wo_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wo_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWo_2 = stablehlo.divide %admnWo_2, %adbc1Wo_2 : tensor<192x192xf32>
    %advhWo_2 = stablehlo.divide %advnWo_2, %adbc2Wo_2 : tensor<192x192xf32>
    %adlrWo_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWo_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWo_2 = stablehlo.sqrt %advhWo_2 : tensor<192x192xf32>
    %addenWo_2 = stablehlo.add %adsqWo_2, %adepsWo_2 : tensor<192x192xf32>
    %adratWo_2 = stablehlo.divide %admhWo_2, %addenWo_2 : tensor<192x192xf32>
    %adstWo_2 = stablehlo.multiply %adlrWo_2, %adratWo_2 : tensor<192x192xf32>
    %adsubWo_2 = stablehlo.subtract %Wo_2, %adstWo_2 : tensor<192x192xf32>
    %adwdWo_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWo_2 = stablehlo.multiply %adwdWo_2, %adlrWo_2 : tensor<192x192xf32>
    %adwdpWo_2 = stablehlo.multiply %adwdlrWo_2, %Wo_2 : tensor<192x192xf32>
    %adnewWo_2 = stablehlo.subtract %adsubWo_2, %adwdpWo_2 : tensor<192x192xf32>
    %adb1bo_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bo_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbo_2 = stablehlo.multiply %adb1bo_2, %bo_2m : tensor<192xf32>
    %admgbo_2 = stablehlo.multiply %adob1bo_2, %vitb2_mdbo : tensor<192xf32>
    %admnbo_2 = stablehlo.add %admsbo_2, %admgbo_2 : tensor<192xf32>
    %adb2bo_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bo_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbo_2 = stablehlo.multiply %adb2bo_2, %bo_2v : tensor<192xf32>
    %adg2bo_2 = stablehlo.multiply %vitb2_mdbo, %vitb2_mdbo : tensor<192xf32>
    %advgbo_2 = stablehlo.multiply %adob2bo_2, %adg2bo_2 : tensor<192xf32>
    %advnbo_2 = stablehlo.add %advsbo_2, %advgbo_2 : tensor<192xf32>
    %adbc1bo_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bo_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbo_2 = stablehlo.divide %admnbo_2, %adbc1bo_2 : tensor<192xf32>
    %advhbo_2 = stablehlo.divide %advnbo_2, %adbc2bo_2 : tensor<192xf32>
    %adlrbo_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbo_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbo_2 = stablehlo.sqrt %advhbo_2 : tensor<192xf32>
    %addenbo_2 = stablehlo.add %adsqbo_2, %adepsbo_2 : tensor<192xf32>
    %adratbo_2 = stablehlo.divide %admhbo_2, %addenbo_2 : tensor<192xf32>
    %adstbo_2 = stablehlo.multiply %adlrbo_2, %adratbo_2 : tensor<192xf32>
    %adsubbo_2 = stablehlo.subtract %bo_2, %adstbo_2 : tensor<192xf32>
    %adwdbo_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbo_2 = stablehlo.multiply %adwdbo_2, %adlrbo_2 : tensor<192xf32>
    %adwdpbo_2 = stablehlo.multiply %adwdlrbo_2, %bo_2 : tensor<192xf32>
    %adnewbo_2 = stablehlo.subtract %adsubbo_2, %adwdpbo_2 : tensor<192xf32>
    %adb1g2_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g2_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg2_2 = stablehlo.multiply %adb1g2_2, %g2_2m : tensor<192xf32>
    %admgg2_2 = stablehlo.multiply %adob1g2_2, %vitb2_2dg : tensor<192xf32>
    %admng2_2 = stablehlo.add %admsg2_2, %admgg2_2 : tensor<192xf32>
    %adb2g2_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g2_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg2_2 = stablehlo.multiply %adb2g2_2, %g2_2v : tensor<192xf32>
    %adg2g2_2 = stablehlo.multiply %vitb2_2dg, %vitb2_2dg : tensor<192xf32>
    %advgg2_2 = stablehlo.multiply %adob2g2_2, %adg2g2_2 : tensor<192xf32>
    %advng2_2 = stablehlo.add %advsg2_2, %advgg2_2 : tensor<192xf32>
    %adbc1g2_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g2_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg2_2 = stablehlo.divide %admng2_2, %adbc1g2_2 : tensor<192xf32>
    %advhg2_2 = stablehlo.divide %advng2_2, %adbc2g2_2 : tensor<192xf32>
    %adlrg2_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg2_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg2_2 = stablehlo.sqrt %advhg2_2 : tensor<192xf32>
    %addeng2_2 = stablehlo.add %adsqg2_2, %adepsg2_2 : tensor<192xf32>
    %adratg2_2 = stablehlo.divide %admhg2_2, %addeng2_2 : tensor<192xf32>
    %adstg2_2 = stablehlo.multiply %adlrg2_2, %adratg2_2 : tensor<192xf32>
    %adsubg2_2 = stablehlo.subtract %g2_2, %adstg2_2 : tensor<192xf32>
    %adwdg2_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg2_2 = stablehlo.multiply %adwdg2_2, %adlrg2_2 : tensor<192xf32>
    %adwdpg2_2 = stablehlo.multiply %adwdlrg2_2, %g2_2 : tensor<192xf32>
    %adnewg2_2 = stablehlo.subtract %adsubg2_2, %adwdpg2_2 : tensor<192xf32>
    %adb1b2_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b2_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb2_2 = stablehlo.multiply %adb1b2_2, %b2_2m : tensor<192xf32>
    %admgb2_2 = stablehlo.multiply %adob1b2_2, %vitb2_2db : tensor<192xf32>
    %admnb2_2 = stablehlo.add %admsb2_2, %admgb2_2 : tensor<192xf32>
    %adb2b2_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b2_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb2_2 = stablehlo.multiply %adb2b2_2, %b2_2v : tensor<192xf32>
    %adg2b2_2 = stablehlo.multiply %vitb2_2db, %vitb2_2db : tensor<192xf32>
    %advgb2_2 = stablehlo.multiply %adob2b2_2, %adg2b2_2 : tensor<192xf32>
    %advnb2_2 = stablehlo.add %advsb2_2, %advgb2_2 : tensor<192xf32>
    %adbc1b2_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b2_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb2_2 = stablehlo.divide %admnb2_2, %adbc1b2_2 : tensor<192xf32>
    %advhb2_2 = stablehlo.divide %advnb2_2, %adbc2b2_2 : tensor<192xf32>
    %adlrb2_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb2_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb2_2 = stablehlo.sqrt %advhb2_2 : tensor<192xf32>
    %addenb2_2 = stablehlo.add %adsqb2_2, %adepsb2_2 : tensor<192xf32>
    %adratb2_2 = stablehlo.divide %admhb2_2, %addenb2_2 : tensor<192xf32>
    %adstb2_2 = stablehlo.multiply %adlrb2_2, %adratb2_2 : tensor<192xf32>
    %adsubb2_2 = stablehlo.subtract %b2_2, %adstb2_2 : tensor<192xf32>
    %adwdb2_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb2_2 = stablehlo.multiply %adwdb2_2, %adlrb2_2 : tensor<192xf32>
    %adwdpb2_2 = stablehlo.multiply %adwdlrb2_2, %b2_2 : tensor<192xf32>
    %adnewb2_2 = stablehlo.subtract %adsubb2_2, %adwdpb2_2 : tensor<192xf32>
    %adb1Wfc1_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob1Wfc1_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admsWfc1_2 = stablehlo.multiply %adb1Wfc1_2, %Wfc1_2m : tensor<192x768xf32>
    %admgWfc1_2 = stablehlo.multiply %adob1Wfc1_2, %vitb2_pdWfc1 : tensor<192x768xf32>
    %admnWfc1_2 = stablehlo.add %admsWfc1_2, %admgWfc1_2 : tensor<192x768xf32>
    %adb2Wfc1_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob2Wfc1_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %advsWfc1_2 = stablehlo.multiply %adb2Wfc1_2, %Wfc1_2v : tensor<192x768xf32>
    %adg2Wfc1_2 = stablehlo.multiply %vitb2_pdWfc1, %vitb2_pdWfc1 : tensor<192x768xf32>
    %advgWfc1_2 = stablehlo.multiply %adob2Wfc1_2, %adg2Wfc1_2 : tensor<192x768xf32>
    %advnWfc1_2 = stablehlo.add %advsWfc1_2, %advgWfc1_2 : tensor<192x768xf32>
    %adbc1Wfc1_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adbc2Wfc1_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admhWfc1_2 = stablehlo.divide %admnWfc1_2, %adbc1Wfc1_2 : tensor<192x768xf32>
    %advhWfc1_2 = stablehlo.divide %advnWfc1_2, %adbc2Wfc1_2 : tensor<192x768xf32>
    %adlrWfc1_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adepsWfc1_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adsqWfc1_2 = stablehlo.sqrt %advhWfc1_2 : tensor<192x768xf32>
    %addenWfc1_2 = stablehlo.add %adsqWfc1_2, %adepsWfc1_2 : tensor<192x768xf32>
    %adratWfc1_2 = stablehlo.divide %admhWfc1_2, %addenWfc1_2 : tensor<192x768xf32>
    %adstWfc1_2 = stablehlo.multiply %adlrWfc1_2, %adratWfc1_2 : tensor<192x768xf32>
    %adsubWfc1_2 = stablehlo.subtract %Wfc1_2, %adstWfc1_2 : tensor<192x768xf32>
    %adwdWfc1_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adwdlrWfc1_2 = stablehlo.multiply %adwdWfc1_2, %adlrWfc1_2 : tensor<192x768xf32>
    %adwdpWfc1_2 = stablehlo.multiply %adwdlrWfc1_2, %Wfc1_2 : tensor<192x768xf32>
    %adnewWfc1_2 = stablehlo.subtract %adsubWfc1_2, %adwdpWfc1_2 : tensor<192x768xf32>
    %adb1bfc1_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1bfc1_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admsbfc1_2 = stablehlo.multiply %adb1bfc1_2, %bfc1_2m : tensor<768xf32>
    %admgbfc1_2 = stablehlo.multiply %adob1bfc1_2, %vitb2_pdbfc1 : tensor<768xf32>
    %admnbfc1_2 = stablehlo.add %admsbfc1_2, %admgbfc1_2 : tensor<768xf32>
    %adb2bfc1_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2bfc1_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advsbfc1_2 = stablehlo.multiply %adb2bfc1_2, %bfc1_2v : tensor<768xf32>
    %adg2bfc1_2 = stablehlo.multiply %vitb2_pdbfc1, %vitb2_pdbfc1 : tensor<768xf32>
    %advgbfc1_2 = stablehlo.multiply %adob2bfc1_2, %adg2bfc1_2 : tensor<768xf32>
    %advnbfc1_2 = stablehlo.add %advsbfc1_2, %advgbfc1_2 : tensor<768xf32>
    %adbc1bfc1_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2bfc1_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhbfc1_2 = stablehlo.divide %admnbfc1_2, %adbc1bfc1_2 : tensor<768xf32>
    %advhbfc1_2 = stablehlo.divide %advnbfc1_2, %adbc2bfc1_2 : tensor<768xf32>
    %adlrbfc1_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepsbfc1_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqbfc1_2 = stablehlo.sqrt %advhbfc1_2 : tensor<768xf32>
    %addenbfc1_2 = stablehlo.add %adsqbfc1_2, %adepsbfc1_2 : tensor<768xf32>
    %adratbfc1_2 = stablehlo.divide %admhbfc1_2, %addenbfc1_2 : tensor<768xf32>
    %adstbfc1_2 = stablehlo.multiply %adlrbfc1_2, %adratbfc1_2 : tensor<768xf32>
    %adsubbfc1_2 = stablehlo.subtract %bfc1_2, %adstbfc1_2 : tensor<768xf32>
    %adwdbfc1_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrbfc1_2 = stablehlo.multiply %adwdbfc1_2, %adlrbfc1_2 : tensor<768xf32>
    %adwdpbfc1_2 = stablehlo.multiply %adwdlrbfc1_2, %bfc1_2 : tensor<768xf32>
    %adnewbfc1_2 = stablehlo.subtract %adsubbfc1_2, %adwdpbfc1_2 : tensor<768xf32>
    %adb1Wfc2_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob1Wfc2_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admsWfc2_2 = stablehlo.multiply %adb1Wfc2_2, %Wfc2_2m : tensor<768x192xf32>
    %admgWfc2_2 = stablehlo.multiply %adob1Wfc2_2, %vitb2_pdWfc2 : tensor<768x192xf32>
    %admnWfc2_2 = stablehlo.add %admsWfc2_2, %admgWfc2_2 : tensor<768x192xf32>
    %adb2Wfc2_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob2Wfc2_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %advsWfc2_2 = stablehlo.multiply %adb2Wfc2_2, %Wfc2_2v : tensor<768x192xf32>
    %adg2Wfc2_2 = stablehlo.multiply %vitb2_pdWfc2, %vitb2_pdWfc2 : tensor<768x192xf32>
    %advgWfc2_2 = stablehlo.multiply %adob2Wfc2_2, %adg2Wfc2_2 : tensor<768x192xf32>
    %advnWfc2_2 = stablehlo.add %advsWfc2_2, %advgWfc2_2 : tensor<768x192xf32>
    %adbc1Wfc2_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adbc2Wfc2_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admhWfc2_2 = stablehlo.divide %admnWfc2_2, %adbc1Wfc2_2 : tensor<768x192xf32>
    %advhWfc2_2 = stablehlo.divide %advnWfc2_2, %adbc2Wfc2_2 : tensor<768x192xf32>
    %adlrWfc2_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adepsWfc2_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adsqWfc2_2 = stablehlo.sqrt %advhWfc2_2 : tensor<768x192xf32>
    %addenWfc2_2 = stablehlo.add %adsqWfc2_2, %adepsWfc2_2 : tensor<768x192xf32>
    %adratWfc2_2 = stablehlo.divide %admhWfc2_2, %addenWfc2_2 : tensor<768x192xf32>
    %adstWfc2_2 = stablehlo.multiply %adlrWfc2_2, %adratWfc2_2 : tensor<768x192xf32>
    %adsubWfc2_2 = stablehlo.subtract %Wfc2_2, %adstWfc2_2 : tensor<768x192xf32>
    %adwdWfc2_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adwdlrWfc2_2 = stablehlo.multiply %adwdWfc2_2, %adlrWfc2_2 : tensor<768x192xf32>
    %adwdpWfc2_2 = stablehlo.multiply %adwdlrWfc2_2, %Wfc2_2 : tensor<768x192xf32>
    %adnewWfc2_2 = stablehlo.subtract %adsubWfc2_2, %adwdpWfc2_2 : tensor<768x192xf32>
    %adb1bfc2_2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bfc2_2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbfc2_2 = stablehlo.multiply %adb1bfc2_2, %bfc2_2m : tensor<192xf32>
    %admgbfc2_2 = stablehlo.multiply %adob1bfc2_2, %vitb2_pdbfc2 : tensor<192xf32>
    %admnbfc2_2 = stablehlo.add %admsbfc2_2, %admgbfc2_2 : tensor<192xf32>
    %adb2bfc2_2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bfc2_2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbfc2_2 = stablehlo.multiply %adb2bfc2_2, %bfc2_2v : tensor<192xf32>
    %adg2bfc2_2 = stablehlo.multiply %vitb2_pdbfc2, %vitb2_pdbfc2 : tensor<192xf32>
    %advgbfc2_2 = stablehlo.multiply %adob2bfc2_2, %adg2bfc2_2 : tensor<192xf32>
    %advnbfc2_2 = stablehlo.add %advsbfc2_2, %advgbfc2_2 : tensor<192xf32>
    %adbc1bfc2_2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bfc2_2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbfc2_2 = stablehlo.divide %admnbfc2_2, %adbc1bfc2_2 : tensor<192xf32>
    %advhbfc2_2 = stablehlo.divide %advnbfc2_2, %adbc2bfc2_2 : tensor<192xf32>
    %adlrbfc2_2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbfc2_2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbfc2_2 = stablehlo.sqrt %advhbfc2_2 : tensor<192xf32>
    %addenbfc2_2 = stablehlo.add %adsqbfc2_2, %adepsbfc2_2 : tensor<192xf32>
    %adratbfc2_2 = stablehlo.divide %admhbfc2_2, %addenbfc2_2 : tensor<192xf32>
    %adstbfc2_2 = stablehlo.multiply %adlrbfc2_2, %adratbfc2_2 : tensor<192xf32>
    %adsubbfc2_2 = stablehlo.subtract %bfc2_2, %adstbfc2_2 : tensor<192xf32>
    %adwdbfc2_2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbfc2_2 = stablehlo.multiply %adwdbfc2_2, %adlrbfc2_2 : tensor<192xf32>
    %adwdpbfc2_2 = stablehlo.multiply %adwdlrbfc2_2, %bfc2_2 : tensor<192xf32>
    %adnewbfc2_2 = stablehlo.subtract %adsubbfc2_2, %adwdpbfc2_2 : tensor<192xf32>
    %adb1g1_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g1_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg1_3 = stablehlo.multiply %adb1g1_3, %g1_3m : tensor<192xf32>
    %admgg1_3 = stablehlo.multiply %adob1g1_3, %vitb3_1dg : tensor<192xf32>
    %admng1_3 = stablehlo.add %admsg1_3, %admgg1_3 : tensor<192xf32>
    %adb2g1_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g1_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg1_3 = stablehlo.multiply %adb2g1_3, %g1_3v : tensor<192xf32>
    %adg2g1_3 = stablehlo.multiply %vitb3_1dg, %vitb3_1dg : tensor<192xf32>
    %advgg1_3 = stablehlo.multiply %adob2g1_3, %adg2g1_3 : tensor<192xf32>
    %advng1_3 = stablehlo.add %advsg1_3, %advgg1_3 : tensor<192xf32>
    %adbc1g1_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g1_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg1_3 = stablehlo.divide %admng1_3, %adbc1g1_3 : tensor<192xf32>
    %advhg1_3 = stablehlo.divide %advng1_3, %adbc2g1_3 : tensor<192xf32>
    %adlrg1_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg1_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg1_3 = stablehlo.sqrt %advhg1_3 : tensor<192xf32>
    %addeng1_3 = stablehlo.add %adsqg1_3, %adepsg1_3 : tensor<192xf32>
    %adratg1_3 = stablehlo.divide %admhg1_3, %addeng1_3 : tensor<192xf32>
    %adstg1_3 = stablehlo.multiply %adlrg1_3, %adratg1_3 : tensor<192xf32>
    %adsubg1_3 = stablehlo.subtract %g1_3, %adstg1_3 : tensor<192xf32>
    %adwdg1_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg1_3 = stablehlo.multiply %adwdg1_3, %adlrg1_3 : tensor<192xf32>
    %adwdpg1_3 = stablehlo.multiply %adwdlrg1_3, %g1_3 : tensor<192xf32>
    %adnewg1_3 = stablehlo.subtract %adsubg1_3, %adwdpg1_3 : tensor<192xf32>
    %adb1b1_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b1_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb1_3 = stablehlo.multiply %adb1b1_3, %b1_3m : tensor<192xf32>
    %admgb1_3 = stablehlo.multiply %adob1b1_3, %vitb3_1db : tensor<192xf32>
    %admnb1_3 = stablehlo.add %admsb1_3, %admgb1_3 : tensor<192xf32>
    %adb2b1_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b1_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb1_3 = stablehlo.multiply %adb2b1_3, %b1_3v : tensor<192xf32>
    %adg2b1_3 = stablehlo.multiply %vitb3_1db, %vitb3_1db : tensor<192xf32>
    %advgb1_3 = stablehlo.multiply %adob2b1_3, %adg2b1_3 : tensor<192xf32>
    %advnb1_3 = stablehlo.add %advsb1_3, %advgb1_3 : tensor<192xf32>
    %adbc1b1_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b1_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb1_3 = stablehlo.divide %admnb1_3, %adbc1b1_3 : tensor<192xf32>
    %advhb1_3 = stablehlo.divide %advnb1_3, %adbc2b1_3 : tensor<192xf32>
    %adlrb1_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb1_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb1_3 = stablehlo.sqrt %advhb1_3 : tensor<192xf32>
    %addenb1_3 = stablehlo.add %adsqb1_3, %adepsb1_3 : tensor<192xf32>
    %adratb1_3 = stablehlo.divide %admhb1_3, %addenb1_3 : tensor<192xf32>
    %adstb1_3 = stablehlo.multiply %adlrb1_3, %adratb1_3 : tensor<192xf32>
    %adsubb1_3 = stablehlo.subtract %b1_3, %adstb1_3 : tensor<192xf32>
    %adwdb1_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb1_3 = stablehlo.multiply %adwdb1_3, %adlrb1_3 : tensor<192xf32>
    %adwdpb1_3 = stablehlo.multiply %adwdlrb1_3, %b1_3 : tensor<192xf32>
    %adnewb1_3 = stablehlo.subtract %adsubb1_3, %adwdpb1_3 : tensor<192xf32>
    %adb1Wq_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wq_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWq_3 = stablehlo.multiply %adb1Wq_3, %Wq_3m : tensor<192x192xf32>
    %admgWq_3 = stablehlo.multiply %adob1Wq_3, %vitb3_mdWQ : tensor<192x192xf32>
    %admnWq_3 = stablehlo.add %admsWq_3, %admgWq_3 : tensor<192x192xf32>
    %adb2Wq_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wq_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWq_3 = stablehlo.multiply %adb2Wq_3, %Wq_3v : tensor<192x192xf32>
    %adg2Wq_3 = stablehlo.multiply %vitb3_mdWQ, %vitb3_mdWQ : tensor<192x192xf32>
    %advgWq_3 = stablehlo.multiply %adob2Wq_3, %adg2Wq_3 : tensor<192x192xf32>
    %advnWq_3 = stablehlo.add %advsWq_3, %advgWq_3 : tensor<192x192xf32>
    %adbc1Wq_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wq_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWq_3 = stablehlo.divide %admnWq_3, %adbc1Wq_3 : tensor<192x192xf32>
    %advhWq_3 = stablehlo.divide %advnWq_3, %adbc2Wq_3 : tensor<192x192xf32>
    %adlrWq_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWq_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWq_3 = stablehlo.sqrt %advhWq_3 : tensor<192x192xf32>
    %addenWq_3 = stablehlo.add %adsqWq_3, %adepsWq_3 : tensor<192x192xf32>
    %adratWq_3 = stablehlo.divide %admhWq_3, %addenWq_3 : tensor<192x192xf32>
    %adstWq_3 = stablehlo.multiply %adlrWq_3, %adratWq_3 : tensor<192x192xf32>
    %adsubWq_3 = stablehlo.subtract %Wq_3, %adstWq_3 : tensor<192x192xf32>
    %adwdWq_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWq_3 = stablehlo.multiply %adwdWq_3, %adlrWq_3 : tensor<192x192xf32>
    %adwdpWq_3 = stablehlo.multiply %adwdlrWq_3, %Wq_3 : tensor<192x192xf32>
    %adnewWq_3 = stablehlo.subtract %adsubWq_3, %adwdpWq_3 : tensor<192x192xf32>
    %adb1bq_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bq_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbq_3 = stablehlo.multiply %adb1bq_3, %bq_3m : tensor<192xf32>
    %admgbq_3 = stablehlo.multiply %adob1bq_3, %vitb3_mdbQ : tensor<192xf32>
    %admnbq_3 = stablehlo.add %admsbq_3, %admgbq_3 : tensor<192xf32>
    %adb2bq_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bq_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbq_3 = stablehlo.multiply %adb2bq_3, %bq_3v : tensor<192xf32>
    %adg2bq_3 = stablehlo.multiply %vitb3_mdbQ, %vitb3_mdbQ : tensor<192xf32>
    %advgbq_3 = stablehlo.multiply %adob2bq_3, %adg2bq_3 : tensor<192xf32>
    %advnbq_3 = stablehlo.add %advsbq_3, %advgbq_3 : tensor<192xf32>
    %adbc1bq_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bq_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbq_3 = stablehlo.divide %admnbq_3, %adbc1bq_3 : tensor<192xf32>
    %advhbq_3 = stablehlo.divide %advnbq_3, %adbc2bq_3 : tensor<192xf32>
    %adlrbq_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbq_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbq_3 = stablehlo.sqrt %advhbq_3 : tensor<192xf32>
    %addenbq_3 = stablehlo.add %adsqbq_3, %adepsbq_3 : tensor<192xf32>
    %adratbq_3 = stablehlo.divide %admhbq_3, %addenbq_3 : tensor<192xf32>
    %adstbq_3 = stablehlo.multiply %adlrbq_3, %adratbq_3 : tensor<192xf32>
    %adsubbq_3 = stablehlo.subtract %bq_3, %adstbq_3 : tensor<192xf32>
    %adwdbq_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbq_3 = stablehlo.multiply %adwdbq_3, %adlrbq_3 : tensor<192xf32>
    %adwdpbq_3 = stablehlo.multiply %adwdlrbq_3, %bq_3 : tensor<192xf32>
    %adnewbq_3 = stablehlo.subtract %adsubbq_3, %adwdpbq_3 : tensor<192xf32>
    %adb1Wk_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wk_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWk_3 = stablehlo.multiply %adb1Wk_3, %Wk_3m : tensor<192x192xf32>
    %admgWk_3 = stablehlo.multiply %adob1Wk_3, %vitb3_mdWK : tensor<192x192xf32>
    %admnWk_3 = stablehlo.add %admsWk_3, %admgWk_3 : tensor<192x192xf32>
    %adb2Wk_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wk_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWk_3 = stablehlo.multiply %adb2Wk_3, %Wk_3v : tensor<192x192xf32>
    %adg2Wk_3 = stablehlo.multiply %vitb3_mdWK, %vitb3_mdWK : tensor<192x192xf32>
    %advgWk_3 = stablehlo.multiply %adob2Wk_3, %adg2Wk_3 : tensor<192x192xf32>
    %advnWk_3 = stablehlo.add %advsWk_3, %advgWk_3 : tensor<192x192xf32>
    %adbc1Wk_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wk_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWk_3 = stablehlo.divide %admnWk_3, %adbc1Wk_3 : tensor<192x192xf32>
    %advhWk_3 = stablehlo.divide %advnWk_3, %adbc2Wk_3 : tensor<192x192xf32>
    %adlrWk_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWk_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWk_3 = stablehlo.sqrt %advhWk_3 : tensor<192x192xf32>
    %addenWk_3 = stablehlo.add %adsqWk_3, %adepsWk_3 : tensor<192x192xf32>
    %adratWk_3 = stablehlo.divide %admhWk_3, %addenWk_3 : tensor<192x192xf32>
    %adstWk_3 = stablehlo.multiply %adlrWk_3, %adratWk_3 : tensor<192x192xf32>
    %adsubWk_3 = stablehlo.subtract %Wk_3, %adstWk_3 : tensor<192x192xf32>
    %adwdWk_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWk_3 = stablehlo.multiply %adwdWk_3, %adlrWk_3 : tensor<192x192xf32>
    %adwdpWk_3 = stablehlo.multiply %adwdlrWk_3, %Wk_3 : tensor<192x192xf32>
    %adnewWk_3 = stablehlo.subtract %adsubWk_3, %adwdpWk_3 : tensor<192x192xf32>
    %adb1bk_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bk_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbk_3 = stablehlo.multiply %adb1bk_3, %bk_3m : tensor<192xf32>
    %admgbk_3 = stablehlo.multiply %adob1bk_3, %vitb3_mdbK : tensor<192xf32>
    %admnbk_3 = stablehlo.add %admsbk_3, %admgbk_3 : tensor<192xf32>
    %adb2bk_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bk_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbk_3 = stablehlo.multiply %adb2bk_3, %bk_3v : tensor<192xf32>
    %adg2bk_3 = stablehlo.multiply %vitb3_mdbK, %vitb3_mdbK : tensor<192xf32>
    %advgbk_3 = stablehlo.multiply %adob2bk_3, %adg2bk_3 : tensor<192xf32>
    %advnbk_3 = stablehlo.add %advsbk_3, %advgbk_3 : tensor<192xf32>
    %adbc1bk_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bk_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbk_3 = stablehlo.divide %admnbk_3, %adbc1bk_3 : tensor<192xf32>
    %advhbk_3 = stablehlo.divide %advnbk_3, %adbc2bk_3 : tensor<192xf32>
    %adlrbk_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbk_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbk_3 = stablehlo.sqrt %advhbk_3 : tensor<192xf32>
    %addenbk_3 = stablehlo.add %adsqbk_3, %adepsbk_3 : tensor<192xf32>
    %adratbk_3 = stablehlo.divide %admhbk_3, %addenbk_3 : tensor<192xf32>
    %adstbk_3 = stablehlo.multiply %adlrbk_3, %adratbk_3 : tensor<192xf32>
    %adsubbk_3 = stablehlo.subtract %bk_3, %adstbk_3 : tensor<192xf32>
    %adwdbk_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbk_3 = stablehlo.multiply %adwdbk_3, %adlrbk_3 : tensor<192xf32>
    %adwdpbk_3 = stablehlo.multiply %adwdlrbk_3, %bk_3 : tensor<192xf32>
    %adnewbk_3 = stablehlo.subtract %adsubbk_3, %adwdpbk_3 : tensor<192xf32>
    %adb1Wv_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wv_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWv_3 = stablehlo.multiply %adb1Wv_3, %Wv_3m : tensor<192x192xf32>
    %admgWv_3 = stablehlo.multiply %adob1Wv_3, %vitb3_mdWV : tensor<192x192xf32>
    %admnWv_3 = stablehlo.add %admsWv_3, %admgWv_3 : tensor<192x192xf32>
    %adb2Wv_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wv_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWv_3 = stablehlo.multiply %adb2Wv_3, %Wv_3v : tensor<192x192xf32>
    %adg2Wv_3 = stablehlo.multiply %vitb3_mdWV, %vitb3_mdWV : tensor<192x192xf32>
    %advgWv_3 = stablehlo.multiply %adob2Wv_3, %adg2Wv_3 : tensor<192x192xf32>
    %advnWv_3 = stablehlo.add %advsWv_3, %advgWv_3 : tensor<192x192xf32>
    %adbc1Wv_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wv_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWv_3 = stablehlo.divide %admnWv_3, %adbc1Wv_3 : tensor<192x192xf32>
    %advhWv_3 = stablehlo.divide %advnWv_3, %adbc2Wv_3 : tensor<192x192xf32>
    %adlrWv_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWv_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWv_3 = stablehlo.sqrt %advhWv_3 : tensor<192x192xf32>
    %addenWv_3 = stablehlo.add %adsqWv_3, %adepsWv_3 : tensor<192x192xf32>
    %adratWv_3 = stablehlo.divide %admhWv_3, %addenWv_3 : tensor<192x192xf32>
    %adstWv_3 = stablehlo.multiply %adlrWv_3, %adratWv_3 : tensor<192x192xf32>
    %adsubWv_3 = stablehlo.subtract %Wv_3, %adstWv_3 : tensor<192x192xf32>
    %adwdWv_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWv_3 = stablehlo.multiply %adwdWv_3, %adlrWv_3 : tensor<192x192xf32>
    %adwdpWv_3 = stablehlo.multiply %adwdlrWv_3, %Wv_3 : tensor<192x192xf32>
    %adnewWv_3 = stablehlo.subtract %adsubWv_3, %adwdpWv_3 : tensor<192x192xf32>
    %adb1bv_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bv_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbv_3 = stablehlo.multiply %adb1bv_3, %bv_3m : tensor<192xf32>
    %admgbv_3 = stablehlo.multiply %adob1bv_3, %vitb3_mdbV : tensor<192xf32>
    %admnbv_3 = stablehlo.add %admsbv_3, %admgbv_3 : tensor<192xf32>
    %adb2bv_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bv_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbv_3 = stablehlo.multiply %adb2bv_3, %bv_3v : tensor<192xf32>
    %adg2bv_3 = stablehlo.multiply %vitb3_mdbV, %vitb3_mdbV : tensor<192xf32>
    %advgbv_3 = stablehlo.multiply %adob2bv_3, %adg2bv_3 : tensor<192xf32>
    %advnbv_3 = stablehlo.add %advsbv_3, %advgbv_3 : tensor<192xf32>
    %adbc1bv_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bv_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbv_3 = stablehlo.divide %admnbv_3, %adbc1bv_3 : tensor<192xf32>
    %advhbv_3 = stablehlo.divide %advnbv_3, %adbc2bv_3 : tensor<192xf32>
    %adlrbv_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbv_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbv_3 = stablehlo.sqrt %advhbv_3 : tensor<192xf32>
    %addenbv_3 = stablehlo.add %adsqbv_3, %adepsbv_3 : tensor<192xf32>
    %adratbv_3 = stablehlo.divide %admhbv_3, %addenbv_3 : tensor<192xf32>
    %adstbv_3 = stablehlo.multiply %adlrbv_3, %adratbv_3 : tensor<192xf32>
    %adsubbv_3 = stablehlo.subtract %bv_3, %adstbv_3 : tensor<192xf32>
    %adwdbv_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbv_3 = stablehlo.multiply %adwdbv_3, %adlrbv_3 : tensor<192xf32>
    %adwdpbv_3 = stablehlo.multiply %adwdlrbv_3, %bv_3 : tensor<192xf32>
    %adnewbv_3 = stablehlo.subtract %adsubbv_3, %adwdpbv_3 : tensor<192xf32>
    %adb1Wo_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wo_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWo_3 = stablehlo.multiply %adb1Wo_3, %Wo_3m : tensor<192x192xf32>
    %admgWo_3 = stablehlo.multiply %adob1Wo_3, %vitb3_mdWo : tensor<192x192xf32>
    %admnWo_3 = stablehlo.add %admsWo_3, %admgWo_3 : tensor<192x192xf32>
    %adb2Wo_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wo_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWo_3 = stablehlo.multiply %adb2Wo_3, %Wo_3v : tensor<192x192xf32>
    %adg2Wo_3 = stablehlo.multiply %vitb3_mdWo, %vitb3_mdWo : tensor<192x192xf32>
    %advgWo_3 = stablehlo.multiply %adob2Wo_3, %adg2Wo_3 : tensor<192x192xf32>
    %advnWo_3 = stablehlo.add %advsWo_3, %advgWo_3 : tensor<192x192xf32>
    %adbc1Wo_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wo_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWo_3 = stablehlo.divide %admnWo_3, %adbc1Wo_3 : tensor<192x192xf32>
    %advhWo_3 = stablehlo.divide %advnWo_3, %adbc2Wo_3 : tensor<192x192xf32>
    %adlrWo_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWo_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWo_3 = stablehlo.sqrt %advhWo_3 : tensor<192x192xf32>
    %addenWo_3 = stablehlo.add %adsqWo_3, %adepsWo_3 : tensor<192x192xf32>
    %adratWo_3 = stablehlo.divide %admhWo_3, %addenWo_3 : tensor<192x192xf32>
    %adstWo_3 = stablehlo.multiply %adlrWo_3, %adratWo_3 : tensor<192x192xf32>
    %adsubWo_3 = stablehlo.subtract %Wo_3, %adstWo_3 : tensor<192x192xf32>
    %adwdWo_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWo_3 = stablehlo.multiply %adwdWo_3, %adlrWo_3 : tensor<192x192xf32>
    %adwdpWo_3 = stablehlo.multiply %adwdlrWo_3, %Wo_3 : tensor<192x192xf32>
    %adnewWo_3 = stablehlo.subtract %adsubWo_3, %adwdpWo_3 : tensor<192x192xf32>
    %adb1bo_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bo_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbo_3 = stablehlo.multiply %adb1bo_3, %bo_3m : tensor<192xf32>
    %admgbo_3 = stablehlo.multiply %adob1bo_3, %vitb3_mdbo : tensor<192xf32>
    %admnbo_3 = stablehlo.add %admsbo_3, %admgbo_3 : tensor<192xf32>
    %adb2bo_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bo_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbo_3 = stablehlo.multiply %adb2bo_3, %bo_3v : tensor<192xf32>
    %adg2bo_3 = stablehlo.multiply %vitb3_mdbo, %vitb3_mdbo : tensor<192xf32>
    %advgbo_3 = stablehlo.multiply %adob2bo_3, %adg2bo_3 : tensor<192xf32>
    %advnbo_3 = stablehlo.add %advsbo_3, %advgbo_3 : tensor<192xf32>
    %adbc1bo_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bo_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbo_3 = stablehlo.divide %admnbo_3, %adbc1bo_3 : tensor<192xf32>
    %advhbo_3 = stablehlo.divide %advnbo_3, %adbc2bo_3 : tensor<192xf32>
    %adlrbo_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbo_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbo_3 = stablehlo.sqrt %advhbo_3 : tensor<192xf32>
    %addenbo_3 = stablehlo.add %adsqbo_3, %adepsbo_3 : tensor<192xf32>
    %adratbo_3 = stablehlo.divide %admhbo_3, %addenbo_3 : tensor<192xf32>
    %adstbo_3 = stablehlo.multiply %adlrbo_3, %adratbo_3 : tensor<192xf32>
    %adsubbo_3 = stablehlo.subtract %bo_3, %adstbo_3 : tensor<192xf32>
    %adwdbo_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbo_3 = stablehlo.multiply %adwdbo_3, %adlrbo_3 : tensor<192xf32>
    %adwdpbo_3 = stablehlo.multiply %adwdlrbo_3, %bo_3 : tensor<192xf32>
    %adnewbo_3 = stablehlo.subtract %adsubbo_3, %adwdpbo_3 : tensor<192xf32>
    %adb1g2_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g2_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg2_3 = stablehlo.multiply %adb1g2_3, %g2_3m : tensor<192xf32>
    %admgg2_3 = stablehlo.multiply %adob1g2_3, %vitb3_2dg : tensor<192xf32>
    %admng2_3 = stablehlo.add %admsg2_3, %admgg2_3 : tensor<192xf32>
    %adb2g2_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g2_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg2_3 = stablehlo.multiply %adb2g2_3, %g2_3v : tensor<192xf32>
    %adg2g2_3 = stablehlo.multiply %vitb3_2dg, %vitb3_2dg : tensor<192xf32>
    %advgg2_3 = stablehlo.multiply %adob2g2_3, %adg2g2_3 : tensor<192xf32>
    %advng2_3 = stablehlo.add %advsg2_3, %advgg2_3 : tensor<192xf32>
    %adbc1g2_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g2_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg2_3 = stablehlo.divide %admng2_3, %adbc1g2_3 : tensor<192xf32>
    %advhg2_3 = stablehlo.divide %advng2_3, %adbc2g2_3 : tensor<192xf32>
    %adlrg2_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg2_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg2_3 = stablehlo.sqrt %advhg2_3 : tensor<192xf32>
    %addeng2_3 = stablehlo.add %adsqg2_3, %adepsg2_3 : tensor<192xf32>
    %adratg2_3 = stablehlo.divide %admhg2_3, %addeng2_3 : tensor<192xf32>
    %adstg2_3 = stablehlo.multiply %adlrg2_3, %adratg2_3 : tensor<192xf32>
    %adsubg2_3 = stablehlo.subtract %g2_3, %adstg2_3 : tensor<192xf32>
    %adwdg2_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg2_3 = stablehlo.multiply %adwdg2_3, %adlrg2_3 : tensor<192xf32>
    %adwdpg2_3 = stablehlo.multiply %adwdlrg2_3, %g2_3 : tensor<192xf32>
    %adnewg2_3 = stablehlo.subtract %adsubg2_3, %adwdpg2_3 : tensor<192xf32>
    %adb1b2_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b2_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb2_3 = stablehlo.multiply %adb1b2_3, %b2_3m : tensor<192xf32>
    %admgb2_3 = stablehlo.multiply %adob1b2_3, %vitb3_2db : tensor<192xf32>
    %admnb2_3 = stablehlo.add %admsb2_3, %admgb2_3 : tensor<192xf32>
    %adb2b2_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b2_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb2_3 = stablehlo.multiply %adb2b2_3, %b2_3v : tensor<192xf32>
    %adg2b2_3 = stablehlo.multiply %vitb3_2db, %vitb3_2db : tensor<192xf32>
    %advgb2_3 = stablehlo.multiply %adob2b2_3, %adg2b2_3 : tensor<192xf32>
    %advnb2_3 = stablehlo.add %advsb2_3, %advgb2_3 : tensor<192xf32>
    %adbc1b2_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b2_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb2_3 = stablehlo.divide %admnb2_3, %adbc1b2_3 : tensor<192xf32>
    %advhb2_3 = stablehlo.divide %advnb2_3, %adbc2b2_3 : tensor<192xf32>
    %adlrb2_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb2_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb2_3 = stablehlo.sqrt %advhb2_3 : tensor<192xf32>
    %addenb2_3 = stablehlo.add %adsqb2_3, %adepsb2_3 : tensor<192xf32>
    %adratb2_3 = stablehlo.divide %admhb2_3, %addenb2_3 : tensor<192xf32>
    %adstb2_3 = stablehlo.multiply %adlrb2_3, %adratb2_3 : tensor<192xf32>
    %adsubb2_3 = stablehlo.subtract %b2_3, %adstb2_3 : tensor<192xf32>
    %adwdb2_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb2_3 = stablehlo.multiply %adwdb2_3, %adlrb2_3 : tensor<192xf32>
    %adwdpb2_3 = stablehlo.multiply %adwdlrb2_3, %b2_3 : tensor<192xf32>
    %adnewb2_3 = stablehlo.subtract %adsubb2_3, %adwdpb2_3 : tensor<192xf32>
    %adb1Wfc1_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob1Wfc1_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admsWfc1_3 = stablehlo.multiply %adb1Wfc1_3, %Wfc1_3m : tensor<192x768xf32>
    %admgWfc1_3 = stablehlo.multiply %adob1Wfc1_3, %vitb3_pdWfc1 : tensor<192x768xf32>
    %admnWfc1_3 = stablehlo.add %admsWfc1_3, %admgWfc1_3 : tensor<192x768xf32>
    %adb2Wfc1_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob2Wfc1_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %advsWfc1_3 = stablehlo.multiply %adb2Wfc1_3, %Wfc1_3v : tensor<192x768xf32>
    %adg2Wfc1_3 = stablehlo.multiply %vitb3_pdWfc1, %vitb3_pdWfc1 : tensor<192x768xf32>
    %advgWfc1_3 = stablehlo.multiply %adob2Wfc1_3, %adg2Wfc1_3 : tensor<192x768xf32>
    %advnWfc1_3 = stablehlo.add %advsWfc1_3, %advgWfc1_3 : tensor<192x768xf32>
    %adbc1Wfc1_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adbc2Wfc1_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admhWfc1_3 = stablehlo.divide %admnWfc1_3, %adbc1Wfc1_3 : tensor<192x768xf32>
    %advhWfc1_3 = stablehlo.divide %advnWfc1_3, %adbc2Wfc1_3 : tensor<192x768xf32>
    %adlrWfc1_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adepsWfc1_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adsqWfc1_3 = stablehlo.sqrt %advhWfc1_3 : tensor<192x768xf32>
    %addenWfc1_3 = stablehlo.add %adsqWfc1_3, %adepsWfc1_3 : tensor<192x768xf32>
    %adratWfc1_3 = stablehlo.divide %admhWfc1_3, %addenWfc1_3 : tensor<192x768xf32>
    %adstWfc1_3 = stablehlo.multiply %adlrWfc1_3, %adratWfc1_3 : tensor<192x768xf32>
    %adsubWfc1_3 = stablehlo.subtract %Wfc1_3, %adstWfc1_3 : tensor<192x768xf32>
    %adwdWfc1_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adwdlrWfc1_3 = stablehlo.multiply %adwdWfc1_3, %adlrWfc1_3 : tensor<192x768xf32>
    %adwdpWfc1_3 = stablehlo.multiply %adwdlrWfc1_3, %Wfc1_3 : tensor<192x768xf32>
    %adnewWfc1_3 = stablehlo.subtract %adsubWfc1_3, %adwdpWfc1_3 : tensor<192x768xf32>
    %adb1bfc1_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1bfc1_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admsbfc1_3 = stablehlo.multiply %adb1bfc1_3, %bfc1_3m : tensor<768xf32>
    %admgbfc1_3 = stablehlo.multiply %adob1bfc1_3, %vitb3_pdbfc1 : tensor<768xf32>
    %admnbfc1_3 = stablehlo.add %admsbfc1_3, %admgbfc1_3 : tensor<768xf32>
    %adb2bfc1_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2bfc1_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advsbfc1_3 = stablehlo.multiply %adb2bfc1_3, %bfc1_3v : tensor<768xf32>
    %adg2bfc1_3 = stablehlo.multiply %vitb3_pdbfc1, %vitb3_pdbfc1 : tensor<768xf32>
    %advgbfc1_3 = stablehlo.multiply %adob2bfc1_3, %adg2bfc1_3 : tensor<768xf32>
    %advnbfc1_3 = stablehlo.add %advsbfc1_3, %advgbfc1_3 : tensor<768xf32>
    %adbc1bfc1_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2bfc1_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhbfc1_3 = stablehlo.divide %admnbfc1_3, %adbc1bfc1_3 : tensor<768xf32>
    %advhbfc1_3 = stablehlo.divide %advnbfc1_3, %adbc2bfc1_3 : tensor<768xf32>
    %adlrbfc1_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepsbfc1_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqbfc1_3 = stablehlo.sqrt %advhbfc1_3 : tensor<768xf32>
    %addenbfc1_3 = stablehlo.add %adsqbfc1_3, %adepsbfc1_3 : tensor<768xf32>
    %adratbfc1_3 = stablehlo.divide %admhbfc1_3, %addenbfc1_3 : tensor<768xf32>
    %adstbfc1_3 = stablehlo.multiply %adlrbfc1_3, %adratbfc1_3 : tensor<768xf32>
    %adsubbfc1_3 = stablehlo.subtract %bfc1_3, %adstbfc1_3 : tensor<768xf32>
    %adwdbfc1_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrbfc1_3 = stablehlo.multiply %adwdbfc1_3, %adlrbfc1_3 : tensor<768xf32>
    %adwdpbfc1_3 = stablehlo.multiply %adwdlrbfc1_3, %bfc1_3 : tensor<768xf32>
    %adnewbfc1_3 = stablehlo.subtract %adsubbfc1_3, %adwdpbfc1_3 : tensor<768xf32>
    %adb1Wfc2_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob1Wfc2_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admsWfc2_3 = stablehlo.multiply %adb1Wfc2_3, %Wfc2_3m : tensor<768x192xf32>
    %admgWfc2_3 = stablehlo.multiply %adob1Wfc2_3, %vitb3_pdWfc2 : tensor<768x192xf32>
    %admnWfc2_3 = stablehlo.add %admsWfc2_3, %admgWfc2_3 : tensor<768x192xf32>
    %adb2Wfc2_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob2Wfc2_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %advsWfc2_3 = stablehlo.multiply %adb2Wfc2_3, %Wfc2_3v : tensor<768x192xf32>
    %adg2Wfc2_3 = stablehlo.multiply %vitb3_pdWfc2, %vitb3_pdWfc2 : tensor<768x192xf32>
    %advgWfc2_3 = stablehlo.multiply %adob2Wfc2_3, %adg2Wfc2_3 : tensor<768x192xf32>
    %advnWfc2_3 = stablehlo.add %advsWfc2_3, %advgWfc2_3 : tensor<768x192xf32>
    %adbc1Wfc2_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adbc2Wfc2_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admhWfc2_3 = stablehlo.divide %admnWfc2_3, %adbc1Wfc2_3 : tensor<768x192xf32>
    %advhWfc2_3 = stablehlo.divide %advnWfc2_3, %adbc2Wfc2_3 : tensor<768x192xf32>
    %adlrWfc2_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adepsWfc2_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adsqWfc2_3 = stablehlo.sqrt %advhWfc2_3 : tensor<768x192xf32>
    %addenWfc2_3 = stablehlo.add %adsqWfc2_3, %adepsWfc2_3 : tensor<768x192xf32>
    %adratWfc2_3 = stablehlo.divide %admhWfc2_3, %addenWfc2_3 : tensor<768x192xf32>
    %adstWfc2_3 = stablehlo.multiply %adlrWfc2_3, %adratWfc2_3 : tensor<768x192xf32>
    %adsubWfc2_3 = stablehlo.subtract %Wfc2_3, %adstWfc2_3 : tensor<768x192xf32>
    %adwdWfc2_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adwdlrWfc2_3 = stablehlo.multiply %adwdWfc2_3, %adlrWfc2_3 : tensor<768x192xf32>
    %adwdpWfc2_3 = stablehlo.multiply %adwdlrWfc2_3, %Wfc2_3 : tensor<768x192xf32>
    %adnewWfc2_3 = stablehlo.subtract %adsubWfc2_3, %adwdpWfc2_3 : tensor<768x192xf32>
    %adb1bfc2_3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bfc2_3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbfc2_3 = stablehlo.multiply %adb1bfc2_3, %bfc2_3m : tensor<192xf32>
    %admgbfc2_3 = stablehlo.multiply %adob1bfc2_3, %vitb3_pdbfc2 : tensor<192xf32>
    %admnbfc2_3 = stablehlo.add %admsbfc2_3, %admgbfc2_3 : tensor<192xf32>
    %adb2bfc2_3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bfc2_3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbfc2_3 = stablehlo.multiply %adb2bfc2_3, %bfc2_3v : tensor<192xf32>
    %adg2bfc2_3 = stablehlo.multiply %vitb3_pdbfc2, %vitb3_pdbfc2 : tensor<192xf32>
    %advgbfc2_3 = stablehlo.multiply %adob2bfc2_3, %adg2bfc2_3 : tensor<192xf32>
    %advnbfc2_3 = stablehlo.add %advsbfc2_3, %advgbfc2_3 : tensor<192xf32>
    %adbc1bfc2_3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bfc2_3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbfc2_3 = stablehlo.divide %admnbfc2_3, %adbc1bfc2_3 : tensor<192xf32>
    %advhbfc2_3 = stablehlo.divide %advnbfc2_3, %adbc2bfc2_3 : tensor<192xf32>
    %adlrbfc2_3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbfc2_3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbfc2_3 = stablehlo.sqrt %advhbfc2_3 : tensor<192xf32>
    %addenbfc2_3 = stablehlo.add %adsqbfc2_3, %adepsbfc2_3 : tensor<192xf32>
    %adratbfc2_3 = stablehlo.divide %admhbfc2_3, %addenbfc2_3 : tensor<192xf32>
    %adstbfc2_3 = stablehlo.multiply %adlrbfc2_3, %adratbfc2_3 : tensor<192xf32>
    %adsubbfc2_3 = stablehlo.subtract %bfc2_3, %adstbfc2_3 : tensor<192xf32>
    %adwdbfc2_3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbfc2_3 = stablehlo.multiply %adwdbfc2_3, %adlrbfc2_3 : tensor<192xf32>
    %adwdpbfc2_3 = stablehlo.multiply %adwdlrbfc2_3, %bfc2_3 : tensor<192xf32>
    %adnewbfc2_3 = stablehlo.subtract %adsubbfc2_3, %adwdpbfc2_3 : tensor<192xf32>
    %adb1g1_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g1_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg1_4 = stablehlo.multiply %adb1g1_4, %g1_4m : tensor<192xf32>
    %admgg1_4 = stablehlo.multiply %adob1g1_4, %vitb4_1dg : tensor<192xf32>
    %admng1_4 = stablehlo.add %admsg1_4, %admgg1_4 : tensor<192xf32>
    %adb2g1_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g1_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg1_4 = stablehlo.multiply %adb2g1_4, %g1_4v : tensor<192xf32>
    %adg2g1_4 = stablehlo.multiply %vitb4_1dg, %vitb4_1dg : tensor<192xf32>
    %advgg1_4 = stablehlo.multiply %adob2g1_4, %adg2g1_4 : tensor<192xf32>
    %advng1_4 = stablehlo.add %advsg1_4, %advgg1_4 : tensor<192xf32>
    %adbc1g1_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g1_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg1_4 = stablehlo.divide %admng1_4, %adbc1g1_4 : tensor<192xf32>
    %advhg1_4 = stablehlo.divide %advng1_4, %adbc2g1_4 : tensor<192xf32>
    %adlrg1_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg1_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg1_4 = stablehlo.sqrt %advhg1_4 : tensor<192xf32>
    %addeng1_4 = stablehlo.add %adsqg1_4, %adepsg1_4 : tensor<192xf32>
    %adratg1_4 = stablehlo.divide %admhg1_4, %addeng1_4 : tensor<192xf32>
    %adstg1_4 = stablehlo.multiply %adlrg1_4, %adratg1_4 : tensor<192xf32>
    %adsubg1_4 = stablehlo.subtract %g1_4, %adstg1_4 : tensor<192xf32>
    %adwdg1_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg1_4 = stablehlo.multiply %adwdg1_4, %adlrg1_4 : tensor<192xf32>
    %adwdpg1_4 = stablehlo.multiply %adwdlrg1_4, %g1_4 : tensor<192xf32>
    %adnewg1_4 = stablehlo.subtract %adsubg1_4, %adwdpg1_4 : tensor<192xf32>
    %adb1b1_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b1_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb1_4 = stablehlo.multiply %adb1b1_4, %b1_4m : tensor<192xf32>
    %admgb1_4 = stablehlo.multiply %adob1b1_4, %vitb4_1db : tensor<192xf32>
    %admnb1_4 = stablehlo.add %admsb1_4, %admgb1_4 : tensor<192xf32>
    %adb2b1_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b1_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb1_4 = stablehlo.multiply %adb2b1_4, %b1_4v : tensor<192xf32>
    %adg2b1_4 = stablehlo.multiply %vitb4_1db, %vitb4_1db : tensor<192xf32>
    %advgb1_4 = stablehlo.multiply %adob2b1_4, %adg2b1_4 : tensor<192xf32>
    %advnb1_4 = stablehlo.add %advsb1_4, %advgb1_4 : tensor<192xf32>
    %adbc1b1_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b1_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb1_4 = stablehlo.divide %admnb1_4, %adbc1b1_4 : tensor<192xf32>
    %advhb1_4 = stablehlo.divide %advnb1_4, %adbc2b1_4 : tensor<192xf32>
    %adlrb1_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb1_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb1_4 = stablehlo.sqrt %advhb1_4 : tensor<192xf32>
    %addenb1_4 = stablehlo.add %adsqb1_4, %adepsb1_4 : tensor<192xf32>
    %adratb1_4 = stablehlo.divide %admhb1_4, %addenb1_4 : tensor<192xf32>
    %adstb1_4 = stablehlo.multiply %adlrb1_4, %adratb1_4 : tensor<192xf32>
    %adsubb1_4 = stablehlo.subtract %b1_4, %adstb1_4 : tensor<192xf32>
    %adwdb1_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb1_4 = stablehlo.multiply %adwdb1_4, %adlrb1_4 : tensor<192xf32>
    %adwdpb1_4 = stablehlo.multiply %adwdlrb1_4, %b1_4 : tensor<192xf32>
    %adnewb1_4 = stablehlo.subtract %adsubb1_4, %adwdpb1_4 : tensor<192xf32>
    %adb1Wq_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wq_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWq_4 = stablehlo.multiply %adb1Wq_4, %Wq_4m : tensor<192x192xf32>
    %admgWq_4 = stablehlo.multiply %adob1Wq_4, %vitb4_mdWQ : tensor<192x192xf32>
    %admnWq_4 = stablehlo.add %admsWq_4, %admgWq_4 : tensor<192x192xf32>
    %adb2Wq_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wq_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWq_4 = stablehlo.multiply %adb2Wq_4, %Wq_4v : tensor<192x192xf32>
    %adg2Wq_4 = stablehlo.multiply %vitb4_mdWQ, %vitb4_mdWQ : tensor<192x192xf32>
    %advgWq_4 = stablehlo.multiply %adob2Wq_4, %adg2Wq_4 : tensor<192x192xf32>
    %advnWq_4 = stablehlo.add %advsWq_4, %advgWq_4 : tensor<192x192xf32>
    %adbc1Wq_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wq_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWq_4 = stablehlo.divide %admnWq_4, %adbc1Wq_4 : tensor<192x192xf32>
    %advhWq_4 = stablehlo.divide %advnWq_4, %adbc2Wq_4 : tensor<192x192xf32>
    %adlrWq_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWq_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWq_4 = stablehlo.sqrt %advhWq_4 : tensor<192x192xf32>
    %addenWq_4 = stablehlo.add %adsqWq_4, %adepsWq_4 : tensor<192x192xf32>
    %adratWq_4 = stablehlo.divide %admhWq_4, %addenWq_4 : tensor<192x192xf32>
    %adstWq_4 = stablehlo.multiply %adlrWq_4, %adratWq_4 : tensor<192x192xf32>
    %adsubWq_4 = stablehlo.subtract %Wq_4, %adstWq_4 : tensor<192x192xf32>
    %adwdWq_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWq_4 = stablehlo.multiply %adwdWq_4, %adlrWq_4 : tensor<192x192xf32>
    %adwdpWq_4 = stablehlo.multiply %adwdlrWq_4, %Wq_4 : tensor<192x192xf32>
    %adnewWq_4 = stablehlo.subtract %adsubWq_4, %adwdpWq_4 : tensor<192x192xf32>
    %adb1bq_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bq_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbq_4 = stablehlo.multiply %adb1bq_4, %bq_4m : tensor<192xf32>
    %admgbq_4 = stablehlo.multiply %adob1bq_4, %vitb4_mdbQ : tensor<192xf32>
    %admnbq_4 = stablehlo.add %admsbq_4, %admgbq_4 : tensor<192xf32>
    %adb2bq_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bq_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbq_4 = stablehlo.multiply %adb2bq_4, %bq_4v : tensor<192xf32>
    %adg2bq_4 = stablehlo.multiply %vitb4_mdbQ, %vitb4_mdbQ : tensor<192xf32>
    %advgbq_4 = stablehlo.multiply %adob2bq_4, %adg2bq_4 : tensor<192xf32>
    %advnbq_4 = stablehlo.add %advsbq_4, %advgbq_4 : tensor<192xf32>
    %adbc1bq_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bq_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbq_4 = stablehlo.divide %admnbq_4, %adbc1bq_4 : tensor<192xf32>
    %advhbq_4 = stablehlo.divide %advnbq_4, %adbc2bq_4 : tensor<192xf32>
    %adlrbq_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbq_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbq_4 = stablehlo.sqrt %advhbq_4 : tensor<192xf32>
    %addenbq_4 = stablehlo.add %adsqbq_4, %adepsbq_4 : tensor<192xf32>
    %adratbq_4 = stablehlo.divide %admhbq_4, %addenbq_4 : tensor<192xf32>
    %adstbq_4 = stablehlo.multiply %adlrbq_4, %adratbq_4 : tensor<192xf32>
    %adsubbq_4 = stablehlo.subtract %bq_4, %adstbq_4 : tensor<192xf32>
    %adwdbq_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbq_4 = stablehlo.multiply %adwdbq_4, %adlrbq_4 : tensor<192xf32>
    %adwdpbq_4 = stablehlo.multiply %adwdlrbq_4, %bq_4 : tensor<192xf32>
    %adnewbq_4 = stablehlo.subtract %adsubbq_4, %adwdpbq_4 : tensor<192xf32>
    %adb1Wk_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wk_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWk_4 = stablehlo.multiply %adb1Wk_4, %Wk_4m : tensor<192x192xf32>
    %admgWk_4 = stablehlo.multiply %adob1Wk_4, %vitb4_mdWK : tensor<192x192xf32>
    %admnWk_4 = stablehlo.add %admsWk_4, %admgWk_4 : tensor<192x192xf32>
    %adb2Wk_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wk_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWk_4 = stablehlo.multiply %adb2Wk_4, %Wk_4v : tensor<192x192xf32>
    %adg2Wk_4 = stablehlo.multiply %vitb4_mdWK, %vitb4_mdWK : tensor<192x192xf32>
    %advgWk_4 = stablehlo.multiply %adob2Wk_4, %adg2Wk_4 : tensor<192x192xf32>
    %advnWk_4 = stablehlo.add %advsWk_4, %advgWk_4 : tensor<192x192xf32>
    %adbc1Wk_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wk_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWk_4 = stablehlo.divide %admnWk_4, %adbc1Wk_4 : tensor<192x192xf32>
    %advhWk_4 = stablehlo.divide %advnWk_4, %adbc2Wk_4 : tensor<192x192xf32>
    %adlrWk_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWk_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWk_4 = stablehlo.sqrt %advhWk_4 : tensor<192x192xf32>
    %addenWk_4 = stablehlo.add %adsqWk_4, %adepsWk_4 : tensor<192x192xf32>
    %adratWk_4 = stablehlo.divide %admhWk_4, %addenWk_4 : tensor<192x192xf32>
    %adstWk_4 = stablehlo.multiply %adlrWk_4, %adratWk_4 : tensor<192x192xf32>
    %adsubWk_4 = stablehlo.subtract %Wk_4, %adstWk_4 : tensor<192x192xf32>
    %adwdWk_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWk_4 = stablehlo.multiply %adwdWk_4, %adlrWk_4 : tensor<192x192xf32>
    %adwdpWk_4 = stablehlo.multiply %adwdlrWk_4, %Wk_4 : tensor<192x192xf32>
    %adnewWk_4 = stablehlo.subtract %adsubWk_4, %adwdpWk_4 : tensor<192x192xf32>
    %adb1bk_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bk_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbk_4 = stablehlo.multiply %adb1bk_4, %bk_4m : tensor<192xf32>
    %admgbk_4 = stablehlo.multiply %adob1bk_4, %vitb4_mdbK : tensor<192xf32>
    %admnbk_4 = stablehlo.add %admsbk_4, %admgbk_4 : tensor<192xf32>
    %adb2bk_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bk_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbk_4 = stablehlo.multiply %adb2bk_4, %bk_4v : tensor<192xf32>
    %adg2bk_4 = stablehlo.multiply %vitb4_mdbK, %vitb4_mdbK : tensor<192xf32>
    %advgbk_4 = stablehlo.multiply %adob2bk_4, %adg2bk_4 : tensor<192xf32>
    %advnbk_4 = stablehlo.add %advsbk_4, %advgbk_4 : tensor<192xf32>
    %adbc1bk_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bk_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbk_4 = stablehlo.divide %admnbk_4, %adbc1bk_4 : tensor<192xf32>
    %advhbk_4 = stablehlo.divide %advnbk_4, %adbc2bk_4 : tensor<192xf32>
    %adlrbk_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbk_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbk_4 = stablehlo.sqrt %advhbk_4 : tensor<192xf32>
    %addenbk_4 = stablehlo.add %adsqbk_4, %adepsbk_4 : tensor<192xf32>
    %adratbk_4 = stablehlo.divide %admhbk_4, %addenbk_4 : tensor<192xf32>
    %adstbk_4 = stablehlo.multiply %adlrbk_4, %adratbk_4 : tensor<192xf32>
    %adsubbk_4 = stablehlo.subtract %bk_4, %adstbk_4 : tensor<192xf32>
    %adwdbk_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbk_4 = stablehlo.multiply %adwdbk_4, %adlrbk_4 : tensor<192xf32>
    %adwdpbk_4 = stablehlo.multiply %adwdlrbk_4, %bk_4 : tensor<192xf32>
    %adnewbk_4 = stablehlo.subtract %adsubbk_4, %adwdpbk_4 : tensor<192xf32>
    %adb1Wv_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wv_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWv_4 = stablehlo.multiply %adb1Wv_4, %Wv_4m : tensor<192x192xf32>
    %admgWv_4 = stablehlo.multiply %adob1Wv_4, %vitb4_mdWV : tensor<192x192xf32>
    %admnWv_4 = stablehlo.add %admsWv_4, %admgWv_4 : tensor<192x192xf32>
    %adb2Wv_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wv_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWv_4 = stablehlo.multiply %adb2Wv_4, %Wv_4v : tensor<192x192xf32>
    %adg2Wv_4 = stablehlo.multiply %vitb4_mdWV, %vitb4_mdWV : tensor<192x192xf32>
    %advgWv_4 = stablehlo.multiply %adob2Wv_4, %adg2Wv_4 : tensor<192x192xf32>
    %advnWv_4 = stablehlo.add %advsWv_4, %advgWv_4 : tensor<192x192xf32>
    %adbc1Wv_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wv_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWv_4 = stablehlo.divide %admnWv_4, %adbc1Wv_4 : tensor<192x192xf32>
    %advhWv_4 = stablehlo.divide %advnWv_4, %adbc2Wv_4 : tensor<192x192xf32>
    %adlrWv_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWv_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWv_4 = stablehlo.sqrt %advhWv_4 : tensor<192x192xf32>
    %addenWv_4 = stablehlo.add %adsqWv_4, %adepsWv_4 : tensor<192x192xf32>
    %adratWv_4 = stablehlo.divide %admhWv_4, %addenWv_4 : tensor<192x192xf32>
    %adstWv_4 = stablehlo.multiply %adlrWv_4, %adratWv_4 : tensor<192x192xf32>
    %adsubWv_4 = stablehlo.subtract %Wv_4, %adstWv_4 : tensor<192x192xf32>
    %adwdWv_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWv_4 = stablehlo.multiply %adwdWv_4, %adlrWv_4 : tensor<192x192xf32>
    %adwdpWv_4 = stablehlo.multiply %adwdlrWv_4, %Wv_4 : tensor<192x192xf32>
    %adnewWv_4 = stablehlo.subtract %adsubWv_4, %adwdpWv_4 : tensor<192x192xf32>
    %adb1bv_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bv_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbv_4 = stablehlo.multiply %adb1bv_4, %bv_4m : tensor<192xf32>
    %admgbv_4 = stablehlo.multiply %adob1bv_4, %vitb4_mdbV : tensor<192xf32>
    %admnbv_4 = stablehlo.add %admsbv_4, %admgbv_4 : tensor<192xf32>
    %adb2bv_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bv_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbv_4 = stablehlo.multiply %adb2bv_4, %bv_4v : tensor<192xf32>
    %adg2bv_4 = stablehlo.multiply %vitb4_mdbV, %vitb4_mdbV : tensor<192xf32>
    %advgbv_4 = stablehlo.multiply %adob2bv_4, %adg2bv_4 : tensor<192xf32>
    %advnbv_4 = stablehlo.add %advsbv_4, %advgbv_4 : tensor<192xf32>
    %adbc1bv_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bv_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbv_4 = stablehlo.divide %admnbv_4, %adbc1bv_4 : tensor<192xf32>
    %advhbv_4 = stablehlo.divide %advnbv_4, %adbc2bv_4 : tensor<192xf32>
    %adlrbv_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbv_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbv_4 = stablehlo.sqrt %advhbv_4 : tensor<192xf32>
    %addenbv_4 = stablehlo.add %adsqbv_4, %adepsbv_4 : tensor<192xf32>
    %adratbv_4 = stablehlo.divide %admhbv_4, %addenbv_4 : tensor<192xf32>
    %adstbv_4 = stablehlo.multiply %adlrbv_4, %adratbv_4 : tensor<192xf32>
    %adsubbv_4 = stablehlo.subtract %bv_4, %adstbv_4 : tensor<192xf32>
    %adwdbv_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbv_4 = stablehlo.multiply %adwdbv_4, %adlrbv_4 : tensor<192xf32>
    %adwdpbv_4 = stablehlo.multiply %adwdlrbv_4, %bv_4 : tensor<192xf32>
    %adnewbv_4 = stablehlo.subtract %adsubbv_4, %adwdpbv_4 : tensor<192xf32>
    %adb1Wo_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wo_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWo_4 = stablehlo.multiply %adb1Wo_4, %Wo_4m : tensor<192x192xf32>
    %admgWo_4 = stablehlo.multiply %adob1Wo_4, %vitb4_mdWo : tensor<192x192xf32>
    %admnWo_4 = stablehlo.add %admsWo_4, %admgWo_4 : tensor<192x192xf32>
    %adb2Wo_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wo_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWo_4 = stablehlo.multiply %adb2Wo_4, %Wo_4v : tensor<192x192xf32>
    %adg2Wo_4 = stablehlo.multiply %vitb4_mdWo, %vitb4_mdWo : tensor<192x192xf32>
    %advgWo_4 = stablehlo.multiply %adob2Wo_4, %adg2Wo_4 : tensor<192x192xf32>
    %advnWo_4 = stablehlo.add %advsWo_4, %advgWo_4 : tensor<192x192xf32>
    %adbc1Wo_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wo_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWo_4 = stablehlo.divide %admnWo_4, %adbc1Wo_4 : tensor<192x192xf32>
    %advhWo_4 = stablehlo.divide %advnWo_4, %adbc2Wo_4 : tensor<192x192xf32>
    %adlrWo_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWo_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWo_4 = stablehlo.sqrt %advhWo_4 : tensor<192x192xf32>
    %addenWo_4 = stablehlo.add %adsqWo_4, %adepsWo_4 : tensor<192x192xf32>
    %adratWo_4 = stablehlo.divide %admhWo_4, %addenWo_4 : tensor<192x192xf32>
    %adstWo_4 = stablehlo.multiply %adlrWo_4, %adratWo_4 : tensor<192x192xf32>
    %adsubWo_4 = stablehlo.subtract %Wo_4, %adstWo_4 : tensor<192x192xf32>
    %adwdWo_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWo_4 = stablehlo.multiply %adwdWo_4, %adlrWo_4 : tensor<192x192xf32>
    %adwdpWo_4 = stablehlo.multiply %adwdlrWo_4, %Wo_4 : tensor<192x192xf32>
    %adnewWo_4 = stablehlo.subtract %adsubWo_4, %adwdpWo_4 : tensor<192x192xf32>
    %adb1bo_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bo_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbo_4 = stablehlo.multiply %adb1bo_4, %bo_4m : tensor<192xf32>
    %admgbo_4 = stablehlo.multiply %adob1bo_4, %vitb4_mdbo : tensor<192xf32>
    %admnbo_4 = stablehlo.add %admsbo_4, %admgbo_4 : tensor<192xf32>
    %adb2bo_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bo_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbo_4 = stablehlo.multiply %adb2bo_4, %bo_4v : tensor<192xf32>
    %adg2bo_4 = stablehlo.multiply %vitb4_mdbo, %vitb4_mdbo : tensor<192xf32>
    %advgbo_4 = stablehlo.multiply %adob2bo_4, %adg2bo_4 : tensor<192xf32>
    %advnbo_4 = stablehlo.add %advsbo_4, %advgbo_4 : tensor<192xf32>
    %adbc1bo_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bo_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbo_4 = stablehlo.divide %admnbo_4, %adbc1bo_4 : tensor<192xf32>
    %advhbo_4 = stablehlo.divide %advnbo_4, %adbc2bo_4 : tensor<192xf32>
    %adlrbo_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbo_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbo_4 = stablehlo.sqrt %advhbo_4 : tensor<192xf32>
    %addenbo_4 = stablehlo.add %adsqbo_4, %adepsbo_4 : tensor<192xf32>
    %adratbo_4 = stablehlo.divide %admhbo_4, %addenbo_4 : tensor<192xf32>
    %adstbo_4 = stablehlo.multiply %adlrbo_4, %adratbo_4 : tensor<192xf32>
    %adsubbo_4 = stablehlo.subtract %bo_4, %adstbo_4 : tensor<192xf32>
    %adwdbo_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbo_4 = stablehlo.multiply %adwdbo_4, %adlrbo_4 : tensor<192xf32>
    %adwdpbo_4 = stablehlo.multiply %adwdlrbo_4, %bo_4 : tensor<192xf32>
    %adnewbo_4 = stablehlo.subtract %adsubbo_4, %adwdpbo_4 : tensor<192xf32>
    %adb1g2_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g2_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg2_4 = stablehlo.multiply %adb1g2_4, %g2_4m : tensor<192xf32>
    %admgg2_4 = stablehlo.multiply %adob1g2_4, %vitb4_2dg : tensor<192xf32>
    %admng2_4 = stablehlo.add %admsg2_4, %admgg2_4 : tensor<192xf32>
    %adb2g2_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g2_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg2_4 = stablehlo.multiply %adb2g2_4, %g2_4v : tensor<192xf32>
    %adg2g2_4 = stablehlo.multiply %vitb4_2dg, %vitb4_2dg : tensor<192xf32>
    %advgg2_4 = stablehlo.multiply %adob2g2_4, %adg2g2_4 : tensor<192xf32>
    %advng2_4 = stablehlo.add %advsg2_4, %advgg2_4 : tensor<192xf32>
    %adbc1g2_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g2_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg2_4 = stablehlo.divide %admng2_4, %adbc1g2_4 : tensor<192xf32>
    %advhg2_4 = stablehlo.divide %advng2_4, %adbc2g2_4 : tensor<192xf32>
    %adlrg2_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg2_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg2_4 = stablehlo.sqrt %advhg2_4 : tensor<192xf32>
    %addeng2_4 = stablehlo.add %adsqg2_4, %adepsg2_4 : tensor<192xf32>
    %adratg2_4 = stablehlo.divide %admhg2_4, %addeng2_4 : tensor<192xf32>
    %adstg2_4 = stablehlo.multiply %adlrg2_4, %adratg2_4 : tensor<192xf32>
    %adsubg2_4 = stablehlo.subtract %g2_4, %adstg2_4 : tensor<192xf32>
    %adwdg2_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg2_4 = stablehlo.multiply %adwdg2_4, %adlrg2_4 : tensor<192xf32>
    %adwdpg2_4 = stablehlo.multiply %adwdlrg2_4, %g2_4 : tensor<192xf32>
    %adnewg2_4 = stablehlo.subtract %adsubg2_4, %adwdpg2_4 : tensor<192xf32>
    %adb1b2_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b2_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb2_4 = stablehlo.multiply %adb1b2_4, %b2_4m : tensor<192xf32>
    %admgb2_4 = stablehlo.multiply %adob1b2_4, %vitb4_2db : tensor<192xf32>
    %admnb2_4 = stablehlo.add %admsb2_4, %admgb2_4 : tensor<192xf32>
    %adb2b2_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b2_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb2_4 = stablehlo.multiply %adb2b2_4, %b2_4v : tensor<192xf32>
    %adg2b2_4 = stablehlo.multiply %vitb4_2db, %vitb4_2db : tensor<192xf32>
    %advgb2_4 = stablehlo.multiply %adob2b2_4, %adg2b2_4 : tensor<192xf32>
    %advnb2_4 = stablehlo.add %advsb2_4, %advgb2_4 : tensor<192xf32>
    %adbc1b2_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b2_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb2_4 = stablehlo.divide %admnb2_4, %adbc1b2_4 : tensor<192xf32>
    %advhb2_4 = stablehlo.divide %advnb2_4, %adbc2b2_4 : tensor<192xf32>
    %adlrb2_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb2_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb2_4 = stablehlo.sqrt %advhb2_4 : tensor<192xf32>
    %addenb2_4 = stablehlo.add %adsqb2_4, %adepsb2_4 : tensor<192xf32>
    %adratb2_4 = stablehlo.divide %admhb2_4, %addenb2_4 : tensor<192xf32>
    %adstb2_4 = stablehlo.multiply %adlrb2_4, %adratb2_4 : tensor<192xf32>
    %adsubb2_4 = stablehlo.subtract %b2_4, %adstb2_4 : tensor<192xf32>
    %adwdb2_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb2_4 = stablehlo.multiply %adwdb2_4, %adlrb2_4 : tensor<192xf32>
    %adwdpb2_4 = stablehlo.multiply %adwdlrb2_4, %b2_4 : tensor<192xf32>
    %adnewb2_4 = stablehlo.subtract %adsubb2_4, %adwdpb2_4 : tensor<192xf32>
    %adb1Wfc1_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob1Wfc1_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admsWfc1_4 = stablehlo.multiply %adb1Wfc1_4, %Wfc1_4m : tensor<192x768xf32>
    %admgWfc1_4 = stablehlo.multiply %adob1Wfc1_4, %vitb4_pdWfc1 : tensor<192x768xf32>
    %admnWfc1_4 = stablehlo.add %admsWfc1_4, %admgWfc1_4 : tensor<192x768xf32>
    %adb2Wfc1_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob2Wfc1_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %advsWfc1_4 = stablehlo.multiply %adb2Wfc1_4, %Wfc1_4v : tensor<192x768xf32>
    %adg2Wfc1_4 = stablehlo.multiply %vitb4_pdWfc1, %vitb4_pdWfc1 : tensor<192x768xf32>
    %advgWfc1_4 = stablehlo.multiply %adob2Wfc1_4, %adg2Wfc1_4 : tensor<192x768xf32>
    %advnWfc1_4 = stablehlo.add %advsWfc1_4, %advgWfc1_4 : tensor<192x768xf32>
    %adbc1Wfc1_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adbc2Wfc1_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admhWfc1_4 = stablehlo.divide %admnWfc1_4, %adbc1Wfc1_4 : tensor<192x768xf32>
    %advhWfc1_4 = stablehlo.divide %advnWfc1_4, %adbc2Wfc1_4 : tensor<192x768xf32>
    %adlrWfc1_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adepsWfc1_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adsqWfc1_4 = stablehlo.sqrt %advhWfc1_4 : tensor<192x768xf32>
    %addenWfc1_4 = stablehlo.add %adsqWfc1_4, %adepsWfc1_4 : tensor<192x768xf32>
    %adratWfc1_4 = stablehlo.divide %admhWfc1_4, %addenWfc1_4 : tensor<192x768xf32>
    %adstWfc1_4 = stablehlo.multiply %adlrWfc1_4, %adratWfc1_4 : tensor<192x768xf32>
    %adsubWfc1_4 = stablehlo.subtract %Wfc1_4, %adstWfc1_4 : tensor<192x768xf32>
    %adwdWfc1_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adwdlrWfc1_4 = stablehlo.multiply %adwdWfc1_4, %adlrWfc1_4 : tensor<192x768xf32>
    %adwdpWfc1_4 = stablehlo.multiply %adwdlrWfc1_4, %Wfc1_4 : tensor<192x768xf32>
    %adnewWfc1_4 = stablehlo.subtract %adsubWfc1_4, %adwdpWfc1_4 : tensor<192x768xf32>
    %adb1bfc1_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1bfc1_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admsbfc1_4 = stablehlo.multiply %adb1bfc1_4, %bfc1_4m : tensor<768xf32>
    %admgbfc1_4 = stablehlo.multiply %adob1bfc1_4, %vitb4_pdbfc1 : tensor<768xf32>
    %admnbfc1_4 = stablehlo.add %admsbfc1_4, %admgbfc1_4 : tensor<768xf32>
    %adb2bfc1_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2bfc1_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advsbfc1_4 = stablehlo.multiply %adb2bfc1_4, %bfc1_4v : tensor<768xf32>
    %adg2bfc1_4 = stablehlo.multiply %vitb4_pdbfc1, %vitb4_pdbfc1 : tensor<768xf32>
    %advgbfc1_4 = stablehlo.multiply %adob2bfc1_4, %adg2bfc1_4 : tensor<768xf32>
    %advnbfc1_4 = stablehlo.add %advsbfc1_4, %advgbfc1_4 : tensor<768xf32>
    %adbc1bfc1_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2bfc1_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhbfc1_4 = stablehlo.divide %admnbfc1_4, %adbc1bfc1_4 : tensor<768xf32>
    %advhbfc1_4 = stablehlo.divide %advnbfc1_4, %adbc2bfc1_4 : tensor<768xf32>
    %adlrbfc1_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepsbfc1_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqbfc1_4 = stablehlo.sqrt %advhbfc1_4 : tensor<768xf32>
    %addenbfc1_4 = stablehlo.add %adsqbfc1_4, %adepsbfc1_4 : tensor<768xf32>
    %adratbfc1_4 = stablehlo.divide %admhbfc1_4, %addenbfc1_4 : tensor<768xf32>
    %adstbfc1_4 = stablehlo.multiply %adlrbfc1_4, %adratbfc1_4 : tensor<768xf32>
    %adsubbfc1_4 = stablehlo.subtract %bfc1_4, %adstbfc1_4 : tensor<768xf32>
    %adwdbfc1_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrbfc1_4 = stablehlo.multiply %adwdbfc1_4, %adlrbfc1_4 : tensor<768xf32>
    %adwdpbfc1_4 = stablehlo.multiply %adwdlrbfc1_4, %bfc1_4 : tensor<768xf32>
    %adnewbfc1_4 = stablehlo.subtract %adsubbfc1_4, %adwdpbfc1_4 : tensor<768xf32>
    %adb1Wfc2_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob1Wfc2_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admsWfc2_4 = stablehlo.multiply %adb1Wfc2_4, %Wfc2_4m : tensor<768x192xf32>
    %admgWfc2_4 = stablehlo.multiply %adob1Wfc2_4, %vitb4_pdWfc2 : tensor<768x192xf32>
    %admnWfc2_4 = stablehlo.add %admsWfc2_4, %admgWfc2_4 : tensor<768x192xf32>
    %adb2Wfc2_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob2Wfc2_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %advsWfc2_4 = stablehlo.multiply %adb2Wfc2_4, %Wfc2_4v : tensor<768x192xf32>
    %adg2Wfc2_4 = stablehlo.multiply %vitb4_pdWfc2, %vitb4_pdWfc2 : tensor<768x192xf32>
    %advgWfc2_4 = stablehlo.multiply %adob2Wfc2_4, %adg2Wfc2_4 : tensor<768x192xf32>
    %advnWfc2_4 = stablehlo.add %advsWfc2_4, %advgWfc2_4 : tensor<768x192xf32>
    %adbc1Wfc2_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adbc2Wfc2_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admhWfc2_4 = stablehlo.divide %admnWfc2_4, %adbc1Wfc2_4 : tensor<768x192xf32>
    %advhWfc2_4 = stablehlo.divide %advnWfc2_4, %adbc2Wfc2_4 : tensor<768x192xf32>
    %adlrWfc2_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adepsWfc2_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adsqWfc2_4 = stablehlo.sqrt %advhWfc2_4 : tensor<768x192xf32>
    %addenWfc2_4 = stablehlo.add %adsqWfc2_4, %adepsWfc2_4 : tensor<768x192xf32>
    %adratWfc2_4 = stablehlo.divide %admhWfc2_4, %addenWfc2_4 : tensor<768x192xf32>
    %adstWfc2_4 = stablehlo.multiply %adlrWfc2_4, %adratWfc2_4 : tensor<768x192xf32>
    %adsubWfc2_4 = stablehlo.subtract %Wfc2_4, %adstWfc2_4 : tensor<768x192xf32>
    %adwdWfc2_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adwdlrWfc2_4 = stablehlo.multiply %adwdWfc2_4, %adlrWfc2_4 : tensor<768x192xf32>
    %adwdpWfc2_4 = stablehlo.multiply %adwdlrWfc2_4, %Wfc2_4 : tensor<768x192xf32>
    %adnewWfc2_4 = stablehlo.subtract %adsubWfc2_4, %adwdpWfc2_4 : tensor<768x192xf32>
    %adb1bfc2_4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bfc2_4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbfc2_4 = stablehlo.multiply %adb1bfc2_4, %bfc2_4m : tensor<192xf32>
    %admgbfc2_4 = stablehlo.multiply %adob1bfc2_4, %vitb4_pdbfc2 : tensor<192xf32>
    %admnbfc2_4 = stablehlo.add %admsbfc2_4, %admgbfc2_4 : tensor<192xf32>
    %adb2bfc2_4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bfc2_4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbfc2_4 = stablehlo.multiply %adb2bfc2_4, %bfc2_4v : tensor<192xf32>
    %adg2bfc2_4 = stablehlo.multiply %vitb4_pdbfc2, %vitb4_pdbfc2 : tensor<192xf32>
    %advgbfc2_4 = stablehlo.multiply %adob2bfc2_4, %adg2bfc2_4 : tensor<192xf32>
    %advnbfc2_4 = stablehlo.add %advsbfc2_4, %advgbfc2_4 : tensor<192xf32>
    %adbc1bfc2_4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bfc2_4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbfc2_4 = stablehlo.divide %admnbfc2_4, %adbc1bfc2_4 : tensor<192xf32>
    %advhbfc2_4 = stablehlo.divide %advnbfc2_4, %adbc2bfc2_4 : tensor<192xf32>
    %adlrbfc2_4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbfc2_4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbfc2_4 = stablehlo.sqrt %advhbfc2_4 : tensor<192xf32>
    %addenbfc2_4 = stablehlo.add %adsqbfc2_4, %adepsbfc2_4 : tensor<192xf32>
    %adratbfc2_4 = stablehlo.divide %admhbfc2_4, %addenbfc2_4 : tensor<192xf32>
    %adstbfc2_4 = stablehlo.multiply %adlrbfc2_4, %adratbfc2_4 : tensor<192xf32>
    %adsubbfc2_4 = stablehlo.subtract %bfc2_4, %adstbfc2_4 : tensor<192xf32>
    %adwdbfc2_4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbfc2_4 = stablehlo.multiply %adwdbfc2_4, %adlrbfc2_4 : tensor<192xf32>
    %adwdpbfc2_4 = stablehlo.multiply %adwdlrbfc2_4, %bfc2_4 : tensor<192xf32>
    %adnewbfc2_4 = stablehlo.subtract %adsubbfc2_4, %adwdpbfc2_4 : tensor<192xf32>
    %adb1g1_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g1_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg1_5 = stablehlo.multiply %adb1g1_5, %g1_5m : tensor<192xf32>
    %admgg1_5 = stablehlo.multiply %adob1g1_5, %vitb5_1dg : tensor<192xf32>
    %admng1_5 = stablehlo.add %admsg1_5, %admgg1_5 : tensor<192xf32>
    %adb2g1_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g1_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg1_5 = stablehlo.multiply %adb2g1_5, %g1_5v : tensor<192xf32>
    %adg2g1_5 = stablehlo.multiply %vitb5_1dg, %vitb5_1dg : tensor<192xf32>
    %advgg1_5 = stablehlo.multiply %adob2g1_5, %adg2g1_5 : tensor<192xf32>
    %advng1_5 = stablehlo.add %advsg1_5, %advgg1_5 : tensor<192xf32>
    %adbc1g1_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g1_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg1_5 = stablehlo.divide %admng1_5, %adbc1g1_5 : tensor<192xf32>
    %advhg1_5 = stablehlo.divide %advng1_5, %adbc2g1_5 : tensor<192xf32>
    %adlrg1_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg1_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg1_5 = stablehlo.sqrt %advhg1_5 : tensor<192xf32>
    %addeng1_5 = stablehlo.add %adsqg1_5, %adepsg1_5 : tensor<192xf32>
    %adratg1_5 = stablehlo.divide %admhg1_5, %addeng1_5 : tensor<192xf32>
    %adstg1_5 = stablehlo.multiply %adlrg1_5, %adratg1_5 : tensor<192xf32>
    %adsubg1_5 = stablehlo.subtract %g1_5, %adstg1_5 : tensor<192xf32>
    %adwdg1_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg1_5 = stablehlo.multiply %adwdg1_5, %adlrg1_5 : tensor<192xf32>
    %adwdpg1_5 = stablehlo.multiply %adwdlrg1_5, %g1_5 : tensor<192xf32>
    %adnewg1_5 = stablehlo.subtract %adsubg1_5, %adwdpg1_5 : tensor<192xf32>
    %adb1b1_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b1_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb1_5 = stablehlo.multiply %adb1b1_5, %b1_5m : tensor<192xf32>
    %admgb1_5 = stablehlo.multiply %adob1b1_5, %vitb5_1db : tensor<192xf32>
    %admnb1_5 = stablehlo.add %admsb1_5, %admgb1_5 : tensor<192xf32>
    %adb2b1_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b1_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb1_5 = stablehlo.multiply %adb2b1_5, %b1_5v : tensor<192xf32>
    %adg2b1_5 = stablehlo.multiply %vitb5_1db, %vitb5_1db : tensor<192xf32>
    %advgb1_5 = stablehlo.multiply %adob2b1_5, %adg2b1_5 : tensor<192xf32>
    %advnb1_5 = stablehlo.add %advsb1_5, %advgb1_5 : tensor<192xf32>
    %adbc1b1_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b1_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb1_5 = stablehlo.divide %admnb1_5, %adbc1b1_5 : tensor<192xf32>
    %advhb1_5 = stablehlo.divide %advnb1_5, %adbc2b1_5 : tensor<192xf32>
    %adlrb1_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb1_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb1_5 = stablehlo.sqrt %advhb1_5 : tensor<192xf32>
    %addenb1_5 = stablehlo.add %adsqb1_5, %adepsb1_5 : tensor<192xf32>
    %adratb1_5 = stablehlo.divide %admhb1_5, %addenb1_5 : tensor<192xf32>
    %adstb1_5 = stablehlo.multiply %adlrb1_5, %adratb1_5 : tensor<192xf32>
    %adsubb1_5 = stablehlo.subtract %b1_5, %adstb1_5 : tensor<192xf32>
    %adwdb1_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb1_5 = stablehlo.multiply %adwdb1_5, %adlrb1_5 : tensor<192xf32>
    %adwdpb1_5 = stablehlo.multiply %adwdlrb1_5, %b1_5 : tensor<192xf32>
    %adnewb1_5 = stablehlo.subtract %adsubb1_5, %adwdpb1_5 : tensor<192xf32>
    %adb1Wq_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wq_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWq_5 = stablehlo.multiply %adb1Wq_5, %Wq_5m : tensor<192x192xf32>
    %admgWq_5 = stablehlo.multiply %adob1Wq_5, %vitb5_mdWQ : tensor<192x192xf32>
    %admnWq_5 = stablehlo.add %admsWq_5, %admgWq_5 : tensor<192x192xf32>
    %adb2Wq_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wq_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWq_5 = stablehlo.multiply %adb2Wq_5, %Wq_5v : tensor<192x192xf32>
    %adg2Wq_5 = stablehlo.multiply %vitb5_mdWQ, %vitb5_mdWQ : tensor<192x192xf32>
    %advgWq_5 = stablehlo.multiply %adob2Wq_5, %adg2Wq_5 : tensor<192x192xf32>
    %advnWq_5 = stablehlo.add %advsWq_5, %advgWq_5 : tensor<192x192xf32>
    %adbc1Wq_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wq_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWq_5 = stablehlo.divide %admnWq_5, %adbc1Wq_5 : tensor<192x192xf32>
    %advhWq_5 = stablehlo.divide %advnWq_5, %adbc2Wq_5 : tensor<192x192xf32>
    %adlrWq_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWq_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWq_5 = stablehlo.sqrt %advhWq_5 : tensor<192x192xf32>
    %addenWq_5 = stablehlo.add %adsqWq_5, %adepsWq_5 : tensor<192x192xf32>
    %adratWq_5 = stablehlo.divide %admhWq_5, %addenWq_5 : tensor<192x192xf32>
    %adstWq_5 = stablehlo.multiply %adlrWq_5, %adratWq_5 : tensor<192x192xf32>
    %adsubWq_5 = stablehlo.subtract %Wq_5, %adstWq_5 : tensor<192x192xf32>
    %adwdWq_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWq_5 = stablehlo.multiply %adwdWq_5, %adlrWq_5 : tensor<192x192xf32>
    %adwdpWq_5 = stablehlo.multiply %adwdlrWq_5, %Wq_5 : tensor<192x192xf32>
    %adnewWq_5 = stablehlo.subtract %adsubWq_5, %adwdpWq_5 : tensor<192x192xf32>
    %adb1bq_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bq_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbq_5 = stablehlo.multiply %adb1bq_5, %bq_5m : tensor<192xf32>
    %admgbq_5 = stablehlo.multiply %adob1bq_5, %vitb5_mdbQ : tensor<192xf32>
    %admnbq_5 = stablehlo.add %admsbq_5, %admgbq_5 : tensor<192xf32>
    %adb2bq_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bq_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbq_5 = stablehlo.multiply %adb2bq_5, %bq_5v : tensor<192xf32>
    %adg2bq_5 = stablehlo.multiply %vitb5_mdbQ, %vitb5_mdbQ : tensor<192xf32>
    %advgbq_5 = stablehlo.multiply %adob2bq_5, %adg2bq_5 : tensor<192xf32>
    %advnbq_5 = stablehlo.add %advsbq_5, %advgbq_5 : tensor<192xf32>
    %adbc1bq_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bq_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbq_5 = stablehlo.divide %admnbq_5, %adbc1bq_5 : tensor<192xf32>
    %advhbq_5 = stablehlo.divide %advnbq_5, %adbc2bq_5 : tensor<192xf32>
    %adlrbq_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbq_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbq_5 = stablehlo.sqrt %advhbq_5 : tensor<192xf32>
    %addenbq_5 = stablehlo.add %adsqbq_5, %adepsbq_5 : tensor<192xf32>
    %adratbq_5 = stablehlo.divide %admhbq_5, %addenbq_5 : tensor<192xf32>
    %adstbq_5 = stablehlo.multiply %adlrbq_5, %adratbq_5 : tensor<192xf32>
    %adsubbq_5 = stablehlo.subtract %bq_5, %adstbq_5 : tensor<192xf32>
    %adwdbq_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbq_5 = stablehlo.multiply %adwdbq_5, %adlrbq_5 : tensor<192xf32>
    %adwdpbq_5 = stablehlo.multiply %adwdlrbq_5, %bq_5 : tensor<192xf32>
    %adnewbq_5 = stablehlo.subtract %adsubbq_5, %adwdpbq_5 : tensor<192xf32>
    %adb1Wk_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wk_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWk_5 = stablehlo.multiply %adb1Wk_5, %Wk_5m : tensor<192x192xf32>
    %admgWk_5 = stablehlo.multiply %adob1Wk_5, %vitb5_mdWK : tensor<192x192xf32>
    %admnWk_5 = stablehlo.add %admsWk_5, %admgWk_5 : tensor<192x192xf32>
    %adb2Wk_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wk_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWk_5 = stablehlo.multiply %adb2Wk_5, %Wk_5v : tensor<192x192xf32>
    %adg2Wk_5 = stablehlo.multiply %vitb5_mdWK, %vitb5_mdWK : tensor<192x192xf32>
    %advgWk_5 = stablehlo.multiply %adob2Wk_5, %adg2Wk_5 : tensor<192x192xf32>
    %advnWk_5 = stablehlo.add %advsWk_5, %advgWk_5 : tensor<192x192xf32>
    %adbc1Wk_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wk_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWk_5 = stablehlo.divide %admnWk_5, %adbc1Wk_5 : tensor<192x192xf32>
    %advhWk_5 = stablehlo.divide %advnWk_5, %adbc2Wk_5 : tensor<192x192xf32>
    %adlrWk_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWk_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWk_5 = stablehlo.sqrt %advhWk_5 : tensor<192x192xf32>
    %addenWk_5 = stablehlo.add %adsqWk_5, %adepsWk_5 : tensor<192x192xf32>
    %adratWk_5 = stablehlo.divide %admhWk_5, %addenWk_5 : tensor<192x192xf32>
    %adstWk_5 = stablehlo.multiply %adlrWk_5, %adratWk_5 : tensor<192x192xf32>
    %adsubWk_5 = stablehlo.subtract %Wk_5, %adstWk_5 : tensor<192x192xf32>
    %adwdWk_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWk_5 = stablehlo.multiply %adwdWk_5, %adlrWk_5 : tensor<192x192xf32>
    %adwdpWk_5 = stablehlo.multiply %adwdlrWk_5, %Wk_5 : tensor<192x192xf32>
    %adnewWk_5 = stablehlo.subtract %adsubWk_5, %adwdpWk_5 : tensor<192x192xf32>
    %adb1bk_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bk_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbk_5 = stablehlo.multiply %adb1bk_5, %bk_5m : tensor<192xf32>
    %admgbk_5 = stablehlo.multiply %adob1bk_5, %vitb5_mdbK : tensor<192xf32>
    %admnbk_5 = stablehlo.add %admsbk_5, %admgbk_5 : tensor<192xf32>
    %adb2bk_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bk_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbk_5 = stablehlo.multiply %adb2bk_5, %bk_5v : tensor<192xf32>
    %adg2bk_5 = stablehlo.multiply %vitb5_mdbK, %vitb5_mdbK : tensor<192xf32>
    %advgbk_5 = stablehlo.multiply %adob2bk_5, %adg2bk_5 : tensor<192xf32>
    %advnbk_5 = stablehlo.add %advsbk_5, %advgbk_5 : tensor<192xf32>
    %adbc1bk_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bk_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbk_5 = stablehlo.divide %admnbk_5, %adbc1bk_5 : tensor<192xf32>
    %advhbk_5 = stablehlo.divide %advnbk_5, %adbc2bk_5 : tensor<192xf32>
    %adlrbk_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbk_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbk_5 = stablehlo.sqrt %advhbk_5 : tensor<192xf32>
    %addenbk_5 = stablehlo.add %adsqbk_5, %adepsbk_5 : tensor<192xf32>
    %adratbk_5 = stablehlo.divide %admhbk_5, %addenbk_5 : tensor<192xf32>
    %adstbk_5 = stablehlo.multiply %adlrbk_5, %adratbk_5 : tensor<192xf32>
    %adsubbk_5 = stablehlo.subtract %bk_5, %adstbk_5 : tensor<192xf32>
    %adwdbk_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbk_5 = stablehlo.multiply %adwdbk_5, %adlrbk_5 : tensor<192xf32>
    %adwdpbk_5 = stablehlo.multiply %adwdlrbk_5, %bk_5 : tensor<192xf32>
    %adnewbk_5 = stablehlo.subtract %adsubbk_5, %adwdpbk_5 : tensor<192xf32>
    %adb1Wv_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wv_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWv_5 = stablehlo.multiply %adb1Wv_5, %Wv_5m : tensor<192x192xf32>
    %admgWv_5 = stablehlo.multiply %adob1Wv_5, %vitb5_mdWV : tensor<192x192xf32>
    %admnWv_5 = stablehlo.add %admsWv_5, %admgWv_5 : tensor<192x192xf32>
    %adb2Wv_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wv_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWv_5 = stablehlo.multiply %adb2Wv_5, %Wv_5v : tensor<192x192xf32>
    %adg2Wv_5 = stablehlo.multiply %vitb5_mdWV, %vitb5_mdWV : tensor<192x192xf32>
    %advgWv_5 = stablehlo.multiply %adob2Wv_5, %adg2Wv_5 : tensor<192x192xf32>
    %advnWv_5 = stablehlo.add %advsWv_5, %advgWv_5 : tensor<192x192xf32>
    %adbc1Wv_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wv_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWv_5 = stablehlo.divide %admnWv_5, %adbc1Wv_5 : tensor<192x192xf32>
    %advhWv_5 = stablehlo.divide %advnWv_5, %adbc2Wv_5 : tensor<192x192xf32>
    %adlrWv_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWv_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWv_5 = stablehlo.sqrt %advhWv_5 : tensor<192x192xf32>
    %addenWv_5 = stablehlo.add %adsqWv_5, %adepsWv_5 : tensor<192x192xf32>
    %adratWv_5 = stablehlo.divide %admhWv_5, %addenWv_5 : tensor<192x192xf32>
    %adstWv_5 = stablehlo.multiply %adlrWv_5, %adratWv_5 : tensor<192x192xf32>
    %adsubWv_5 = stablehlo.subtract %Wv_5, %adstWv_5 : tensor<192x192xf32>
    %adwdWv_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWv_5 = stablehlo.multiply %adwdWv_5, %adlrWv_5 : tensor<192x192xf32>
    %adwdpWv_5 = stablehlo.multiply %adwdlrWv_5, %Wv_5 : tensor<192x192xf32>
    %adnewWv_5 = stablehlo.subtract %adsubWv_5, %adwdpWv_5 : tensor<192x192xf32>
    %adb1bv_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bv_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbv_5 = stablehlo.multiply %adb1bv_5, %bv_5m : tensor<192xf32>
    %admgbv_5 = stablehlo.multiply %adob1bv_5, %vitb5_mdbV : tensor<192xf32>
    %admnbv_5 = stablehlo.add %admsbv_5, %admgbv_5 : tensor<192xf32>
    %adb2bv_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bv_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbv_5 = stablehlo.multiply %adb2bv_5, %bv_5v : tensor<192xf32>
    %adg2bv_5 = stablehlo.multiply %vitb5_mdbV, %vitb5_mdbV : tensor<192xf32>
    %advgbv_5 = stablehlo.multiply %adob2bv_5, %adg2bv_5 : tensor<192xf32>
    %advnbv_5 = stablehlo.add %advsbv_5, %advgbv_5 : tensor<192xf32>
    %adbc1bv_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bv_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbv_5 = stablehlo.divide %admnbv_5, %adbc1bv_5 : tensor<192xf32>
    %advhbv_5 = stablehlo.divide %advnbv_5, %adbc2bv_5 : tensor<192xf32>
    %adlrbv_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbv_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbv_5 = stablehlo.sqrt %advhbv_5 : tensor<192xf32>
    %addenbv_5 = stablehlo.add %adsqbv_5, %adepsbv_5 : tensor<192xf32>
    %adratbv_5 = stablehlo.divide %admhbv_5, %addenbv_5 : tensor<192xf32>
    %adstbv_5 = stablehlo.multiply %adlrbv_5, %adratbv_5 : tensor<192xf32>
    %adsubbv_5 = stablehlo.subtract %bv_5, %adstbv_5 : tensor<192xf32>
    %adwdbv_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbv_5 = stablehlo.multiply %adwdbv_5, %adlrbv_5 : tensor<192xf32>
    %adwdpbv_5 = stablehlo.multiply %adwdlrbv_5, %bv_5 : tensor<192xf32>
    %adnewbv_5 = stablehlo.subtract %adsubbv_5, %adwdpbv_5 : tensor<192xf32>
    %adb1Wo_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wo_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWo_5 = stablehlo.multiply %adb1Wo_5, %Wo_5m : tensor<192x192xf32>
    %admgWo_5 = stablehlo.multiply %adob1Wo_5, %vitb5_mdWo : tensor<192x192xf32>
    %admnWo_5 = stablehlo.add %admsWo_5, %admgWo_5 : tensor<192x192xf32>
    %adb2Wo_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wo_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWo_5 = stablehlo.multiply %adb2Wo_5, %Wo_5v : tensor<192x192xf32>
    %adg2Wo_5 = stablehlo.multiply %vitb5_mdWo, %vitb5_mdWo : tensor<192x192xf32>
    %advgWo_5 = stablehlo.multiply %adob2Wo_5, %adg2Wo_5 : tensor<192x192xf32>
    %advnWo_5 = stablehlo.add %advsWo_5, %advgWo_5 : tensor<192x192xf32>
    %adbc1Wo_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wo_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWo_5 = stablehlo.divide %admnWo_5, %adbc1Wo_5 : tensor<192x192xf32>
    %advhWo_5 = stablehlo.divide %advnWo_5, %adbc2Wo_5 : tensor<192x192xf32>
    %adlrWo_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWo_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWo_5 = stablehlo.sqrt %advhWo_5 : tensor<192x192xf32>
    %addenWo_5 = stablehlo.add %adsqWo_5, %adepsWo_5 : tensor<192x192xf32>
    %adratWo_5 = stablehlo.divide %admhWo_5, %addenWo_5 : tensor<192x192xf32>
    %adstWo_5 = stablehlo.multiply %adlrWo_5, %adratWo_5 : tensor<192x192xf32>
    %adsubWo_5 = stablehlo.subtract %Wo_5, %adstWo_5 : tensor<192x192xf32>
    %adwdWo_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWo_5 = stablehlo.multiply %adwdWo_5, %adlrWo_5 : tensor<192x192xf32>
    %adwdpWo_5 = stablehlo.multiply %adwdlrWo_5, %Wo_5 : tensor<192x192xf32>
    %adnewWo_5 = stablehlo.subtract %adsubWo_5, %adwdpWo_5 : tensor<192x192xf32>
    %adb1bo_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bo_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbo_5 = stablehlo.multiply %adb1bo_5, %bo_5m : tensor<192xf32>
    %admgbo_5 = stablehlo.multiply %adob1bo_5, %vitb5_mdbo : tensor<192xf32>
    %admnbo_5 = stablehlo.add %admsbo_5, %admgbo_5 : tensor<192xf32>
    %adb2bo_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bo_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbo_5 = stablehlo.multiply %adb2bo_5, %bo_5v : tensor<192xf32>
    %adg2bo_5 = stablehlo.multiply %vitb5_mdbo, %vitb5_mdbo : tensor<192xf32>
    %advgbo_5 = stablehlo.multiply %adob2bo_5, %adg2bo_5 : tensor<192xf32>
    %advnbo_5 = stablehlo.add %advsbo_5, %advgbo_5 : tensor<192xf32>
    %adbc1bo_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bo_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbo_5 = stablehlo.divide %admnbo_5, %adbc1bo_5 : tensor<192xf32>
    %advhbo_5 = stablehlo.divide %advnbo_5, %adbc2bo_5 : tensor<192xf32>
    %adlrbo_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbo_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbo_5 = stablehlo.sqrt %advhbo_5 : tensor<192xf32>
    %addenbo_5 = stablehlo.add %adsqbo_5, %adepsbo_5 : tensor<192xf32>
    %adratbo_5 = stablehlo.divide %admhbo_5, %addenbo_5 : tensor<192xf32>
    %adstbo_5 = stablehlo.multiply %adlrbo_5, %adratbo_5 : tensor<192xf32>
    %adsubbo_5 = stablehlo.subtract %bo_5, %adstbo_5 : tensor<192xf32>
    %adwdbo_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbo_5 = stablehlo.multiply %adwdbo_5, %adlrbo_5 : tensor<192xf32>
    %adwdpbo_5 = stablehlo.multiply %adwdlrbo_5, %bo_5 : tensor<192xf32>
    %adnewbo_5 = stablehlo.subtract %adsubbo_5, %adwdpbo_5 : tensor<192xf32>
    %adb1g2_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g2_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg2_5 = stablehlo.multiply %adb1g2_5, %g2_5m : tensor<192xf32>
    %admgg2_5 = stablehlo.multiply %adob1g2_5, %vitb5_2dg : tensor<192xf32>
    %admng2_5 = stablehlo.add %admsg2_5, %admgg2_5 : tensor<192xf32>
    %adb2g2_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g2_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg2_5 = stablehlo.multiply %adb2g2_5, %g2_5v : tensor<192xf32>
    %adg2g2_5 = stablehlo.multiply %vitb5_2dg, %vitb5_2dg : tensor<192xf32>
    %advgg2_5 = stablehlo.multiply %adob2g2_5, %adg2g2_5 : tensor<192xf32>
    %advng2_5 = stablehlo.add %advsg2_5, %advgg2_5 : tensor<192xf32>
    %adbc1g2_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g2_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg2_5 = stablehlo.divide %admng2_5, %adbc1g2_5 : tensor<192xf32>
    %advhg2_5 = stablehlo.divide %advng2_5, %adbc2g2_5 : tensor<192xf32>
    %adlrg2_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg2_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg2_5 = stablehlo.sqrt %advhg2_5 : tensor<192xf32>
    %addeng2_5 = stablehlo.add %adsqg2_5, %adepsg2_5 : tensor<192xf32>
    %adratg2_5 = stablehlo.divide %admhg2_5, %addeng2_5 : tensor<192xf32>
    %adstg2_5 = stablehlo.multiply %adlrg2_5, %adratg2_5 : tensor<192xf32>
    %adsubg2_5 = stablehlo.subtract %g2_5, %adstg2_5 : tensor<192xf32>
    %adwdg2_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg2_5 = stablehlo.multiply %adwdg2_5, %adlrg2_5 : tensor<192xf32>
    %adwdpg2_5 = stablehlo.multiply %adwdlrg2_5, %g2_5 : tensor<192xf32>
    %adnewg2_5 = stablehlo.subtract %adsubg2_5, %adwdpg2_5 : tensor<192xf32>
    %adb1b2_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b2_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb2_5 = stablehlo.multiply %adb1b2_5, %b2_5m : tensor<192xf32>
    %admgb2_5 = stablehlo.multiply %adob1b2_5, %vitb5_2db : tensor<192xf32>
    %admnb2_5 = stablehlo.add %admsb2_5, %admgb2_5 : tensor<192xf32>
    %adb2b2_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b2_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb2_5 = stablehlo.multiply %adb2b2_5, %b2_5v : tensor<192xf32>
    %adg2b2_5 = stablehlo.multiply %vitb5_2db, %vitb5_2db : tensor<192xf32>
    %advgb2_5 = stablehlo.multiply %adob2b2_5, %adg2b2_5 : tensor<192xf32>
    %advnb2_5 = stablehlo.add %advsb2_5, %advgb2_5 : tensor<192xf32>
    %adbc1b2_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b2_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb2_5 = stablehlo.divide %admnb2_5, %adbc1b2_5 : tensor<192xf32>
    %advhb2_5 = stablehlo.divide %advnb2_5, %adbc2b2_5 : tensor<192xf32>
    %adlrb2_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb2_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb2_5 = stablehlo.sqrt %advhb2_5 : tensor<192xf32>
    %addenb2_5 = stablehlo.add %adsqb2_5, %adepsb2_5 : tensor<192xf32>
    %adratb2_5 = stablehlo.divide %admhb2_5, %addenb2_5 : tensor<192xf32>
    %adstb2_5 = stablehlo.multiply %adlrb2_5, %adratb2_5 : tensor<192xf32>
    %adsubb2_5 = stablehlo.subtract %b2_5, %adstb2_5 : tensor<192xf32>
    %adwdb2_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb2_5 = stablehlo.multiply %adwdb2_5, %adlrb2_5 : tensor<192xf32>
    %adwdpb2_5 = stablehlo.multiply %adwdlrb2_5, %b2_5 : tensor<192xf32>
    %adnewb2_5 = stablehlo.subtract %adsubb2_5, %adwdpb2_5 : tensor<192xf32>
    %adb1Wfc1_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob1Wfc1_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admsWfc1_5 = stablehlo.multiply %adb1Wfc1_5, %Wfc1_5m : tensor<192x768xf32>
    %admgWfc1_5 = stablehlo.multiply %adob1Wfc1_5, %vitb5_pdWfc1 : tensor<192x768xf32>
    %admnWfc1_5 = stablehlo.add %admsWfc1_5, %admgWfc1_5 : tensor<192x768xf32>
    %adb2Wfc1_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob2Wfc1_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %advsWfc1_5 = stablehlo.multiply %adb2Wfc1_5, %Wfc1_5v : tensor<192x768xf32>
    %adg2Wfc1_5 = stablehlo.multiply %vitb5_pdWfc1, %vitb5_pdWfc1 : tensor<192x768xf32>
    %advgWfc1_5 = stablehlo.multiply %adob2Wfc1_5, %adg2Wfc1_5 : tensor<192x768xf32>
    %advnWfc1_5 = stablehlo.add %advsWfc1_5, %advgWfc1_5 : tensor<192x768xf32>
    %adbc1Wfc1_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adbc2Wfc1_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admhWfc1_5 = stablehlo.divide %admnWfc1_5, %adbc1Wfc1_5 : tensor<192x768xf32>
    %advhWfc1_5 = stablehlo.divide %advnWfc1_5, %adbc2Wfc1_5 : tensor<192x768xf32>
    %adlrWfc1_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adepsWfc1_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adsqWfc1_5 = stablehlo.sqrt %advhWfc1_5 : tensor<192x768xf32>
    %addenWfc1_5 = stablehlo.add %adsqWfc1_5, %adepsWfc1_5 : tensor<192x768xf32>
    %adratWfc1_5 = stablehlo.divide %admhWfc1_5, %addenWfc1_5 : tensor<192x768xf32>
    %adstWfc1_5 = stablehlo.multiply %adlrWfc1_5, %adratWfc1_5 : tensor<192x768xf32>
    %adsubWfc1_5 = stablehlo.subtract %Wfc1_5, %adstWfc1_5 : tensor<192x768xf32>
    %adwdWfc1_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adwdlrWfc1_5 = stablehlo.multiply %adwdWfc1_5, %adlrWfc1_5 : tensor<192x768xf32>
    %adwdpWfc1_5 = stablehlo.multiply %adwdlrWfc1_5, %Wfc1_5 : tensor<192x768xf32>
    %adnewWfc1_5 = stablehlo.subtract %adsubWfc1_5, %adwdpWfc1_5 : tensor<192x768xf32>
    %adb1bfc1_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1bfc1_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admsbfc1_5 = stablehlo.multiply %adb1bfc1_5, %bfc1_5m : tensor<768xf32>
    %admgbfc1_5 = stablehlo.multiply %adob1bfc1_5, %vitb5_pdbfc1 : tensor<768xf32>
    %admnbfc1_5 = stablehlo.add %admsbfc1_5, %admgbfc1_5 : tensor<768xf32>
    %adb2bfc1_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2bfc1_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advsbfc1_5 = stablehlo.multiply %adb2bfc1_5, %bfc1_5v : tensor<768xf32>
    %adg2bfc1_5 = stablehlo.multiply %vitb5_pdbfc1, %vitb5_pdbfc1 : tensor<768xf32>
    %advgbfc1_5 = stablehlo.multiply %adob2bfc1_5, %adg2bfc1_5 : tensor<768xf32>
    %advnbfc1_5 = stablehlo.add %advsbfc1_5, %advgbfc1_5 : tensor<768xf32>
    %adbc1bfc1_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2bfc1_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhbfc1_5 = stablehlo.divide %admnbfc1_5, %adbc1bfc1_5 : tensor<768xf32>
    %advhbfc1_5 = stablehlo.divide %advnbfc1_5, %adbc2bfc1_5 : tensor<768xf32>
    %adlrbfc1_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepsbfc1_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqbfc1_5 = stablehlo.sqrt %advhbfc1_5 : tensor<768xf32>
    %addenbfc1_5 = stablehlo.add %adsqbfc1_5, %adepsbfc1_5 : tensor<768xf32>
    %adratbfc1_5 = stablehlo.divide %admhbfc1_5, %addenbfc1_5 : tensor<768xf32>
    %adstbfc1_5 = stablehlo.multiply %adlrbfc1_5, %adratbfc1_5 : tensor<768xf32>
    %adsubbfc1_5 = stablehlo.subtract %bfc1_5, %adstbfc1_5 : tensor<768xf32>
    %adwdbfc1_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrbfc1_5 = stablehlo.multiply %adwdbfc1_5, %adlrbfc1_5 : tensor<768xf32>
    %adwdpbfc1_5 = stablehlo.multiply %adwdlrbfc1_5, %bfc1_5 : tensor<768xf32>
    %adnewbfc1_5 = stablehlo.subtract %adsubbfc1_5, %adwdpbfc1_5 : tensor<768xf32>
    %adb1Wfc2_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob1Wfc2_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admsWfc2_5 = stablehlo.multiply %adb1Wfc2_5, %Wfc2_5m : tensor<768x192xf32>
    %admgWfc2_5 = stablehlo.multiply %adob1Wfc2_5, %vitb5_pdWfc2 : tensor<768x192xf32>
    %admnWfc2_5 = stablehlo.add %admsWfc2_5, %admgWfc2_5 : tensor<768x192xf32>
    %adb2Wfc2_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob2Wfc2_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %advsWfc2_5 = stablehlo.multiply %adb2Wfc2_5, %Wfc2_5v : tensor<768x192xf32>
    %adg2Wfc2_5 = stablehlo.multiply %vitb5_pdWfc2, %vitb5_pdWfc2 : tensor<768x192xf32>
    %advgWfc2_5 = stablehlo.multiply %adob2Wfc2_5, %adg2Wfc2_5 : tensor<768x192xf32>
    %advnWfc2_5 = stablehlo.add %advsWfc2_5, %advgWfc2_5 : tensor<768x192xf32>
    %adbc1Wfc2_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adbc2Wfc2_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admhWfc2_5 = stablehlo.divide %admnWfc2_5, %adbc1Wfc2_5 : tensor<768x192xf32>
    %advhWfc2_5 = stablehlo.divide %advnWfc2_5, %adbc2Wfc2_5 : tensor<768x192xf32>
    %adlrWfc2_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adepsWfc2_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adsqWfc2_5 = stablehlo.sqrt %advhWfc2_5 : tensor<768x192xf32>
    %addenWfc2_5 = stablehlo.add %adsqWfc2_5, %adepsWfc2_5 : tensor<768x192xf32>
    %adratWfc2_5 = stablehlo.divide %admhWfc2_5, %addenWfc2_5 : tensor<768x192xf32>
    %adstWfc2_5 = stablehlo.multiply %adlrWfc2_5, %adratWfc2_5 : tensor<768x192xf32>
    %adsubWfc2_5 = stablehlo.subtract %Wfc2_5, %adstWfc2_5 : tensor<768x192xf32>
    %adwdWfc2_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adwdlrWfc2_5 = stablehlo.multiply %adwdWfc2_5, %adlrWfc2_5 : tensor<768x192xf32>
    %adwdpWfc2_5 = stablehlo.multiply %adwdlrWfc2_5, %Wfc2_5 : tensor<768x192xf32>
    %adnewWfc2_5 = stablehlo.subtract %adsubWfc2_5, %adwdpWfc2_5 : tensor<768x192xf32>
    %adb1bfc2_5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bfc2_5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbfc2_5 = stablehlo.multiply %adb1bfc2_5, %bfc2_5m : tensor<192xf32>
    %admgbfc2_5 = stablehlo.multiply %adob1bfc2_5, %vitb5_pdbfc2 : tensor<192xf32>
    %admnbfc2_5 = stablehlo.add %admsbfc2_5, %admgbfc2_5 : tensor<192xf32>
    %adb2bfc2_5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bfc2_5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbfc2_5 = stablehlo.multiply %adb2bfc2_5, %bfc2_5v : tensor<192xf32>
    %adg2bfc2_5 = stablehlo.multiply %vitb5_pdbfc2, %vitb5_pdbfc2 : tensor<192xf32>
    %advgbfc2_5 = stablehlo.multiply %adob2bfc2_5, %adg2bfc2_5 : tensor<192xf32>
    %advnbfc2_5 = stablehlo.add %advsbfc2_5, %advgbfc2_5 : tensor<192xf32>
    %adbc1bfc2_5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bfc2_5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbfc2_5 = stablehlo.divide %admnbfc2_5, %adbc1bfc2_5 : tensor<192xf32>
    %advhbfc2_5 = stablehlo.divide %advnbfc2_5, %adbc2bfc2_5 : tensor<192xf32>
    %adlrbfc2_5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbfc2_5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbfc2_5 = stablehlo.sqrt %advhbfc2_5 : tensor<192xf32>
    %addenbfc2_5 = stablehlo.add %adsqbfc2_5, %adepsbfc2_5 : tensor<192xf32>
    %adratbfc2_5 = stablehlo.divide %admhbfc2_5, %addenbfc2_5 : tensor<192xf32>
    %adstbfc2_5 = stablehlo.multiply %adlrbfc2_5, %adratbfc2_5 : tensor<192xf32>
    %adsubbfc2_5 = stablehlo.subtract %bfc2_5, %adstbfc2_5 : tensor<192xf32>
    %adwdbfc2_5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbfc2_5 = stablehlo.multiply %adwdbfc2_5, %adlrbfc2_5 : tensor<192xf32>
    %adwdpbfc2_5 = stablehlo.multiply %adwdlrbfc2_5, %bfc2_5 : tensor<192xf32>
    %adnewbfc2_5 = stablehlo.subtract %adsubbfc2_5, %adwdpbfc2_5 : tensor<192xf32>
    %adb1g1_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g1_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg1_6 = stablehlo.multiply %adb1g1_6, %g1_6m : tensor<192xf32>
    %admgg1_6 = stablehlo.multiply %adob1g1_6, %vitb6_1dg : tensor<192xf32>
    %admng1_6 = stablehlo.add %admsg1_6, %admgg1_6 : tensor<192xf32>
    %adb2g1_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g1_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg1_6 = stablehlo.multiply %adb2g1_6, %g1_6v : tensor<192xf32>
    %adg2g1_6 = stablehlo.multiply %vitb6_1dg, %vitb6_1dg : tensor<192xf32>
    %advgg1_6 = stablehlo.multiply %adob2g1_6, %adg2g1_6 : tensor<192xf32>
    %advng1_6 = stablehlo.add %advsg1_6, %advgg1_6 : tensor<192xf32>
    %adbc1g1_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g1_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg1_6 = stablehlo.divide %admng1_6, %adbc1g1_6 : tensor<192xf32>
    %advhg1_6 = stablehlo.divide %advng1_6, %adbc2g1_6 : tensor<192xf32>
    %adlrg1_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg1_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg1_6 = stablehlo.sqrt %advhg1_6 : tensor<192xf32>
    %addeng1_6 = stablehlo.add %adsqg1_6, %adepsg1_6 : tensor<192xf32>
    %adratg1_6 = stablehlo.divide %admhg1_6, %addeng1_6 : tensor<192xf32>
    %adstg1_6 = stablehlo.multiply %adlrg1_6, %adratg1_6 : tensor<192xf32>
    %adsubg1_6 = stablehlo.subtract %g1_6, %adstg1_6 : tensor<192xf32>
    %adwdg1_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg1_6 = stablehlo.multiply %adwdg1_6, %adlrg1_6 : tensor<192xf32>
    %adwdpg1_6 = stablehlo.multiply %adwdlrg1_6, %g1_6 : tensor<192xf32>
    %adnewg1_6 = stablehlo.subtract %adsubg1_6, %adwdpg1_6 : tensor<192xf32>
    %adb1b1_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b1_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb1_6 = stablehlo.multiply %adb1b1_6, %b1_6m : tensor<192xf32>
    %admgb1_6 = stablehlo.multiply %adob1b1_6, %vitb6_1db : tensor<192xf32>
    %admnb1_6 = stablehlo.add %admsb1_6, %admgb1_6 : tensor<192xf32>
    %adb2b1_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b1_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb1_6 = stablehlo.multiply %adb2b1_6, %b1_6v : tensor<192xf32>
    %adg2b1_6 = stablehlo.multiply %vitb6_1db, %vitb6_1db : tensor<192xf32>
    %advgb1_6 = stablehlo.multiply %adob2b1_6, %adg2b1_6 : tensor<192xf32>
    %advnb1_6 = stablehlo.add %advsb1_6, %advgb1_6 : tensor<192xf32>
    %adbc1b1_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b1_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb1_6 = stablehlo.divide %admnb1_6, %adbc1b1_6 : tensor<192xf32>
    %advhb1_6 = stablehlo.divide %advnb1_6, %adbc2b1_6 : tensor<192xf32>
    %adlrb1_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb1_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb1_6 = stablehlo.sqrt %advhb1_6 : tensor<192xf32>
    %addenb1_6 = stablehlo.add %adsqb1_6, %adepsb1_6 : tensor<192xf32>
    %adratb1_6 = stablehlo.divide %admhb1_6, %addenb1_6 : tensor<192xf32>
    %adstb1_6 = stablehlo.multiply %adlrb1_6, %adratb1_6 : tensor<192xf32>
    %adsubb1_6 = stablehlo.subtract %b1_6, %adstb1_6 : tensor<192xf32>
    %adwdb1_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb1_6 = stablehlo.multiply %adwdb1_6, %adlrb1_6 : tensor<192xf32>
    %adwdpb1_6 = stablehlo.multiply %adwdlrb1_6, %b1_6 : tensor<192xf32>
    %adnewb1_6 = stablehlo.subtract %adsubb1_6, %adwdpb1_6 : tensor<192xf32>
    %adb1Wq_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wq_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWq_6 = stablehlo.multiply %adb1Wq_6, %Wq_6m : tensor<192x192xf32>
    %admgWq_6 = stablehlo.multiply %adob1Wq_6, %vitb6_mdWQ : tensor<192x192xf32>
    %admnWq_6 = stablehlo.add %admsWq_6, %admgWq_6 : tensor<192x192xf32>
    %adb2Wq_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wq_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWq_6 = stablehlo.multiply %adb2Wq_6, %Wq_6v : tensor<192x192xf32>
    %adg2Wq_6 = stablehlo.multiply %vitb6_mdWQ, %vitb6_mdWQ : tensor<192x192xf32>
    %advgWq_6 = stablehlo.multiply %adob2Wq_6, %adg2Wq_6 : tensor<192x192xf32>
    %advnWq_6 = stablehlo.add %advsWq_6, %advgWq_6 : tensor<192x192xf32>
    %adbc1Wq_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wq_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWq_6 = stablehlo.divide %admnWq_6, %adbc1Wq_6 : tensor<192x192xf32>
    %advhWq_6 = stablehlo.divide %advnWq_6, %adbc2Wq_6 : tensor<192x192xf32>
    %adlrWq_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWq_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWq_6 = stablehlo.sqrt %advhWq_6 : tensor<192x192xf32>
    %addenWq_6 = stablehlo.add %adsqWq_6, %adepsWq_6 : tensor<192x192xf32>
    %adratWq_6 = stablehlo.divide %admhWq_6, %addenWq_6 : tensor<192x192xf32>
    %adstWq_6 = stablehlo.multiply %adlrWq_6, %adratWq_6 : tensor<192x192xf32>
    %adsubWq_6 = stablehlo.subtract %Wq_6, %adstWq_6 : tensor<192x192xf32>
    %adwdWq_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWq_6 = stablehlo.multiply %adwdWq_6, %adlrWq_6 : tensor<192x192xf32>
    %adwdpWq_6 = stablehlo.multiply %adwdlrWq_6, %Wq_6 : tensor<192x192xf32>
    %adnewWq_6 = stablehlo.subtract %adsubWq_6, %adwdpWq_6 : tensor<192x192xf32>
    %adb1bq_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bq_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbq_6 = stablehlo.multiply %adb1bq_6, %bq_6m : tensor<192xf32>
    %admgbq_6 = stablehlo.multiply %adob1bq_6, %vitb6_mdbQ : tensor<192xf32>
    %admnbq_6 = stablehlo.add %admsbq_6, %admgbq_6 : tensor<192xf32>
    %adb2bq_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bq_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbq_6 = stablehlo.multiply %adb2bq_6, %bq_6v : tensor<192xf32>
    %adg2bq_6 = stablehlo.multiply %vitb6_mdbQ, %vitb6_mdbQ : tensor<192xf32>
    %advgbq_6 = stablehlo.multiply %adob2bq_6, %adg2bq_6 : tensor<192xf32>
    %advnbq_6 = stablehlo.add %advsbq_6, %advgbq_6 : tensor<192xf32>
    %adbc1bq_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bq_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbq_6 = stablehlo.divide %admnbq_6, %adbc1bq_6 : tensor<192xf32>
    %advhbq_6 = stablehlo.divide %advnbq_6, %adbc2bq_6 : tensor<192xf32>
    %adlrbq_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbq_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbq_6 = stablehlo.sqrt %advhbq_6 : tensor<192xf32>
    %addenbq_6 = stablehlo.add %adsqbq_6, %adepsbq_6 : tensor<192xf32>
    %adratbq_6 = stablehlo.divide %admhbq_6, %addenbq_6 : tensor<192xf32>
    %adstbq_6 = stablehlo.multiply %adlrbq_6, %adratbq_6 : tensor<192xf32>
    %adsubbq_6 = stablehlo.subtract %bq_6, %adstbq_6 : tensor<192xf32>
    %adwdbq_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbq_6 = stablehlo.multiply %adwdbq_6, %adlrbq_6 : tensor<192xf32>
    %adwdpbq_6 = stablehlo.multiply %adwdlrbq_6, %bq_6 : tensor<192xf32>
    %adnewbq_6 = stablehlo.subtract %adsubbq_6, %adwdpbq_6 : tensor<192xf32>
    %adb1Wk_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wk_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWk_6 = stablehlo.multiply %adb1Wk_6, %Wk_6m : tensor<192x192xf32>
    %admgWk_6 = stablehlo.multiply %adob1Wk_6, %vitb6_mdWK : tensor<192x192xf32>
    %admnWk_6 = stablehlo.add %admsWk_6, %admgWk_6 : tensor<192x192xf32>
    %adb2Wk_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wk_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWk_6 = stablehlo.multiply %adb2Wk_6, %Wk_6v : tensor<192x192xf32>
    %adg2Wk_6 = stablehlo.multiply %vitb6_mdWK, %vitb6_mdWK : tensor<192x192xf32>
    %advgWk_6 = stablehlo.multiply %adob2Wk_6, %adg2Wk_6 : tensor<192x192xf32>
    %advnWk_6 = stablehlo.add %advsWk_6, %advgWk_6 : tensor<192x192xf32>
    %adbc1Wk_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wk_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWk_6 = stablehlo.divide %admnWk_6, %adbc1Wk_6 : tensor<192x192xf32>
    %advhWk_6 = stablehlo.divide %advnWk_6, %adbc2Wk_6 : tensor<192x192xf32>
    %adlrWk_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWk_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWk_6 = stablehlo.sqrt %advhWk_6 : tensor<192x192xf32>
    %addenWk_6 = stablehlo.add %adsqWk_6, %adepsWk_6 : tensor<192x192xf32>
    %adratWk_6 = stablehlo.divide %admhWk_6, %addenWk_6 : tensor<192x192xf32>
    %adstWk_6 = stablehlo.multiply %adlrWk_6, %adratWk_6 : tensor<192x192xf32>
    %adsubWk_6 = stablehlo.subtract %Wk_6, %adstWk_6 : tensor<192x192xf32>
    %adwdWk_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWk_6 = stablehlo.multiply %adwdWk_6, %adlrWk_6 : tensor<192x192xf32>
    %adwdpWk_6 = stablehlo.multiply %adwdlrWk_6, %Wk_6 : tensor<192x192xf32>
    %adnewWk_6 = stablehlo.subtract %adsubWk_6, %adwdpWk_6 : tensor<192x192xf32>
    %adb1bk_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bk_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbk_6 = stablehlo.multiply %adb1bk_6, %bk_6m : tensor<192xf32>
    %admgbk_6 = stablehlo.multiply %adob1bk_6, %vitb6_mdbK : tensor<192xf32>
    %admnbk_6 = stablehlo.add %admsbk_6, %admgbk_6 : tensor<192xf32>
    %adb2bk_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bk_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbk_6 = stablehlo.multiply %adb2bk_6, %bk_6v : tensor<192xf32>
    %adg2bk_6 = stablehlo.multiply %vitb6_mdbK, %vitb6_mdbK : tensor<192xf32>
    %advgbk_6 = stablehlo.multiply %adob2bk_6, %adg2bk_6 : tensor<192xf32>
    %advnbk_6 = stablehlo.add %advsbk_6, %advgbk_6 : tensor<192xf32>
    %adbc1bk_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bk_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbk_6 = stablehlo.divide %admnbk_6, %adbc1bk_6 : tensor<192xf32>
    %advhbk_6 = stablehlo.divide %advnbk_6, %adbc2bk_6 : tensor<192xf32>
    %adlrbk_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbk_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbk_6 = stablehlo.sqrt %advhbk_6 : tensor<192xf32>
    %addenbk_6 = stablehlo.add %adsqbk_6, %adepsbk_6 : tensor<192xf32>
    %adratbk_6 = stablehlo.divide %admhbk_6, %addenbk_6 : tensor<192xf32>
    %adstbk_6 = stablehlo.multiply %adlrbk_6, %adratbk_6 : tensor<192xf32>
    %adsubbk_6 = stablehlo.subtract %bk_6, %adstbk_6 : tensor<192xf32>
    %adwdbk_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbk_6 = stablehlo.multiply %adwdbk_6, %adlrbk_6 : tensor<192xf32>
    %adwdpbk_6 = stablehlo.multiply %adwdlrbk_6, %bk_6 : tensor<192xf32>
    %adnewbk_6 = stablehlo.subtract %adsubbk_6, %adwdpbk_6 : tensor<192xf32>
    %adb1Wv_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wv_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWv_6 = stablehlo.multiply %adb1Wv_6, %Wv_6m : tensor<192x192xf32>
    %admgWv_6 = stablehlo.multiply %adob1Wv_6, %vitb6_mdWV : tensor<192x192xf32>
    %admnWv_6 = stablehlo.add %admsWv_6, %admgWv_6 : tensor<192x192xf32>
    %adb2Wv_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wv_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWv_6 = stablehlo.multiply %adb2Wv_6, %Wv_6v : tensor<192x192xf32>
    %adg2Wv_6 = stablehlo.multiply %vitb6_mdWV, %vitb6_mdWV : tensor<192x192xf32>
    %advgWv_6 = stablehlo.multiply %adob2Wv_6, %adg2Wv_6 : tensor<192x192xf32>
    %advnWv_6 = stablehlo.add %advsWv_6, %advgWv_6 : tensor<192x192xf32>
    %adbc1Wv_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wv_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWv_6 = stablehlo.divide %admnWv_6, %adbc1Wv_6 : tensor<192x192xf32>
    %advhWv_6 = stablehlo.divide %advnWv_6, %adbc2Wv_6 : tensor<192x192xf32>
    %adlrWv_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWv_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWv_6 = stablehlo.sqrt %advhWv_6 : tensor<192x192xf32>
    %addenWv_6 = stablehlo.add %adsqWv_6, %adepsWv_6 : tensor<192x192xf32>
    %adratWv_6 = stablehlo.divide %admhWv_6, %addenWv_6 : tensor<192x192xf32>
    %adstWv_6 = stablehlo.multiply %adlrWv_6, %adratWv_6 : tensor<192x192xf32>
    %adsubWv_6 = stablehlo.subtract %Wv_6, %adstWv_6 : tensor<192x192xf32>
    %adwdWv_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWv_6 = stablehlo.multiply %adwdWv_6, %adlrWv_6 : tensor<192x192xf32>
    %adwdpWv_6 = stablehlo.multiply %adwdlrWv_6, %Wv_6 : tensor<192x192xf32>
    %adnewWv_6 = stablehlo.subtract %adsubWv_6, %adwdpWv_6 : tensor<192x192xf32>
    %adb1bv_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bv_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbv_6 = stablehlo.multiply %adb1bv_6, %bv_6m : tensor<192xf32>
    %admgbv_6 = stablehlo.multiply %adob1bv_6, %vitb6_mdbV : tensor<192xf32>
    %admnbv_6 = stablehlo.add %admsbv_6, %admgbv_6 : tensor<192xf32>
    %adb2bv_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bv_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbv_6 = stablehlo.multiply %adb2bv_6, %bv_6v : tensor<192xf32>
    %adg2bv_6 = stablehlo.multiply %vitb6_mdbV, %vitb6_mdbV : tensor<192xf32>
    %advgbv_6 = stablehlo.multiply %adob2bv_6, %adg2bv_6 : tensor<192xf32>
    %advnbv_6 = stablehlo.add %advsbv_6, %advgbv_6 : tensor<192xf32>
    %adbc1bv_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bv_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbv_6 = stablehlo.divide %admnbv_6, %adbc1bv_6 : tensor<192xf32>
    %advhbv_6 = stablehlo.divide %advnbv_6, %adbc2bv_6 : tensor<192xf32>
    %adlrbv_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbv_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbv_6 = stablehlo.sqrt %advhbv_6 : tensor<192xf32>
    %addenbv_6 = stablehlo.add %adsqbv_6, %adepsbv_6 : tensor<192xf32>
    %adratbv_6 = stablehlo.divide %admhbv_6, %addenbv_6 : tensor<192xf32>
    %adstbv_6 = stablehlo.multiply %adlrbv_6, %adratbv_6 : tensor<192xf32>
    %adsubbv_6 = stablehlo.subtract %bv_6, %adstbv_6 : tensor<192xf32>
    %adwdbv_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbv_6 = stablehlo.multiply %adwdbv_6, %adlrbv_6 : tensor<192xf32>
    %adwdpbv_6 = stablehlo.multiply %adwdlrbv_6, %bv_6 : tensor<192xf32>
    %adnewbv_6 = stablehlo.subtract %adsubbv_6, %adwdpbv_6 : tensor<192xf32>
    %adb1Wo_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wo_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWo_6 = stablehlo.multiply %adb1Wo_6, %Wo_6m : tensor<192x192xf32>
    %admgWo_6 = stablehlo.multiply %adob1Wo_6, %vitb6_mdWo : tensor<192x192xf32>
    %admnWo_6 = stablehlo.add %admsWo_6, %admgWo_6 : tensor<192x192xf32>
    %adb2Wo_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wo_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWo_6 = stablehlo.multiply %adb2Wo_6, %Wo_6v : tensor<192x192xf32>
    %adg2Wo_6 = stablehlo.multiply %vitb6_mdWo, %vitb6_mdWo : tensor<192x192xf32>
    %advgWo_6 = stablehlo.multiply %adob2Wo_6, %adg2Wo_6 : tensor<192x192xf32>
    %advnWo_6 = stablehlo.add %advsWo_6, %advgWo_6 : tensor<192x192xf32>
    %adbc1Wo_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wo_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWo_6 = stablehlo.divide %admnWo_6, %adbc1Wo_6 : tensor<192x192xf32>
    %advhWo_6 = stablehlo.divide %advnWo_6, %adbc2Wo_6 : tensor<192x192xf32>
    %adlrWo_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWo_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWo_6 = stablehlo.sqrt %advhWo_6 : tensor<192x192xf32>
    %addenWo_6 = stablehlo.add %adsqWo_6, %adepsWo_6 : tensor<192x192xf32>
    %adratWo_6 = stablehlo.divide %admhWo_6, %addenWo_6 : tensor<192x192xf32>
    %adstWo_6 = stablehlo.multiply %adlrWo_6, %adratWo_6 : tensor<192x192xf32>
    %adsubWo_6 = stablehlo.subtract %Wo_6, %adstWo_6 : tensor<192x192xf32>
    %adwdWo_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWo_6 = stablehlo.multiply %adwdWo_6, %adlrWo_6 : tensor<192x192xf32>
    %adwdpWo_6 = stablehlo.multiply %adwdlrWo_6, %Wo_6 : tensor<192x192xf32>
    %adnewWo_6 = stablehlo.subtract %adsubWo_6, %adwdpWo_6 : tensor<192x192xf32>
    %adb1bo_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bo_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbo_6 = stablehlo.multiply %adb1bo_6, %bo_6m : tensor<192xf32>
    %admgbo_6 = stablehlo.multiply %adob1bo_6, %vitb6_mdbo : tensor<192xf32>
    %admnbo_6 = stablehlo.add %admsbo_6, %admgbo_6 : tensor<192xf32>
    %adb2bo_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bo_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbo_6 = stablehlo.multiply %adb2bo_6, %bo_6v : tensor<192xf32>
    %adg2bo_6 = stablehlo.multiply %vitb6_mdbo, %vitb6_mdbo : tensor<192xf32>
    %advgbo_6 = stablehlo.multiply %adob2bo_6, %adg2bo_6 : tensor<192xf32>
    %advnbo_6 = stablehlo.add %advsbo_6, %advgbo_6 : tensor<192xf32>
    %adbc1bo_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bo_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbo_6 = stablehlo.divide %admnbo_6, %adbc1bo_6 : tensor<192xf32>
    %advhbo_6 = stablehlo.divide %advnbo_6, %adbc2bo_6 : tensor<192xf32>
    %adlrbo_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbo_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbo_6 = stablehlo.sqrt %advhbo_6 : tensor<192xf32>
    %addenbo_6 = stablehlo.add %adsqbo_6, %adepsbo_6 : tensor<192xf32>
    %adratbo_6 = stablehlo.divide %admhbo_6, %addenbo_6 : tensor<192xf32>
    %adstbo_6 = stablehlo.multiply %adlrbo_6, %adratbo_6 : tensor<192xf32>
    %adsubbo_6 = stablehlo.subtract %bo_6, %adstbo_6 : tensor<192xf32>
    %adwdbo_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbo_6 = stablehlo.multiply %adwdbo_6, %adlrbo_6 : tensor<192xf32>
    %adwdpbo_6 = stablehlo.multiply %adwdlrbo_6, %bo_6 : tensor<192xf32>
    %adnewbo_6 = stablehlo.subtract %adsubbo_6, %adwdpbo_6 : tensor<192xf32>
    %adb1g2_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g2_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg2_6 = stablehlo.multiply %adb1g2_6, %g2_6m : tensor<192xf32>
    %admgg2_6 = stablehlo.multiply %adob1g2_6, %vitb6_2dg : tensor<192xf32>
    %admng2_6 = stablehlo.add %admsg2_6, %admgg2_6 : tensor<192xf32>
    %adb2g2_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g2_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg2_6 = stablehlo.multiply %adb2g2_6, %g2_6v : tensor<192xf32>
    %adg2g2_6 = stablehlo.multiply %vitb6_2dg, %vitb6_2dg : tensor<192xf32>
    %advgg2_6 = stablehlo.multiply %adob2g2_6, %adg2g2_6 : tensor<192xf32>
    %advng2_6 = stablehlo.add %advsg2_6, %advgg2_6 : tensor<192xf32>
    %adbc1g2_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g2_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg2_6 = stablehlo.divide %admng2_6, %adbc1g2_6 : tensor<192xf32>
    %advhg2_6 = stablehlo.divide %advng2_6, %adbc2g2_6 : tensor<192xf32>
    %adlrg2_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg2_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg2_6 = stablehlo.sqrt %advhg2_6 : tensor<192xf32>
    %addeng2_6 = stablehlo.add %adsqg2_6, %adepsg2_6 : tensor<192xf32>
    %adratg2_6 = stablehlo.divide %admhg2_6, %addeng2_6 : tensor<192xf32>
    %adstg2_6 = stablehlo.multiply %adlrg2_6, %adratg2_6 : tensor<192xf32>
    %adsubg2_6 = stablehlo.subtract %g2_6, %adstg2_6 : tensor<192xf32>
    %adwdg2_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg2_6 = stablehlo.multiply %adwdg2_6, %adlrg2_6 : tensor<192xf32>
    %adwdpg2_6 = stablehlo.multiply %adwdlrg2_6, %g2_6 : tensor<192xf32>
    %adnewg2_6 = stablehlo.subtract %adsubg2_6, %adwdpg2_6 : tensor<192xf32>
    %adb1b2_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b2_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb2_6 = stablehlo.multiply %adb1b2_6, %b2_6m : tensor<192xf32>
    %admgb2_6 = stablehlo.multiply %adob1b2_6, %vitb6_2db : tensor<192xf32>
    %admnb2_6 = stablehlo.add %admsb2_6, %admgb2_6 : tensor<192xf32>
    %adb2b2_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b2_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb2_6 = stablehlo.multiply %adb2b2_6, %b2_6v : tensor<192xf32>
    %adg2b2_6 = stablehlo.multiply %vitb6_2db, %vitb6_2db : tensor<192xf32>
    %advgb2_6 = stablehlo.multiply %adob2b2_6, %adg2b2_6 : tensor<192xf32>
    %advnb2_6 = stablehlo.add %advsb2_6, %advgb2_6 : tensor<192xf32>
    %adbc1b2_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b2_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb2_6 = stablehlo.divide %admnb2_6, %adbc1b2_6 : tensor<192xf32>
    %advhb2_6 = stablehlo.divide %advnb2_6, %adbc2b2_6 : tensor<192xf32>
    %adlrb2_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb2_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb2_6 = stablehlo.sqrt %advhb2_6 : tensor<192xf32>
    %addenb2_6 = stablehlo.add %adsqb2_6, %adepsb2_6 : tensor<192xf32>
    %adratb2_6 = stablehlo.divide %admhb2_6, %addenb2_6 : tensor<192xf32>
    %adstb2_6 = stablehlo.multiply %adlrb2_6, %adratb2_6 : tensor<192xf32>
    %adsubb2_6 = stablehlo.subtract %b2_6, %adstb2_6 : tensor<192xf32>
    %adwdb2_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb2_6 = stablehlo.multiply %adwdb2_6, %adlrb2_6 : tensor<192xf32>
    %adwdpb2_6 = stablehlo.multiply %adwdlrb2_6, %b2_6 : tensor<192xf32>
    %adnewb2_6 = stablehlo.subtract %adsubb2_6, %adwdpb2_6 : tensor<192xf32>
    %adb1Wfc1_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob1Wfc1_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admsWfc1_6 = stablehlo.multiply %adb1Wfc1_6, %Wfc1_6m : tensor<192x768xf32>
    %admgWfc1_6 = stablehlo.multiply %adob1Wfc1_6, %vitb6_pdWfc1 : tensor<192x768xf32>
    %admnWfc1_6 = stablehlo.add %admsWfc1_6, %admgWfc1_6 : tensor<192x768xf32>
    %adb2Wfc1_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob2Wfc1_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %advsWfc1_6 = stablehlo.multiply %adb2Wfc1_6, %Wfc1_6v : tensor<192x768xf32>
    %adg2Wfc1_6 = stablehlo.multiply %vitb6_pdWfc1, %vitb6_pdWfc1 : tensor<192x768xf32>
    %advgWfc1_6 = stablehlo.multiply %adob2Wfc1_6, %adg2Wfc1_6 : tensor<192x768xf32>
    %advnWfc1_6 = stablehlo.add %advsWfc1_6, %advgWfc1_6 : tensor<192x768xf32>
    %adbc1Wfc1_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adbc2Wfc1_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admhWfc1_6 = stablehlo.divide %admnWfc1_6, %adbc1Wfc1_6 : tensor<192x768xf32>
    %advhWfc1_6 = stablehlo.divide %advnWfc1_6, %adbc2Wfc1_6 : tensor<192x768xf32>
    %adlrWfc1_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adepsWfc1_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adsqWfc1_6 = stablehlo.sqrt %advhWfc1_6 : tensor<192x768xf32>
    %addenWfc1_6 = stablehlo.add %adsqWfc1_6, %adepsWfc1_6 : tensor<192x768xf32>
    %adratWfc1_6 = stablehlo.divide %admhWfc1_6, %addenWfc1_6 : tensor<192x768xf32>
    %adstWfc1_6 = stablehlo.multiply %adlrWfc1_6, %adratWfc1_6 : tensor<192x768xf32>
    %adsubWfc1_6 = stablehlo.subtract %Wfc1_6, %adstWfc1_6 : tensor<192x768xf32>
    %adwdWfc1_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adwdlrWfc1_6 = stablehlo.multiply %adwdWfc1_6, %adlrWfc1_6 : tensor<192x768xf32>
    %adwdpWfc1_6 = stablehlo.multiply %adwdlrWfc1_6, %Wfc1_6 : tensor<192x768xf32>
    %adnewWfc1_6 = stablehlo.subtract %adsubWfc1_6, %adwdpWfc1_6 : tensor<192x768xf32>
    %adb1bfc1_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1bfc1_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admsbfc1_6 = stablehlo.multiply %adb1bfc1_6, %bfc1_6m : tensor<768xf32>
    %admgbfc1_6 = stablehlo.multiply %adob1bfc1_6, %vitb6_pdbfc1 : tensor<768xf32>
    %admnbfc1_6 = stablehlo.add %admsbfc1_6, %admgbfc1_6 : tensor<768xf32>
    %adb2bfc1_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2bfc1_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advsbfc1_6 = stablehlo.multiply %adb2bfc1_6, %bfc1_6v : tensor<768xf32>
    %adg2bfc1_6 = stablehlo.multiply %vitb6_pdbfc1, %vitb6_pdbfc1 : tensor<768xf32>
    %advgbfc1_6 = stablehlo.multiply %adob2bfc1_6, %adg2bfc1_6 : tensor<768xf32>
    %advnbfc1_6 = stablehlo.add %advsbfc1_6, %advgbfc1_6 : tensor<768xf32>
    %adbc1bfc1_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2bfc1_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhbfc1_6 = stablehlo.divide %admnbfc1_6, %adbc1bfc1_6 : tensor<768xf32>
    %advhbfc1_6 = stablehlo.divide %advnbfc1_6, %adbc2bfc1_6 : tensor<768xf32>
    %adlrbfc1_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepsbfc1_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqbfc1_6 = stablehlo.sqrt %advhbfc1_6 : tensor<768xf32>
    %addenbfc1_6 = stablehlo.add %adsqbfc1_6, %adepsbfc1_6 : tensor<768xf32>
    %adratbfc1_6 = stablehlo.divide %admhbfc1_6, %addenbfc1_6 : tensor<768xf32>
    %adstbfc1_6 = stablehlo.multiply %adlrbfc1_6, %adratbfc1_6 : tensor<768xf32>
    %adsubbfc1_6 = stablehlo.subtract %bfc1_6, %adstbfc1_6 : tensor<768xf32>
    %adwdbfc1_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrbfc1_6 = stablehlo.multiply %adwdbfc1_6, %adlrbfc1_6 : tensor<768xf32>
    %adwdpbfc1_6 = stablehlo.multiply %adwdlrbfc1_6, %bfc1_6 : tensor<768xf32>
    %adnewbfc1_6 = stablehlo.subtract %adsubbfc1_6, %adwdpbfc1_6 : tensor<768xf32>
    %adb1Wfc2_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob1Wfc2_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admsWfc2_6 = stablehlo.multiply %adb1Wfc2_6, %Wfc2_6m : tensor<768x192xf32>
    %admgWfc2_6 = stablehlo.multiply %adob1Wfc2_6, %vitb6_pdWfc2 : tensor<768x192xf32>
    %admnWfc2_6 = stablehlo.add %admsWfc2_6, %admgWfc2_6 : tensor<768x192xf32>
    %adb2Wfc2_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob2Wfc2_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %advsWfc2_6 = stablehlo.multiply %adb2Wfc2_6, %Wfc2_6v : tensor<768x192xf32>
    %adg2Wfc2_6 = stablehlo.multiply %vitb6_pdWfc2, %vitb6_pdWfc2 : tensor<768x192xf32>
    %advgWfc2_6 = stablehlo.multiply %adob2Wfc2_6, %adg2Wfc2_6 : tensor<768x192xf32>
    %advnWfc2_6 = stablehlo.add %advsWfc2_6, %advgWfc2_6 : tensor<768x192xf32>
    %adbc1Wfc2_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adbc2Wfc2_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admhWfc2_6 = stablehlo.divide %admnWfc2_6, %adbc1Wfc2_6 : tensor<768x192xf32>
    %advhWfc2_6 = stablehlo.divide %advnWfc2_6, %adbc2Wfc2_6 : tensor<768x192xf32>
    %adlrWfc2_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adepsWfc2_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adsqWfc2_6 = stablehlo.sqrt %advhWfc2_6 : tensor<768x192xf32>
    %addenWfc2_6 = stablehlo.add %adsqWfc2_6, %adepsWfc2_6 : tensor<768x192xf32>
    %adratWfc2_6 = stablehlo.divide %admhWfc2_6, %addenWfc2_6 : tensor<768x192xf32>
    %adstWfc2_6 = stablehlo.multiply %adlrWfc2_6, %adratWfc2_6 : tensor<768x192xf32>
    %adsubWfc2_6 = stablehlo.subtract %Wfc2_6, %adstWfc2_6 : tensor<768x192xf32>
    %adwdWfc2_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adwdlrWfc2_6 = stablehlo.multiply %adwdWfc2_6, %adlrWfc2_6 : tensor<768x192xf32>
    %adwdpWfc2_6 = stablehlo.multiply %adwdlrWfc2_6, %Wfc2_6 : tensor<768x192xf32>
    %adnewWfc2_6 = stablehlo.subtract %adsubWfc2_6, %adwdpWfc2_6 : tensor<768x192xf32>
    %adb1bfc2_6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bfc2_6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbfc2_6 = stablehlo.multiply %adb1bfc2_6, %bfc2_6m : tensor<192xf32>
    %admgbfc2_6 = stablehlo.multiply %adob1bfc2_6, %vitb6_pdbfc2 : tensor<192xf32>
    %admnbfc2_6 = stablehlo.add %admsbfc2_6, %admgbfc2_6 : tensor<192xf32>
    %adb2bfc2_6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bfc2_6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbfc2_6 = stablehlo.multiply %adb2bfc2_6, %bfc2_6v : tensor<192xf32>
    %adg2bfc2_6 = stablehlo.multiply %vitb6_pdbfc2, %vitb6_pdbfc2 : tensor<192xf32>
    %advgbfc2_6 = stablehlo.multiply %adob2bfc2_6, %adg2bfc2_6 : tensor<192xf32>
    %advnbfc2_6 = stablehlo.add %advsbfc2_6, %advgbfc2_6 : tensor<192xf32>
    %adbc1bfc2_6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bfc2_6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbfc2_6 = stablehlo.divide %admnbfc2_6, %adbc1bfc2_6 : tensor<192xf32>
    %advhbfc2_6 = stablehlo.divide %advnbfc2_6, %adbc2bfc2_6 : tensor<192xf32>
    %adlrbfc2_6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbfc2_6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbfc2_6 = stablehlo.sqrt %advhbfc2_6 : tensor<192xf32>
    %addenbfc2_6 = stablehlo.add %adsqbfc2_6, %adepsbfc2_6 : tensor<192xf32>
    %adratbfc2_6 = stablehlo.divide %admhbfc2_6, %addenbfc2_6 : tensor<192xf32>
    %adstbfc2_6 = stablehlo.multiply %adlrbfc2_6, %adratbfc2_6 : tensor<192xf32>
    %adsubbfc2_6 = stablehlo.subtract %bfc2_6, %adstbfc2_6 : tensor<192xf32>
    %adwdbfc2_6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbfc2_6 = stablehlo.multiply %adwdbfc2_6, %adlrbfc2_6 : tensor<192xf32>
    %adwdpbfc2_6 = stablehlo.multiply %adwdlrbfc2_6, %bfc2_6 : tensor<192xf32>
    %adnewbfc2_6 = stablehlo.subtract %adsubbfc2_6, %adwdpbfc2_6 : tensor<192xf32>
    %adb1g1_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g1_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg1_7 = stablehlo.multiply %adb1g1_7, %g1_7m : tensor<192xf32>
    %admgg1_7 = stablehlo.multiply %adob1g1_7, %vitb7_1dg : tensor<192xf32>
    %admng1_7 = stablehlo.add %admsg1_7, %admgg1_7 : tensor<192xf32>
    %adb2g1_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g1_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg1_7 = stablehlo.multiply %adb2g1_7, %g1_7v : tensor<192xf32>
    %adg2g1_7 = stablehlo.multiply %vitb7_1dg, %vitb7_1dg : tensor<192xf32>
    %advgg1_7 = stablehlo.multiply %adob2g1_7, %adg2g1_7 : tensor<192xf32>
    %advng1_7 = stablehlo.add %advsg1_7, %advgg1_7 : tensor<192xf32>
    %adbc1g1_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g1_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg1_7 = stablehlo.divide %admng1_7, %adbc1g1_7 : tensor<192xf32>
    %advhg1_7 = stablehlo.divide %advng1_7, %adbc2g1_7 : tensor<192xf32>
    %adlrg1_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg1_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg1_7 = stablehlo.sqrt %advhg1_7 : tensor<192xf32>
    %addeng1_7 = stablehlo.add %adsqg1_7, %adepsg1_7 : tensor<192xf32>
    %adratg1_7 = stablehlo.divide %admhg1_7, %addeng1_7 : tensor<192xf32>
    %adstg1_7 = stablehlo.multiply %adlrg1_7, %adratg1_7 : tensor<192xf32>
    %adsubg1_7 = stablehlo.subtract %g1_7, %adstg1_7 : tensor<192xf32>
    %adwdg1_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg1_7 = stablehlo.multiply %adwdg1_7, %adlrg1_7 : tensor<192xf32>
    %adwdpg1_7 = stablehlo.multiply %adwdlrg1_7, %g1_7 : tensor<192xf32>
    %adnewg1_7 = stablehlo.subtract %adsubg1_7, %adwdpg1_7 : tensor<192xf32>
    %adb1b1_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b1_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb1_7 = stablehlo.multiply %adb1b1_7, %b1_7m : tensor<192xf32>
    %admgb1_7 = stablehlo.multiply %adob1b1_7, %vitb7_1db : tensor<192xf32>
    %admnb1_7 = stablehlo.add %admsb1_7, %admgb1_7 : tensor<192xf32>
    %adb2b1_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b1_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb1_7 = stablehlo.multiply %adb2b1_7, %b1_7v : tensor<192xf32>
    %adg2b1_7 = stablehlo.multiply %vitb7_1db, %vitb7_1db : tensor<192xf32>
    %advgb1_7 = stablehlo.multiply %adob2b1_7, %adg2b1_7 : tensor<192xf32>
    %advnb1_7 = stablehlo.add %advsb1_7, %advgb1_7 : tensor<192xf32>
    %adbc1b1_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b1_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb1_7 = stablehlo.divide %admnb1_7, %adbc1b1_7 : tensor<192xf32>
    %advhb1_7 = stablehlo.divide %advnb1_7, %adbc2b1_7 : tensor<192xf32>
    %adlrb1_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb1_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb1_7 = stablehlo.sqrt %advhb1_7 : tensor<192xf32>
    %addenb1_7 = stablehlo.add %adsqb1_7, %adepsb1_7 : tensor<192xf32>
    %adratb1_7 = stablehlo.divide %admhb1_7, %addenb1_7 : tensor<192xf32>
    %adstb1_7 = stablehlo.multiply %adlrb1_7, %adratb1_7 : tensor<192xf32>
    %adsubb1_7 = stablehlo.subtract %b1_7, %adstb1_7 : tensor<192xf32>
    %adwdb1_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb1_7 = stablehlo.multiply %adwdb1_7, %adlrb1_7 : tensor<192xf32>
    %adwdpb1_7 = stablehlo.multiply %adwdlrb1_7, %b1_7 : tensor<192xf32>
    %adnewb1_7 = stablehlo.subtract %adsubb1_7, %adwdpb1_7 : tensor<192xf32>
    %adb1Wq_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wq_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWq_7 = stablehlo.multiply %adb1Wq_7, %Wq_7m : tensor<192x192xf32>
    %admgWq_7 = stablehlo.multiply %adob1Wq_7, %vitb7_mdWQ : tensor<192x192xf32>
    %admnWq_7 = stablehlo.add %admsWq_7, %admgWq_7 : tensor<192x192xf32>
    %adb2Wq_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wq_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWq_7 = stablehlo.multiply %adb2Wq_7, %Wq_7v : tensor<192x192xf32>
    %adg2Wq_7 = stablehlo.multiply %vitb7_mdWQ, %vitb7_mdWQ : tensor<192x192xf32>
    %advgWq_7 = stablehlo.multiply %adob2Wq_7, %adg2Wq_7 : tensor<192x192xf32>
    %advnWq_7 = stablehlo.add %advsWq_7, %advgWq_7 : tensor<192x192xf32>
    %adbc1Wq_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wq_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWq_7 = stablehlo.divide %admnWq_7, %adbc1Wq_7 : tensor<192x192xf32>
    %advhWq_7 = stablehlo.divide %advnWq_7, %adbc2Wq_7 : tensor<192x192xf32>
    %adlrWq_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWq_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWq_7 = stablehlo.sqrt %advhWq_7 : tensor<192x192xf32>
    %addenWq_7 = stablehlo.add %adsqWq_7, %adepsWq_7 : tensor<192x192xf32>
    %adratWq_7 = stablehlo.divide %admhWq_7, %addenWq_7 : tensor<192x192xf32>
    %adstWq_7 = stablehlo.multiply %adlrWq_7, %adratWq_7 : tensor<192x192xf32>
    %adsubWq_7 = stablehlo.subtract %Wq_7, %adstWq_7 : tensor<192x192xf32>
    %adwdWq_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWq_7 = stablehlo.multiply %adwdWq_7, %adlrWq_7 : tensor<192x192xf32>
    %adwdpWq_7 = stablehlo.multiply %adwdlrWq_7, %Wq_7 : tensor<192x192xf32>
    %adnewWq_7 = stablehlo.subtract %adsubWq_7, %adwdpWq_7 : tensor<192x192xf32>
    %adb1bq_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bq_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbq_7 = stablehlo.multiply %adb1bq_7, %bq_7m : tensor<192xf32>
    %admgbq_7 = stablehlo.multiply %adob1bq_7, %vitb7_mdbQ : tensor<192xf32>
    %admnbq_7 = stablehlo.add %admsbq_7, %admgbq_7 : tensor<192xf32>
    %adb2bq_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bq_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbq_7 = stablehlo.multiply %adb2bq_7, %bq_7v : tensor<192xf32>
    %adg2bq_7 = stablehlo.multiply %vitb7_mdbQ, %vitb7_mdbQ : tensor<192xf32>
    %advgbq_7 = stablehlo.multiply %adob2bq_7, %adg2bq_7 : tensor<192xf32>
    %advnbq_7 = stablehlo.add %advsbq_7, %advgbq_7 : tensor<192xf32>
    %adbc1bq_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bq_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbq_7 = stablehlo.divide %admnbq_7, %adbc1bq_7 : tensor<192xf32>
    %advhbq_7 = stablehlo.divide %advnbq_7, %adbc2bq_7 : tensor<192xf32>
    %adlrbq_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbq_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbq_7 = stablehlo.sqrt %advhbq_7 : tensor<192xf32>
    %addenbq_7 = stablehlo.add %adsqbq_7, %adepsbq_7 : tensor<192xf32>
    %adratbq_7 = stablehlo.divide %admhbq_7, %addenbq_7 : tensor<192xf32>
    %adstbq_7 = stablehlo.multiply %adlrbq_7, %adratbq_7 : tensor<192xf32>
    %adsubbq_7 = stablehlo.subtract %bq_7, %adstbq_7 : tensor<192xf32>
    %adwdbq_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbq_7 = stablehlo.multiply %adwdbq_7, %adlrbq_7 : tensor<192xf32>
    %adwdpbq_7 = stablehlo.multiply %adwdlrbq_7, %bq_7 : tensor<192xf32>
    %adnewbq_7 = stablehlo.subtract %adsubbq_7, %adwdpbq_7 : tensor<192xf32>
    %adb1Wk_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wk_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWk_7 = stablehlo.multiply %adb1Wk_7, %Wk_7m : tensor<192x192xf32>
    %admgWk_7 = stablehlo.multiply %adob1Wk_7, %vitb7_mdWK : tensor<192x192xf32>
    %admnWk_7 = stablehlo.add %admsWk_7, %admgWk_7 : tensor<192x192xf32>
    %adb2Wk_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wk_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWk_7 = stablehlo.multiply %adb2Wk_7, %Wk_7v : tensor<192x192xf32>
    %adg2Wk_7 = stablehlo.multiply %vitb7_mdWK, %vitb7_mdWK : tensor<192x192xf32>
    %advgWk_7 = stablehlo.multiply %adob2Wk_7, %adg2Wk_7 : tensor<192x192xf32>
    %advnWk_7 = stablehlo.add %advsWk_7, %advgWk_7 : tensor<192x192xf32>
    %adbc1Wk_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wk_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWk_7 = stablehlo.divide %admnWk_7, %adbc1Wk_7 : tensor<192x192xf32>
    %advhWk_7 = stablehlo.divide %advnWk_7, %adbc2Wk_7 : tensor<192x192xf32>
    %adlrWk_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWk_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWk_7 = stablehlo.sqrt %advhWk_7 : tensor<192x192xf32>
    %addenWk_7 = stablehlo.add %adsqWk_7, %adepsWk_7 : tensor<192x192xf32>
    %adratWk_7 = stablehlo.divide %admhWk_7, %addenWk_7 : tensor<192x192xf32>
    %adstWk_7 = stablehlo.multiply %adlrWk_7, %adratWk_7 : tensor<192x192xf32>
    %adsubWk_7 = stablehlo.subtract %Wk_7, %adstWk_7 : tensor<192x192xf32>
    %adwdWk_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWk_7 = stablehlo.multiply %adwdWk_7, %adlrWk_7 : tensor<192x192xf32>
    %adwdpWk_7 = stablehlo.multiply %adwdlrWk_7, %Wk_7 : tensor<192x192xf32>
    %adnewWk_7 = stablehlo.subtract %adsubWk_7, %adwdpWk_7 : tensor<192x192xf32>
    %adb1bk_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bk_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbk_7 = stablehlo.multiply %adb1bk_7, %bk_7m : tensor<192xf32>
    %admgbk_7 = stablehlo.multiply %adob1bk_7, %vitb7_mdbK : tensor<192xf32>
    %admnbk_7 = stablehlo.add %admsbk_7, %admgbk_7 : tensor<192xf32>
    %adb2bk_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bk_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbk_7 = stablehlo.multiply %adb2bk_7, %bk_7v : tensor<192xf32>
    %adg2bk_7 = stablehlo.multiply %vitb7_mdbK, %vitb7_mdbK : tensor<192xf32>
    %advgbk_7 = stablehlo.multiply %adob2bk_7, %adg2bk_7 : tensor<192xf32>
    %advnbk_7 = stablehlo.add %advsbk_7, %advgbk_7 : tensor<192xf32>
    %adbc1bk_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bk_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbk_7 = stablehlo.divide %admnbk_7, %adbc1bk_7 : tensor<192xf32>
    %advhbk_7 = stablehlo.divide %advnbk_7, %adbc2bk_7 : tensor<192xf32>
    %adlrbk_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbk_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbk_7 = stablehlo.sqrt %advhbk_7 : tensor<192xf32>
    %addenbk_7 = stablehlo.add %adsqbk_7, %adepsbk_7 : tensor<192xf32>
    %adratbk_7 = stablehlo.divide %admhbk_7, %addenbk_7 : tensor<192xf32>
    %adstbk_7 = stablehlo.multiply %adlrbk_7, %adratbk_7 : tensor<192xf32>
    %adsubbk_7 = stablehlo.subtract %bk_7, %adstbk_7 : tensor<192xf32>
    %adwdbk_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbk_7 = stablehlo.multiply %adwdbk_7, %adlrbk_7 : tensor<192xf32>
    %adwdpbk_7 = stablehlo.multiply %adwdlrbk_7, %bk_7 : tensor<192xf32>
    %adnewbk_7 = stablehlo.subtract %adsubbk_7, %adwdpbk_7 : tensor<192xf32>
    %adb1Wv_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wv_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWv_7 = stablehlo.multiply %adb1Wv_7, %Wv_7m : tensor<192x192xf32>
    %admgWv_7 = stablehlo.multiply %adob1Wv_7, %vitb7_mdWV : tensor<192x192xf32>
    %admnWv_7 = stablehlo.add %admsWv_7, %admgWv_7 : tensor<192x192xf32>
    %adb2Wv_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wv_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWv_7 = stablehlo.multiply %adb2Wv_7, %Wv_7v : tensor<192x192xf32>
    %adg2Wv_7 = stablehlo.multiply %vitb7_mdWV, %vitb7_mdWV : tensor<192x192xf32>
    %advgWv_7 = stablehlo.multiply %adob2Wv_7, %adg2Wv_7 : tensor<192x192xf32>
    %advnWv_7 = stablehlo.add %advsWv_7, %advgWv_7 : tensor<192x192xf32>
    %adbc1Wv_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wv_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWv_7 = stablehlo.divide %admnWv_7, %adbc1Wv_7 : tensor<192x192xf32>
    %advhWv_7 = stablehlo.divide %advnWv_7, %adbc2Wv_7 : tensor<192x192xf32>
    %adlrWv_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWv_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWv_7 = stablehlo.sqrt %advhWv_7 : tensor<192x192xf32>
    %addenWv_7 = stablehlo.add %adsqWv_7, %adepsWv_7 : tensor<192x192xf32>
    %adratWv_7 = stablehlo.divide %admhWv_7, %addenWv_7 : tensor<192x192xf32>
    %adstWv_7 = stablehlo.multiply %adlrWv_7, %adratWv_7 : tensor<192x192xf32>
    %adsubWv_7 = stablehlo.subtract %Wv_7, %adstWv_7 : tensor<192x192xf32>
    %adwdWv_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWv_7 = stablehlo.multiply %adwdWv_7, %adlrWv_7 : tensor<192x192xf32>
    %adwdpWv_7 = stablehlo.multiply %adwdlrWv_7, %Wv_7 : tensor<192x192xf32>
    %adnewWv_7 = stablehlo.subtract %adsubWv_7, %adwdpWv_7 : tensor<192x192xf32>
    %adb1bv_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bv_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbv_7 = stablehlo.multiply %adb1bv_7, %bv_7m : tensor<192xf32>
    %admgbv_7 = stablehlo.multiply %adob1bv_7, %vitb7_mdbV : tensor<192xf32>
    %admnbv_7 = stablehlo.add %admsbv_7, %admgbv_7 : tensor<192xf32>
    %adb2bv_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bv_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbv_7 = stablehlo.multiply %adb2bv_7, %bv_7v : tensor<192xf32>
    %adg2bv_7 = stablehlo.multiply %vitb7_mdbV, %vitb7_mdbV : tensor<192xf32>
    %advgbv_7 = stablehlo.multiply %adob2bv_7, %adg2bv_7 : tensor<192xf32>
    %advnbv_7 = stablehlo.add %advsbv_7, %advgbv_7 : tensor<192xf32>
    %adbc1bv_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bv_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbv_7 = stablehlo.divide %admnbv_7, %adbc1bv_7 : tensor<192xf32>
    %advhbv_7 = stablehlo.divide %advnbv_7, %adbc2bv_7 : tensor<192xf32>
    %adlrbv_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbv_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbv_7 = stablehlo.sqrt %advhbv_7 : tensor<192xf32>
    %addenbv_7 = stablehlo.add %adsqbv_7, %adepsbv_7 : tensor<192xf32>
    %adratbv_7 = stablehlo.divide %admhbv_7, %addenbv_7 : tensor<192xf32>
    %adstbv_7 = stablehlo.multiply %adlrbv_7, %adratbv_7 : tensor<192xf32>
    %adsubbv_7 = stablehlo.subtract %bv_7, %adstbv_7 : tensor<192xf32>
    %adwdbv_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbv_7 = stablehlo.multiply %adwdbv_7, %adlrbv_7 : tensor<192xf32>
    %adwdpbv_7 = stablehlo.multiply %adwdlrbv_7, %bv_7 : tensor<192xf32>
    %adnewbv_7 = stablehlo.subtract %adsubbv_7, %adwdpbv_7 : tensor<192xf32>
    %adb1Wo_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wo_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWo_7 = stablehlo.multiply %adb1Wo_7, %Wo_7m : tensor<192x192xf32>
    %admgWo_7 = stablehlo.multiply %adob1Wo_7, %vitb7_mdWo : tensor<192x192xf32>
    %admnWo_7 = stablehlo.add %admsWo_7, %admgWo_7 : tensor<192x192xf32>
    %adb2Wo_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wo_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWo_7 = stablehlo.multiply %adb2Wo_7, %Wo_7v : tensor<192x192xf32>
    %adg2Wo_7 = stablehlo.multiply %vitb7_mdWo, %vitb7_mdWo : tensor<192x192xf32>
    %advgWo_7 = stablehlo.multiply %adob2Wo_7, %adg2Wo_7 : tensor<192x192xf32>
    %advnWo_7 = stablehlo.add %advsWo_7, %advgWo_7 : tensor<192x192xf32>
    %adbc1Wo_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wo_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWo_7 = stablehlo.divide %admnWo_7, %adbc1Wo_7 : tensor<192x192xf32>
    %advhWo_7 = stablehlo.divide %advnWo_7, %adbc2Wo_7 : tensor<192x192xf32>
    %adlrWo_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWo_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWo_7 = stablehlo.sqrt %advhWo_7 : tensor<192x192xf32>
    %addenWo_7 = stablehlo.add %adsqWo_7, %adepsWo_7 : tensor<192x192xf32>
    %adratWo_7 = stablehlo.divide %admhWo_7, %addenWo_7 : tensor<192x192xf32>
    %adstWo_7 = stablehlo.multiply %adlrWo_7, %adratWo_7 : tensor<192x192xf32>
    %adsubWo_7 = stablehlo.subtract %Wo_7, %adstWo_7 : tensor<192x192xf32>
    %adwdWo_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWo_7 = stablehlo.multiply %adwdWo_7, %adlrWo_7 : tensor<192x192xf32>
    %adwdpWo_7 = stablehlo.multiply %adwdlrWo_7, %Wo_7 : tensor<192x192xf32>
    %adnewWo_7 = stablehlo.subtract %adsubWo_7, %adwdpWo_7 : tensor<192x192xf32>
    %adb1bo_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bo_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbo_7 = stablehlo.multiply %adb1bo_7, %bo_7m : tensor<192xf32>
    %admgbo_7 = stablehlo.multiply %adob1bo_7, %vitb7_mdbo : tensor<192xf32>
    %admnbo_7 = stablehlo.add %admsbo_7, %admgbo_7 : tensor<192xf32>
    %adb2bo_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bo_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbo_7 = stablehlo.multiply %adb2bo_7, %bo_7v : tensor<192xf32>
    %adg2bo_7 = stablehlo.multiply %vitb7_mdbo, %vitb7_mdbo : tensor<192xf32>
    %advgbo_7 = stablehlo.multiply %adob2bo_7, %adg2bo_7 : tensor<192xf32>
    %advnbo_7 = stablehlo.add %advsbo_7, %advgbo_7 : tensor<192xf32>
    %adbc1bo_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bo_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbo_7 = stablehlo.divide %admnbo_7, %adbc1bo_7 : tensor<192xf32>
    %advhbo_7 = stablehlo.divide %advnbo_7, %adbc2bo_7 : tensor<192xf32>
    %adlrbo_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbo_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbo_7 = stablehlo.sqrt %advhbo_7 : tensor<192xf32>
    %addenbo_7 = stablehlo.add %adsqbo_7, %adepsbo_7 : tensor<192xf32>
    %adratbo_7 = stablehlo.divide %admhbo_7, %addenbo_7 : tensor<192xf32>
    %adstbo_7 = stablehlo.multiply %adlrbo_7, %adratbo_7 : tensor<192xf32>
    %adsubbo_7 = stablehlo.subtract %bo_7, %adstbo_7 : tensor<192xf32>
    %adwdbo_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbo_7 = stablehlo.multiply %adwdbo_7, %adlrbo_7 : tensor<192xf32>
    %adwdpbo_7 = stablehlo.multiply %adwdlrbo_7, %bo_7 : tensor<192xf32>
    %adnewbo_7 = stablehlo.subtract %adsubbo_7, %adwdpbo_7 : tensor<192xf32>
    %adb1g2_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g2_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg2_7 = stablehlo.multiply %adb1g2_7, %g2_7m : tensor<192xf32>
    %admgg2_7 = stablehlo.multiply %adob1g2_7, %vitb7_2dg : tensor<192xf32>
    %admng2_7 = stablehlo.add %admsg2_7, %admgg2_7 : tensor<192xf32>
    %adb2g2_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g2_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg2_7 = stablehlo.multiply %adb2g2_7, %g2_7v : tensor<192xf32>
    %adg2g2_7 = stablehlo.multiply %vitb7_2dg, %vitb7_2dg : tensor<192xf32>
    %advgg2_7 = stablehlo.multiply %adob2g2_7, %adg2g2_7 : tensor<192xf32>
    %advng2_7 = stablehlo.add %advsg2_7, %advgg2_7 : tensor<192xf32>
    %adbc1g2_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g2_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg2_7 = stablehlo.divide %admng2_7, %adbc1g2_7 : tensor<192xf32>
    %advhg2_7 = stablehlo.divide %advng2_7, %adbc2g2_7 : tensor<192xf32>
    %adlrg2_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg2_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg2_7 = stablehlo.sqrt %advhg2_7 : tensor<192xf32>
    %addeng2_7 = stablehlo.add %adsqg2_7, %adepsg2_7 : tensor<192xf32>
    %adratg2_7 = stablehlo.divide %admhg2_7, %addeng2_7 : tensor<192xf32>
    %adstg2_7 = stablehlo.multiply %adlrg2_7, %adratg2_7 : tensor<192xf32>
    %adsubg2_7 = stablehlo.subtract %g2_7, %adstg2_7 : tensor<192xf32>
    %adwdg2_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg2_7 = stablehlo.multiply %adwdg2_7, %adlrg2_7 : tensor<192xf32>
    %adwdpg2_7 = stablehlo.multiply %adwdlrg2_7, %g2_7 : tensor<192xf32>
    %adnewg2_7 = stablehlo.subtract %adsubg2_7, %adwdpg2_7 : tensor<192xf32>
    %adb1b2_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b2_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb2_7 = stablehlo.multiply %adb1b2_7, %b2_7m : tensor<192xf32>
    %admgb2_7 = stablehlo.multiply %adob1b2_7, %vitb7_2db : tensor<192xf32>
    %admnb2_7 = stablehlo.add %admsb2_7, %admgb2_7 : tensor<192xf32>
    %adb2b2_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b2_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb2_7 = stablehlo.multiply %adb2b2_7, %b2_7v : tensor<192xf32>
    %adg2b2_7 = stablehlo.multiply %vitb7_2db, %vitb7_2db : tensor<192xf32>
    %advgb2_7 = stablehlo.multiply %adob2b2_7, %adg2b2_7 : tensor<192xf32>
    %advnb2_7 = stablehlo.add %advsb2_7, %advgb2_7 : tensor<192xf32>
    %adbc1b2_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b2_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb2_7 = stablehlo.divide %admnb2_7, %adbc1b2_7 : tensor<192xf32>
    %advhb2_7 = stablehlo.divide %advnb2_7, %adbc2b2_7 : tensor<192xf32>
    %adlrb2_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb2_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb2_7 = stablehlo.sqrt %advhb2_7 : tensor<192xf32>
    %addenb2_7 = stablehlo.add %adsqb2_7, %adepsb2_7 : tensor<192xf32>
    %adratb2_7 = stablehlo.divide %admhb2_7, %addenb2_7 : tensor<192xf32>
    %adstb2_7 = stablehlo.multiply %adlrb2_7, %adratb2_7 : tensor<192xf32>
    %adsubb2_7 = stablehlo.subtract %b2_7, %adstb2_7 : tensor<192xf32>
    %adwdb2_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb2_7 = stablehlo.multiply %adwdb2_7, %adlrb2_7 : tensor<192xf32>
    %adwdpb2_7 = stablehlo.multiply %adwdlrb2_7, %b2_7 : tensor<192xf32>
    %adnewb2_7 = stablehlo.subtract %adsubb2_7, %adwdpb2_7 : tensor<192xf32>
    %adb1Wfc1_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob1Wfc1_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admsWfc1_7 = stablehlo.multiply %adb1Wfc1_7, %Wfc1_7m : tensor<192x768xf32>
    %admgWfc1_7 = stablehlo.multiply %adob1Wfc1_7, %vitb7_pdWfc1 : tensor<192x768xf32>
    %admnWfc1_7 = stablehlo.add %admsWfc1_7, %admgWfc1_7 : tensor<192x768xf32>
    %adb2Wfc1_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob2Wfc1_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %advsWfc1_7 = stablehlo.multiply %adb2Wfc1_7, %Wfc1_7v : tensor<192x768xf32>
    %adg2Wfc1_7 = stablehlo.multiply %vitb7_pdWfc1, %vitb7_pdWfc1 : tensor<192x768xf32>
    %advgWfc1_7 = stablehlo.multiply %adob2Wfc1_7, %adg2Wfc1_7 : tensor<192x768xf32>
    %advnWfc1_7 = stablehlo.add %advsWfc1_7, %advgWfc1_7 : tensor<192x768xf32>
    %adbc1Wfc1_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adbc2Wfc1_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admhWfc1_7 = stablehlo.divide %admnWfc1_7, %adbc1Wfc1_7 : tensor<192x768xf32>
    %advhWfc1_7 = stablehlo.divide %advnWfc1_7, %adbc2Wfc1_7 : tensor<192x768xf32>
    %adlrWfc1_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adepsWfc1_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adsqWfc1_7 = stablehlo.sqrt %advhWfc1_7 : tensor<192x768xf32>
    %addenWfc1_7 = stablehlo.add %adsqWfc1_7, %adepsWfc1_7 : tensor<192x768xf32>
    %adratWfc1_7 = stablehlo.divide %admhWfc1_7, %addenWfc1_7 : tensor<192x768xf32>
    %adstWfc1_7 = stablehlo.multiply %adlrWfc1_7, %adratWfc1_7 : tensor<192x768xf32>
    %adsubWfc1_7 = stablehlo.subtract %Wfc1_7, %adstWfc1_7 : tensor<192x768xf32>
    %adwdWfc1_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adwdlrWfc1_7 = stablehlo.multiply %adwdWfc1_7, %adlrWfc1_7 : tensor<192x768xf32>
    %adwdpWfc1_7 = stablehlo.multiply %adwdlrWfc1_7, %Wfc1_7 : tensor<192x768xf32>
    %adnewWfc1_7 = stablehlo.subtract %adsubWfc1_7, %adwdpWfc1_7 : tensor<192x768xf32>
    %adb1bfc1_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1bfc1_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admsbfc1_7 = stablehlo.multiply %adb1bfc1_7, %bfc1_7m : tensor<768xf32>
    %admgbfc1_7 = stablehlo.multiply %adob1bfc1_7, %vitb7_pdbfc1 : tensor<768xf32>
    %admnbfc1_7 = stablehlo.add %admsbfc1_7, %admgbfc1_7 : tensor<768xf32>
    %adb2bfc1_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2bfc1_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advsbfc1_7 = stablehlo.multiply %adb2bfc1_7, %bfc1_7v : tensor<768xf32>
    %adg2bfc1_7 = stablehlo.multiply %vitb7_pdbfc1, %vitb7_pdbfc1 : tensor<768xf32>
    %advgbfc1_7 = stablehlo.multiply %adob2bfc1_7, %adg2bfc1_7 : tensor<768xf32>
    %advnbfc1_7 = stablehlo.add %advsbfc1_7, %advgbfc1_7 : tensor<768xf32>
    %adbc1bfc1_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2bfc1_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhbfc1_7 = stablehlo.divide %admnbfc1_7, %adbc1bfc1_7 : tensor<768xf32>
    %advhbfc1_7 = stablehlo.divide %advnbfc1_7, %adbc2bfc1_7 : tensor<768xf32>
    %adlrbfc1_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepsbfc1_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqbfc1_7 = stablehlo.sqrt %advhbfc1_7 : tensor<768xf32>
    %addenbfc1_7 = stablehlo.add %adsqbfc1_7, %adepsbfc1_7 : tensor<768xf32>
    %adratbfc1_7 = stablehlo.divide %admhbfc1_7, %addenbfc1_7 : tensor<768xf32>
    %adstbfc1_7 = stablehlo.multiply %adlrbfc1_7, %adratbfc1_7 : tensor<768xf32>
    %adsubbfc1_7 = stablehlo.subtract %bfc1_7, %adstbfc1_7 : tensor<768xf32>
    %adwdbfc1_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrbfc1_7 = stablehlo.multiply %adwdbfc1_7, %adlrbfc1_7 : tensor<768xf32>
    %adwdpbfc1_7 = stablehlo.multiply %adwdlrbfc1_7, %bfc1_7 : tensor<768xf32>
    %adnewbfc1_7 = stablehlo.subtract %adsubbfc1_7, %adwdpbfc1_7 : tensor<768xf32>
    %adb1Wfc2_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob1Wfc2_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admsWfc2_7 = stablehlo.multiply %adb1Wfc2_7, %Wfc2_7m : tensor<768x192xf32>
    %admgWfc2_7 = stablehlo.multiply %adob1Wfc2_7, %vitb7_pdWfc2 : tensor<768x192xf32>
    %admnWfc2_7 = stablehlo.add %admsWfc2_7, %admgWfc2_7 : tensor<768x192xf32>
    %adb2Wfc2_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob2Wfc2_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %advsWfc2_7 = stablehlo.multiply %adb2Wfc2_7, %Wfc2_7v : tensor<768x192xf32>
    %adg2Wfc2_7 = stablehlo.multiply %vitb7_pdWfc2, %vitb7_pdWfc2 : tensor<768x192xf32>
    %advgWfc2_7 = stablehlo.multiply %adob2Wfc2_7, %adg2Wfc2_7 : tensor<768x192xf32>
    %advnWfc2_7 = stablehlo.add %advsWfc2_7, %advgWfc2_7 : tensor<768x192xf32>
    %adbc1Wfc2_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adbc2Wfc2_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admhWfc2_7 = stablehlo.divide %admnWfc2_7, %adbc1Wfc2_7 : tensor<768x192xf32>
    %advhWfc2_7 = stablehlo.divide %advnWfc2_7, %adbc2Wfc2_7 : tensor<768x192xf32>
    %adlrWfc2_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adepsWfc2_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adsqWfc2_7 = stablehlo.sqrt %advhWfc2_7 : tensor<768x192xf32>
    %addenWfc2_7 = stablehlo.add %adsqWfc2_7, %adepsWfc2_7 : tensor<768x192xf32>
    %adratWfc2_7 = stablehlo.divide %admhWfc2_7, %addenWfc2_7 : tensor<768x192xf32>
    %adstWfc2_7 = stablehlo.multiply %adlrWfc2_7, %adratWfc2_7 : tensor<768x192xf32>
    %adsubWfc2_7 = stablehlo.subtract %Wfc2_7, %adstWfc2_7 : tensor<768x192xf32>
    %adwdWfc2_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adwdlrWfc2_7 = stablehlo.multiply %adwdWfc2_7, %adlrWfc2_7 : tensor<768x192xf32>
    %adwdpWfc2_7 = stablehlo.multiply %adwdlrWfc2_7, %Wfc2_7 : tensor<768x192xf32>
    %adnewWfc2_7 = stablehlo.subtract %adsubWfc2_7, %adwdpWfc2_7 : tensor<768x192xf32>
    %adb1bfc2_7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bfc2_7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbfc2_7 = stablehlo.multiply %adb1bfc2_7, %bfc2_7m : tensor<192xf32>
    %admgbfc2_7 = stablehlo.multiply %adob1bfc2_7, %vitb7_pdbfc2 : tensor<192xf32>
    %admnbfc2_7 = stablehlo.add %admsbfc2_7, %admgbfc2_7 : tensor<192xf32>
    %adb2bfc2_7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bfc2_7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbfc2_7 = stablehlo.multiply %adb2bfc2_7, %bfc2_7v : tensor<192xf32>
    %adg2bfc2_7 = stablehlo.multiply %vitb7_pdbfc2, %vitb7_pdbfc2 : tensor<192xf32>
    %advgbfc2_7 = stablehlo.multiply %adob2bfc2_7, %adg2bfc2_7 : tensor<192xf32>
    %advnbfc2_7 = stablehlo.add %advsbfc2_7, %advgbfc2_7 : tensor<192xf32>
    %adbc1bfc2_7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bfc2_7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbfc2_7 = stablehlo.divide %admnbfc2_7, %adbc1bfc2_7 : tensor<192xf32>
    %advhbfc2_7 = stablehlo.divide %advnbfc2_7, %adbc2bfc2_7 : tensor<192xf32>
    %adlrbfc2_7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbfc2_7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbfc2_7 = stablehlo.sqrt %advhbfc2_7 : tensor<192xf32>
    %addenbfc2_7 = stablehlo.add %adsqbfc2_7, %adepsbfc2_7 : tensor<192xf32>
    %adratbfc2_7 = stablehlo.divide %admhbfc2_7, %addenbfc2_7 : tensor<192xf32>
    %adstbfc2_7 = stablehlo.multiply %adlrbfc2_7, %adratbfc2_7 : tensor<192xf32>
    %adsubbfc2_7 = stablehlo.subtract %bfc2_7, %adstbfc2_7 : tensor<192xf32>
    %adwdbfc2_7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbfc2_7 = stablehlo.multiply %adwdbfc2_7, %adlrbfc2_7 : tensor<192xf32>
    %adwdpbfc2_7 = stablehlo.multiply %adwdlrbfc2_7, %bfc2_7 : tensor<192xf32>
    %adnewbfc2_7 = stablehlo.subtract %adsubbfc2_7, %adwdpbfc2_7 : tensor<192xf32>
    %adb1g1_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g1_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg1_8 = stablehlo.multiply %adb1g1_8, %g1_8m : tensor<192xf32>
    %admgg1_8 = stablehlo.multiply %adob1g1_8, %vitb8_1dg : tensor<192xf32>
    %admng1_8 = stablehlo.add %admsg1_8, %admgg1_8 : tensor<192xf32>
    %adb2g1_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g1_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg1_8 = stablehlo.multiply %adb2g1_8, %g1_8v : tensor<192xf32>
    %adg2g1_8 = stablehlo.multiply %vitb8_1dg, %vitb8_1dg : tensor<192xf32>
    %advgg1_8 = stablehlo.multiply %adob2g1_8, %adg2g1_8 : tensor<192xf32>
    %advng1_8 = stablehlo.add %advsg1_8, %advgg1_8 : tensor<192xf32>
    %adbc1g1_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g1_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg1_8 = stablehlo.divide %admng1_8, %adbc1g1_8 : tensor<192xf32>
    %advhg1_8 = stablehlo.divide %advng1_8, %adbc2g1_8 : tensor<192xf32>
    %adlrg1_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg1_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg1_8 = stablehlo.sqrt %advhg1_8 : tensor<192xf32>
    %addeng1_8 = stablehlo.add %adsqg1_8, %adepsg1_8 : tensor<192xf32>
    %adratg1_8 = stablehlo.divide %admhg1_8, %addeng1_8 : tensor<192xf32>
    %adstg1_8 = stablehlo.multiply %adlrg1_8, %adratg1_8 : tensor<192xf32>
    %adsubg1_8 = stablehlo.subtract %g1_8, %adstg1_8 : tensor<192xf32>
    %adwdg1_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg1_8 = stablehlo.multiply %adwdg1_8, %adlrg1_8 : tensor<192xf32>
    %adwdpg1_8 = stablehlo.multiply %adwdlrg1_8, %g1_8 : tensor<192xf32>
    %adnewg1_8 = stablehlo.subtract %adsubg1_8, %adwdpg1_8 : tensor<192xf32>
    %adb1b1_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b1_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb1_8 = stablehlo.multiply %adb1b1_8, %b1_8m : tensor<192xf32>
    %admgb1_8 = stablehlo.multiply %adob1b1_8, %vitb8_1db : tensor<192xf32>
    %admnb1_8 = stablehlo.add %admsb1_8, %admgb1_8 : tensor<192xf32>
    %adb2b1_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b1_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb1_8 = stablehlo.multiply %adb2b1_8, %b1_8v : tensor<192xf32>
    %adg2b1_8 = stablehlo.multiply %vitb8_1db, %vitb8_1db : tensor<192xf32>
    %advgb1_8 = stablehlo.multiply %adob2b1_8, %adg2b1_8 : tensor<192xf32>
    %advnb1_8 = stablehlo.add %advsb1_8, %advgb1_8 : tensor<192xf32>
    %adbc1b1_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b1_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb1_8 = stablehlo.divide %admnb1_8, %adbc1b1_8 : tensor<192xf32>
    %advhb1_8 = stablehlo.divide %advnb1_8, %adbc2b1_8 : tensor<192xf32>
    %adlrb1_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb1_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb1_8 = stablehlo.sqrt %advhb1_8 : tensor<192xf32>
    %addenb1_8 = stablehlo.add %adsqb1_8, %adepsb1_8 : tensor<192xf32>
    %adratb1_8 = stablehlo.divide %admhb1_8, %addenb1_8 : tensor<192xf32>
    %adstb1_8 = stablehlo.multiply %adlrb1_8, %adratb1_8 : tensor<192xf32>
    %adsubb1_8 = stablehlo.subtract %b1_8, %adstb1_8 : tensor<192xf32>
    %adwdb1_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb1_8 = stablehlo.multiply %adwdb1_8, %adlrb1_8 : tensor<192xf32>
    %adwdpb1_8 = stablehlo.multiply %adwdlrb1_8, %b1_8 : tensor<192xf32>
    %adnewb1_8 = stablehlo.subtract %adsubb1_8, %adwdpb1_8 : tensor<192xf32>
    %adb1Wq_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wq_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWq_8 = stablehlo.multiply %adb1Wq_8, %Wq_8m : tensor<192x192xf32>
    %admgWq_8 = stablehlo.multiply %adob1Wq_8, %vitb8_mdWQ : tensor<192x192xf32>
    %admnWq_8 = stablehlo.add %admsWq_8, %admgWq_8 : tensor<192x192xf32>
    %adb2Wq_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wq_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWq_8 = stablehlo.multiply %adb2Wq_8, %Wq_8v : tensor<192x192xf32>
    %adg2Wq_8 = stablehlo.multiply %vitb8_mdWQ, %vitb8_mdWQ : tensor<192x192xf32>
    %advgWq_8 = stablehlo.multiply %adob2Wq_8, %adg2Wq_8 : tensor<192x192xf32>
    %advnWq_8 = stablehlo.add %advsWq_8, %advgWq_8 : tensor<192x192xf32>
    %adbc1Wq_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wq_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWq_8 = stablehlo.divide %admnWq_8, %adbc1Wq_8 : tensor<192x192xf32>
    %advhWq_8 = stablehlo.divide %advnWq_8, %adbc2Wq_8 : tensor<192x192xf32>
    %adlrWq_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWq_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWq_8 = stablehlo.sqrt %advhWq_8 : tensor<192x192xf32>
    %addenWq_8 = stablehlo.add %adsqWq_8, %adepsWq_8 : tensor<192x192xf32>
    %adratWq_8 = stablehlo.divide %admhWq_8, %addenWq_8 : tensor<192x192xf32>
    %adstWq_8 = stablehlo.multiply %adlrWq_8, %adratWq_8 : tensor<192x192xf32>
    %adsubWq_8 = stablehlo.subtract %Wq_8, %adstWq_8 : tensor<192x192xf32>
    %adwdWq_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWq_8 = stablehlo.multiply %adwdWq_8, %adlrWq_8 : tensor<192x192xf32>
    %adwdpWq_8 = stablehlo.multiply %adwdlrWq_8, %Wq_8 : tensor<192x192xf32>
    %adnewWq_8 = stablehlo.subtract %adsubWq_8, %adwdpWq_8 : tensor<192x192xf32>
    %adb1bq_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bq_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbq_8 = stablehlo.multiply %adb1bq_8, %bq_8m : tensor<192xf32>
    %admgbq_8 = stablehlo.multiply %adob1bq_8, %vitb8_mdbQ : tensor<192xf32>
    %admnbq_8 = stablehlo.add %admsbq_8, %admgbq_8 : tensor<192xf32>
    %adb2bq_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bq_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbq_8 = stablehlo.multiply %adb2bq_8, %bq_8v : tensor<192xf32>
    %adg2bq_8 = stablehlo.multiply %vitb8_mdbQ, %vitb8_mdbQ : tensor<192xf32>
    %advgbq_8 = stablehlo.multiply %adob2bq_8, %adg2bq_8 : tensor<192xf32>
    %advnbq_8 = stablehlo.add %advsbq_8, %advgbq_8 : tensor<192xf32>
    %adbc1bq_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bq_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbq_8 = stablehlo.divide %admnbq_8, %adbc1bq_8 : tensor<192xf32>
    %advhbq_8 = stablehlo.divide %advnbq_8, %adbc2bq_8 : tensor<192xf32>
    %adlrbq_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbq_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbq_8 = stablehlo.sqrt %advhbq_8 : tensor<192xf32>
    %addenbq_8 = stablehlo.add %adsqbq_8, %adepsbq_8 : tensor<192xf32>
    %adratbq_8 = stablehlo.divide %admhbq_8, %addenbq_8 : tensor<192xf32>
    %adstbq_8 = stablehlo.multiply %adlrbq_8, %adratbq_8 : tensor<192xf32>
    %adsubbq_8 = stablehlo.subtract %bq_8, %adstbq_8 : tensor<192xf32>
    %adwdbq_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbq_8 = stablehlo.multiply %adwdbq_8, %adlrbq_8 : tensor<192xf32>
    %adwdpbq_8 = stablehlo.multiply %adwdlrbq_8, %bq_8 : tensor<192xf32>
    %adnewbq_8 = stablehlo.subtract %adsubbq_8, %adwdpbq_8 : tensor<192xf32>
    %adb1Wk_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wk_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWk_8 = stablehlo.multiply %adb1Wk_8, %Wk_8m : tensor<192x192xf32>
    %admgWk_8 = stablehlo.multiply %adob1Wk_8, %vitb8_mdWK : tensor<192x192xf32>
    %admnWk_8 = stablehlo.add %admsWk_8, %admgWk_8 : tensor<192x192xf32>
    %adb2Wk_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wk_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWk_8 = stablehlo.multiply %adb2Wk_8, %Wk_8v : tensor<192x192xf32>
    %adg2Wk_8 = stablehlo.multiply %vitb8_mdWK, %vitb8_mdWK : tensor<192x192xf32>
    %advgWk_8 = stablehlo.multiply %adob2Wk_8, %adg2Wk_8 : tensor<192x192xf32>
    %advnWk_8 = stablehlo.add %advsWk_8, %advgWk_8 : tensor<192x192xf32>
    %adbc1Wk_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wk_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWk_8 = stablehlo.divide %admnWk_8, %adbc1Wk_8 : tensor<192x192xf32>
    %advhWk_8 = stablehlo.divide %advnWk_8, %adbc2Wk_8 : tensor<192x192xf32>
    %adlrWk_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWk_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWk_8 = stablehlo.sqrt %advhWk_8 : tensor<192x192xf32>
    %addenWk_8 = stablehlo.add %adsqWk_8, %adepsWk_8 : tensor<192x192xf32>
    %adratWk_8 = stablehlo.divide %admhWk_8, %addenWk_8 : tensor<192x192xf32>
    %adstWk_8 = stablehlo.multiply %adlrWk_8, %adratWk_8 : tensor<192x192xf32>
    %adsubWk_8 = stablehlo.subtract %Wk_8, %adstWk_8 : tensor<192x192xf32>
    %adwdWk_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWk_8 = stablehlo.multiply %adwdWk_8, %adlrWk_8 : tensor<192x192xf32>
    %adwdpWk_8 = stablehlo.multiply %adwdlrWk_8, %Wk_8 : tensor<192x192xf32>
    %adnewWk_8 = stablehlo.subtract %adsubWk_8, %adwdpWk_8 : tensor<192x192xf32>
    %adb1bk_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bk_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbk_8 = stablehlo.multiply %adb1bk_8, %bk_8m : tensor<192xf32>
    %admgbk_8 = stablehlo.multiply %adob1bk_8, %vitb8_mdbK : tensor<192xf32>
    %admnbk_8 = stablehlo.add %admsbk_8, %admgbk_8 : tensor<192xf32>
    %adb2bk_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bk_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbk_8 = stablehlo.multiply %adb2bk_8, %bk_8v : tensor<192xf32>
    %adg2bk_8 = stablehlo.multiply %vitb8_mdbK, %vitb8_mdbK : tensor<192xf32>
    %advgbk_8 = stablehlo.multiply %adob2bk_8, %adg2bk_8 : tensor<192xf32>
    %advnbk_8 = stablehlo.add %advsbk_8, %advgbk_8 : tensor<192xf32>
    %adbc1bk_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bk_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbk_8 = stablehlo.divide %admnbk_8, %adbc1bk_8 : tensor<192xf32>
    %advhbk_8 = stablehlo.divide %advnbk_8, %adbc2bk_8 : tensor<192xf32>
    %adlrbk_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbk_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbk_8 = stablehlo.sqrt %advhbk_8 : tensor<192xf32>
    %addenbk_8 = stablehlo.add %adsqbk_8, %adepsbk_8 : tensor<192xf32>
    %adratbk_8 = stablehlo.divide %admhbk_8, %addenbk_8 : tensor<192xf32>
    %adstbk_8 = stablehlo.multiply %adlrbk_8, %adratbk_8 : tensor<192xf32>
    %adsubbk_8 = stablehlo.subtract %bk_8, %adstbk_8 : tensor<192xf32>
    %adwdbk_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbk_8 = stablehlo.multiply %adwdbk_8, %adlrbk_8 : tensor<192xf32>
    %adwdpbk_8 = stablehlo.multiply %adwdlrbk_8, %bk_8 : tensor<192xf32>
    %adnewbk_8 = stablehlo.subtract %adsubbk_8, %adwdpbk_8 : tensor<192xf32>
    %adb1Wv_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wv_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWv_8 = stablehlo.multiply %adb1Wv_8, %Wv_8m : tensor<192x192xf32>
    %admgWv_8 = stablehlo.multiply %adob1Wv_8, %vitb8_mdWV : tensor<192x192xf32>
    %admnWv_8 = stablehlo.add %admsWv_8, %admgWv_8 : tensor<192x192xf32>
    %adb2Wv_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wv_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWv_8 = stablehlo.multiply %adb2Wv_8, %Wv_8v : tensor<192x192xf32>
    %adg2Wv_8 = stablehlo.multiply %vitb8_mdWV, %vitb8_mdWV : tensor<192x192xf32>
    %advgWv_8 = stablehlo.multiply %adob2Wv_8, %adg2Wv_8 : tensor<192x192xf32>
    %advnWv_8 = stablehlo.add %advsWv_8, %advgWv_8 : tensor<192x192xf32>
    %adbc1Wv_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wv_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWv_8 = stablehlo.divide %admnWv_8, %adbc1Wv_8 : tensor<192x192xf32>
    %advhWv_8 = stablehlo.divide %advnWv_8, %adbc2Wv_8 : tensor<192x192xf32>
    %adlrWv_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWv_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWv_8 = stablehlo.sqrt %advhWv_8 : tensor<192x192xf32>
    %addenWv_8 = stablehlo.add %adsqWv_8, %adepsWv_8 : tensor<192x192xf32>
    %adratWv_8 = stablehlo.divide %admhWv_8, %addenWv_8 : tensor<192x192xf32>
    %adstWv_8 = stablehlo.multiply %adlrWv_8, %adratWv_8 : tensor<192x192xf32>
    %adsubWv_8 = stablehlo.subtract %Wv_8, %adstWv_8 : tensor<192x192xf32>
    %adwdWv_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWv_8 = stablehlo.multiply %adwdWv_8, %adlrWv_8 : tensor<192x192xf32>
    %adwdpWv_8 = stablehlo.multiply %adwdlrWv_8, %Wv_8 : tensor<192x192xf32>
    %adnewWv_8 = stablehlo.subtract %adsubWv_8, %adwdpWv_8 : tensor<192x192xf32>
    %adb1bv_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bv_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbv_8 = stablehlo.multiply %adb1bv_8, %bv_8m : tensor<192xf32>
    %admgbv_8 = stablehlo.multiply %adob1bv_8, %vitb8_mdbV : tensor<192xf32>
    %admnbv_8 = stablehlo.add %admsbv_8, %admgbv_8 : tensor<192xf32>
    %adb2bv_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bv_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbv_8 = stablehlo.multiply %adb2bv_8, %bv_8v : tensor<192xf32>
    %adg2bv_8 = stablehlo.multiply %vitb8_mdbV, %vitb8_mdbV : tensor<192xf32>
    %advgbv_8 = stablehlo.multiply %adob2bv_8, %adg2bv_8 : tensor<192xf32>
    %advnbv_8 = stablehlo.add %advsbv_8, %advgbv_8 : tensor<192xf32>
    %adbc1bv_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bv_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbv_8 = stablehlo.divide %admnbv_8, %adbc1bv_8 : tensor<192xf32>
    %advhbv_8 = stablehlo.divide %advnbv_8, %adbc2bv_8 : tensor<192xf32>
    %adlrbv_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbv_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbv_8 = stablehlo.sqrt %advhbv_8 : tensor<192xf32>
    %addenbv_8 = stablehlo.add %adsqbv_8, %adepsbv_8 : tensor<192xf32>
    %adratbv_8 = stablehlo.divide %admhbv_8, %addenbv_8 : tensor<192xf32>
    %adstbv_8 = stablehlo.multiply %adlrbv_8, %adratbv_8 : tensor<192xf32>
    %adsubbv_8 = stablehlo.subtract %bv_8, %adstbv_8 : tensor<192xf32>
    %adwdbv_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbv_8 = stablehlo.multiply %adwdbv_8, %adlrbv_8 : tensor<192xf32>
    %adwdpbv_8 = stablehlo.multiply %adwdlrbv_8, %bv_8 : tensor<192xf32>
    %adnewbv_8 = stablehlo.subtract %adsubbv_8, %adwdpbv_8 : tensor<192xf32>
    %adb1Wo_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wo_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWo_8 = stablehlo.multiply %adb1Wo_8, %Wo_8m : tensor<192x192xf32>
    %admgWo_8 = stablehlo.multiply %adob1Wo_8, %vitb8_mdWo : tensor<192x192xf32>
    %admnWo_8 = stablehlo.add %admsWo_8, %admgWo_8 : tensor<192x192xf32>
    %adb2Wo_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wo_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWo_8 = stablehlo.multiply %adb2Wo_8, %Wo_8v : tensor<192x192xf32>
    %adg2Wo_8 = stablehlo.multiply %vitb8_mdWo, %vitb8_mdWo : tensor<192x192xf32>
    %advgWo_8 = stablehlo.multiply %adob2Wo_8, %adg2Wo_8 : tensor<192x192xf32>
    %advnWo_8 = stablehlo.add %advsWo_8, %advgWo_8 : tensor<192x192xf32>
    %adbc1Wo_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wo_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWo_8 = stablehlo.divide %admnWo_8, %adbc1Wo_8 : tensor<192x192xf32>
    %advhWo_8 = stablehlo.divide %advnWo_8, %adbc2Wo_8 : tensor<192x192xf32>
    %adlrWo_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWo_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWo_8 = stablehlo.sqrt %advhWo_8 : tensor<192x192xf32>
    %addenWo_8 = stablehlo.add %adsqWo_8, %adepsWo_8 : tensor<192x192xf32>
    %adratWo_8 = stablehlo.divide %admhWo_8, %addenWo_8 : tensor<192x192xf32>
    %adstWo_8 = stablehlo.multiply %adlrWo_8, %adratWo_8 : tensor<192x192xf32>
    %adsubWo_8 = stablehlo.subtract %Wo_8, %adstWo_8 : tensor<192x192xf32>
    %adwdWo_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWo_8 = stablehlo.multiply %adwdWo_8, %adlrWo_8 : tensor<192x192xf32>
    %adwdpWo_8 = stablehlo.multiply %adwdlrWo_8, %Wo_8 : tensor<192x192xf32>
    %adnewWo_8 = stablehlo.subtract %adsubWo_8, %adwdpWo_8 : tensor<192x192xf32>
    %adb1bo_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bo_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbo_8 = stablehlo.multiply %adb1bo_8, %bo_8m : tensor<192xf32>
    %admgbo_8 = stablehlo.multiply %adob1bo_8, %vitb8_mdbo : tensor<192xf32>
    %admnbo_8 = stablehlo.add %admsbo_8, %admgbo_8 : tensor<192xf32>
    %adb2bo_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bo_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbo_8 = stablehlo.multiply %adb2bo_8, %bo_8v : tensor<192xf32>
    %adg2bo_8 = stablehlo.multiply %vitb8_mdbo, %vitb8_mdbo : tensor<192xf32>
    %advgbo_8 = stablehlo.multiply %adob2bo_8, %adg2bo_8 : tensor<192xf32>
    %advnbo_8 = stablehlo.add %advsbo_8, %advgbo_8 : tensor<192xf32>
    %adbc1bo_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bo_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbo_8 = stablehlo.divide %admnbo_8, %adbc1bo_8 : tensor<192xf32>
    %advhbo_8 = stablehlo.divide %advnbo_8, %adbc2bo_8 : tensor<192xf32>
    %adlrbo_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbo_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbo_8 = stablehlo.sqrt %advhbo_8 : tensor<192xf32>
    %addenbo_8 = stablehlo.add %adsqbo_8, %adepsbo_8 : tensor<192xf32>
    %adratbo_8 = stablehlo.divide %admhbo_8, %addenbo_8 : tensor<192xf32>
    %adstbo_8 = stablehlo.multiply %adlrbo_8, %adratbo_8 : tensor<192xf32>
    %adsubbo_8 = stablehlo.subtract %bo_8, %adstbo_8 : tensor<192xf32>
    %adwdbo_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbo_8 = stablehlo.multiply %adwdbo_8, %adlrbo_8 : tensor<192xf32>
    %adwdpbo_8 = stablehlo.multiply %adwdlrbo_8, %bo_8 : tensor<192xf32>
    %adnewbo_8 = stablehlo.subtract %adsubbo_8, %adwdpbo_8 : tensor<192xf32>
    %adb1g2_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g2_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg2_8 = stablehlo.multiply %adb1g2_8, %g2_8m : tensor<192xf32>
    %admgg2_8 = stablehlo.multiply %adob1g2_8, %vitb8_2dg : tensor<192xf32>
    %admng2_8 = stablehlo.add %admsg2_8, %admgg2_8 : tensor<192xf32>
    %adb2g2_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g2_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg2_8 = stablehlo.multiply %adb2g2_8, %g2_8v : tensor<192xf32>
    %adg2g2_8 = stablehlo.multiply %vitb8_2dg, %vitb8_2dg : tensor<192xf32>
    %advgg2_8 = stablehlo.multiply %adob2g2_8, %adg2g2_8 : tensor<192xf32>
    %advng2_8 = stablehlo.add %advsg2_8, %advgg2_8 : tensor<192xf32>
    %adbc1g2_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g2_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg2_8 = stablehlo.divide %admng2_8, %adbc1g2_8 : tensor<192xf32>
    %advhg2_8 = stablehlo.divide %advng2_8, %adbc2g2_8 : tensor<192xf32>
    %adlrg2_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg2_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg2_8 = stablehlo.sqrt %advhg2_8 : tensor<192xf32>
    %addeng2_8 = stablehlo.add %adsqg2_8, %adepsg2_8 : tensor<192xf32>
    %adratg2_8 = stablehlo.divide %admhg2_8, %addeng2_8 : tensor<192xf32>
    %adstg2_8 = stablehlo.multiply %adlrg2_8, %adratg2_8 : tensor<192xf32>
    %adsubg2_8 = stablehlo.subtract %g2_8, %adstg2_8 : tensor<192xf32>
    %adwdg2_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg2_8 = stablehlo.multiply %adwdg2_8, %adlrg2_8 : tensor<192xf32>
    %adwdpg2_8 = stablehlo.multiply %adwdlrg2_8, %g2_8 : tensor<192xf32>
    %adnewg2_8 = stablehlo.subtract %adsubg2_8, %adwdpg2_8 : tensor<192xf32>
    %adb1b2_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b2_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb2_8 = stablehlo.multiply %adb1b2_8, %b2_8m : tensor<192xf32>
    %admgb2_8 = stablehlo.multiply %adob1b2_8, %vitb8_2db : tensor<192xf32>
    %admnb2_8 = stablehlo.add %admsb2_8, %admgb2_8 : tensor<192xf32>
    %adb2b2_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b2_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb2_8 = stablehlo.multiply %adb2b2_8, %b2_8v : tensor<192xf32>
    %adg2b2_8 = stablehlo.multiply %vitb8_2db, %vitb8_2db : tensor<192xf32>
    %advgb2_8 = stablehlo.multiply %adob2b2_8, %adg2b2_8 : tensor<192xf32>
    %advnb2_8 = stablehlo.add %advsb2_8, %advgb2_8 : tensor<192xf32>
    %adbc1b2_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b2_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb2_8 = stablehlo.divide %admnb2_8, %adbc1b2_8 : tensor<192xf32>
    %advhb2_8 = stablehlo.divide %advnb2_8, %adbc2b2_8 : tensor<192xf32>
    %adlrb2_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb2_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb2_8 = stablehlo.sqrt %advhb2_8 : tensor<192xf32>
    %addenb2_8 = stablehlo.add %adsqb2_8, %adepsb2_8 : tensor<192xf32>
    %adratb2_8 = stablehlo.divide %admhb2_8, %addenb2_8 : tensor<192xf32>
    %adstb2_8 = stablehlo.multiply %adlrb2_8, %adratb2_8 : tensor<192xf32>
    %adsubb2_8 = stablehlo.subtract %b2_8, %adstb2_8 : tensor<192xf32>
    %adwdb2_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb2_8 = stablehlo.multiply %adwdb2_8, %adlrb2_8 : tensor<192xf32>
    %adwdpb2_8 = stablehlo.multiply %adwdlrb2_8, %b2_8 : tensor<192xf32>
    %adnewb2_8 = stablehlo.subtract %adsubb2_8, %adwdpb2_8 : tensor<192xf32>
    %adb1Wfc1_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob1Wfc1_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admsWfc1_8 = stablehlo.multiply %adb1Wfc1_8, %Wfc1_8m : tensor<192x768xf32>
    %admgWfc1_8 = stablehlo.multiply %adob1Wfc1_8, %vitb8_pdWfc1 : tensor<192x768xf32>
    %admnWfc1_8 = stablehlo.add %admsWfc1_8, %admgWfc1_8 : tensor<192x768xf32>
    %adb2Wfc1_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob2Wfc1_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %advsWfc1_8 = stablehlo.multiply %adb2Wfc1_8, %Wfc1_8v : tensor<192x768xf32>
    %adg2Wfc1_8 = stablehlo.multiply %vitb8_pdWfc1, %vitb8_pdWfc1 : tensor<192x768xf32>
    %advgWfc1_8 = stablehlo.multiply %adob2Wfc1_8, %adg2Wfc1_8 : tensor<192x768xf32>
    %advnWfc1_8 = stablehlo.add %advsWfc1_8, %advgWfc1_8 : tensor<192x768xf32>
    %adbc1Wfc1_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adbc2Wfc1_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admhWfc1_8 = stablehlo.divide %admnWfc1_8, %adbc1Wfc1_8 : tensor<192x768xf32>
    %advhWfc1_8 = stablehlo.divide %advnWfc1_8, %adbc2Wfc1_8 : tensor<192x768xf32>
    %adlrWfc1_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adepsWfc1_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adsqWfc1_8 = stablehlo.sqrt %advhWfc1_8 : tensor<192x768xf32>
    %addenWfc1_8 = stablehlo.add %adsqWfc1_8, %adepsWfc1_8 : tensor<192x768xf32>
    %adratWfc1_8 = stablehlo.divide %admhWfc1_8, %addenWfc1_8 : tensor<192x768xf32>
    %adstWfc1_8 = stablehlo.multiply %adlrWfc1_8, %adratWfc1_8 : tensor<192x768xf32>
    %adsubWfc1_8 = stablehlo.subtract %Wfc1_8, %adstWfc1_8 : tensor<192x768xf32>
    %adwdWfc1_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adwdlrWfc1_8 = stablehlo.multiply %adwdWfc1_8, %adlrWfc1_8 : tensor<192x768xf32>
    %adwdpWfc1_8 = stablehlo.multiply %adwdlrWfc1_8, %Wfc1_8 : tensor<192x768xf32>
    %adnewWfc1_8 = stablehlo.subtract %adsubWfc1_8, %adwdpWfc1_8 : tensor<192x768xf32>
    %adb1bfc1_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1bfc1_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admsbfc1_8 = stablehlo.multiply %adb1bfc1_8, %bfc1_8m : tensor<768xf32>
    %admgbfc1_8 = stablehlo.multiply %adob1bfc1_8, %vitb8_pdbfc1 : tensor<768xf32>
    %admnbfc1_8 = stablehlo.add %admsbfc1_8, %admgbfc1_8 : tensor<768xf32>
    %adb2bfc1_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2bfc1_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advsbfc1_8 = stablehlo.multiply %adb2bfc1_8, %bfc1_8v : tensor<768xf32>
    %adg2bfc1_8 = stablehlo.multiply %vitb8_pdbfc1, %vitb8_pdbfc1 : tensor<768xf32>
    %advgbfc1_8 = stablehlo.multiply %adob2bfc1_8, %adg2bfc1_8 : tensor<768xf32>
    %advnbfc1_8 = stablehlo.add %advsbfc1_8, %advgbfc1_8 : tensor<768xf32>
    %adbc1bfc1_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2bfc1_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhbfc1_8 = stablehlo.divide %admnbfc1_8, %adbc1bfc1_8 : tensor<768xf32>
    %advhbfc1_8 = stablehlo.divide %advnbfc1_8, %adbc2bfc1_8 : tensor<768xf32>
    %adlrbfc1_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepsbfc1_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqbfc1_8 = stablehlo.sqrt %advhbfc1_8 : tensor<768xf32>
    %addenbfc1_8 = stablehlo.add %adsqbfc1_8, %adepsbfc1_8 : tensor<768xf32>
    %adratbfc1_8 = stablehlo.divide %admhbfc1_8, %addenbfc1_8 : tensor<768xf32>
    %adstbfc1_8 = stablehlo.multiply %adlrbfc1_8, %adratbfc1_8 : tensor<768xf32>
    %adsubbfc1_8 = stablehlo.subtract %bfc1_8, %adstbfc1_8 : tensor<768xf32>
    %adwdbfc1_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrbfc1_8 = stablehlo.multiply %adwdbfc1_8, %adlrbfc1_8 : tensor<768xf32>
    %adwdpbfc1_8 = stablehlo.multiply %adwdlrbfc1_8, %bfc1_8 : tensor<768xf32>
    %adnewbfc1_8 = stablehlo.subtract %adsubbfc1_8, %adwdpbfc1_8 : tensor<768xf32>
    %adb1Wfc2_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob1Wfc2_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admsWfc2_8 = stablehlo.multiply %adb1Wfc2_8, %Wfc2_8m : tensor<768x192xf32>
    %admgWfc2_8 = stablehlo.multiply %adob1Wfc2_8, %vitb8_pdWfc2 : tensor<768x192xf32>
    %admnWfc2_8 = stablehlo.add %admsWfc2_8, %admgWfc2_8 : tensor<768x192xf32>
    %adb2Wfc2_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob2Wfc2_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %advsWfc2_8 = stablehlo.multiply %adb2Wfc2_8, %Wfc2_8v : tensor<768x192xf32>
    %adg2Wfc2_8 = stablehlo.multiply %vitb8_pdWfc2, %vitb8_pdWfc2 : tensor<768x192xf32>
    %advgWfc2_8 = stablehlo.multiply %adob2Wfc2_8, %adg2Wfc2_8 : tensor<768x192xf32>
    %advnWfc2_8 = stablehlo.add %advsWfc2_8, %advgWfc2_8 : tensor<768x192xf32>
    %adbc1Wfc2_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adbc2Wfc2_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admhWfc2_8 = stablehlo.divide %admnWfc2_8, %adbc1Wfc2_8 : tensor<768x192xf32>
    %advhWfc2_8 = stablehlo.divide %advnWfc2_8, %adbc2Wfc2_8 : tensor<768x192xf32>
    %adlrWfc2_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adepsWfc2_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adsqWfc2_8 = stablehlo.sqrt %advhWfc2_8 : tensor<768x192xf32>
    %addenWfc2_8 = stablehlo.add %adsqWfc2_8, %adepsWfc2_8 : tensor<768x192xf32>
    %adratWfc2_8 = stablehlo.divide %admhWfc2_8, %addenWfc2_8 : tensor<768x192xf32>
    %adstWfc2_8 = stablehlo.multiply %adlrWfc2_8, %adratWfc2_8 : tensor<768x192xf32>
    %adsubWfc2_8 = stablehlo.subtract %Wfc2_8, %adstWfc2_8 : tensor<768x192xf32>
    %adwdWfc2_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adwdlrWfc2_8 = stablehlo.multiply %adwdWfc2_8, %adlrWfc2_8 : tensor<768x192xf32>
    %adwdpWfc2_8 = stablehlo.multiply %adwdlrWfc2_8, %Wfc2_8 : tensor<768x192xf32>
    %adnewWfc2_8 = stablehlo.subtract %adsubWfc2_8, %adwdpWfc2_8 : tensor<768x192xf32>
    %adb1bfc2_8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bfc2_8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbfc2_8 = stablehlo.multiply %adb1bfc2_8, %bfc2_8m : tensor<192xf32>
    %admgbfc2_8 = stablehlo.multiply %adob1bfc2_8, %vitb8_pdbfc2 : tensor<192xf32>
    %admnbfc2_8 = stablehlo.add %admsbfc2_8, %admgbfc2_8 : tensor<192xf32>
    %adb2bfc2_8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bfc2_8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbfc2_8 = stablehlo.multiply %adb2bfc2_8, %bfc2_8v : tensor<192xf32>
    %adg2bfc2_8 = stablehlo.multiply %vitb8_pdbfc2, %vitb8_pdbfc2 : tensor<192xf32>
    %advgbfc2_8 = stablehlo.multiply %adob2bfc2_8, %adg2bfc2_8 : tensor<192xf32>
    %advnbfc2_8 = stablehlo.add %advsbfc2_8, %advgbfc2_8 : tensor<192xf32>
    %adbc1bfc2_8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bfc2_8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbfc2_8 = stablehlo.divide %admnbfc2_8, %adbc1bfc2_8 : tensor<192xf32>
    %advhbfc2_8 = stablehlo.divide %advnbfc2_8, %adbc2bfc2_8 : tensor<192xf32>
    %adlrbfc2_8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbfc2_8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbfc2_8 = stablehlo.sqrt %advhbfc2_8 : tensor<192xf32>
    %addenbfc2_8 = stablehlo.add %adsqbfc2_8, %adepsbfc2_8 : tensor<192xf32>
    %adratbfc2_8 = stablehlo.divide %admhbfc2_8, %addenbfc2_8 : tensor<192xf32>
    %adstbfc2_8 = stablehlo.multiply %adlrbfc2_8, %adratbfc2_8 : tensor<192xf32>
    %adsubbfc2_8 = stablehlo.subtract %bfc2_8, %adstbfc2_8 : tensor<192xf32>
    %adwdbfc2_8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbfc2_8 = stablehlo.multiply %adwdbfc2_8, %adlrbfc2_8 : tensor<192xf32>
    %adwdpbfc2_8 = stablehlo.multiply %adwdlrbfc2_8, %bfc2_8 : tensor<192xf32>
    %adnewbfc2_8 = stablehlo.subtract %adsubbfc2_8, %adwdpbfc2_8 : tensor<192xf32>
    %adb1g1_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g1_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg1_9 = stablehlo.multiply %adb1g1_9, %g1_9m : tensor<192xf32>
    %admgg1_9 = stablehlo.multiply %adob1g1_9, %vitb9_1dg : tensor<192xf32>
    %admng1_9 = stablehlo.add %admsg1_9, %admgg1_9 : tensor<192xf32>
    %adb2g1_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g1_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg1_9 = stablehlo.multiply %adb2g1_9, %g1_9v : tensor<192xf32>
    %adg2g1_9 = stablehlo.multiply %vitb9_1dg, %vitb9_1dg : tensor<192xf32>
    %advgg1_9 = stablehlo.multiply %adob2g1_9, %adg2g1_9 : tensor<192xf32>
    %advng1_9 = stablehlo.add %advsg1_9, %advgg1_9 : tensor<192xf32>
    %adbc1g1_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g1_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg1_9 = stablehlo.divide %admng1_9, %adbc1g1_9 : tensor<192xf32>
    %advhg1_9 = stablehlo.divide %advng1_9, %adbc2g1_9 : tensor<192xf32>
    %adlrg1_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg1_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg1_9 = stablehlo.sqrt %advhg1_9 : tensor<192xf32>
    %addeng1_9 = stablehlo.add %adsqg1_9, %adepsg1_9 : tensor<192xf32>
    %adratg1_9 = stablehlo.divide %admhg1_9, %addeng1_9 : tensor<192xf32>
    %adstg1_9 = stablehlo.multiply %adlrg1_9, %adratg1_9 : tensor<192xf32>
    %adsubg1_9 = stablehlo.subtract %g1_9, %adstg1_9 : tensor<192xf32>
    %adwdg1_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg1_9 = stablehlo.multiply %adwdg1_9, %adlrg1_9 : tensor<192xf32>
    %adwdpg1_9 = stablehlo.multiply %adwdlrg1_9, %g1_9 : tensor<192xf32>
    %adnewg1_9 = stablehlo.subtract %adsubg1_9, %adwdpg1_9 : tensor<192xf32>
    %adb1b1_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b1_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb1_9 = stablehlo.multiply %adb1b1_9, %b1_9m : tensor<192xf32>
    %admgb1_9 = stablehlo.multiply %adob1b1_9, %vitb9_1db : tensor<192xf32>
    %admnb1_9 = stablehlo.add %admsb1_9, %admgb1_9 : tensor<192xf32>
    %adb2b1_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b1_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb1_9 = stablehlo.multiply %adb2b1_9, %b1_9v : tensor<192xf32>
    %adg2b1_9 = stablehlo.multiply %vitb9_1db, %vitb9_1db : tensor<192xf32>
    %advgb1_9 = stablehlo.multiply %adob2b1_9, %adg2b1_9 : tensor<192xf32>
    %advnb1_9 = stablehlo.add %advsb1_9, %advgb1_9 : tensor<192xf32>
    %adbc1b1_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b1_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb1_9 = stablehlo.divide %admnb1_9, %adbc1b1_9 : tensor<192xf32>
    %advhb1_9 = stablehlo.divide %advnb1_9, %adbc2b1_9 : tensor<192xf32>
    %adlrb1_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb1_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb1_9 = stablehlo.sqrt %advhb1_9 : tensor<192xf32>
    %addenb1_9 = stablehlo.add %adsqb1_9, %adepsb1_9 : tensor<192xf32>
    %adratb1_9 = stablehlo.divide %admhb1_9, %addenb1_9 : tensor<192xf32>
    %adstb1_9 = stablehlo.multiply %adlrb1_9, %adratb1_9 : tensor<192xf32>
    %adsubb1_9 = stablehlo.subtract %b1_9, %adstb1_9 : tensor<192xf32>
    %adwdb1_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb1_9 = stablehlo.multiply %adwdb1_9, %adlrb1_9 : tensor<192xf32>
    %adwdpb1_9 = stablehlo.multiply %adwdlrb1_9, %b1_9 : tensor<192xf32>
    %adnewb1_9 = stablehlo.subtract %adsubb1_9, %adwdpb1_9 : tensor<192xf32>
    %adb1Wq_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wq_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWq_9 = stablehlo.multiply %adb1Wq_9, %Wq_9m : tensor<192x192xf32>
    %admgWq_9 = stablehlo.multiply %adob1Wq_9, %vitb9_mdWQ : tensor<192x192xf32>
    %admnWq_9 = stablehlo.add %admsWq_9, %admgWq_9 : tensor<192x192xf32>
    %adb2Wq_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wq_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWq_9 = stablehlo.multiply %adb2Wq_9, %Wq_9v : tensor<192x192xf32>
    %adg2Wq_9 = stablehlo.multiply %vitb9_mdWQ, %vitb9_mdWQ : tensor<192x192xf32>
    %advgWq_9 = stablehlo.multiply %adob2Wq_9, %adg2Wq_9 : tensor<192x192xf32>
    %advnWq_9 = stablehlo.add %advsWq_9, %advgWq_9 : tensor<192x192xf32>
    %adbc1Wq_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wq_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWq_9 = stablehlo.divide %admnWq_9, %adbc1Wq_9 : tensor<192x192xf32>
    %advhWq_9 = stablehlo.divide %advnWq_9, %adbc2Wq_9 : tensor<192x192xf32>
    %adlrWq_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWq_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWq_9 = stablehlo.sqrt %advhWq_9 : tensor<192x192xf32>
    %addenWq_9 = stablehlo.add %adsqWq_9, %adepsWq_9 : tensor<192x192xf32>
    %adratWq_9 = stablehlo.divide %admhWq_9, %addenWq_9 : tensor<192x192xf32>
    %adstWq_9 = stablehlo.multiply %adlrWq_9, %adratWq_9 : tensor<192x192xf32>
    %adsubWq_9 = stablehlo.subtract %Wq_9, %adstWq_9 : tensor<192x192xf32>
    %adwdWq_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWq_9 = stablehlo.multiply %adwdWq_9, %adlrWq_9 : tensor<192x192xf32>
    %adwdpWq_9 = stablehlo.multiply %adwdlrWq_9, %Wq_9 : tensor<192x192xf32>
    %adnewWq_9 = stablehlo.subtract %adsubWq_9, %adwdpWq_9 : tensor<192x192xf32>
    %adb1bq_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bq_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbq_9 = stablehlo.multiply %adb1bq_9, %bq_9m : tensor<192xf32>
    %admgbq_9 = stablehlo.multiply %adob1bq_9, %vitb9_mdbQ : tensor<192xf32>
    %admnbq_9 = stablehlo.add %admsbq_9, %admgbq_9 : tensor<192xf32>
    %adb2bq_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bq_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbq_9 = stablehlo.multiply %adb2bq_9, %bq_9v : tensor<192xf32>
    %adg2bq_9 = stablehlo.multiply %vitb9_mdbQ, %vitb9_mdbQ : tensor<192xf32>
    %advgbq_9 = stablehlo.multiply %adob2bq_9, %adg2bq_9 : tensor<192xf32>
    %advnbq_9 = stablehlo.add %advsbq_9, %advgbq_9 : tensor<192xf32>
    %adbc1bq_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bq_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbq_9 = stablehlo.divide %admnbq_9, %adbc1bq_9 : tensor<192xf32>
    %advhbq_9 = stablehlo.divide %advnbq_9, %adbc2bq_9 : tensor<192xf32>
    %adlrbq_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbq_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbq_9 = stablehlo.sqrt %advhbq_9 : tensor<192xf32>
    %addenbq_9 = stablehlo.add %adsqbq_9, %adepsbq_9 : tensor<192xf32>
    %adratbq_9 = stablehlo.divide %admhbq_9, %addenbq_9 : tensor<192xf32>
    %adstbq_9 = stablehlo.multiply %adlrbq_9, %adratbq_9 : tensor<192xf32>
    %adsubbq_9 = stablehlo.subtract %bq_9, %adstbq_9 : tensor<192xf32>
    %adwdbq_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbq_9 = stablehlo.multiply %adwdbq_9, %adlrbq_9 : tensor<192xf32>
    %adwdpbq_9 = stablehlo.multiply %adwdlrbq_9, %bq_9 : tensor<192xf32>
    %adnewbq_9 = stablehlo.subtract %adsubbq_9, %adwdpbq_9 : tensor<192xf32>
    %adb1Wk_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wk_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWk_9 = stablehlo.multiply %adb1Wk_9, %Wk_9m : tensor<192x192xf32>
    %admgWk_9 = stablehlo.multiply %adob1Wk_9, %vitb9_mdWK : tensor<192x192xf32>
    %admnWk_9 = stablehlo.add %admsWk_9, %admgWk_9 : tensor<192x192xf32>
    %adb2Wk_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wk_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWk_9 = stablehlo.multiply %adb2Wk_9, %Wk_9v : tensor<192x192xf32>
    %adg2Wk_9 = stablehlo.multiply %vitb9_mdWK, %vitb9_mdWK : tensor<192x192xf32>
    %advgWk_9 = stablehlo.multiply %adob2Wk_9, %adg2Wk_9 : tensor<192x192xf32>
    %advnWk_9 = stablehlo.add %advsWk_9, %advgWk_9 : tensor<192x192xf32>
    %adbc1Wk_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wk_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWk_9 = stablehlo.divide %admnWk_9, %adbc1Wk_9 : tensor<192x192xf32>
    %advhWk_9 = stablehlo.divide %advnWk_9, %adbc2Wk_9 : tensor<192x192xf32>
    %adlrWk_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWk_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWk_9 = stablehlo.sqrt %advhWk_9 : tensor<192x192xf32>
    %addenWk_9 = stablehlo.add %adsqWk_9, %adepsWk_9 : tensor<192x192xf32>
    %adratWk_9 = stablehlo.divide %admhWk_9, %addenWk_9 : tensor<192x192xf32>
    %adstWk_9 = stablehlo.multiply %adlrWk_9, %adratWk_9 : tensor<192x192xf32>
    %adsubWk_9 = stablehlo.subtract %Wk_9, %adstWk_9 : tensor<192x192xf32>
    %adwdWk_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWk_9 = stablehlo.multiply %adwdWk_9, %adlrWk_9 : tensor<192x192xf32>
    %adwdpWk_9 = stablehlo.multiply %adwdlrWk_9, %Wk_9 : tensor<192x192xf32>
    %adnewWk_9 = stablehlo.subtract %adsubWk_9, %adwdpWk_9 : tensor<192x192xf32>
    %adb1bk_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bk_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbk_9 = stablehlo.multiply %adb1bk_9, %bk_9m : tensor<192xf32>
    %admgbk_9 = stablehlo.multiply %adob1bk_9, %vitb9_mdbK : tensor<192xf32>
    %admnbk_9 = stablehlo.add %admsbk_9, %admgbk_9 : tensor<192xf32>
    %adb2bk_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bk_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbk_9 = stablehlo.multiply %adb2bk_9, %bk_9v : tensor<192xf32>
    %adg2bk_9 = stablehlo.multiply %vitb9_mdbK, %vitb9_mdbK : tensor<192xf32>
    %advgbk_9 = stablehlo.multiply %adob2bk_9, %adg2bk_9 : tensor<192xf32>
    %advnbk_9 = stablehlo.add %advsbk_9, %advgbk_9 : tensor<192xf32>
    %adbc1bk_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bk_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbk_9 = stablehlo.divide %admnbk_9, %adbc1bk_9 : tensor<192xf32>
    %advhbk_9 = stablehlo.divide %advnbk_9, %adbc2bk_9 : tensor<192xf32>
    %adlrbk_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbk_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbk_9 = stablehlo.sqrt %advhbk_9 : tensor<192xf32>
    %addenbk_9 = stablehlo.add %adsqbk_9, %adepsbk_9 : tensor<192xf32>
    %adratbk_9 = stablehlo.divide %admhbk_9, %addenbk_9 : tensor<192xf32>
    %adstbk_9 = stablehlo.multiply %adlrbk_9, %adratbk_9 : tensor<192xf32>
    %adsubbk_9 = stablehlo.subtract %bk_9, %adstbk_9 : tensor<192xf32>
    %adwdbk_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbk_9 = stablehlo.multiply %adwdbk_9, %adlrbk_9 : tensor<192xf32>
    %adwdpbk_9 = stablehlo.multiply %adwdlrbk_9, %bk_9 : tensor<192xf32>
    %adnewbk_9 = stablehlo.subtract %adsubbk_9, %adwdpbk_9 : tensor<192xf32>
    %adb1Wv_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wv_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWv_9 = stablehlo.multiply %adb1Wv_9, %Wv_9m : tensor<192x192xf32>
    %admgWv_9 = stablehlo.multiply %adob1Wv_9, %vitb9_mdWV : tensor<192x192xf32>
    %admnWv_9 = stablehlo.add %admsWv_9, %admgWv_9 : tensor<192x192xf32>
    %adb2Wv_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wv_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWv_9 = stablehlo.multiply %adb2Wv_9, %Wv_9v : tensor<192x192xf32>
    %adg2Wv_9 = stablehlo.multiply %vitb9_mdWV, %vitb9_mdWV : tensor<192x192xf32>
    %advgWv_9 = stablehlo.multiply %adob2Wv_9, %adg2Wv_9 : tensor<192x192xf32>
    %advnWv_9 = stablehlo.add %advsWv_9, %advgWv_9 : tensor<192x192xf32>
    %adbc1Wv_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wv_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWv_9 = stablehlo.divide %admnWv_9, %adbc1Wv_9 : tensor<192x192xf32>
    %advhWv_9 = stablehlo.divide %advnWv_9, %adbc2Wv_9 : tensor<192x192xf32>
    %adlrWv_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWv_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWv_9 = stablehlo.sqrt %advhWv_9 : tensor<192x192xf32>
    %addenWv_9 = stablehlo.add %adsqWv_9, %adepsWv_9 : tensor<192x192xf32>
    %adratWv_9 = stablehlo.divide %admhWv_9, %addenWv_9 : tensor<192x192xf32>
    %adstWv_9 = stablehlo.multiply %adlrWv_9, %adratWv_9 : tensor<192x192xf32>
    %adsubWv_9 = stablehlo.subtract %Wv_9, %adstWv_9 : tensor<192x192xf32>
    %adwdWv_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWv_9 = stablehlo.multiply %adwdWv_9, %adlrWv_9 : tensor<192x192xf32>
    %adwdpWv_9 = stablehlo.multiply %adwdlrWv_9, %Wv_9 : tensor<192x192xf32>
    %adnewWv_9 = stablehlo.subtract %adsubWv_9, %adwdpWv_9 : tensor<192x192xf32>
    %adb1bv_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bv_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbv_9 = stablehlo.multiply %adb1bv_9, %bv_9m : tensor<192xf32>
    %admgbv_9 = stablehlo.multiply %adob1bv_9, %vitb9_mdbV : tensor<192xf32>
    %admnbv_9 = stablehlo.add %admsbv_9, %admgbv_9 : tensor<192xf32>
    %adb2bv_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bv_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbv_9 = stablehlo.multiply %adb2bv_9, %bv_9v : tensor<192xf32>
    %adg2bv_9 = stablehlo.multiply %vitb9_mdbV, %vitb9_mdbV : tensor<192xf32>
    %advgbv_9 = stablehlo.multiply %adob2bv_9, %adg2bv_9 : tensor<192xf32>
    %advnbv_9 = stablehlo.add %advsbv_9, %advgbv_9 : tensor<192xf32>
    %adbc1bv_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bv_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbv_9 = stablehlo.divide %admnbv_9, %adbc1bv_9 : tensor<192xf32>
    %advhbv_9 = stablehlo.divide %advnbv_9, %adbc2bv_9 : tensor<192xf32>
    %adlrbv_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbv_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbv_9 = stablehlo.sqrt %advhbv_9 : tensor<192xf32>
    %addenbv_9 = stablehlo.add %adsqbv_9, %adepsbv_9 : tensor<192xf32>
    %adratbv_9 = stablehlo.divide %admhbv_9, %addenbv_9 : tensor<192xf32>
    %adstbv_9 = stablehlo.multiply %adlrbv_9, %adratbv_9 : tensor<192xf32>
    %adsubbv_9 = stablehlo.subtract %bv_9, %adstbv_9 : tensor<192xf32>
    %adwdbv_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbv_9 = stablehlo.multiply %adwdbv_9, %adlrbv_9 : tensor<192xf32>
    %adwdpbv_9 = stablehlo.multiply %adwdlrbv_9, %bv_9 : tensor<192xf32>
    %adnewbv_9 = stablehlo.subtract %adsubbv_9, %adwdpbv_9 : tensor<192xf32>
    %adb1Wo_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wo_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWo_9 = stablehlo.multiply %adb1Wo_9, %Wo_9m : tensor<192x192xf32>
    %admgWo_9 = stablehlo.multiply %adob1Wo_9, %vitb9_mdWo : tensor<192x192xf32>
    %admnWo_9 = stablehlo.add %admsWo_9, %admgWo_9 : tensor<192x192xf32>
    %adb2Wo_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wo_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWo_9 = stablehlo.multiply %adb2Wo_9, %Wo_9v : tensor<192x192xf32>
    %adg2Wo_9 = stablehlo.multiply %vitb9_mdWo, %vitb9_mdWo : tensor<192x192xf32>
    %advgWo_9 = stablehlo.multiply %adob2Wo_9, %adg2Wo_9 : tensor<192x192xf32>
    %advnWo_9 = stablehlo.add %advsWo_9, %advgWo_9 : tensor<192x192xf32>
    %adbc1Wo_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wo_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWo_9 = stablehlo.divide %admnWo_9, %adbc1Wo_9 : tensor<192x192xf32>
    %advhWo_9 = stablehlo.divide %advnWo_9, %adbc2Wo_9 : tensor<192x192xf32>
    %adlrWo_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWo_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWo_9 = stablehlo.sqrt %advhWo_9 : tensor<192x192xf32>
    %addenWo_9 = stablehlo.add %adsqWo_9, %adepsWo_9 : tensor<192x192xf32>
    %adratWo_9 = stablehlo.divide %admhWo_9, %addenWo_9 : tensor<192x192xf32>
    %adstWo_9 = stablehlo.multiply %adlrWo_9, %adratWo_9 : tensor<192x192xf32>
    %adsubWo_9 = stablehlo.subtract %Wo_9, %adstWo_9 : tensor<192x192xf32>
    %adwdWo_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWo_9 = stablehlo.multiply %adwdWo_9, %adlrWo_9 : tensor<192x192xf32>
    %adwdpWo_9 = stablehlo.multiply %adwdlrWo_9, %Wo_9 : tensor<192x192xf32>
    %adnewWo_9 = stablehlo.subtract %adsubWo_9, %adwdpWo_9 : tensor<192x192xf32>
    %adb1bo_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bo_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbo_9 = stablehlo.multiply %adb1bo_9, %bo_9m : tensor<192xf32>
    %admgbo_9 = stablehlo.multiply %adob1bo_9, %vitb9_mdbo : tensor<192xf32>
    %admnbo_9 = stablehlo.add %admsbo_9, %admgbo_9 : tensor<192xf32>
    %adb2bo_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bo_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbo_9 = stablehlo.multiply %adb2bo_9, %bo_9v : tensor<192xf32>
    %adg2bo_9 = stablehlo.multiply %vitb9_mdbo, %vitb9_mdbo : tensor<192xf32>
    %advgbo_9 = stablehlo.multiply %adob2bo_9, %adg2bo_9 : tensor<192xf32>
    %advnbo_9 = stablehlo.add %advsbo_9, %advgbo_9 : tensor<192xf32>
    %adbc1bo_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bo_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbo_9 = stablehlo.divide %admnbo_9, %adbc1bo_9 : tensor<192xf32>
    %advhbo_9 = stablehlo.divide %advnbo_9, %adbc2bo_9 : tensor<192xf32>
    %adlrbo_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbo_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbo_9 = stablehlo.sqrt %advhbo_9 : tensor<192xf32>
    %addenbo_9 = stablehlo.add %adsqbo_9, %adepsbo_9 : tensor<192xf32>
    %adratbo_9 = stablehlo.divide %admhbo_9, %addenbo_9 : tensor<192xf32>
    %adstbo_9 = stablehlo.multiply %adlrbo_9, %adratbo_9 : tensor<192xf32>
    %adsubbo_9 = stablehlo.subtract %bo_9, %adstbo_9 : tensor<192xf32>
    %adwdbo_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbo_9 = stablehlo.multiply %adwdbo_9, %adlrbo_9 : tensor<192xf32>
    %adwdpbo_9 = stablehlo.multiply %adwdlrbo_9, %bo_9 : tensor<192xf32>
    %adnewbo_9 = stablehlo.subtract %adsubbo_9, %adwdpbo_9 : tensor<192xf32>
    %adb1g2_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g2_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg2_9 = stablehlo.multiply %adb1g2_9, %g2_9m : tensor<192xf32>
    %admgg2_9 = stablehlo.multiply %adob1g2_9, %vitb9_2dg : tensor<192xf32>
    %admng2_9 = stablehlo.add %admsg2_9, %admgg2_9 : tensor<192xf32>
    %adb2g2_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g2_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg2_9 = stablehlo.multiply %adb2g2_9, %g2_9v : tensor<192xf32>
    %adg2g2_9 = stablehlo.multiply %vitb9_2dg, %vitb9_2dg : tensor<192xf32>
    %advgg2_9 = stablehlo.multiply %adob2g2_9, %adg2g2_9 : tensor<192xf32>
    %advng2_9 = stablehlo.add %advsg2_9, %advgg2_9 : tensor<192xf32>
    %adbc1g2_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g2_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg2_9 = stablehlo.divide %admng2_9, %adbc1g2_9 : tensor<192xf32>
    %advhg2_9 = stablehlo.divide %advng2_9, %adbc2g2_9 : tensor<192xf32>
    %adlrg2_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg2_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg2_9 = stablehlo.sqrt %advhg2_9 : tensor<192xf32>
    %addeng2_9 = stablehlo.add %adsqg2_9, %adepsg2_9 : tensor<192xf32>
    %adratg2_9 = stablehlo.divide %admhg2_9, %addeng2_9 : tensor<192xf32>
    %adstg2_9 = stablehlo.multiply %adlrg2_9, %adratg2_9 : tensor<192xf32>
    %adsubg2_9 = stablehlo.subtract %g2_9, %adstg2_9 : tensor<192xf32>
    %adwdg2_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg2_9 = stablehlo.multiply %adwdg2_9, %adlrg2_9 : tensor<192xf32>
    %adwdpg2_9 = stablehlo.multiply %adwdlrg2_9, %g2_9 : tensor<192xf32>
    %adnewg2_9 = stablehlo.subtract %adsubg2_9, %adwdpg2_9 : tensor<192xf32>
    %adb1b2_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b2_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb2_9 = stablehlo.multiply %adb1b2_9, %b2_9m : tensor<192xf32>
    %admgb2_9 = stablehlo.multiply %adob1b2_9, %vitb9_2db : tensor<192xf32>
    %admnb2_9 = stablehlo.add %admsb2_9, %admgb2_9 : tensor<192xf32>
    %adb2b2_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b2_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb2_9 = stablehlo.multiply %adb2b2_9, %b2_9v : tensor<192xf32>
    %adg2b2_9 = stablehlo.multiply %vitb9_2db, %vitb9_2db : tensor<192xf32>
    %advgb2_9 = stablehlo.multiply %adob2b2_9, %adg2b2_9 : tensor<192xf32>
    %advnb2_9 = stablehlo.add %advsb2_9, %advgb2_9 : tensor<192xf32>
    %adbc1b2_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b2_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb2_9 = stablehlo.divide %admnb2_9, %adbc1b2_9 : tensor<192xf32>
    %advhb2_9 = stablehlo.divide %advnb2_9, %adbc2b2_9 : tensor<192xf32>
    %adlrb2_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb2_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb2_9 = stablehlo.sqrt %advhb2_9 : tensor<192xf32>
    %addenb2_9 = stablehlo.add %adsqb2_9, %adepsb2_9 : tensor<192xf32>
    %adratb2_9 = stablehlo.divide %admhb2_9, %addenb2_9 : tensor<192xf32>
    %adstb2_9 = stablehlo.multiply %adlrb2_9, %adratb2_9 : tensor<192xf32>
    %adsubb2_9 = stablehlo.subtract %b2_9, %adstb2_9 : tensor<192xf32>
    %adwdb2_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb2_9 = stablehlo.multiply %adwdb2_9, %adlrb2_9 : tensor<192xf32>
    %adwdpb2_9 = stablehlo.multiply %adwdlrb2_9, %b2_9 : tensor<192xf32>
    %adnewb2_9 = stablehlo.subtract %adsubb2_9, %adwdpb2_9 : tensor<192xf32>
    %adb1Wfc1_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob1Wfc1_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admsWfc1_9 = stablehlo.multiply %adb1Wfc1_9, %Wfc1_9m : tensor<192x768xf32>
    %admgWfc1_9 = stablehlo.multiply %adob1Wfc1_9, %vitb9_pdWfc1 : tensor<192x768xf32>
    %admnWfc1_9 = stablehlo.add %admsWfc1_9, %admgWfc1_9 : tensor<192x768xf32>
    %adb2Wfc1_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob2Wfc1_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %advsWfc1_9 = stablehlo.multiply %adb2Wfc1_9, %Wfc1_9v : tensor<192x768xf32>
    %adg2Wfc1_9 = stablehlo.multiply %vitb9_pdWfc1, %vitb9_pdWfc1 : tensor<192x768xf32>
    %advgWfc1_9 = stablehlo.multiply %adob2Wfc1_9, %adg2Wfc1_9 : tensor<192x768xf32>
    %advnWfc1_9 = stablehlo.add %advsWfc1_9, %advgWfc1_9 : tensor<192x768xf32>
    %adbc1Wfc1_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adbc2Wfc1_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admhWfc1_9 = stablehlo.divide %admnWfc1_9, %adbc1Wfc1_9 : tensor<192x768xf32>
    %advhWfc1_9 = stablehlo.divide %advnWfc1_9, %adbc2Wfc1_9 : tensor<192x768xf32>
    %adlrWfc1_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adepsWfc1_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adsqWfc1_9 = stablehlo.sqrt %advhWfc1_9 : tensor<192x768xf32>
    %addenWfc1_9 = stablehlo.add %adsqWfc1_9, %adepsWfc1_9 : tensor<192x768xf32>
    %adratWfc1_9 = stablehlo.divide %admhWfc1_9, %addenWfc1_9 : tensor<192x768xf32>
    %adstWfc1_9 = stablehlo.multiply %adlrWfc1_9, %adratWfc1_9 : tensor<192x768xf32>
    %adsubWfc1_9 = stablehlo.subtract %Wfc1_9, %adstWfc1_9 : tensor<192x768xf32>
    %adwdWfc1_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adwdlrWfc1_9 = stablehlo.multiply %adwdWfc1_9, %adlrWfc1_9 : tensor<192x768xf32>
    %adwdpWfc1_9 = stablehlo.multiply %adwdlrWfc1_9, %Wfc1_9 : tensor<192x768xf32>
    %adnewWfc1_9 = stablehlo.subtract %adsubWfc1_9, %adwdpWfc1_9 : tensor<192x768xf32>
    %adb1bfc1_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1bfc1_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admsbfc1_9 = stablehlo.multiply %adb1bfc1_9, %bfc1_9m : tensor<768xf32>
    %admgbfc1_9 = stablehlo.multiply %adob1bfc1_9, %vitb9_pdbfc1 : tensor<768xf32>
    %admnbfc1_9 = stablehlo.add %admsbfc1_9, %admgbfc1_9 : tensor<768xf32>
    %adb2bfc1_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2bfc1_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advsbfc1_9 = stablehlo.multiply %adb2bfc1_9, %bfc1_9v : tensor<768xf32>
    %adg2bfc1_9 = stablehlo.multiply %vitb9_pdbfc1, %vitb9_pdbfc1 : tensor<768xf32>
    %advgbfc1_9 = stablehlo.multiply %adob2bfc1_9, %adg2bfc1_9 : tensor<768xf32>
    %advnbfc1_9 = stablehlo.add %advsbfc1_9, %advgbfc1_9 : tensor<768xf32>
    %adbc1bfc1_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2bfc1_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhbfc1_9 = stablehlo.divide %admnbfc1_9, %adbc1bfc1_9 : tensor<768xf32>
    %advhbfc1_9 = stablehlo.divide %advnbfc1_9, %adbc2bfc1_9 : tensor<768xf32>
    %adlrbfc1_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepsbfc1_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqbfc1_9 = stablehlo.sqrt %advhbfc1_9 : tensor<768xf32>
    %addenbfc1_9 = stablehlo.add %adsqbfc1_9, %adepsbfc1_9 : tensor<768xf32>
    %adratbfc1_9 = stablehlo.divide %admhbfc1_9, %addenbfc1_9 : tensor<768xf32>
    %adstbfc1_9 = stablehlo.multiply %adlrbfc1_9, %adratbfc1_9 : tensor<768xf32>
    %adsubbfc1_9 = stablehlo.subtract %bfc1_9, %adstbfc1_9 : tensor<768xf32>
    %adwdbfc1_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrbfc1_9 = stablehlo.multiply %adwdbfc1_9, %adlrbfc1_9 : tensor<768xf32>
    %adwdpbfc1_9 = stablehlo.multiply %adwdlrbfc1_9, %bfc1_9 : tensor<768xf32>
    %adnewbfc1_9 = stablehlo.subtract %adsubbfc1_9, %adwdpbfc1_9 : tensor<768xf32>
    %adb1Wfc2_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob1Wfc2_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admsWfc2_9 = stablehlo.multiply %adb1Wfc2_9, %Wfc2_9m : tensor<768x192xf32>
    %admgWfc2_9 = stablehlo.multiply %adob1Wfc2_9, %vitb9_pdWfc2 : tensor<768x192xf32>
    %admnWfc2_9 = stablehlo.add %admsWfc2_9, %admgWfc2_9 : tensor<768x192xf32>
    %adb2Wfc2_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob2Wfc2_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %advsWfc2_9 = stablehlo.multiply %adb2Wfc2_9, %Wfc2_9v : tensor<768x192xf32>
    %adg2Wfc2_9 = stablehlo.multiply %vitb9_pdWfc2, %vitb9_pdWfc2 : tensor<768x192xf32>
    %advgWfc2_9 = stablehlo.multiply %adob2Wfc2_9, %adg2Wfc2_9 : tensor<768x192xf32>
    %advnWfc2_9 = stablehlo.add %advsWfc2_9, %advgWfc2_9 : tensor<768x192xf32>
    %adbc1Wfc2_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adbc2Wfc2_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admhWfc2_9 = stablehlo.divide %admnWfc2_9, %adbc1Wfc2_9 : tensor<768x192xf32>
    %advhWfc2_9 = stablehlo.divide %advnWfc2_9, %adbc2Wfc2_9 : tensor<768x192xf32>
    %adlrWfc2_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adepsWfc2_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adsqWfc2_9 = stablehlo.sqrt %advhWfc2_9 : tensor<768x192xf32>
    %addenWfc2_9 = stablehlo.add %adsqWfc2_9, %adepsWfc2_9 : tensor<768x192xf32>
    %adratWfc2_9 = stablehlo.divide %admhWfc2_9, %addenWfc2_9 : tensor<768x192xf32>
    %adstWfc2_9 = stablehlo.multiply %adlrWfc2_9, %adratWfc2_9 : tensor<768x192xf32>
    %adsubWfc2_9 = stablehlo.subtract %Wfc2_9, %adstWfc2_9 : tensor<768x192xf32>
    %adwdWfc2_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adwdlrWfc2_9 = stablehlo.multiply %adwdWfc2_9, %adlrWfc2_9 : tensor<768x192xf32>
    %adwdpWfc2_9 = stablehlo.multiply %adwdlrWfc2_9, %Wfc2_9 : tensor<768x192xf32>
    %adnewWfc2_9 = stablehlo.subtract %adsubWfc2_9, %adwdpWfc2_9 : tensor<768x192xf32>
    %adb1bfc2_9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bfc2_9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbfc2_9 = stablehlo.multiply %adb1bfc2_9, %bfc2_9m : tensor<192xf32>
    %admgbfc2_9 = stablehlo.multiply %adob1bfc2_9, %vitb9_pdbfc2 : tensor<192xf32>
    %admnbfc2_9 = stablehlo.add %admsbfc2_9, %admgbfc2_9 : tensor<192xf32>
    %adb2bfc2_9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bfc2_9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbfc2_9 = stablehlo.multiply %adb2bfc2_9, %bfc2_9v : tensor<192xf32>
    %adg2bfc2_9 = stablehlo.multiply %vitb9_pdbfc2, %vitb9_pdbfc2 : tensor<192xf32>
    %advgbfc2_9 = stablehlo.multiply %adob2bfc2_9, %adg2bfc2_9 : tensor<192xf32>
    %advnbfc2_9 = stablehlo.add %advsbfc2_9, %advgbfc2_9 : tensor<192xf32>
    %adbc1bfc2_9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bfc2_9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbfc2_9 = stablehlo.divide %admnbfc2_9, %adbc1bfc2_9 : tensor<192xf32>
    %advhbfc2_9 = stablehlo.divide %advnbfc2_9, %adbc2bfc2_9 : tensor<192xf32>
    %adlrbfc2_9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbfc2_9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbfc2_9 = stablehlo.sqrt %advhbfc2_9 : tensor<192xf32>
    %addenbfc2_9 = stablehlo.add %adsqbfc2_9, %adepsbfc2_9 : tensor<192xf32>
    %adratbfc2_9 = stablehlo.divide %admhbfc2_9, %addenbfc2_9 : tensor<192xf32>
    %adstbfc2_9 = stablehlo.multiply %adlrbfc2_9, %adratbfc2_9 : tensor<192xf32>
    %adsubbfc2_9 = stablehlo.subtract %bfc2_9, %adstbfc2_9 : tensor<192xf32>
    %adwdbfc2_9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbfc2_9 = stablehlo.multiply %adwdbfc2_9, %adlrbfc2_9 : tensor<192xf32>
    %adwdpbfc2_9 = stablehlo.multiply %adwdlrbfc2_9, %bfc2_9 : tensor<192xf32>
    %adnewbfc2_9 = stablehlo.subtract %adsubbfc2_9, %adwdpbfc2_9 : tensor<192xf32>
    %adb1g1_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g1_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg1_10 = stablehlo.multiply %adb1g1_10, %g1_10m : tensor<192xf32>
    %admgg1_10 = stablehlo.multiply %adob1g1_10, %vitb10_1dg : tensor<192xf32>
    %admng1_10 = stablehlo.add %admsg1_10, %admgg1_10 : tensor<192xf32>
    %adb2g1_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g1_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg1_10 = stablehlo.multiply %adb2g1_10, %g1_10v : tensor<192xf32>
    %adg2g1_10 = stablehlo.multiply %vitb10_1dg, %vitb10_1dg : tensor<192xf32>
    %advgg1_10 = stablehlo.multiply %adob2g1_10, %adg2g1_10 : tensor<192xf32>
    %advng1_10 = stablehlo.add %advsg1_10, %advgg1_10 : tensor<192xf32>
    %adbc1g1_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g1_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg1_10 = stablehlo.divide %admng1_10, %adbc1g1_10 : tensor<192xf32>
    %advhg1_10 = stablehlo.divide %advng1_10, %adbc2g1_10 : tensor<192xf32>
    %adlrg1_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg1_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg1_10 = stablehlo.sqrt %advhg1_10 : tensor<192xf32>
    %addeng1_10 = stablehlo.add %adsqg1_10, %adepsg1_10 : tensor<192xf32>
    %adratg1_10 = stablehlo.divide %admhg1_10, %addeng1_10 : tensor<192xf32>
    %adstg1_10 = stablehlo.multiply %adlrg1_10, %adratg1_10 : tensor<192xf32>
    %adsubg1_10 = stablehlo.subtract %g1_10, %adstg1_10 : tensor<192xf32>
    %adwdg1_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg1_10 = stablehlo.multiply %adwdg1_10, %adlrg1_10 : tensor<192xf32>
    %adwdpg1_10 = stablehlo.multiply %adwdlrg1_10, %g1_10 : tensor<192xf32>
    %adnewg1_10 = stablehlo.subtract %adsubg1_10, %adwdpg1_10 : tensor<192xf32>
    %adb1b1_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b1_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb1_10 = stablehlo.multiply %adb1b1_10, %b1_10m : tensor<192xf32>
    %admgb1_10 = stablehlo.multiply %adob1b1_10, %vitb10_1db : tensor<192xf32>
    %admnb1_10 = stablehlo.add %admsb1_10, %admgb1_10 : tensor<192xf32>
    %adb2b1_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b1_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb1_10 = stablehlo.multiply %adb2b1_10, %b1_10v : tensor<192xf32>
    %adg2b1_10 = stablehlo.multiply %vitb10_1db, %vitb10_1db : tensor<192xf32>
    %advgb1_10 = stablehlo.multiply %adob2b1_10, %adg2b1_10 : tensor<192xf32>
    %advnb1_10 = stablehlo.add %advsb1_10, %advgb1_10 : tensor<192xf32>
    %adbc1b1_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b1_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb1_10 = stablehlo.divide %admnb1_10, %adbc1b1_10 : tensor<192xf32>
    %advhb1_10 = stablehlo.divide %advnb1_10, %adbc2b1_10 : tensor<192xf32>
    %adlrb1_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb1_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb1_10 = stablehlo.sqrt %advhb1_10 : tensor<192xf32>
    %addenb1_10 = stablehlo.add %adsqb1_10, %adepsb1_10 : tensor<192xf32>
    %adratb1_10 = stablehlo.divide %admhb1_10, %addenb1_10 : tensor<192xf32>
    %adstb1_10 = stablehlo.multiply %adlrb1_10, %adratb1_10 : tensor<192xf32>
    %adsubb1_10 = stablehlo.subtract %b1_10, %adstb1_10 : tensor<192xf32>
    %adwdb1_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb1_10 = stablehlo.multiply %adwdb1_10, %adlrb1_10 : tensor<192xf32>
    %adwdpb1_10 = stablehlo.multiply %adwdlrb1_10, %b1_10 : tensor<192xf32>
    %adnewb1_10 = stablehlo.subtract %adsubb1_10, %adwdpb1_10 : tensor<192xf32>
    %adb1Wq_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wq_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWq_10 = stablehlo.multiply %adb1Wq_10, %Wq_10m : tensor<192x192xf32>
    %admgWq_10 = stablehlo.multiply %adob1Wq_10, %vitb10_mdWQ : tensor<192x192xf32>
    %admnWq_10 = stablehlo.add %admsWq_10, %admgWq_10 : tensor<192x192xf32>
    %adb2Wq_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wq_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWq_10 = stablehlo.multiply %adb2Wq_10, %Wq_10v : tensor<192x192xf32>
    %adg2Wq_10 = stablehlo.multiply %vitb10_mdWQ, %vitb10_mdWQ : tensor<192x192xf32>
    %advgWq_10 = stablehlo.multiply %adob2Wq_10, %adg2Wq_10 : tensor<192x192xf32>
    %advnWq_10 = stablehlo.add %advsWq_10, %advgWq_10 : tensor<192x192xf32>
    %adbc1Wq_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wq_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWq_10 = stablehlo.divide %admnWq_10, %adbc1Wq_10 : tensor<192x192xf32>
    %advhWq_10 = stablehlo.divide %advnWq_10, %adbc2Wq_10 : tensor<192x192xf32>
    %adlrWq_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWq_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWq_10 = stablehlo.sqrt %advhWq_10 : tensor<192x192xf32>
    %addenWq_10 = stablehlo.add %adsqWq_10, %adepsWq_10 : tensor<192x192xf32>
    %adratWq_10 = stablehlo.divide %admhWq_10, %addenWq_10 : tensor<192x192xf32>
    %adstWq_10 = stablehlo.multiply %adlrWq_10, %adratWq_10 : tensor<192x192xf32>
    %adsubWq_10 = stablehlo.subtract %Wq_10, %adstWq_10 : tensor<192x192xf32>
    %adwdWq_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWq_10 = stablehlo.multiply %adwdWq_10, %adlrWq_10 : tensor<192x192xf32>
    %adwdpWq_10 = stablehlo.multiply %adwdlrWq_10, %Wq_10 : tensor<192x192xf32>
    %adnewWq_10 = stablehlo.subtract %adsubWq_10, %adwdpWq_10 : tensor<192x192xf32>
    %adb1bq_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bq_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbq_10 = stablehlo.multiply %adb1bq_10, %bq_10m : tensor<192xf32>
    %admgbq_10 = stablehlo.multiply %adob1bq_10, %vitb10_mdbQ : tensor<192xf32>
    %admnbq_10 = stablehlo.add %admsbq_10, %admgbq_10 : tensor<192xf32>
    %adb2bq_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bq_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbq_10 = stablehlo.multiply %adb2bq_10, %bq_10v : tensor<192xf32>
    %adg2bq_10 = stablehlo.multiply %vitb10_mdbQ, %vitb10_mdbQ : tensor<192xf32>
    %advgbq_10 = stablehlo.multiply %adob2bq_10, %adg2bq_10 : tensor<192xf32>
    %advnbq_10 = stablehlo.add %advsbq_10, %advgbq_10 : tensor<192xf32>
    %adbc1bq_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bq_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbq_10 = stablehlo.divide %admnbq_10, %adbc1bq_10 : tensor<192xf32>
    %advhbq_10 = stablehlo.divide %advnbq_10, %adbc2bq_10 : tensor<192xf32>
    %adlrbq_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbq_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbq_10 = stablehlo.sqrt %advhbq_10 : tensor<192xf32>
    %addenbq_10 = stablehlo.add %adsqbq_10, %adepsbq_10 : tensor<192xf32>
    %adratbq_10 = stablehlo.divide %admhbq_10, %addenbq_10 : tensor<192xf32>
    %adstbq_10 = stablehlo.multiply %adlrbq_10, %adratbq_10 : tensor<192xf32>
    %adsubbq_10 = stablehlo.subtract %bq_10, %adstbq_10 : tensor<192xf32>
    %adwdbq_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbq_10 = stablehlo.multiply %adwdbq_10, %adlrbq_10 : tensor<192xf32>
    %adwdpbq_10 = stablehlo.multiply %adwdlrbq_10, %bq_10 : tensor<192xf32>
    %adnewbq_10 = stablehlo.subtract %adsubbq_10, %adwdpbq_10 : tensor<192xf32>
    %adb1Wk_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wk_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWk_10 = stablehlo.multiply %adb1Wk_10, %Wk_10m : tensor<192x192xf32>
    %admgWk_10 = stablehlo.multiply %adob1Wk_10, %vitb10_mdWK : tensor<192x192xf32>
    %admnWk_10 = stablehlo.add %admsWk_10, %admgWk_10 : tensor<192x192xf32>
    %adb2Wk_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wk_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWk_10 = stablehlo.multiply %adb2Wk_10, %Wk_10v : tensor<192x192xf32>
    %adg2Wk_10 = stablehlo.multiply %vitb10_mdWK, %vitb10_mdWK : tensor<192x192xf32>
    %advgWk_10 = stablehlo.multiply %adob2Wk_10, %adg2Wk_10 : tensor<192x192xf32>
    %advnWk_10 = stablehlo.add %advsWk_10, %advgWk_10 : tensor<192x192xf32>
    %adbc1Wk_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wk_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWk_10 = stablehlo.divide %admnWk_10, %adbc1Wk_10 : tensor<192x192xf32>
    %advhWk_10 = stablehlo.divide %advnWk_10, %adbc2Wk_10 : tensor<192x192xf32>
    %adlrWk_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWk_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWk_10 = stablehlo.sqrt %advhWk_10 : tensor<192x192xf32>
    %addenWk_10 = stablehlo.add %adsqWk_10, %adepsWk_10 : tensor<192x192xf32>
    %adratWk_10 = stablehlo.divide %admhWk_10, %addenWk_10 : tensor<192x192xf32>
    %adstWk_10 = stablehlo.multiply %adlrWk_10, %adratWk_10 : tensor<192x192xf32>
    %adsubWk_10 = stablehlo.subtract %Wk_10, %adstWk_10 : tensor<192x192xf32>
    %adwdWk_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWk_10 = stablehlo.multiply %adwdWk_10, %adlrWk_10 : tensor<192x192xf32>
    %adwdpWk_10 = stablehlo.multiply %adwdlrWk_10, %Wk_10 : tensor<192x192xf32>
    %adnewWk_10 = stablehlo.subtract %adsubWk_10, %adwdpWk_10 : tensor<192x192xf32>
    %adb1bk_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bk_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbk_10 = stablehlo.multiply %adb1bk_10, %bk_10m : tensor<192xf32>
    %admgbk_10 = stablehlo.multiply %adob1bk_10, %vitb10_mdbK : tensor<192xf32>
    %admnbk_10 = stablehlo.add %admsbk_10, %admgbk_10 : tensor<192xf32>
    %adb2bk_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bk_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbk_10 = stablehlo.multiply %adb2bk_10, %bk_10v : tensor<192xf32>
    %adg2bk_10 = stablehlo.multiply %vitb10_mdbK, %vitb10_mdbK : tensor<192xf32>
    %advgbk_10 = stablehlo.multiply %adob2bk_10, %adg2bk_10 : tensor<192xf32>
    %advnbk_10 = stablehlo.add %advsbk_10, %advgbk_10 : tensor<192xf32>
    %adbc1bk_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bk_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbk_10 = stablehlo.divide %admnbk_10, %adbc1bk_10 : tensor<192xf32>
    %advhbk_10 = stablehlo.divide %advnbk_10, %adbc2bk_10 : tensor<192xf32>
    %adlrbk_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbk_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbk_10 = stablehlo.sqrt %advhbk_10 : tensor<192xf32>
    %addenbk_10 = stablehlo.add %adsqbk_10, %adepsbk_10 : tensor<192xf32>
    %adratbk_10 = stablehlo.divide %admhbk_10, %addenbk_10 : tensor<192xf32>
    %adstbk_10 = stablehlo.multiply %adlrbk_10, %adratbk_10 : tensor<192xf32>
    %adsubbk_10 = stablehlo.subtract %bk_10, %adstbk_10 : tensor<192xf32>
    %adwdbk_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbk_10 = stablehlo.multiply %adwdbk_10, %adlrbk_10 : tensor<192xf32>
    %adwdpbk_10 = stablehlo.multiply %adwdlrbk_10, %bk_10 : tensor<192xf32>
    %adnewbk_10 = stablehlo.subtract %adsubbk_10, %adwdpbk_10 : tensor<192xf32>
    %adb1Wv_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wv_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWv_10 = stablehlo.multiply %adb1Wv_10, %Wv_10m : tensor<192x192xf32>
    %admgWv_10 = stablehlo.multiply %adob1Wv_10, %vitb10_mdWV : tensor<192x192xf32>
    %admnWv_10 = stablehlo.add %admsWv_10, %admgWv_10 : tensor<192x192xf32>
    %adb2Wv_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wv_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWv_10 = stablehlo.multiply %adb2Wv_10, %Wv_10v : tensor<192x192xf32>
    %adg2Wv_10 = stablehlo.multiply %vitb10_mdWV, %vitb10_mdWV : tensor<192x192xf32>
    %advgWv_10 = stablehlo.multiply %adob2Wv_10, %adg2Wv_10 : tensor<192x192xf32>
    %advnWv_10 = stablehlo.add %advsWv_10, %advgWv_10 : tensor<192x192xf32>
    %adbc1Wv_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wv_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWv_10 = stablehlo.divide %admnWv_10, %adbc1Wv_10 : tensor<192x192xf32>
    %advhWv_10 = stablehlo.divide %advnWv_10, %adbc2Wv_10 : tensor<192x192xf32>
    %adlrWv_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWv_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWv_10 = stablehlo.sqrt %advhWv_10 : tensor<192x192xf32>
    %addenWv_10 = stablehlo.add %adsqWv_10, %adepsWv_10 : tensor<192x192xf32>
    %adratWv_10 = stablehlo.divide %admhWv_10, %addenWv_10 : tensor<192x192xf32>
    %adstWv_10 = stablehlo.multiply %adlrWv_10, %adratWv_10 : tensor<192x192xf32>
    %adsubWv_10 = stablehlo.subtract %Wv_10, %adstWv_10 : tensor<192x192xf32>
    %adwdWv_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWv_10 = stablehlo.multiply %adwdWv_10, %adlrWv_10 : tensor<192x192xf32>
    %adwdpWv_10 = stablehlo.multiply %adwdlrWv_10, %Wv_10 : tensor<192x192xf32>
    %adnewWv_10 = stablehlo.subtract %adsubWv_10, %adwdpWv_10 : tensor<192x192xf32>
    %adb1bv_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bv_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbv_10 = stablehlo.multiply %adb1bv_10, %bv_10m : tensor<192xf32>
    %admgbv_10 = stablehlo.multiply %adob1bv_10, %vitb10_mdbV : tensor<192xf32>
    %admnbv_10 = stablehlo.add %admsbv_10, %admgbv_10 : tensor<192xf32>
    %adb2bv_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bv_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbv_10 = stablehlo.multiply %adb2bv_10, %bv_10v : tensor<192xf32>
    %adg2bv_10 = stablehlo.multiply %vitb10_mdbV, %vitb10_mdbV : tensor<192xf32>
    %advgbv_10 = stablehlo.multiply %adob2bv_10, %adg2bv_10 : tensor<192xf32>
    %advnbv_10 = stablehlo.add %advsbv_10, %advgbv_10 : tensor<192xf32>
    %adbc1bv_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bv_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbv_10 = stablehlo.divide %admnbv_10, %adbc1bv_10 : tensor<192xf32>
    %advhbv_10 = stablehlo.divide %advnbv_10, %adbc2bv_10 : tensor<192xf32>
    %adlrbv_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbv_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbv_10 = stablehlo.sqrt %advhbv_10 : tensor<192xf32>
    %addenbv_10 = stablehlo.add %adsqbv_10, %adepsbv_10 : tensor<192xf32>
    %adratbv_10 = stablehlo.divide %admhbv_10, %addenbv_10 : tensor<192xf32>
    %adstbv_10 = stablehlo.multiply %adlrbv_10, %adratbv_10 : tensor<192xf32>
    %adsubbv_10 = stablehlo.subtract %bv_10, %adstbv_10 : tensor<192xf32>
    %adwdbv_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbv_10 = stablehlo.multiply %adwdbv_10, %adlrbv_10 : tensor<192xf32>
    %adwdpbv_10 = stablehlo.multiply %adwdlrbv_10, %bv_10 : tensor<192xf32>
    %adnewbv_10 = stablehlo.subtract %adsubbv_10, %adwdpbv_10 : tensor<192xf32>
    %adb1Wo_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wo_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWo_10 = stablehlo.multiply %adb1Wo_10, %Wo_10m : tensor<192x192xf32>
    %admgWo_10 = stablehlo.multiply %adob1Wo_10, %vitb10_mdWo : tensor<192x192xf32>
    %admnWo_10 = stablehlo.add %admsWo_10, %admgWo_10 : tensor<192x192xf32>
    %adb2Wo_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wo_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWo_10 = stablehlo.multiply %adb2Wo_10, %Wo_10v : tensor<192x192xf32>
    %adg2Wo_10 = stablehlo.multiply %vitb10_mdWo, %vitb10_mdWo : tensor<192x192xf32>
    %advgWo_10 = stablehlo.multiply %adob2Wo_10, %adg2Wo_10 : tensor<192x192xf32>
    %advnWo_10 = stablehlo.add %advsWo_10, %advgWo_10 : tensor<192x192xf32>
    %adbc1Wo_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wo_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWo_10 = stablehlo.divide %admnWo_10, %adbc1Wo_10 : tensor<192x192xf32>
    %advhWo_10 = stablehlo.divide %advnWo_10, %adbc2Wo_10 : tensor<192x192xf32>
    %adlrWo_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWo_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWo_10 = stablehlo.sqrt %advhWo_10 : tensor<192x192xf32>
    %addenWo_10 = stablehlo.add %adsqWo_10, %adepsWo_10 : tensor<192x192xf32>
    %adratWo_10 = stablehlo.divide %admhWo_10, %addenWo_10 : tensor<192x192xf32>
    %adstWo_10 = stablehlo.multiply %adlrWo_10, %adratWo_10 : tensor<192x192xf32>
    %adsubWo_10 = stablehlo.subtract %Wo_10, %adstWo_10 : tensor<192x192xf32>
    %adwdWo_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWo_10 = stablehlo.multiply %adwdWo_10, %adlrWo_10 : tensor<192x192xf32>
    %adwdpWo_10 = stablehlo.multiply %adwdlrWo_10, %Wo_10 : tensor<192x192xf32>
    %adnewWo_10 = stablehlo.subtract %adsubWo_10, %adwdpWo_10 : tensor<192x192xf32>
    %adb1bo_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bo_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbo_10 = stablehlo.multiply %adb1bo_10, %bo_10m : tensor<192xf32>
    %admgbo_10 = stablehlo.multiply %adob1bo_10, %vitb10_mdbo : tensor<192xf32>
    %admnbo_10 = stablehlo.add %admsbo_10, %admgbo_10 : tensor<192xf32>
    %adb2bo_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bo_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbo_10 = stablehlo.multiply %adb2bo_10, %bo_10v : tensor<192xf32>
    %adg2bo_10 = stablehlo.multiply %vitb10_mdbo, %vitb10_mdbo : tensor<192xf32>
    %advgbo_10 = stablehlo.multiply %adob2bo_10, %adg2bo_10 : tensor<192xf32>
    %advnbo_10 = stablehlo.add %advsbo_10, %advgbo_10 : tensor<192xf32>
    %adbc1bo_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bo_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbo_10 = stablehlo.divide %admnbo_10, %adbc1bo_10 : tensor<192xf32>
    %advhbo_10 = stablehlo.divide %advnbo_10, %adbc2bo_10 : tensor<192xf32>
    %adlrbo_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbo_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbo_10 = stablehlo.sqrt %advhbo_10 : tensor<192xf32>
    %addenbo_10 = stablehlo.add %adsqbo_10, %adepsbo_10 : tensor<192xf32>
    %adratbo_10 = stablehlo.divide %admhbo_10, %addenbo_10 : tensor<192xf32>
    %adstbo_10 = stablehlo.multiply %adlrbo_10, %adratbo_10 : tensor<192xf32>
    %adsubbo_10 = stablehlo.subtract %bo_10, %adstbo_10 : tensor<192xf32>
    %adwdbo_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbo_10 = stablehlo.multiply %adwdbo_10, %adlrbo_10 : tensor<192xf32>
    %adwdpbo_10 = stablehlo.multiply %adwdlrbo_10, %bo_10 : tensor<192xf32>
    %adnewbo_10 = stablehlo.subtract %adsubbo_10, %adwdpbo_10 : tensor<192xf32>
    %adb1g2_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g2_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg2_10 = stablehlo.multiply %adb1g2_10, %g2_10m : tensor<192xf32>
    %admgg2_10 = stablehlo.multiply %adob1g2_10, %vitb10_2dg : tensor<192xf32>
    %admng2_10 = stablehlo.add %admsg2_10, %admgg2_10 : tensor<192xf32>
    %adb2g2_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g2_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg2_10 = stablehlo.multiply %adb2g2_10, %g2_10v : tensor<192xf32>
    %adg2g2_10 = stablehlo.multiply %vitb10_2dg, %vitb10_2dg : tensor<192xf32>
    %advgg2_10 = stablehlo.multiply %adob2g2_10, %adg2g2_10 : tensor<192xf32>
    %advng2_10 = stablehlo.add %advsg2_10, %advgg2_10 : tensor<192xf32>
    %adbc1g2_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g2_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg2_10 = stablehlo.divide %admng2_10, %adbc1g2_10 : tensor<192xf32>
    %advhg2_10 = stablehlo.divide %advng2_10, %adbc2g2_10 : tensor<192xf32>
    %adlrg2_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg2_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg2_10 = stablehlo.sqrt %advhg2_10 : tensor<192xf32>
    %addeng2_10 = stablehlo.add %adsqg2_10, %adepsg2_10 : tensor<192xf32>
    %adratg2_10 = stablehlo.divide %admhg2_10, %addeng2_10 : tensor<192xf32>
    %adstg2_10 = stablehlo.multiply %adlrg2_10, %adratg2_10 : tensor<192xf32>
    %adsubg2_10 = stablehlo.subtract %g2_10, %adstg2_10 : tensor<192xf32>
    %adwdg2_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg2_10 = stablehlo.multiply %adwdg2_10, %adlrg2_10 : tensor<192xf32>
    %adwdpg2_10 = stablehlo.multiply %adwdlrg2_10, %g2_10 : tensor<192xf32>
    %adnewg2_10 = stablehlo.subtract %adsubg2_10, %adwdpg2_10 : tensor<192xf32>
    %adb1b2_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b2_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb2_10 = stablehlo.multiply %adb1b2_10, %b2_10m : tensor<192xf32>
    %admgb2_10 = stablehlo.multiply %adob1b2_10, %vitb10_2db : tensor<192xf32>
    %admnb2_10 = stablehlo.add %admsb2_10, %admgb2_10 : tensor<192xf32>
    %adb2b2_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b2_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb2_10 = stablehlo.multiply %adb2b2_10, %b2_10v : tensor<192xf32>
    %adg2b2_10 = stablehlo.multiply %vitb10_2db, %vitb10_2db : tensor<192xf32>
    %advgb2_10 = stablehlo.multiply %adob2b2_10, %adg2b2_10 : tensor<192xf32>
    %advnb2_10 = stablehlo.add %advsb2_10, %advgb2_10 : tensor<192xf32>
    %adbc1b2_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b2_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb2_10 = stablehlo.divide %admnb2_10, %adbc1b2_10 : tensor<192xf32>
    %advhb2_10 = stablehlo.divide %advnb2_10, %adbc2b2_10 : tensor<192xf32>
    %adlrb2_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb2_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb2_10 = stablehlo.sqrt %advhb2_10 : tensor<192xf32>
    %addenb2_10 = stablehlo.add %adsqb2_10, %adepsb2_10 : tensor<192xf32>
    %adratb2_10 = stablehlo.divide %admhb2_10, %addenb2_10 : tensor<192xf32>
    %adstb2_10 = stablehlo.multiply %adlrb2_10, %adratb2_10 : tensor<192xf32>
    %adsubb2_10 = stablehlo.subtract %b2_10, %adstb2_10 : tensor<192xf32>
    %adwdb2_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb2_10 = stablehlo.multiply %adwdb2_10, %adlrb2_10 : tensor<192xf32>
    %adwdpb2_10 = stablehlo.multiply %adwdlrb2_10, %b2_10 : tensor<192xf32>
    %adnewb2_10 = stablehlo.subtract %adsubb2_10, %adwdpb2_10 : tensor<192xf32>
    %adb1Wfc1_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob1Wfc1_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admsWfc1_10 = stablehlo.multiply %adb1Wfc1_10, %Wfc1_10m : tensor<192x768xf32>
    %admgWfc1_10 = stablehlo.multiply %adob1Wfc1_10, %vitb10_pdWfc1 : tensor<192x768xf32>
    %admnWfc1_10 = stablehlo.add %admsWfc1_10, %admgWfc1_10 : tensor<192x768xf32>
    %adb2Wfc1_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob2Wfc1_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %advsWfc1_10 = stablehlo.multiply %adb2Wfc1_10, %Wfc1_10v : tensor<192x768xf32>
    %adg2Wfc1_10 = stablehlo.multiply %vitb10_pdWfc1, %vitb10_pdWfc1 : tensor<192x768xf32>
    %advgWfc1_10 = stablehlo.multiply %adob2Wfc1_10, %adg2Wfc1_10 : tensor<192x768xf32>
    %advnWfc1_10 = stablehlo.add %advsWfc1_10, %advgWfc1_10 : tensor<192x768xf32>
    %adbc1Wfc1_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adbc2Wfc1_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admhWfc1_10 = stablehlo.divide %admnWfc1_10, %adbc1Wfc1_10 : tensor<192x768xf32>
    %advhWfc1_10 = stablehlo.divide %advnWfc1_10, %adbc2Wfc1_10 : tensor<192x768xf32>
    %adlrWfc1_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adepsWfc1_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adsqWfc1_10 = stablehlo.sqrt %advhWfc1_10 : tensor<192x768xf32>
    %addenWfc1_10 = stablehlo.add %adsqWfc1_10, %adepsWfc1_10 : tensor<192x768xf32>
    %adratWfc1_10 = stablehlo.divide %admhWfc1_10, %addenWfc1_10 : tensor<192x768xf32>
    %adstWfc1_10 = stablehlo.multiply %adlrWfc1_10, %adratWfc1_10 : tensor<192x768xf32>
    %adsubWfc1_10 = stablehlo.subtract %Wfc1_10, %adstWfc1_10 : tensor<192x768xf32>
    %adwdWfc1_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adwdlrWfc1_10 = stablehlo.multiply %adwdWfc1_10, %adlrWfc1_10 : tensor<192x768xf32>
    %adwdpWfc1_10 = stablehlo.multiply %adwdlrWfc1_10, %Wfc1_10 : tensor<192x768xf32>
    %adnewWfc1_10 = stablehlo.subtract %adsubWfc1_10, %adwdpWfc1_10 : tensor<192x768xf32>
    %adb1bfc1_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1bfc1_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admsbfc1_10 = stablehlo.multiply %adb1bfc1_10, %bfc1_10m : tensor<768xf32>
    %admgbfc1_10 = stablehlo.multiply %adob1bfc1_10, %vitb10_pdbfc1 : tensor<768xf32>
    %admnbfc1_10 = stablehlo.add %admsbfc1_10, %admgbfc1_10 : tensor<768xf32>
    %adb2bfc1_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2bfc1_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advsbfc1_10 = stablehlo.multiply %adb2bfc1_10, %bfc1_10v : tensor<768xf32>
    %adg2bfc1_10 = stablehlo.multiply %vitb10_pdbfc1, %vitb10_pdbfc1 : tensor<768xf32>
    %advgbfc1_10 = stablehlo.multiply %adob2bfc1_10, %adg2bfc1_10 : tensor<768xf32>
    %advnbfc1_10 = stablehlo.add %advsbfc1_10, %advgbfc1_10 : tensor<768xf32>
    %adbc1bfc1_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2bfc1_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhbfc1_10 = stablehlo.divide %admnbfc1_10, %adbc1bfc1_10 : tensor<768xf32>
    %advhbfc1_10 = stablehlo.divide %advnbfc1_10, %adbc2bfc1_10 : tensor<768xf32>
    %adlrbfc1_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepsbfc1_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqbfc1_10 = stablehlo.sqrt %advhbfc1_10 : tensor<768xf32>
    %addenbfc1_10 = stablehlo.add %adsqbfc1_10, %adepsbfc1_10 : tensor<768xf32>
    %adratbfc1_10 = stablehlo.divide %admhbfc1_10, %addenbfc1_10 : tensor<768xf32>
    %adstbfc1_10 = stablehlo.multiply %adlrbfc1_10, %adratbfc1_10 : tensor<768xf32>
    %adsubbfc1_10 = stablehlo.subtract %bfc1_10, %adstbfc1_10 : tensor<768xf32>
    %adwdbfc1_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrbfc1_10 = stablehlo.multiply %adwdbfc1_10, %adlrbfc1_10 : tensor<768xf32>
    %adwdpbfc1_10 = stablehlo.multiply %adwdlrbfc1_10, %bfc1_10 : tensor<768xf32>
    %adnewbfc1_10 = stablehlo.subtract %adsubbfc1_10, %adwdpbfc1_10 : tensor<768xf32>
    %adb1Wfc2_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob1Wfc2_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admsWfc2_10 = stablehlo.multiply %adb1Wfc2_10, %Wfc2_10m : tensor<768x192xf32>
    %admgWfc2_10 = stablehlo.multiply %adob1Wfc2_10, %vitb10_pdWfc2 : tensor<768x192xf32>
    %admnWfc2_10 = stablehlo.add %admsWfc2_10, %admgWfc2_10 : tensor<768x192xf32>
    %adb2Wfc2_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob2Wfc2_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %advsWfc2_10 = stablehlo.multiply %adb2Wfc2_10, %Wfc2_10v : tensor<768x192xf32>
    %adg2Wfc2_10 = stablehlo.multiply %vitb10_pdWfc2, %vitb10_pdWfc2 : tensor<768x192xf32>
    %advgWfc2_10 = stablehlo.multiply %adob2Wfc2_10, %adg2Wfc2_10 : tensor<768x192xf32>
    %advnWfc2_10 = stablehlo.add %advsWfc2_10, %advgWfc2_10 : tensor<768x192xf32>
    %adbc1Wfc2_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adbc2Wfc2_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admhWfc2_10 = stablehlo.divide %admnWfc2_10, %adbc1Wfc2_10 : tensor<768x192xf32>
    %advhWfc2_10 = stablehlo.divide %advnWfc2_10, %adbc2Wfc2_10 : tensor<768x192xf32>
    %adlrWfc2_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adepsWfc2_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adsqWfc2_10 = stablehlo.sqrt %advhWfc2_10 : tensor<768x192xf32>
    %addenWfc2_10 = stablehlo.add %adsqWfc2_10, %adepsWfc2_10 : tensor<768x192xf32>
    %adratWfc2_10 = stablehlo.divide %admhWfc2_10, %addenWfc2_10 : tensor<768x192xf32>
    %adstWfc2_10 = stablehlo.multiply %adlrWfc2_10, %adratWfc2_10 : tensor<768x192xf32>
    %adsubWfc2_10 = stablehlo.subtract %Wfc2_10, %adstWfc2_10 : tensor<768x192xf32>
    %adwdWfc2_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adwdlrWfc2_10 = stablehlo.multiply %adwdWfc2_10, %adlrWfc2_10 : tensor<768x192xf32>
    %adwdpWfc2_10 = stablehlo.multiply %adwdlrWfc2_10, %Wfc2_10 : tensor<768x192xf32>
    %adnewWfc2_10 = stablehlo.subtract %adsubWfc2_10, %adwdpWfc2_10 : tensor<768x192xf32>
    %adb1bfc2_10 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bfc2_10 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbfc2_10 = stablehlo.multiply %adb1bfc2_10, %bfc2_10m : tensor<192xf32>
    %admgbfc2_10 = stablehlo.multiply %adob1bfc2_10, %vitb10_pdbfc2 : tensor<192xf32>
    %admnbfc2_10 = stablehlo.add %admsbfc2_10, %admgbfc2_10 : tensor<192xf32>
    %adb2bfc2_10 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bfc2_10 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbfc2_10 = stablehlo.multiply %adb2bfc2_10, %bfc2_10v : tensor<192xf32>
    %adg2bfc2_10 = stablehlo.multiply %vitb10_pdbfc2, %vitb10_pdbfc2 : tensor<192xf32>
    %advgbfc2_10 = stablehlo.multiply %adob2bfc2_10, %adg2bfc2_10 : tensor<192xf32>
    %advnbfc2_10 = stablehlo.add %advsbfc2_10, %advgbfc2_10 : tensor<192xf32>
    %adbc1bfc2_10 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bfc2_10 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbfc2_10 = stablehlo.divide %admnbfc2_10, %adbc1bfc2_10 : tensor<192xf32>
    %advhbfc2_10 = stablehlo.divide %advnbfc2_10, %adbc2bfc2_10 : tensor<192xf32>
    %adlrbfc2_10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbfc2_10 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbfc2_10 = stablehlo.sqrt %advhbfc2_10 : tensor<192xf32>
    %addenbfc2_10 = stablehlo.add %adsqbfc2_10, %adepsbfc2_10 : tensor<192xf32>
    %adratbfc2_10 = stablehlo.divide %admhbfc2_10, %addenbfc2_10 : tensor<192xf32>
    %adstbfc2_10 = stablehlo.multiply %adlrbfc2_10, %adratbfc2_10 : tensor<192xf32>
    %adsubbfc2_10 = stablehlo.subtract %bfc2_10, %adstbfc2_10 : tensor<192xf32>
    %adwdbfc2_10 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbfc2_10 = stablehlo.multiply %adwdbfc2_10, %adlrbfc2_10 : tensor<192xf32>
    %adwdpbfc2_10 = stablehlo.multiply %adwdlrbfc2_10, %bfc2_10 : tensor<192xf32>
    %adnewbfc2_10 = stablehlo.subtract %adsubbfc2_10, %adwdpbfc2_10 : tensor<192xf32>
    %adb1g1_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g1_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg1_11 = stablehlo.multiply %adb1g1_11, %g1_11m : tensor<192xf32>
    %admgg1_11 = stablehlo.multiply %adob1g1_11, %vitb11_1dg : tensor<192xf32>
    %admng1_11 = stablehlo.add %admsg1_11, %admgg1_11 : tensor<192xf32>
    %adb2g1_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g1_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg1_11 = stablehlo.multiply %adb2g1_11, %g1_11v : tensor<192xf32>
    %adg2g1_11 = stablehlo.multiply %vitb11_1dg, %vitb11_1dg : tensor<192xf32>
    %advgg1_11 = stablehlo.multiply %adob2g1_11, %adg2g1_11 : tensor<192xf32>
    %advng1_11 = stablehlo.add %advsg1_11, %advgg1_11 : tensor<192xf32>
    %adbc1g1_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g1_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg1_11 = stablehlo.divide %admng1_11, %adbc1g1_11 : tensor<192xf32>
    %advhg1_11 = stablehlo.divide %advng1_11, %adbc2g1_11 : tensor<192xf32>
    %adlrg1_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg1_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg1_11 = stablehlo.sqrt %advhg1_11 : tensor<192xf32>
    %addeng1_11 = stablehlo.add %adsqg1_11, %adepsg1_11 : tensor<192xf32>
    %adratg1_11 = stablehlo.divide %admhg1_11, %addeng1_11 : tensor<192xf32>
    %adstg1_11 = stablehlo.multiply %adlrg1_11, %adratg1_11 : tensor<192xf32>
    %adsubg1_11 = stablehlo.subtract %g1_11, %adstg1_11 : tensor<192xf32>
    %adwdg1_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg1_11 = stablehlo.multiply %adwdg1_11, %adlrg1_11 : tensor<192xf32>
    %adwdpg1_11 = stablehlo.multiply %adwdlrg1_11, %g1_11 : tensor<192xf32>
    %adnewg1_11 = stablehlo.subtract %adsubg1_11, %adwdpg1_11 : tensor<192xf32>
    %adb1b1_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b1_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb1_11 = stablehlo.multiply %adb1b1_11, %b1_11m : tensor<192xf32>
    %admgb1_11 = stablehlo.multiply %adob1b1_11, %vitb11_1db : tensor<192xf32>
    %admnb1_11 = stablehlo.add %admsb1_11, %admgb1_11 : tensor<192xf32>
    %adb2b1_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b1_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb1_11 = stablehlo.multiply %adb2b1_11, %b1_11v : tensor<192xf32>
    %adg2b1_11 = stablehlo.multiply %vitb11_1db, %vitb11_1db : tensor<192xf32>
    %advgb1_11 = stablehlo.multiply %adob2b1_11, %adg2b1_11 : tensor<192xf32>
    %advnb1_11 = stablehlo.add %advsb1_11, %advgb1_11 : tensor<192xf32>
    %adbc1b1_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b1_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb1_11 = stablehlo.divide %admnb1_11, %adbc1b1_11 : tensor<192xf32>
    %advhb1_11 = stablehlo.divide %advnb1_11, %adbc2b1_11 : tensor<192xf32>
    %adlrb1_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb1_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb1_11 = stablehlo.sqrt %advhb1_11 : tensor<192xf32>
    %addenb1_11 = stablehlo.add %adsqb1_11, %adepsb1_11 : tensor<192xf32>
    %adratb1_11 = stablehlo.divide %admhb1_11, %addenb1_11 : tensor<192xf32>
    %adstb1_11 = stablehlo.multiply %adlrb1_11, %adratb1_11 : tensor<192xf32>
    %adsubb1_11 = stablehlo.subtract %b1_11, %adstb1_11 : tensor<192xf32>
    %adwdb1_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb1_11 = stablehlo.multiply %adwdb1_11, %adlrb1_11 : tensor<192xf32>
    %adwdpb1_11 = stablehlo.multiply %adwdlrb1_11, %b1_11 : tensor<192xf32>
    %adnewb1_11 = stablehlo.subtract %adsubb1_11, %adwdpb1_11 : tensor<192xf32>
    %adb1Wq_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wq_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWq_11 = stablehlo.multiply %adb1Wq_11, %Wq_11m : tensor<192x192xf32>
    %admgWq_11 = stablehlo.multiply %adob1Wq_11, %vitb11_mdWQ : tensor<192x192xf32>
    %admnWq_11 = stablehlo.add %admsWq_11, %admgWq_11 : tensor<192x192xf32>
    %adb2Wq_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wq_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWq_11 = stablehlo.multiply %adb2Wq_11, %Wq_11v : tensor<192x192xf32>
    %adg2Wq_11 = stablehlo.multiply %vitb11_mdWQ, %vitb11_mdWQ : tensor<192x192xf32>
    %advgWq_11 = stablehlo.multiply %adob2Wq_11, %adg2Wq_11 : tensor<192x192xf32>
    %advnWq_11 = stablehlo.add %advsWq_11, %advgWq_11 : tensor<192x192xf32>
    %adbc1Wq_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wq_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWq_11 = stablehlo.divide %admnWq_11, %adbc1Wq_11 : tensor<192x192xf32>
    %advhWq_11 = stablehlo.divide %advnWq_11, %adbc2Wq_11 : tensor<192x192xf32>
    %adlrWq_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWq_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWq_11 = stablehlo.sqrt %advhWq_11 : tensor<192x192xf32>
    %addenWq_11 = stablehlo.add %adsqWq_11, %adepsWq_11 : tensor<192x192xf32>
    %adratWq_11 = stablehlo.divide %admhWq_11, %addenWq_11 : tensor<192x192xf32>
    %adstWq_11 = stablehlo.multiply %adlrWq_11, %adratWq_11 : tensor<192x192xf32>
    %adsubWq_11 = stablehlo.subtract %Wq_11, %adstWq_11 : tensor<192x192xf32>
    %adwdWq_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWq_11 = stablehlo.multiply %adwdWq_11, %adlrWq_11 : tensor<192x192xf32>
    %adwdpWq_11 = stablehlo.multiply %adwdlrWq_11, %Wq_11 : tensor<192x192xf32>
    %adnewWq_11 = stablehlo.subtract %adsubWq_11, %adwdpWq_11 : tensor<192x192xf32>
    %adb1bq_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bq_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbq_11 = stablehlo.multiply %adb1bq_11, %bq_11m : tensor<192xf32>
    %admgbq_11 = stablehlo.multiply %adob1bq_11, %vitb11_mdbQ : tensor<192xf32>
    %admnbq_11 = stablehlo.add %admsbq_11, %admgbq_11 : tensor<192xf32>
    %adb2bq_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bq_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbq_11 = stablehlo.multiply %adb2bq_11, %bq_11v : tensor<192xf32>
    %adg2bq_11 = stablehlo.multiply %vitb11_mdbQ, %vitb11_mdbQ : tensor<192xf32>
    %advgbq_11 = stablehlo.multiply %adob2bq_11, %adg2bq_11 : tensor<192xf32>
    %advnbq_11 = stablehlo.add %advsbq_11, %advgbq_11 : tensor<192xf32>
    %adbc1bq_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bq_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbq_11 = stablehlo.divide %admnbq_11, %adbc1bq_11 : tensor<192xf32>
    %advhbq_11 = stablehlo.divide %advnbq_11, %adbc2bq_11 : tensor<192xf32>
    %adlrbq_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbq_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbq_11 = stablehlo.sqrt %advhbq_11 : tensor<192xf32>
    %addenbq_11 = stablehlo.add %adsqbq_11, %adepsbq_11 : tensor<192xf32>
    %adratbq_11 = stablehlo.divide %admhbq_11, %addenbq_11 : tensor<192xf32>
    %adstbq_11 = stablehlo.multiply %adlrbq_11, %adratbq_11 : tensor<192xf32>
    %adsubbq_11 = stablehlo.subtract %bq_11, %adstbq_11 : tensor<192xf32>
    %adwdbq_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbq_11 = stablehlo.multiply %adwdbq_11, %adlrbq_11 : tensor<192xf32>
    %adwdpbq_11 = stablehlo.multiply %adwdlrbq_11, %bq_11 : tensor<192xf32>
    %adnewbq_11 = stablehlo.subtract %adsubbq_11, %adwdpbq_11 : tensor<192xf32>
    %adb1Wk_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wk_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWk_11 = stablehlo.multiply %adb1Wk_11, %Wk_11m : tensor<192x192xf32>
    %admgWk_11 = stablehlo.multiply %adob1Wk_11, %vitb11_mdWK : tensor<192x192xf32>
    %admnWk_11 = stablehlo.add %admsWk_11, %admgWk_11 : tensor<192x192xf32>
    %adb2Wk_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wk_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWk_11 = stablehlo.multiply %adb2Wk_11, %Wk_11v : tensor<192x192xf32>
    %adg2Wk_11 = stablehlo.multiply %vitb11_mdWK, %vitb11_mdWK : tensor<192x192xf32>
    %advgWk_11 = stablehlo.multiply %adob2Wk_11, %adg2Wk_11 : tensor<192x192xf32>
    %advnWk_11 = stablehlo.add %advsWk_11, %advgWk_11 : tensor<192x192xf32>
    %adbc1Wk_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wk_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWk_11 = stablehlo.divide %admnWk_11, %adbc1Wk_11 : tensor<192x192xf32>
    %advhWk_11 = stablehlo.divide %advnWk_11, %adbc2Wk_11 : tensor<192x192xf32>
    %adlrWk_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWk_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWk_11 = stablehlo.sqrt %advhWk_11 : tensor<192x192xf32>
    %addenWk_11 = stablehlo.add %adsqWk_11, %adepsWk_11 : tensor<192x192xf32>
    %adratWk_11 = stablehlo.divide %admhWk_11, %addenWk_11 : tensor<192x192xf32>
    %adstWk_11 = stablehlo.multiply %adlrWk_11, %adratWk_11 : tensor<192x192xf32>
    %adsubWk_11 = stablehlo.subtract %Wk_11, %adstWk_11 : tensor<192x192xf32>
    %adwdWk_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWk_11 = stablehlo.multiply %adwdWk_11, %adlrWk_11 : tensor<192x192xf32>
    %adwdpWk_11 = stablehlo.multiply %adwdlrWk_11, %Wk_11 : tensor<192x192xf32>
    %adnewWk_11 = stablehlo.subtract %adsubWk_11, %adwdpWk_11 : tensor<192x192xf32>
    %adb1bk_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bk_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbk_11 = stablehlo.multiply %adb1bk_11, %bk_11m : tensor<192xf32>
    %admgbk_11 = stablehlo.multiply %adob1bk_11, %vitb11_mdbK : tensor<192xf32>
    %admnbk_11 = stablehlo.add %admsbk_11, %admgbk_11 : tensor<192xf32>
    %adb2bk_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bk_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbk_11 = stablehlo.multiply %adb2bk_11, %bk_11v : tensor<192xf32>
    %adg2bk_11 = stablehlo.multiply %vitb11_mdbK, %vitb11_mdbK : tensor<192xf32>
    %advgbk_11 = stablehlo.multiply %adob2bk_11, %adg2bk_11 : tensor<192xf32>
    %advnbk_11 = stablehlo.add %advsbk_11, %advgbk_11 : tensor<192xf32>
    %adbc1bk_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bk_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbk_11 = stablehlo.divide %admnbk_11, %adbc1bk_11 : tensor<192xf32>
    %advhbk_11 = stablehlo.divide %advnbk_11, %adbc2bk_11 : tensor<192xf32>
    %adlrbk_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbk_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbk_11 = stablehlo.sqrt %advhbk_11 : tensor<192xf32>
    %addenbk_11 = stablehlo.add %adsqbk_11, %adepsbk_11 : tensor<192xf32>
    %adratbk_11 = stablehlo.divide %admhbk_11, %addenbk_11 : tensor<192xf32>
    %adstbk_11 = stablehlo.multiply %adlrbk_11, %adratbk_11 : tensor<192xf32>
    %adsubbk_11 = stablehlo.subtract %bk_11, %adstbk_11 : tensor<192xf32>
    %adwdbk_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbk_11 = stablehlo.multiply %adwdbk_11, %adlrbk_11 : tensor<192xf32>
    %adwdpbk_11 = stablehlo.multiply %adwdlrbk_11, %bk_11 : tensor<192xf32>
    %adnewbk_11 = stablehlo.subtract %adsubbk_11, %adwdpbk_11 : tensor<192xf32>
    %adb1Wv_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wv_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWv_11 = stablehlo.multiply %adb1Wv_11, %Wv_11m : tensor<192x192xf32>
    %admgWv_11 = stablehlo.multiply %adob1Wv_11, %vitb11_mdWV : tensor<192x192xf32>
    %admnWv_11 = stablehlo.add %admsWv_11, %admgWv_11 : tensor<192x192xf32>
    %adb2Wv_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wv_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWv_11 = stablehlo.multiply %adb2Wv_11, %Wv_11v : tensor<192x192xf32>
    %adg2Wv_11 = stablehlo.multiply %vitb11_mdWV, %vitb11_mdWV : tensor<192x192xf32>
    %advgWv_11 = stablehlo.multiply %adob2Wv_11, %adg2Wv_11 : tensor<192x192xf32>
    %advnWv_11 = stablehlo.add %advsWv_11, %advgWv_11 : tensor<192x192xf32>
    %adbc1Wv_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wv_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWv_11 = stablehlo.divide %admnWv_11, %adbc1Wv_11 : tensor<192x192xf32>
    %advhWv_11 = stablehlo.divide %advnWv_11, %adbc2Wv_11 : tensor<192x192xf32>
    %adlrWv_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWv_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWv_11 = stablehlo.sqrt %advhWv_11 : tensor<192x192xf32>
    %addenWv_11 = stablehlo.add %adsqWv_11, %adepsWv_11 : tensor<192x192xf32>
    %adratWv_11 = stablehlo.divide %admhWv_11, %addenWv_11 : tensor<192x192xf32>
    %adstWv_11 = stablehlo.multiply %adlrWv_11, %adratWv_11 : tensor<192x192xf32>
    %adsubWv_11 = stablehlo.subtract %Wv_11, %adstWv_11 : tensor<192x192xf32>
    %adwdWv_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWv_11 = stablehlo.multiply %adwdWv_11, %adlrWv_11 : tensor<192x192xf32>
    %adwdpWv_11 = stablehlo.multiply %adwdlrWv_11, %Wv_11 : tensor<192x192xf32>
    %adnewWv_11 = stablehlo.subtract %adsubWv_11, %adwdpWv_11 : tensor<192x192xf32>
    %adb1bv_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bv_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbv_11 = stablehlo.multiply %adb1bv_11, %bv_11m : tensor<192xf32>
    %admgbv_11 = stablehlo.multiply %adob1bv_11, %vitb11_mdbV : tensor<192xf32>
    %admnbv_11 = stablehlo.add %admsbv_11, %admgbv_11 : tensor<192xf32>
    %adb2bv_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bv_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbv_11 = stablehlo.multiply %adb2bv_11, %bv_11v : tensor<192xf32>
    %adg2bv_11 = stablehlo.multiply %vitb11_mdbV, %vitb11_mdbV : tensor<192xf32>
    %advgbv_11 = stablehlo.multiply %adob2bv_11, %adg2bv_11 : tensor<192xf32>
    %advnbv_11 = stablehlo.add %advsbv_11, %advgbv_11 : tensor<192xf32>
    %adbc1bv_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bv_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbv_11 = stablehlo.divide %admnbv_11, %adbc1bv_11 : tensor<192xf32>
    %advhbv_11 = stablehlo.divide %advnbv_11, %adbc2bv_11 : tensor<192xf32>
    %adlrbv_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbv_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbv_11 = stablehlo.sqrt %advhbv_11 : tensor<192xf32>
    %addenbv_11 = stablehlo.add %adsqbv_11, %adepsbv_11 : tensor<192xf32>
    %adratbv_11 = stablehlo.divide %admhbv_11, %addenbv_11 : tensor<192xf32>
    %adstbv_11 = stablehlo.multiply %adlrbv_11, %adratbv_11 : tensor<192xf32>
    %adsubbv_11 = stablehlo.subtract %bv_11, %adstbv_11 : tensor<192xf32>
    %adwdbv_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbv_11 = stablehlo.multiply %adwdbv_11, %adlrbv_11 : tensor<192xf32>
    %adwdpbv_11 = stablehlo.multiply %adwdlrbv_11, %bv_11 : tensor<192xf32>
    %adnewbv_11 = stablehlo.subtract %adsubbv_11, %adwdpbv_11 : tensor<192xf32>
    %adb1Wo_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob1Wo_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admsWo_11 = stablehlo.multiply %adb1Wo_11, %Wo_11m : tensor<192x192xf32>
    %admgWo_11 = stablehlo.multiply %adob1Wo_11, %vitb11_mdWo : tensor<192x192xf32>
    %admnWo_11 = stablehlo.add %admsWo_11, %admgWo_11 : tensor<192x192xf32>
    %adb2Wo_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adob2Wo_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %advsWo_11 = stablehlo.multiply %adb2Wo_11, %Wo_11v : tensor<192x192xf32>
    %adg2Wo_11 = stablehlo.multiply %vitb11_mdWo, %vitb11_mdWo : tensor<192x192xf32>
    %advgWo_11 = stablehlo.multiply %adob2Wo_11, %adg2Wo_11 : tensor<192x192xf32>
    %advnWo_11 = stablehlo.add %advsWo_11, %advgWo_11 : tensor<192x192xf32>
    %adbc1Wo_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adbc2Wo_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %admhWo_11 = stablehlo.divide %admnWo_11, %adbc1Wo_11 : tensor<192x192xf32>
    %advhWo_11 = stablehlo.divide %advnWo_11, %adbc2Wo_11 : tensor<192x192xf32>
    %adlrWo_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adepsWo_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adsqWo_11 = stablehlo.sqrt %advhWo_11 : tensor<192x192xf32>
    %addenWo_11 = stablehlo.add %adsqWo_11, %adepsWo_11 : tensor<192x192xf32>
    %adratWo_11 = stablehlo.divide %admhWo_11, %addenWo_11 : tensor<192x192xf32>
    %adstWo_11 = stablehlo.multiply %adlrWo_11, %adratWo_11 : tensor<192x192xf32>
    %adsubWo_11 = stablehlo.subtract %Wo_11, %adstWo_11 : tensor<192x192xf32>
    %adwdWo_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x192xf32>
    %adwdlrWo_11 = stablehlo.multiply %adwdWo_11, %adlrWo_11 : tensor<192x192xf32>
    %adwdpWo_11 = stablehlo.multiply %adwdlrWo_11, %Wo_11 : tensor<192x192xf32>
    %adnewWo_11 = stablehlo.subtract %adsubWo_11, %adwdpWo_11 : tensor<192x192xf32>
    %adb1bo_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bo_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbo_11 = stablehlo.multiply %adb1bo_11, %bo_11m : tensor<192xf32>
    %admgbo_11 = stablehlo.multiply %adob1bo_11, %vitb11_mdbo : tensor<192xf32>
    %admnbo_11 = stablehlo.add %admsbo_11, %admgbo_11 : tensor<192xf32>
    %adb2bo_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bo_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbo_11 = stablehlo.multiply %adb2bo_11, %bo_11v : tensor<192xf32>
    %adg2bo_11 = stablehlo.multiply %vitb11_mdbo, %vitb11_mdbo : tensor<192xf32>
    %advgbo_11 = stablehlo.multiply %adob2bo_11, %adg2bo_11 : tensor<192xf32>
    %advnbo_11 = stablehlo.add %advsbo_11, %advgbo_11 : tensor<192xf32>
    %adbc1bo_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bo_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbo_11 = stablehlo.divide %admnbo_11, %adbc1bo_11 : tensor<192xf32>
    %advhbo_11 = stablehlo.divide %advnbo_11, %adbc2bo_11 : tensor<192xf32>
    %adlrbo_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbo_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbo_11 = stablehlo.sqrt %advhbo_11 : tensor<192xf32>
    %addenbo_11 = stablehlo.add %adsqbo_11, %adepsbo_11 : tensor<192xf32>
    %adratbo_11 = stablehlo.divide %admhbo_11, %addenbo_11 : tensor<192xf32>
    %adstbo_11 = stablehlo.multiply %adlrbo_11, %adratbo_11 : tensor<192xf32>
    %adsubbo_11 = stablehlo.subtract %bo_11, %adstbo_11 : tensor<192xf32>
    %adwdbo_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbo_11 = stablehlo.multiply %adwdbo_11, %adlrbo_11 : tensor<192xf32>
    %adwdpbo_11 = stablehlo.multiply %adwdlrbo_11, %bo_11 : tensor<192xf32>
    %adnewbo_11 = stablehlo.subtract %adsubbo_11, %adwdpbo_11 : tensor<192xf32>
    %adb1g2_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1g2_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsg2_11 = stablehlo.multiply %adb1g2_11, %g2_11m : tensor<192xf32>
    %admgg2_11 = stablehlo.multiply %adob1g2_11, %vitb11_2dg : tensor<192xf32>
    %admng2_11 = stablehlo.add %admsg2_11, %admgg2_11 : tensor<192xf32>
    %adb2g2_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2g2_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsg2_11 = stablehlo.multiply %adb2g2_11, %g2_11v : tensor<192xf32>
    %adg2g2_11 = stablehlo.multiply %vitb11_2dg, %vitb11_2dg : tensor<192xf32>
    %advgg2_11 = stablehlo.multiply %adob2g2_11, %adg2g2_11 : tensor<192xf32>
    %advng2_11 = stablehlo.add %advsg2_11, %advgg2_11 : tensor<192xf32>
    %adbc1g2_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2g2_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhg2_11 = stablehlo.divide %admng2_11, %adbc1g2_11 : tensor<192xf32>
    %advhg2_11 = stablehlo.divide %advng2_11, %adbc2g2_11 : tensor<192xf32>
    %adlrg2_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsg2_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqg2_11 = stablehlo.sqrt %advhg2_11 : tensor<192xf32>
    %addeng2_11 = stablehlo.add %adsqg2_11, %adepsg2_11 : tensor<192xf32>
    %adratg2_11 = stablehlo.divide %admhg2_11, %addeng2_11 : tensor<192xf32>
    %adstg2_11 = stablehlo.multiply %adlrg2_11, %adratg2_11 : tensor<192xf32>
    %adsubg2_11 = stablehlo.subtract %g2_11, %adstg2_11 : tensor<192xf32>
    %adwdg2_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrg2_11 = stablehlo.multiply %adwdg2_11, %adlrg2_11 : tensor<192xf32>
    %adwdpg2_11 = stablehlo.multiply %adwdlrg2_11, %g2_11 : tensor<192xf32>
    %adnewg2_11 = stablehlo.subtract %adsubg2_11, %adwdpg2_11 : tensor<192xf32>
    %adb1b2_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1b2_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsb2_11 = stablehlo.multiply %adb1b2_11, %b2_11m : tensor<192xf32>
    %admgb2_11 = stablehlo.multiply %adob1b2_11, %vitb11_2db : tensor<192xf32>
    %admnb2_11 = stablehlo.add %admsb2_11, %admgb2_11 : tensor<192xf32>
    %adb2b2_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2b2_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsb2_11 = stablehlo.multiply %adb2b2_11, %b2_11v : tensor<192xf32>
    %adg2b2_11 = stablehlo.multiply %vitb11_2db, %vitb11_2db : tensor<192xf32>
    %advgb2_11 = stablehlo.multiply %adob2b2_11, %adg2b2_11 : tensor<192xf32>
    %advnb2_11 = stablehlo.add %advsb2_11, %advgb2_11 : tensor<192xf32>
    %adbc1b2_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2b2_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhb2_11 = stablehlo.divide %admnb2_11, %adbc1b2_11 : tensor<192xf32>
    %advhb2_11 = stablehlo.divide %advnb2_11, %adbc2b2_11 : tensor<192xf32>
    %adlrb2_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsb2_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqb2_11 = stablehlo.sqrt %advhb2_11 : tensor<192xf32>
    %addenb2_11 = stablehlo.add %adsqb2_11, %adepsb2_11 : tensor<192xf32>
    %adratb2_11 = stablehlo.divide %admhb2_11, %addenb2_11 : tensor<192xf32>
    %adstb2_11 = stablehlo.multiply %adlrb2_11, %adratb2_11 : tensor<192xf32>
    %adsubb2_11 = stablehlo.subtract %b2_11, %adstb2_11 : tensor<192xf32>
    %adwdb2_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrb2_11 = stablehlo.multiply %adwdb2_11, %adlrb2_11 : tensor<192xf32>
    %adwdpb2_11 = stablehlo.multiply %adwdlrb2_11, %b2_11 : tensor<192xf32>
    %adnewb2_11 = stablehlo.subtract %adsubb2_11, %adwdpb2_11 : tensor<192xf32>
    %adb1Wfc1_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob1Wfc1_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admsWfc1_11 = stablehlo.multiply %adb1Wfc1_11, %Wfc1_11m : tensor<192x768xf32>
    %admgWfc1_11 = stablehlo.multiply %adob1Wfc1_11, %vitb11_pdWfc1 : tensor<192x768xf32>
    %admnWfc1_11 = stablehlo.add %admsWfc1_11, %admgWfc1_11 : tensor<192x768xf32>
    %adb2Wfc1_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adob2Wfc1_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %advsWfc1_11 = stablehlo.multiply %adb2Wfc1_11, %Wfc1_11v : tensor<192x768xf32>
    %adg2Wfc1_11 = stablehlo.multiply %vitb11_pdWfc1, %vitb11_pdWfc1 : tensor<192x768xf32>
    %advgWfc1_11 = stablehlo.multiply %adob2Wfc1_11, %adg2Wfc1_11 : tensor<192x768xf32>
    %advnWfc1_11 = stablehlo.add %advsWfc1_11, %advgWfc1_11 : tensor<192x768xf32>
    %adbc1Wfc1_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adbc2Wfc1_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %admhWfc1_11 = stablehlo.divide %admnWfc1_11, %adbc1Wfc1_11 : tensor<192x768xf32>
    %advhWfc1_11 = stablehlo.divide %advnWfc1_11, %adbc2Wfc1_11 : tensor<192x768xf32>
    %adlrWfc1_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adepsWfc1_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adsqWfc1_11 = stablehlo.sqrt %advhWfc1_11 : tensor<192x768xf32>
    %addenWfc1_11 = stablehlo.add %adsqWfc1_11, %adepsWfc1_11 : tensor<192x768xf32>
    %adratWfc1_11 = stablehlo.divide %admhWfc1_11, %addenWfc1_11 : tensor<192x768xf32>
    %adstWfc1_11 = stablehlo.multiply %adlrWfc1_11, %adratWfc1_11 : tensor<192x768xf32>
    %adsubWfc1_11 = stablehlo.subtract %Wfc1_11, %adstWfc1_11 : tensor<192x768xf32>
    %adwdWfc1_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768xf32>
    %adwdlrWfc1_11 = stablehlo.multiply %adwdWfc1_11, %adlrWfc1_11 : tensor<192x768xf32>
    %adwdpWfc1_11 = stablehlo.multiply %adwdlrWfc1_11, %Wfc1_11 : tensor<192x768xf32>
    %adnewWfc1_11 = stablehlo.subtract %adsubWfc1_11, %adwdpWfc1_11 : tensor<192x768xf32>
    %adb1bfc1_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1bfc1_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admsbfc1_11 = stablehlo.multiply %adb1bfc1_11, %bfc1_11m : tensor<768xf32>
    %admgbfc1_11 = stablehlo.multiply %adob1bfc1_11, %vitb11_pdbfc1 : tensor<768xf32>
    %admnbfc1_11 = stablehlo.add %admsbfc1_11, %admgbfc1_11 : tensor<768xf32>
    %adb2bfc1_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2bfc1_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advsbfc1_11 = stablehlo.multiply %adb2bfc1_11, %bfc1_11v : tensor<768xf32>
    %adg2bfc1_11 = stablehlo.multiply %vitb11_pdbfc1, %vitb11_pdbfc1 : tensor<768xf32>
    %advgbfc1_11 = stablehlo.multiply %adob2bfc1_11, %adg2bfc1_11 : tensor<768xf32>
    %advnbfc1_11 = stablehlo.add %advsbfc1_11, %advgbfc1_11 : tensor<768xf32>
    %adbc1bfc1_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2bfc1_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhbfc1_11 = stablehlo.divide %admnbfc1_11, %adbc1bfc1_11 : tensor<768xf32>
    %advhbfc1_11 = stablehlo.divide %advnbfc1_11, %adbc2bfc1_11 : tensor<768xf32>
    %adlrbfc1_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepsbfc1_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqbfc1_11 = stablehlo.sqrt %advhbfc1_11 : tensor<768xf32>
    %addenbfc1_11 = stablehlo.add %adsqbfc1_11, %adepsbfc1_11 : tensor<768xf32>
    %adratbfc1_11 = stablehlo.divide %admhbfc1_11, %addenbfc1_11 : tensor<768xf32>
    %adstbfc1_11 = stablehlo.multiply %adlrbfc1_11, %adratbfc1_11 : tensor<768xf32>
    %adsubbfc1_11 = stablehlo.subtract %bfc1_11, %adstbfc1_11 : tensor<768xf32>
    %adwdbfc1_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrbfc1_11 = stablehlo.multiply %adwdbfc1_11, %adlrbfc1_11 : tensor<768xf32>
    %adwdpbfc1_11 = stablehlo.multiply %adwdlrbfc1_11, %bfc1_11 : tensor<768xf32>
    %adnewbfc1_11 = stablehlo.subtract %adsubbfc1_11, %adwdpbfc1_11 : tensor<768xf32>
    %adb1Wfc2_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob1Wfc2_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admsWfc2_11 = stablehlo.multiply %adb1Wfc2_11, %Wfc2_11m : tensor<768x192xf32>
    %admgWfc2_11 = stablehlo.multiply %adob1Wfc2_11, %vitb11_pdWfc2 : tensor<768x192xf32>
    %admnWfc2_11 = stablehlo.add %admsWfc2_11, %admgWfc2_11 : tensor<768x192xf32>
    %adb2Wfc2_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adob2Wfc2_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %advsWfc2_11 = stablehlo.multiply %adb2Wfc2_11, %Wfc2_11v : tensor<768x192xf32>
    %adg2Wfc2_11 = stablehlo.multiply %vitb11_pdWfc2, %vitb11_pdWfc2 : tensor<768x192xf32>
    %advgWfc2_11 = stablehlo.multiply %adob2Wfc2_11, %adg2Wfc2_11 : tensor<768x192xf32>
    %advnWfc2_11 = stablehlo.add %advsWfc2_11, %advgWfc2_11 : tensor<768x192xf32>
    %adbc1Wfc2_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adbc2Wfc2_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %admhWfc2_11 = stablehlo.divide %admnWfc2_11, %adbc1Wfc2_11 : tensor<768x192xf32>
    %advhWfc2_11 = stablehlo.divide %advnWfc2_11, %adbc2Wfc2_11 : tensor<768x192xf32>
    %adlrWfc2_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adepsWfc2_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adsqWfc2_11 = stablehlo.sqrt %advhWfc2_11 : tensor<768x192xf32>
    %addenWfc2_11 = stablehlo.add %adsqWfc2_11, %adepsWfc2_11 : tensor<768x192xf32>
    %adratWfc2_11 = stablehlo.divide %admhWfc2_11, %addenWfc2_11 : tensor<768x192xf32>
    %adstWfc2_11 = stablehlo.multiply %adlrWfc2_11, %adratWfc2_11 : tensor<768x192xf32>
    %adsubWfc2_11 = stablehlo.subtract %Wfc2_11, %adstWfc2_11 : tensor<768x192xf32>
    %adwdWfc2_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192xf32>
    %adwdlrWfc2_11 = stablehlo.multiply %adwdWfc2_11, %adlrWfc2_11 : tensor<768x192xf32>
    %adwdpWfc2_11 = stablehlo.multiply %adwdlrWfc2_11, %Wfc2_11 : tensor<768x192xf32>
    %adnewWfc2_11 = stablehlo.subtract %adsubWfc2_11, %adwdpWfc2_11 : tensor<768x192xf32>
    %adb1bfc2_11 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bfc2_11 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbfc2_11 = stablehlo.multiply %adb1bfc2_11, %bfc2_11m : tensor<192xf32>
    %admgbfc2_11 = stablehlo.multiply %adob1bfc2_11, %vitb11_pdbfc2 : tensor<192xf32>
    %admnbfc2_11 = stablehlo.add %admsbfc2_11, %admgbfc2_11 : tensor<192xf32>
    %adb2bfc2_11 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bfc2_11 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbfc2_11 = stablehlo.multiply %adb2bfc2_11, %bfc2_11v : tensor<192xf32>
    %adg2bfc2_11 = stablehlo.multiply %vitb11_pdbfc2, %vitb11_pdbfc2 : tensor<192xf32>
    %advgbfc2_11 = stablehlo.multiply %adob2bfc2_11, %adg2bfc2_11 : tensor<192xf32>
    %advnbfc2_11 = stablehlo.add %advsbfc2_11, %advgbfc2_11 : tensor<192xf32>
    %adbc1bfc2_11 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bfc2_11 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbfc2_11 = stablehlo.divide %admnbfc2_11, %adbc1bfc2_11 : tensor<192xf32>
    %advhbfc2_11 = stablehlo.divide %advnbfc2_11, %adbc2bfc2_11 : tensor<192xf32>
    %adlrbfc2_11 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbfc2_11 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbfc2_11 = stablehlo.sqrt %advhbfc2_11 : tensor<192xf32>
    %addenbfc2_11 = stablehlo.add %adsqbfc2_11, %adepsbfc2_11 : tensor<192xf32>
    %adratbfc2_11 = stablehlo.divide %admhbfc2_11, %addenbfc2_11 : tensor<192xf32>
    %adstbfc2_11 = stablehlo.multiply %adlrbfc2_11, %adratbfc2_11 : tensor<192xf32>
    %adsubbfc2_11 = stablehlo.subtract %bfc2_11, %adstbfc2_11 : tensor<192xf32>
    %adwdbfc2_11 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbfc2_11 = stablehlo.multiply %adwdbfc2_11, %adlrbfc2_11 : tensor<192xf32>
    %adwdpbfc2_11 = stablehlo.multiply %adwdlrbfc2_11, %bfc2_11 : tensor<192xf32>
    %adnewbfc2_11 = stablehlo.subtract %adsubbfc2_11, %adwdpbfc2_11 : tensor<192xf32>
    %adb1gF = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1gF = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsgF = stablehlo.multiply %adb1gF, %gFm : tensor<192xf32>
    %admggF = stablehlo.multiply %adob1gF, %vitflndg : tensor<192xf32>
    %admngF = stablehlo.add %admsgF, %admggF : tensor<192xf32>
    %adb2gF = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2gF = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsgF = stablehlo.multiply %adb2gF, %gFv : tensor<192xf32>
    %adg2gF = stablehlo.multiply %vitflndg, %vitflndg : tensor<192xf32>
    %advggF = stablehlo.multiply %adob2gF, %adg2gF : tensor<192xf32>
    %advngF = stablehlo.add %advsgF, %advggF : tensor<192xf32>
    %adbc1gF = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2gF = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhgF = stablehlo.divide %admngF, %adbc1gF : tensor<192xf32>
    %advhgF = stablehlo.divide %advngF, %adbc2gF : tensor<192xf32>
    %adlrgF = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsgF = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqgF = stablehlo.sqrt %advhgF : tensor<192xf32>
    %addengF = stablehlo.add %adsqgF, %adepsgF : tensor<192xf32>
    %adratgF = stablehlo.divide %admhgF, %addengF : tensor<192xf32>
    %adstgF = stablehlo.multiply %adlrgF, %adratgF : tensor<192xf32>
    %adsubgF = stablehlo.subtract %gF, %adstgF : tensor<192xf32>
    %adwdgF = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrgF = stablehlo.multiply %adwdgF, %adlrgF : tensor<192xf32>
    %adwdpgF = stablehlo.multiply %adwdlrgF, %gF : tensor<192xf32>
    %adnewgF = stablehlo.subtract %adsubgF, %adwdpgF : tensor<192xf32>
    %adb1bF = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1bF = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsbF = stablehlo.multiply %adb1bF, %bFm : tensor<192xf32>
    %admgbF = stablehlo.multiply %adob1bF, %vitflndb : tensor<192xf32>
    %admnbF = stablehlo.add %admsbF, %admgbF : tensor<192xf32>
    %adb2bF = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2bF = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsbF = stablehlo.multiply %adb2bF, %bFv : tensor<192xf32>
    %adg2bF = stablehlo.multiply %vitflndb, %vitflndb : tensor<192xf32>
    %advgbF = stablehlo.multiply %adob2bF, %adg2bF : tensor<192xf32>
    %advnbF = stablehlo.add %advsbF, %advgbF : tensor<192xf32>
    %adbc1bF = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2bF = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhbF = stablehlo.divide %admnbF, %adbc1bF : tensor<192xf32>
    %advhbF = stablehlo.divide %advnbF, %adbc2bF : tensor<192xf32>
    %adlrbF = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsbF = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqbF = stablehlo.sqrt %advhbF : tensor<192xf32>
    %addenbF = stablehlo.add %adsqbF, %adepsbF : tensor<192xf32>
    %adratbF = stablehlo.divide %admhbF, %addenbF : tensor<192xf32>
    %adstbF = stablehlo.multiply %adlrbF, %adratbF : tensor<192xf32>
    %adsubbF = stablehlo.subtract %bF, %adstbF : tensor<192xf32>
    %adwdbF = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrbF = stablehlo.multiply %adwdbF, %adlrbF : tensor<192xf32>
    %adwdpbF = stablehlo.multiply %adwdlrbF, %bF : tensor<192xf32>
    %adnewbF = stablehlo.subtract %adsubbF, %adwdpbF : tensor<192xf32>
    %adb1Wc = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x10xf32>
    %adob1Wc = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x10xf32>
    %admsWc = stablehlo.multiply %adb1Wc, %Wcm : tensor<192x10xf32>
    %admgWc = stablehlo.multiply %adob1Wc, %vithddWc : tensor<192x10xf32>
    %admnWc = stablehlo.add %admsWc, %admgWc : tensor<192x10xf32>
    %adb2Wc = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x10xf32>
    %adob2Wc = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x10xf32>
    %advsWc = stablehlo.multiply %adb2Wc, %Wcv : tensor<192x10xf32>
    %adg2Wc = stablehlo.multiply %vithddWc, %vithddWc : tensor<192x10xf32>
    %advgWc = stablehlo.multiply %adob2Wc, %adg2Wc : tensor<192x10xf32>
    %advnWc = stablehlo.add %advsWc, %advgWc : tensor<192x10xf32>
    %adbc1Wc = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x10xf32>
    %adbc2Wc = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x10xf32>
    %admhWc = stablehlo.divide %admnWc, %adbc1Wc : tensor<192x10xf32>
    %advhWc = stablehlo.divide %advnWc, %adbc2Wc : tensor<192x10xf32>
    %adlrWc = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x10xf32>
    %adepsWc = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x10xf32>
    %adsqWc = stablehlo.sqrt %advhWc : tensor<192x10xf32>
    %addenWc = stablehlo.add %adsqWc, %adepsWc : tensor<192x10xf32>
    %adratWc = stablehlo.divide %admhWc, %addenWc : tensor<192x10xf32>
    %adstWc = stablehlo.multiply %adlrWc, %adratWc : tensor<192x10xf32>
    %adsubWc = stablehlo.subtract %Wc, %adstWc : tensor<192x10xf32>
    %adwdWc = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x10xf32>
    %adwdlrWc = stablehlo.multiply %adwdWc, %adlrWc : tensor<192x10xf32>
    %adwdpWc = stablehlo.multiply %adwdlrWc, %Wc : tensor<192x10xf32>
    %adnewWc = stablehlo.subtract %adsubWc, %adwdpWc : tensor<192x10xf32>
    %adb1bc = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adob1bc = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %admsbc = stablehlo.multiply %adb1bc, %bcm : tensor<10xf32>
    %admgbc = stablehlo.multiply %adob1bc, %vithddbc : tensor<10xf32>
    %admnbc = stablehlo.add %admsbc, %admgbc : tensor<10xf32>
    %adb2bc = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adob2bc = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %advsbc = stablehlo.multiply %adb2bc, %bcv : tensor<10xf32>
    %adg2bc = stablehlo.multiply %vithddbc, %vithddbc : tensor<10xf32>
    %advgbc = stablehlo.multiply %adob2bc, %adg2bc : tensor<10xf32>
    %advnbc = stablehlo.add %advsbc, %advgbc : tensor<10xf32>
    %adbc1bc = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adbc2bc = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %admhbc = stablehlo.divide %admnbc, %adbc1bc : tensor<10xf32>
    %advhbc = stablehlo.divide %advnbc, %adbc2bc : tensor<10xf32>
    %adlrbc = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adepsbc = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adsqbc = stablehlo.sqrt %advhbc : tensor<10xf32>
    %addenbc = stablehlo.add %adsqbc, %adepsbc : tensor<10xf32>
    %adratbc = stablehlo.divide %admhbc, %addenbc : tensor<10xf32>
    %adstbc = stablehlo.multiply %adlrbc, %adratbc : tensor<10xf32>
    %adsubbc = stablehlo.subtract %bc, %adstbc : tensor<10xf32>
    %adwdbc = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adwdlrbc = stablehlo.multiply %adwdbc, %adlrbc : tensor<10xf32>
    %adwdpbc = stablehlo.multiply %adwdlrbc, %bc : tensor<10xf32>
    %adnewbc = stablehlo.subtract %adsubbc, %adwdpbc : tensor<10xf32>
    return %adnewwConv, %adnewbConv, %adnewcls, %adnewpos, %adnewg1_0, %adnewb1_0, %adnewWq_0, %adnewbq_0, %adnewWk_0, %adnewbk_0, %adnewWv_0, %adnewbv_0, %adnewWo_0, %adnewbo_0, %adnewg2_0, %adnewb2_0, %adnewWfc1_0, %adnewbfc1_0, %adnewWfc2_0, %adnewbfc2_0, %adnewg1_1, %adnewb1_1, %adnewWq_1, %adnewbq_1, %adnewWk_1, %adnewbk_1, %adnewWv_1, %adnewbv_1, %adnewWo_1, %adnewbo_1, %adnewg2_1, %adnewb2_1, %adnewWfc1_1, %adnewbfc1_1, %adnewWfc2_1, %adnewbfc2_1, %adnewg1_2, %adnewb1_2, %adnewWq_2, %adnewbq_2, %adnewWk_2, %adnewbk_2, %adnewWv_2, %adnewbv_2, %adnewWo_2, %adnewbo_2, %adnewg2_2, %adnewb2_2, %adnewWfc1_2, %adnewbfc1_2, %adnewWfc2_2, %adnewbfc2_2, %adnewg1_3, %adnewb1_3, %adnewWq_3, %adnewbq_3, %adnewWk_3, %adnewbk_3, %adnewWv_3, %adnewbv_3, %adnewWo_3, %adnewbo_3, %adnewg2_3, %adnewb2_3, %adnewWfc1_3, %adnewbfc1_3, %adnewWfc2_3, %adnewbfc2_3, %adnewg1_4, %adnewb1_4, %adnewWq_4, %adnewbq_4, %adnewWk_4, %adnewbk_4, %adnewWv_4, %adnewbv_4, %adnewWo_4, %adnewbo_4, %adnewg2_4, %adnewb2_4, %adnewWfc1_4, %adnewbfc1_4, %adnewWfc2_4, %adnewbfc2_4, %adnewg1_5, %adnewb1_5, %adnewWq_5, %adnewbq_5, %adnewWk_5, %adnewbk_5, %adnewWv_5, %adnewbv_5, %adnewWo_5, %adnewbo_5, %adnewg2_5, %adnewb2_5, %adnewWfc1_5, %adnewbfc1_5, %adnewWfc2_5, %adnewbfc2_5, %adnewg1_6, %adnewb1_6, %adnewWq_6, %adnewbq_6, %adnewWk_6, %adnewbk_6, %adnewWv_6, %adnewbv_6, %adnewWo_6, %adnewbo_6, %adnewg2_6, %adnewb2_6, %adnewWfc1_6, %adnewbfc1_6, %adnewWfc2_6, %adnewbfc2_6, %adnewg1_7, %adnewb1_7, %adnewWq_7, %adnewbq_7, %adnewWk_7, %adnewbk_7, %adnewWv_7, %adnewbv_7, %adnewWo_7, %adnewbo_7, %adnewg2_7, %adnewb2_7, %adnewWfc1_7, %adnewbfc1_7, %adnewWfc2_7, %adnewbfc2_7, %adnewg1_8, %adnewb1_8, %adnewWq_8, %adnewbq_8, %adnewWk_8, %adnewbk_8, %adnewWv_8, %adnewbv_8, %adnewWo_8, %adnewbo_8, %adnewg2_8, %adnewb2_8, %adnewWfc1_8, %adnewbfc1_8, %adnewWfc2_8, %adnewbfc2_8, %adnewg1_9, %adnewb1_9, %adnewWq_9, %adnewbq_9, %adnewWk_9, %adnewbk_9, %adnewWv_9, %adnewbv_9, %adnewWo_9, %adnewbo_9, %adnewg2_9, %adnewb2_9, %adnewWfc1_9, %adnewbfc1_9, %adnewWfc2_9, %adnewbfc2_9, %adnewg1_10, %adnewb1_10, %adnewWq_10, %adnewbq_10, %adnewWk_10, %adnewbk_10, %adnewWv_10, %adnewbv_10, %adnewWo_10, %adnewbo_10, %adnewg2_10, %adnewb2_10, %adnewWfc1_10, %adnewbfc1_10, %adnewWfc2_10, %adnewbfc2_10, %adnewg1_11, %adnewb1_11, %adnewWq_11, %adnewbq_11, %adnewWk_11, %adnewbk_11, %adnewWv_11, %adnewbv_11, %adnewWo_11, %adnewbo_11, %adnewg2_11, %adnewb2_11, %adnewWfc1_11, %adnewbfc1_11, %adnewWfc2_11, %adnewbfc2_11, %adnewgF, %adnewbF, %adnewWc, %adnewbc, %admnwConv, %admnbConv, %admncls, %admnpos, %admng1_0, %admnb1_0, %admnWq_0, %admnbq_0, %admnWk_0, %admnbk_0, %admnWv_0, %admnbv_0, %admnWo_0, %admnbo_0, %admng2_0, %admnb2_0, %admnWfc1_0, %admnbfc1_0, %admnWfc2_0, %admnbfc2_0, %admng1_1, %admnb1_1, %admnWq_1, %admnbq_1, %admnWk_1, %admnbk_1, %admnWv_1, %admnbv_1, %admnWo_1, %admnbo_1, %admng2_1, %admnb2_1, %admnWfc1_1, %admnbfc1_1, %admnWfc2_1, %admnbfc2_1, %admng1_2, %admnb1_2, %admnWq_2, %admnbq_2, %admnWk_2, %admnbk_2, %admnWv_2, %admnbv_2, %admnWo_2, %admnbo_2, %admng2_2, %admnb2_2, %admnWfc1_2, %admnbfc1_2, %admnWfc2_2, %admnbfc2_2, %admng1_3, %admnb1_3, %admnWq_3, %admnbq_3, %admnWk_3, %admnbk_3, %admnWv_3, %admnbv_3, %admnWo_3, %admnbo_3, %admng2_3, %admnb2_3, %admnWfc1_3, %admnbfc1_3, %admnWfc2_3, %admnbfc2_3, %admng1_4, %admnb1_4, %admnWq_4, %admnbq_4, %admnWk_4, %admnbk_4, %admnWv_4, %admnbv_4, %admnWo_4, %admnbo_4, %admng2_4, %admnb2_4, %admnWfc1_4, %admnbfc1_4, %admnWfc2_4, %admnbfc2_4, %admng1_5, %admnb1_5, %admnWq_5, %admnbq_5, %admnWk_5, %admnbk_5, %admnWv_5, %admnbv_5, %admnWo_5, %admnbo_5, %admng2_5, %admnb2_5, %admnWfc1_5, %admnbfc1_5, %admnWfc2_5, %admnbfc2_5, %admng1_6, %admnb1_6, %admnWq_6, %admnbq_6, %admnWk_6, %admnbk_6, %admnWv_6, %admnbv_6, %admnWo_6, %admnbo_6, %admng2_6, %admnb2_6, %admnWfc1_6, %admnbfc1_6, %admnWfc2_6, %admnbfc2_6, %admng1_7, %admnb1_7, %admnWq_7, %admnbq_7, %admnWk_7, %admnbk_7, %admnWv_7, %admnbv_7, %admnWo_7, %admnbo_7, %admng2_7, %admnb2_7, %admnWfc1_7, %admnbfc1_7, %admnWfc2_7, %admnbfc2_7, %admng1_8, %admnb1_8, %admnWq_8, %admnbq_8, %admnWk_8, %admnbk_8, %admnWv_8, %admnbv_8, %admnWo_8, %admnbo_8, %admng2_8, %admnb2_8, %admnWfc1_8, %admnbfc1_8, %admnWfc2_8, %admnbfc2_8, %admng1_9, %admnb1_9, %admnWq_9, %admnbq_9, %admnWk_9, %admnbk_9, %admnWv_9, %admnbv_9, %admnWo_9, %admnbo_9, %admng2_9, %admnb2_9, %admnWfc1_9, %admnbfc1_9, %admnWfc2_9, %admnbfc2_9, %admng1_10, %admnb1_10, %admnWq_10, %admnbq_10, %admnWk_10, %admnbk_10, %admnWv_10, %admnbv_10, %admnWo_10, %admnbo_10, %admng2_10, %admnb2_10, %admnWfc1_10, %admnbfc1_10, %admnWfc2_10, %admnbfc2_10, %admng1_11, %admnb1_11, %admnWq_11, %admnbq_11, %admnWk_11, %admnbk_11, %admnWv_11, %admnbv_11, %admnWo_11, %admnbo_11, %admng2_11, %admnb2_11, %admnWfc1_11, %admnbfc1_11, %admnWfc2_11, %admnbfc2_11, %admngF, %admnbF, %admnWc, %admnbc, %advnwConv, %advnbConv, %advncls, %advnpos, %advng1_0, %advnb1_0, %advnWq_0, %advnbq_0, %advnWk_0, %advnbk_0, %advnWv_0, %advnbv_0, %advnWo_0, %advnbo_0, %advng2_0, %advnb2_0, %advnWfc1_0, %advnbfc1_0, %advnWfc2_0, %advnbfc2_0, %advng1_1, %advnb1_1, %advnWq_1, %advnbq_1, %advnWk_1, %advnbk_1, %advnWv_1, %advnbv_1, %advnWo_1, %advnbo_1, %advng2_1, %advnb2_1, %advnWfc1_1, %advnbfc1_1, %advnWfc2_1, %advnbfc2_1, %advng1_2, %advnb1_2, %advnWq_2, %advnbq_2, %advnWk_2, %advnbk_2, %advnWv_2, %advnbv_2, %advnWo_2, %advnbo_2, %advng2_2, %advnb2_2, %advnWfc1_2, %advnbfc1_2, %advnWfc2_2, %advnbfc2_2, %advng1_3, %advnb1_3, %advnWq_3, %advnbq_3, %advnWk_3, %advnbk_3, %advnWv_3, %advnbv_3, %advnWo_3, %advnbo_3, %advng2_3, %advnb2_3, %advnWfc1_3, %advnbfc1_3, %advnWfc2_3, %advnbfc2_3, %advng1_4, %advnb1_4, %advnWq_4, %advnbq_4, %advnWk_4, %advnbk_4, %advnWv_4, %advnbv_4, %advnWo_4, %advnbo_4, %advng2_4, %advnb2_4, %advnWfc1_4, %advnbfc1_4, %advnWfc2_4, %advnbfc2_4, %advng1_5, %advnb1_5, %advnWq_5, %advnbq_5, %advnWk_5, %advnbk_5, %advnWv_5, %advnbv_5, %advnWo_5, %advnbo_5, %advng2_5, %advnb2_5, %advnWfc1_5, %advnbfc1_5, %advnWfc2_5, %advnbfc2_5, %advng1_6, %advnb1_6, %advnWq_6, %advnbq_6, %advnWk_6, %advnbk_6, %advnWv_6, %advnbv_6, %advnWo_6, %advnbo_6, %advng2_6, %advnb2_6, %advnWfc1_6, %advnbfc1_6, %advnWfc2_6, %advnbfc2_6, %advng1_7, %advnb1_7, %advnWq_7, %advnbq_7, %advnWk_7, %advnbk_7, %advnWv_7, %advnbv_7, %advnWo_7, %advnbo_7, %advng2_7, %advnb2_7, %advnWfc1_7, %advnbfc1_7, %advnWfc2_7, %advnbfc2_7, %advng1_8, %advnb1_8, %advnWq_8, %advnbq_8, %advnWk_8, %advnbk_8, %advnWv_8, %advnbv_8, %advnWo_8, %advnbo_8, %advng2_8, %advnb2_8, %advnWfc1_8, %advnbfc1_8, %advnWfc2_8, %advnbfc2_8, %advng1_9, %advnb1_9, %advnWq_9, %advnbq_9, %advnWk_9, %advnbk_9, %advnWv_9, %advnbv_9, %advnWo_9, %advnbo_9, %advng2_9, %advnb2_9, %advnWfc1_9, %advnbfc1_9, %advnWfc2_9, %advnbfc2_9, %advng1_10, %advnb1_10, %advnWq_10, %advnbq_10, %advnWk_10, %advnbk_10, %advnWv_10, %advnbv_10, %advnWo_10, %advnbo_10, %advng2_10, %advnb2_10, %advnWfc1_10, %advnbfc1_10, %advnWfc2_10, %advnbfc2_10, %advng1_11, %advnb1_11, %advnWq_11, %advnbq_11, %advnWk_11, %advnbk_11, %advnWv_11, %advnbv_11, %advnWo_11, %advnbo_11, %advng2_11, %advnb2_11, %advnWfc1_11, %advnbfc1_11, %advnWfc2_11, %advnbfc2_11, %advngF, %advnbF, %advnWc, %advnbc, %loss, %bc1, %bc2 : tensor<192x3x16x16xf32>, tensor<192xf32>, tensor<192xf32>, tensor<197x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x10xf32>, tensor<10xf32>, tensor<192x3x16x16xf32>, tensor<192xf32>, tensor<192xf32>, tensor<197x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x10xf32>, tensor<10xf32>, tensor<192x3x16x16xf32>, tensor<192xf32>, tensor<192xf32>, tensor<197x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}

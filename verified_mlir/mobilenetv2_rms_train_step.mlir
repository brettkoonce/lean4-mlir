module @m {
  func.func @mobilenetv2_rms_train_step(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x3x3xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4pW: tensor<32x144x1x1xf32>, %b4pg: tensor<32xf32>, %b4pbt: tensor<32xf32>, %b5eW: tensor<192x32x1x1xf32>, %b5eg: tensor<192xf32>, %b5ebt: tensor<192xf32>, %b5dW: tensor<192x1x3x3xf32>, %b5dg: tensor<192xf32>, %b5dbt: tensor<192xf32>, %b5pW: tensor<32x192x1x1xf32>, %b5pg: tensor<32xf32>, %b5pbt: tensor<32xf32>, %b6eW: tensor<192x32x1x1xf32>, %b6eg: tensor<192xf32>, %b6ebt: tensor<192xf32>, %b6dW: tensor<192x1x3x3xf32>, %b6dg: tensor<192xf32>, %b6dbt: tensor<192xf32>, %b6pW: tensor<32x192x1x1xf32>, %b6pg: tensor<32xf32>, %b6pbt: tensor<32xf32>, %b7eW: tensor<192x32x1x1xf32>, %b7eg: tensor<192xf32>, %b7ebt: tensor<192xf32>, %b7dW: tensor<192x1x3x3xf32>, %b7dg: tensor<192xf32>, %b7dbt: tensor<192xf32>, %b7pW: tensor<64x192x1x1xf32>, %b7pg: tensor<64xf32>, %b7pbt: tensor<64xf32>, %b8eW: tensor<384x64x1x1xf32>, %b8eg: tensor<384xf32>, %b8ebt: tensor<384xf32>, %b8dW: tensor<384x1x3x3xf32>, %b8dg: tensor<384xf32>, %b8dbt: tensor<384xf32>, %b8pW: tensor<64x384x1x1xf32>, %b8pg: tensor<64xf32>, %b8pbt: tensor<64xf32>, %b9eW: tensor<384x64x1x1xf32>, %b9eg: tensor<384xf32>, %b9ebt: tensor<384xf32>, %b9dW: tensor<384x1x3x3xf32>, %b9dg: tensor<384xf32>, %b9dbt: tensor<384xf32>, %b9pW: tensor<64x384x1x1xf32>, %b9pg: tensor<64xf32>, %b9pbt: tensor<64xf32>, %b10eW: tensor<384x64x1x1xf32>, %b10eg: tensor<384xf32>, %b10ebt: tensor<384xf32>, %b10dW: tensor<384x1x3x3xf32>, %b10dg: tensor<384xf32>, %b10dbt: tensor<384xf32>, %b10pW: tensor<64x384x1x1xf32>, %b10pg: tensor<64xf32>, %b10pbt: tensor<64xf32>, %b11eW: tensor<384x64x1x1xf32>, %b11eg: tensor<384xf32>, %b11ebt: tensor<384xf32>, %b11dW: tensor<384x1x3x3xf32>, %b11dg: tensor<384xf32>, %b11dbt: tensor<384xf32>, %b11pW: tensor<96x384x1x1xf32>, %b11pg: tensor<96xf32>, %b11pbt: tensor<96xf32>, %b12eW: tensor<576x96x1x1xf32>, %b12eg: tensor<576xf32>, %b12ebt: tensor<576xf32>, %b12dW: tensor<576x1x3x3xf32>, %b12dg: tensor<576xf32>, %b12dbt: tensor<576xf32>, %b12pW: tensor<96x576x1x1xf32>, %b12pg: tensor<96xf32>, %b12pbt: tensor<96xf32>, %b13eW: tensor<576x96x1x1xf32>, %b13eg: tensor<576xf32>, %b13ebt: tensor<576xf32>, %b13dW: tensor<576x1x3x3xf32>, %b13dg: tensor<576xf32>, %b13dbt: tensor<576xf32>, %b13pW: tensor<96x576x1x1xf32>, %b13pg: tensor<96xf32>, %b13pbt: tensor<96xf32>, %b14eW: tensor<576x96x1x1xf32>, %b14eg: tensor<576xf32>, %b14ebt: tensor<576xf32>, %b14dW: tensor<576x1x3x3xf32>, %b14dg: tensor<576xf32>, %b14dbt: tensor<576xf32>, %b14pW: tensor<160x576x1x1xf32>, %b14pg: tensor<160xf32>, %b14pbt: tensor<160xf32>, %b15eW: tensor<960x160x1x1xf32>, %b15eg: tensor<960xf32>, %b15ebt: tensor<960xf32>, %b15dW: tensor<960x1x3x3xf32>, %b15dg: tensor<960xf32>, %b15dbt: tensor<960xf32>, %b15pW: tensor<160x960x1x1xf32>, %b15pg: tensor<160xf32>, %b15pbt: tensor<160xf32>, %b16eW: tensor<960x160x1x1xf32>, %b16eg: tensor<960xf32>, %b16ebt: tensor<960xf32>, %b16dW: tensor<960x1x3x3xf32>, %b16dg: tensor<960xf32>, %b16dbt: tensor<960xf32>, %b16pW: tensor<160x960x1x1xf32>, %b16pg: tensor<160xf32>, %b16pbt: tensor<160xf32>, %b17eW: tensor<960x160x1x1xf32>, %b17eg: tensor<960xf32>, %b17ebt: tensor<960xf32>, %b17dW: tensor<960x1x3x3xf32>, %b17dg: tensor<960xf32>, %b17dbt: tensor<960xf32>, %b17pW: tensor<320x960x1x1xf32>, %b17pg: tensor<320xf32>, %b17pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %sWm: tensor<32x3x3x3xf32>, %sgm: tensor<32xf32>, %sbtm: tensor<32xf32>, %b1dWm: tensor<32x1x3x3xf32>, %b1dgm: tensor<32xf32>, %b1dbtm: tensor<32xf32>, %b1pWm: tensor<16x32x1x1xf32>, %b1pgm: tensor<16xf32>, %b1pbtm: tensor<16xf32>, %b2eWm: tensor<96x16x1x1xf32>, %b2egm: tensor<96xf32>, %b2ebtm: tensor<96xf32>, %b2dWm: tensor<96x1x3x3xf32>, %b2dgm: tensor<96xf32>, %b2dbtm: tensor<96xf32>, %b2pWm: tensor<24x96x1x1xf32>, %b2pgm: tensor<24xf32>, %b2pbtm: tensor<24xf32>, %b3eWm: tensor<144x24x1x1xf32>, %b3egm: tensor<144xf32>, %b3ebtm: tensor<144xf32>, %b3dWm: tensor<144x1x3x3xf32>, %b3dgm: tensor<144xf32>, %b3dbtm: tensor<144xf32>, %b3pWm: tensor<24x144x1x1xf32>, %b3pgm: tensor<24xf32>, %b3pbtm: tensor<24xf32>, %b4eWm: tensor<144x24x1x1xf32>, %b4egm: tensor<144xf32>, %b4ebtm: tensor<144xf32>, %b4dWm: tensor<144x1x3x3xf32>, %b4dgm: tensor<144xf32>, %b4dbtm: tensor<144xf32>, %b4pWm: tensor<32x144x1x1xf32>, %b4pgm: tensor<32xf32>, %b4pbtm: tensor<32xf32>, %b5eWm: tensor<192x32x1x1xf32>, %b5egm: tensor<192xf32>, %b5ebtm: tensor<192xf32>, %b5dWm: tensor<192x1x3x3xf32>, %b5dgm: tensor<192xf32>, %b5dbtm: tensor<192xf32>, %b5pWm: tensor<32x192x1x1xf32>, %b5pgm: tensor<32xf32>, %b5pbtm: tensor<32xf32>, %b6eWm: tensor<192x32x1x1xf32>, %b6egm: tensor<192xf32>, %b6ebtm: tensor<192xf32>, %b6dWm: tensor<192x1x3x3xf32>, %b6dgm: tensor<192xf32>, %b6dbtm: tensor<192xf32>, %b6pWm: tensor<32x192x1x1xf32>, %b6pgm: tensor<32xf32>, %b6pbtm: tensor<32xf32>, %b7eWm: tensor<192x32x1x1xf32>, %b7egm: tensor<192xf32>, %b7ebtm: tensor<192xf32>, %b7dWm: tensor<192x1x3x3xf32>, %b7dgm: tensor<192xf32>, %b7dbtm: tensor<192xf32>, %b7pWm: tensor<64x192x1x1xf32>, %b7pgm: tensor<64xf32>, %b7pbtm: tensor<64xf32>, %b8eWm: tensor<384x64x1x1xf32>, %b8egm: tensor<384xf32>, %b8ebtm: tensor<384xf32>, %b8dWm: tensor<384x1x3x3xf32>, %b8dgm: tensor<384xf32>, %b8dbtm: tensor<384xf32>, %b8pWm: tensor<64x384x1x1xf32>, %b8pgm: tensor<64xf32>, %b8pbtm: tensor<64xf32>, %b9eWm: tensor<384x64x1x1xf32>, %b9egm: tensor<384xf32>, %b9ebtm: tensor<384xf32>, %b9dWm: tensor<384x1x3x3xf32>, %b9dgm: tensor<384xf32>, %b9dbtm: tensor<384xf32>, %b9pWm: tensor<64x384x1x1xf32>, %b9pgm: tensor<64xf32>, %b9pbtm: tensor<64xf32>, %b10eWm: tensor<384x64x1x1xf32>, %b10egm: tensor<384xf32>, %b10ebtm: tensor<384xf32>, %b10dWm: tensor<384x1x3x3xf32>, %b10dgm: tensor<384xf32>, %b10dbtm: tensor<384xf32>, %b10pWm: tensor<64x384x1x1xf32>, %b10pgm: tensor<64xf32>, %b10pbtm: tensor<64xf32>, %b11eWm: tensor<384x64x1x1xf32>, %b11egm: tensor<384xf32>, %b11ebtm: tensor<384xf32>, %b11dWm: tensor<384x1x3x3xf32>, %b11dgm: tensor<384xf32>, %b11dbtm: tensor<384xf32>, %b11pWm: tensor<96x384x1x1xf32>, %b11pgm: tensor<96xf32>, %b11pbtm: tensor<96xf32>, %b12eWm: tensor<576x96x1x1xf32>, %b12egm: tensor<576xf32>, %b12ebtm: tensor<576xf32>, %b12dWm: tensor<576x1x3x3xf32>, %b12dgm: tensor<576xf32>, %b12dbtm: tensor<576xf32>, %b12pWm: tensor<96x576x1x1xf32>, %b12pgm: tensor<96xf32>, %b12pbtm: tensor<96xf32>, %b13eWm: tensor<576x96x1x1xf32>, %b13egm: tensor<576xf32>, %b13ebtm: tensor<576xf32>, %b13dWm: tensor<576x1x3x3xf32>, %b13dgm: tensor<576xf32>, %b13dbtm: tensor<576xf32>, %b13pWm: tensor<96x576x1x1xf32>, %b13pgm: tensor<96xf32>, %b13pbtm: tensor<96xf32>, %b14eWm: tensor<576x96x1x1xf32>, %b14egm: tensor<576xf32>, %b14ebtm: tensor<576xf32>, %b14dWm: tensor<576x1x3x3xf32>, %b14dgm: tensor<576xf32>, %b14dbtm: tensor<576xf32>, %b14pWm: tensor<160x576x1x1xf32>, %b14pgm: tensor<160xf32>, %b14pbtm: tensor<160xf32>, %b15eWm: tensor<960x160x1x1xf32>, %b15egm: tensor<960xf32>, %b15ebtm: tensor<960xf32>, %b15dWm: tensor<960x1x3x3xf32>, %b15dgm: tensor<960xf32>, %b15dbtm: tensor<960xf32>, %b15pWm: tensor<160x960x1x1xf32>, %b15pgm: tensor<160xf32>, %b15pbtm: tensor<160xf32>, %b16eWm: tensor<960x160x1x1xf32>, %b16egm: tensor<960xf32>, %b16ebtm: tensor<960xf32>, %b16dWm: tensor<960x1x3x3xf32>, %b16dgm: tensor<960xf32>, %b16dbtm: tensor<960xf32>, %b16pWm: tensor<160x960x1x1xf32>, %b16pgm: tensor<160xf32>, %b16pbtm: tensor<160xf32>, %b17eWm: tensor<960x160x1x1xf32>, %b17egm: tensor<960xf32>, %b17ebtm: tensor<960xf32>, %b17dWm: tensor<960x1x3x3xf32>, %b17dgm: tensor<960xf32>, %b17dbtm: tensor<960xf32>, %b17pWm: tensor<320x960x1x1xf32>, %b17pgm: tensor<320xf32>, %b17pbtm: tensor<320xf32>, %hWm: tensor<1280x320x1x1xf32>, %hgm: tensor<1280xf32>, %hbtm: tensor<1280xf32>, %Wdm: tensor<1280x10xf32>, %bdm: tensor<10xf32>, %sWv: tensor<32x3x3x3xf32>, %sgv: tensor<32xf32>, %sbtv: tensor<32xf32>, %b1dWv: tensor<32x1x3x3xf32>, %b1dgv: tensor<32xf32>, %b1dbtv: tensor<32xf32>, %b1pWv: tensor<16x32x1x1xf32>, %b1pgv: tensor<16xf32>, %b1pbtv: tensor<16xf32>, %b2eWv: tensor<96x16x1x1xf32>, %b2egv: tensor<96xf32>, %b2ebtv: tensor<96xf32>, %b2dWv: tensor<96x1x3x3xf32>, %b2dgv: tensor<96xf32>, %b2dbtv: tensor<96xf32>, %b2pWv: tensor<24x96x1x1xf32>, %b2pgv: tensor<24xf32>, %b2pbtv: tensor<24xf32>, %b3eWv: tensor<144x24x1x1xf32>, %b3egv: tensor<144xf32>, %b3ebtv: tensor<144xf32>, %b3dWv: tensor<144x1x3x3xf32>, %b3dgv: tensor<144xf32>, %b3dbtv: tensor<144xf32>, %b3pWv: tensor<24x144x1x1xf32>, %b3pgv: tensor<24xf32>, %b3pbtv: tensor<24xf32>, %b4eWv: tensor<144x24x1x1xf32>, %b4egv: tensor<144xf32>, %b4ebtv: tensor<144xf32>, %b4dWv: tensor<144x1x3x3xf32>, %b4dgv: tensor<144xf32>, %b4dbtv: tensor<144xf32>, %b4pWv: tensor<32x144x1x1xf32>, %b4pgv: tensor<32xf32>, %b4pbtv: tensor<32xf32>, %b5eWv: tensor<192x32x1x1xf32>, %b5egv: tensor<192xf32>, %b5ebtv: tensor<192xf32>, %b5dWv: tensor<192x1x3x3xf32>, %b5dgv: tensor<192xf32>, %b5dbtv: tensor<192xf32>, %b5pWv: tensor<32x192x1x1xf32>, %b5pgv: tensor<32xf32>, %b5pbtv: tensor<32xf32>, %b6eWv: tensor<192x32x1x1xf32>, %b6egv: tensor<192xf32>, %b6ebtv: tensor<192xf32>, %b6dWv: tensor<192x1x3x3xf32>, %b6dgv: tensor<192xf32>, %b6dbtv: tensor<192xf32>, %b6pWv: tensor<32x192x1x1xf32>, %b6pgv: tensor<32xf32>, %b6pbtv: tensor<32xf32>, %b7eWv: tensor<192x32x1x1xf32>, %b7egv: tensor<192xf32>, %b7ebtv: tensor<192xf32>, %b7dWv: tensor<192x1x3x3xf32>, %b7dgv: tensor<192xf32>, %b7dbtv: tensor<192xf32>, %b7pWv: tensor<64x192x1x1xf32>, %b7pgv: tensor<64xf32>, %b7pbtv: tensor<64xf32>, %b8eWv: tensor<384x64x1x1xf32>, %b8egv: tensor<384xf32>, %b8ebtv: tensor<384xf32>, %b8dWv: tensor<384x1x3x3xf32>, %b8dgv: tensor<384xf32>, %b8dbtv: tensor<384xf32>, %b8pWv: tensor<64x384x1x1xf32>, %b8pgv: tensor<64xf32>, %b8pbtv: tensor<64xf32>, %b9eWv: tensor<384x64x1x1xf32>, %b9egv: tensor<384xf32>, %b9ebtv: tensor<384xf32>, %b9dWv: tensor<384x1x3x3xf32>, %b9dgv: tensor<384xf32>, %b9dbtv: tensor<384xf32>, %b9pWv: tensor<64x384x1x1xf32>, %b9pgv: tensor<64xf32>, %b9pbtv: tensor<64xf32>, %b10eWv: tensor<384x64x1x1xf32>, %b10egv: tensor<384xf32>, %b10ebtv: tensor<384xf32>, %b10dWv: tensor<384x1x3x3xf32>, %b10dgv: tensor<384xf32>, %b10dbtv: tensor<384xf32>, %b10pWv: tensor<64x384x1x1xf32>, %b10pgv: tensor<64xf32>, %b10pbtv: tensor<64xf32>, %b11eWv: tensor<384x64x1x1xf32>, %b11egv: tensor<384xf32>, %b11ebtv: tensor<384xf32>, %b11dWv: tensor<384x1x3x3xf32>, %b11dgv: tensor<384xf32>, %b11dbtv: tensor<384xf32>, %b11pWv: tensor<96x384x1x1xf32>, %b11pgv: tensor<96xf32>, %b11pbtv: tensor<96xf32>, %b12eWv: tensor<576x96x1x1xf32>, %b12egv: tensor<576xf32>, %b12ebtv: tensor<576xf32>, %b12dWv: tensor<576x1x3x3xf32>, %b12dgv: tensor<576xf32>, %b12dbtv: tensor<576xf32>, %b12pWv: tensor<96x576x1x1xf32>, %b12pgv: tensor<96xf32>, %b12pbtv: tensor<96xf32>, %b13eWv: tensor<576x96x1x1xf32>, %b13egv: tensor<576xf32>, %b13ebtv: tensor<576xf32>, %b13dWv: tensor<576x1x3x3xf32>, %b13dgv: tensor<576xf32>, %b13dbtv: tensor<576xf32>, %b13pWv: tensor<96x576x1x1xf32>, %b13pgv: tensor<96xf32>, %b13pbtv: tensor<96xf32>, %b14eWv: tensor<576x96x1x1xf32>, %b14egv: tensor<576xf32>, %b14ebtv: tensor<576xf32>, %b14dWv: tensor<576x1x3x3xf32>, %b14dgv: tensor<576xf32>, %b14dbtv: tensor<576xf32>, %b14pWv: tensor<160x576x1x1xf32>, %b14pgv: tensor<160xf32>, %b14pbtv: tensor<160xf32>, %b15eWv: tensor<960x160x1x1xf32>, %b15egv: tensor<960xf32>, %b15ebtv: tensor<960xf32>, %b15dWv: tensor<960x1x3x3xf32>, %b15dgv: tensor<960xf32>, %b15dbtv: tensor<960xf32>, %b15pWv: tensor<160x960x1x1xf32>, %b15pgv: tensor<160xf32>, %b15pbtv: tensor<160xf32>, %b16eWv: tensor<960x160x1x1xf32>, %b16egv: tensor<960xf32>, %b16ebtv: tensor<960xf32>, %b16dWv: tensor<960x1x3x3xf32>, %b16dgv: tensor<960xf32>, %b16dbtv: tensor<960xf32>, %b16pWv: tensor<160x960x1x1xf32>, %b16pgv: tensor<160xf32>, %b16pbtv: tensor<160xf32>, %b17eWv: tensor<960x160x1x1xf32>, %b17egv: tensor<960xf32>, %b17ebtv: tensor<960xf32>, %b17dWv: tensor<960x1x3x3xf32>, %b17dgv: tensor<960xf32>, %b17dbtv: tensor<960xf32>, %b17pWv: tensor<320x960x1x1xf32>, %b17pgv: tensor<320xf32>, %b17pbtv: tensor<320xf32>, %hWv: tensor<1280x320x1x1xf32>, %hgv: tensor<1280xf32>, %hbtv: tensor<1280xf32>, %Wdv: tensor<1280x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<32xf32>, %stnvari: tensor<32xf32>, %b1dnmui: tensor<32xf32>, %b1dnvari: tensor<32xf32>, %b1pnmui: tensor<16xf32>, %b1pnvari: tensor<16xf32>, %b2enmui: tensor<96xf32>, %b2envari: tensor<96xf32>, %b2dnmui: tensor<96xf32>, %b2dnvari: tensor<96xf32>, %b2pnmui: tensor<24xf32>, %b2pnvari: tensor<24xf32>, %b3enmui: tensor<144xf32>, %b3envari: tensor<144xf32>, %b3dnmui: tensor<144xf32>, %b3dnvari: tensor<144xf32>, %b3pnmui: tensor<24xf32>, %b3pnvari: tensor<24xf32>, %b4enmui: tensor<144xf32>, %b4envari: tensor<144xf32>, %b4dnmui: tensor<144xf32>, %b4dnvari: tensor<144xf32>, %b4pnmui: tensor<32xf32>, %b4pnvari: tensor<32xf32>, %b5enmui: tensor<192xf32>, %b5envari: tensor<192xf32>, %b5dnmui: tensor<192xf32>, %b5dnvari: tensor<192xf32>, %b5pnmui: tensor<32xf32>, %b5pnvari: tensor<32xf32>, %b6enmui: tensor<192xf32>, %b6envari: tensor<192xf32>, %b6dnmui: tensor<192xf32>, %b6dnvari: tensor<192xf32>, %b6pnmui: tensor<32xf32>, %b6pnvari: tensor<32xf32>, %b7enmui: tensor<192xf32>, %b7envari: tensor<192xf32>, %b7dnmui: tensor<192xf32>, %b7dnvari: tensor<192xf32>, %b7pnmui: tensor<64xf32>, %b7pnvari: tensor<64xf32>, %b8enmui: tensor<384xf32>, %b8envari: tensor<384xf32>, %b8dnmui: tensor<384xf32>, %b8dnvari: tensor<384xf32>, %b8pnmui: tensor<64xf32>, %b8pnvari: tensor<64xf32>, %b9enmui: tensor<384xf32>, %b9envari: tensor<384xf32>, %b9dnmui: tensor<384xf32>, %b9dnvari: tensor<384xf32>, %b9pnmui: tensor<64xf32>, %b9pnvari: tensor<64xf32>, %b10enmui: tensor<384xf32>, %b10envari: tensor<384xf32>, %b10dnmui: tensor<384xf32>, %b10dnvari: tensor<384xf32>, %b10pnmui: tensor<64xf32>, %b10pnvari: tensor<64xf32>, %b11enmui: tensor<384xf32>, %b11envari: tensor<384xf32>, %b11dnmui: tensor<384xf32>, %b11dnvari: tensor<384xf32>, %b11pnmui: tensor<96xf32>, %b11pnvari: tensor<96xf32>, %b12enmui: tensor<576xf32>, %b12envari: tensor<576xf32>, %b12dnmui: tensor<576xf32>, %b12dnvari: tensor<576xf32>, %b12pnmui: tensor<96xf32>, %b12pnvari: tensor<96xf32>, %b13enmui: tensor<576xf32>, %b13envari: tensor<576xf32>, %b13dnmui: tensor<576xf32>, %b13dnvari: tensor<576xf32>, %b13pnmui: tensor<96xf32>, %b13pnvari: tensor<96xf32>, %b14enmui: tensor<576xf32>, %b14envari: tensor<576xf32>, %b14dnmui: tensor<576xf32>, %b14dnvari: tensor<576xf32>, %b14pnmui: tensor<160xf32>, %b14pnvari: tensor<160xf32>, %b15enmui: tensor<960xf32>, %b15envari: tensor<960xf32>, %b15dnmui: tensor<960xf32>, %b15dnvari: tensor<960xf32>, %b15pnmui: tensor<160xf32>, %b15pnvari: tensor<160xf32>, %b16enmui: tensor<960xf32>, %b16envari: tensor<960xf32>, %b16dnmui: tensor<960xf32>, %b16dnvari: tensor<960xf32>, %b16pnmui: tensor<160xf32>, %b16pnvari: tensor<160xf32>, %b17enmui: tensor<960xf32>, %b17envari: tensor<960xf32>, %b17dnmui: tensor<960xf32>, %b17dnvari: tensor<960xf32>, %b17pnmui: tensor<320xf32>, %b17pnvari: tensor<320xf32>, %hnmui: tensor<1280xf32>, %hnvari: tensor<1280xf32>, %onehot: tensor<32x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280xf32>, tensor<1280xf32>) {
    // ── OPTIMIZER: RMSProp + momentum, TENSORFLOW flavour (the MobileNetV2 reference's
    //    own: jax/MainMobilenetV2Imagenet.lean). Per parameter, in this order:
    //      g  <- g + wd*θ        COUPLED L2, BEFORE the accumulator  (momVNextF)
    //      s' <- ρ*s + (1-ρ)*g²                                      (adamVNextF at ρ)
    //      b' <- μ*b + g/sqrt(s' + ε)   ⚠ ε INSIDE the sqrt          (rmsBufNextF)
    //      θ' <- θ - lr*b'                                           (sgdParamF)
    //    Packed [θ|m|v] is reused with m = momentum buffer, v = mean-square, so the
    //    interface is byte-identical to the AdamW render's apart from the entry name.
    //    %bc1/%bc2 are Adam bias corrections: unused here, passed through unchanged.
    //    ⚠ The mean-square must be INITIALISED TO 1.0, not 0 — part of the recipe, not
    //    an implementation detail, since this optimizer is not bias-corrected.
    // ── MobileNetV2 batch-BN AdamW train step: every line is pretty(verified AST node) ──
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb16 = stablehlo.constant dense<0.0> : tensor<16xf32>
    %zb24 = stablehlo.constant dense<0.0> : tensor<24xf32>
    %zb32 = stablehlo.constant dense<0.0> : tensor<32xf32>
    %zb64 = stablehlo.constant dense<0.0> : tensor<64xf32>
    %zb96 = stablehlo.constant dense<0.0> : tensor<96xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb144 = stablehlo.constant dense<0.0> : tensor<144xf32>
    %zb160 = stablehlo.constant dense<0.0> : tensor<160xf32>
    %zb192 = stablehlo.constant dense<0.0> : tensor<192xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb320 = stablehlo.constant dense<0.0> : tensor<320xf32>
    %zb384 = stablehlo.constant dense<0.0> : tensor<384xf32>
    %zb576 = stablehlo.constant dense<0.0> : tensor<576xf32>
    %zb960 = stablehlo.constant dense<0.0> : tensor<960xf32>
    %zb1280 = stablehlo.constant dense<0.0> : tensor<1280xf32>
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<32x3x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x32x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<32x32x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<32x32x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<32x32x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<32x32x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<32x32x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<32x32x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<32x32x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<32x32x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<32x32x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v25 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v26 = stablehlo.constant dense<6.0> : tensor<32x401408xf32>
    %v27 = stablehlo.maximum %v24, %v25 : tensor<32x401408xf32>
    %v28 = stablehlo.minimum %v27, %v26 : tensor<32x401408xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v30 = stablehlo.convolution(%v29, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v31 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<32x32x112x112xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v35 = stablehlo.constant dense<0.0> : tensor<f32>
    %v36 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v37 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v38 = stablehlo.reduce(%v34 init: %v35) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v39 = stablehlo.broadcast_in_dim %v38, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v40 = stablehlo.divide %v39, %v36 : tensor<32x32x112x112xf32>
    %v41 = stablehlo.subtract %v34, %v40 : tensor<32x32x112x112xf32>
    %v42 = stablehlo.multiply %v41, %v41 : tensor<32x32x112x112xf32>
    %v43 = stablehlo.reduce(%v42 init: %v35) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v44 = stablehlo.broadcast_in_dim %v43, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v45 = stablehlo.divide %v44, %v36 : tensor<32x32x112x112xf32>
    %v46 = stablehlo.add %v45, %v37 : tensor<32x32x112x112xf32>
    %v47 = stablehlo.rsqrt %v46 : tensor<32x32x112x112xf32>
    %v48 = stablehlo.multiply %v41, %v47 : tensor<32x32x112x112xf32>
    %v49 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v50 = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v51 = stablehlo.multiply %v48, %v49 : tensor<32x32x112x112xf32>
    %v52 = stablehlo.add %v51, %v50 : tensor<32x32x112x112xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v54 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v55 = stablehlo.constant dense<6.0> : tensor<32x401408xf32>
    %v56 = stablehlo.maximum %v53, %v54 : tensor<32x401408xf32>
    %v57 = stablehlo.minimum %v56, %v55 : tensor<32x401408xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v59 = stablehlo.convolution(%v58, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v60 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<32x16x112x112xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v64 = stablehlo.constant dense<0.0> : tensor<f32>
    %v65 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v66 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v67 = stablehlo.reduce(%v63 init: %v64) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v69 = stablehlo.divide %v68, %v65 : tensor<32x16x112x112xf32>
    %v70 = stablehlo.subtract %v63, %v69 : tensor<32x16x112x112xf32>
    %v71 = stablehlo.multiply %v70, %v70 : tensor<32x16x112x112xf32>
    %v72 = stablehlo.reduce(%v71 init: %v64) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v73 = stablehlo.broadcast_in_dim %v72, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v74 = stablehlo.divide %v73, %v65 : tensor<32x16x112x112xf32>
    %v75 = stablehlo.add %v74, %v66 : tensor<32x16x112x112xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<32x16x112x112xf32>
    %v77 = stablehlo.multiply %v70, %v76 : tensor<32x16x112x112xf32>
    %v78 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v79 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v80 = stablehlo.multiply %v77, %v78 : tensor<32x16x112x112xf32>
    %v81 = stablehlo.add %v80, %v79 : tensor<32x16x112x112xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v84 = stablehlo.convolution(%v83, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v85 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v86 = stablehlo.add %v84, %v85 : tensor<32x96x112x112xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v89 = stablehlo.constant dense<0.0> : tensor<f32>
    %v90 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v91 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v92 = stablehlo.reduce(%v88 init: %v89) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v93 = stablehlo.broadcast_in_dim %v92, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v94 = stablehlo.divide %v93, %v90 : tensor<32x96x112x112xf32>
    %v95 = stablehlo.subtract %v88, %v94 : tensor<32x96x112x112xf32>
    %v96 = stablehlo.multiply %v95, %v95 : tensor<32x96x112x112xf32>
    %v97 = stablehlo.reduce(%v96 init: %v89) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v98 = stablehlo.broadcast_in_dim %v97, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v99 = stablehlo.divide %v98, %v90 : tensor<32x96x112x112xf32>
    %v100 = stablehlo.add %v99, %v91 : tensor<32x96x112x112xf32>
    %v101 = stablehlo.rsqrt %v100 : tensor<32x96x112x112xf32>
    %v102 = stablehlo.multiply %v95, %v101 : tensor<32x96x112x112xf32>
    %v103 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v104 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v105 = stablehlo.multiply %v102, %v103 : tensor<32x96x112x112xf32>
    %v106 = stablehlo.add %v105, %v104 : tensor<32x96x112x112xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v108 = stablehlo.constant dense<0.0> : tensor<32x1204224xf32>
    %v109 = stablehlo.constant dense<6.0> : tensor<32x1204224xf32>
    %v110 = stablehlo.maximum %v107, %v108 : tensor<32x1204224xf32>
    %v111 = stablehlo.minimum %v110, %v109 : tensor<32x1204224xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v113 = stablehlo.convolution(%v112, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v114 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v115 = stablehlo.add %v113, %v114 : tensor<32x96x56x56xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v119 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v120 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v121 = stablehlo.reduce(%v117 init: %v118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v122 = stablehlo.broadcast_in_dim %v121, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v123 = stablehlo.divide %v122, %v119 : tensor<32x96x56x56xf32>
    %v124 = stablehlo.subtract %v117, %v123 : tensor<32x96x56x56xf32>
    %v125 = stablehlo.multiply %v124, %v124 : tensor<32x96x56x56xf32>
    %v126 = stablehlo.reduce(%v125 init: %v118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v127 = stablehlo.broadcast_in_dim %v126, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v128 = stablehlo.divide %v127, %v119 : tensor<32x96x56x56xf32>
    %v129 = stablehlo.add %v128, %v120 : tensor<32x96x56x56xf32>
    %v130 = stablehlo.rsqrt %v129 : tensor<32x96x56x56xf32>
    %v131 = stablehlo.multiply %v124, %v130 : tensor<32x96x56x56xf32>
    %v132 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v133 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v134 = stablehlo.multiply %v131, %v132 : tensor<32x96x56x56xf32>
    %v135 = stablehlo.add %v134, %v133 : tensor<32x96x56x56xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v137 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v138 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v139 = stablehlo.maximum %v136, %v137 : tensor<32x301056xf32>
    %v140 = stablehlo.minimum %v139, %v138 : tensor<32x301056xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v142 = stablehlo.convolution(%v141, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v143 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v144 = stablehlo.add %v142, %v143 : tensor<32x24x56x56xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v148 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v149 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v150 = stablehlo.reduce(%v146 init: %v147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v151 = stablehlo.broadcast_in_dim %v150, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v152 = stablehlo.divide %v151, %v148 : tensor<32x24x56x56xf32>
    %v153 = stablehlo.subtract %v146, %v152 : tensor<32x24x56x56xf32>
    %v154 = stablehlo.multiply %v153, %v153 : tensor<32x24x56x56xf32>
    %v155 = stablehlo.reduce(%v154 init: %v147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v156 = stablehlo.broadcast_in_dim %v155, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v157 = stablehlo.divide %v156, %v148 : tensor<32x24x56x56xf32>
    %v158 = stablehlo.add %v157, %v149 : tensor<32x24x56x56xf32>
    %v159 = stablehlo.rsqrt %v158 : tensor<32x24x56x56xf32>
    %v160 = stablehlo.multiply %v153, %v159 : tensor<32x24x56x56xf32>
    %v161 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v162 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v163 = stablehlo.multiply %v160, %v161 : tensor<32x24x56x56xf32>
    %v164 = stablehlo.add %v163, %v162 : tensor<32x24x56x56xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v167 = stablehlo.convolution(%v166, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v168 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v169 = stablehlo.add %v167, %v168 : tensor<32x144x56x56xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v173 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v174 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v175 = stablehlo.reduce(%v171 init: %v172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v176 = stablehlo.broadcast_in_dim %v175, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v177 = stablehlo.divide %v176, %v173 : tensor<32x144x56x56xf32>
    %v178 = stablehlo.subtract %v171, %v177 : tensor<32x144x56x56xf32>
    %v179 = stablehlo.multiply %v178, %v178 : tensor<32x144x56x56xf32>
    %v180 = stablehlo.reduce(%v179 init: %v172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v181 = stablehlo.broadcast_in_dim %v180, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v182 = stablehlo.divide %v181, %v173 : tensor<32x144x56x56xf32>
    %v183 = stablehlo.add %v182, %v174 : tensor<32x144x56x56xf32>
    %v184 = stablehlo.rsqrt %v183 : tensor<32x144x56x56xf32>
    %v185 = stablehlo.multiply %v178, %v184 : tensor<32x144x56x56xf32>
    %v186 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v187 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v188 = stablehlo.multiply %v185, %v186 : tensor<32x144x56x56xf32>
    %v189 = stablehlo.add %v188, %v187 : tensor<32x144x56x56xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v191 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v192 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v193 = stablehlo.maximum %v190, %v191 : tensor<32x451584xf32>
    %v194 = stablehlo.minimum %v193, %v192 : tensor<32x451584xf32>
    %v195 = stablehlo.reshape %v194 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v196 = stablehlo.convolution(%v195, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v197 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v198 = stablehlo.add %v196, %v197 : tensor<32x144x56x56xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v201 = stablehlo.constant dense<0.0> : tensor<f32>
    %v202 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v203 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v204 = stablehlo.reduce(%v200 init: %v201) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v205 = stablehlo.broadcast_in_dim %v204, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v206 = stablehlo.divide %v205, %v202 : tensor<32x144x56x56xf32>
    %v207 = stablehlo.subtract %v200, %v206 : tensor<32x144x56x56xf32>
    %v208 = stablehlo.multiply %v207, %v207 : tensor<32x144x56x56xf32>
    %v209 = stablehlo.reduce(%v208 init: %v201) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v210 = stablehlo.broadcast_in_dim %v209, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v211 = stablehlo.divide %v210, %v202 : tensor<32x144x56x56xf32>
    %v212 = stablehlo.add %v211, %v203 : tensor<32x144x56x56xf32>
    %v213 = stablehlo.rsqrt %v212 : tensor<32x144x56x56xf32>
    %v214 = stablehlo.multiply %v207, %v213 : tensor<32x144x56x56xf32>
    %v215 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v216 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v217 = stablehlo.multiply %v214, %v215 : tensor<32x144x56x56xf32>
    %v218 = stablehlo.add %v217, %v216 : tensor<32x144x56x56xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v220 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v221 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v222 = stablehlo.maximum %v219, %v220 : tensor<32x451584xf32>
    %v223 = stablehlo.minimum %v222, %v221 : tensor<32x451584xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v225 = stablehlo.convolution(%v224, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v226 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v227 = stablehlo.add %v225, %v226 : tensor<32x24x56x56xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v231 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v232 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v233 = stablehlo.reduce(%v229 init: %v230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v234 = stablehlo.broadcast_in_dim %v233, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v235 = stablehlo.divide %v234, %v231 : tensor<32x24x56x56xf32>
    %v236 = stablehlo.subtract %v229, %v235 : tensor<32x24x56x56xf32>
    %v237 = stablehlo.multiply %v236, %v236 : tensor<32x24x56x56xf32>
    %v238 = stablehlo.reduce(%v237 init: %v230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v239 = stablehlo.broadcast_in_dim %v238, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v240 = stablehlo.divide %v239, %v231 : tensor<32x24x56x56xf32>
    %v241 = stablehlo.add %v240, %v232 : tensor<32x24x56x56xf32>
    %v242 = stablehlo.rsqrt %v241 : tensor<32x24x56x56xf32>
    %v243 = stablehlo.multiply %v236, %v242 : tensor<32x24x56x56xf32>
    %v244 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v245 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v246 = stablehlo.multiply %v243, %v244 : tensor<32x24x56x56xf32>
    %v247 = stablehlo.add %v246, %v245 : tensor<32x24x56x56xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v249 = stablehlo.add %v248, %v165 : tensor<32x75264xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v251 = stablehlo.convolution(%v250, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v252 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v253 = stablehlo.add %v251, %v252 : tensor<32x144x56x56xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v257 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v258 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v259 = stablehlo.reduce(%v255 init: %v256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v260 = stablehlo.broadcast_in_dim %v259, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v261 = stablehlo.divide %v260, %v257 : tensor<32x144x56x56xf32>
    %v262 = stablehlo.subtract %v255, %v261 : tensor<32x144x56x56xf32>
    %v263 = stablehlo.multiply %v262, %v262 : tensor<32x144x56x56xf32>
    %v264 = stablehlo.reduce(%v263 init: %v256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v265 = stablehlo.broadcast_in_dim %v264, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v266 = stablehlo.divide %v265, %v257 : tensor<32x144x56x56xf32>
    %v267 = stablehlo.add %v266, %v258 : tensor<32x144x56x56xf32>
    %v268 = stablehlo.rsqrt %v267 : tensor<32x144x56x56xf32>
    %v269 = stablehlo.multiply %v262, %v268 : tensor<32x144x56x56xf32>
    %v270 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v271 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v272 = stablehlo.multiply %v269, %v270 : tensor<32x144x56x56xf32>
    %v273 = stablehlo.add %v272, %v271 : tensor<32x144x56x56xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v275 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v276 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v277 = stablehlo.maximum %v274, %v275 : tensor<32x451584xf32>
    %v278 = stablehlo.minimum %v277, %v276 : tensor<32x451584xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v280 = stablehlo.convolution(%v279, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x28x28xf32>
    %v281 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v282 = stablehlo.add %v280, %v281 : tensor<32x144x28x28xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v286 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v287 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v288 = stablehlo.reduce(%v284 init: %v285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v289 = stablehlo.broadcast_in_dim %v288, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v290 = stablehlo.divide %v289, %v286 : tensor<32x144x28x28xf32>
    %v291 = stablehlo.subtract %v284, %v290 : tensor<32x144x28x28xf32>
    %v292 = stablehlo.multiply %v291, %v291 : tensor<32x144x28x28xf32>
    %v293 = stablehlo.reduce(%v292 init: %v285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v294 = stablehlo.broadcast_in_dim %v293, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v295 = stablehlo.divide %v294, %v286 : tensor<32x144x28x28xf32>
    %v296 = stablehlo.add %v295, %v287 : tensor<32x144x28x28xf32>
    %v297 = stablehlo.rsqrt %v296 : tensor<32x144x28x28xf32>
    %v298 = stablehlo.multiply %v291, %v297 : tensor<32x144x28x28xf32>
    %v299 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v300 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v301 = stablehlo.multiply %v298, %v299 : tensor<32x144x28x28xf32>
    %v302 = stablehlo.add %v301, %v300 : tensor<32x144x28x28xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v304 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v305 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v306 = stablehlo.maximum %v303, %v304 : tensor<32x112896xf32>
    %v307 = stablehlo.minimum %v306, %v305 : tensor<32x112896xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v309 = stablehlo.convolution(%v308, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<32x144x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v310 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v311 = stablehlo.add %v309, %v310 : tensor<32x32x28x28xf32>
    %v312 = stablehlo.reshape %v311 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v315 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v316 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v317 = stablehlo.reduce(%v313 init: %v314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v318 = stablehlo.broadcast_in_dim %v317, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v319 = stablehlo.divide %v318, %v315 : tensor<32x32x28x28xf32>
    %v320 = stablehlo.subtract %v313, %v319 : tensor<32x32x28x28xf32>
    %v321 = stablehlo.multiply %v320, %v320 : tensor<32x32x28x28xf32>
    %v322 = stablehlo.reduce(%v321 init: %v314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v323 = stablehlo.broadcast_in_dim %v322, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v324 = stablehlo.divide %v323, %v315 : tensor<32x32x28x28xf32>
    %v325 = stablehlo.add %v324, %v316 : tensor<32x32x28x28xf32>
    %v326 = stablehlo.rsqrt %v325 : tensor<32x32x28x28xf32>
    %v327 = stablehlo.multiply %v320, %v326 : tensor<32x32x28x28xf32>
    %v328 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v329 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v330 = stablehlo.multiply %v327, %v328 : tensor<32x32x28x28xf32>
    %v331 = stablehlo.add %v330, %v329 : tensor<32x32x28x28xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v334 = stablehlo.convolution(%v333, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v335 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v336 = stablehlo.add %v334, %v335 : tensor<32x192x28x28xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v340 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v341 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v342 = stablehlo.reduce(%v338 init: %v339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v343 = stablehlo.broadcast_in_dim %v342, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v344 = stablehlo.divide %v343, %v340 : tensor<32x192x28x28xf32>
    %v345 = stablehlo.subtract %v338, %v344 : tensor<32x192x28x28xf32>
    %v346 = stablehlo.multiply %v345, %v345 : tensor<32x192x28x28xf32>
    %v347 = stablehlo.reduce(%v346 init: %v339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v348 = stablehlo.broadcast_in_dim %v347, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v349 = stablehlo.divide %v348, %v340 : tensor<32x192x28x28xf32>
    %v350 = stablehlo.add %v349, %v341 : tensor<32x192x28x28xf32>
    %v351 = stablehlo.rsqrt %v350 : tensor<32x192x28x28xf32>
    %v352 = stablehlo.multiply %v345, %v351 : tensor<32x192x28x28xf32>
    %v353 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v354 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v355 = stablehlo.multiply %v352, %v353 : tensor<32x192x28x28xf32>
    %v356 = stablehlo.add %v355, %v354 : tensor<32x192x28x28xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v358 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v359 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v360 = stablehlo.maximum %v357, %v358 : tensor<32x150528xf32>
    %v361 = stablehlo.minimum %v360, %v359 : tensor<32x150528xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v363 = stablehlo.convolution(%v362, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v364 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v365 = stablehlo.add %v363, %v364 : tensor<32x192x28x28xf32>
    %v366 = stablehlo.reshape %v365 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v369 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v370 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v371 = stablehlo.reduce(%v367 init: %v368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v372 = stablehlo.broadcast_in_dim %v371, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v373 = stablehlo.divide %v372, %v369 : tensor<32x192x28x28xf32>
    %v374 = stablehlo.subtract %v367, %v373 : tensor<32x192x28x28xf32>
    %v375 = stablehlo.multiply %v374, %v374 : tensor<32x192x28x28xf32>
    %v376 = stablehlo.reduce(%v375 init: %v368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v377 = stablehlo.broadcast_in_dim %v376, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v378 = stablehlo.divide %v377, %v369 : tensor<32x192x28x28xf32>
    %v379 = stablehlo.add %v378, %v370 : tensor<32x192x28x28xf32>
    %v380 = stablehlo.rsqrt %v379 : tensor<32x192x28x28xf32>
    %v381 = stablehlo.multiply %v374, %v380 : tensor<32x192x28x28xf32>
    %v382 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v383 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v384 = stablehlo.multiply %v381, %v382 : tensor<32x192x28x28xf32>
    %v385 = stablehlo.add %v384, %v383 : tensor<32x192x28x28xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v387 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v388 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v389 = stablehlo.maximum %v386, %v387 : tensor<32x150528xf32>
    %v390 = stablehlo.minimum %v389, %v388 : tensor<32x150528xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v392 = stablehlo.convolution(%v391, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v393 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v394 = stablehlo.add %v392, %v393 : tensor<32x32x28x28xf32>
    %v395 = stablehlo.reshape %v394 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v397 = stablehlo.constant dense<0.0> : tensor<f32>
    %v398 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v399 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v400 = stablehlo.reduce(%v396 init: %v397) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v401 = stablehlo.broadcast_in_dim %v400, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v402 = stablehlo.divide %v401, %v398 : tensor<32x32x28x28xf32>
    %v403 = stablehlo.subtract %v396, %v402 : tensor<32x32x28x28xf32>
    %v404 = stablehlo.multiply %v403, %v403 : tensor<32x32x28x28xf32>
    %v405 = stablehlo.reduce(%v404 init: %v397) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v406 = stablehlo.broadcast_in_dim %v405, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v407 = stablehlo.divide %v406, %v398 : tensor<32x32x28x28xf32>
    %v408 = stablehlo.add %v407, %v399 : tensor<32x32x28x28xf32>
    %v409 = stablehlo.rsqrt %v408 : tensor<32x32x28x28xf32>
    %v410 = stablehlo.multiply %v403, %v409 : tensor<32x32x28x28xf32>
    %v411 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v412 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v413 = stablehlo.multiply %v410, %v411 : tensor<32x32x28x28xf32>
    %v414 = stablehlo.add %v413, %v412 : tensor<32x32x28x28xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v416 = stablehlo.add %v415, %v332 : tensor<32x25088xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v418 = stablehlo.convolution(%v417, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v419 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v420 = stablehlo.add %v418, %v419 : tensor<32x192x28x28xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v424 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v425 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v426 = stablehlo.reduce(%v422 init: %v423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v427 = stablehlo.broadcast_in_dim %v426, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v428 = stablehlo.divide %v427, %v424 : tensor<32x192x28x28xf32>
    %v429 = stablehlo.subtract %v422, %v428 : tensor<32x192x28x28xf32>
    %v430 = stablehlo.multiply %v429, %v429 : tensor<32x192x28x28xf32>
    %v431 = stablehlo.reduce(%v430 init: %v423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v432 = stablehlo.broadcast_in_dim %v431, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v433 = stablehlo.divide %v432, %v424 : tensor<32x192x28x28xf32>
    %v434 = stablehlo.add %v433, %v425 : tensor<32x192x28x28xf32>
    %v435 = stablehlo.rsqrt %v434 : tensor<32x192x28x28xf32>
    %v436 = stablehlo.multiply %v429, %v435 : tensor<32x192x28x28xf32>
    %v437 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v438 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v439 = stablehlo.multiply %v436, %v437 : tensor<32x192x28x28xf32>
    %v440 = stablehlo.add %v439, %v438 : tensor<32x192x28x28xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v442 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v443 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v444 = stablehlo.maximum %v441, %v442 : tensor<32x150528xf32>
    %v445 = stablehlo.minimum %v444, %v443 : tensor<32x150528xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v447 = stablehlo.convolution(%v446, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v448 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v449 = stablehlo.add %v447, %v448 : tensor<32x192x28x28xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v453 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v454 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v455 = stablehlo.reduce(%v451 init: %v452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v456 = stablehlo.broadcast_in_dim %v455, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v457 = stablehlo.divide %v456, %v453 : tensor<32x192x28x28xf32>
    %v458 = stablehlo.subtract %v451, %v457 : tensor<32x192x28x28xf32>
    %v459 = stablehlo.multiply %v458, %v458 : tensor<32x192x28x28xf32>
    %v460 = stablehlo.reduce(%v459 init: %v452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v461 = stablehlo.broadcast_in_dim %v460, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v462 = stablehlo.divide %v461, %v453 : tensor<32x192x28x28xf32>
    %v463 = stablehlo.add %v462, %v454 : tensor<32x192x28x28xf32>
    %v464 = stablehlo.rsqrt %v463 : tensor<32x192x28x28xf32>
    %v465 = stablehlo.multiply %v458, %v464 : tensor<32x192x28x28xf32>
    %v466 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v467 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v468 = stablehlo.multiply %v465, %v466 : tensor<32x192x28x28xf32>
    %v469 = stablehlo.add %v468, %v467 : tensor<32x192x28x28xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v471 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v472 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v473 = stablehlo.maximum %v470, %v471 : tensor<32x150528xf32>
    %v474 = stablehlo.minimum %v473, %v472 : tensor<32x150528xf32>
    %v475 = stablehlo.reshape %v474 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v476 = stablehlo.convolution(%v475, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v477 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v478 = stablehlo.add %v476, %v477 : tensor<32x32x28x28xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v481 = stablehlo.constant dense<0.0> : tensor<f32>
    %v482 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v483 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v484 = stablehlo.reduce(%v480 init: %v481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v485 = stablehlo.broadcast_in_dim %v484, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v486 = stablehlo.divide %v485, %v482 : tensor<32x32x28x28xf32>
    %v487 = stablehlo.subtract %v480, %v486 : tensor<32x32x28x28xf32>
    %v488 = stablehlo.multiply %v487, %v487 : tensor<32x32x28x28xf32>
    %v489 = stablehlo.reduce(%v488 init: %v481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v490 = stablehlo.broadcast_in_dim %v489, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v491 = stablehlo.divide %v490, %v482 : tensor<32x32x28x28xf32>
    %v492 = stablehlo.add %v491, %v483 : tensor<32x32x28x28xf32>
    %v493 = stablehlo.rsqrt %v492 : tensor<32x32x28x28xf32>
    %v494 = stablehlo.multiply %v487, %v493 : tensor<32x32x28x28xf32>
    %v495 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v496 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v497 = stablehlo.multiply %v494, %v495 : tensor<32x32x28x28xf32>
    %v498 = stablehlo.add %v497, %v496 : tensor<32x32x28x28xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v500 = stablehlo.add %v499, %v416 : tensor<32x25088xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v502 = stablehlo.convolution(%v501, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v503 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v504 = stablehlo.add %v502, %v503 : tensor<32x192x28x28xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v506 = stablehlo.reshape %v505 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v507 = stablehlo.constant dense<0.0> : tensor<f32>
    %v508 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v509 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v510 = stablehlo.reduce(%v506 init: %v507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v511 = stablehlo.broadcast_in_dim %v510, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v512 = stablehlo.divide %v511, %v508 : tensor<32x192x28x28xf32>
    %v513 = stablehlo.subtract %v506, %v512 : tensor<32x192x28x28xf32>
    %v514 = stablehlo.multiply %v513, %v513 : tensor<32x192x28x28xf32>
    %v515 = stablehlo.reduce(%v514 init: %v507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v516 = stablehlo.broadcast_in_dim %v515, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v517 = stablehlo.divide %v516, %v508 : tensor<32x192x28x28xf32>
    %v518 = stablehlo.add %v517, %v509 : tensor<32x192x28x28xf32>
    %v519 = stablehlo.rsqrt %v518 : tensor<32x192x28x28xf32>
    %v520 = stablehlo.multiply %v513, %v519 : tensor<32x192x28x28xf32>
    %v521 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v522 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v523 = stablehlo.multiply %v520, %v521 : tensor<32x192x28x28xf32>
    %v524 = stablehlo.add %v523, %v522 : tensor<32x192x28x28xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v526 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v527 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v528 = stablehlo.maximum %v525, %v526 : tensor<32x150528xf32>
    %v529 = stablehlo.minimum %v528, %v527 : tensor<32x150528xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v531 = stablehlo.convolution(%v530, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x14x14xf32>
    %v532 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v533 = stablehlo.add %v531, %v532 : tensor<32x192x14x14xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v537 = stablehlo.constant dense<6272.0> : tensor<32x192x14x14xf32>
    %v538 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v539 = stablehlo.reduce(%v535 init: %v536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v540 = stablehlo.broadcast_in_dim %v539, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v541 = stablehlo.divide %v540, %v537 : tensor<32x192x14x14xf32>
    %v542 = stablehlo.subtract %v535, %v541 : tensor<32x192x14x14xf32>
    %v543 = stablehlo.multiply %v542, %v542 : tensor<32x192x14x14xf32>
    %v544 = stablehlo.reduce(%v543 init: %v536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v545 = stablehlo.broadcast_in_dim %v544, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v546 = stablehlo.divide %v545, %v537 : tensor<32x192x14x14xf32>
    %v547 = stablehlo.add %v546, %v538 : tensor<32x192x14x14xf32>
    %v548 = stablehlo.rsqrt %v547 : tensor<32x192x14x14xf32>
    %v549 = stablehlo.multiply %v542, %v548 : tensor<32x192x14x14xf32>
    %v550 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v551 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v552 = stablehlo.multiply %v549, %v550 : tensor<32x192x14x14xf32>
    %v553 = stablehlo.add %v552, %v551 : tensor<32x192x14x14xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v555 = stablehlo.constant dense<0.0> : tensor<32x37632xf32>
    %v556 = stablehlo.constant dense<6.0> : tensor<32x37632xf32>
    %v557 = stablehlo.maximum %v554, %v555 : tensor<32x37632xf32>
    %v558 = stablehlo.minimum %v557, %v556 : tensor<32x37632xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v560 = stablehlo.convolution(%v559, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x14x14xf32>, tensor<64x192x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v561 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v562 = stablehlo.add %v560, %v561 : tensor<32x64x14x14xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v566 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v567 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v568 = stablehlo.reduce(%v564 init: %v565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v569 = stablehlo.broadcast_in_dim %v568, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v570 = stablehlo.divide %v569, %v566 : tensor<32x64x14x14xf32>
    %v571 = stablehlo.subtract %v564, %v570 : tensor<32x64x14x14xf32>
    %v572 = stablehlo.multiply %v571, %v571 : tensor<32x64x14x14xf32>
    %v573 = stablehlo.reduce(%v572 init: %v565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v574 = stablehlo.broadcast_in_dim %v573, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v575 = stablehlo.divide %v574, %v566 : tensor<32x64x14x14xf32>
    %v576 = stablehlo.add %v575, %v567 : tensor<32x64x14x14xf32>
    %v577 = stablehlo.rsqrt %v576 : tensor<32x64x14x14xf32>
    %v578 = stablehlo.multiply %v571, %v577 : tensor<32x64x14x14xf32>
    %v579 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v580 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v581 = stablehlo.multiply %v578, %v579 : tensor<32x64x14x14xf32>
    %v582 = stablehlo.add %v581, %v580 : tensor<32x64x14x14xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v585 = stablehlo.convolution(%v584, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v586 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v587 = stablehlo.add %v585, %v586 : tensor<32x384x14x14xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v591 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v592 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v593 = stablehlo.reduce(%v589 init: %v590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v594 = stablehlo.broadcast_in_dim %v593, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v595 = stablehlo.divide %v594, %v591 : tensor<32x384x14x14xf32>
    %v596 = stablehlo.subtract %v589, %v595 : tensor<32x384x14x14xf32>
    %v597 = stablehlo.multiply %v596, %v596 : tensor<32x384x14x14xf32>
    %v598 = stablehlo.reduce(%v597 init: %v590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v599 = stablehlo.broadcast_in_dim %v598, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v600 = stablehlo.divide %v599, %v591 : tensor<32x384x14x14xf32>
    %v601 = stablehlo.add %v600, %v592 : tensor<32x384x14x14xf32>
    %v602 = stablehlo.rsqrt %v601 : tensor<32x384x14x14xf32>
    %v603 = stablehlo.multiply %v596, %v602 : tensor<32x384x14x14xf32>
    %v604 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v605 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v606 = stablehlo.multiply %v603, %v604 : tensor<32x384x14x14xf32>
    %v607 = stablehlo.add %v606, %v605 : tensor<32x384x14x14xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v609 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v610 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v611 = stablehlo.maximum %v608, %v609 : tensor<32x75264xf32>
    %v612 = stablehlo.minimum %v611, %v610 : tensor<32x75264xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v614 = stablehlo.convolution(%v613, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v615 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v616 = stablehlo.add %v614, %v615 : tensor<32x384x14x14xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v618 = stablehlo.reshape %v617 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v619 = stablehlo.constant dense<0.0> : tensor<f32>
    %v620 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v621 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v622 = stablehlo.reduce(%v618 init: %v619) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v623 = stablehlo.broadcast_in_dim %v622, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v624 = stablehlo.divide %v623, %v620 : tensor<32x384x14x14xf32>
    %v625 = stablehlo.subtract %v618, %v624 : tensor<32x384x14x14xf32>
    %v626 = stablehlo.multiply %v625, %v625 : tensor<32x384x14x14xf32>
    %v627 = stablehlo.reduce(%v626 init: %v619) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v628 = stablehlo.broadcast_in_dim %v627, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v629 = stablehlo.divide %v628, %v620 : tensor<32x384x14x14xf32>
    %v630 = stablehlo.add %v629, %v621 : tensor<32x384x14x14xf32>
    %v631 = stablehlo.rsqrt %v630 : tensor<32x384x14x14xf32>
    %v632 = stablehlo.multiply %v625, %v631 : tensor<32x384x14x14xf32>
    %v633 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v634 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v635 = stablehlo.multiply %v632, %v633 : tensor<32x384x14x14xf32>
    %v636 = stablehlo.add %v635, %v634 : tensor<32x384x14x14xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v638 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v639 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v640 = stablehlo.maximum %v637, %v638 : tensor<32x75264xf32>
    %v641 = stablehlo.minimum %v640, %v639 : tensor<32x75264xf32>
    %v642 = stablehlo.reshape %v641 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v643 = stablehlo.convolution(%v642, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v644 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v645 = stablehlo.add %v643, %v644 : tensor<32x64x14x14xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v648 = stablehlo.constant dense<0.0> : tensor<f32>
    %v649 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v650 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v651 = stablehlo.reduce(%v647 init: %v648) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v652 = stablehlo.broadcast_in_dim %v651, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v653 = stablehlo.divide %v652, %v649 : tensor<32x64x14x14xf32>
    %v654 = stablehlo.subtract %v647, %v653 : tensor<32x64x14x14xf32>
    %v655 = stablehlo.multiply %v654, %v654 : tensor<32x64x14x14xf32>
    %v656 = stablehlo.reduce(%v655 init: %v648) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v657 = stablehlo.broadcast_in_dim %v656, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v658 = stablehlo.divide %v657, %v649 : tensor<32x64x14x14xf32>
    %v659 = stablehlo.add %v658, %v650 : tensor<32x64x14x14xf32>
    %v660 = stablehlo.rsqrt %v659 : tensor<32x64x14x14xf32>
    %v661 = stablehlo.multiply %v654, %v660 : tensor<32x64x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v663 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v664 = stablehlo.multiply %v661, %v662 : tensor<32x64x14x14xf32>
    %v665 = stablehlo.add %v664, %v663 : tensor<32x64x14x14xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v667 = stablehlo.add %v666, %v583 : tensor<32x12544xf32>
    %v668 = stablehlo.reshape %v667 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v669 = stablehlo.convolution(%v668, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v670 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v671 = stablehlo.add %v669, %v670 : tensor<32x384x14x14xf32>
    %v672 = stablehlo.reshape %v671 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v674 = stablehlo.constant dense<0.0> : tensor<f32>
    %v675 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v676 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v677 = stablehlo.reduce(%v673 init: %v674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v678 = stablehlo.broadcast_in_dim %v677, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v679 = stablehlo.divide %v678, %v675 : tensor<32x384x14x14xf32>
    %v680 = stablehlo.subtract %v673, %v679 : tensor<32x384x14x14xf32>
    %v681 = stablehlo.multiply %v680, %v680 : tensor<32x384x14x14xf32>
    %v682 = stablehlo.reduce(%v681 init: %v674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v683 = stablehlo.broadcast_in_dim %v682, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v684 = stablehlo.divide %v683, %v675 : tensor<32x384x14x14xf32>
    %v685 = stablehlo.add %v684, %v676 : tensor<32x384x14x14xf32>
    %v686 = stablehlo.rsqrt %v685 : tensor<32x384x14x14xf32>
    %v687 = stablehlo.multiply %v680, %v686 : tensor<32x384x14x14xf32>
    %v688 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v689 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v690 = stablehlo.multiply %v687, %v688 : tensor<32x384x14x14xf32>
    %v691 = stablehlo.add %v690, %v689 : tensor<32x384x14x14xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v693 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v694 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v695 = stablehlo.maximum %v692, %v693 : tensor<32x75264xf32>
    %v696 = stablehlo.minimum %v695, %v694 : tensor<32x75264xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v698 = stablehlo.convolution(%v697, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v699 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v700 = stablehlo.add %v698, %v699 : tensor<32x384x14x14xf32>
    %v701 = stablehlo.reshape %v700 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v702 = stablehlo.reshape %v701 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v703 = stablehlo.constant dense<0.0> : tensor<f32>
    %v704 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v705 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v706 = stablehlo.reduce(%v702 init: %v703) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v707 = stablehlo.broadcast_in_dim %v706, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v708 = stablehlo.divide %v707, %v704 : tensor<32x384x14x14xf32>
    %v709 = stablehlo.subtract %v702, %v708 : tensor<32x384x14x14xf32>
    %v710 = stablehlo.multiply %v709, %v709 : tensor<32x384x14x14xf32>
    %v711 = stablehlo.reduce(%v710 init: %v703) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v712 = stablehlo.broadcast_in_dim %v711, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v713 = stablehlo.divide %v712, %v704 : tensor<32x384x14x14xf32>
    %v714 = stablehlo.add %v713, %v705 : tensor<32x384x14x14xf32>
    %v715 = stablehlo.rsqrt %v714 : tensor<32x384x14x14xf32>
    %v716 = stablehlo.multiply %v709, %v715 : tensor<32x384x14x14xf32>
    %v717 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v718 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v719 = stablehlo.multiply %v716, %v717 : tensor<32x384x14x14xf32>
    %v720 = stablehlo.add %v719, %v718 : tensor<32x384x14x14xf32>
    %v721 = stablehlo.reshape %v720 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v722 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v723 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v724 = stablehlo.maximum %v721, %v722 : tensor<32x75264xf32>
    %v725 = stablehlo.minimum %v724, %v723 : tensor<32x75264xf32>
    %v726 = stablehlo.reshape %v725 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v727 = stablehlo.convolution(%v726, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v728 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v729 = stablehlo.add %v727, %v728 : tensor<32x64x14x14xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v732 = stablehlo.constant dense<0.0> : tensor<f32>
    %v733 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v734 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v735 = stablehlo.reduce(%v731 init: %v732) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v736 = stablehlo.broadcast_in_dim %v735, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v737 = stablehlo.divide %v736, %v733 : tensor<32x64x14x14xf32>
    %v738 = stablehlo.subtract %v731, %v737 : tensor<32x64x14x14xf32>
    %v739 = stablehlo.multiply %v738, %v738 : tensor<32x64x14x14xf32>
    %v740 = stablehlo.reduce(%v739 init: %v732) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v741 = stablehlo.broadcast_in_dim %v740, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v742 = stablehlo.divide %v741, %v733 : tensor<32x64x14x14xf32>
    %v743 = stablehlo.add %v742, %v734 : tensor<32x64x14x14xf32>
    %v744 = stablehlo.rsqrt %v743 : tensor<32x64x14x14xf32>
    %v745 = stablehlo.multiply %v738, %v744 : tensor<32x64x14x14xf32>
    %v746 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v747 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v748 = stablehlo.multiply %v745, %v746 : tensor<32x64x14x14xf32>
    %v749 = stablehlo.add %v748, %v747 : tensor<32x64x14x14xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v751 = stablehlo.add %v750, %v667 : tensor<32x12544xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v753 = stablehlo.convolution(%v752, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v754 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v755 = stablehlo.add %v753, %v754 : tensor<32x384x14x14xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v758 = stablehlo.constant dense<0.0> : tensor<f32>
    %v759 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v760 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v761 = stablehlo.reduce(%v757 init: %v758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v762 = stablehlo.broadcast_in_dim %v761, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v763 = stablehlo.divide %v762, %v759 : tensor<32x384x14x14xf32>
    %v764 = stablehlo.subtract %v757, %v763 : tensor<32x384x14x14xf32>
    %v765 = stablehlo.multiply %v764, %v764 : tensor<32x384x14x14xf32>
    %v766 = stablehlo.reduce(%v765 init: %v758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v767 = stablehlo.broadcast_in_dim %v766, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v768 = stablehlo.divide %v767, %v759 : tensor<32x384x14x14xf32>
    %v769 = stablehlo.add %v768, %v760 : tensor<32x384x14x14xf32>
    %v770 = stablehlo.rsqrt %v769 : tensor<32x384x14x14xf32>
    %v771 = stablehlo.multiply %v764, %v770 : tensor<32x384x14x14xf32>
    %v772 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v773 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v774 = stablehlo.multiply %v771, %v772 : tensor<32x384x14x14xf32>
    %v775 = stablehlo.add %v774, %v773 : tensor<32x384x14x14xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v777 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v778 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v779 = stablehlo.maximum %v776, %v777 : tensor<32x75264xf32>
    %v780 = stablehlo.minimum %v779, %v778 : tensor<32x75264xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v782 = stablehlo.convolution(%v781, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v783 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v784 = stablehlo.add %v782, %v783 : tensor<32x384x14x14xf32>
    %v785 = stablehlo.reshape %v784 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v787 = stablehlo.constant dense<0.0> : tensor<f32>
    %v788 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v789 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v790 = stablehlo.reduce(%v786 init: %v787) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v791 = stablehlo.broadcast_in_dim %v790, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v792 = stablehlo.divide %v791, %v788 : tensor<32x384x14x14xf32>
    %v793 = stablehlo.subtract %v786, %v792 : tensor<32x384x14x14xf32>
    %v794 = stablehlo.multiply %v793, %v793 : tensor<32x384x14x14xf32>
    %v795 = stablehlo.reduce(%v794 init: %v787) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v796 = stablehlo.broadcast_in_dim %v795, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v797 = stablehlo.divide %v796, %v788 : tensor<32x384x14x14xf32>
    %v798 = stablehlo.add %v797, %v789 : tensor<32x384x14x14xf32>
    %v799 = stablehlo.rsqrt %v798 : tensor<32x384x14x14xf32>
    %v800 = stablehlo.multiply %v793, %v799 : tensor<32x384x14x14xf32>
    %v801 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v802 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v803 = stablehlo.multiply %v800, %v801 : tensor<32x384x14x14xf32>
    %v804 = stablehlo.add %v803, %v802 : tensor<32x384x14x14xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v806 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v807 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v808 = stablehlo.maximum %v805, %v806 : tensor<32x75264xf32>
    %v809 = stablehlo.minimum %v808, %v807 : tensor<32x75264xf32>
    %v810 = stablehlo.reshape %v809 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v811 = stablehlo.convolution(%v810, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v812 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v813 = stablehlo.add %v811, %v812 : tensor<32x64x14x14xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v817 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v818 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v819 = stablehlo.reduce(%v815 init: %v816) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v820 = stablehlo.broadcast_in_dim %v819, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v821 = stablehlo.divide %v820, %v817 : tensor<32x64x14x14xf32>
    %v822 = stablehlo.subtract %v815, %v821 : tensor<32x64x14x14xf32>
    %v823 = stablehlo.multiply %v822, %v822 : tensor<32x64x14x14xf32>
    %v824 = stablehlo.reduce(%v823 init: %v816) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v825 = stablehlo.broadcast_in_dim %v824, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v826 = stablehlo.divide %v825, %v817 : tensor<32x64x14x14xf32>
    %v827 = stablehlo.add %v826, %v818 : tensor<32x64x14x14xf32>
    %v828 = stablehlo.rsqrt %v827 : tensor<32x64x14x14xf32>
    %v829 = stablehlo.multiply %v822, %v828 : tensor<32x64x14x14xf32>
    %v830 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v831 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v832 = stablehlo.multiply %v829, %v830 : tensor<32x64x14x14xf32>
    %v833 = stablehlo.add %v832, %v831 : tensor<32x64x14x14xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v835 = stablehlo.add %v834, %v751 : tensor<32x12544xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v837 = stablehlo.convolution(%v836, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v838 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v839 = stablehlo.add %v837, %v838 : tensor<32x384x14x14xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v843 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v844 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v845 = stablehlo.reduce(%v841 init: %v842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v846 = stablehlo.broadcast_in_dim %v845, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v847 = stablehlo.divide %v846, %v843 : tensor<32x384x14x14xf32>
    %v848 = stablehlo.subtract %v841, %v847 : tensor<32x384x14x14xf32>
    %v849 = stablehlo.multiply %v848, %v848 : tensor<32x384x14x14xf32>
    %v850 = stablehlo.reduce(%v849 init: %v842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v851 = stablehlo.broadcast_in_dim %v850, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v852 = stablehlo.divide %v851, %v843 : tensor<32x384x14x14xf32>
    %v853 = stablehlo.add %v852, %v844 : tensor<32x384x14x14xf32>
    %v854 = stablehlo.rsqrt %v853 : tensor<32x384x14x14xf32>
    %v855 = stablehlo.multiply %v848, %v854 : tensor<32x384x14x14xf32>
    %v856 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v857 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v858 = stablehlo.multiply %v855, %v856 : tensor<32x384x14x14xf32>
    %v859 = stablehlo.add %v858, %v857 : tensor<32x384x14x14xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v861 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v862 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v863 = stablehlo.maximum %v860, %v861 : tensor<32x75264xf32>
    %v864 = stablehlo.minimum %v863, %v862 : tensor<32x75264xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v866 = stablehlo.convolution(%v865, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v867 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v868 = stablehlo.add %v866, %v867 : tensor<32x384x14x14xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v871 = stablehlo.constant dense<0.0> : tensor<f32>
    %v872 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v873 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v874 = stablehlo.reduce(%v870 init: %v871) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v875 = stablehlo.broadcast_in_dim %v874, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v876 = stablehlo.divide %v875, %v872 : tensor<32x384x14x14xf32>
    %v877 = stablehlo.subtract %v870, %v876 : tensor<32x384x14x14xf32>
    %v878 = stablehlo.multiply %v877, %v877 : tensor<32x384x14x14xf32>
    %v879 = stablehlo.reduce(%v878 init: %v871) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v880 = stablehlo.broadcast_in_dim %v879, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v881 = stablehlo.divide %v880, %v872 : tensor<32x384x14x14xf32>
    %v882 = stablehlo.add %v881, %v873 : tensor<32x384x14x14xf32>
    %v883 = stablehlo.rsqrt %v882 : tensor<32x384x14x14xf32>
    %v884 = stablehlo.multiply %v877, %v883 : tensor<32x384x14x14xf32>
    %v885 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v886 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v887 = stablehlo.multiply %v884, %v885 : tensor<32x384x14x14xf32>
    %v888 = stablehlo.add %v887, %v886 : tensor<32x384x14x14xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v890 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v891 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v892 = stablehlo.maximum %v889, %v890 : tensor<32x75264xf32>
    %v893 = stablehlo.minimum %v892, %v891 : tensor<32x75264xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v895 = stablehlo.convolution(%v894, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v896 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v897 = stablehlo.add %v895, %v896 : tensor<32x96x14x14xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v899 = stablehlo.reshape %v898 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v901 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v902 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v903 = stablehlo.reduce(%v899 init: %v900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v904 = stablehlo.broadcast_in_dim %v903, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v905 = stablehlo.divide %v904, %v901 : tensor<32x96x14x14xf32>
    %v906 = stablehlo.subtract %v899, %v905 : tensor<32x96x14x14xf32>
    %v907 = stablehlo.multiply %v906, %v906 : tensor<32x96x14x14xf32>
    %v908 = stablehlo.reduce(%v907 init: %v900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v909 = stablehlo.broadcast_in_dim %v908, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v910 = stablehlo.divide %v909, %v901 : tensor<32x96x14x14xf32>
    %v911 = stablehlo.add %v910, %v902 : tensor<32x96x14x14xf32>
    %v912 = stablehlo.rsqrt %v911 : tensor<32x96x14x14xf32>
    %v913 = stablehlo.multiply %v906, %v912 : tensor<32x96x14x14xf32>
    %v914 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v915 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v916 = stablehlo.multiply %v913, %v914 : tensor<32x96x14x14xf32>
    %v917 = stablehlo.add %v916, %v915 : tensor<32x96x14x14xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v920 = stablehlo.convolution(%v919, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v921 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v922 = stablehlo.add %v920, %v921 : tensor<32x576x14x14xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v924 = stablehlo.reshape %v923 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v925 = stablehlo.constant dense<0.0> : tensor<f32>
    %v926 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v927 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v928 = stablehlo.reduce(%v924 init: %v925) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v929 = stablehlo.broadcast_in_dim %v928, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v930 = stablehlo.divide %v929, %v926 : tensor<32x576x14x14xf32>
    %v931 = stablehlo.subtract %v924, %v930 : tensor<32x576x14x14xf32>
    %v932 = stablehlo.multiply %v931, %v931 : tensor<32x576x14x14xf32>
    %v933 = stablehlo.reduce(%v932 init: %v925) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v934 = stablehlo.broadcast_in_dim %v933, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v935 = stablehlo.divide %v934, %v926 : tensor<32x576x14x14xf32>
    %v936 = stablehlo.add %v935, %v927 : tensor<32x576x14x14xf32>
    %v937 = stablehlo.rsqrt %v936 : tensor<32x576x14x14xf32>
    %v938 = stablehlo.multiply %v931, %v937 : tensor<32x576x14x14xf32>
    %v939 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v940 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v941 = stablehlo.multiply %v938, %v939 : tensor<32x576x14x14xf32>
    %v942 = stablehlo.add %v941, %v940 : tensor<32x576x14x14xf32>
    %v943 = stablehlo.reshape %v942 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v944 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v945 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v946 = stablehlo.maximum %v943, %v944 : tensor<32x112896xf32>
    %v947 = stablehlo.minimum %v946, %v945 : tensor<32x112896xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v949 = stablehlo.convolution(%v948, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v950 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v951 = stablehlo.add %v949, %v950 : tensor<32x576x14x14xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v955 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v956 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v957 = stablehlo.reduce(%v953 init: %v954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v958 = stablehlo.broadcast_in_dim %v957, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v959 = stablehlo.divide %v958, %v955 : tensor<32x576x14x14xf32>
    %v960 = stablehlo.subtract %v953, %v959 : tensor<32x576x14x14xf32>
    %v961 = stablehlo.multiply %v960, %v960 : tensor<32x576x14x14xf32>
    %v962 = stablehlo.reduce(%v961 init: %v954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v963 = stablehlo.broadcast_in_dim %v962, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v964 = stablehlo.divide %v963, %v955 : tensor<32x576x14x14xf32>
    %v965 = stablehlo.add %v964, %v956 : tensor<32x576x14x14xf32>
    %v966 = stablehlo.rsqrt %v965 : tensor<32x576x14x14xf32>
    %v967 = stablehlo.multiply %v960, %v966 : tensor<32x576x14x14xf32>
    %v968 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v969 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v970 = stablehlo.multiply %v967, %v968 : tensor<32x576x14x14xf32>
    %v971 = stablehlo.add %v970, %v969 : tensor<32x576x14x14xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v973 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v974 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v975 = stablehlo.maximum %v972, %v973 : tensor<32x112896xf32>
    %v976 = stablehlo.minimum %v975, %v974 : tensor<32x112896xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v978 = stablehlo.convolution(%v977, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v979 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v980 = stablehlo.add %v978, %v979 : tensor<32x96x14x14xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v983 = stablehlo.constant dense<0.0> : tensor<f32>
    %v984 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v985 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v986 = stablehlo.reduce(%v982 init: %v983) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v987 = stablehlo.broadcast_in_dim %v986, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v988 = stablehlo.divide %v987, %v984 : tensor<32x96x14x14xf32>
    %v989 = stablehlo.subtract %v982, %v988 : tensor<32x96x14x14xf32>
    %v990 = stablehlo.multiply %v989, %v989 : tensor<32x96x14x14xf32>
    %v991 = stablehlo.reduce(%v990 init: %v983) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v992 = stablehlo.broadcast_in_dim %v991, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v993 = stablehlo.divide %v992, %v984 : tensor<32x96x14x14xf32>
    %v994 = stablehlo.add %v993, %v985 : tensor<32x96x14x14xf32>
    %v995 = stablehlo.rsqrt %v994 : tensor<32x96x14x14xf32>
    %v996 = stablehlo.multiply %v989, %v995 : tensor<32x96x14x14xf32>
    %v997 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v998 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v999 = stablehlo.multiply %v996, %v997 : tensor<32x96x14x14xf32>
    %v1000 = stablehlo.add %v999, %v998 : tensor<32x96x14x14xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1002 = stablehlo.add %v1001, %v918 : tensor<32x18816xf32>
    %v1003 = stablehlo.reshape %v1002 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1004 = stablehlo.convolution(%v1003, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v1005 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1006 = stablehlo.add %v1004, %v1005 : tensor<32x576x14x14xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1010 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v1011 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v1012 = stablehlo.reduce(%v1008 init: %v1009) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v1013 = stablehlo.broadcast_in_dim %v1012, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1014 = stablehlo.divide %v1013, %v1010 : tensor<32x576x14x14xf32>
    %v1015 = stablehlo.subtract %v1008, %v1014 : tensor<32x576x14x14xf32>
    %v1016 = stablehlo.multiply %v1015, %v1015 : tensor<32x576x14x14xf32>
    %v1017 = stablehlo.reduce(%v1016 init: %v1009) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v1018 = stablehlo.broadcast_in_dim %v1017, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1019 = stablehlo.divide %v1018, %v1010 : tensor<32x576x14x14xf32>
    %v1020 = stablehlo.add %v1019, %v1011 : tensor<32x576x14x14xf32>
    %v1021 = stablehlo.rsqrt %v1020 : tensor<32x576x14x14xf32>
    %v1022 = stablehlo.multiply %v1015, %v1021 : tensor<32x576x14x14xf32>
    %v1023 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1024 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1025 = stablehlo.multiply %v1022, %v1023 : tensor<32x576x14x14xf32>
    %v1026 = stablehlo.add %v1025, %v1024 : tensor<32x576x14x14xf32>
    %v1027 = stablehlo.reshape %v1026 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1028 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v1029 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v1030 = stablehlo.maximum %v1027, %v1028 : tensor<32x112896xf32>
    %v1031 = stablehlo.minimum %v1030, %v1029 : tensor<32x112896xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1033 = stablehlo.convolution(%v1032, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v1034 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1035 = stablehlo.add %v1033, %v1034 : tensor<32x576x14x14xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1038 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1039 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v1040 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v1041 = stablehlo.reduce(%v1037 init: %v1038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v1042 = stablehlo.broadcast_in_dim %v1041, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1043 = stablehlo.divide %v1042, %v1039 : tensor<32x576x14x14xf32>
    %v1044 = stablehlo.subtract %v1037, %v1043 : tensor<32x576x14x14xf32>
    %v1045 = stablehlo.multiply %v1044, %v1044 : tensor<32x576x14x14xf32>
    %v1046 = stablehlo.reduce(%v1045 init: %v1038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v1047 = stablehlo.broadcast_in_dim %v1046, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1048 = stablehlo.divide %v1047, %v1039 : tensor<32x576x14x14xf32>
    %v1049 = stablehlo.add %v1048, %v1040 : tensor<32x576x14x14xf32>
    %v1050 = stablehlo.rsqrt %v1049 : tensor<32x576x14x14xf32>
    %v1051 = stablehlo.multiply %v1044, %v1050 : tensor<32x576x14x14xf32>
    %v1052 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1053 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1054 = stablehlo.multiply %v1051, %v1052 : tensor<32x576x14x14xf32>
    %v1055 = stablehlo.add %v1054, %v1053 : tensor<32x576x14x14xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1057 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v1058 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v1059 = stablehlo.maximum %v1056, %v1057 : tensor<32x112896xf32>
    %v1060 = stablehlo.minimum %v1059, %v1058 : tensor<32x112896xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1062 = stablehlo.convolution(%v1061, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v1063 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1064 = stablehlo.add %v1062, %v1063 : tensor<32x96x14x14xf32>
    %v1065 = stablehlo.reshape %v1064 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1067 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1068 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v1069 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v1070 = stablehlo.reduce(%v1066 init: %v1067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v1071 = stablehlo.broadcast_in_dim %v1070, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1072 = stablehlo.divide %v1071, %v1068 : tensor<32x96x14x14xf32>
    %v1073 = stablehlo.subtract %v1066, %v1072 : tensor<32x96x14x14xf32>
    %v1074 = stablehlo.multiply %v1073, %v1073 : tensor<32x96x14x14xf32>
    %v1075 = stablehlo.reduce(%v1074 init: %v1067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v1076 = stablehlo.broadcast_in_dim %v1075, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1077 = stablehlo.divide %v1076, %v1068 : tensor<32x96x14x14xf32>
    %v1078 = stablehlo.add %v1077, %v1069 : tensor<32x96x14x14xf32>
    %v1079 = stablehlo.rsqrt %v1078 : tensor<32x96x14x14xf32>
    %v1080 = stablehlo.multiply %v1073, %v1079 : tensor<32x96x14x14xf32>
    %v1081 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1082 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1083 = stablehlo.multiply %v1080, %v1081 : tensor<32x96x14x14xf32>
    %v1084 = stablehlo.add %v1083, %v1082 : tensor<32x96x14x14xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1086 = stablehlo.add %v1085, %v1002 : tensor<32x18816xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1088 = stablehlo.convolution(%v1087, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v1089 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1090 = stablehlo.add %v1088, %v1089 : tensor<32x576x14x14xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1094 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v1095 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v1096 = stablehlo.reduce(%v1092 init: %v1093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v1097 = stablehlo.broadcast_in_dim %v1096, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1098 = stablehlo.divide %v1097, %v1094 : tensor<32x576x14x14xf32>
    %v1099 = stablehlo.subtract %v1092, %v1098 : tensor<32x576x14x14xf32>
    %v1100 = stablehlo.multiply %v1099, %v1099 : tensor<32x576x14x14xf32>
    %v1101 = stablehlo.reduce(%v1100 init: %v1093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v1102 = stablehlo.broadcast_in_dim %v1101, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1103 = stablehlo.divide %v1102, %v1094 : tensor<32x576x14x14xf32>
    %v1104 = stablehlo.add %v1103, %v1095 : tensor<32x576x14x14xf32>
    %v1105 = stablehlo.rsqrt %v1104 : tensor<32x576x14x14xf32>
    %v1106 = stablehlo.multiply %v1099, %v1105 : tensor<32x576x14x14xf32>
    %v1107 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1108 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1109 = stablehlo.multiply %v1106, %v1107 : tensor<32x576x14x14xf32>
    %v1110 = stablehlo.add %v1109, %v1108 : tensor<32x576x14x14xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1112 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v1113 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v1114 = stablehlo.maximum %v1111, %v1112 : tensor<32x112896xf32>
    %v1115 = stablehlo.minimum %v1114, %v1113 : tensor<32x112896xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1117 = stablehlo.convolution(%v1116, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x7x7xf32>
    %v1118 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1119 = stablehlo.add %v1117, %v1118 : tensor<32x576x7x7xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v1122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1123 = stablehlo.constant dense<1568.0> : tensor<32x576x7x7xf32>
    %v1124 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v1125 = stablehlo.reduce(%v1121 init: %v1122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v1126 = stablehlo.broadcast_in_dim %v1125, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1127 = stablehlo.divide %v1126, %v1123 : tensor<32x576x7x7xf32>
    %v1128 = stablehlo.subtract %v1121, %v1127 : tensor<32x576x7x7xf32>
    %v1129 = stablehlo.multiply %v1128, %v1128 : tensor<32x576x7x7xf32>
    %v1130 = stablehlo.reduce(%v1129 init: %v1122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v1131 = stablehlo.broadcast_in_dim %v1130, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1132 = stablehlo.divide %v1131, %v1123 : tensor<32x576x7x7xf32>
    %v1133 = stablehlo.add %v1132, %v1124 : tensor<32x576x7x7xf32>
    %v1134 = stablehlo.rsqrt %v1133 : tensor<32x576x7x7xf32>
    %v1135 = stablehlo.multiply %v1128, %v1134 : tensor<32x576x7x7xf32>
    %v1136 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1137 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1138 = stablehlo.multiply %v1135, %v1136 : tensor<32x576x7x7xf32>
    %v1139 = stablehlo.add %v1138, %v1137 : tensor<32x576x7x7xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v1141 = stablehlo.constant dense<0.0> : tensor<32x28224xf32>
    %v1142 = stablehlo.constant dense<6.0> : tensor<32x28224xf32>
    %v1143 = stablehlo.maximum %v1140, %v1141 : tensor<32x28224xf32>
    %v1144 = stablehlo.minimum %v1143, %v1142 : tensor<32x28224xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v1146 = stablehlo.convolution(%v1145, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x7x7xf32>, tensor<160x576x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1147 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1148 = stablehlo.add %v1146, %v1147 : tensor<32x160x7x7xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1152 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v1153 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1154 = stablehlo.reduce(%v1150 init: %v1151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1155 = stablehlo.broadcast_in_dim %v1154, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1156 = stablehlo.divide %v1155, %v1152 : tensor<32x160x7x7xf32>
    %v1157 = stablehlo.subtract %v1150, %v1156 : tensor<32x160x7x7xf32>
    %v1158 = stablehlo.multiply %v1157, %v1157 : tensor<32x160x7x7xf32>
    %v1159 = stablehlo.reduce(%v1158 init: %v1151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1160 = stablehlo.broadcast_in_dim %v1159, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1161 = stablehlo.divide %v1160, %v1152 : tensor<32x160x7x7xf32>
    %v1162 = stablehlo.add %v1161, %v1153 : tensor<32x160x7x7xf32>
    %v1163 = stablehlo.rsqrt %v1162 : tensor<32x160x7x7xf32>
    %v1164 = stablehlo.multiply %v1157, %v1163 : tensor<32x160x7x7xf32>
    %v1165 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1166 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1167 = stablehlo.multiply %v1164, %v1165 : tensor<32x160x7x7xf32>
    %v1168 = stablehlo.add %v1167, %v1166 : tensor<32x160x7x7xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1170 = stablehlo.reshape %v1169 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1171 = stablehlo.convolution(%v1170, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1172 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1173 = stablehlo.add %v1171, %v1172 : tensor<32x960x7x7xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1177 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1178 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1179 = stablehlo.reduce(%v1175 init: %v1176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1180 = stablehlo.broadcast_in_dim %v1179, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1181 = stablehlo.divide %v1180, %v1177 : tensor<32x960x7x7xf32>
    %v1182 = stablehlo.subtract %v1175, %v1181 : tensor<32x960x7x7xf32>
    %v1183 = stablehlo.multiply %v1182, %v1182 : tensor<32x960x7x7xf32>
    %v1184 = stablehlo.reduce(%v1183 init: %v1176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1185 = stablehlo.broadcast_in_dim %v1184, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1186 = stablehlo.divide %v1185, %v1177 : tensor<32x960x7x7xf32>
    %v1187 = stablehlo.add %v1186, %v1178 : tensor<32x960x7x7xf32>
    %v1188 = stablehlo.rsqrt %v1187 : tensor<32x960x7x7xf32>
    %v1189 = stablehlo.multiply %v1182, %v1188 : tensor<32x960x7x7xf32>
    %v1190 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1191 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1192 = stablehlo.multiply %v1189, %v1190 : tensor<32x960x7x7xf32>
    %v1193 = stablehlo.add %v1192, %v1191 : tensor<32x960x7x7xf32>
    %v1194 = stablehlo.reshape %v1193 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1195 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1196 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1197 = stablehlo.maximum %v1194, %v1195 : tensor<32x47040xf32>
    %v1198 = stablehlo.minimum %v1197, %v1196 : tensor<32x47040xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1200 = stablehlo.convolution(%v1199, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1201 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1202 = stablehlo.add %v1200, %v1201 : tensor<32x960x7x7xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1205 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1206 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1207 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1208 = stablehlo.reduce(%v1204 init: %v1205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1209 = stablehlo.broadcast_in_dim %v1208, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1210 = stablehlo.divide %v1209, %v1206 : tensor<32x960x7x7xf32>
    %v1211 = stablehlo.subtract %v1204, %v1210 : tensor<32x960x7x7xf32>
    %v1212 = stablehlo.multiply %v1211, %v1211 : tensor<32x960x7x7xf32>
    %v1213 = stablehlo.reduce(%v1212 init: %v1205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1214 = stablehlo.broadcast_in_dim %v1213, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1215 = stablehlo.divide %v1214, %v1206 : tensor<32x960x7x7xf32>
    %v1216 = stablehlo.add %v1215, %v1207 : tensor<32x960x7x7xf32>
    %v1217 = stablehlo.rsqrt %v1216 : tensor<32x960x7x7xf32>
    %v1218 = stablehlo.multiply %v1211, %v1217 : tensor<32x960x7x7xf32>
    %v1219 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1220 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1221 = stablehlo.multiply %v1218, %v1219 : tensor<32x960x7x7xf32>
    %v1222 = stablehlo.add %v1221, %v1220 : tensor<32x960x7x7xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1224 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1225 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1226 = stablehlo.maximum %v1223, %v1224 : tensor<32x47040xf32>
    %v1227 = stablehlo.minimum %v1226, %v1225 : tensor<32x47040xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1229 = stablehlo.convolution(%v1228, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1230 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1231 = stablehlo.add %v1229, %v1230 : tensor<32x160x7x7xf32>
    %v1232 = stablehlo.reshape %v1231 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1233 = stablehlo.reshape %v1232 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1234 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1235 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v1236 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1237 = stablehlo.reduce(%v1233 init: %v1234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1238 = stablehlo.broadcast_in_dim %v1237, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1239 = stablehlo.divide %v1238, %v1235 : tensor<32x160x7x7xf32>
    %v1240 = stablehlo.subtract %v1233, %v1239 : tensor<32x160x7x7xf32>
    %v1241 = stablehlo.multiply %v1240, %v1240 : tensor<32x160x7x7xf32>
    %v1242 = stablehlo.reduce(%v1241 init: %v1234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1243 = stablehlo.broadcast_in_dim %v1242, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1244 = stablehlo.divide %v1243, %v1235 : tensor<32x160x7x7xf32>
    %v1245 = stablehlo.add %v1244, %v1236 : tensor<32x160x7x7xf32>
    %v1246 = stablehlo.rsqrt %v1245 : tensor<32x160x7x7xf32>
    %v1247 = stablehlo.multiply %v1240, %v1246 : tensor<32x160x7x7xf32>
    %v1248 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1249 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1250 = stablehlo.multiply %v1247, %v1248 : tensor<32x160x7x7xf32>
    %v1251 = stablehlo.add %v1250, %v1249 : tensor<32x160x7x7xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1253 = stablehlo.add %v1252, %v1169 : tensor<32x7840xf32>
    %v1254 = stablehlo.reshape %v1253 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1255 = stablehlo.convolution(%v1254, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1256 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1257 = stablehlo.add %v1255, %v1256 : tensor<32x960x7x7xf32>
    %v1258 = stablehlo.reshape %v1257 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1261 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1262 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1263 = stablehlo.reduce(%v1259 init: %v1260) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1264 = stablehlo.broadcast_in_dim %v1263, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1265 = stablehlo.divide %v1264, %v1261 : tensor<32x960x7x7xf32>
    %v1266 = stablehlo.subtract %v1259, %v1265 : tensor<32x960x7x7xf32>
    %v1267 = stablehlo.multiply %v1266, %v1266 : tensor<32x960x7x7xf32>
    %v1268 = stablehlo.reduce(%v1267 init: %v1260) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1269 = stablehlo.broadcast_in_dim %v1268, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1270 = stablehlo.divide %v1269, %v1261 : tensor<32x960x7x7xf32>
    %v1271 = stablehlo.add %v1270, %v1262 : tensor<32x960x7x7xf32>
    %v1272 = stablehlo.rsqrt %v1271 : tensor<32x960x7x7xf32>
    %v1273 = stablehlo.multiply %v1266, %v1272 : tensor<32x960x7x7xf32>
    %v1274 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1275 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1276 = stablehlo.multiply %v1273, %v1274 : tensor<32x960x7x7xf32>
    %v1277 = stablehlo.add %v1276, %v1275 : tensor<32x960x7x7xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1279 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1280 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1281 = stablehlo.maximum %v1278, %v1279 : tensor<32x47040xf32>
    %v1282 = stablehlo.minimum %v1281, %v1280 : tensor<32x47040xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1284 = stablehlo.convolution(%v1283, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1285 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1286 = stablehlo.add %v1284, %v1285 : tensor<32x960x7x7xf32>
    %v1287 = stablehlo.reshape %v1286 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1288 = stablehlo.reshape %v1287 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1290 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1291 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1292 = stablehlo.reduce(%v1288 init: %v1289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1293 = stablehlo.broadcast_in_dim %v1292, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1294 = stablehlo.divide %v1293, %v1290 : tensor<32x960x7x7xf32>
    %v1295 = stablehlo.subtract %v1288, %v1294 : tensor<32x960x7x7xf32>
    %v1296 = stablehlo.multiply %v1295, %v1295 : tensor<32x960x7x7xf32>
    %v1297 = stablehlo.reduce(%v1296 init: %v1289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1298 = stablehlo.broadcast_in_dim %v1297, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1299 = stablehlo.divide %v1298, %v1290 : tensor<32x960x7x7xf32>
    %v1300 = stablehlo.add %v1299, %v1291 : tensor<32x960x7x7xf32>
    %v1301 = stablehlo.rsqrt %v1300 : tensor<32x960x7x7xf32>
    %v1302 = stablehlo.multiply %v1295, %v1301 : tensor<32x960x7x7xf32>
    %v1303 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1304 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1305 = stablehlo.multiply %v1302, %v1303 : tensor<32x960x7x7xf32>
    %v1306 = stablehlo.add %v1305, %v1304 : tensor<32x960x7x7xf32>
    %v1307 = stablehlo.reshape %v1306 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1308 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1309 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1310 = stablehlo.maximum %v1307, %v1308 : tensor<32x47040xf32>
    %v1311 = stablehlo.minimum %v1310, %v1309 : tensor<32x47040xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1313 = stablehlo.convolution(%v1312, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1314 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1315 = stablehlo.add %v1313, %v1314 : tensor<32x160x7x7xf32>
    %v1316 = stablehlo.reshape %v1315 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1317 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1318 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1319 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v1320 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1321 = stablehlo.reduce(%v1317 init: %v1318) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1322 = stablehlo.broadcast_in_dim %v1321, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1323 = stablehlo.divide %v1322, %v1319 : tensor<32x160x7x7xf32>
    %v1324 = stablehlo.subtract %v1317, %v1323 : tensor<32x160x7x7xf32>
    %v1325 = stablehlo.multiply %v1324, %v1324 : tensor<32x160x7x7xf32>
    %v1326 = stablehlo.reduce(%v1325 init: %v1318) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1327 = stablehlo.broadcast_in_dim %v1326, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1328 = stablehlo.divide %v1327, %v1319 : tensor<32x160x7x7xf32>
    %v1329 = stablehlo.add %v1328, %v1320 : tensor<32x160x7x7xf32>
    %v1330 = stablehlo.rsqrt %v1329 : tensor<32x160x7x7xf32>
    %v1331 = stablehlo.multiply %v1324, %v1330 : tensor<32x160x7x7xf32>
    %v1332 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1333 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1334 = stablehlo.multiply %v1331, %v1332 : tensor<32x160x7x7xf32>
    %v1335 = stablehlo.add %v1334, %v1333 : tensor<32x160x7x7xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1337 = stablehlo.add %v1336, %v1253 : tensor<32x7840xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1339 = stablehlo.convolution(%v1338, %b17eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1340 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1341 = stablehlo.add %v1339, %v1340 : tensor<32x960x7x7xf32>
    %v1342 = stablehlo.reshape %v1341 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1343 = stablehlo.reshape %v1342 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1344 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1345 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1346 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1347 = stablehlo.reduce(%v1343 init: %v1344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1348 = stablehlo.broadcast_in_dim %v1347, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1349 = stablehlo.divide %v1348, %v1345 : tensor<32x960x7x7xf32>
    %v1350 = stablehlo.subtract %v1343, %v1349 : tensor<32x960x7x7xf32>
    %v1351 = stablehlo.multiply %v1350, %v1350 : tensor<32x960x7x7xf32>
    %v1352 = stablehlo.reduce(%v1351 init: %v1344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1353 = stablehlo.broadcast_in_dim %v1352, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1354 = stablehlo.divide %v1353, %v1345 : tensor<32x960x7x7xf32>
    %v1355 = stablehlo.add %v1354, %v1346 : tensor<32x960x7x7xf32>
    %v1356 = stablehlo.rsqrt %v1355 : tensor<32x960x7x7xf32>
    %v1357 = stablehlo.multiply %v1350, %v1356 : tensor<32x960x7x7xf32>
    %v1358 = stablehlo.broadcast_in_dim %b17eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1359 = stablehlo.broadcast_in_dim %b17ebt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1360 = stablehlo.multiply %v1357, %v1358 : tensor<32x960x7x7xf32>
    %v1361 = stablehlo.add %v1360, %v1359 : tensor<32x960x7x7xf32>
    %v1362 = stablehlo.reshape %v1361 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1363 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1364 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1365 = stablehlo.maximum %v1362, %v1363 : tensor<32x47040xf32>
    %v1366 = stablehlo.minimum %v1365, %v1364 : tensor<32x47040xf32>
    %v1367 = stablehlo.reshape %v1366 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1368 = stablehlo.convolution(%v1367, %b17dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1369 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1370 = stablehlo.add %v1368, %v1369 : tensor<32x960x7x7xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1373 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1374 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1375 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1376 = stablehlo.reduce(%v1372 init: %v1373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1377 = stablehlo.broadcast_in_dim %v1376, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1378 = stablehlo.divide %v1377, %v1374 : tensor<32x960x7x7xf32>
    %v1379 = stablehlo.subtract %v1372, %v1378 : tensor<32x960x7x7xf32>
    %v1380 = stablehlo.multiply %v1379, %v1379 : tensor<32x960x7x7xf32>
    %v1381 = stablehlo.reduce(%v1380 init: %v1373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1382 = stablehlo.broadcast_in_dim %v1381, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1383 = stablehlo.divide %v1382, %v1374 : tensor<32x960x7x7xf32>
    %v1384 = stablehlo.add %v1383, %v1375 : tensor<32x960x7x7xf32>
    %v1385 = stablehlo.rsqrt %v1384 : tensor<32x960x7x7xf32>
    %v1386 = stablehlo.multiply %v1379, %v1385 : tensor<32x960x7x7xf32>
    %v1387 = stablehlo.broadcast_in_dim %b17dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1388 = stablehlo.broadcast_in_dim %b17dbt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1389 = stablehlo.multiply %v1386, %v1387 : tensor<32x960x7x7xf32>
    %v1390 = stablehlo.add %v1389, %v1388 : tensor<32x960x7x7xf32>
    %v1391 = stablehlo.reshape %v1390 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1392 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1393 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1394 = stablehlo.maximum %v1391, %v1392 : tensor<32x47040xf32>
    %v1395 = stablehlo.minimum %v1394, %v1393 : tensor<32x47040xf32>
    %v1396 = stablehlo.reshape %v1395 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1397 = stablehlo.convolution(%v1396, %b17pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<320x960x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1398 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1399 = stablehlo.add %v1397, %v1398 : tensor<32x320x7x7xf32>
    %v1400 = stablehlo.reshape %v1399 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1401 = stablehlo.reshape %v1400 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1403 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1404 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1405 = stablehlo.reduce(%v1401 init: %v1402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1406 = stablehlo.broadcast_in_dim %v1405, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1407 = stablehlo.divide %v1406, %v1403 : tensor<32x320x7x7xf32>
    %v1408 = stablehlo.subtract %v1401, %v1407 : tensor<32x320x7x7xf32>
    %v1409 = stablehlo.multiply %v1408, %v1408 : tensor<32x320x7x7xf32>
    %v1410 = stablehlo.reduce(%v1409 init: %v1402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1411 = stablehlo.broadcast_in_dim %v1410, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1412 = stablehlo.divide %v1411, %v1403 : tensor<32x320x7x7xf32>
    %v1413 = stablehlo.add %v1412, %v1404 : tensor<32x320x7x7xf32>
    %v1414 = stablehlo.rsqrt %v1413 : tensor<32x320x7x7xf32>
    %v1415 = stablehlo.multiply %v1408, %v1414 : tensor<32x320x7x7xf32>
    %v1416 = stablehlo.broadcast_in_dim %b17pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1417 = stablehlo.broadcast_in_dim %b17pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1418 = stablehlo.multiply %v1415, %v1416 : tensor<32x320x7x7xf32>
    %v1419 = stablehlo.add %v1418, %v1417 : tensor<32x320x7x7xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1421 = stablehlo.reshape %v1420 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1422 = stablehlo.convolution(%v1421, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1423 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1424 = stablehlo.add %v1422, %v1423 : tensor<32x1280x7x7xf32>
    %v1425 = stablehlo.reshape %v1424 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1426 = stablehlo.reshape %v1425 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1427 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1428 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1429 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1430 = stablehlo.reduce(%v1426 init: %v1427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1431 = stablehlo.broadcast_in_dim %v1430, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1432 = stablehlo.divide %v1431, %v1428 : tensor<32x1280x7x7xf32>
    %v1433 = stablehlo.subtract %v1426, %v1432 : tensor<32x1280x7x7xf32>
    %v1434 = stablehlo.multiply %v1433, %v1433 : tensor<32x1280x7x7xf32>
    %v1435 = stablehlo.reduce(%v1434 init: %v1427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1436 = stablehlo.broadcast_in_dim %v1435, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1437 = stablehlo.divide %v1436, %v1428 : tensor<32x1280x7x7xf32>
    %v1438 = stablehlo.add %v1437, %v1429 : tensor<32x1280x7x7xf32>
    %v1439 = stablehlo.rsqrt %v1438 : tensor<32x1280x7x7xf32>
    %v1440 = stablehlo.multiply %v1433, %v1439 : tensor<32x1280x7x7xf32>
    %v1441 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1442 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1443 = stablehlo.multiply %v1440, %v1441 : tensor<32x1280x7x7xf32>
    %v1444 = stablehlo.add %v1443, %v1442 : tensor<32x1280x7x7xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1446 = stablehlo.constant dense<0.0> : tensor<32x62720xf32>
    %v1447 = stablehlo.constant dense<6.0> : tensor<32x62720xf32>
    %v1448 = stablehlo.maximum %v1445, %v1446 : tensor<32x62720xf32>
    %v1449 = stablehlo.minimum %v1448, %v1447 : tensor<32x62720xf32>
    %v1450 = stablehlo.reshape %v1449 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1451 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1452 = stablehlo.reduce(%v1450 init: %v1451) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1453 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1454 = stablehlo.divide %v1452, %v1453 : tensor<32x1280xf32>
    %v1455 = stablehlo.dot_general %v1454, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1456 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1457 = stablehlo.add %v1455, %v1456 : tensor<32x10xf32>
    %v1458 = stablehlo.reshape %v1457 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1459 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1460 = stablehlo.exponential %v1458 : tensor<32x1x10xf32>
    %v1461 = stablehlo.reduce(%v1460 init: %v1459) applies stablehlo.add across dimensions = [2] : (tensor<32x1x10xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1462 = stablehlo.broadcast_in_dim %v1461, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x10xf32>
    %v1463 = stablehlo.divide %v1460, %v1462 : tensor<32x1x10xf32>
    %v1464 = stablehlo.reshape %v1463 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v1465 = stablehlo.subtract %v1464, %onehot : tensor<32x10xf32>
    %v1466 = stablehlo.constant dense<0.100000> : tensor<32x10xf32>
    %v1467 = stablehlo.multiply %onehot, %v1466 : tensor<32x10xf32>
    %v1468 = stablehlo.add %v1465, %v1467 : tensor<32x10xf32>
    %v1469 = stablehlo.constant dense<-0.010000> : tensor<32x10xf32>
    %v1470 = stablehlo.add %v1468, %v1469 : tensor<32x10xf32>
    %v1471 = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %v1472 = stablehlo.divide %v1470, %v1471 : tensor<32x10xf32>
    %v1473 = stablehlo.reshape %v1472 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1474 = stablehlo.dot_general %v1473, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x10xf32>, tensor<1280x10xf32>) -> tensor<32x1x1280xf32>
    %v1475 = stablehlo.reshape %v1474 : (tensor<32x1x1280xf32>) -> tensor<32x1280xf32>
    %v1476 = stablehlo.dot_general %v1454, %v1472, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<32x10xf32>) -> tensor<1280x10xf32>
    %v1477 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1478 = stablehlo.reduce(%v1472 init: %v1477) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1479 = stablehlo.broadcast_in_dim %v1475, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1480 = stablehlo.constant dense<49.0> : tensor<32x1280x7x7xf32>
    %v1481 = stablehlo.divide %v1479, %v1480 : tensor<32x1280x7x7xf32>
    %v1482 = stablehlo.reshape %v1481 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1483 = stablehlo.constant dense<0.0> : tensor<32x62720xf32>
    %v1484 = stablehlo.constant dense<6.0> : tensor<32x62720xf32>
    %v1485 = stablehlo.compare GT, %v1445, %v1483 : (tensor<32x62720xf32>, tensor<32x62720xf32>) -> tensor<32x62720xi1>
    %v1486 = stablehlo.compare LT, %v1445, %v1484 : (tensor<32x62720xf32>, tensor<32x62720xf32>) -> tensor<32x62720xi1>
    %v1487 = stablehlo.and %v1485, %v1486 : tensor<32x62720xi1>
    %v1488 = stablehlo.select %v1487, %v1482, %v1483 : tensor<32x62720xi1>, tensor<32x62720xf32>
    %v1489 = stablehlo.reshape %v1425 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1490 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1491 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1492 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1493 = stablehlo.reduce(%v1489 init: %v1490) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1494 = stablehlo.broadcast_in_dim %v1493, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1495 = stablehlo.divide %v1494, %v1491 : tensor<32x1280x7x7xf32>
    %v1496 = stablehlo.subtract %v1489, %v1495 : tensor<32x1280x7x7xf32>
    %v1497 = stablehlo.multiply %v1496, %v1496 : tensor<32x1280x7x7xf32>
    %v1498 = stablehlo.reduce(%v1497 init: %v1490) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1499 = stablehlo.broadcast_in_dim %v1498, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1500 = stablehlo.divide %v1499, %v1491 : tensor<32x1280x7x7xf32>
    %v1501 = stablehlo.add %v1500, %v1492 : tensor<32x1280x7x7xf32>
    %v1502 = stablehlo.rsqrt %v1501 : tensor<32x1280x7x7xf32>
    %v1503 = stablehlo.multiply %v1496, %v1502 : tensor<32x1280x7x7xf32>
    %v1504 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1505 = stablehlo.reshape %v1488 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1506 = stablehlo.multiply %v1504, %v1505 : tensor<32x1280x7x7xf32>
    %v1507 = stablehlo.reduce(%v1506 init: %v1490) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1508 = stablehlo.broadcast_in_dim %v1507, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1509 = stablehlo.multiply %v1503, %v1506 : tensor<32x1280x7x7xf32>
    %v1510 = stablehlo.reduce(%v1509 init: %v1490) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1511 = stablehlo.broadcast_in_dim %v1510, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1512 = stablehlo.multiply %v1506, %v1491 : tensor<32x1280x7x7xf32>
    %v1513 = stablehlo.subtract %v1512, %v1508 : tensor<32x1280x7x7xf32>
    %v1514 = stablehlo.multiply %v1503, %v1511 : tensor<32x1280x7x7xf32>
    %v1515 = stablehlo.subtract %v1513, %v1514 : tensor<32x1280x7x7xf32>
    %v1516 = stablehlo.divide %v1502, %v1491 : tensor<32x1280x7x7xf32>
    %v1517 = stablehlo.multiply %v1516, %v1515 : tensor<32x1280x7x7xf32>
    %v1518 = stablehlo.reshape %v1517 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1520 = stablehlo.reverse %hW, dims = [2, 3] : tensor<1280x320x1x1xf32>
    %v1521 = stablehlo.transpose %v1520, dims = [1, 0, 2, 3] : (tensor<1280x320x1x1xf32>) -> tensor<320x1280x1x1xf32>
    %v1522 = stablehlo.convolution(%v1519, %v1521)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1280x7x7xf32>, tensor<320x1280x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1523 = stablehlo.reshape %v1522 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1524 = stablehlo.reshape %v1420 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1525 = stablehlo.reshape %v1518 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1526 = stablehlo.transpose %v1524, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v1527 = stablehlo.transpose %v1525, dims = [1, 0, 2, 3] : (tensor<32x1280x7x7xf32>) -> tensor<1280x32x7x7xf32>
    %v1528 = stablehlo.convolution(%v1526, %v1527)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<320x32x7x7xf32>, tensor<1280x32x7x7xf32>) -> tensor<320x1280x1x1xf32>
    %v1529 = stablehlo.transpose %v1528, dims = [1, 0, 2, 3] : (tensor<320x1280x1x1xf32>) -> tensor<1280x320x1x1xf32>
    %v1530 = stablehlo.reshape %v1425 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1531 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1532 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1533 = stablehlo.reduce(%v1530 init: %v1531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1534 = stablehlo.broadcast_in_dim %v1533, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1535 = stablehlo.divide %v1534, %v1532 : tensor<32x1280x7x7xf32>
    %v1536 = stablehlo.subtract %v1530, %v1535 : tensor<32x1280x7x7xf32>
    %v1537 = stablehlo.multiply %v1536, %v1536 : tensor<32x1280x7x7xf32>
    %v1538 = stablehlo.reduce(%v1537 init: %v1531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1539 = stablehlo.broadcast_in_dim %v1538, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1540 = stablehlo.divide %v1539, %v1532 : tensor<32x1280x7x7xf32>
    %v1541 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1542 = stablehlo.add %v1540, %v1541 : tensor<32x1280x7x7xf32>
    %v1543 = stablehlo.rsqrt %v1542 : tensor<32x1280x7x7xf32>
    %v1544 = stablehlo.multiply %v1536, %v1543 : tensor<32x1280x7x7xf32>
    %v1545 = stablehlo.reshape %v1488 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1546 = stablehlo.multiply %v1545, %v1544 : tensor<32x1280x7x7xf32>
    %v1547 = stablehlo.reduce(%v1546 init: %v1531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1548 = stablehlo.reshape %v1488 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1550 = stablehlo.reduce(%v1548 init: %v1549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1551 = stablehlo.reshape %v1400 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1553 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1554 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1555 = stablehlo.reduce(%v1551 init: %v1552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1556 = stablehlo.broadcast_in_dim %v1555, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1557 = stablehlo.divide %v1556, %v1553 : tensor<32x320x7x7xf32>
    %v1558 = stablehlo.subtract %v1551, %v1557 : tensor<32x320x7x7xf32>
    %v1559 = stablehlo.multiply %v1558, %v1558 : tensor<32x320x7x7xf32>
    %v1560 = stablehlo.reduce(%v1559 init: %v1552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1561 = stablehlo.broadcast_in_dim %v1560, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1562 = stablehlo.divide %v1561, %v1553 : tensor<32x320x7x7xf32>
    %v1563 = stablehlo.add %v1562, %v1554 : tensor<32x320x7x7xf32>
    %v1564 = stablehlo.rsqrt %v1563 : tensor<32x320x7x7xf32>
    %v1565 = stablehlo.multiply %v1558, %v1564 : tensor<32x320x7x7xf32>
    %v1566 = stablehlo.broadcast_in_dim %b17pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1567 = stablehlo.reshape %v1523 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1568 = stablehlo.multiply %v1566, %v1567 : tensor<32x320x7x7xf32>
    %v1569 = stablehlo.reduce(%v1568 init: %v1552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1570 = stablehlo.broadcast_in_dim %v1569, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1571 = stablehlo.multiply %v1565, %v1568 : tensor<32x320x7x7xf32>
    %v1572 = stablehlo.reduce(%v1571 init: %v1552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1573 = stablehlo.broadcast_in_dim %v1572, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1574 = stablehlo.multiply %v1568, %v1553 : tensor<32x320x7x7xf32>
    %v1575 = stablehlo.subtract %v1574, %v1570 : tensor<32x320x7x7xf32>
    %v1576 = stablehlo.multiply %v1565, %v1573 : tensor<32x320x7x7xf32>
    %v1577 = stablehlo.subtract %v1575, %v1576 : tensor<32x320x7x7xf32>
    %v1578 = stablehlo.divide %v1564, %v1553 : tensor<32x320x7x7xf32>
    %v1579 = stablehlo.multiply %v1578, %v1577 : tensor<32x320x7x7xf32>
    %v1580 = stablehlo.reshape %v1579 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1581 = stablehlo.reshape %v1580 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1582 = stablehlo.reverse %b17pW, dims = [2, 3] : tensor<320x960x1x1xf32>
    %v1583 = stablehlo.transpose %v1582, dims = [1, 0, 2, 3] : (tensor<320x960x1x1xf32>) -> tensor<960x320x1x1xf32>
    %v1584 = stablehlo.convolution(%v1581, %v1583)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<960x320x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1586 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1587 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1588 = stablehlo.compare GT, %v1391, %v1586 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1589 = stablehlo.compare LT, %v1391, %v1587 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1590 = stablehlo.and %v1588, %v1589 : tensor<32x47040xi1>
    %v1591 = stablehlo.select %v1590, %v1585, %v1586 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v1592 = stablehlo.reshape %v1371 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1594 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1595 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1596 = stablehlo.reduce(%v1592 init: %v1593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1597 = stablehlo.broadcast_in_dim %v1596, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1598 = stablehlo.divide %v1597, %v1594 : tensor<32x960x7x7xf32>
    %v1599 = stablehlo.subtract %v1592, %v1598 : tensor<32x960x7x7xf32>
    %v1600 = stablehlo.multiply %v1599, %v1599 : tensor<32x960x7x7xf32>
    %v1601 = stablehlo.reduce(%v1600 init: %v1593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1602 = stablehlo.broadcast_in_dim %v1601, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1603 = stablehlo.divide %v1602, %v1594 : tensor<32x960x7x7xf32>
    %v1604 = stablehlo.add %v1603, %v1595 : tensor<32x960x7x7xf32>
    %v1605 = stablehlo.rsqrt %v1604 : tensor<32x960x7x7xf32>
    %v1606 = stablehlo.multiply %v1599, %v1605 : tensor<32x960x7x7xf32>
    %v1607 = stablehlo.broadcast_in_dim %b17dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1608 = stablehlo.reshape %v1591 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1609 = stablehlo.multiply %v1607, %v1608 : tensor<32x960x7x7xf32>
    %v1610 = stablehlo.reduce(%v1609 init: %v1593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1611 = stablehlo.broadcast_in_dim %v1610, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1612 = stablehlo.multiply %v1606, %v1609 : tensor<32x960x7x7xf32>
    %v1613 = stablehlo.reduce(%v1612 init: %v1593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1614 = stablehlo.broadcast_in_dim %v1613, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1615 = stablehlo.multiply %v1609, %v1594 : tensor<32x960x7x7xf32>
    %v1616 = stablehlo.subtract %v1615, %v1611 : tensor<32x960x7x7xf32>
    %v1617 = stablehlo.multiply %v1606, %v1614 : tensor<32x960x7x7xf32>
    %v1618 = stablehlo.subtract %v1616, %v1617 : tensor<32x960x7x7xf32>
    %v1619 = stablehlo.divide %v1605, %v1594 : tensor<32x960x7x7xf32>
    %v1620 = stablehlo.multiply %v1619, %v1618 : tensor<32x960x7x7xf32>
    %v1621 = stablehlo.reshape %v1620 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1622 = stablehlo.reshape %v1621 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1623 = stablehlo.reverse %b17dW, dims = [2, 3] : tensor<960x1x3x3xf32>
    %v1624 = stablehlo.convolution(%v1622, %v1623)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1625 = stablehlo.reshape %v1624 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1626 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1627 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1628 = stablehlo.compare GT, %v1362, %v1626 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1629 = stablehlo.compare LT, %v1362, %v1627 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1630 = stablehlo.and %v1628, %v1629 : tensor<32x47040xi1>
    %v1631 = stablehlo.select %v1630, %v1625, %v1626 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v1632 = stablehlo.reshape %v1342 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1633 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1634 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1635 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1636 = stablehlo.reduce(%v1632 init: %v1633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1637 = stablehlo.broadcast_in_dim %v1636, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1638 = stablehlo.divide %v1637, %v1634 : tensor<32x960x7x7xf32>
    %v1639 = stablehlo.subtract %v1632, %v1638 : tensor<32x960x7x7xf32>
    %v1640 = stablehlo.multiply %v1639, %v1639 : tensor<32x960x7x7xf32>
    %v1641 = stablehlo.reduce(%v1640 init: %v1633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1642 = stablehlo.broadcast_in_dim %v1641, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1643 = stablehlo.divide %v1642, %v1634 : tensor<32x960x7x7xf32>
    %v1644 = stablehlo.add %v1643, %v1635 : tensor<32x960x7x7xf32>
    %v1645 = stablehlo.rsqrt %v1644 : tensor<32x960x7x7xf32>
    %v1646 = stablehlo.multiply %v1639, %v1645 : tensor<32x960x7x7xf32>
    %v1647 = stablehlo.broadcast_in_dim %b17eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1648 = stablehlo.reshape %v1631 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1649 = stablehlo.multiply %v1647, %v1648 : tensor<32x960x7x7xf32>
    %v1650 = stablehlo.reduce(%v1649 init: %v1633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1651 = stablehlo.broadcast_in_dim %v1650, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1652 = stablehlo.multiply %v1646, %v1649 : tensor<32x960x7x7xf32>
    %v1653 = stablehlo.reduce(%v1652 init: %v1633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1654 = stablehlo.broadcast_in_dim %v1653, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1655 = stablehlo.multiply %v1649, %v1634 : tensor<32x960x7x7xf32>
    %v1656 = stablehlo.subtract %v1655, %v1651 : tensor<32x960x7x7xf32>
    %v1657 = stablehlo.multiply %v1646, %v1654 : tensor<32x960x7x7xf32>
    %v1658 = stablehlo.subtract %v1656, %v1657 : tensor<32x960x7x7xf32>
    %v1659 = stablehlo.divide %v1645, %v1634 : tensor<32x960x7x7xf32>
    %v1660 = stablehlo.multiply %v1659, %v1658 : tensor<32x960x7x7xf32>
    %v1661 = stablehlo.reshape %v1660 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1662 = stablehlo.reshape %v1661 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1663 = stablehlo.reverse %b17eW, dims = [2, 3] : tensor<960x160x1x1xf32>
    %v1664 = stablehlo.transpose %v1663, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v1665 = stablehlo.convolution(%v1662, %v1664)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1666 = stablehlo.reshape %v1665 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1667 = stablehlo.reshape %v1337 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1668 = stablehlo.reshape %v1661 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1669 = stablehlo.transpose %v1667, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v1670 = stablehlo.transpose %v1668, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1671 = stablehlo.convolution(%v1669, %v1670)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<160x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<160x960x1x1xf32>
    %v1672 = stablehlo.transpose %v1671, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v1673 = stablehlo.reshape %v1342 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1674 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1675 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1676 = stablehlo.reduce(%v1673 init: %v1674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1677 = stablehlo.broadcast_in_dim %v1676, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1678 = stablehlo.divide %v1677, %v1675 : tensor<32x960x7x7xf32>
    %v1679 = stablehlo.subtract %v1673, %v1678 : tensor<32x960x7x7xf32>
    %v1680 = stablehlo.multiply %v1679, %v1679 : tensor<32x960x7x7xf32>
    %v1681 = stablehlo.reduce(%v1680 init: %v1674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1682 = stablehlo.broadcast_in_dim %v1681, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1683 = stablehlo.divide %v1682, %v1675 : tensor<32x960x7x7xf32>
    %v1684 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1685 = stablehlo.add %v1683, %v1684 : tensor<32x960x7x7xf32>
    %v1686 = stablehlo.rsqrt %v1685 : tensor<32x960x7x7xf32>
    %v1687 = stablehlo.multiply %v1679, %v1686 : tensor<32x960x7x7xf32>
    %v1688 = stablehlo.reshape %v1631 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1689 = stablehlo.multiply %v1688, %v1687 : tensor<32x960x7x7xf32>
    %v1690 = stablehlo.reduce(%v1689 init: %v1674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1691 = stablehlo.reshape %v1631 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1693 = stablehlo.reduce(%v1691 init: %v1692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1694 = stablehlo.reshape %v1366 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1695 = stablehlo.reshape %v1621 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1696 = stablehlo.transpose %v1694, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1697 = stablehlo.transpose %v1695, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1698 = stablehlo.convolution(%v1696, %v1697)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 960 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<1x960x3x3xf32>
    %v1699 = stablehlo.reshape %v1698 : (tensor<1x960x3x3xf32>) -> tensor<960x1x3x3xf32>
    %v1700 = stablehlo.reshape %v1371 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1702 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1703 = stablehlo.reduce(%v1700 init: %v1701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1704 = stablehlo.broadcast_in_dim %v1703, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1705 = stablehlo.divide %v1704, %v1702 : tensor<32x960x7x7xf32>
    %v1706 = stablehlo.subtract %v1700, %v1705 : tensor<32x960x7x7xf32>
    %v1707 = stablehlo.multiply %v1706, %v1706 : tensor<32x960x7x7xf32>
    %v1708 = stablehlo.reduce(%v1707 init: %v1701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1709 = stablehlo.broadcast_in_dim %v1708, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1710 = stablehlo.divide %v1709, %v1702 : tensor<32x960x7x7xf32>
    %v1711 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1712 = stablehlo.add %v1710, %v1711 : tensor<32x960x7x7xf32>
    %v1713 = stablehlo.rsqrt %v1712 : tensor<32x960x7x7xf32>
    %v1714 = stablehlo.multiply %v1706, %v1713 : tensor<32x960x7x7xf32>
    %v1715 = stablehlo.reshape %v1591 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1716 = stablehlo.multiply %v1715, %v1714 : tensor<32x960x7x7xf32>
    %v1717 = stablehlo.reduce(%v1716 init: %v1701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1718 = stablehlo.reshape %v1591 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1720 = stablehlo.reduce(%v1718 init: %v1719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1721 = stablehlo.reshape %v1395 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1722 = stablehlo.reshape %v1580 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1723 = stablehlo.transpose %v1721, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1724 = stablehlo.transpose %v1722, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v1725 = stablehlo.convolution(%v1723, %v1724)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<320x32x7x7xf32>) -> tensor<960x320x1x1xf32>
    %v1726 = stablehlo.transpose %v1725, dims = [1, 0, 2, 3] : (tensor<960x320x1x1xf32>) -> tensor<320x960x1x1xf32>
    %v1727 = stablehlo.reshape %v1400 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1729 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1730 = stablehlo.reduce(%v1727 init: %v1728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1731 = stablehlo.broadcast_in_dim %v1730, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1732 = stablehlo.divide %v1731, %v1729 : tensor<32x320x7x7xf32>
    %v1733 = stablehlo.subtract %v1727, %v1732 : tensor<32x320x7x7xf32>
    %v1734 = stablehlo.multiply %v1733, %v1733 : tensor<32x320x7x7xf32>
    %v1735 = stablehlo.reduce(%v1734 init: %v1728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1736 = stablehlo.broadcast_in_dim %v1735, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1737 = stablehlo.divide %v1736, %v1729 : tensor<32x320x7x7xf32>
    %v1738 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1739 = stablehlo.add %v1737, %v1738 : tensor<32x320x7x7xf32>
    %v1740 = stablehlo.rsqrt %v1739 : tensor<32x320x7x7xf32>
    %v1741 = stablehlo.multiply %v1733, %v1740 : tensor<32x320x7x7xf32>
    %v1742 = stablehlo.reshape %v1523 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1743 = stablehlo.multiply %v1742, %v1741 : tensor<32x320x7x7xf32>
    %v1744 = stablehlo.reduce(%v1743 init: %v1728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1745 = stablehlo.reshape %v1523 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1747 = stablehlo.reduce(%v1745 init: %v1746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1748 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1750 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v1751 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1752 = stablehlo.reduce(%v1748 init: %v1749) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1753 = stablehlo.broadcast_in_dim %v1752, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1754 = stablehlo.divide %v1753, %v1750 : tensor<32x160x7x7xf32>
    %v1755 = stablehlo.subtract %v1748, %v1754 : tensor<32x160x7x7xf32>
    %v1756 = stablehlo.multiply %v1755, %v1755 : tensor<32x160x7x7xf32>
    %v1757 = stablehlo.reduce(%v1756 init: %v1749) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1758 = stablehlo.broadcast_in_dim %v1757, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1759 = stablehlo.divide %v1758, %v1750 : tensor<32x160x7x7xf32>
    %v1760 = stablehlo.add %v1759, %v1751 : tensor<32x160x7x7xf32>
    %v1761 = stablehlo.rsqrt %v1760 : tensor<32x160x7x7xf32>
    %v1762 = stablehlo.multiply %v1755, %v1761 : tensor<32x160x7x7xf32>
    %v1763 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1764 = stablehlo.reshape %v1666 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1765 = stablehlo.multiply %v1763, %v1764 : tensor<32x160x7x7xf32>
    %v1766 = stablehlo.reduce(%v1765 init: %v1749) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1767 = stablehlo.broadcast_in_dim %v1766, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1768 = stablehlo.multiply %v1762, %v1765 : tensor<32x160x7x7xf32>
    %v1769 = stablehlo.reduce(%v1768 init: %v1749) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1770 = stablehlo.broadcast_in_dim %v1769, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1771 = stablehlo.multiply %v1765, %v1750 : tensor<32x160x7x7xf32>
    %v1772 = stablehlo.subtract %v1771, %v1767 : tensor<32x160x7x7xf32>
    %v1773 = stablehlo.multiply %v1762, %v1770 : tensor<32x160x7x7xf32>
    %v1774 = stablehlo.subtract %v1772, %v1773 : tensor<32x160x7x7xf32>
    %v1775 = stablehlo.divide %v1761, %v1750 : tensor<32x160x7x7xf32>
    %v1776 = stablehlo.multiply %v1775, %v1774 : tensor<32x160x7x7xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1778 = stablehlo.reshape %v1777 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1779 = stablehlo.reverse %b16pW, dims = [2, 3] : tensor<160x960x1x1xf32>
    %v1780 = stablehlo.transpose %v1779, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v1781 = stablehlo.convolution(%v1778, %v1780)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1782 = stablehlo.reshape %v1781 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1783 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1784 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1785 = stablehlo.compare GT, %v1307, %v1783 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1786 = stablehlo.compare LT, %v1307, %v1784 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1787 = stablehlo.and %v1785, %v1786 : tensor<32x47040xi1>
    %v1788 = stablehlo.select %v1787, %v1782, %v1783 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v1789 = stablehlo.reshape %v1287 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1790 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1791 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1792 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1793 = stablehlo.reduce(%v1789 init: %v1790) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1794 = stablehlo.broadcast_in_dim %v1793, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1795 = stablehlo.divide %v1794, %v1791 : tensor<32x960x7x7xf32>
    %v1796 = stablehlo.subtract %v1789, %v1795 : tensor<32x960x7x7xf32>
    %v1797 = stablehlo.multiply %v1796, %v1796 : tensor<32x960x7x7xf32>
    %v1798 = stablehlo.reduce(%v1797 init: %v1790) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1799 = stablehlo.broadcast_in_dim %v1798, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1800 = stablehlo.divide %v1799, %v1791 : tensor<32x960x7x7xf32>
    %v1801 = stablehlo.add %v1800, %v1792 : tensor<32x960x7x7xf32>
    %v1802 = stablehlo.rsqrt %v1801 : tensor<32x960x7x7xf32>
    %v1803 = stablehlo.multiply %v1796, %v1802 : tensor<32x960x7x7xf32>
    %v1804 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1805 = stablehlo.reshape %v1788 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1806 = stablehlo.multiply %v1804, %v1805 : tensor<32x960x7x7xf32>
    %v1807 = stablehlo.reduce(%v1806 init: %v1790) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1808 = stablehlo.broadcast_in_dim %v1807, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1809 = stablehlo.multiply %v1803, %v1806 : tensor<32x960x7x7xf32>
    %v1810 = stablehlo.reduce(%v1809 init: %v1790) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1811 = stablehlo.broadcast_in_dim %v1810, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1812 = stablehlo.multiply %v1806, %v1791 : tensor<32x960x7x7xf32>
    %v1813 = stablehlo.subtract %v1812, %v1808 : tensor<32x960x7x7xf32>
    %v1814 = stablehlo.multiply %v1803, %v1811 : tensor<32x960x7x7xf32>
    %v1815 = stablehlo.subtract %v1813, %v1814 : tensor<32x960x7x7xf32>
    %v1816 = stablehlo.divide %v1802, %v1791 : tensor<32x960x7x7xf32>
    %v1817 = stablehlo.multiply %v1816, %v1815 : tensor<32x960x7x7xf32>
    %v1818 = stablehlo.reshape %v1817 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1819 = stablehlo.reshape %v1818 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1820 = stablehlo.reverse %b16dW, dims = [2, 3] : tensor<960x1x3x3xf32>
    %v1821 = stablehlo.convolution(%v1819, %v1820)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1822 = stablehlo.reshape %v1821 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1823 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1824 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1825 = stablehlo.compare GT, %v1278, %v1823 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1826 = stablehlo.compare LT, %v1278, %v1824 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1827 = stablehlo.and %v1825, %v1826 : tensor<32x47040xi1>
    %v1828 = stablehlo.select %v1827, %v1822, %v1823 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v1829 = stablehlo.reshape %v1258 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1831 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1832 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1833 = stablehlo.reduce(%v1829 init: %v1830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1834 = stablehlo.broadcast_in_dim %v1833, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1835 = stablehlo.divide %v1834, %v1831 : tensor<32x960x7x7xf32>
    %v1836 = stablehlo.subtract %v1829, %v1835 : tensor<32x960x7x7xf32>
    %v1837 = stablehlo.multiply %v1836, %v1836 : tensor<32x960x7x7xf32>
    %v1838 = stablehlo.reduce(%v1837 init: %v1830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1839 = stablehlo.broadcast_in_dim %v1838, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1840 = stablehlo.divide %v1839, %v1831 : tensor<32x960x7x7xf32>
    %v1841 = stablehlo.add %v1840, %v1832 : tensor<32x960x7x7xf32>
    %v1842 = stablehlo.rsqrt %v1841 : tensor<32x960x7x7xf32>
    %v1843 = stablehlo.multiply %v1836, %v1842 : tensor<32x960x7x7xf32>
    %v1844 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1845 = stablehlo.reshape %v1828 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1846 = stablehlo.multiply %v1844, %v1845 : tensor<32x960x7x7xf32>
    %v1847 = stablehlo.reduce(%v1846 init: %v1830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1848 = stablehlo.broadcast_in_dim %v1847, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1849 = stablehlo.multiply %v1843, %v1846 : tensor<32x960x7x7xf32>
    %v1850 = stablehlo.reduce(%v1849 init: %v1830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1851 = stablehlo.broadcast_in_dim %v1850, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1852 = stablehlo.multiply %v1846, %v1831 : tensor<32x960x7x7xf32>
    %v1853 = stablehlo.subtract %v1852, %v1848 : tensor<32x960x7x7xf32>
    %v1854 = stablehlo.multiply %v1843, %v1851 : tensor<32x960x7x7xf32>
    %v1855 = stablehlo.subtract %v1853, %v1854 : tensor<32x960x7x7xf32>
    %v1856 = stablehlo.divide %v1842, %v1831 : tensor<32x960x7x7xf32>
    %v1857 = stablehlo.multiply %v1856, %v1855 : tensor<32x960x7x7xf32>
    %v1858 = stablehlo.reshape %v1857 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1859 = stablehlo.reshape %v1858 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1860 = stablehlo.reverse %b16eW, dims = [2, 3] : tensor<960x160x1x1xf32>
    %v1861 = stablehlo.transpose %v1860, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v1862 = stablehlo.convolution(%v1859, %v1861)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1863 = stablehlo.reshape %v1862 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1864 = stablehlo.add %v1863, %v1666 : tensor<32x7840xf32>
    %v1865 = stablehlo.reshape %v1253 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1866 = stablehlo.reshape %v1858 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1867 = stablehlo.transpose %v1865, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v1868 = stablehlo.transpose %v1866, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1869 = stablehlo.convolution(%v1867, %v1868)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<160x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<160x960x1x1xf32>
    %v1870 = stablehlo.transpose %v1869, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v1871 = stablehlo.reshape %v1258 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1872 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1873 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1874 = stablehlo.reduce(%v1871 init: %v1872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1875 = stablehlo.broadcast_in_dim %v1874, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1876 = stablehlo.divide %v1875, %v1873 : tensor<32x960x7x7xf32>
    %v1877 = stablehlo.subtract %v1871, %v1876 : tensor<32x960x7x7xf32>
    %v1878 = stablehlo.multiply %v1877, %v1877 : tensor<32x960x7x7xf32>
    %v1879 = stablehlo.reduce(%v1878 init: %v1872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1880 = stablehlo.broadcast_in_dim %v1879, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1881 = stablehlo.divide %v1880, %v1873 : tensor<32x960x7x7xf32>
    %v1882 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1883 = stablehlo.add %v1881, %v1882 : tensor<32x960x7x7xf32>
    %v1884 = stablehlo.rsqrt %v1883 : tensor<32x960x7x7xf32>
    %v1885 = stablehlo.multiply %v1877, %v1884 : tensor<32x960x7x7xf32>
    %v1886 = stablehlo.reshape %v1828 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1887 = stablehlo.multiply %v1886, %v1885 : tensor<32x960x7x7xf32>
    %v1888 = stablehlo.reduce(%v1887 init: %v1872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1889 = stablehlo.reshape %v1828 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1891 = stablehlo.reduce(%v1889 init: %v1890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1892 = stablehlo.reshape %v1282 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1893 = stablehlo.reshape %v1818 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1894 = stablehlo.transpose %v1892, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1895 = stablehlo.transpose %v1893, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1896 = stablehlo.convolution(%v1894, %v1895)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 960 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<1x960x3x3xf32>
    %v1897 = stablehlo.reshape %v1896 : (tensor<1x960x3x3xf32>) -> tensor<960x1x3x3xf32>
    %v1898 = stablehlo.reshape %v1287 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1899 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1900 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1901 = stablehlo.reduce(%v1898 init: %v1899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1902 = stablehlo.broadcast_in_dim %v1901, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1903 = stablehlo.divide %v1902, %v1900 : tensor<32x960x7x7xf32>
    %v1904 = stablehlo.subtract %v1898, %v1903 : tensor<32x960x7x7xf32>
    %v1905 = stablehlo.multiply %v1904, %v1904 : tensor<32x960x7x7xf32>
    %v1906 = stablehlo.reduce(%v1905 init: %v1899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1907 = stablehlo.broadcast_in_dim %v1906, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1908 = stablehlo.divide %v1907, %v1900 : tensor<32x960x7x7xf32>
    %v1909 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1910 = stablehlo.add %v1908, %v1909 : tensor<32x960x7x7xf32>
    %v1911 = stablehlo.rsqrt %v1910 : tensor<32x960x7x7xf32>
    %v1912 = stablehlo.multiply %v1904, %v1911 : tensor<32x960x7x7xf32>
    %v1913 = stablehlo.reshape %v1788 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1914 = stablehlo.multiply %v1913, %v1912 : tensor<32x960x7x7xf32>
    %v1915 = stablehlo.reduce(%v1914 init: %v1899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1916 = stablehlo.reshape %v1788 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1918 = stablehlo.reduce(%v1916 init: %v1917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1919 = stablehlo.reshape %v1311 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1920 = stablehlo.reshape %v1777 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1921 = stablehlo.transpose %v1919, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1922 = stablehlo.transpose %v1920, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v1923 = stablehlo.convolution(%v1921, %v1922)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<960x160x1x1xf32>
    %v1924 = stablehlo.transpose %v1923, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v1925 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1927 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v1928 = stablehlo.reduce(%v1925 init: %v1926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1929 = stablehlo.broadcast_in_dim %v1928, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1930 = stablehlo.divide %v1929, %v1927 : tensor<32x160x7x7xf32>
    %v1931 = stablehlo.subtract %v1925, %v1930 : tensor<32x160x7x7xf32>
    %v1932 = stablehlo.multiply %v1931, %v1931 : tensor<32x160x7x7xf32>
    %v1933 = stablehlo.reduce(%v1932 init: %v1926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1934 = stablehlo.broadcast_in_dim %v1933, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1935 = stablehlo.divide %v1934, %v1927 : tensor<32x160x7x7xf32>
    %v1936 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1937 = stablehlo.add %v1935, %v1936 : tensor<32x160x7x7xf32>
    %v1938 = stablehlo.rsqrt %v1937 : tensor<32x160x7x7xf32>
    %v1939 = stablehlo.multiply %v1931, %v1938 : tensor<32x160x7x7xf32>
    %v1940 = stablehlo.reshape %v1666 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1941 = stablehlo.multiply %v1940, %v1939 : tensor<32x160x7x7xf32>
    %v1942 = stablehlo.reduce(%v1941 init: %v1926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1943 = stablehlo.reshape %v1666 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1944 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1945 = stablehlo.reduce(%v1943 init: %v1944) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1946 = stablehlo.reshape %v1232 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1948 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v1949 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1950 = stablehlo.reduce(%v1946 init: %v1947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1951 = stablehlo.broadcast_in_dim %v1950, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1952 = stablehlo.divide %v1951, %v1948 : tensor<32x160x7x7xf32>
    %v1953 = stablehlo.subtract %v1946, %v1952 : tensor<32x160x7x7xf32>
    %v1954 = stablehlo.multiply %v1953, %v1953 : tensor<32x160x7x7xf32>
    %v1955 = stablehlo.reduce(%v1954 init: %v1947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1956 = stablehlo.broadcast_in_dim %v1955, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1957 = stablehlo.divide %v1956, %v1948 : tensor<32x160x7x7xf32>
    %v1958 = stablehlo.add %v1957, %v1949 : tensor<32x160x7x7xf32>
    %v1959 = stablehlo.rsqrt %v1958 : tensor<32x160x7x7xf32>
    %v1960 = stablehlo.multiply %v1953, %v1959 : tensor<32x160x7x7xf32>
    %v1961 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1962 = stablehlo.reshape %v1864 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1963 = stablehlo.multiply %v1961, %v1962 : tensor<32x160x7x7xf32>
    %v1964 = stablehlo.reduce(%v1963 init: %v1947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1965 = stablehlo.broadcast_in_dim %v1964, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1966 = stablehlo.multiply %v1960, %v1963 : tensor<32x160x7x7xf32>
    %v1967 = stablehlo.reduce(%v1966 init: %v1947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1968 = stablehlo.broadcast_in_dim %v1967, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1969 = stablehlo.multiply %v1963, %v1948 : tensor<32x160x7x7xf32>
    %v1970 = stablehlo.subtract %v1969, %v1965 : tensor<32x160x7x7xf32>
    %v1971 = stablehlo.multiply %v1960, %v1968 : tensor<32x160x7x7xf32>
    %v1972 = stablehlo.subtract %v1970, %v1971 : tensor<32x160x7x7xf32>
    %v1973 = stablehlo.divide %v1959, %v1948 : tensor<32x160x7x7xf32>
    %v1974 = stablehlo.multiply %v1973, %v1972 : tensor<32x160x7x7xf32>
    %v1975 = stablehlo.reshape %v1974 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1976 = stablehlo.reshape %v1975 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1977 = stablehlo.reverse %b15pW, dims = [2, 3] : tensor<160x960x1x1xf32>
    %v1978 = stablehlo.transpose %v1977, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v1979 = stablehlo.convolution(%v1976, %v1978)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1980 = stablehlo.reshape %v1979 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1981 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1982 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1983 = stablehlo.compare GT, %v1223, %v1981 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1984 = stablehlo.compare LT, %v1223, %v1982 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1985 = stablehlo.and %v1983, %v1984 : tensor<32x47040xi1>
    %v1986 = stablehlo.select %v1985, %v1980, %v1981 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v1987 = stablehlo.reshape %v1203 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1988 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1989 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1990 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1991 = stablehlo.reduce(%v1987 init: %v1988) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1992 = stablehlo.broadcast_in_dim %v1991, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1993 = stablehlo.divide %v1992, %v1989 : tensor<32x960x7x7xf32>
    %v1994 = stablehlo.subtract %v1987, %v1993 : tensor<32x960x7x7xf32>
    %v1995 = stablehlo.multiply %v1994, %v1994 : tensor<32x960x7x7xf32>
    %v1996 = stablehlo.reduce(%v1995 init: %v1988) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1997 = stablehlo.broadcast_in_dim %v1996, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1998 = stablehlo.divide %v1997, %v1989 : tensor<32x960x7x7xf32>
    %v1999 = stablehlo.add %v1998, %v1990 : tensor<32x960x7x7xf32>
    %v2000 = stablehlo.rsqrt %v1999 : tensor<32x960x7x7xf32>
    %v2001 = stablehlo.multiply %v1994, %v2000 : tensor<32x960x7x7xf32>
    %v2002 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2003 = stablehlo.reshape %v1986 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2004 = stablehlo.multiply %v2002, %v2003 : tensor<32x960x7x7xf32>
    %v2005 = stablehlo.reduce(%v2004 init: %v1988) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2006 = stablehlo.broadcast_in_dim %v2005, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2007 = stablehlo.multiply %v2001, %v2004 : tensor<32x960x7x7xf32>
    %v2008 = stablehlo.reduce(%v2007 init: %v1988) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2009 = stablehlo.broadcast_in_dim %v2008, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2010 = stablehlo.multiply %v2004, %v1989 : tensor<32x960x7x7xf32>
    %v2011 = stablehlo.subtract %v2010, %v2006 : tensor<32x960x7x7xf32>
    %v2012 = stablehlo.multiply %v2001, %v2009 : tensor<32x960x7x7xf32>
    %v2013 = stablehlo.subtract %v2011, %v2012 : tensor<32x960x7x7xf32>
    %v2014 = stablehlo.divide %v2000, %v1989 : tensor<32x960x7x7xf32>
    %v2015 = stablehlo.multiply %v2014, %v2013 : tensor<32x960x7x7xf32>
    %v2016 = stablehlo.reshape %v2015 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2017 = stablehlo.reshape %v2016 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2018 = stablehlo.reverse %b15dW, dims = [2, 3] : tensor<960x1x3x3xf32>
    %v2019 = stablehlo.convolution(%v2017, %v2018)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v2020 = stablehlo.reshape %v2019 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2021 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v2022 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v2023 = stablehlo.compare GT, %v1194, %v2021 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v2024 = stablehlo.compare LT, %v1194, %v2022 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v2025 = stablehlo.and %v2023, %v2024 : tensor<32x47040xi1>
    %v2026 = stablehlo.select %v2025, %v2020, %v2021 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v2027 = stablehlo.reshape %v1174 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2028 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2029 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v2030 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2031 = stablehlo.reduce(%v2027 init: %v2028) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2032 = stablehlo.broadcast_in_dim %v2031, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2033 = stablehlo.divide %v2032, %v2029 : tensor<32x960x7x7xf32>
    %v2034 = stablehlo.subtract %v2027, %v2033 : tensor<32x960x7x7xf32>
    %v2035 = stablehlo.multiply %v2034, %v2034 : tensor<32x960x7x7xf32>
    %v2036 = stablehlo.reduce(%v2035 init: %v2028) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2037 = stablehlo.broadcast_in_dim %v2036, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2038 = stablehlo.divide %v2037, %v2029 : tensor<32x960x7x7xf32>
    %v2039 = stablehlo.add %v2038, %v2030 : tensor<32x960x7x7xf32>
    %v2040 = stablehlo.rsqrt %v2039 : tensor<32x960x7x7xf32>
    %v2041 = stablehlo.multiply %v2034, %v2040 : tensor<32x960x7x7xf32>
    %v2042 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2043 = stablehlo.reshape %v2026 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2044 = stablehlo.multiply %v2042, %v2043 : tensor<32x960x7x7xf32>
    %v2045 = stablehlo.reduce(%v2044 init: %v2028) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2046 = stablehlo.broadcast_in_dim %v2045, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2047 = stablehlo.multiply %v2041, %v2044 : tensor<32x960x7x7xf32>
    %v2048 = stablehlo.reduce(%v2047 init: %v2028) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2049 = stablehlo.broadcast_in_dim %v2048, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2050 = stablehlo.multiply %v2044, %v2029 : tensor<32x960x7x7xf32>
    %v2051 = stablehlo.subtract %v2050, %v2046 : tensor<32x960x7x7xf32>
    %v2052 = stablehlo.multiply %v2041, %v2049 : tensor<32x960x7x7xf32>
    %v2053 = stablehlo.subtract %v2051, %v2052 : tensor<32x960x7x7xf32>
    %v2054 = stablehlo.divide %v2040, %v2029 : tensor<32x960x7x7xf32>
    %v2055 = stablehlo.multiply %v2054, %v2053 : tensor<32x960x7x7xf32>
    %v2056 = stablehlo.reshape %v2055 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2057 = stablehlo.reshape %v2056 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2058 = stablehlo.reverse %b15eW, dims = [2, 3] : tensor<960x160x1x1xf32>
    %v2059 = stablehlo.transpose %v2058, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v2060 = stablehlo.convolution(%v2057, %v2059)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v2061 = stablehlo.reshape %v2060 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2062 = stablehlo.add %v2061, %v1864 : tensor<32x7840xf32>
    %v2063 = stablehlo.reshape %v1169 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2064 = stablehlo.reshape %v2056 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2065 = stablehlo.transpose %v2063, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2066 = stablehlo.transpose %v2064, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2067 = stablehlo.convolution(%v2065, %v2066)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<160x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<160x960x1x1xf32>
    %v2068 = stablehlo.transpose %v2067, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v2069 = stablehlo.reshape %v1174 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2070 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2071 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v2072 = stablehlo.reduce(%v2069 init: %v2070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2073 = stablehlo.broadcast_in_dim %v2072, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2074 = stablehlo.divide %v2073, %v2071 : tensor<32x960x7x7xf32>
    %v2075 = stablehlo.subtract %v2069, %v2074 : tensor<32x960x7x7xf32>
    %v2076 = stablehlo.multiply %v2075, %v2075 : tensor<32x960x7x7xf32>
    %v2077 = stablehlo.reduce(%v2076 init: %v2070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2078 = stablehlo.broadcast_in_dim %v2077, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2079 = stablehlo.divide %v2078, %v2071 : tensor<32x960x7x7xf32>
    %v2080 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2081 = stablehlo.add %v2079, %v2080 : tensor<32x960x7x7xf32>
    %v2082 = stablehlo.rsqrt %v2081 : tensor<32x960x7x7xf32>
    %v2083 = stablehlo.multiply %v2075, %v2082 : tensor<32x960x7x7xf32>
    %v2084 = stablehlo.reshape %v2026 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2085 = stablehlo.multiply %v2084, %v2083 : tensor<32x960x7x7xf32>
    %v2086 = stablehlo.reduce(%v2085 init: %v2070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2087 = stablehlo.reshape %v2026 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2088 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2089 = stablehlo.reduce(%v2087 init: %v2088) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2090 = stablehlo.reshape %v1198 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2091 = stablehlo.reshape %v2016 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2092 = stablehlo.transpose %v2090, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2093 = stablehlo.transpose %v2091, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2094 = stablehlo.convolution(%v2092, %v2093)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 960 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<1x960x3x3xf32>
    %v2095 = stablehlo.reshape %v2094 : (tensor<1x960x3x3xf32>) -> tensor<960x1x3x3xf32>
    %v2096 = stablehlo.reshape %v1203 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2098 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v2099 = stablehlo.reduce(%v2096 init: %v2097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2100 = stablehlo.broadcast_in_dim %v2099, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2101 = stablehlo.divide %v2100, %v2098 : tensor<32x960x7x7xf32>
    %v2102 = stablehlo.subtract %v2096, %v2101 : tensor<32x960x7x7xf32>
    %v2103 = stablehlo.multiply %v2102, %v2102 : tensor<32x960x7x7xf32>
    %v2104 = stablehlo.reduce(%v2103 init: %v2097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2105 = stablehlo.broadcast_in_dim %v2104, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2106 = stablehlo.divide %v2105, %v2098 : tensor<32x960x7x7xf32>
    %v2107 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2108 = stablehlo.add %v2106, %v2107 : tensor<32x960x7x7xf32>
    %v2109 = stablehlo.rsqrt %v2108 : tensor<32x960x7x7xf32>
    %v2110 = stablehlo.multiply %v2102, %v2109 : tensor<32x960x7x7xf32>
    %v2111 = stablehlo.reshape %v1986 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2112 = stablehlo.multiply %v2111, %v2110 : tensor<32x960x7x7xf32>
    %v2113 = stablehlo.reduce(%v2112 init: %v2097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2114 = stablehlo.reshape %v1986 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2116 = stablehlo.reduce(%v2114 init: %v2115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2117 = stablehlo.reshape %v1227 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2118 = stablehlo.reshape %v1975 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2119 = stablehlo.transpose %v2117, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2120 = stablehlo.transpose %v2118, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2121 = stablehlo.convolution(%v2119, %v2120)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<960x160x1x1xf32>
    %v2122 = stablehlo.transpose %v2121, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v2123 = stablehlo.reshape %v1232 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2125 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v2126 = stablehlo.reduce(%v2123 init: %v2124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2127 = stablehlo.broadcast_in_dim %v2126, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2128 = stablehlo.divide %v2127, %v2125 : tensor<32x160x7x7xf32>
    %v2129 = stablehlo.subtract %v2123, %v2128 : tensor<32x160x7x7xf32>
    %v2130 = stablehlo.multiply %v2129, %v2129 : tensor<32x160x7x7xf32>
    %v2131 = stablehlo.reduce(%v2130 init: %v2124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2132 = stablehlo.broadcast_in_dim %v2131, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2133 = stablehlo.divide %v2132, %v2125 : tensor<32x160x7x7xf32>
    %v2134 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2135 = stablehlo.add %v2133, %v2134 : tensor<32x160x7x7xf32>
    %v2136 = stablehlo.rsqrt %v2135 : tensor<32x160x7x7xf32>
    %v2137 = stablehlo.multiply %v2129, %v2136 : tensor<32x160x7x7xf32>
    %v2138 = stablehlo.reshape %v1864 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2139 = stablehlo.multiply %v2138, %v2137 : tensor<32x160x7x7xf32>
    %v2140 = stablehlo.reduce(%v2139 init: %v2124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2141 = stablehlo.reshape %v1864 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2142 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2143 = stablehlo.reduce(%v2141 init: %v2142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2144 = stablehlo.reshape %v1149 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2145 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2146 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v2147 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2148 = stablehlo.reduce(%v2144 init: %v2145) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2149 = stablehlo.broadcast_in_dim %v2148, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2150 = stablehlo.divide %v2149, %v2146 : tensor<32x160x7x7xf32>
    %v2151 = stablehlo.subtract %v2144, %v2150 : tensor<32x160x7x7xf32>
    %v2152 = stablehlo.multiply %v2151, %v2151 : tensor<32x160x7x7xf32>
    %v2153 = stablehlo.reduce(%v2152 init: %v2145) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2154 = stablehlo.broadcast_in_dim %v2153, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2155 = stablehlo.divide %v2154, %v2146 : tensor<32x160x7x7xf32>
    %v2156 = stablehlo.add %v2155, %v2147 : tensor<32x160x7x7xf32>
    %v2157 = stablehlo.rsqrt %v2156 : tensor<32x160x7x7xf32>
    %v2158 = stablehlo.multiply %v2151, %v2157 : tensor<32x160x7x7xf32>
    %v2159 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2160 = stablehlo.reshape %v2062 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2161 = stablehlo.multiply %v2159, %v2160 : tensor<32x160x7x7xf32>
    %v2162 = stablehlo.reduce(%v2161 init: %v2145) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2163 = stablehlo.broadcast_in_dim %v2162, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2164 = stablehlo.multiply %v2158, %v2161 : tensor<32x160x7x7xf32>
    %v2165 = stablehlo.reduce(%v2164 init: %v2145) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2166 = stablehlo.broadcast_in_dim %v2165, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2167 = stablehlo.multiply %v2161, %v2146 : tensor<32x160x7x7xf32>
    %v2168 = stablehlo.subtract %v2167, %v2163 : tensor<32x160x7x7xf32>
    %v2169 = stablehlo.multiply %v2158, %v2166 : tensor<32x160x7x7xf32>
    %v2170 = stablehlo.subtract %v2168, %v2169 : tensor<32x160x7x7xf32>
    %v2171 = stablehlo.divide %v2157, %v2146 : tensor<32x160x7x7xf32>
    %v2172 = stablehlo.multiply %v2171, %v2170 : tensor<32x160x7x7xf32>
    %v2173 = stablehlo.reshape %v2172 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2174 = stablehlo.reshape %v2173 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2175 = stablehlo.reverse %b14pW, dims = [2, 3] : tensor<160x576x1x1xf32>
    %v2176 = stablehlo.transpose %v2175, dims = [1, 0, 2, 3] : (tensor<160x576x1x1xf32>) -> tensor<576x160x1x1xf32>
    %v2177 = stablehlo.convolution(%v2174, %v2176)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<576x160x1x1xf32>) -> tensor<32x576x7x7xf32>
    %v2178 = stablehlo.reshape %v2177 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v2179 = stablehlo.constant dense<0.0> : tensor<32x28224xf32>
    %v2180 = stablehlo.constant dense<6.0> : tensor<32x28224xf32>
    %v2181 = stablehlo.compare GT, %v1140, %v2179 : (tensor<32x28224xf32>, tensor<32x28224xf32>) -> tensor<32x28224xi1>
    %v2182 = stablehlo.compare LT, %v1140, %v2180 : (tensor<32x28224xf32>, tensor<32x28224xf32>) -> tensor<32x28224xi1>
    %v2183 = stablehlo.and %v2181, %v2182 : tensor<32x28224xi1>
    %v2184 = stablehlo.select %v2183, %v2178, %v2179 : tensor<32x28224xi1>, tensor<32x28224xf32>
    %v2185 = stablehlo.reshape %v1120 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2187 = stablehlo.constant dense<1568.0> : tensor<32x576x7x7xf32>
    %v2188 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v2189 = stablehlo.reduce(%v2185 init: %v2186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2190 = stablehlo.broadcast_in_dim %v2189, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2191 = stablehlo.divide %v2190, %v2187 : tensor<32x576x7x7xf32>
    %v2192 = stablehlo.subtract %v2185, %v2191 : tensor<32x576x7x7xf32>
    %v2193 = stablehlo.multiply %v2192, %v2192 : tensor<32x576x7x7xf32>
    %v2194 = stablehlo.reduce(%v2193 init: %v2186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2195 = stablehlo.broadcast_in_dim %v2194, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2196 = stablehlo.divide %v2195, %v2187 : tensor<32x576x7x7xf32>
    %v2197 = stablehlo.add %v2196, %v2188 : tensor<32x576x7x7xf32>
    %v2198 = stablehlo.rsqrt %v2197 : tensor<32x576x7x7xf32>
    %v2199 = stablehlo.multiply %v2192, %v2198 : tensor<32x576x7x7xf32>
    %v2200 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2201 = stablehlo.reshape %v2184 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2202 = stablehlo.multiply %v2200, %v2201 : tensor<32x576x7x7xf32>
    %v2203 = stablehlo.reduce(%v2202 init: %v2186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2204 = stablehlo.broadcast_in_dim %v2203, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2205 = stablehlo.multiply %v2199, %v2202 : tensor<32x576x7x7xf32>
    %v2206 = stablehlo.reduce(%v2205 init: %v2186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2207 = stablehlo.broadcast_in_dim %v2206, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2208 = stablehlo.multiply %v2202, %v2187 : tensor<32x576x7x7xf32>
    %v2209 = stablehlo.subtract %v2208, %v2204 : tensor<32x576x7x7xf32>
    %v2210 = stablehlo.multiply %v2199, %v2207 : tensor<32x576x7x7xf32>
    %v2211 = stablehlo.subtract %v2209, %v2210 : tensor<32x576x7x7xf32>
    %v2212 = stablehlo.divide %v2198, %v2187 : tensor<32x576x7x7xf32>
    %v2213 = stablehlo.multiply %v2212, %v2211 : tensor<32x576x7x7xf32>
    %v2214 = stablehlo.reshape %v2213 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v2215 = stablehlo.reshape %v2214 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2216 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2217 = stablehlo.pad %v2215, %v2216, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576x14x14xf32>
    %v2218 = stablehlo.reverse %b14dW, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2219 = stablehlo.convolution(%v2217, %v2218)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2220 = stablehlo.reshape %v2219 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2221 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v2222 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v2223 = stablehlo.compare GT, %v1111, %v2221 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2224 = stablehlo.compare LT, %v1111, %v2222 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2225 = stablehlo.and %v2223, %v2224 : tensor<32x112896xi1>
    %v2226 = stablehlo.select %v2225, %v2220, %v2221 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v2227 = stablehlo.reshape %v1091 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2228 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2229 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2230 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2231 = stablehlo.reduce(%v2227 init: %v2228) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2232 = stablehlo.broadcast_in_dim %v2231, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2233 = stablehlo.divide %v2232, %v2229 : tensor<32x576x14x14xf32>
    %v2234 = stablehlo.subtract %v2227, %v2233 : tensor<32x576x14x14xf32>
    %v2235 = stablehlo.multiply %v2234, %v2234 : tensor<32x576x14x14xf32>
    %v2236 = stablehlo.reduce(%v2235 init: %v2228) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2237 = stablehlo.broadcast_in_dim %v2236, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2238 = stablehlo.divide %v2237, %v2229 : tensor<32x576x14x14xf32>
    %v2239 = stablehlo.add %v2238, %v2230 : tensor<32x576x14x14xf32>
    %v2240 = stablehlo.rsqrt %v2239 : tensor<32x576x14x14xf32>
    %v2241 = stablehlo.multiply %v2234, %v2240 : tensor<32x576x14x14xf32>
    %v2242 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2243 = stablehlo.reshape %v2226 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2244 = stablehlo.multiply %v2242, %v2243 : tensor<32x576x14x14xf32>
    %v2245 = stablehlo.reduce(%v2244 init: %v2228) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2246 = stablehlo.broadcast_in_dim %v2245, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2247 = stablehlo.multiply %v2241, %v2244 : tensor<32x576x14x14xf32>
    %v2248 = stablehlo.reduce(%v2247 init: %v2228) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2249 = stablehlo.broadcast_in_dim %v2248, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2250 = stablehlo.multiply %v2244, %v2229 : tensor<32x576x14x14xf32>
    %v2251 = stablehlo.subtract %v2250, %v2246 : tensor<32x576x14x14xf32>
    %v2252 = stablehlo.multiply %v2241, %v2249 : tensor<32x576x14x14xf32>
    %v2253 = stablehlo.subtract %v2251, %v2252 : tensor<32x576x14x14xf32>
    %v2254 = stablehlo.divide %v2240, %v2229 : tensor<32x576x14x14xf32>
    %v2255 = stablehlo.multiply %v2254, %v2253 : tensor<32x576x14x14xf32>
    %v2256 = stablehlo.reshape %v2255 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2257 = stablehlo.reshape %v2256 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2258 = stablehlo.reverse %b14eW, dims = [2, 3] : tensor<576x96x1x1xf32>
    %v2259 = stablehlo.transpose %v2258, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2260 = stablehlo.convolution(%v2257, %v2259)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2261 = stablehlo.reshape %v2260 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2262 = stablehlo.reshape %v1086 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2263 = stablehlo.reshape %v2256 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2264 = stablehlo.transpose %v2262, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2265 = stablehlo.transpose %v2263, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2266 = stablehlo.convolution(%v2264, %v2265)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2267 = stablehlo.transpose %v2266, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2268 = stablehlo.reshape %v1091 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2269 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2270 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2271 = stablehlo.reduce(%v2268 init: %v2269) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2272 = stablehlo.broadcast_in_dim %v2271, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2273 = stablehlo.divide %v2272, %v2270 : tensor<32x576x14x14xf32>
    %v2274 = stablehlo.subtract %v2268, %v2273 : tensor<32x576x14x14xf32>
    %v2275 = stablehlo.multiply %v2274, %v2274 : tensor<32x576x14x14xf32>
    %v2276 = stablehlo.reduce(%v2275 init: %v2269) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2277 = stablehlo.broadcast_in_dim %v2276, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2278 = stablehlo.divide %v2277, %v2270 : tensor<32x576x14x14xf32>
    %v2279 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2280 = stablehlo.add %v2278, %v2279 : tensor<32x576x14x14xf32>
    %v2281 = stablehlo.rsqrt %v2280 : tensor<32x576x14x14xf32>
    %v2282 = stablehlo.multiply %v2274, %v2281 : tensor<32x576x14x14xf32>
    %v2283 = stablehlo.reshape %v2226 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2284 = stablehlo.multiply %v2283, %v2282 : tensor<32x576x14x14xf32>
    %v2285 = stablehlo.reduce(%v2284 init: %v2269) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2286 = stablehlo.reshape %v2226 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2287 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2288 = stablehlo.reduce(%v2286 init: %v2287) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2289 = stablehlo.reshape %v1115 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2290 = stablehlo.reshape %v2214 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2291 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2292 = stablehlo.pad %v2290, %v2291, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576x14x14xf32>
    %v2293 = stablehlo.transpose %v2289, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2294 = stablehlo.transpose %v2292, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2295 = stablehlo.convolution(%v2293, %v2294)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2296 = stablehlo.reshape %v2295 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2297 = stablehlo.reshape %v1120 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2299 = stablehlo.constant dense<1568.0> : tensor<32x576x7x7xf32>
    %v2300 = stablehlo.reduce(%v2297 init: %v2298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2301 = stablehlo.broadcast_in_dim %v2300, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2302 = stablehlo.divide %v2301, %v2299 : tensor<32x576x7x7xf32>
    %v2303 = stablehlo.subtract %v2297, %v2302 : tensor<32x576x7x7xf32>
    %v2304 = stablehlo.multiply %v2303, %v2303 : tensor<32x576x7x7xf32>
    %v2305 = stablehlo.reduce(%v2304 init: %v2298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2306 = stablehlo.broadcast_in_dim %v2305, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2307 = stablehlo.divide %v2306, %v2299 : tensor<32x576x7x7xf32>
    %v2308 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v2309 = stablehlo.add %v2307, %v2308 : tensor<32x576x7x7xf32>
    %v2310 = stablehlo.rsqrt %v2309 : tensor<32x576x7x7xf32>
    %v2311 = stablehlo.multiply %v2303, %v2310 : tensor<32x576x7x7xf32>
    %v2312 = stablehlo.reshape %v2184 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2313 = stablehlo.multiply %v2312, %v2311 : tensor<32x576x7x7xf32>
    %v2314 = stablehlo.reduce(%v2313 init: %v2298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2315 = stablehlo.reshape %v2184 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2317 = stablehlo.reduce(%v2315 init: %v2316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2318 = stablehlo.reshape %v1144 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2319 = stablehlo.reshape %v2173 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2320 = stablehlo.transpose %v2318, dims = [1, 0, 2, 3] : (tensor<32x576x7x7xf32>) -> tensor<576x32x7x7xf32>
    %v2321 = stablehlo.transpose %v2319, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2322 = stablehlo.convolution(%v2320, %v2321)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<576x160x1x1xf32>
    %v2323 = stablehlo.transpose %v2322, dims = [1, 0, 2, 3] : (tensor<576x160x1x1xf32>) -> tensor<160x576x1x1xf32>
    %v2324 = stablehlo.reshape %v1149 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2325 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2326 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v2327 = stablehlo.reduce(%v2324 init: %v2325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2328 = stablehlo.broadcast_in_dim %v2327, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2329 = stablehlo.divide %v2328, %v2326 : tensor<32x160x7x7xf32>
    %v2330 = stablehlo.subtract %v2324, %v2329 : tensor<32x160x7x7xf32>
    %v2331 = stablehlo.multiply %v2330, %v2330 : tensor<32x160x7x7xf32>
    %v2332 = stablehlo.reduce(%v2331 init: %v2325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2333 = stablehlo.broadcast_in_dim %v2332, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2334 = stablehlo.divide %v2333, %v2326 : tensor<32x160x7x7xf32>
    %v2335 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2336 = stablehlo.add %v2334, %v2335 : tensor<32x160x7x7xf32>
    %v2337 = stablehlo.rsqrt %v2336 : tensor<32x160x7x7xf32>
    %v2338 = stablehlo.multiply %v2330, %v2337 : tensor<32x160x7x7xf32>
    %v2339 = stablehlo.reshape %v2062 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2340 = stablehlo.multiply %v2339, %v2338 : tensor<32x160x7x7xf32>
    %v2341 = stablehlo.reduce(%v2340 init: %v2325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2342 = stablehlo.reshape %v2062 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2344 = stablehlo.reduce(%v2342 init: %v2343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2345 = stablehlo.reshape %v1065 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2346 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2347 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2348 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2349 = stablehlo.reduce(%v2345 init: %v2346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2350 = stablehlo.broadcast_in_dim %v2349, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2351 = stablehlo.divide %v2350, %v2347 : tensor<32x96x14x14xf32>
    %v2352 = stablehlo.subtract %v2345, %v2351 : tensor<32x96x14x14xf32>
    %v2353 = stablehlo.multiply %v2352, %v2352 : tensor<32x96x14x14xf32>
    %v2354 = stablehlo.reduce(%v2353 init: %v2346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2355 = stablehlo.broadcast_in_dim %v2354, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2356 = stablehlo.divide %v2355, %v2347 : tensor<32x96x14x14xf32>
    %v2357 = stablehlo.add %v2356, %v2348 : tensor<32x96x14x14xf32>
    %v2358 = stablehlo.rsqrt %v2357 : tensor<32x96x14x14xf32>
    %v2359 = stablehlo.multiply %v2352, %v2358 : tensor<32x96x14x14xf32>
    %v2360 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2361 = stablehlo.reshape %v2261 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2362 = stablehlo.multiply %v2360, %v2361 : tensor<32x96x14x14xf32>
    %v2363 = stablehlo.reduce(%v2362 init: %v2346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2364 = stablehlo.broadcast_in_dim %v2363, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2365 = stablehlo.multiply %v2359, %v2362 : tensor<32x96x14x14xf32>
    %v2366 = stablehlo.reduce(%v2365 init: %v2346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2367 = stablehlo.broadcast_in_dim %v2366, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2368 = stablehlo.multiply %v2362, %v2347 : tensor<32x96x14x14xf32>
    %v2369 = stablehlo.subtract %v2368, %v2364 : tensor<32x96x14x14xf32>
    %v2370 = stablehlo.multiply %v2359, %v2367 : tensor<32x96x14x14xf32>
    %v2371 = stablehlo.subtract %v2369, %v2370 : tensor<32x96x14x14xf32>
    %v2372 = stablehlo.divide %v2358, %v2347 : tensor<32x96x14x14xf32>
    %v2373 = stablehlo.multiply %v2372, %v2371 : tensor<32x96x14x14xf32>
    %v2374 = stablehlo.reshape %v2373 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2375 = stablehlo.reshape %v2374 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2376 = stablehlo.reverse %b13pW, dims = [2, 3] : tensor<96x576x1x1xf32>
    %v2377 = stablehlo.transpose %v2376, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2378 = stablehlo.convolution(%v2375, %v2377)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v2379 = stablehlo.reshape %v2378 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2380 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v2381 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v2382 = stablehlo.compare GT, %v1056, %v2380 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2383 = stablehlo.compare LT, %v1056, %v2381 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2384 = stablehlo.and %v2382, %v2383 : tensor<32x112896xi1>
    %v2385 = stablehlo.select %v2384, %v2379, %v2380 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v2386 = stablehlo.reshape %v1036 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2387 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2388 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2389 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2390 = stablehlo.reduce(%v2386 init: %v2387) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2391 = stablehlo.broadcast_in_dim %v2390, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2392 = stablehlo.divide %v2391, %v2388 : tensor<32x576x14x14xf32>
    %v2393 = stablehlo.subtract %v2386, %v2392 : tensor<32x576x14x14xf32>
    %v2394 = stablehlo.multiply %v2393, %v2393 : tensor<32x576x14x14xf32>
    %v2395 = stablehlo.reduce(%v2394 init: %v2387) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2396 = stablehlo.broadcast_in_dim %v2395, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2397 = stablehlo.divide %v2396, %v2388 : tensor<32x576x14x14xf32>
    %v2398 = stablehlo.add %v2397, %v2389 : tensor<32x576x14x14xf32>
    %v2399 = stablehlo.rsqrt %v2398 : tensor<32x576x14x14xf32>
    %v2400 = stablehlo.multiply %v2393, %v2399 : tensor<32x576x14x14xf32>
    %v2401 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2402 = stablehlo.reshape %v2385 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2403 = stablehlo.multiply %v2401, %v2402 : tensor<32x576x14x14xf32>
    %v2404 = stablehlo.reduce(%v2403 init: %v2387) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2405 = stablehlo.broadcast_in_dim %v2404, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2406 = stablehlo.multiply %v2400, %v2403 : tensor<32x576x14x14xf32>
    %v2407 = stablehlo.reduce(%v2406 init: %v2387) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2408 = stablehlo.broadcast_in_dim %v2407, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2409 = stablehlo.multiply %v2403, %v2388 : tensor<32x576x14x14xf32>
    %v2410 = stablehlo.subtract %v2409, %v2405 : tensor<32x576x14x14xf32>
    %v2411 = stablehlo.multiply %v2400, %v2408 : tensor<32x576x14x14xf32>
    %v2412 = stablehlo.subtract %v2410, %v2411 : tensor<32x576x14x14xf32>
    %v2413 = stablehlo.divide %v2399, %v2388 : tensor<32x576x14x14xf32>
    %v2414 = stablehlo.multiply %v2413, %v2412 : tensor<32x576x14x14xf32>
    %v2415 = stablehlo.reshape %v2414 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2416 = stablehlo.reshape %v2415 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2417 = stablehlo.reverse %b13dW, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2418 = stablehlo.convolution(%v2416, %v2417)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2419 = stablehlo.reshape %v2418 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2420 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v2421 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v2422 = stablehlo.compare GT, %v1027, %v2420 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2423 = stablehlo.compare LT, %v1027, %v2421 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2424 = stablehlo.and %v2422, %v2423 : tensor<32x112896xi1>
    %v2425 = stablehlo.select %v2424, %v2419, %v2420 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v2426 = stablehlo.reshape %v1007 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2427 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2428 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2429 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2430 = stablehlo.reduce(%v2426 init: %v2427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2431 = stablehlo.broadcast_in_dim %v2430, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2432 = stablehlo.divide %v2431, %v2428 : tensor<32x576x14x14xf32>
    %v2433 = stablehlo.subtract %v2426, %v2432 : tensor<32x576x14x14xf32>
    %v2434 = stablehlo.multiply %v2433, %v2433 : tensor<32x576x14x14xf32>
    %v2435 = stablehlo.reduce(%v2434 init: %v2427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2436 = stablehlo.broadcast_in_dim %v2435, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2437 = stablehlo.divide %v2436, %v2428 : tensor<32x576x14x14xf32>
    %v2438 = stablehlo.add %v2437, %v2429 : tensor<32x576x14x14xf32>
    %v2439 = stablehlo.rsqrt %v2438 : tensor<32x576x14x14xf32>
    %v2440 = stablehlo.multiply %v2433, %v2439 : tensor<32x576x14x14xf32>
    %v2441 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2442 = stablehlo.reshape %v2425 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2443 = stablehlo.multiply %v2441, %v2442 : tensor<32x576x14x14xf32>
    %v2444 = stablehlo.reduce(%v2443 init: %v2427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2445 = stablehlo.broadcast_in_dim %v2444, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2446 = stablehlo.multiply %v2440, %v2443 : tensor<32x576x14x14xf32>
    %v2447 = stablehlo.reduce(%v2446 init: %v2427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2448 = stablehlo.broadcast_in_dim %v2447, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2449 = stablehlo.multiply %v2443, %v2428 : tensor<32x576x14x14xf32>
    %v2450 = stablehlo.subtract %v2449, %v2445 : tensor<32x576x14x14xf32>
    %v2451 = stablehlo.multiply %v2440, %v2448 : tensor<32x576x14x14xf32>
    %v2452 = stablehlo.subtract %v2450, %v2451 : tensor<32x576x14x14xf32>
    %v2453 = stablehlo.divide %v2439, %v2428 : tensor<32x576x14x14xf32>
    %v2454 = stablehlo.multiply %v2453, %v2452 : tensor<32x576x14x14xf32>
    %v2455 = stablehlo.reshape %v2454 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2456 = stablehlo.reshape %v2455 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2457 = stablehlo.reverse %b13eW, dims = [2, 3] : tensor<576x96x1x1xf32>
    %v2458 = stablehlo.transpose %v2457, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2459 = stablehlo.convolution(%v2456, %v2458)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2460 = stablehlo.reshape %v2459 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2461 = stablehlo.add %v2460, %v2261 : tensor<32x18816xf32>
    %v2462 = stablehlo.reshape %v1002 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2463 = stablehlo.reshape %v2455 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2464 = stablehlo.transpose %v2462, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2465 = stablehlo.transpose %v2463, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2466 = stablehlo.convolution(%v2464, %v2465)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2467 = stablehlo.transpose %v2466, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2468 = stablehlo.reshape %v1007 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2470 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2471 = stablehlo.reduce(%v2468 init: %v2469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2472 = stablehlo.broadcast_in_dim %v2471, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2473 = stablehlo.divide %v2472, %v2470 : tensor<32x576x14x14xf32>
    %v2474 = stablehlo.subtract %v2468, %v2473 : tensor<32x576x14x14xf32>
    %v2475 = stablehlo.multiply %v2474, %v2474 : tensor<32x576x14x14xf32>
    %v2476 = stablehlo.reduce(%v2475 init: %v2469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2477 = stablehlo.broadcast_in_dim %v2476, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2478 = stablehlo.divide %v2477, %v2470 : tensor<32x576x14x14xf32>
    %v2479 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2480 = stablehlo.add %v2478, %v2479 : tensor<32x576x14x14xf32>
    %v2481 = stablehlo.rsqrt %v2480 : tensor<32x576x14x14xf32>
    %v2482 = stablehlo.multiply %v2474, %v2481 : tensor<32x576x14x14xf32>
    %v2483 = stablehlo.reshape %v2425 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2484 = stablehlo.multiply %v2483, %v2482 : tensor<32x576x14x14xf32>
    %v2485 = stablehlo.reduce(%v2484 init: %v2469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2486 = stablehlo.reshape %v2425 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2487 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2488 = stablehlo.reduce(%v2486 init: %v2487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2489 = stablehlo.reshape %v1031 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2490 = stablehlo.reshape %v2415 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2491 = stablehlo.transpose %v2489, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2492 = stablehlo.transpose %v2490, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2493 = stablehlo.convolution(%v2491, %v2492)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2494 = stablehlo.reshape %v2493 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2495 = stablehlo.reshape %v1036 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2496 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2497 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2498 = stablehlo.reduce(%v2495 init: %v2496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2499 = stablehlo.broadcast_in_dim %v2498, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2500 = stablehlo.divide %v2499, %v2497 : tensor<32x576x14x14xf32>
    %v2501 = stablehlo.subtract %v2495, %v2500 : tensor<32x576x14x14xf32>
    %v2502 = stablehlo.multiply %v2501, %v2501 : tensor<32x576x14x14xf32>
    %v2503 = stablehlo.reduce(%v2502 init: %v2496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2504 = stablehlo.broadcast_in_dim %v2503, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2505 = stablehlo.divide %v2504, %v2497 : tensor<32x576x14x14xf32>
    %v2506 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2507 = stablehlo.add %v2505, %v2506 : tensor<32x576x14x14xf32>
    %v2508 = stablehlo.rsqrt %v2507 : tensor<32x576x14x14xf32>
    %v2509 = stablehlo.multiply %v2501, %v2508 : tensor<32x576x14x14xf32>
    %v2510 = stablehlo.reshape %v2385 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2511 = stablehlo.multiply %v2510, %v2509 : tensor<32x576x14x14xf32>
    %v2512 = stablehlo.reduce(%v2511 init: %v2496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2513 = stablehlo.reshape %v2385 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2515 = stablehlo.reduce(%v2513 init: %v2514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2516 = stablehlo.reshape %v1060 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2517 = stablehlo.reshape %v2374 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2518 = stablehlo.transpose %v2516, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2519 = stablehlo.transpose %v2517, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2520 = stablehlo.convolution(%v2518, %v2519)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<576x96x1x1xf32>
    %v2521 = stablehlo.transpose %v2520, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2522 = stablehlo.reshape %v1065 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2523 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2524 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2525 = stablehlo.reduce(%v2522 init: %v2523) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2526 = stablehlo.broadcast_in_dim %v2525, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2527 = stablehlo.divide %v2526, %v2524 : tensor<32x96x14x14xf32>
    %v2528 = stablehlo.subtract %v2522, %v2527 : tensor<32x96x14x14xf32>
    %v2529 = stablehlo.multiply %v2528, %v2528 : tensor<32x96x14x14xf32>
    %v2530 = stablehlo.reduce(%v2529 init: %v2523) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2531 = stablehlo.broadcast_in_dim %v2530, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2532 = stablehlo.divide %v2531, %v2524 : tensor<32x96x14x14xf32>
    %v2533 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2534 = stablehlo.add %v2532, %v2533 : tensor<32x96x14x14xf32>
    %v2535 = stablehlo.rsqrt %v2534 : tensor<32x96x14x14xf32>
    %v2536 = stablehlo.multiply %v2528, %v2535 : tensor<32x96x14x14xf32>
    %v2537 = stablehlo.reshape %v2261 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2538 = stablehlo.multiply %v2537, %v2536 : tensor<32x96x14x14xf32>
    %v2539 = stablehlo.reduce(%v2538 init: %v2523) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2540 = stablehlo.reshape %v2261 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2541 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2542 = stablehlo.reduce(%v2540 init: %v2541) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2543 = stablehlo.reshape %v981 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2544 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2545 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2546 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2547 = stablehlo.reduce(%v2543 init: %v2544) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2548 = stablehlo.broadcast_in_dim %v2547, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2549 = stablehlo.divide %v2548, %v2545 : tensor<32x96x14x14xf32>
    %v2550 = stablehlo.subtract %v2543, %v2549 : tensor<32x96x14x14xf32>
    %v2551 = stablehlo.multiply %v2550, %v2550 : tensor<32x96x14x14xf32>
    %v2552 = stablehlo.reduce(%v2551 init: %v2544) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2553 = stablehlo.broadcast_in_dim %v2552, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2554 = stablehlo.divide %v2553, %v2545 : tensor<32x96x14x14xf32>
    %v2555 = stablehlo.add %v2554, %v2546 : tensor<32x96x14x14xf32>
    %v2556 = stablehlo.rsqrt %v2555 : tensor<32x96x14x14xf32>
    %v2557 = stablehlo.multiply %v2550, %v2556 : tensor<32x96x14x14xf32>
    %v2558 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2559 = stablehlo.reshape %v2461 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2560 = stablehlo.multiply %v2558, %v2559 : tensor<32x96x14x14xf32>
    %v2561 = stablehlo.reduce(%v2560 init: %v2544) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2562 = stablehlo.broadcast_in_dim %v2561, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2563 = stablehlo.multiply %v2557, %v2560 : tensor<32x96x14x14xf32>
    %v2564 = stablehlo.reduce(%v2563 init: %v2544) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2565 = stablehlo.broadcast_in_dim %v2564, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2566 = stablehlo.multiply %v2560, %v2545 : tensor<32x96x14x14xf32>
    %v2567 = stablehlo.subtract %v2566, %v2562 : tensor<32x96x14x14xf32>
    %v2568 = stablehlo.multiply %v2557, %v2565 : tensor<32x96x14x14xf32>
    %v2569 = stablehlo.subtract %v2567, %v2568 : tensor<32x96x14x14xf32>
    %v2570 = stablehlo.divide %v2556, %v2545 : tensor<32x96x14x14xf32>
    %v2571 = stablehlo.multiply %v2570, %v2569 : tensor<32x96x14x14xf32>
    %v2572 = stablehlo.reshape %v2571 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2573 = stablehlo.reshape %v2572 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2574 = stablehlo.reverse %b12pW, dims = [2, 3] : tensor<96x576x1x1xf32>
    %v2575 = stablehlo.transpose %v2574, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2576 = stablehlo.convolution(%v2573, %v2575)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v2577 = stablehlo.reshape %v2576 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2578 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v2579 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v2580 = stablehlo.compare GT, %v972, %v2578 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2581 = stablehlo.compare LT, %v972, %v2579 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2582 = stablehlo.and %v2580, %v2581 : tensor<32x112896xi1>
    %v2583 = stablehlo.select %v2582, %v2577, %v2578 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v2584 = stablehlo.reshape %v952 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2586 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2587 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2588 = stablehlo.reduce(%v2584 init: %v2585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2589 = stablehlo.broadcast_in_dim %v2588, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2590 = stablehlo.divide %v2589, %v2586 : tensor<32x576x14x14xf32>
    %v2591 = stablehlo.subtract %v2584, %v2590 : tensor<32x576x14x14xf32>
    %v2592 = stablehlo.multiply %v2591, %v2591 : tensor<32x576x14x14xf32>
    %v2593 = stablehlo.reduce(%v2592 init: %v2585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2594 = stablehlo.broadcast_in_dim %v2593, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2595 = stablehlo.divide %v2594, %v2586 : tensor<32x576x14x14xf32>
    %v2596 = stablehlo.add %v2595, %v2587 : tensor<32x576x14x14xf32>
    %v2597 = stablehlo.rsqrt %v2596 : tensor<32x576x14x14xf32>
    %v2598 = stablehlo.multiply %v2591, %v2597 : tensor<32x576x14x14xf32>
    %v2599 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2600 = stablehlo.reshape %v2583 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2601 = stablehlo.multiply %v2599, %v2600 : tensor<32x576x14x14xf32>
    %v2602 = stablehlo.reduce(%v2601 init: %v2585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2603 = stablehlo.broadcast_in_dim %v2602, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2604 = stablehlo.multiply %v2598, %v2601 : tensor<32x576x14x14xf32>
    %v2605 = stablehlo.reduce(%v2604 init: %v2585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2606 = stablehlo.broadcast_in_dim %v2605, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2607 = stablehlo.multiply %v2601, %v2586 : tensor<32x576x14x14xf32>
    %v2608 = stablehlo.subtract %v2607, %v2603 : tensor<32x576x14x14xf32>
    %v2609 = stablehlo.multiply %v2598, %v2606 : tensor<32x576x14x14xf32>
    %v2610 = stablehlo.subtract %v2608, %v2609 : tensor<32x576x14x14xf32>
    %v2611 = stablehlo.divide %v2597, %v2586 : tensor<32x576x14x14xf32>
    %v2612 = stablehlo.multiply %v2611, %v2610 : tensor<32x576x14x14xf32>
    %v2613 = stablehlo.reshape %v2612 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2614 = stablehlo.reshape %v2613 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2615 = stablehlo.reverse %b12dW, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2616 = stablehlo.convolution(%v2614, %v2615)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2617 = stablehlo.reshape %v2616 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2618 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v2619 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v2620 = stablehlo.compare GT, %v943, %v2618 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2621 = stablehlo.compare LT, %v943, %v2619 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2622 = stablehlo.and %v2620, %v2621 : tensor<32x112896xi1>
    %v2623 = stablehlo.select %v2622, %v2617, %v2618 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v2624 = stablehlo.reshape %v923 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2625 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2626 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2627 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2628 = stablehlo.reduce(%v2624 init: %v2625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2629 = stablehlo.broadcast_in_dim %v2628, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2630 = stablehlo.divide %v2629, %v2626 : tensor<32x576x14x14xf32>
    %v2631 = stablehlo.subtract %v2624, %v2630 : tensor<32x576x14x14xf32>
    %v2632 = stablehlo.multiply %v2631, %v2631 : tensor<32x576x14x14xf32>
    %v2633 = stablehlo.reduce(%v2632 init: %v2625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2634 = stablehlo.broadcast_in_dim %v2633, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2635 = stablehlo.divide %v2634, %v2626 : tensor<32x576x14x14xf32>
    %v2636 = stablehlo.add %v2635, %v2627 : tensor<32x576x14x14xf32>
    %v2637 = stablehlo.rsqrt %v2636 : tensor<32x576x14x14xf32>
    %v2638 = stablehlo.multiply %v2631, %v2637 : tensor<32x576x14x14xf32>
    %v2639 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2640 = stablehlo.reshape %v2623 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2641 = stablehlo.multiply %v2639, %v2640 : tensor<32x576x14x14xf32>
    %v2642 = stablehlo.reduce(%v2641 init: %v2625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2643 = stablehlo.broadcast_in_dim %v2642, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2644 = stablehlo.multiply %v2638, %v2641 : tensor<32x576x14x14xf32>
    %v2645 = stablehlo.reduce(%v2644 init: %v2625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2646 = stablehlo.broadcast_in_dim %v2645, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2647 = stablehlo.multiply %v2641, %v2626 : tensor<32x576x14x14xf32>
    %v2648 = stablehlo.subtract %v2647, %v2643 : tensor<32x576x14x14xf32>
    %v2649 = stablehlo.multiply %v2638, %v2646 : tensor<32x576x14x14xf32>
    %v2650 = stablehlo.subtract %v2648, %v2649 : tensor<32x576x14x14xf32>
    %v2651 = stablehlo.divide %v2637, %v2626 : tensor<32x576x14x14xf32>
    %v2652 = stablehlo.multiply %v2651, %v2650 : tensor<32x576x14x14xf32>
    %v2653 = stablehlo.reshape %v2652 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2654 = stablehlo.reshape %v2653 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2655 = stablehlo.reverse %b12eW, dims = [2, 3] : tensor<576x96x1x1xf32>
    %v2656 = stablehlo.transpose %v2655, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2657 = stablehlo.convolution(%v2654, %v2656)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2658 = stablehlo.reshape %v2657 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2659 = stablehlo.add %v2658, %v2461 : tensor<32x18816xf32>
    %v2660 = stablehlo.reshape %v918 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2661 = stablehlo.reshape %v2653 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2662 = stablehlo.transpose %v2660, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2663 = stablehlo.transpose %v2661, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2664 = stablehlo.convolution(%v2662, %v2663)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2665 = stablehlo.transpose %v2664, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2666 = stablehlo.reshape %v923 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2667 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2668 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2669 = stablehlo.reduce(%v2666 init: %v2667) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2670 = stablehlo.broadcast_in_dim %v2669, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2671 = stablehlo.divide %v2670, %v2668 : tensor<32x576x14x14xf32>
    %v2672 = stablehlo.subtract %v2666, %v2671 : tensor<32x576x14x14xf32>
    %v2673 = stablehlo.multiply %v2672, %v2672 : tensor<32x576x14x14xf32>
    %v2674 = stablehlo.reduce(%v2673 init: %v2667) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2675 = stablehlo.broadcast_in_dim %v2674, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2676 = stablehlo.divide %v2675, %v2668 : tensor<32x576x14x14xf32>
    %v2677 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2678 = stablehlo.add %v2676, %v2677 : tensor<32x576x14x14xf32>
    %v2679 = stablehlo.rsqrt %v2678 : tensor<32x576x14x14xf32>
    %v2680 = stablehlo.multiply %v2672, %v2679 : tensor<32x576x14x14xf32>
    %v2681 = stablehlo.reshape %v2623 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2682 = stablehlo.multiply %v2681, %v2680 : tensor<32x576x14x14xf32>
    %v2683 = stablehlo.reduce(%v2682 init: %v2667) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2684 = stablehlo.reshape %v2623 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2685 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2686 = stablehlo.reduce(%v2684 init: %v2685) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2687 = stablehlo.reshape %v947 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2688 = stablehlo.reshape %v2613 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2689 = stablehlo.transpose %v2687, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2690 = stablehlo.transpose %v2688, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2691 = stablehlo.convolution(%v2689, %v2690)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2692 = stablehlo.reshape %v2691 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2693 = stablehlo.reshape %v952 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2695 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2696 = stablehlo.reduce(%v2693 init: %v2694) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2697 = stablehlo.broadcast_in_dim %v2696, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2698 = stablehlo.divide %v2697, %v2695 : tensor<32x576x14x14xf32>
    %v2699 = stablehlo.subtract %v2693, %v2698 : tensor<32x576x14x14xf32>
    %v2700 = stablehlo.multiply %v2699, %v2699 : tensor<32x576x14x14xf32>
    %v2701 = stablehlo.reduce(%v2700 init: %v2694) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2702 = stablehlo.broadcast_in_dim %v2701, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2703 = stablehlo.divide %v2702, %v2695 : tensor<32x576x14x14xf32>
    %v2704 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2705 = stablehlo.add %v2703, %v2704 : tensor<32x576x14x14xf32>
    %v2706 = stablehlo.rsqrt %v2705 : tensor<32x576x14x14xf32>
    %v2707 = stablehlo.multiply %v2699, %v2706 : tensor<32x576x14x14xf32>
    %v2708 = stablehlo.reshape %v2583 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2709 = stablehlo.multiply %v2708, %v2707 : tensor<32x576x14x14xf32>
    %v2710 = stablehlo.reduce(%v2709 init: %v2694) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2711 = stablehlo.reshape %v2583 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2712 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2713 = stablehlo.reduce(%v2711 init: %v2712) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2714 = stablehlo.reshape %v976 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2715 = stablehlo.reshape %v2572 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2716 = stablehlo.transpose %v2714, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2717 = stablehlo.transpose %v2715, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2718 = stablehlo.convolution(%v2716, %v2717)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<576x96x1x1xf32>
    %v2719 = stablehlo.transpose %v2718, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2720 = stablehlo.reshape %v981 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2721 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2722 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2723 = stablehlo.reduce(%v2720 init: %v2721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2724 = stablehlo.broadcast_in_dim %v2723, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2725 = stablehlo.divide %v2724, %v2722 : tensor<32x96x14x14xf32>
    %v2726 = stablehlo.subtract %v2720, %v2725 : tensor<32x96x14x14xf32>
    %v2727 = stablehlo.multiply %v2726, %v2726 : tensor<32x96x14x14xf32>
    %v2728 = stablehlo.reduce(%v2727 init: %v2721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2729 = stablehlo.broadcast_in_dim %v2728, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2730 = stablehlo.divide %v2729, %v2722 : tensor<32x96x14x14xf32>
    %v2731 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2732 = stablehlo.add %v2730, %v2731 : tensor<32x96x14x14xf32>
    %v2733 = stablehlo.rsqrt %v2732 : tensor<32x96x14x14xf32>
    %v2734 = stablehlo.multiply %v2726, %v2733 : tensor<32x96x14x14xf32>
    %v2735 = stablehlo.reshape %v2461 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2736 = stablehlo.multiply %v2735, %v2734 : tensor<32x96x14x14xf32>
    %v2737 = stablehlo.reduce(%v2736 init: %v2721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2738 = stablehlo.reshape %v2461 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2740 = stablehlo.reduce(%v2738 init: %v2739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2741 = stablehlo.reshape %v898 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2743 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2744 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2745 = stablehlo.reduce(%v2741 init: %v2742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2746 = stablehlo.broadcast_in_dim %v2745, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2747 = stablehlo.divide %v2746, %v2743 : tensor<32x96x14x14xf32>
    %v2748 = stablehlo.subtract %v2741, %v2747 : tensor<32x96x14x14xf32>
    %v2749 = stablehlo.multiply %v2748, %v2748 : tensor<32x96x14x14xf32>
    %v2750 = stablehlo.reduce(%v2749 init: %v2742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2751 = stablehlo.broadcast_in_dim %v2750, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2752 = stablehlo.divide %v2751, %v2743 : tensor<32x96x14x14xf32>
    %v2753 = stablehlo.add %v2752, %v2744 : tensor<32x96x14x14xf32>
    %v2754 = stablehlo.rsqrt %v2753 : tensor<32x96x14x14xf32>
    %v2755 = stablehlo.multiply %v2748, %v2754 : tensor<32x96x14x14xf32>
    %v2756 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2757 = stablehlo.reshape %v2659 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2758 = stablehlo.multiply %v2756, %v2757 : tensor<32x96x14x14xf32>
    %v2759 = stablehlo.reduce(%v2758 init: %v2742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2760 = stablehlo.broadcast_in_dim %v2759, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2761 = stablehlo.multiply %v2755, %v2758 : tensor<32x96x14x14xf32>
    %v2762 = stablehlo.reduce(%v2761 init: %v2742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2763 = stablehlo.broadcast_in_dim %v2762, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2764 = stablehlo.multiply %v2758, %v2743 : tensor<32x96x14x14xf32>
    %v2765 = stablehlo.subtract %v2764, %v2760 : tensor<32x96x14x14xf32>
    %v2766 = stablehlo.multiply %v2755, %v2763 : tensor<32x96x14x14xf32>
    %v2767 = stablehlo.subtract %v2765, %v2766 : tensor<32x96x14x14xf32>
    %v2768 = stablehlo.divide %v2754, %v2743 : tensor<32x96x14x14xf32>
    %v2769 = stablehlo.multiply %v2768, %v2767 : tensor<32x96x14x14xf32>
    %v2770 = stablehlo.reshape %v2769 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2771 = stablehlo.reshape %v2770 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2772 = stablehlo.reverse %b11pW, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v2773 = stablehlo.transpose %v2772, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v2774 = stablehlo.convolution(%v2771, %v2773)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2775 = stablehlo.reshape %v2774 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2776 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v2777 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v2778 = stablehlo.compare GT, %v889, %v2776 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v2779 = stablehlo.compare LT, %v889, %v2777 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v2780 = stablehlo.and %v2778, %v2779 : tensor<32x75264xi1>
    %v2781 = stablehlo.select %v2780, %v2775, %v2776 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v2782 = stablehlo.reshape %v869 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2784 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v2785 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v2786 = stablehlo.reduce(%v2782 init: %v2783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2787 = stablehlo.broadcast_in_dim %v2786, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2788 = stablehlo.divide %v2787, %v2784 : tensor<32x384x14x14xf32>
    %v2789 = stablehlo.subtract %v2782, %v2788 : tensor<32x384x14x14xf32>
    %v2790 = stablehlo.multiply %v2789, %v2789 : tensor<32x384x14x14xf32>
    %v2791 = stablehlo.reduce(%v2790 init: %v2783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2792 = stablehlo.broadcast_in_dim %v2791, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2793 = stablehlo.divide %v2792, %v2784 : tensor<32x384x14x14xf32>
    %v2794 = stablehlo.add %v2793, %v2785 : tensor<32x384x14x14xf32>
    %v2795 = stablehlo.rsqrt %v2794 : tensor<32x384x14x14xf32>
    %v2796 = stablehlo.multiply %v2789, %v2795 : tensor<32x384x14x14xf32>
    %v2797 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2798 = stablehlo.reshape %v2781 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2799 = stablehlo.multiply %v2797, %v2798 : tensor<32x384x14x14xf32>
    %v2800 = stablehlo.reduce(%v2799 init: %v2783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2801 = stablehlo.broadcast_in_dim %v2800, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2802 = stablehlo.multiply %v2796, %v2799 : tensor<32x384x14x14xf32>
    %v2803 = stablehlo.reduce(%v2802 init: %v2783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2804 = stablehlo.broadcast_in_dim %v2803, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2805 = stablehlo.multiply %v2799, %v2784 : tensor<32x384x14x14xf32>
    %v2806 = stablehlo.subtract %v2805, %v2801 : tensor<32x384x14x14xf32>
    %v2807 = stablehlo.multiply %v2796, %v2804 : tensor<32x384x14x14xf32>
    %v2808 = stablehlo.subtract %v2806, %v2807 : tensor<32x384x14x14xf32>
    %v2809 = stablehlo.divide %v2795, %v2784 : tensor<32x384x14x14xf32>
    %v2810 = stablehlo.multiply %v2809, %v2808 : tensor<32x384x14x14xf32>
    %v2811 = stablehlo.reshape %v2810 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2812 = stablehlo.reshape %v2811 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2813 = stablehlo.reverse %b11dW, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v2814 = stablehlo.convolution(%v2812, %v2813)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v2815 = stablehlo.reshape %v2814 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2816 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v2817 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v2818 = stablehlo.compare GT, %v860, %v2816 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v2819 = stablehlo.compare LT, %v860, %v2817 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v2820 = stablehlo.and %v2818, %v2819 : tensor<32x75264xi1>
    %v2821 = stablehlo.select %v2820, %v2815, %v2816 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v2822 = stablehlo.reshape %v840 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2823 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2824 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v2825 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v2826 = stablehlo.reduce(%v2822 init: %v2823) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2827 = stablehlo.broadcast_in_dim %v2826, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2828 = stablehlo.divide %v2827, %v2824 : tensor<32x384x14x14xf32>
    %v2829 = stablehlo.subtract %v2822, %v2828 : tensor<32x384x14x14xf32>
    %v2830 = stablehlo.multiply %v2829, %v2829 : tensor<32x384x14x14xf32>
    %v2831 = stablehlo.reduce(%v2830 init: %v2823) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2832 = stablehlo.broadcast_in_dim %v2831, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2833 = stablehlo.divide %v2832, %v2824 : tensor<32x384x14x14xf32>
    %v2834 = stablehlo.add %v2833, %v2825 : tensor<32x384x14x14xf32>
    %v2835 = stablehlo.rsqrt %v2834 : tensor<32x384x14x14xf32>
    %v2836 = stablehlo.multiply %v2829, %v2835 : tensor<32x384x14x14xf32>
    %v2837 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2838 = stablehlo.reshape %v2821 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2839 = stablehlo.multiply %v2837, %v2838 : tensor<32x384x14x14xf32>
    %v2840 = stablehlo.reduce(%v2839 init: %v2823) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2841 = stablehlo.broadcast_in_dim %v2840, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2842 = stablehlo.multiply %v2836, %v2839 : tensor<32x384x14x14xf32>
    %v2843 = stablehlo.reduce(%v2842 init: %v2823) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2844 = stablehlo.broadcast_in_dim %v2843, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2845 = stablehlo.multiply %v2839, %v2824 : tensor<32x384x14x14xf32>
    %v2846 = stablehlo.subtract %v2845, %v2841 : tensor<32x384x14x14xf32>
    %v2847 = stablehlo.multiply %v2836, %v2844 : tensor<32x384x14x14xf32>
    %v2848 = stablehlo.subtract %v2846, %v2847 : tensor<32x384x14x14xf32>
    %v2849 = stablehlo.divide %v2835, %v2824 : tensor<32x384x14x14xf32>
    %v2850 = stablehlo.multiply %v2849, %v2848 : tensor<32x384x14x14xf32>
    %v2851 = stablehlo.reshape %v2850 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2852 = stablehlo.reshape %v2851 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2853 = stablehlo.reverse %b11eW, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v2854 = stablehlo.transpose %v2853, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v2855 = stablehlo.convolution(%v2852, %v2854)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v2856 = stablehlo.reshape %v2855 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v2857 = stablehlo.reshape %v835 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v2858 = stablehlo.reshape %v2851 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2859 = stablehlo.transpose %v2857, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v2860 = stablehlo.transpose %v2858, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2861 = stablehlo.convolution(%v2859, %v2860)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v2862 = stablehlo.transpose %v2861, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v2863 = stablehlo.reshape %v840 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2865 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v2866 = stablehlo.reduce(%v2863 init: %v2864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2867 = stablehlo.broadcast_in_dim %v2866, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2868 = stablehlo.divide %v2867, %v2865 : tensor<32x384x14x14xf32>
    %v2869 = stablehlo.subtract %v2863, %v2868 : tensor<32x384x14x14xf32>
    %v2870 = stablehlo.multiply %v2869, %v2869 : tensor<32x384x14x14xf32>
    %v2871 = stablehlo.reduce(%v2870 init: %v2864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2872 = stablehlo.broadcast_in_dim %v2871, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2873 = stablehlo.divide %v2872, %v2865 : tensor<32x384x14x14xf32>
    %v2874 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v2875 = stablehlo.add %v2873, %v2874 : tensor<32x384x14x14xf32>
    %v2876 = stablehlo.rsqrt %v2875 : tensor<32x384x14x14xf32>
    %v2877 = stablehlo.multiply %v2869, %v2876 : tensor<32x384x14x14xf32>
    %v2878 = stablehlo.reshape %v2821 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2879 = stablehlo.multiply %v2878, %v2877 : tensor<32x384x14x14xf32>
    %v2880 = stablehlo.reduce(%v2879 init: %v2864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2881 = stablehlo.reshape %v2821 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2883 = stablehlo.reduce(%v2881 init: %v2882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2884 = stablehlo.reshape %v864 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2885 = stablehlo.reshape %v2811 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2886 = stablehlo.transpose %v2884, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2887 = stablehlo.transpose %v2885, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2888 = stablehlo.convolution(%v2886, %v2887)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v2889 = stablehlo.reshape %v2888 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v2890 = stablehlo.reshape %v869 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2892 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v2893 = stablehlo.reduce(%v2890 init: %v2891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2894 = stablehlo.broadcast_in_dim %v2893, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2895 = stablehlo.divide %v2894, %v2892 : tensor<32x384x14x14xf32>
    %v2896 = stablehlo.subtract %v2890, %v2895 : tensor<32x384x14x14xf32>
    %v2897 = stablehlo.multiply %v2896, %v2896 : tensor<32x384x14x14xf32>
    %v2898 = stablehlo.reduce(%v2897 init: %v2891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2899 = stablehlo.broadcast_in_dim %v2898, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2900 = stablehlo.divide %v2899, %v2892 : tensor<32x384x14x14xf32>
    %v2901 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v2902 = stablehlo.add %v2900, %v2901 : tensor<32x384x14x14xf32>
    %v2903 = stablehlo.rsqrt %v2902 : tensor<32x384x14x14xf32>
    %v2904 = stablehlo.multiply %v2896, %v2903 : tensor<32x384x14x14xf32>
    %v2905 = stablehlo.reshape %v2781 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2906 = stablehlo.multiply %v2905, %v2904 : tensor<32x384x14x14xf32>
    %v2907 = stablehlo.reduce(%v2906 init: %v2891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2908 = stablehlo.reshape %v2781 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2910 = stablehlo.reduce(%v2908 init: %v2909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2911 = stablehlo.reshape %v893 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2912 = stablehlo.reshape %v2770 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2913 = stablehlo.transpose %v2911, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2914 = stablehlo.transpose %v2912, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2915 = stablehlo.convolution(%v2913, %v2914)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<384x96x1x1xf32>
    %v2916 = stablehlo.transpose %v2915, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v2917 = stablehlo.reshape %v898 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2919 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2920 = stablehlo.reduce(%v2917 init: %v2918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2921 = stablehlo.broadcast_in_dim %v2920, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2922 = stablehlo.divide %v2921, %v2919 : tensor<32x96x14x14xf32>
    %v2923 = stablehlo.subtract %v2917, %v2922 : tensor<32x96x14x14xf32>
    %v2924 = stablehlo.multiply %v2923, %v2923 : tensor<32x96x14x14xf32>
    %v2925 = stablehlo.reduce(%v2924 init: %v2918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2926 = stablehlo.broadcast_in_dim %v2925, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2927 = stablehlo.divide %v2926, %v2919 : tensor<32x96x14x14xf32>
    %v2928 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2929 = stablehlo.add %v2927, %v2928 : tensor<32x96x14x14xf32>
    %v2930 = stablehlo.rsqrt %v2929 : tensor<32x96x14x14xf32>
    %v2931 = stablehlo.multiply %v2923, %v2930 : tensor<32x96x14x14xf32>
    %v2932 = stablehlo.reshape %v2659 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2933 = stablehlo.multiply %v2932, %v2931 : tensor<32x96x14x14xf32>
    %v2934 = stablehlo.reduce(%v2933 init: %v2918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2935 = stablehlo.reshape %v2659 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2937 = stablehlo.reduce(%v2935 init: %v2936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2938 = stablehlo.reshape %v814 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v2939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2940 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v2941 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v2942 = stablehlo.reduce(%v2938 init: %v2939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v2943 = stablehlo.broadcast_in_dim %v2942, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v2944 = stablehlo.divide %v2943, %v2940 : tensor<32x64x14x14xf32>
    %v2945 = stablehlo.subtract %v2938, %v2944 : tensor<32x64x14x14xf32>
    %v2946 = stablehlo.multiply %v2945, %v2945 : tensor<32x64x14x14xf32>
    %v2947 = stablehlo.reduce(%v2946 init: %v2939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v2948 = stablehlo.broadcast_in_dim %v2947, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v2949 = stablehlo.divide %v2948, %v2940 : tensor<32x64x14x14xf32>
    %v2950 = stablehlo.add %v2949, %v2941 : tensor<32x64x14x14xf32>
    %v2951 = stablehlo.rsqrt %v2950 : tensor<32x64x14x14xf32>
    %v2952 = stablehlo.multiply %v2945, %v2951 : tensor<32x64x14x14xf32>
    %v2953 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v2954 = stablehlo.reshape %v2856 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v2955 = stablehlo.multiply %v2953, %v2954 : tensor<32x64x14x14xf32>
    %v2956 = stablehlo.reduce(%v2955 init: %v2939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v2957 = stablehlo.broadcast_in_dim %v2956, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v2958 = stablehlo.multiply %v2952, %v2955 : tensor<32x64x14x14xf32>
    %v2959 = stablehlo.reduce(%v2958 init: %v2939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v2960 = stablehlo.broadcast_in_dim %v2959, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v2961 = stablehlo.multiply %v2955, %v2940 : tensor<32x64x14x14xf32>
    %v2962 = stablehlo.subtract %v2961, %v2957 : tensor<32x64x14x14xf32>
    %v2963 = stablehlo.multiply %v2952, %v2960 : tensor<32x64x14x14xf32>
    %v2964 = stablehlo.subtract %v2962, %v2963 : tensor<32x64x14x14xf32>
    %v2965 = stablehlo.divide %v2951, %v2940 : tensor<32x64x14x14xf32>
    %v2966 = stablehlo.multiply %v2965, %v2964 : tensor<32x64x14x14xf32>
    %v2967 = stablehlo.reshape %v2966 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v2968 = stablehlo.reshape %v2967 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v2969 = stablehlo.reverse %b10pW, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v2970 = stablehlo.transpose %v2969, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v2971 = stablehlo.convolution(%v2968, %v2970)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2972 = stablehlo.reshape %v2971 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2973 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v2974 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v2975 = stablehlo.compare GT, %v805, %v2973 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v2976 = stablehlo.compare LT, %v805, %v2974 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v2977 = stablehlo.and %v2975, %v2976 : tensor<32x75264xi1>
    %v2978 = stablehlo.select %v2977, %v2972, %v2973 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v2979 = stablehlo.reshape %v785 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2980 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2981 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v2982 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v2983 = stablehlo.reduce(%v2979 init: %v2980) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2984 = stablehlo.broadcast_in_dim %v2983, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2985 = stablehlo.divide %v2984, %v2981 : tensor<32x384x14x14xf32>
    %v2986 = stablehlo.subtract %v2979, %v2985 : tensor<32x384x14x14xf32>
    %v2987 = stablehlo.multiply %v2986, %v2986 : tensor<32x384x14x14xf32>
    %v2988 = stablehlo.reduce(%v2987 init: %v2980) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2989 = stablehlo.broadcast_in_dim %v2988, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2990 = stablehlo.divide %v2989, %v2981 : tensor<32x384x14x14xf32>
    %v2991 = stablehlo.add %v2990, %v2982 : tensor<32x384x14x14xf32>
    %v2992 = stablehlo.rsqrt %v2991 : tensor<32x384x14x14xf32>
    %v2993 = stablehlo.multiply %v2986, %v2992 : tensor<32x384x14x14xf32>
    %v2994 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2995 = stablehlo.reshape %v2978 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2996 = stablehlo.multiply %v2994, %v2995 : tensor<32x384x14x14xf32>
    %v2997 = stablehlo.reduce(%v2996 init: %v2980) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2998 = stablehlo.broadcast_in_dim %v2997, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2999 = stablehlo.multiply %v2993, %v2996 : tensor<32x384x14x14xf32>
    %v3000 = stablehlo.reduce(%v2999 init: %v2980) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3001 = stablehlo.broadcast_in_dim %v3000, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3002 = stablehlo.multiply %v2996, %v2981 : tensor<32x384x14x14xf32>
    %v3003 = stablehlo.subtract %v3002, %v2998 : tensor<32x384x14x14xf32>
    %v3004 = stablehlo.multiply %v2993, %v3001 : tensor<32x384x14x14xf32>
    %v3005 = stablehlo.subtract %v3003, %v3004 : tensor<32x384x14x14xf32>
    %v3006 = stablehlo.divide %v2992, %v2981 : tensor<32x384x14x14xf32>
    %v3007 = stablehlo.multiply %v3006, %v3005 : tensor<32x384x14x14xf32>
    %v3008 = stablehlo.reshape %v3007 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3009 = stablehlo.reshape %v3008 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3010 = stablehlo.reverse %b10dW, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3011 = stablehlo.convolution(%v3009, %v3010)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3012 = stablehlo.reshape %v3011 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3013 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3014 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3015 = stablehlo.compare GT, %v776, %v3013 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3016 = stablehlo.compare LT, %v776, %v3014 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3017 = stablehlo.and %v3015, %v3016 : tensor<32x75264xi1>
    %v3018 = stablehlo.select %v3017, %v3012, %v3013 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3019 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3020 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3021 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3022 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3023 = stablehlo.reduce(%v3019 init: %v3020) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3024 = stablehlo.broadcast_in_dim %v3023, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3025 = stablehlo.divide %v3024, %v3021 : tensor<32x384x14x14xf32>
    %v3026 = stablehlo.subtract %v3019, %v3025 : tensor<32x384x14x14xf32>
    %v3027 = stablehlo.multiply %v3026, %v3026 : tensor<32x384x14x14xf32>
    %v3028 = stablehlo.reduce(%v3027 init: %v3020) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3029 = stablehlo.broadcast_in_dim %v3028, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3030 = stablehlo.divide %v3029, %v3021 : tensor<32x384x14x14xf32>
    %v3031 = stablehlo.add %v3030, %v3022 : tensor<32x384x14x14xf32>
    %v3032 = stablehlo.rsqrt %v3031 : tensor<32x384x14x14xf32>
    %v3033 = stablehlo.multiply %v3026, %v3032 : tensor<32x384x14x14xf32>
    %v3034 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3035 = stablehlo.reshape %v3018 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3036 = stablehlo.multiply %v3034, %v3035 : tensor<32x384x14x14xf32>
    %v3037 = stablehlo.reduce(%v3036 init: %v3020) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3038 = stablehlo.broadcast_in_dim %v3037, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3039 = stablehlo.multiply %v3033, %v3036 : tensor<32x384x14x14xf32>
    %v3040 = stablehlo.reduce(%v3039 init: %v3020) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3041 = stablehlo.broadcast_in_dim %v3040, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3042 = stablehlo.multiply %v3036, %v3021 : tensor<32x384x14x14xf32>
    %v3043 = stablehlo.subtract %v3042, %v3038 : tensor<32x384x14x14xf32>
    %v3044 = stablehlo.multiply %v3033, %v3041 : tensor<32x384x14x14xf32>
    %v3045 = stablehlo.subtract %v3043, %v3044 : tensor<32x384x14x14xf32>
    %v3046 = stablehlo.divide %v3032, %v3021 : tensor<32x384x14x14xf32>
    %v3047 = stablehlo.multiply %v3046, %v3045 : tensor<32x384x14x14xf32>
    %v3048 = stablehlo.reshape %v3047 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3049 = stablehlo.reshape %v3048 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3050 = stablehlo.reverse %b10eW, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3051 = stablehlo.transpose %v3050, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3052 = stablehlo.convolution(%v3049, %v3051)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3053 = stablehlo.reshape %v3052 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3054 = stablehlo.add %v3053, %v2856 : tensor<32x12544xf32>
    %v3055 = stablehlo.reshape %v751 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3056 = stablehlo.reshape %v3048 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3057 = stablehlo.transpose %v3055, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3058 = stablehlo.transpose %v3056, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3059 = stablehlo.convolution(%v3057, %v3058)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3060 = stablehlo.transpose %v3059, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3061 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3062 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3063 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3064 = stablehlo.reduce(%v3061 init: %v3062) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3065 = stablehlo.broadcast_in_dim %v3064, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3066 = stablehlo.divide %v3065, %v3063 : tensor<32x384x14x14xf32>
    %v3067 = stablehlo.subtract %v3061, %v3066 : tensor<32x384x14x14xf32>
    %v3068 = stablehlo.multiply %v3067, %v3067 : tensor<32x384x14x14xf32>
    %v3069 = stablehlo.reduce(%v3068 init: %v3062) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3070 = stablehlo.broadcast_in_dim %v3069, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3071 = stablehlo.divide %v3070, %v3063 : tensor<32x384x14x14xf32>
    %v3072 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3073 = stablehlo.add %v3071, %v3072 : tensor<32x384x14x14xf32>
    %v3074 = stablehlo.rsqrt %v3073 : tensor<32x384x14x14xf32>
    %v3075 = stablehlo.multiply %v3067, %v3074 : tensor<32x384x14x14xf32>
    %v3076 = stablehlo.reshape %v3018 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3077 = stablehlo.multiply %v3076, %v3075 : tensor<32x384x14x14xf32>
    %v3078 = stablehlo.reduce(%v3077 init: %v3062) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3079 = stablehlo.reshape %v3018 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3080 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3081 = stablehlo.reduce(%v3079 init: %v3080) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3082 = stablehlo.reshape %v780 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3083 = stablehlo.reshape %v3008 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3084 = stablehlo.transpose %v3082, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3085 = stablehlo.transpose %v3083, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3086 = stablehlo.convolution(%v3084, %v3085)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3087 = stablehlo.reshape %v3086 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3088 = stablehlo.reshape %v785 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3089 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3090 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3091 = stablehlo.reduce(%v3088 init: %v3089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3092 = stablehlo.broadcast_in_dim %v3091, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3093 = stablehlo.divide %v3092, %v3090 : tensor<32x384x14x14xf32>
    %v3094 = stablehlo.subtract %v3088, %v3093 : tensor<32x384x14x14xf32>
    %v3095 = stablehlo.multiply %v3094, %v3094 : tensor<32x384x14x14xf32>
    %v3096 = stablehlo.reduce(%v3095 init: %v3089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3097 = stablehlo.broadcast_in_dim %v3096, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3098 = stablehlo.divide %v3097, %v3090 : tensor<32x384x14x14xf32>
    %v3099 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3100 = stablehlo.add %v3098, %v3099 : tensor<32x384x14x14xf32>
    %v3101 = stablehlo.rsqrt %v3100 : tensor<32x384x14x14xf32>
    %v3102 = stablehlo.multiply %v3094, %v3101 : tensor<32x384x14x14xf32>
    %v3103 = stablehlo.reshape %v2978 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3104 = stablehlo.multiply %v3103, %v3102 : tensor<32x384x14x14xf32>
    %v3105 = stablehlo.reduce(%v3104 init: %v3089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3106 = stablehlo.reshape %v2978 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3107 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3108 = stablehlo.reduce(%v3106 init: %v3107) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3109 = stablehlo.reshape %v809 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3110 = stablehlo.reshape %v2967 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3111 = stablehlo.transpose %v3109, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3112 = stablehlo.transpose %v3110, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3113 = stablehlo.convolution(%v3111, %v3112)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3114 = stablehlo.transpose %v3113, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3115 = stablehlo.reshape %v814 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3116 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3117 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3118 = stablehlo.reduce(%v3115 init: %v3116) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3119 = stablehlo.broadcast_in_dim %v3118, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3120 = stablehlo.divide %v3119, %v3117 : tensor<32x64x14x14xf32>
    %v3121 = stablehlo.subtract %v3115, %v3120 : tensor<32x64x14x14xf32>
    %v3122 = stablehlo.multiply %v3121, %v3121 : tensor<32x64x14x14xf32>
    %v3123 = stablehlo.reduce(%v3122 init: %v3116) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3124 = stablehlo.broadcast_in_dim %v3123, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3125 = stablehlo.divide %v3124, %v3117 : tensor<32x64x14x14xf32>
    %v3126 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3127 = stablehlo.add %v3125, %v3126 : tensor<32x64x14x14xf32>
    %v3128 = stablehlo.rsqrt %v3127 : tensor<32x64x14x14xf32>
    %v3129 = stablehlo.multiply %v3121, %v3128 : tensor<32x64x14x14xf32>
    %v3130 = stablehlo.reshape %v2856 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3131 = stablehlo.multiply %v3130, %v3129 : tensor<32x64x14x14xf32>
    %v3132 = stablehlo.reduce(%v3131 init: %v3116) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3133 = stablehlo.reshape %v2856 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3134 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3135 = stablehlo.reduce(%v3133 init: %v3134) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3136 = stablehlo.reshape %v730 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3137 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3138 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3139 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3140 = stablehlo.reduce(%v3136 init: %v3137) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3141 = stablehlo.broadcast_in_dim %v3140, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3142 = stablehlo.divide %v3141, %v3138 : tensor<32x64x14x14xf32>
    %v3143 = stablehlo.subtract %v3136, %v3142 : tensor<32x64x14x14xf32>
    %v3144 = stablehlo.multiply %v3143, %v3143 : tensor<32x64x14x14xf32>
    %v3145 = stablehlo.reduce(%v3144 init: %v3137) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3146 = stablehlo.broadcast_in_dim %v3145, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3147 = stablehlo.divide %v3146, %v3138 : tensor<32x64x14x14xf32>
    %v3148 = stablehlo.add %v3147, %v3139 : tensor<32x64x14x14xf32>
    %v3149 = stablehlo.rsqrt %v3148 : tensor<32x64x14x14xf32>
    %v3150 = stablehlo.multiply %v3143, %v3149 : tensor<32x64x14x14xf32>
    %v3151 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3152 = stablehlo.reshape %v3054 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3153 = stablehlo.multiply %v3151, %v3152 : tensor<32x64x14x14xf32>
    %v3154 = stablehlo.reduce(%v3153 init: %v3137) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3155 = stablehlo.broadcast_in_dim %v3154, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3156 = stablehlo.multiply %v3150, %v3153 : tensor<32x64x14x14xf32>
    %v3157 = stablehlo.reduce(%v3156 init: %v3137) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3158 = stablehlo.broadcast_in_dim %v3157, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3159 = stablehlo.multiply %v3153, %v3138 : tensor<32x64x14x14xf32>
    %v3160 = stablehlo.subtract %v3159, %v3155 : tensor<32x64x14x14xf32>
    %v3161 = stablehlo.multiply %v3150, %v3158 : tensor<32x64x14x14xf32>
    %v3162 = stablehlo.subtract %v3160, %v3161 : tensor<32x64x14x14xf32>
    %v3163 = stablehlo.divide %v3149, %v3138 : tensor<32x64x14x14xf32>
    %v3164 = stablehlo.multiply %v3163, %v3162 : tensor<32x64x14x14xf32>
    %v3165 = stablehlo.reshape %v3164 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3166 = stablehlo.reshape %v3165 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3167 = stablehlo.reverse %b9pW, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3168 = stablehlo.transpose %v3167, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3169 = stablehlo.convolution(%v3166, %v3168)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3170 = stablehlo.reshape %v3169 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3171 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3172 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3173 = stablehlo.compare GT, %v721, %v3171 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3174 = stablehlo.compare LT, %v721, %v3172 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3175 = stablehlo.and %v3173, %v3174 : tensor<32x75264xi1>
    %v3176 = stablehlo.select %v3175, %v3170, %v3171 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3177 = stablehlo.reshape %v701 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3179 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3180 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3181 = stablehlo.reduce(%v3177 init: %v3178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3182 = stablehlo.broadcast_in_dim %v3181, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3183 = stablehlo.divide %v3182, %v3179 : tensor<32x384x14x14xf32>
    %v3184 = stablehlo.subtract %v3177, %v3183 : tensor<32x384x14x14xf32>
    %v3185 = stablehlo.multiply %v3184, %v3184 : tensor<32x384x14x14xf32>
    %v3186 = stablehlo.reduce(%v3185 init: %v3178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3187 = stablehlo.broadcast_in_dim %v3186, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3188 = stablehlo.divide %v3187, %v3179 : tensor<32x384x14x14xf32>
    %v3189 = stablehlo.add %v3188, %v3180 : tensor<32x384x14x14xf32>
    %v3190 = stablehlo.rsqrt %v3189 : tensor<32x384x14x14xf32>
    %v3191 = stablehlo.multiply %v3184, %v3190 : tensor<32x384x14x14xf32>
    %v3192 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3193 = stablehlo.reshape %v3176 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3194 = stablehlo.multiply %v3192, %v3193 : tensor<32x384x14x14xf32>
    %v3195 = stablehlo.reduce(%v3194 init: %v3178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3196 = stablehlo.broadcast_in_dim %v3195, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3197 = stablehlo.multiply %v3191, %v3194 : tensor<32x384x14x14xf32>
    %v3198 = stablehlo.reduce(%v3197 init: %v3178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3199 = stablehlo.broadcast_in_dim %v3198, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3200 = stablehlo.multiply %v3194, %v3179 : tensor<32x384x14x14xf32>
    %v3201 = stablehlo.subtract %v3200, %v3196 : tensor<32x384x14x14xf32>
    %v3202 = stablehlo.multiply %v3191, %v3199 : tensor<32x384x14x14xf32>
    %v3203 = stablehlo.subtract %v3201, %v3202 : tensor<32x384x14x14xf32>
    %v3204 = stablehlo.divide %v3190, %v3179 : tensor<32x384x14x14xf32>
    %v3205 = stablehlo.multiply %v3204, %v3203 : tensor<32x384x14x14xf32>
    %v3206 = stablehlo.reshape %v3205 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3207 = stablehlo.reshape %v3206 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3208 = stablehlo.reverse %b9dW, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3209 = stablehlo.convolution(%v3207, %v3208)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3210 = stablehlo.reshape %v3209 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3211 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3212 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3213 = stablehlo.compare GT, %v692, %v3211 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3214 = stablehlo.compare LT, %v692, %v3212 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3215 = stablehlo.and %v3213, %v3214 : tensor<32x75264xi1>
    %v3216 = stablehlo.select %v3215, %v3210, %v3211 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3217 = stablehlo.reshape %v672 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3219 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3220 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3221 = stablehlo.reduce(%v3217 init: %v3218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3222 = stablehlo.broadcast_in_dim %v3221, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3223 = stablehlo.divide %v3222, %v3219 : tensor<32x384x14x14xf32>
    %v3224 = stablehlo.subtract %v3217, %v3223 : tensor<32x384x14x14xf32>
    %v3225 = stablehlo.multiply %v3224, %v3224 : tensor<32x384x14x14xf32>
    %v3226 = stablehlo.reduce(%v3225 init: %v3218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3227 = stablehlo.broadcast_in_dim %v3226, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3228 = stablehlo.divide %v3227, %v3219 : tensor<32x384x14x14xf32>
    %v3229 = stablehlo.add %v3228, %v3220 : tensor<32x384x14x14xf32>
    %v3230 = stablehlo.rsqrt %v3229 : tensor<32x384x14x14xf32>
    %v3231 = stablehlo.multiply %v3224, %v3230 : tensor<32x384x14x14xf32>
    %v3232 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3233 = stablehlo.reshape %v3216 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3234 = stablehlo.multiply %v3232, %v3233 : tensor<32x384x14x14xf32>
    %v3235 = stablehlo.reduce(%v3234 init: %v3218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3236 = stablehlo.broadcast_in_dim %v3235, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3237 = stablehlo.multiply %v3231, %v3234 : tensor<32x384x14x14xf32>
    %v3238 = stablehlo.reduce(%v3237 init: %v3218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3239 = stablehlo.broadcast_in_dim %v3238, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3240 = stablehlo.multiply %v3234, %v3219 : tensor<32x384x14x14xf32>
    %v3241 = stablehlo.subtract %v3240, %v3236 : tensor<32x384x14x14xf32>
    %v3242 = stablehlo.multiply %v3231, %v3239 : tensor<32x384x14x14xf32>
    %v3243 = stablehlo.subtract %v3241, %v3242 : tensor<32x384x14x14xf32>
    %v3244 = stablehlo.divide %v3230, %v3219 : tensor<32x384x14x14xf32>
    %v3245 = stablehlo.multiply %v3244, %v3243 : tensor<32x384x14x14xf32>
    %v3246 = stablehlo.reshape %v3245 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3247 = stablehlo.reshape %v3246 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3248 = stablehlo.reverse %b9eW, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3249 = stablehlo.transpose %v3248, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3250 = stablehlo.convolution(%v3247, %v3249)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3251 = stablehlo.reshape %v3250 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3252 = stablehlo.add %v3251, %v3054 : tensor<32x12544xf32>
    %v3253 = stablehlo.reshape %v667 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3254 = stablehlo.reshape %v3246 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3255 = stablehlo.transpose %v3253, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3256 = stablehlo.transpose %v3254, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3257 = stablehlo.convolution(%v3255, %v3256)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3258 = stablehlo.transpose %v3257, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3259 = stablehlo.reshape %v672 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3261 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3262 = stablehlo.reduce(%v3259 init: %v3260) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3263 = stablehlo.broadcast_in_dim %v3262, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3264 = stablehlo.divide %v3263, %v3261 : tensor<32x384x14x14xf32>
    %v3265 = stablehlo.subtract %v3259, %v3264 : tensor<32x384x14x14xf32>
    %v3266 = stablehlo.multiply %v3265, %v3265 : tensor<32x384x14x14xf32>
    %v3267 = stablehlo.reduce(%v3266 init: %v3260) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3268 = stablehlo.broadcast_in_dim %v3267, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3269 = stablehlo.divide %v3268, %v3261 : tensor<32x384x14x14xf32>
    %v3270 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3271 = stablehlo.add %v3269, %v3270 : tensor<32x384x14x14xf32>
    %v3272 = stablehlo.rsqrt %v3271 : tensor<32x384x14x14xf32>
    %v3273 = stablehlo.multiply %v3265, %v3272 : tensor<32x384x14x14xf32>
    %v3274 = stablehlo.reshape %v3216 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3275 = stablehlo.multiply %v3274, %v3273 : tensor<32x384x14x14xf32>
    %v3276 = stablehlo.reduce(%v3275 init: %v3260) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3277 = stablehlo.reshape %v3216 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3279 = stablehlo.reduce(%v3277 init: %v3278) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3280 = stablehlo.reshape %v696 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3281 = stablehlo.reshape %v3206 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3282 = stablehlo.transpose %v3280, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3283 = stablehlo.transpose %v3281, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3284 = stablehlo.convolution(%v3282, %v3283)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3285 = stablehlo.reshape %v3284 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3286 = stablehlo.reshape %v701 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3287 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3288 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3289 = stablehlo.reduce(%v3286 init: %v3287) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3290 = stablehlo.broadcast_in_dim %v3289, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3291 = stablehlo.divide %v3290, %v3288 : tensor<32x384x14x14xf32>
    %v3292 = stablehlo.subtract %v3286, %v3291 : tensor<32x384x14x14xf32>
    %v3293 = stablehlo.multiply %v3292, %v3292 : tensor<32x384x14x14xf32>
    %v3294 = stablehlo.reduce(%v3293 init: %v3287) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3295 = stablehlo.broadcast_in_dim %v3294, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3296 = stablehlo.divide %v3295, %v3288 : tensor<32x384x14x14xf32>
    %v3297 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3298 = stablehlo.add %v3296, %v3297 : tensor<32x384x14x14xf32>
    %v3299 = stablehlo.rsqrt %v3298 : tensor<32x384x14x14xf32>
    %v3300 = stablehlo.multiply %v3292, %v3299 : tensor<32x384x14x14xf32>
    %v3301 = stablehlo.reshape %v3176 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3302 = stablehlo.multiply %v3301, %v3300 : tensor<32x384x14x14xf32>
    %v3303 = stablehlo.reduce(%v3302 init: %v3287) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3304 = stablehlo.reshape %v3176 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3305 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3306 = stablehlo.reduce(%v3304 init: %v3305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3307 = stablehlo.reshape %v725 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3308 = stablehlo.reshape %v3165 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3309 = stablehlo.transpose %v3307, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3310 = stablehlo.transpose %v3308, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3311 = stablehlo.convolution(%v3309, %v3310)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3312 = stablehlo.transpose %v3311, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3313 = stablehlo.reshape %v730 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3315 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3316 = stablehlo.reduce(%v3313 init: %v3314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3317 = stablehlo.broadcast_in_dim %v3316, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3318 = stablehlo.divide %v3317, %v3315 : tensor<32x64x14x14xf32>
    %v3319 = stablehlo.subtract %v3313, %v3318 : tensor<32x64x14x14xf32>
    %v3320 = stablehlo.multiply %v3319, %v3319 : tensor<32x64x14x14xf32>
    %v3321 = stablehlo.reduce(%v3320 init: %v3314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3322 = stablehlo.broadcast_in_dim %v3321, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3323 = stablehlo.divide %v3322, %v3315 : tensor<32x64x14x14xf32>
    %v3324 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3325 = stablehlo.add %v3323, %v3324 : tensor<32x64x14x14xf32>
    %v3326 = stablehlo.rsqrt %v3325 : tensor<32x64x14x14xf32>
    %v3327 = stablehlo.multiply %v3319, %v3326 : tensor<32x64x14x14xf32>
    %v3328 = stablehlo.reshape %v3054 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3329 = stablehlo.multiply %v3328, %v3327 : tensor<32x64x14x14xf32>
    %v3330 = stablehlo.reduce(%v3329 init: %v3314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3331 = stablehlo.reshape %v3054 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3333 = stablehlo.reduce(%v3331 init: %v3332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3334 = stablehlo.reshape %v646 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3336 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3337 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3338 = stablehlo.reduce(%v3334 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3339 = stablehlo.broadcast_in_dim %v3338, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3340 = stablehlo.divide %v3339, %v3336 : tensor<32x64x14x14xf32>
    %v3341 = stablehlo.subtract %v3334, %v3340 : tensor<32x64x14x14xf32>
    %v3342 = stablehlo.multiply %v3341, %v3341 : tensor<32x64x14x14xf32>
    %v3343 = stablehlo.reduce(%v3342 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3344 = stablehlo.broadcast_in_dim %v3343, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3345 = stablehlo.divide %v3344, %v3336 : tensor<32x64x14x14xf32>
    %v3346 = stablehlo.add %v3345, %v3337 : tensor<32x64x14x14xf32>
    %v3347 = stablehlo.rsqrt %v3346 : tensor<32x64x14x14xf32>
    %v3348 = stablehlo.multiply %v3341, %v3347 : tensor<32x64x14x14xf32>
    %v3349 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3350 = stablehlo.reshape %v3252 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3351 = stablehlo.multiply %v3349, %v3350 : tensor<32x64x14x14xf32>
    %v3352 = stablehlo.reduce(%v3351 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3353 = stablehlo.broadcast_in_dim %v3352, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3354 = stablehlo.multiply %v3348, %v3351 : tensor<32x64x14x14xf32>
    %v3355 = stablehlo.reduce(%v3354 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3356 = stablehlo.broadcast_in_dim %v3355, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3357 = stablehlo.multiply %v3351, %v3336 : tensor<32x64x14x14xf32>
    %v3358 = stablehlo.subtract %v3357, %v3353 : tensor<32x64x14x14xf32>
    %v3359 = stablehlo.multiply %v3348, %v3356 : tensor<32x64x14x14xf32>
    %v3360 = stablehlo.subtract %v3358, %v3359 : tensor<32x64x14x14xf32>
    %v3361 = stablehlo.divide %v3347, %v3336 : tensor<32x64x14x14xf32>
    %v3362 = stablehlo.multiply %v3361, %v3360 : tensor<32x64x14x14xf32>
    %v3363 = stablehlo.reshape %v3362 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3364 = stablehlo.reshape %v3363 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3365 = stablehlo.reverse %b8pW, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3366 = stablehlo.transpose %v3365, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3367 = stablehlo.convolution(%v3364, %v3366)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3368 = stablehlo.reshape %v3367 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3369 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3370 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3371 = stablehlo.compare GT, %v637, %v3369 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3372 = stablehlo.compare LT, %v637, %v3370 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3373 = stablehlo.and %v3371, %v3372 : tensor<32x75264xi1>
    %v3374 = stablehlo.select %v3373, %v3368, %v3369 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3375 = stablehlo.reshape %v617 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3376 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3377 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3378 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3379 = stablehlo.reduce(%v3375 init: %v3376) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3380 = stablehlo.broadcast_in_dim %v3379, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3381 = stablehlo.divide %v3380, %v3377 : tensor<32x384x14x14xf32>
    %v3382 = stablehlo.subtract %v3375, %v3381 : tensor<32x384x14x14xf32>
    %v3383 = stablehlo.multiply %v3382, %v3382 : tensor<32x384x14x14xf32>
    %v3384 = stablehlo.reduce(%v3383 init: %v3376) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3385 = stablehlo.broadcast_in_dim %v3384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3386 = stablehlo.divide %v3385, %v3377 : tensor<32x384x14x14xf32>
    %v3387 = stablehlo.add %v3386, %v3378 : tensor<32x384x14x14xf32>
    %v3388 = stablehlo.rsqrt %v3387 : tensor<32x384x14x14xf32>
    %v3389 = stablehlo.multiply %v3382, %v3388 : tensor<32x384x14x14xf32>
    %v3390 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3391 = stablehlo.reshape %v3374 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3392 = stablehlo.multiply %v3390, %v3391 : tensor<32x384x14x14xf32>
    %v3393 = stablehlo.reduce(%v3392 init: %v3376) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3394 = stablehlo.broadcast_in_dim %v3393, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3395 = stablehlo.multiply %v3389, %v3392 : tensor<32x384x14x14xf32>
    %v3396 = stablehlo.reduce(%v3395 init: %v3376) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3397 = stablehlo.broadcast_in_dim %v3396, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3398 = stablehlo.multiply %v3392, %v3377 : tensor<32x384x14x14xf32>
    %v3399 = stablehlo.subtract %v3398, %v3394 : tensor<32x384x14x14xf32>
    %v3400 = stablehlo.multiply %v3389, %v3397 : tensor<32x384x14x14xf32>
    %v3401 = stablehlo.subtract %v3399, %v3400 : tensor<32x384x14x14xf32>
    %v3402 = stablehlo.divide %v3388, %v3377 : tensor<32x384x14x14xf32>
    %v3403 = stablehlo.multiply %v3402, %v3401 : tensor<32x384x14x14xf32>
    %v3404 = stablehlo.reshape %v3403 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3405 = stablehlo.reshape %v3404 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3406 = stablehlo.reverse %b8dW, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3407 = stablehlo.convolution(%v3405, %v3406)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3408 = stablehlo.reshape %v3407 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3409 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3410 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3411 = stablehlo.compare GT, %v608, %v3409 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3412 = stablehlo.compare LT, %v608, %v3410 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3413 = stablehlo.and %v3411, %v3412 : tensor<32x75264xi1>
    %v3414 = stablehlo.select %v3413, %v3408, %v3409 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3415 = stablehlo.reshape %v588 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3416 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3417 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3418 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3419 = stablehlo.reduce(%v3415 init: %v3416) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3420 = stablehlo.broadcast_in_dim %v3419, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3421 = stablehlo.divide %v3420, %v3417 : tensor<32x384x14x14xf32>
    %v3422 = stablehlo.subtract %v3415, %v3421 : tensor<32x384x14x14xf32>
    %v3423 = stablehlo.multiply %v3422, %v3422 : tensor<32x384x14x14xf32>
    %v3424 = stablehlo.reduce(%v3423 init: %v3416) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3425 = stablehlo.broadcast_in_dim %v3424, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3426 = stablehlo.divide %v3425, %v3417 : tensor<32x384x14x14xf32>
    %v3427 = stablehlo.add %v3426, %v3418 : tensor<32x384x14x14xf32>
    %v3428 = stablehlo.rsqrt %v3427 : tensor<32x384x14x14xf32>
    %v3429 = stablehlo.multiply %v3422, %v3428 : tensor<32x384x14x14xf32>
    %v3430 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3431 = stablehlo.reshape %v3414 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3432 = stablehlo.multiply %v3430, %v3431 : tensor<32x384x14x14xf32>
    %v3433 = stablehlo.reduce(%v3432 init: %v3416) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3434 = stablehlo.broadcast_in_dim %v3433, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3435 = stablehlo.multiply %v3429, %v3432 : tensor<32x384x14x14xf32>
    %v3436 = stablehlo.reduce(%v3435 init: %v3416) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3437 = stablehlo.broadcast_in_dim %v3436, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3438 = stablehlo.multiply %v3432, %v3417 : tensor<32x384x14x14xf32>
    %v3439 = stablehlo.subtract %v3438, %v3434 : tensor<32x384x14x14xf32>
    %v3440 = stablehlo.multiply %v3429, %v3437 : tensor<32x384x14x14xf32>
    %v3441 = stablehlo.subtract %v3439, %v3440 : tensor<32x384x14x14xf32>
    %v3442 = stablehlo.divide %v3428, %v3417 : tensor<32x384x14x14xf32>
    %v3443 = stablehlo.multiply %v3442, %v3441 : tensor<32x384x14x14xf32>
    %v3444 = stablehlo.reshape %v3443 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3445 = stablehlo.reshape %v3444 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3446 = stablehlo.reverse %b8eW, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3447 = stablehlo.transpose %v3446, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3448 = stablehlo.convolution(%v3445, %v3447)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3449 = stablehlo.reshape %v3448 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3450 = stablehlo.add %v3449, %v3252 : tensor<32x12544xf32>
    %v3451 = stablehlo.reshape %v583 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3452 = stablehlo.reshape %v3444 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3453 = stablehlo.transpose %v3451, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3454 = stablehlo.transpose %v3452, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3455 = stablehlo.convolution(%v3453, %v3454)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3456 = stablehlo.transpose %v3455, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3457 = stablehlo.reshape %v588 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3459 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3460 = stablehlo.reduce(%v3457 init: %v3458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3461 = stablehlo.broadcast_in_dim %v3460, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3462 = stablehlo.divide %v3461, %v3459 : tensor<32x384x14x14xf32>
    %v3463 = stablehlo.subtract %v3457, %v3462 : tensor<32x384x14x14xf32>
    %v3464 = stablehlo.multiply %v3463, %v3463 : tensor<32x384x14x14xf32>
    %v3465 = stablehlo.reduce(%v3464 init: %v3458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3466 = stablehlo.broadcast_in_dim %v3465, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3467 = stablehlo.divide %v3466, %v3459 : tensor<32x384x14x14xf32>
    %v3468 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3469 = stablehlo.add %v3467, %v3468 : tensor<32x384x14x14xf32>
    %v3470 = stablehlo.rsqrt %v3469 : tensor<32x384x14x14xf32>
    %v3471 = stablehlo.multiply %v3463, %v3470 : tensor<32x384x14x14xf32>
    %v3472 = stablehlo.reshape %v3414 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3473 = stablehlo.multiply %v3472, %v3471 : tensor<32x384x14x14xf32>
    %v3474 = stablehlo.reduce(%v3473 init: %v3458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3475 = stablehlo.reshape %v3414 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3476 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3477 = stablehlo.reduce(%v3475 init: %v3476) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3478 = stablehlo.reshape %v612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3479 = stablehlo.reshape %v3404 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3480 = stablehlo.transpose %v3478, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3481 = stablehlo.transpose %v3479, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3482 = stablehlo.convolution(%v3480, %v3481)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3483 = stablehlo.reshape %v3482 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3484 = stablehlo.reshape %v617 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3485 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3486 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3487 = stablehlo.reduce(%v3484 init: %v3485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3488 = stablehlo.broadcast_in_dim %v3487, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3489 = stablehlo.divide %v3488, %v3486 : tensor<32x384x14x14xf32>
    %v3490 = stablehlo.subtract %v3484, %v3489 : tensor<32x384x14x14xf32>
    %v3491 = stablehlo.multiply %v3490, %v3490 : tensor<32x384x14x14xf32>
    %v3492 = stablehlo.reduce(%v3491 init: %v3485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3493 = stablehlo.broadcast_in_dim %v3492, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3494 = stablehlo.divide %v3493, %v3486 : tensor<32x384x14x14xf32>
    %v3495 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3496 = stablehlo.add %v3494, %v3495 : tensor<32x384x14x14xf32>
    %v3497 = stablehlo.rsqrt %v3496 : tensor<32x384x14x14xf32>
    %v3498 = stablehlo.multiply %v3490, %v3497 : tensor<32x384x14x14xf32>
    %v3499 = stablehlo.reshape %v3374 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3500 = stablehlo.multiply %v3499, %v3498 : tensor<32x384x14x14xf32>
    %v3501 = stablehlo.reduce(%v3500 init: %v3485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3502 = stablehlo.reshape %v3374 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3504 = stablehlo.reduce(%v3502 init: %v3503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3505 = stablehlo.reshape %v641 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3506 = stablehlo.reshape %v3363 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3507 = stablehlo.transpose %v3505, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3508 = stablehlo.transpose %v3506, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3509 = stablehlo.convolution(%v3507, %v3508)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3510 = stablehlo.transpose %v3509, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3511 = stablehlo.reshape %v646 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3512 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3513 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3514 = stablehlo.reduce(%v3511 init: %v3512) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3515 = stablehlo.broadcast_in_dim %v3514, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3516 = stablehlo.divide %v3515, %v3513 : tensor<32x64x14x14xf32>
    %v3517 = stablehlo.subtract %v3511, %v3516 : tensor<32x64x14x14xf32>
    %v3518 = stablehlo.multiply %v3517, %v3517 : tensor<32x64x14x14xf32>
    %v3519 = stablehlo.reduce(%v3518 init: %v3512) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3520 = stablehlo.broadcast_in_dim %v3519, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3521 = stablehlo.divide %v3520, %v3513 : tensor<32x64x14x14xf32>
    %v3522 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3523 = stablehlo.add %v3521, %v3522 : tensor<32x64x14x14xf32>
    %v3524 = stablehlo.rsqrt %v3523 : tensor<32x64x14x14xf32>
    %v3525 = stablehlo.multiply %v3517, %v3524 : tensor<32x64x14x14xf32>
    %v3526 = stablehlo.reshape %v3252 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3527 = stablehlo.multiply %v3526, %v3525 : tensor<32x64x14x14xf32>
    %v3528 = stablehlo.reduce(%v3527 init: %v3512) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3529 = stablehlo.reshape %v3252 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3531 = stablehlo.reduce(%v3529 init: %v3530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3532 = stablehlo.reshape %v563 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3534 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3535 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3536 = stablehlo.reduce(%v3532 init: %v3533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3537 = stablehlo.broadcast_in_dim %v3536, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3538 = stablehlo.divide %v3537, %v3534 : tensor<32x64x14x14xf32>
    %v3539 = stablehlo.subtract %v3532, %v3538 : tensor<32x64x14x14xf32>
    %v3540 = stablehlo.multiply %v3539, %v3539 : tensor<32x64x14x14xf32>
    %v3541 = stablehlo.reduce(%v3540 init: %v3533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3542 = stablehlo.broadcast_in_dim %v3541, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3543 = stablehlo.divide %v3542, %v3534 : tensor<32x64x14x14xf32>
    %v3544 = stablehlo.add %v3543, %v3535 : tensor<32x64x14x14xf32>
    %v3545 = stablehlo.rsqrt %v3544 : tensor<32x64x14x14xf32>
    %v3546 = stablehlo.multiply %v3539, %v3545 : tensor<32x64x14x14xf32>
    %v3547 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3548 = stablehlo.reshape %v3450 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3549 = stablehlo.multiply %v3547, %v3548 : tensor<32x64x14x14xf32>
    %v3550 = stablehlo.reduce(%v3549 init: %v3533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3551 = stablehlo.broadcast_in_dim %v3550, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3552 = stablehlo.multiply %v3546, %v3549 : tensor<32x64x14x14xf32>
    %v3553 = stablehlo.reduce(%v3552 init: %v3533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3554 = stablehlo.broadcast_in_dim %v3553, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3555 = stablehlo.multiply %v3549, %v3534 : tensor<32x64x14x14xf32>
    %v3556 = stablehlo.subtract %v3555, %v3551 : tensor<32x64x14x14xf32>
    %v3557 = stablehlo.multiply %v3546, %v3554 : tensor<32x64x14x14xf32>
    %v3558 = stablehlo.subtract %v3556, %v3557 : tensor<32x64x14x14xf32>
    %v3559 = stablehlo.divide %v3545, %v3534 : tensor<32x64x14x14xf32>
    %v3560 = stablehlo.multiply %v3559, %v3558 : tensor<32x64x14x14xf32>
    %v3561 = stablehlo.reshape %v3560 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3562 = stablehlo.reshape %v3561 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3563 = stablehlo.reverse %b7pW, dims = [2, 3] : tensor<64x192x1x1xf32>
    %v3564 = stablehlo.transpose %v3563, dims = [1, 0, 2, 3] : (tensor<64x192x1x1xf32>) -> tensor<192x64x1x1xf32>
    %v3565 = stablehlo.convolution(%v3562, %v3564)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<192x64x1x1xf32>) -> tensor<32x192x14x14xf32>
    %v3566 = stablehlo.reshape %v3565 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v3567 = stablehlo.constant dense<0.0> : tensor<32x37632xf32>
    %v3568 = stablehlo.constant dense<6.0> : tensor<32x37632xf32>
    %v3569 = stablehlo.compare GT, %v554, %v3567 : (tensor<32x37632xf32>, tensor<32x37632xf32>) -> tensor<32x37632xi1>
    %v3570 = stablehlo.compare LT, %v554, %v3568 : (tensor<32x37632xf32>, tensor<32x37632xf32>) -> tensor<32x37632xi1>
    %v3571 = stablehlo.and %v3569, %v3570 : tensor<32x37632xi1>
    %v3572 = stablehlo.select %v3571, %v3566, %v3567 : tensor<32x37632xi1>, tensor<32x37632xf32>
    %v3573 = stablehlo.reshape %v534 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3574 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3575 = stablehlo.constant dense<6272.0> : tensor<32x192x14x14xf32>
    %v3576 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v3577 = stablehlo.reduce(%v3573 init: %v3574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3578 = stablehlo.broadcast_in_dim %v3577, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3579 = stablehlo.divide %v3578, %v3575 : tensor<32x192x14x14xf32>
    %v3580 = stablehlo.subtract %v3573, %v3579 : tensor<32x192x14x14xf32>
    %v3581 = stablehlo.multiply %v3580, %v3580 : tensor<32x192x14x14xf32>
    %v3582 = stablehlo.reduce(%v3581 init: %v3574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3583 = stablehlo.broadcast_in_dim %v3582, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3584 = stablehlo.divide %v3583, %v3575 : tensor<32x192x14x14xf32>
    %v3585 = stablehlo.add %v3584, %v3576 : tensor<32x192x14x14xf32>
    %v3586 = stablehlo.rsqrt %v3585 : tensor<32x192x14x14xf32>
    %v3587 = stablehlo.multiply %v3580, %v3586 : tensor<32x192x14x14xf32>
    %v3588 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3589 = stablehlo.reshape %v3572 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3590 = stablehlo.multiply %v3588, %v3589 : tensor<32x192x14x14xf32>
    %v3591 = stablehlo.reduce(%v3590 init: %v3574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3592 = stablehlo.broadcast_in_dim %v3591, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3593 = stablehlo.multiply %v3587, %v3590 : tensor<32x192x14x14xf32>
    %v3594 = stablehlo.reduce(%v3593 init: %v3574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3595 = stablehlo.broadcast_in_dim %v3594, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3596 = stablehlo.multiply %v3590, %v3575 : tensor<32x192x14x14xf32>
    %v3597 = stablehlo.subtract %v3596, %v3592 : tensor<32x192x14x14xf32>
    %v3598 = stablehlo.multiply %v3587, %v3595 : tensor<32x192x14x14xf32>
    %v3599 = stablehlo.subtract %v3597, %v3598 : tensor<32x192x14x14xf32>
    %v3600 = stablehlo.divide %v3586, %v3575 : tensor<32x192x14x14xf32>
    %v3601 = stablehlo.multiply %v3600, %v3599 : tensor<32x192x14x14xf32>
    %v3602 = stablehlo.reshape %v3601 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v3603 = stablehlo.reshape %v3602 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3604 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3605 = stablehlo.pad %v3603, %v3604, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192x28x28xf32>
    %v3606 = stablehlo.reverse %b7dW, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v3607 = stablehlo.convolution(%v3605, %v3606)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v3608 = stablehlo.reshape %v3607 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3609 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v3610 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v3611 = stablehlo.compare GT, %v525, %v3609 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v3612 = stablehlo.compare LT, %v525, %v3610 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v3613 = stablehlo.and %v3611, %v3612 : tensor<32x150528xi1>
    %v3614 = stablehlo.select %v3613, %v3608, %v3609 : tensor<32x150528xi1>, tensor<32x150528xf32>
    %v3615 = stablehlo.reshape %v505 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3616 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3617 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v3618 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v3619 = stablehlo.reduce(%v3615 init: %v3616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3620 = stablehlo.broadcast_in_dim %v3619, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3621 = stablehlo.divide %v3620, %v3617 : tensor<32x192x28x28xf32>
    %v3622 = stablehlo.subtract %v3615, %v3621 : tensor<32x192x28x28xf32>
    %v3623 = stablehlo.multiply %v3622, %v3622 : tensor<32x192x28x28xf32>
    %v3624 = stablehlo.reduce(%v3623 init: %v3616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3625 = stablehlo.broadcast_in_dim %v3624, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3626 = stablehlo.divide %v3625, %v3617 : tensor<32x192x28x28xf32>
    %v3627 = stablehlo.add %v3626, %v3618 : tensor<32x192x28x28xf32>
    %v3628 = stablehlo.rsqrt %v3627 : tensor<32x192x28x28xf32>
    %v3629 = stablehlo.multiply %v3622, %v3628 : tensor<32x192x28x28xf32>
    %v3630 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3631 = stablehlo.reshape %v3614 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3632 = stablehlo.multiply %v3630, %v3631 : tensor<32x192x28x28xf32>
    %v3633 = stablehlo.reduce(%v3632 init: %v3616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3634 = stablehlo.broadcast_in_dim %v3633, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3635 = stablehlo.multiply %v3629, %v3632 : tensor<32x192x28x28xf32>
    %v3636 = stablehlo.reduce(%v3635 init: %v3616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3637 = stablehlo.broadcast_in_dim %v3636, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3638 = stablehlo.multiply %v3632, %v3617 : tensor<32x192x28x28xf32>
    %v3639 = stablehlo.subtract %v3638, %v3634 : tensor<32x192x28x28xf32>
    %v3640 = stablehlo.multiply %v3629, %v3637 : tensor<32x192x28x28xf32>
    %v3641 = stablehlo.subtract %v3639, %v3640 : tensor<32x192x28x28xf32>
    %v3642 = stablehlo.divide %v3628, %v3617 : tensor<32x192x28x28xf32>
    %v3643 = stablehlo.multiply %v3642, %v3641 : tensor<32x192x28x28xf32>
    %v3644 = stablehlo.reshape %v3643 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3645 = stablehlo.reshape %v3644 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3646 = stablehlo.reverse %b7eW, dims = [2, 3] : tensor<192x32x1x1xf32>
    %v3647 = stablehlo.transpose %v3646, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v3648 = stablehlo.convolution(%v3645, %v3647)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v3649 = stablehlo.reshape %v3648 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v3650 = stablehlo.reshape %v500 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3651 = stablehlo.reshape %v3644 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3652 = stablehlo.transpose %v3650, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v3653 = stablehlo.transpose %v3651, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3654 = stablehlo.convolution(%v3652, %v3653)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v3655 = stablehlo.transpose %v3654, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v3656 = stablehlo.reshape %v505 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3657 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3658 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v3659 = stablehlo.reduce(%v3656 init: %v3657) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3660 = stablehlo.broadcast_in_dim %v3659, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3661 = stablehlo.divide %v3660, %v3658 : tensor<32x192x28x28xf32>
    %v3662 = stablehlo.subtract %v3656, %v3661 : tensor<32x192x28x28xf32>
    %v3663 = stablehlo.multiply %v3662, %v3662 : tensor<32x192x28x28xf32>
    %v3664 = stablehlo.reduce(%v3663 init: %v3657) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3665 = stablehlo.broadcast_in_dim %v3664, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3666 = stablehlo.divide %v3665, %v3658 : tensor<32x192x28x28xf32>
    %v3667 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v3668 = stablehlo.add %v3666, %v3667 : tensor<32x192x28x28xf32>
    %v3669 = stablehlo.rsqrt %v3668 : tensor<32x192x28x28xf32>
    %v3670 = stablehlo.multiply %v3662, %v3669 : tensor<32x192x28x28xf32>
    %v3671 = stablehlo.reshape %v3614 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3672 = stablehlo.multiply %v3671, %v3670 : tensor<32x192x28x28xf32>
    %v3673 = stablehlo.reduce(%v3672 init: %v3657) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3674 = stablehlo.reshape %v3614 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3676 = stablehlo.reduce(%v3674 init: %v3675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3677 = stablehlo.reshape %v529 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3678 = stablehlo.reshape %v3602 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3679 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3680 = stablehlo.pad %v3678, %v3679, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192x28x28xf32>
    %v3681 = stablehlo.transpose %v3677, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3682 = stablehlo.transpose %v3680, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3683 = stablehlo.convolution(%v3681, %v3682)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v3684 = stablehlo.reshape %v3683 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v3685 = stablehlo.reshape %v534 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3687 = stablehlo.constant dense<6272.0> : tensor<32x192x14x14xf32>
    %v3688 = stablehlo.reduce(%v3685 init: %v3686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3689 = stablehlo.broadcast_in_dim %v3688, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3690 = stablehlo.divide %v3689, %v3687 : tensor<32x192x14x14xf32>
    %v3691 = stablehlo.subtract %v3685, %v3690 : tensor<32x192x14x14xf32>
    %v3692 = stablehlo.multiply %v3691, %v3691 : tensor<32x192x14x14xf32>
    %v3693 = stablehlo.reduce(%v3692 init: %v3686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3694 = stablehlo.broadcast_in_dim %v3693, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3695 = stablehlo.divide %v3694, %v3687 : tensor<32x192x14x14xf32>
    %v3696 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v3697 = stablehlo.add %v3695, %v3696 : tensor<32x192x14x14xf32>
    %v3698 = stablehlo.rsqrt %v3697 : tensor<32x192x14x14xf32>
    %v3699 = stablehlo.multiply %v3691, %v3698 : tensor<32x192x14x14xf32>
    %v3700 = stablehlo.reshape %v3572 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3701 = stablehlo.multiply %v3700, %v3699 : tensor<32x192x14x14xf32>
    %v3702 = stablehlo.reduce(%v3701 init: %v3686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3703 = stablehlo.reshape %v3572 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3704 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3705 = stablehlo.reduce(%v3703 init: %v3704) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3706 = stablehlo.reshape %v558 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3707 = stablehlo.reshape %v3561 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3708 = stablehlo.transpose %v3706, dims = [1, 0, 2, 3] : (tensor<32x192x14x14xf32>) -> tensor<192x32x14x14xf32>
    %v3709 = stablehlo.transpose %v3707, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3710 = stablehlo.convolution(%v3708, %v3709)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<192x64x1x1xf32>
    %v3711 = stablehlo.transpose %v3710, dims = [1, 0, 2, 3] : (tensor<192x64x1x1xf32>) -> tensor<64x192x1x1xf32>
    %v3712 = stablehlo.reshape %v563 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3714 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3715 = stablehlo.reduce(%v3712 init: %v3713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3716 = stablehlo.broadcast_in_dim %v3715, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3717 = stablehlo.divide %v3716, %v3714 : tensor<32x64x14x14xf32>
    %v3718 = stablehlo.subtract %v3712, %v3717 : tensor<32x64x14x14xf32>
    %v3719 = stablehlo.multiply %v3718, %v3718 : tensor<32x64x14x14xf32>
    %v3720 = stablehlo.reduce(%v3719 init: %v3713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3721 = stablehlo.broadcast_in_dim %v3720, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3722 = stablehlo.divide %v3721, %v3714 : tensor<32x64x14x14xf32>
    %v3723 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3724 = stablehlo.add %v3722, %v3723 : tensor<32x64x14x14xf32>
    %v3725 = stablehlo.rsqrt %v3724 : tensor<32x64x14x14xf32>
    %v3726 = stablehlo.multiply %v3718, %v3725 : tensor<32x64x14x14xf32>
    %v3727 = stablehlo.reshape %v3450 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3728 = stablehlo.multiply %v3727, %v3726 : tensor<32x64x14x14xf32>
    %v3729 = stablehlo.reduce(%v3728 init: %v3713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3730 = stablehlo.reshape %v3450 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3731 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3732 = stablehlo.reduce(%v3730 init: %v3731) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3733 = stablehlo.reshape %v479 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3735 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v3736 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v3737 = stablehlo.reduce(%v3733 init: %v3734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3738 = stablehlo.broadcast_in_dim %v3737, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3739 = stablehlo.divide %v3738, %v3735 : tensor<32x32x28x28xf32>
    %v3740 = stablehlo.subtract %v3733, %v3739 : tensor<32x32x28x28xf32>
    %v3741 = stablehlo.multiply %v3740, %v3740 : tensor<32x32x28x28xf32>
    %v3742 = stablehlo.reduce(%v3741 init: %v3734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3743 = stablehlo.broadcast_in_dim %v3742, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3744 = stablehlo.divide %v3743, %v3735 : tensor<32x32x28x28xf32>
    %v3745 = stablehlo.add %v3744, %v3736 : tensor<32x32x28x28xf32>
    %v3746 = stablehlo.rsqrt %v3745 : tensor<32x32x28x28xf32>
    %v3747 = stablehlo.multiply %v3740, %v3746 : tensor<32x32x28x28xf32>
    %v3748 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3749 = stablehlo.reshape %v3649 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3750 = stablehlo.multiply %v3748, %v3749 : tensor<32x32x28x28xf32>
    %v3751 = stablehlo.reduce(%v3750 init: %v3734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3752 = stablehlo.broadcast_in_dim %v3751, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3753 = stablehlo.multiply %v3747, %v3750 : tensor<32x32x28x28xf32>
    %v3754 = stablehlo.reduce(%v3753 init: %v3734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3755 = stablehlo.broadcast_in_dim %v3754, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3756 = stablehlo.multiply %v3750, %v3735 : tensor<32x32x28x28xf32>
    %v3757 = stablehlo.subtract %v3756, %v3752 : tensor<32x32x28x28xf32>
    %v3758 = stablehlo.multiply %v3747, %v3755 : tensor<32x32x28x28xf32>
    %v3759 = stablehlo.subtract %v3757, %v3758 : tensor<32x32x28x28xf32>
    %v3760 = stablehlo.divide %v3746, %v3735 : tensor<32x32x28x28xf32>
    %v3761 = stablehlo.multiply %v3760, %v3759 : tensor<32x32x28x28xf32>
    %v3762 = stablehlo.reshape %v3761 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v3763 = stablehlo.reshape %v3762 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3764 = stablehlo.reverse %b6pW, dims = [2, 3] : tensor<32x192x1x1xf32>
    %v3765 = stablehlo.transpose %v3764, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v3766 = stablehlo.convolution(%v3763, %v3765)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3767 = stablehlo.reshape %v3766 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3768 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v3769 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v3770 = stablehlo.compare GT, %v470, %v3768 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v3771 = stablehlo.compare LT, %v470, %v3769 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v3772 = stablehlo.and %v3770, %v3771 : tensor<32x150528xi1>
    %v3773 = stablehlo.select %v3772, %v3767, %v3768 : tensor<32x150528xi1>, tensor<32x150528xf32>
    %v3774 = stablehlo.reshape %v450 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3775 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3776 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v3777 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v3778 = stablehlo.reduce(%v3774 init: %v3775) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3779 = stablehlo.broadcast_in_dim %v3778, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3780 = stablehlo.divide %v3779, %v3776 : tensor<32x192x28x28xf32>
    %v3781 = stablehlo.subtract %v3774, %v3780 : tensor<32x192x28x28xf32>
    %v3782 = stablehlo.multiply %v3781, %v3781 : tensor<32x192x28x28xf32>
    %v3783 = stablehlo.reduce(%v3782 init: %v3775) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3784 = stablehlo.broadcast_in_dim %v3783, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3785 = stablehlo.divide %v3784, %v3776 : tensor<32x192x28x28xf32>
    %v3786 = stablehlo.add %v3785, %v3777 : tensor<32x192x28x28xf32>
    %v3787 = stablehlo.rsqrt %v3786 : tensor<32x192x28x28xf32>
    %v3788 = stablehlo.multiply %v3781, %v3787 : tensor<32x192x28x28xf32>
    %v3789 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3790 = stablehlo.reshape %v3773 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3791 = stablehlo.multiply %v3789, %v3790 : tensor<32x192x28x28xf32>
    %v3792 = stablehlo.reduce(%v3791 init: %v3775) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3793 = stablehlo.broadcast_in_dim %v3792, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3794 = stablehlo.multiply %v3788, %v3791 : tensor<32x192x28x28xf32>
    %v3795 = stablehlo.reduce(%v3794 init: %v3775) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3796 = stablehlo.broadcast_in_dim %v3795, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3797 = stablehlo.multiply %v3791, %v3776 : tensor<32x192x28x28xf32>
    %v3798 = stablehlo.subtract %v3797, %v3793 : tensor<32x192x28x28xf32>
    %v3799 = stablehlo.multiply %v3788, %v3796 : tensor<32x192x28x28xf32>
    %v3800 = stablehlo.subtract %v3798, %v3799 : tensor<32x192x28x28xf32>
    %v3801 = stablehlo.divide %v3787, %v3776 : tensor<32x192x28x28xf32>
    %v3802 = stablehlo.multiply %v3801, %v3800 : tensor<32x192x28x28xf32>
    %v3803 = stablehlo.reshape %v3802 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3804 = stablehlo.reshape %v3803 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3805 = stablehlo.reverse %b6dW, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v3806 = stablehlo.convolution(%v3804, %v3805)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v3807 = stablehlo.reshape %v3806 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3808 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v3809 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v3810 = stablehlo.compare GT, %v441, %v3808 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v3811 = stablehlo.compare LT, %v441, %v3809 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v3812 = stablehlo.and %v3810, %v3811 : tensor<32x150528xi1>
    %v3813 = stablehlo.select %v3812, %v3807, %v3808 : tensor<32x150528xi1>, tensor<32x150528xf32>
    %v3814 = stablehlo.reshape %v421 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3815 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3816 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v3817 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v3818 = stablehlo.reduce(%v3814 init: %v3815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3819 = stablehlo.broadcast_in_dim %v3818, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3820 = stablehlo.divide %v3819, %v3816 : tensor<32x192x28x28xf32>
    %v3821 = stablehlo.subtract %v3814, %v3820 : tensor<32x192x28x28xf32>
    %v3822 = stablehlo.multiply %v3821, %v3821 : tensor<32x192x28x28xf32>
    %v3823 = stablehlo.reduce(%v3822 init: %v3815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3824 = stablehlo.broadcast_in_dim %v3823, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3825 = stablehlo.divide %v3824, %v3816 : tensor<32x192x28x28xf32>
    %v3826 = stablehlo.add %v3825, %v3817 : tensor<32x192x28x28xf32>
    %v3827 = stablehlo.rsqrt %v3826 : tensor<32x192x28x28xf32>
    %v3828 = stablehlo.multiply %v3821, %v3827 : tensor<32x192x28x28xf32>
    %v3829 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3830 = stablehlo.reshape %v3813 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3831 = stablehlo.multiply %v3829, %v3830 : tensor<32x192x28x28xf32>
    %v3832 = stablehlo.reduce(%v3831 init: %v3815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3833 = stablehlo.broadcast_in_dim %v3832, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3834 = stablehlo.multiply %v3828, %v3831 : tensor<32x192x28x28xf32>
    %v3835 = stablehlo.reduce(%v3834 init: %v3815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3836 = stablehlo.broadcast_in_dim %v3835, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3837 = stablehlo.multiply %v3831, %v3816 : tensor<32x192x28x28xf32>
    %v3838 = stablehlo.subtract %v3837, %v3833 : tensor<32x192x28x28xf32>
    %v3839 = stablehlo.multiply %v3828, %v3836 : tensor<32x192x28x28xf32>
    %v3840 = stablehlo.subtract %v3838, %v3839 : tensor<32x192x28x28xf32>
    %v3841 = stablehlo.divide %v3827, %v3816 : tensor<32x192x28x28xf32>
    %v3842 = stablehlo.multiply %v3841, %v3840 : tensor<32x192x28x28xf32>
    %v3843 = stablehlo.reshape %v3842 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3844 = stablehlo.reshape %v3843 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3845 = stablehlo.reverse %b6eW, dims = [2, 3] : tensor<192x32x1x1xf32>
    %v3846 = stablehlo.transpose %v3845, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v3847 = stablehlo.convolution(%v3844, %v3846)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v3848 = stablehlo.reshape %v3847 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v3849 = stablehlo.add %v3848, %v3649 : tensor<32x25088xf32>
    %v3850 = stablehlo.reshape %v416 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3851 = stablehlo.reshape %v3843 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3852 = stablehlo.transpose %v3850, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v3853 = stablehlo.transpose %v3851, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3854 = stablehlo.convolution(%v3852, %v3853)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v3855 = stablehlo.transpose %v3854, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v3856 = stablehlo.reshape %v421 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3857 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3858 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v3859 = stablehlo.reduce(%v3856 init: %v3857) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3860 = stablehlo.broadcast_in_dim %v3859, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3861 = stablehlo.divide %v3860, %v3858 : tensor<32x192x28x28xf32>
    %v3862 = stablehlo.subtract %v3856, %v3861 : tensor<32x192x28x28xf32>
    %v3863 = stablehlo.multiply %v3862, %v3862 : tensor<32x192x28x28xf32>
    %v3864 = stablehlo.reduce(%v3863 init: %v3857) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3865 = stablehlo.broadcast_in_dim %v3864, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3866 = stablehlo.divide %v3865, %v3858 : tensor<32x192x28x28xf32>
    %v3867 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v3868 = stablehlo.add %v3866, %v3867 : tensor<32x192x28x28xf32>
    %v3869 = stablehlo.rsqrt %v3868 : tensor<32x192x28x28xf32>
    %v3870 = stablehlo.multiply %v3862, %v3869 : tensor<32x192x28x28xf32>
    %v3871 = stablehlo.reshape %v3813 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3872 = stablehlo.multiply %v3871, %v3870 : tensor<32x192x28x28xf32>
    %v3873 = stablehlo.reduce(%v3872 init: %v3857) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3874 = stablehlo.reshape %v3813 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3875 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3876 = stablehlo.reduce(%v3874 init: %v3875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3877 = stablehlo.reshape %v445 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3878 = stablehlo.reshape %v3803 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3879 = stablehlo.transpose %v3877, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3880 = stablehlo.transpose %v3878, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3881 = stablehlo.convolution(%v3879, %v3880)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v3882 = stablehlo.reshape %v3881 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v3883 = stablehlo.reshape %v450 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3884 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3885 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v3886 = stablehlo.reduce(%v3883 init: %v3884) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3887 = stablehlo.broadcast_in_dim %v3886, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3888 = stablehlo.divide %v3887, %v3885 : tensor<32x192x28x28xf32>
    %v3889 = stablehlo.subtract %v3883, %v3888 : tensor<32x192x28x28xf32>
    %v3890 = stablehlo.multiply %v3889, %v3889 : tensor<32x192x28x28xf32>
    %v3891 = stablehlo.reduce(%v3890 init: %v3884) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3892 = stablehlo.broadcast_in_dim %v3891, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3893 = stablehlo.divide %v3892, %v3885 : tensor<32x192x28x28xf32>
    %v3894 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v3895 = stablehlo.add %v3893, %v3894 : tensor<32x192x28x28xf32>
    %v3896 = stablehlo.rsqrt %v3895 : tensor<32x192x28x28xf32>
    %v3897 = stablehlo.multiply %v3889, %v3896 : tensor<32x192x28x28xf32>
    %v3898 = stablehlo.reshape %v3773 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3899 = stablehlo.multiply %v3898, %v3897 : tensor<32x192x28x28xf32>
    %v3900 = stablehlo.reduce(%v3899 init: %v3884) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3901 = stablehlo.reshape %v3773 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3902 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3903 = stablehlo.reduce(%v3901 init: %v3902) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3904 = stablehlo.reshape %v474 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3905 = stablehlo.reshape %v3762 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3906 = stablehlo.transpose %v3904, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3907 = stablehlo.transpose %v3905, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v3908 = stablehlo.convolution(%v3906, %v3907)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<192x32x1x1xf32>
    %v3909 = stablehlo.transpose %v3908, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v3910 = stablehlo.reshape %v479 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3912 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v3913 = stablehlo.reduce(%v3910 init: %v3911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3914 = stablehlo.broadcast_in_dim %v3913, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3915 = stablehlo.divide %v3914, %v3912 : tensor<32x32x28x28xf32>
    %v3916 = stablehlo.subtract %v3910, %v3915 : tensor<32x32x28x28xf32>
    %v3917 = stablehlo.multiply %v3916, %v3916 : tensor<32x32x28x28xf32>
    %v3918 = stablehlo.reduce(%v3917 init: %v3911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3919 = stablehlo.broadcast_in_dim %v3918, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3920 = stablehlo.divide %v3919, %v3912 : tensor<32x32x28x28xf32>
    %v3921 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v3922 = stablehlo.add %v3920, %v3921 : tensor<32x32x28x28xf32>
    %v3923 = stablehlo.rsqrt %v3922 : tensor<32x32x28x28xf32>
    %v3924 = stablehlo.multiply %v3916, %v3923 : tensor<32x32x28x28xf32>
    %v3925 = stablehlo.reshape %v3649 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3926 = stablehlo.multiply %v3925, %v3924 : tensor<32x32x28x28xf32>
    %v3927 = stablehlo.reduce(%v3926 init: %v3911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3928 = stablehlo.reshape %v3649 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3929 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3930 = stablehlo.reduce(%v3928 init: %v3929) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3931 = stablehlo.reshape %v395 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3933 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v3934 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v3935 = stablehlo.reduce(%v3931 init: %v3932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3936 = stablehlo.broadcast_in_dim %v3935, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3937 = stablehlo.divide %v3936, %v3933 : tensor<32x32x28x28xf32>
    %v3938 = stablehlo.subtract %v3931, %v3937 : tensor<32x32x28x28xf32>
    %v3939 = stablehlo.multiply %v3938, %v3938 : tensor<32x32x28x28xf32>
    %v3940 = stablehlo.reduce(%v3939 init: %v3932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3941 = stablehlo.broadcast_in_dim %v3940, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3942 = stablehlo.divide %v3941, %v3933 : tensor<32x32x28x28xf32>
    %v3943 = stablehlo.add %v3942, %v3934 : tensor<32x32x28x28xf32>
    %v3944 = stablehlo.rsqrt %v3943 : tensor<32x32x28x28xf32>
    %v3945 = stablehlo.multiply %v3938, %v3944 : tensor<32x32x28x28xf32>
    %v3946 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3947 = stablehlo.reshape %v3849 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3948 = stablehlo.multiply %v3946, %v3947 : tensor<32x32x28x28xf32>
    %v3949 = stablehlo.reduce(%v3948 init: %v3932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3950 = stablehlo.broadcast_in_dim %v3949, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3951 = stablehlo.multiply %v3945, %v3948 : tensor<32x32x28x28xf32>
    %v3952 = stablehlo.reduce(%v3951 init: %v3932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3953 = stablehlo.broadcast_in_dim %v3952, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3954 = stablehlo.multiply %v3948, %v3933 : tensor<32x32x28x28xf32>
    %v3955 = stablehlo.subtract %v3954, %v3950 : tensor<32x32x28x28xf32>
    %v3956 = stablehlo.multiply %v3945, %v3953 : tensor<32x32x28x28xf32>
    %v3957 = stablehlo.subtract %v3955, %v3956 : tensor<32x32x28x28xf32>
    %v3958 = stablehlo.divide %v3944, %v3933 : tensor<32x32x28x28xf32>
    %v3959 = stablehlo.multiply %v3958, %v3957 : tensor<32x32x28x28xf32>
    %v3960 = stablehlo.reshape %v3959 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v3961 = stablehlo.reshape %v3960 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3962 = stablehlo.reverse %b5pW, dims = [2, 3] : tensor<32x192x1x1xf32>
    %v3963 = stablehlo.transpose %v3962, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v3964 = stablehlo.convolution(%v3961, %v3963)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3965 = stablehlo.reshape %v3964 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3966 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v3967 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v3968 = stablehlo.compare GT, %v386, %v3966 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v3969 = stablehlo.compare LT, %v386, %v3967 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v3970 = stablehlo.and %v3968, %v3969 : tensor<32x150528xi1>
    %v3971 = stablehlo.select %v3970, %v3965, %v3966 : tensor<32x150528xi1>, tensor<32x150528xf32>
    %v3972 = stablehlo.reshape %v366 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3973 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3974 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v3975 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v3976 = stablehlo.reduce(%v3972 init: %v3973) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3977 = stablehlo.broadcast_in_dim %v3976, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3978 = stablehlo.divide %v3977, %v3974 : tensor<32x192x28x28xf32>
    %v3979 = stablehlo.subtract %v3972, %v3978 : tensor<32x192x28x28xf32>
    %v3980 = stablehlo.multiply %v3979, %v3979 : tensor<32x192x28x28xf32>
    %v3981 = stablehlo.reduce(%v3980 init: %v3973) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3982 = stablehlo.broadcast_in_dim %v3981, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3983 = stablehlo.divide %v3982, %v3974 : tensor<32x192x28x28xf32>
    %v3984 = stablehlo.add %v3983, %v3975 : tensor<32x192x28x28xf32>
    %v3985 = stablehlo.rsqrt %v3984 : tensor<32x192x28x28xf32>
    %v3986 = stablehlo.multiply %v3979, %v3985 : tensor<32x192x28x28xf32>
    %v3987 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3988 = stablehlo.reshape %v3971 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3989 = stablehlo.multiply %v3987, %v3988 : tensor<32x192x28x28xf32>
    %v3990 = stablehlo.reduce(%v3989 init: %v3973) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3991 = stablehlo.broadcast_in_dim %v3990, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3992 = stablehlo.multiply %v3986, %v3989 : tensor<32x192x28x28xf32>
    %v3993 = stablehlo.reduce(%v3992 init: %v3973) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3994 = stablehlo.broadcast_in_dim %v3993, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3995 = stablehlo.multiply %v3989, %v3974 : tensor<32x192x28x28xf32>
    %v3996 = stablehlo.subtract %v3995, %v3991 : tensor<32x192x28x28xf32>
    %v3997 = stablehlo.multiply %v3986, %v3994 : tensor<32x192x28x28xf32>
    %v3998 = stablehlo.subtract %v3996, %v3997 : tensor<32x192x28x28xf32>
    %v3999 = stablehlo.divide %v3985, %v3974 : tensor<32x192x28x28xf32>
    %v4000 = stablehlo.multiply %v3999, %v3998 : tensor<32x192x28x28xf32>
    %v4001 = stablehlo.reshape %v4000 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4002 = stablehlo.reshape %v4001 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4003 = stablehlo.reverse %b5dW, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v4004 = stablehlo.convolution(%v4002, %v4003)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v4005 = stablehlo.reshape %v4004 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4006 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v4007 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v4008 = stablehlo.compare GT, %v357, %v4006 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4009 = stablehlo.compare LT, %v357, %v4007 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4010 = stablehlo.and %v4008, %v4009 : tensor<32x150528xi1>
    %v4011 = stablehlo.select %v4010, %v4005, %v4006 : tensor<32x150528xi1>, tensor<32x150528xf32>
    %v4012 = stablehlo.reshape %v337 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4014 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4015 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4016 = stablehlo.reduce(%v4012 init: %v4013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4017 = stablehlo.broadcast_in_dim %v4016, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4018 = stablehlo.divide %v4017, %v4014 : tensor<32x192x28x28xf32>
    %v4019 = stablehlo.subtract %v4012, %v4018 : tensor<32x192x28x28xf32>
    %v4020 = stablehlo.multiply %v4019, %v4019 : tensor<32x192x28x28xf32>
    %v4021 = stablehlo.reduce(%v4020 init: %v4013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4022 = stablehlo.broadcast_in_dim %v4021, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4023 = stablehlo.divide %v4022, %v4014 : tensor<32x192x28x28xf32>
    %v4024 = stablehlo.add %v4023, %v4015 : tensor<32x192x28x28xf32>
    %v4025 = stablehlo.rsqrt %v4024 : tensor<32x192x28x28xf32>
    %v4026 = stablehlo.multiply %v4019, %v4025 : tensor<32x192x28x28xf32>
    %v4027 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4028 = stablehlo.reshape %v4011 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4029 = stablehlo.multiply %v4027, %v4028 : tensor<32x192x28x28xf32>
    %v4030 = stablehlo.reduce(%v4029 init: %v4013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4031 = stablehlo.broadcast_in_dim %v4030, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4032 = stablehlo.multiply %v4026, %v4029 : tensor<32x192x28x28xf32>
    %v4033 = stablehlo.reduce(%v4032 init: %v4013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4034 = stablehlo.broadcast_in_dim %v4033, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4035 = stablehlo.multiply %v4029, %v4014 : tensor<32x192x28x28xf32>
    %v4036 = stablehlo.subtract %v4035, %v4031 : tensor<32x192x28x28xf32>
    %v4037 = stablehlo.multiply %v4026, %v4034 : tensor<32x192x28x28xf32>
    %v4038 = stablehlo.subtract %v4036, %v4037 : tensor<32x192x28x28xf32>
    %v4039 = stablehlo.divide %v4025, %v4014 : tensor<32x192x28x28xf32>
    %v4040 = stablehlo.multiply %v4039, %v4038 : tensor<32x192x28x28xf32>
    %v4041 = stablehlo.reshape %v4040 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4042 = stablehlo.reshape %v4041 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4043 = stablehlo.reverse %b5eW, dims = [2, 3] : tensor<192x32x1x1xf32>
    %v4044 = stablehlo.transpose %v4043, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4045 = stablehlo.convolution(%v4042, %v4044)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v4046 = stablehlo.reshape %v4045 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4047 = stablehlo.add %v4046, %v3849 : tensor<32x25088xf32>
    %v4048 = stablehlo.reshape %v332 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4049 = stablehlo.reshape %v4041 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4050 = stablehlo.transpose %v4048, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4051 = stablehlo.transpose %v4049, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4052 = stablehlo.convolution(%v4050, %v4051)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v4053 = stablehlo.transpose %v4052, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4054 = stablehlo.reshape %v337 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4056 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4057 = stablehlo.reduce(%v4054 init: %v4055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4058 = stablehlo.broadcast_in_dim %v4057, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4059 = stablehlo.divide %v4058, %v4056 : tensor<32x192x28x28xf32>
    %v4060 = stablehlo.subtract %v4054, %v4059 : tensor<32x192x28x28xf32>
    %v4061 = stablehlo.multiply %v4060, %v4060 : tensor<32x192x28x28xf32>
    %v4062 = stablehlo.reduce(%v4061 init: %v4055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4063 = stablehlo.broadcast_in_dim %v4062, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4064 = stablehlo.divide %v4063, %v4056 : tensor<32x192x28x28xf32>
    %v4065 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4066 = stablehlo.add %v4064, %v4065 : tensor<32x192x28x28xf32>
    %v4067 = stablehlo.rsqrt %v4066 : tensor<32x192x28x28xf32>
    %v4068 = stablehlo.multiply %v4060, %v4067 : tensor<32x192x28x28xf32>
    %v4069 = stablehlo.reshape %v4011 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4070 = stablehlo.multiply %v4069, %v4068 : tensor<32x192x28x28xf32>
    %v4071 = stablehlo.reduce(%v4070 init: %v4055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4072 = stablehlo.reshape %v4011 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4074 = stablehlo.reduce(%v4072 init: %v4073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4075 = stablehlo.reshape %v361 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4076 = stablehlo.reshape %v4001 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4077 = stablehlo.transpose %v4075, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4078 = stablehlo.transpose %v4076, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4079 = stablehlo.convolution(%v4077, %v4078)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v4080 = stablehlo.reshape %v4079 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v4081 = stablehlo.reshape %v366 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4082 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4083 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4084 = stablehlo.reduce(%v4081 init: %v4082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4085 = stablehlo.broadcast_in_dim %v4084, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4086 = stablehlo.divide %v4085, %v4083 : tensor<32x192x28x28xf32>
    %v4087 = stablehlo.subtract %v4081, %v4086 : tensor<32x192x28x28xf32>
    %v4088 = stablehlo.multiply %v4087, %v4087 : tensor<32x192x28x28xf32>
    %v4089 = stablehlo.reduce(%v4088 init: %v4082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4090 = stablehlo.broadcast_in_dim %v4089, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4091 = stablehlo.divide %v4090, %v4083 : tensor<32x192x28x28xf32>
    %v4092 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4093 = stablehlo.add %v4091, %v4092 : tensor<32x192x28x28xf32>
    %v4094 = stablehlo.rsqrt %v4093 : tensor<32x192x28x28xf32>
    %v4095 = stablehlo.multiply %v4087, %v4094 : tensor<32x192x28x28xf32>
    %v4096 = stablehlo.reshape %v3971 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4097 = stablehlo.multiply %v4096, %v4095 : tensor<32x192x28x28xf32>
    %v4098 = stablehlo.reduce(%v4097 init: %v4082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4099 = stablehlo.reshape %v3971 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4100 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4101 = stablehlo.reduce(%v4099 init: %v4100) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4102 = stablehlo.reshape %v390 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4103 = stablehlo.reshape %v3960 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4104 = stablehlo.transpose %v4102, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4105 = stablehlo.transpose %v4103, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4106 = stablehlo.convolution(%v4104, %v4105)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<192x32x1x1xf32>
    %v4107 = stablehlo.transpose %v4106, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4108 = stablehlo.reshape %v395 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4110 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v4111 = stablehlo.reduce(%v4108 init: %v4109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4112 = stablehlo.broadcast_in_dim %v4111, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4113 = stablehlo.divide %v4112, %v4110 : tensor<32x32x28x28xf32>
    %v4114 = stablehlo.subtract %v4108, %v4113 : tensor<32x32x28x28xf32>
    %v4115 = stablehlo.multiply %v4114, %v4114 : tensor<32x32x28x28xf32>
    %v4116 = stablehlo.reduce(%v4115 init: %v4109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4117 = stablehlo.broadcast_in_dim %v4116, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4118 = stablehlo.divide %v4117, %v4110 : tensor<32x32x28x28xf32>
    %v4119 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4120 = stablehlo.add %v4118, %v4119 : tensor<32x32x28x28xf32>
    %v4121 = stablehlo.rsqrt %v4120 : tensor<32x32x28x28xf32>
    %v4122 = stablehlo.multiply %v4114, %v4121 : tensor<32x32x28x28xf32>
    %v4123 = stablehlo.reshape %v3849 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4124 = stablehlo.multiply %v4123, %v4122 : tensor<32x32x28x28xf32>
    %v4125 = stablehlo.reduce(%v4124 init: %v4109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4126 = stablehlo.reshape %v3849 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4128 = stablehlo.reduce(%v4126 init: %v4127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4129 = stablehlo.reshape %v312 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4131 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v4132 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4133 = stablehlo.reduce(%v4129 init: %v4130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4134 = stablehlo.broadcast_in_dim %v4133, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4135 = stablehlo.divide %v4134, %v4131 : tensor<32x32x28x28xf32>
    %v4136 = stablehlo.subtract %v4129, %v4135 : tensor<32x32x28x28xf32>
    %v4137 = stablehlo.multiply %v4136, %v4136 : tensor<32x32x28x28xf32>
    %v4138 = stablehlo.reduce(%v4137 init: %v4130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4139 = stablehlo.broadcast_in_dim %v4138, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4140 = stablehlo.divide %v4139, %v4131 : tensor<32x32x28x28xf32>
    %v4141 = stablehlo.add %v4140, %v4132 : tensor<32x32x28x28xf32>
    %v4142 = stablehlo.rsqrt %v4141 : tensor<32x32x28x28xf32>
    %v4143 = stablehlo.multiply %v4136, %v4142 : tensor<32x32x28x28xf32>
    %v4144 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4145 = stablehlo.reshape %v4047 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4146 = stablehlo.multiply %v4144, %v4145 : tensor<32x32x28x28xf32>
    %v4147 = stablehlo.reduce(%v4146 init: %v4130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4148 = stablehlo.broadcast_in_dim %v4147, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4149 = stablehlo.multiply %v4143, %v4146 : tensor<32x32x28x28xf32>
    %v4150 = stablehlo.reduce(%v4149 init: %v4130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4151 = stablehlo.broadcast_in_dim %v4150, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4152 = stablehlo.multiply %v4146, %v4131 : tensor<32x32x28x28xf32>
    %v4153 = stablehlo.subtract %v4152, %v4148 : tensor<32x32x28x28xf32>
    %v4154 = stablehlo.multiply %v4143, %v4151 : tensor<32x32x28x28xf32>
    %v4155 = stablehlo.subtract %v4153, %v4154 : tensor<32x32x28x28xf32>
    %v4156 = stablehlo.divide %v4142, %v4131 : tensor<32x32x28x28xf32>
    %v4157 = stablehlo.multiply %v4156, %v4155 : tensor<32x32x28x28xf32>
    %v4158 = stablehlo.reshape %v4157 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4159 = stablehlo.reshape %v4158 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4160 = stablehlo.reverse %b4pW, dims = [2, 3] : tensor<32x144x1x1xf32>
    %v4161 = stablehlo.transpose %v4160, dims = [1, 0, 2, 3] : (tensor<32x144x1x1xf32>) -> tensor<144x32x1x1xf32>
    %v4162 = stablehlo.convolution(%v4159, %v4161)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<144x32x1x1xf32>) -> tensor<32x144x28x28xf32>
    %v4163 = stablehlo.reshape %v4162 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v4164 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v4165 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v4166 = stablehlo.compare GT, %v303, %v4164 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v4167 = stablehlo.compare LT, %v303, %v4165 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v4168 = stablehlo.and %v4166, %v4167 : tensor<32x112896xi1>
    %v4169 = stablehlo.select %v4168, %v4163, %v4164 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v4170 = stablehlo.reshape %v283 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4172 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v4173 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v4174 = stablehlo.reduce(%v4170 init: %v4171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4175 = stablehlo.broadcast_in_dim %v4174, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4176 = stablehlo.divide %v4175, %v4172 : tensor<32x144x28x28xf32>
    %v4177 = stablehlo.subtract %v4170, %v4176 : tensor<32x144x28x28xf32>
    %v4178 = stablehlo.multiply %v4177, %v4177 : tensor<32x144x28x28xf32>
    %v4179 = stablehlo.reduce(%v4178 init: %v4171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4180 = stablehlo.broadcast_in_dim %v4179, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4181 = stablehlo.divide %v4180, %v4172 : tensor<32x144x28x28xf32>
    %v4182 = stablehlo.add %v4181, %v4173 : tensor<32x144x28x28xf32>
    %v4183 = stablehlo.rsqrt %v4182 : tensor<32x144x28x28xf32>
    %v4184 = stablehlo.multiply %v4177, %v4183 : tensor<32x144x28x28xf32>
    %v4185 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4186 = stablehlo.reshape %v4169 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4187 = stablehlo.multiply %v4185, %v4186 : tensor<32x144x28x28xf32>
    %v4188 = stablehlo.reduce(%v4187 init: %v4171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4189 = stablehlo.broadcast_in_dim %v4188, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4190 = stablehlo.multiply %v4184, %v4187 : tensor<32x144x28x28xf32>
    %v4191 = stablehlo.reduce(%v4190 init: %v4171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4192 = stablehlo.broadcast_in_dim %v4191, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4193 = stablehlo.multiply %v4187, %v4172 : tensor<32x144x28x28xf32>
    %v4194 = stablehlo.subtract %v4193, %v4189 : tensor<32x144x28x28xf32>
    %v4195 = stablehlo.multiply %v4184, %v4192 : tensor<32x144x28x28xf32>
    %v4196 = stablehlo.subtract %v4194, %v4195 : tensor<32x144x28x28xf32>
    %v4197 = stablehlo.divide %v4183, %v4172 : tensor<32x144x28x28xf32>
    %v4198 = stablehlo.multiply %v4197, %v4196 : tensor<32x144x28x28xf32>
    %v4199 = stablehlo.reshape %v4198 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v4200 = stablehlo.reshape %v4199 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4201 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4202 = stablehlo.pad %v4200, %v4201, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v4203 = stablehlo.reverse %b4dW, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v4204 = stablehlo.convolution(%v4202, %v4203)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v4205 = stablehlo.reshape %v4204 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4206 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v4207 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v4208 = stablehlo.compare GT, %v274, %v4206 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v4209 = stablehlo.compare LT, %v274, %v4207 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v4210 = stablehlo.and %v4208, %v4209 : tensor<32x451584xi1>
    %v4211 = stablehlo.select %v4210, %v4205, %v4206 : tensor<32x451584xi1>, tensor<32x451584xf32>
    %v4212 = stablehlo.reshape %v254 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4213 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4214 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4215 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4216 = stablehlo.reduce(%v4212 init: %v4213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4217 = stablehlo.broadcast_in_dim %v4216, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4218 = stablehlo.divide %v4217, %v4214 : tensor<32x144x56x56xf32>
    %v4219 = stablehlo.subtract %v4212, %v4218 : tensor<32x144x56x56xf32>
    %v4220 = stablehlo.multiply %v4219, %v4219 : tensor<32x144x56x56xf32>
    %v4221 = stablehlo.reduce(%v4220 init: %v4213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4222 = stablehlo.broadcast_in_dim %v4221, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4223 = stablehlo.divide %v4222, %v4214 : tensor<32x144x56x56xf32>
    %v4224 = stablehlo.add %v4223, %v4215 : tensor<32x144x56x56xf32>
    %v4225 = stablehlo.rsqrt %v4224 : tensor<32x144x56x56xf32>
    %v4226 = stablehlo.multiply %v4219, %v4225 : tensor<32x144x56x56xf32>
    %v4227 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4228 = stablehlo.reshape %v4211 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4229 = stablehlo.multiply %v4227, %v4228 : tensor<32x144x56x56xf32>
    %v4230 = stablehlo.reduce(%v4229 init: %v4213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4231 = stablehlo.broadcast_in_dim %v4230, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4232 = stablehlo.multiply %v4226, %v4229 : tensor<32x144x56x56xf32>
    %v4233 = stablehlo.reduce(%v4232 init: %v4213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4234 = stablehlo.broadcast_in_dim %v4233, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4235 = stablehlo.multiply %v4229, %v4214 : tensor<32x144x56x56xf32>
    %v4236 = stablehlo.subtract %v4235, %v4231 : tensor<32x144x56x56xf32>
    %v4237 = stablehlo.multiply %v4226, %v4234 : tensor<32x144x56x56xf32>
    %v4238 = stablehlo.subtract %v4236, %v4237 : tensor<32x144x56x56xf32>
    %v4239 = stablehlo.divide %v4225, %v4214 : tensor<32x144x56x56xf32>
    %v4240 = stablehlo.multiply %v4239, %v4238 : tensor<32x144x56x56xf32>
    %v4241 = stablehlo.reshape %v4240 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4242 = stablehlo.reshape %v4241 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4243 = stablehlo.reverse %b4eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v4244 = stablehlo.transpose %v4243, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v4245 = stablehlo.convolution(%v4242, %v4244)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v4246 = stablehlo.reshape %v4245 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4247 = stablehlo.reshape %v249 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4248 = stablehlo.reshape %v4241 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4249 = stablehlo.transpose %v4247, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4250 = stablehlo.transpose %v4248, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4251 = stablehlo.convolution(%v4249, %v4250)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v4252 = stablehlo.transpose %v4251, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v4253 = stablehlo.reshape %v254 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4255 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4256 = stablehlo.reduce(%v4253 init: %v4254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4257 = stablehlo.broadcast_in_dim %v4256, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4258 = stablehlo.divide %v4257, %v4255 : tensor<32x144x56x56xf32>
    %v4259 = stablehlo.subtract %v4253, %v4258 : tensor<32x144x56x56xf32>
    %v4260 = stablehlo.multiply %v4259, %v4259 : tensor<32x144x56x56xf32>
    %v4261 = stablehlo.reduce(%v4260 init: %v4254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4262 = stablehlo.broadcast_in_dim %v4261, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4263 = stablehlo.divide %v4262, %v4255 : tensor<32x144x56x56xf32>
    %v4264 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4265 = stablehlo.add %v4263, %v4264 : tensor<32x144x56x56xf32>
    %v4266 = stablehlo.rsqrt %v4265 : tensor<32x144x56x56xf32>
    %v4267 = stablehlo.multiply %v4259, %v4266 : tensor<32x144x56x56xf32>
    %v4268 = stablehlo.reshape %v4211 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4269 = stablehlo.multiply %v4268, %v4267 : tensor<32x144x56x56xf32>
    %v4270 = stablehlo.reduce(%v4269 init: %v4254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4271 = stablehlo.reshape %v4211 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4273 = stablehlo.reduce(%v4271 init: %v4272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4274 = stablehlo.reshape %v278 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4275 = stablehlo.reshape %v4199 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4276 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4277 = stablehlo.pad %v4275, %v4276, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v4278 = stablehlo.transpose %v4274, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4279 = stablehlo.transpose %v4277, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4280 = stablehlo.convolution(%v4278, %v4279)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v4281 = stablehlo.reshape %v4280 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v4282 = stablehlo.reshape %v283 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4284 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v4285 = stablehlo.reduce(%v4282 init: %v4283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4286 = stablehlo.broadcast_in_dim %v4285, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4287 = stablehlo.divide %v4286, %v4284 : tensor<32x144x28x28xf32>
    %v4288 = stablehlo.subtract %v4282, %v4287 : tensor<32x144x28x28xf32>
    %v4289 = stablehlo.multiply %v4288, %v4288 : tensor<32x144x28x28xf32>
    %v4290 = stablehlo.reduce(%v4289 init: %v4283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4291 = stablehlo.broadcast_in_dim %v4290, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4292 = stablehlo.divide %v4291, %v4284 : tensor<32x144x28x28xf32>
    %v4293 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v4294 = stablehlo.add %v4292, %v4293 : tensor<32x144x28x28xf32>
    %v4295 = stablehlo.rsqrt %v4294 : tensor<32x144x28x28xf32>
    %v4296 = stablehlo.multiply %v4288, %v4295 : tensor<32x144x28x28xf32>
    %v4297 = stablehlo.reshape %v4169 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4298 = stablehlo.multiply %v4297, %v4296 : tensor<32x144x28x28xf32>
    %v4299 = stablehlo.reduce(%v4298 init: %v4283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4300 = stablehlo.reshape %v4169 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4302 = stablehlo.reduce(%v4300 init: %v4301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4303 = stablehlo.reshape %v307 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4304 = stablehlo.reshape %v4158 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4305 = stablehlo.transpose %v4303, dims = [1, 0, 2, 3] : (tensor<32x144x28x28xf32>) -> tensor<144x32x28x28xf32>
    %v4306 = stablehlo.transpose %v4304, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4307 = stablehlo.convolution(%v4305, %v4306)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<144x32x1x1xf32>
    %v4308 = stablehlo.transpose %v4307, dims = [1, 0, 2, 3] : (tensor<144x32x1x1xf32>) -> tensor<32x144x1x1xf32>
    %v4309 = stablehlo.reshape %v312 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4311 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v4312 = stablehlo.reduce(%v4309 init: %v4310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4313 = stablehlo.broadcast_in_dim %v4312, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4314 = stablehlo.divide %v4313, %v4311 : tensor<32x32x28x28xf32>
    %v4315 = stablehlo.subtract %v4309, %v4314 : tensor<32x32x28x28xf32>
    %v4316 = stablehlo.multiply %v4315, %v4315 : tensor<32x32x28x28xf32>
    %v4317 = stablehlo.reduce(%v4316 init: %v4310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4318 = stablehlo.broadcast_in_dim %v4317, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4319 = stablehlo.divide %v4318, %v4311 : tensor<32x32x28x28xf32>
    %v4320 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4321 = stablehlo.add %v4319, %v4320 : tensor<32x32x28x28xf32>
    %v4322 = stablehlo.rsqrt %v4321 : tensor<32x32x28x28xf32>
    %v4323 = stablehlo.multiply %v4315, %v4322 : tensor<32x32x28x28xf32>
    %v4324 = stablehlo.reshape %v4047 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4325 = stablehlo.multiply %v4324, %v4323 : tensor<32x32x28x28xf32>
    %v4326 = stablehlo.reduce(%v4325 init: %v4310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4327 = stablehlo.reshape %v4047 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4328 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4329 = stablehlo.reduce(%v4327 init: %v4328) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4330 = stablehlo.reshape %v228 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4332 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v4333 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4334 = stablehlo.reduce(%v4330 init: %v4331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4335 = stablehlo.broadcast_in_dim %v4334, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4336 = stablehlo.divide %v4335, %v4332 : tensor<32x24x56x56xf32>
    %v4337 = stablehlo.subtract %v4330, %v4336 : tensor<32x24x56x56xf32>
    %v4338 = stablehlo.multiply %v4337, %v4337 : tensor<32x24x56x56xf32>
    %v4339 = stablehlo.reduce(%v4338 init: %v4331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4340 = stablehlo.broadcast_in_dim %v4339, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4341 = stablehlo.divide %v4340, %v4332 : tensor<32x24x56x56xf32>
    %v4342 = stablehlo.add %v4341, %v4333 : tensor<32x24x56x56xf32>
    %v4343 = stablehlo.rsqrt %v4342 : tensor<32x24x56x56xf32>
    %v4344 = stablehlo.multiply %v4337, %v4343 : tensor<32x24x56x56xf32>
    %v4345 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4346 = stablehlo.reshape %v4246 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4347 = stablehlo.multiply %v4345, %v4346 : tensor<32x24x56x56xf32>
    %v4348 = stablehlo.reduce(%v4347 init: %v4331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4349 = stablehlo.broadcast_in_dim %v4348, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4350 = stablehlo.multiply %v4344, %v4347 : tensor<32x24x56x56xf32>
    %v4351 = stablehlo.reduce(%v4350 init: %v4331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4352 = stablehlo.broadcast_in_dim %v4351, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4353 = stablehlo.multiply %v4347, %v4332 : tensor<32x24x56x56xf32>
    %v4354 = stablehlo.subtract %v4353, %v4349 : tensor<32x24x56x56xf32>
    %v4355 = stablehlo.multiply %v4344, %v4352 : tensor<32x24x56x56xf32>
    %v4356 = stablehlo.subtract %v4354, %v4355 : tensor<32x24x56x56xf32>
    %v4357 = stablehlo.divide %v4343, %v4332 : tensor<32x24x56x56xf32>
    %v4358 = stablehlo.multiply %v4357, %v4356 : tensor<32x24x56x56xf32>
    %v4359 = stablehlo.reshape %v4358 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4360 = stablehlo.reshape %v4359 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4361 = stablehlo.reverse %b3pW, dims = [2, 3] : tensor<24x144x1x1xf32>
    %v4362 = stablehlo.transpose %v4361, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v4363 = stablehlo.convolution(%v4360, %v4362)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v4364 = stablehlo.reshape %v4363 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4365 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v4366 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v4367 = stablehlo.compare GT, %v219, %v4365 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v4368 = stablehlo.compare LT, %v219, %v4366 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v4369 = stablehlo.and %v4367, %v4368 : tensor<32x451584xi1>
    %v4370 = stablehlo.select %v4369, %v4364, %v4365 : tensor<32x451584xi1>, tensor<32x451584xf32>
    %v4371 = stablehlo.reshape %v199 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4372 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4373 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4374 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4375 = stablehlo.reduce(%v4371 init: %v4372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4376 = stablehlo.broadcast_in_dim %v4375, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4377 = stablehlo.divide %v4376, %v4373 : tensor<32x144x56x56xf32>
    %v4378 = stablehlo.subtract %v4371, %v4377 : tensor<32x144x56x56xf32>
    %v4379 = stablehlo.multiply %v4378, %v4378 : tensor<32x144x56x56xf32>
    %v4380 = stablehlo.reduce(%v4379 init: %v4372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4381 = stablehlo.broadcast_in_dim %v4380, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4382 = stablehlo.divide %v4381, %v4373 : tensor<32x144x56x56xf32>
    %v4383 = stablehlo.add %v4382, %v4374 : tensor<32x144x56x56xf32>
    %v4384 = stablehlo.rsqrt %v4383 : tensor<32x144x56x56xf32>
    %v4385 = stablehlo.multiply %v4378, %v4384 : tensor<32x144x56x56xf32>
    %v4386 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4387 = stablehlo.reshape %v4370 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4388 = stablehlo.multiply %v4386, %v4387 : tensor<32x144x56x56xf32>
    %v4389 = stablehlo.reduce(%v4388 init: %v4372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4390 = stablehlo.broadcast_in_dim %v4389, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4391 = stablehlo.multiply %v4385, %v4388 : tensor<32x144x56x56xf32>
    %v4392 = stablehlo.reduce(%v4391 init: %v4372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4393 = stablehlo.broadcast_in_dim %v4392, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4394 = stablehlo.multiply %v4388, %v4373 : tensor<32x144x56x56xf32>
    %v4395 = stablehlo.subtract %v4394, %v4390 : tensor<32x144x56x56xf32>
    %v4396 = stablehlo.multiply %v4385, %v4393 : tensor<32x144x56x56xf32>
    %v4397 = stablehlo.subtract %v4395, %v4396 : tensor<32x144x56x56xf32>
    %v4398 = stablehlo.divide %v4384, %v4373 : tensor<32x144x56x56xf32>
    %v4399 = stablehlo.multiply %v4398, %v4397 : tensor<32x144x56x56xf32>
    %v4400 = stablehlo.reshape %v4399 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4401 = stablehlo.reshape %v4400 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4402 = stablehlo.reverse %b3dW, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v4403 = stablehlo.convolution(%v4401, %v4402)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v4404 = stablehlo.reshape %v4403 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4405 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v4406 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v4407 = stablehlo.compare GT, %v190, %v4405 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v4408 = stablehlo.compare LT, %v190, %v4406 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v4409 = stablehlo.and %v4407, %v4408 : tensor<32x451584xi1>
    %v4410 = stablehlo.select %v4409, %v4404, %v4405 : tensor<32x451584xi1>, tensor<32x451584xf32>
    %v4411 = stablehlo.reshape %v170 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4412 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4413 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4414 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4415 = stablehlo.reduce(%v4411 init: %v4412) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4416 = stablehlo.broadcast_in_dim %v4415, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4417 = stablehlo.divide %v4416, %v4413 : tensor<32x144x56x56xf32>
    %v4418 = stablehlo.subtract %v4411, %v4417 : tensor<32x144x56x56xf32>
    %v4419 = stablehlo.multiply %v4418, %v4418 : tensor<32x144x56x56xf32>
    %v4420 = stablehlo.reduce(%v4419 init: %v4412) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4421 = stablehlo.broadcast_in_dim %v4420, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4422 = stablehlo.divide %v4421, %v4413 : tensor<32x144x56x56xf32>
    %v4423 = stablehlo.add %v4422, %v4414 : tensor<32x144x56x56xf32>
    %v4424 = stablehlo.rsqrt %v4423 : tensor<32x144x56x56xf32>
    %v4425 = stablehlo.multiply %v4418, %v4424 : tensor<32x144x56x56xf32>
    %v4426 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4427 = stablehlo.reshape %v4410 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4428 = stablehlo.multiply %v4426, %v4427 : tensor<32x144x56x56xf32>
    %v4429 = stablehlo.reduce(%v4428 init: %v4412) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4430 = stablehlo.broadcast_in_dim %v4429, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4431 = stablehlo.multiply %v4425, %v4428 : tensor<32x144x56x56xf32>
    %v4432 = stablehlo.reduce(%v4431 init: %v4412) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4433 = stablehlo.broadcast_in_dim %v4432, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4434 = stablehlo.multiply %v4428, %v4413 : tensor<32x144x56x56xf32>
    %v4435 = stablehlo.subtract %v4434, %v4430 : tensor<32x144x56x56xf32>
    %v4436 = stablehlo.multiply %v4425, %v4433 : tensor<32x144x56x56xf32>
    %v4437 = stablehlo.subtract %v4435, %v4436 : tensor<32x144x56x56xf32>
    %v4438 = stablehlo.divide %v4424, %v4413 : tensor<32x144x56x56xf32>
    %v4439 = stablehlo.multiply %v4438, %v4437 : tensor<32x144x56x56xf32>
    %v4440 = stablehlo.reshape %v4439 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4441 = stablehlo.reshape %v4440 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4442 = stablehlo.reverse %b3eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v4443 = stablehlo.transpose %v4442, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v4444 = stablehlo.convolution(%v4441, %v4443)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v4445 = stablehlo.reshape %v4444 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4446 = stablehlo.add %v4445, %v4246 : tensor<32x75264xf32>
    %v4447 = stablehlo.reshape %v165 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4448 = stablehlo.reshape %v4440 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4449 = stablehlo.transpose %v4447, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4450 = stablehlo.transpose %v4448, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4451 = stablehlo.convolution(%v4449, %v4450)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v4452 = stablehlo.transpose %v4451, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v4453 = stablehlo.reshape %v170 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4454 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4455 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4456 = stablehlo.reduce(%v4453 init: %v4454) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4457 = stablehlo.broadcast_in_dim %v4456, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4458 = stablehlo.divide %v4457, %v4455 : tensor<32x144x56x56xf32>
    %v4459 = stablehlo.subtract %v4453, %v4458 : tensor<32x144x56x56xf32>
    %v4460 = stablehlo.multiply %v4459, %v4459 : tensor<32x144x56x56xf32>
    %v4461 = stablehlo.reduce(%v4460 init: %v4454) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4462 = stablehlo.broadcast_in_dim %v4461, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4463 = stablehlo.divide %v4462, %v4455 : tensor<32x144x56x56xf32>
    %v4464 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4465 = stablehlo.add %v4463, %v4464 : tensor<32x144x56x56xf32>
    %v4466 = stablehlo.rsqrt %v4465 : tensor<32x144x56x56xf32>
    %v4467 = stablehlo.multiply %v4459, %v4466 : tensor<32x144x56x56xf32>
    %v4468 = stablehlo.reshape %v4410 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4469 = stablehlo.multiply %v4468, %v4467 : tensor<32x144x56x56xf32>
    %v4470 = stablehlo.reduce(%v4469 init: %v4454) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4471 = stablehlo.reshape %v4410 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4472 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4473 = stablehlo.reduce(%v4471 init: %v4472) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4474 = stablehlo.reshape %v194 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4475 = stablehlo.reshape %v4400 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4476 = stablehlo.transpose %v4474, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4477 = stablehlo.transpose %v4475, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4478 = stablehlo.convolution(%v4476, %v4477)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v4479 = stablehlo.reshape %v4478 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v4480 = stablehlo.reshape %v199 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4481 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4482 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4483 = stablehlo.reduce(%v4480 init: %v4481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4484 = stablehlo.broadcast_in_dim %v4483, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4485 = stablehlo.divide %v4484, %v4482 : tensor<32x144x56x56xf32>
    %v4486 = stablehlo.subtract %v4480, %v4485 : tensor<32x144x56x56xf32>
    %v4487 = stablehlo.multiply %v4486, %v4486 : tensor<32x144x56x56xf32>
    %v4488 = stablehlo.reduce(%v4487 init: %v4481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4489 = stablehlo.broadcast_in_dim %v4488, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4490 = stablehlo.divide %v4489, %v4482 : tensor<32x144x56x56xf32>
    %v4491 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4492 = stablehlo.add %v4490, %v4491 : tensor<32x144x56x56xf32>
    %v4493 = stablehlo.rsqrt %v4492 : tensor<32x144x56x56xf32>
    %v4494 = stablehlo.multiply %v4486, %v4493 : tensor<32x144x56x56xf32>
    %v4495 = stablehlo.reshape %v4370 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4496 = stablehlo.multiply %v4495, %v4494 : tensor<32x144x56x56xf32>
    %v4497 = stablehlo.reduce(%v4496 init: %v4481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4498 = stablehlo.reshape %v4370 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4499 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4500 = stablehlo.reduce(%v4498 init: %v4499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4501 = stablehlo.reshape %v223 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4502 = stablehlo.reshape %v4359 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4503 = stablehlo.transpose %v4501, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4504 = stablehlo.transpose %v4502, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4505 = stablehlo.convolution(%v4503, %v4504)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<144x24x1x1xf32>
    %v4506 = stablehlo.transpose %v4505, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v4507 = stablehlo.reshape %v228 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4508 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4509 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v4510 = stablehlo.reduce(%v4507 init: %v4508) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4511 = stablehlo.broadcast_in_dim %v4510, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4512 = stablehlo.divide %v4511, %v4509 : tensor<32x24x56x56xf32>
    %v4513 = stablehlo.subtract %v4507, %v4512 : tensor<32x24x56x56xf32>
    %v4514 = stablehlo.multiply %v4513, %v4513 : tensor<32x24x56x56xf32>
    %v4515 = stablehlo.reduce(%v4514 init: %v4508) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4516 = stablehlo.broadcast_in_dim %v4515, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4517 = stablehlo.divide %v4516, %v4509 : tensor<32x24x56x56xf32>
    %v4518 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4519 = stablehlo.add %v4517, %v4518 : tensor<32x24x56x56xf32>
    %v4520 = stablehlo.rsqrt %v4519 : tensor<32x24x56x56xf32>
    %v4521 = stablehlo.multiply %v4513, %v4520 : tensor<32x24x56x56xf32>
    %v4522 = stablehlo.reshape %v4246 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4523 = stablehlo.multiply %v4522, %v4521 : tensor<32x24x56x56xf32>
    %v4524 = stablehlo.reduce(%v4523 init: %v4508) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4525 = stablehlo.reshape %v4246 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4526 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4527 = stablehlo.reduce(%v4525 init: %v4526) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4528 = stablehlo.reshape %v145 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4530 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v4531 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4532 = stablehlo.reduce(%v4528 init: %v4529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4533 = stablehlo.broadcast_in_dim %v4532, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4534 = stablehlo.divide %v4533, %v4530 : tensor<32x24x56x56xf32>
    %v4535 = stablehlo.subtract %v4528, %v4534 : tensor<32x24x56x56xf32>
    %v4536 = stablehlo.multiply %v4535, %v4535 : tensor<32x24x56x56xf32>
    %v4537 = stablehlo.reduce(%v4536 init: %v4529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4538 = stablehlo.broadcast_in_dim %v4537, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4539 = stablehlo.divide %v4538, %v4530 : tensor<32x24x56x56xf32>
    %v4540 = stablehlo.add %v4539, %v4531 : tensor<32x24x56x56xf32>
    %v4541 = stablehlo.rsqrt %v4540 : tensor<32x24x56x56xf32>
    %v4542 = stablehlo.multiply %v4535, %v4541 : tensor<32x24x56x56xf32>
    %v4543 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4544 = stablehlo.reshape %v4446 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4545 = stablehlo.multiply %v4543, %v4544 : tensor<32x24x56x56xf32>
    %v4546 = stablehlo.reduce(%v4545 init: %v4529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4547 = stablehlo.broadcast_in_dim %v4546, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4548 = stablehlo.multiply %v4542, %v4545 : tensor<32x24x56x56xf32>
    %v4549 = stablehlo.reduce(%v4548 init: %v4529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4550 = stablehlo.broadcast_in_dim %v4549, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4551 = stablehlo.multiply %v4545, %v4530 : tensor<32x24x56x56xf32>
    %v4552 = stablehlo.subtract %v4551, %v4547 : tensor<32x24x56x56xf32>
    %v4553 = stablehlo.multiply %v4542, %v4550 : tensor<32x24x56x56xf32>
    %v4554 = stablehlo.subtract %v4552, %v4553 : tensor<32x24x56x56xf32>
    %v4555 = stablehlo.divide %v4541, %v4530 : tensor<32x24x56x56xf32>
    %v4556 = stablehlo.multiply %v4555, %v4554 : tensor<32x24x56x56xf32>
    %v4557 = stablehlo.reshape %v4556 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4558 = stablehlo.reshape %v4557 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4559 = stablehlo.reverse %b2pW, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v4560 = stablehlo.transpose %v4559, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v4561 = stablehlo.convolution(%v4558, %v4560)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v4562 = stablehlo.reshape %v4561 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4563 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v4564 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v4565 = stablehlo.compare GT, %v136, %v4563 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v4566 = stablehlo.compare LT, %v136, %v4564 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v4567 = stablehlo.and %v4565, %v4566 : tensor<32x301056xi1>
    %v4568 = stablehlo.select %v4567, %v4562, %v4563 : tensor<32x301056xi1>, tensor<32x301056xf32>
    %v4569 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4571 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v4572 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v4573 = stablehlo.reduce(%v4569 init: %v4570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4574 = stablehlo.broadcast_in_dim %v4573, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4575 = stablehlo.divide %v4574, %v4571 : tensor<32x96x56x56xf32>
    %v4576 = stablehlo.subtract %v4569, %v4575 : tensor<32x96x56x56xf32>
    %v4577 = stablehlo.multiply %v4576, %v4576 : tensor<32x96x56x56xf32>
    %v4578 = stablehlo.reduce(%v4577 init: %v4570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4579 = stablehlo.broadcast_in_dim %v4578, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4580 = stablehlo.divide %v4579, %v4571 : tensor<32x96x56x56xf32>
    %v4581 = stablehlo.add %v4580, %v4572 : tensor<32x96x56x56xf32>
    %v4582 = stablehlo.rsqrt %v4581 : tensor<32x96x56x56xf32>
    %v4583 = stablehlo.multiply %v4576, %v4582 : tensor<32x96x56x56xf32>
    %v4584 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4585 = stablehlo.reshape %v4568 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4586 = stablehlo.multiply %v4584, %v4585 : tensor<32x96x56x56xf32>
    %v4587 = stablehlo.reduce(%v4586 init: %v4570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4588 = stablehlo.broadcast_in_dim %v4587, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4589 = stablehlo.multiply %v4583, %v4586 : tensor<32x96x56x56xf32>
    %v4590 = stablehlo.reduce(%v4589 init: %v4570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4591 = stablehlo.broadcast_in_dim %v4590, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4592 = stablehlo.multiply %v4586, %v4571 : tensor<32x96x56x56xf32>
    %v4593 = stablehlo.subtract %v4592, %v4588 : tensor<32x96x56x56xf32>
    %v4594 = stablehlo.multiply %v4583, %v4591 : tensor<32x96x56x56xf32>
    %v4595 = stablehlo.subtract %v4593, %v4594 : tensor<32x96x56x56xf32>
    %v4596 = stablehlo.divide %v4582, %v4571 : tensor<32x96x56x56xf32>
    %v4597 = stablehlo.multiply %v4596, %v4595 : tensor<32x96x56x56xf32>
    %v4598 = stablehlo.reshape %v4597 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4599 = stablehlo.reshape %v4598 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4600 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4601 = stablehlo.pad %v4599, %v4600, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v4602 = stablehlo.reverse %b2dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v4603 = stablehlo.convolution(%v4601, %v4602)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x112x112xf32>
    %v4604 = stablehlo.reshape %v4603 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v4605 = stablehlo.constant dense<0.0> : tensor<32x1204224xf32>
    %v4606 = stablehlo.constant dense<6.0> : tensor<32x1204224xf32>
    %v4607 = stablehlo.compare GT, %v107, %v4605 : (tensor<32x1204224xf32>, tensor<32x1204224xf32>) -> tensor<32x1204224xi1>
    %v4608 = stablehlo.compare LT, %v107, %v4606 : (tensor<32x1204224xf32>, tensor<32x1204224xf32>) -> tensor<32x1204224xi1>
    %v4609 = stablehlo.and %v4607, %v4608 : tensor<32x1204224xi1>
    %v4610 = stablehlo.select %v4609, %v4604, %v4605 : tensor<32x1204224xi1>, tensor<32x1204224xf32>
    %v4611 = stablehlo.reshape %v87 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4612 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4613 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v4614 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v4615 = stablehlo.reduce(%v4611 init: %v4612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4616 = stablehlo.broadcast_in_dim %v4615, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4617 = stablehlo.divide %v4616, %v4613 : tensor<32x96x112x112xf32>
    %v4618 = stablehlo.subtract %v4611, %v4617 : tensor<32x96x112x112xf32>
    %v4619 = stablehlo.multiply %v4618, %v4618 : tensor<32x96x112x112xf32>
    %v4620 = stablehlo.reduce(%v4619 init: %v4612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4621 = stablehlo.broadcast_in_dim %v4620, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4622 = stablehlo.divide %v4621, %v4613 : tensor<32x96x112x112xf32>
    %v4623 = stablehlo.add %v4622, %v4614 : tensor<32x96x112x112xf32>
    %v4624 = stablehlo.rsqrt %v4623 : tensor<32x96x112x112xf32>
    %v4625 = stablehlo.multiply %v4618, %v4624 : tensor<32x96x112x112xf32>
    %v4626 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4627 = stablehlo.reshape %v4610 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4628 = stablehlo.multiply %v4626, %v4627 : tensor<32x96x112x112xf32>
    %v4629 = stablehlo.reduce(%v4628 init: %v4612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4630 = stablehlo.broadcast_in_dim %v4629, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4631 = stablehlo.multiply %v4625, %v4628 : tensor<32x96x112x112xf32>
    %v4632 = stablehlo.reduce(%v4631 init: %v4612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4633 = stablehlo.broadcast_in_dim %v4632, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4634 = stablehlo.multiply %v4628, %v4613 : tensor<32x96x112x112xf32>
    %v4635 = stablehlo.subtract %v4634, %v4630 : tensor<32x96x112x112xf32>
    %v4636 = stablehlo.multiply %v4625, %v4633 : tensor<32x96x112x112xf32>
    %v4637 = stablehlo.subtract %v4635, %v4636 : tensor<32x96x112x112xf32>
    %v4638 = stablehlo.divide %v4624, %v4613 : tensor<32x96x112x112xf32>
    %v4639 = stablehlo.multiply %v4638, %v4637 : tensor<32x96x112x112xf32>
    %v4640 = stablehlo.reshape %v4639 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v4641 = stablehlo.reshape %v4640 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4642 = stablehlo.reverse %b2eW, dims = [2, 3] : tensor<96x16x1x1xf32>
    %v4643 = stablehlo.transpose %v4642, dims = [1, 0, 2, 3] : (tensor<96x16x1x1xf32>) -> tensor<16x96x1x1xf32>
    %v4644 = stablehlo.convolution(%v4641, %v4643)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x112x112xf32>, tensor<16x96x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v4645 = stablehlo.reshape %v4644 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v4646 = stablehlo.reshape %v82 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4647 = stablehlo.reshape %v4640 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4648 = stablehlo.transpose %v4646, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v4649 = stablehlo.transpose %v4647, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v4650 = stablehlo.convolution(%v4648, %v4649)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<16x96x1x1xf32>
    %v4651 = stablehlo.transpose %v4650, dims = [1, 0, 2, 3] : (tensor<16x96x1x1xf32>) -> tensor<96x16x1x1xf32>
    %v4652 = stablehlo.reshape %v87 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4653 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4654 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v4655 = stablehlo.reduce(%v4652 init: %v4653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4656 = stablehlo.broadcast_in_dim %v4655, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4657 = stablehlo.divide %v4656, %v4654 : tensor<32x96x112x112xf32>
    %v4658 = stablehlo.subtract %v4652, %v4657 : tensor<32x96x112x112xf32>
    %v4659 = stablehlo.multiply %v4658, %v4658 : tensor<32x96x112x112xf32>
    %v4660 = stablehlo.reduce(%v4659 init: %v4653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4661 = stablehlo.broadcast_in_dim %v4660, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4662 = stablehlo.divide %v4661, %v4654 : tensor<32x96x112x112xf32>
    %v4663 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v4664 = stablehlo.add %v4662, %v4663 : tensor<32x96x112x112xf32>
    %v4665 = stablehlo.rsqrt %v4664 : tensor<32x96x112x112xf32>
    %v4666 = stablehlo.multiply %v4658, %v4665 : tensor<32x96x112x112xf32>
    %v4667 = stablehlo.reshape %v4610 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4668 = stablehlo.multiply %v4667, %v4666 : tensor<32x96x112x112xf32>
    %v4669 = stablehlo.reduce(%v4668 init: %v4653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4670 = stablehlo.reshape %v4610 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4672 = stablehlo.reduce(%v4670 init: %v4671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4673 = stablehlo.reshape %v111 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4674 = stablehlo.reshape %v4598 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4676 = stablehlo.pad %v4674, %v4675, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v4677 = stablehlo.transpose %v4673, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v4678 = stablehlo.transpose %v4676, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v4679 = stablehlo.convolution(%v4677, %v4678)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<1x96x3x3xf32>
    %v4680 = stablehlo.reshape %v4679 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v4681 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4682 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4683 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v4684 = stablehlo.reduce(%v4681 init: %v4682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4685 = stablehlo.broadcast_in_dim %v4684, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4686 = stablehlo.divide %v4685, %v4683 : tensor<32x96x56x56xf32>
    %v4687 = stablehlo.subtract %v4681, %v4686 : tensor<32x96x56x56xf32>
    %v4688 = stablehlo.multiply %v4687, %v4687 : tensor<32x96x56x56xf32>
    %v4689 = stablehlo.reduce(%v4688 init: %v4682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4690 = stablehlo.broadcast_in_dim %v4689, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4691 = stablehlo.divide %v4690, %v4683 : tensor<32x96x56x56xf32>
    %v4692 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v4693 = stablehlo.add %v4691, %v4692 : tensor<32x96x56x56xf32>
    %v4694 = stablehlo.rsqrt %v4693 : tensor<32x96x56x56xf32>
    %v4695 = stablehlo.multiply %v4687, %v4694 : tensor<32x96x56x56xf32>
    %v4696 = stablehlo.reshape %v4568 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4697 = stablehlo.multiply %v4696, %v4695 : tensor<32x96x56x56xf32>
    %v4698 = stablehlo.reduce(%v4697 init: %v4682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4699 = stablehlo.reshape %v4568 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4700 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4701 = stablehlo.reduce(%v4699 init: %v4700) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4702 = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4703 = stablehlo.reshape %v4557 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4704 = stablehlo.transpose %v4702, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4705 = stablehlo.transpose %v4703, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4706 = stablehlo.convolution(%v4704, %v4705)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %v4707 = stablehlo.transpose %v4706, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v4708 = stablehlo.reshape %v145 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4710 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v4711 = stablehlo.reduce(%v4708 init: %v4709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4712 = stablehlo.broadcast_in_dim %v4711, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4713 = stablehlo.divide %v4712, %v4710 : tensor<32x24x56x56xf32>
    %v4714 = stablehlo.subtract %v4708, %v4713 : tensor<32x24x56x56xf32>
    %v4715 = stablehlo.multiply %v4714, %v4714 : tensor<32x24x56x56xf32>
    %v4716 = stablehlo.reduce(%v4715 init: %v4709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4717 = stablehlo.broadcast_in_dim %v4716, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4718 = stablehlo.divide %v4717, %v4710 : tensor<32x24x56x56xf32>
    %v4719 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4720 = stablehlo.add %v4718, %v4719 : tensor<32x24x56x56xf32>
    %v4721 = stablehlo.rsqrt %v4720 : tensor<32x24x56x56xf32>
    %v4722 = stablehlo.multiply %v4714, %v4721 : tensor<32x24x56x56xf32>
    %v4723 = stablehlo.reshape %v4446 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4724 = stablehlo.multiply %v4723, %v4722 : tensor<32x24x56x56xf32>
    %v4725 = stablehlo.reduce(%v4724 init: %v4709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4726 = stablehlo.reshape %v4446 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4727 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4728 = stablehlo.reduce(%v4726 init: %v4727) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4729 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4731 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v4732 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v4733 = stablehlo.reduce(%v4729 init: %v4730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4734 = stablehlo.broadcast_in_dim %v4733, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4735 = stablehlo.divide %v4734, %v4731 : tensor<32x16x112x112xf32>
    %v4736 = stablehlo.subtract %v4729, %v4735 : tensor<32x16x112x112xf32>
    %v4737 = stablehlo.multiply %v4736, %v4736 : tensor<32x16x112x112xf32>
    %v4738 = stablehlo.reduce(%v4737 init: %v4730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4739 = stablehlo.broadcast_in_dim %v4738, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4740 = stablehlo.divide %v4739, %v4731 : tensor<32x16x112x112xf32>
    %v4741 = stablehlo.add %v4740, %v4732 : tensor<32x16x112x112xf32>
    %v4742 = stablehlo.rsqrt %v4741 : tensor<32x16x112x112xf32>
    %v4743 = stablehlo.multiply %v4736, %v4742 : tensor<32x16x112x112xf32>
    %v4744 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4745 = stablehlo.reshape %v4645 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4746 = stablehlo.multiply %v4744, %v4745 : tensor<32x16x112x112xf32>
    %v4747 = stablehlo.reduce(%v4746 init: %v4730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4748 = stablehlo.broadcast_in_dim %v4747, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4749 = stablehlo.multiply %v4743, %v4746 : tensor<32x16x112x112xf32>
    %v4750 = stablehlo.reduce(%v4749 init: %v4730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4751 = stablehlo.broadcast_in_dim %v4750, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4752 = stablehlo.multiply %v4746, %v4731 : tensor<32x16x112x112xf32>
    %v4753 = stablehlo.subtract %v4752, %v4748 : tensor<32x16x112x112xf32>
    %v4754 = stablehlo.multiply %v4743, %v4751 : tensor<32x16x112x112xf32>
    %v4755 = stablehlo.subtract %v4753, %v4754 : tensor<32x16x112x112xf32>
    %v4756 = stablehlo.divide %v4742, %v4731 : tensor<32x16x112x112xf32>
    %v4757 = stablehlo.multiply %v4756, %v4755 : tensor<32x16x112x112xf32>
    %v4758 = stablehlo.reshape %v4757 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v4759 = stablehlo.reshape %v4758 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4760 = stablehlo.reverse %b1pW, dims = [2, 3] : tensor<16x32x1x1xf32>
    %v4761 = stablehlo.transpose %v4760, dims = [1, 0, 2, 3] : (tensor<16x32x1x1xf32>) -> tensor<32x16x1x1xf32>
    %v4762 = stablehlo.convolution(%v4759, %v4761)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<32x16x1x1xf32>) -> tensor<32x32x112x112xf32>
    %v4763 = stablehlo.reshape %v4762 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v4764 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v4765 = stablehlo.constant dense<6.0> : tensor<32x401408xf32>
    %v4766 = stablehlo.compare GT, %v53, %v4764 : (tensor<32x401408xf32>, tensor<32x401408xf32>) -> tensor<32x401408xi1>
    %v4767 = stablehlo.compare LT, %v53, %v4765 : (tensor<32x401408xf32>, tensor<32x401408xf32>) -> tensor<32x401408xi1>
    %v4768 = stablehlo.and %v4766, %v4767 : tensor<32x401408xi1>
    %v4769 = stablehlo.select %v4768, %v4763, %v4764 : tensor<32x401408xi1>, tensor<32x401408xf32>
    %v4770 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4772 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v4773 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v4774 = stablehlo.reduce(%v4770 init: %v4771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4775 = stablehlo.broadcast_in_dim %v4774, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4776 = stablehlo.divide %v4775, %v4772 : tensor<32x32x112x112xf32>
    %v4777 = stablehlo.subtract %v4770, %v4776 : tensor<32x32x112x112xf32>
    %v4778 = stablehlo.multiply %v4777, %v4777 : tensor<32x32x112x112xf32>
    %v4779 = stablehlo.reduce(%v4778 init: %v4771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4780 = stablehlo.broadcast_in_dim %v4779, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4781 = stablehlo.divide %v4780, %v4772 : tensor<32x32x112x112xf32>
    %v4782 = stablehlo.add %v4781, %v4773 : tensor<32x32x112x112xf32>
    %v4783 = stablehlo.rsqrt %v4782 : tensor<32x32x112x112xf32>
    %v4784 = stablehlo.multiply %v4777, %v4783 : tensor<32x32x112x112xf32>
    %v4785 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4786 = stablehlo.reshape %v4769 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4787 = stablehlo.multiply %v4785, %v4786 : tensor<32x32x112x112xf32>
    %v4788 = stablehlo.reduce(%v4787 init: %v4771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4789 = stablehlo.broadcast_in_dim %v4788, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4790 = stablehlo.multiply %v4784, %v4787 : tensor<32x32x112x112xf32>
    %v4791 = stablehlo.reduce(%v4790 init: %v4771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4792 = stablehlo.broadcast_in_dim %v4791, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4793 = stablehlo.multiply %v4787, %v4772 : tensor<32x32x112x112xf32>
    %v4794 = stablehlo.subtract %v4793, %v4789 : tensor<32x32x112x112xf32>
    %v4795 = stablehlo.multiply %v4784, %v4792 : tensor<32x32x112x112xf32>
    %v4796 = stablehlo.subtract %v4794, %v4795 : tensor<32x32x112x112xf32>
    %v4797 = stablehlo.divide %v4783, %v4772 : tensor<32x32x112x112xf32>
    %v4798 = stablehlo.multiply %v4797, %v4796 : tensor<32x32x112x112xf32>
    %v4799 = stablehlo.reshape %v4798 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v4800 = stablehlo.reshape %v4799 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4801 = stablehlo.reverse %b1dW, dims = [2, 3] : tensor<32x1x3x3xf32>
    %v4802 = stablehlo.convolution(%v4800, %v4801)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v4803 = stablehlo.reshape %v4802 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v4804 = stablehlo.reshape %v28 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4805 = stablehlo.reshape %v4799 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4806 = stablehlo.transpose %v4804, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v4807 = stablehlo.transpose %v4805, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v4808 = stablehlo.convolution(%v4806, %v4807)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 32 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<1x32x3x3xf32>
    %v4809 = stablehlo.reshape %v4808 : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %v4810 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4811 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4812 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v4813 = stablehlo.reduce(%v4810 init: %v4811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4814 = stablehlo.broadcast_in_dim %v4813, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4815 = stablehlo.divide %v4814, %v4812 : tensor<32x32x112x112xf32>
    %v4816 = stablehlo.subtract %v4810, %v4815 : tensor<32x32x112x112xf32>
    %v4817 = stablehlo.multiply %v4816, %v4816 : tensor<32x32x112x112xf32>
    %v4818 = stablehlo.reduce(%v4817 init: %v4811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4819 = stablehlo.broadcast_in_dim %v4818, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4820 = stablehlo.divide %v4819, %v4812 : tensor<32x32x112x112xf32>
    %v4821 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v4822 = stablehlo.add %v4820, %v4821 : tensor<32x32x112x112xf32>
    %v4823 = stablehlo.rsqrt %v4822 : tensor<32x32x112x112xf32>
    %v4824 = stablehlo.multiply %v4816, %v4823 : tensor<32x32x112x112xf32>
    %v4825 = stablehlo.reshape %v4769 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4826 = stablehlo.multiply %v4825, %v4824 : tensor<32x32x112x112xf32>
    %v4827 = stablehlo.reduce(%v4826 init: %v4811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4828 = stablehlo.reshape %v4769 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4830 = stablehlo.reduce(%v4828 init: %v4829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4831 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4832 = stablehlo.reshape %v4758 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4833 = stablehlo.transpose %v4831, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v4834 = stablehlo.transpose %v4832, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v4835 = stablehlo.convolution(%v4833, %v4834)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x112x112xf32>) -> tensor<32x16x1x1xf32>
    %v4836 = stablehlo.transpose %v4835, dims = [1, 0, 2, 3] : (tensor<32x16x1x1xf32>) -> tensor<16x32x1x1xf32>
    %v4837 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4839 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v4840 = stablehlo.reduce(%v4837 init: %v4838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4841 = stablehlo.broadcast_in_dim %v4840, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4842 = stablehlo.divide %v4841, %v4839 : tensor<32x16x112x112xf32>
    %v4843 = stablehlo.subtract %v4837, %v4842 : tensor<32x16x112x112xf32>
    %v4844 = stablehlo.multiply %v4843, %v4843 : tensor<32x16x112x112xf32>
    %v4845 = stablehlo.reduce(%v4844 init: %v4838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4846 = stablehlo.broadcast_in_dim %v4845, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4847 = stablehlo.divide %v4846, %v4839 : tensor<32x16x112x112xf32>
    %v4848 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v4849 = stablehlo.add %v4847, %v4848 : tensor<32x16x112x112xf32>
    %v4850 = stablehlo.rsqrt %v4849 : tensor<32x16x112x112xf32>
    %v4851 = stablehlo.multiply %v4843, %v4850 : tensor<32x16x112x112xf32>
    %v4852 = stablehlo.reshape %v4645 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4853 = stablehlo.multiply %v4852, %v4851 : tensor<32x16x112x112xf32>
    %v4854 = stablehlo.reduce(%v4853 init: %v4838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4855 = stablehlo.reshape %v4645 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4857 = stablehlo.reduce(%v4855 init: %v4856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4858 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v4859 = stablehlo.constant dense<6.0> : tensor<32x401408xf32>
    %v4860 = stablehlo.compare GT, %v24, %v4858 : (tensor<32x401408xf32>, tensor<32x401408xf32>) -> tensor<32x401408xi1>
    %v4861 = stablehlo.compare LT, %v24, %v4859 : (tensor<32x401408xf32>, tensor<32x401408xf32>) -> tensor<32x401408xi1>
    %v4862 = stablehlo.and %v4860, %v4861 : tensor<32x401408xi1>
    %v4863 = stablehlo.select %v4862, %v4803, %v4858 : tensor<32x401408xi1>, tensor<32x401408xf32>
    %v4864 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4865 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4866 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v4867 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v4868 = stablehlo.reduce(%v4864 init: %v4865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4869 = stablehlo.broadcast_in_dim %v4868, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4870 = stablehlo.divide %v4869, %v4866 : tensor<32x32x112x112xf32>
    %v4871 = stablehlo.subtract %v4864, %v4870 : tensor<32x32x112x112xf32>
    %v4872 = stablehlo.multiply %v4871, %v4871 : tensor<32x32x112x112xf32>
    %v4873 = stablehlo.reduce(%v4872 init: %v4865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4874 = stablehlo.broadcast_in_dim %v4873, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4875 = stablehlo.divide %v4874, %v4866 : tensor<32x32x112x112xf32>
    %v4876 = stablehlo.add %v4875, %v4867 : tensor<32x32x112x112xf32>
    %v4877 = stablehlo.rsqrt %v4876 : tensor<32x32x112x112xf32>
    %v4878 = stablehlo.multiply %v4871, %v4877 : tensor<32x32x112x112xf32>
    %v4879 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4880 = stablehlo.reshape %v4863 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4881 = stablehlo.multiply %v4879, %v4880 : tensor<32x32x112x112xf32>
    %v4882 = stablehlo.reduce(%v4881 init: %v4865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4883 = stablehlo.broadcast_in_dim %v4882, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4884 = stablehlo.multiply %v4878, %v4881 : tensor<32x32x112x112xf32>
    %v4885 = stablehlo.reduce(%v4884 init: %v4865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4886 = stablehlo.broadcast_in_dim %v4885, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4887 = stablehlo.multiply %v4881, %v4866 : tensor<32x32x112x112xf32>
    %v4888 = stablehlo.subtract %v4887, %v4883 : tensor<32x32x112x112xf32>
    %v4889 = stablehlo.multiply %v4878, %v4886 : tensor<32x32x112x112xf32>
    %v4890 = stablehlo.subtract %v4888, %v4889 : tensor<32x32x112x112xf32>
    %v4891 = stablehlo.divide %v4877, %v4866 : tensor<32x32x112x112xf32>
    %v4892 = stablehlo.multiply %v4891, %v4890 : tensor<32x32x112x112xf32>
    %v4893 = stablehlo.reshape %v4892 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v4894 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v4895 = stablehlo.reshape %v4893 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4896 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4897 = stablehlo.pad %v4895, %v4896, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32x224x224xf32>
    %v4898 = stablehlo.transpose %v4894, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v4899 = stablehlo.transpose %v4897, dims = [1, 0, 2, 3] : (tensor<32x32x224x224xf32>) -> tensor<32x32x224x224xf32>
    %v4900 = stablehlo.convolution(%v4898, %v4899)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<32x32x224x224xf32>) -> tensor<3x32x3x3xf32>
    %v4901 = stablehlo.transpose %v4900, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %v4902 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4903 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4904 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v4905 = stablehlo.reduce(%v4902 init: %v4903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4906 = stablehlo.broadcast_in_dim %v4905, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4907 = stablehlo.divide %v4906, %v4904 : tensor<32x32x112x112xf32>
    %v4908 = stablehlo.subtract %v4902, %v4907 : tensor<32x32x112x112xf32>
    %v4909 = stablehlo.multiply %v4908, %v4908 : tensor<32x32x112x112xf32>
    %v4910 = stablehlo.reduce(%v4909 init: %v4903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4911 = stablehlo.broadcast_in_dim %v4910, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4912 = stablehlo.divide %v4911, %v4904 : tensor<32x32x112x112xf32>
    %v4913 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v4914 = stablehlo.add %v4912, %v4913 : tensor<32x32x112x112xf32>
    %v4915 = stablehlo.rsqrt %v4914 : tensor<32x32x112x112xf32>
    %v4916 = stablehlo.multiply %v4908, %v4915 : tensor<32x32x112x112xf32>
    %v4917 = stablehlo.reshape %v4863 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4918 = stablehlo.multiply %v4917, %v4916 : tensor<32x32x112x112xf32>
    %v4919 = stablehlo.reduce(%v4918 init: %v4903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4920 = stablehlo.reshape %v4863 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4922 = stablehlo.reduce(%v4920 init: %v4921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4923 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4925 = stablehlo.constant dense<401408.0> : tensor<32xf32>
    %v4926 = stablehlo.reduce(%v4923 init: %v4924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4927 = stablehlo.divide %v4926, %v4925 : tensor<32xf32>
    %v4928 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4929 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4930 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v4931 = stablehlo.reduce(%v4928 init: %v4929) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4932 = stablehlo.broadcast_in_dim %v4931, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4933 = stablehlo.divide %v4932, %v4930 : tensor<32x32x112x112xf32>
    %v4934 = stablehlo.subtract %v4928, %v4933 : tensor<32x32x112x112xf32>
    %v4935 = stablehlo.multiply %v4934, %v4934 : tensor<32x32x112x112xf32>
    %v4936 = stablehlo.reduce(%v4935 init: %v4929) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4937 = stablehlo.constant dense<401408.0> : tensor<32xf32>
    %v4938 = stablehlo.divide %v4936, %v4937 : tensor<32xf32>
    %v4939 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4940 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4941 = stablehlo.constant dense<401408.0> : tensor<32xf32>
    %v4942 = stablehlo.reduce(%v4939 init: %v4940) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4943 = stablehlo.divide %v4942, %v4941 : tensor<32xf32>
    %v4944 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4946 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v4947 = stablehlo.reduce(%v4944 init: %v4945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4948 = stablehlo.broadcast_in_dim %v4947, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v4949 = stablehlo.divide %v4948, %v4946 : tensor<32x32x112x112xf32>
    %v4950 = stablehlo.subtract %v4944, %v4949 : tensor<32x32x112x112xf32>
    %v4951 = stablehlo.multiply %v4950, %v4950 : tensor<32x32x112x112xf32>
    %v4952 = stablehlo.reduce(%v4951 init: %v4945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v4953 = stablehlo.constant dense<401408.0> : tensor<32xf32>
    %v4954 = stablehlo.divide %v4952, %v4953 : tensor<32xf32>
    %v4955 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4956 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4957 = stablehlo.constant dense<401408.0> : tensor<16xf32>
    %v4958 = stablehlo.reduce(%v4955 init: %v4956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4959 = stablehlo.divide %v4958, %v4957 : tensor<16xf32>
    %v4960 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4961 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4962 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v4963 = stablehlo.reduce(%v4960 init: %v4961) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4964 = stablehlo.broadcast_in_dim %v4963, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4965 = stablehlo.divide %v4964, %v4962 : tensor<32x16x112x112xf32>
    %v4966 = stablehlo.subtract %v4960, %v4965 : tensor<32x16x112x112xf32>
    %v4967 = stablehlo.multiply %v4966, %v4966 : tensor<32x16x112x112xf32>
    %v4968 = stablehlo.reduce(%v4967 init: %v4961) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4969 = stablehlo.constant dense<401408.0> : tensor<16xf32>
    %v4970 = stablehlo.divide %v4968, %v4969 : tensor<16xf32>
    %v4971 = stablehlo.reshape %v87 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4972 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4973 = stablehlo.constant dense<401408.0> : tensor<96xf32>
    %v4974 = stablehlo.reduce(%v4971 init: %v4972) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4975 = stablehlo.divide %v4974, %v4973 : tensor<96xf32>
    %v4976 = stablehlo.reshape %v87 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4977 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4978 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v4979 = stablehlo.reduce(%v4976 init: %v4977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4980 = stablehlo.broadcast_in_dim %v4979, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4981 = stablehlo.divide %v4980, %v4978 : tensor<32x96x112x112xf32>
    %v4982 = stablehlo.subtract %v4976, %v4981 : tensor<32x96x112x112xf32>
    %v4983 = stablehlo.multiply %v4982, %v4982 : tensor<32x96x112x112xf32>
    %v4984 = stablehlo.reduce(%v4983 init: %v4977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4985 = stablehlo.constant dense<401408.0> : tensor<96xf32>
    %v4986 = stablehlo.divide %v4984, %v4985 : tensor<96xf32>
    %v4987 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4988 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4989 = stablehlo.constant dense<100352.0> : tensor<96xf32>
    %v4990 = stablehlo.reduce(%v4987 init: %v4988) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4991 = stablehlo.divide %v4990, %v4989 : tensor<96xf32>
    %v4992 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4993 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4994 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v4995 = stablehlo.reduce(%v4992 init: %v4993) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4996 = stablehlo.broadcast_in_dim %v4995, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4997 = stablehlo.divide %v4996, %v4994 : tensor<32x96x56x56xf32>
    %v4998 = stablehlo.subtract %v4992, %v4997 : tensor<32x96x56x56xf32>
    %v4999 = stablehlo.multiply %v4998, %v4998 : tensor<32x96x56x56xf32>
    %v5000 = stablehlo.reduce(%v4999 init: %v4993) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5001 = stablehlo.constant dense<100352.0> : tensor<96xf32>
    %v5002 = stablehlo.divide %v5000, %v5001 : tensor<96xf32>
    %v5003 = stablehlo.reshape %v145 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5004 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5005 = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %v5006 = stablehlo.reduce(%v5003 init: %v5004) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5007 = stablehlo.divide %v5006, %v5005 : tensor<24xf32>
    %v5008 = stablehlo.reshape %v145 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5010 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v5011 = stablehlo.reduce(%v5008 init: %v5009) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5012 = stablehlo.broadcast_in_dim %v5011, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5013 = stablehlo.divide %v5012, %v5010 : tensor<32x24x56x56xf32>
    %v5014 = stablehlo.subtract %v5008, %v5013 : tensor<32x24x56x56xf32>
    %v5015 = stablehlo.multiply %v5014, %v5014 : tensor<32x24x56x56xf32>
    %v5016 = stablehlo.reduce(%v5015 init: %v5009) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5017 = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %v5018 = stablehlo.divide %v5016, %v5017 : tensor<24xf32>
    %v5019 = stablehlo.reshape %v170 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5020 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5021 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5022 = stablehlo.reduce(%v5019 init: %v5020) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5023 = stablehlo.divide %v5022, %v5021 : tensor<144xf32>
    %v5024 = stablehlo.reshape %v170 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5025 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5026 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5027 = stablehlo.reduce(%v5024 init: %v5025) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5028 = stablehlo.broadcast_in_dim %v5027, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5029 = stablehlo.divide %v5028, %v5026 : tensor<32x144x56x56xf32>
    %v5030 = stablehlo.subtract %v5024, %v5029 : tensor<32x144x56x56xf32>
    %v5031 = stablehlo.multiply %v5030, %v5030 : tensor<32x144x56x56xf32>
    %v5032 = stablehlo.reduce(%v5031 init: %v5025) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5033 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5034 = stablehlo.divide %v5032, %v5033 : tensor<144xf32>
    %v5035 = stablehlo.reshape %v199 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5036 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5037 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5038 = stablehlo.reduce(%v5035 init: %v5036) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5039 = stablehlo.divide %v5038, %v5037 : tensor<144xf32>
    %v5040 = stablehlo.reshape %v199 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5041 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5042 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5043 = stablehlo.reduce(%v5040 init: %v5041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5044 = stablehlo.broadcast_in_dim %v5043, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5045 = stablehlo.divide %v5044, %v5042 : tensor<32x144x56x56xf32>
    %v5046 = stablehlo.subtract %v5040, %v5045 : tensor<32x144x56x56xf32>
    %v5047 = stablehlo.multiply %v5046, %v5046 : tensor<32x144x56x56xf32>
    %v5048 = stablehlo.reduce(%v5047 init: %v5041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5049 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5050 = stablehlo.divide %v5048, %v5049 : tensor<144xf32>
    %v5051 = stablehlo.reshape %v228 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5052 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5053 = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %v5054 = stablehlo.reduce(%v5051 init: %v5052) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5055 = stablehlo.divide %v5054, %v5053 : tensor<24xf32>
    %v5056 = stablehlo.reshape %v228 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5057 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5058 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v5059 = stablehlo.reduce(%v5056 init: %v5057) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5060 = stablehlo.broadcast_in_dim %v5059, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5061 = stablehlo.divide %v5060, %v5058 : tensor<32x24x56x56xf32>
    %v5062 = stablehlo.subtract %v5056, %v5061 : tensor<32x24x56x56xf32>
    %v5063 = stablehlo.multiply %v5062, %v5062 : tensor<32x24x56x56xf32>
    %v5064 = stablehlo.reduce(%v5063 init: %v5057) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5065 = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %v5066 = stablehlo.divide %v5064, %v5065 : tensor<24xf32>
    %v5067 = stablehlo.reshape %v254 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5069 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5070 = stablehlo.reduce(%v5067 init: %v5068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5071 = stablehlo.divide %v5070, %v5069 : tensor<144xf32>
    %v5072 = stablehlo.reshape %v254 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5074 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5075 = stablehlo.reduce(%v5072 init: %v5073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5076 = stablehlo.broadcast_in_dim %v5075, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5077 = stablehlo.divide %v5076, %v5074 : tensor<32x144x56x56xf32>
    %v5078 = stablehlo.subtract %v5072, %v5077 : tensor<32x144x56x56xf32>
    %v5079 = stablehlo.multiply %v5078, %v5078 : tensor<32x144x56x56xf32>
    %v5080 = stablehlo.reduce(%v5079 init: %v5073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5081 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5082 = stablehlo.divide %v5080, %v5081 : tensor<144xf32>
    %v5083 = stablehlo.reshape %v283 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5084 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5085 = stablehlo.constant dense<25088.0> : tensor<144xf32>
    %v5086 = stablehlo.reduce(%v5083 init: %v5084) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5087 = stablehlo.divide %v5086, %v5085 : tensor<144xf32>
    %v5088 = stablehlo.reshape %v283 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5089 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5090 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v5091 = stablehlo.reduce(%v5088 init: %v5089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5092 = stablehlo.broadcast_in_dim %v5091, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5093 = stablehlo.divide %v5092, %v5090 : tensor<32x144x28x28xf32>
    %v5094 = stablehlo.subtract %v5088, %v5093 : tensor<32x144x28x28xf32>
    %v5095 = stablehlo.multiply %v5094, %v5094 : tensor<32x144x28x28xf32>
    %v5096 = stablehlo.reduce(%v5095 init: %v5089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5097 = stablehlo.constant dense<25088.0> : tensor<144xf32>
    %v5098 = stablehlo.divide %v5096, %v5097 : tensor<144xf32>
    %v5099 = stablehlo.reshape %v312 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5100 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5101 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5102 = stablehlo.reduce(%v5099 init: %v5100) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5103 = stablehlo.divide %v5102, %v5101 : tensor<32xf32>
    %v5104 = stablehlo.reshape %v312 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5105 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5106 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v5107 = stablehlo.reduce(%v5104 init: %v5105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5108 = stablehlo.broadcast_in_dim %v5107, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v5109 = stablehlo.divide %v5108, %v5106 : tensor<32x32x28x28xf32>
    %v5110 = stablehlo.subtract %v5104, %v5109 : tensor<32x32x28x28xf32>
    %v5111 = stablehlo.multiply %v5110, %v5110 : tensor<32x32x28x28xf32>
    %v5112 = stablehlo.reduce(%v5111 init: %v5105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5113 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5114 = stablehlo.divide %v5112, %v5113 : tensor<32xf32>
    %v5115 = stablehlo.reshape %v337 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5116 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5117 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5118 = stablehlo.reduce(%v5115 init: %v5116) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5119 = stablehlo.divide %v5118, %v5117 : tensor<192xf32>
    %v5120 = stablehlo.reshape %v337 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5122 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v5123 = stablehlo.reduce(%v5120 init: %v5121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5124 = stablehlo.broadcast_in_dim %v5123, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v5125 = stablehlo.divide %v5124, %v5122 : tensor<32x192x28x28xf32>
    %v5126 = stablehlo.subtract %v5120, %v5125 : tensor<32x192x28x28xf32>
    %v5127 = stablehlo.multiply %v5126, %v5126 : tensor<32x192x28x28xf32>
    %v5128 = stablehlo.reduce(%v5127 init: %v5121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5129 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5130 = stablehlo.divide %v5128, %v5129 : tensor<192xf32>
    %v5131 = stablehlo.reshape %v366 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5132 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5133 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5134 = stablehlo.reduce(%v5131 init: %v5132) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5135 = stablehlo.divide %v5134, %v5133 : tensor<192xf32>
    %v5136 = stablehlo.reshape %v366 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5137 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5138 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v5139 = stablehlo.reduce(%v5136 init: %v5137) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5140 = stablehlo.broadcast_in_dim %v5139, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v5141 = stablehlo.divide %v5140, %v5138 : tensor<32x192x28x28xf32>
    %v5142 = stablehlo.subtract %v5136, %v5141 : tensor<32x192x28x28xf32>
    %v5143 = stablehlo.multiply %v5142, %v5142 : tensor<32x192x28x28xf32>
    %v5144 = stablehlo.reduce(%v5143 init: %v5137) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5145 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5146 = stablehlo.divide %v5144, %v5145 : tensor<192xf32>
    %v5147 = stablehlo.reshape %v395 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5148 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5149 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5150 = stablehlo.reduce(%v5147 init: %v5148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5151 = stablehlo.divide %v5150, %v5149 : tensor<32xf32>
    %v5152 = stablehlo.reshape %v395 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5153 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5154 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v5155 = stablehlo.reduce(%v5152 init: %v5153) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5156 = stablehlo.broadcast_in_dim %v5155, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v5157 = stablehlo.divide %v5156, %v5154 : tensor<32x32x28x28xf32>
    %v5158 = stablehlo.subtract %v5152, %v5157 : tensor<32x32x28x28xf32>
    %v5159 = stablehlo.multiply %v5158, %v5158 : tensor<32x32x28x28xf32>
    %v5160 = stablehlo.reduce(%v5159 init: %v5153) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5161 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5162 = stablehlo.divide %v5160, %v5161 : tensor<32xf32>
    %v5163 = stablehlo.reshape %v421 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5165 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5166 = stablehlo.reduce(%v5163 init: %v5164) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5167 = stablehlo.divide %v5166, %v5165 : tensor<192xf32>
    %v5168 = stablehlo.reshape %v421 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5169 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5170 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v5171 = stablehlo.reduce(%v5168 init: %v5169) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5172 = stablehlo.broadcast_in_dim %v5171, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v5173 = stablehlo.divide %v5172, %v5170 : tensor<32x192x28x28xf32>
    %v5174 = stablehlo.subtract %v5168, %v5173 : tensor<32x192x28x28xf32>
    %v5175 = stablehlo.multiply %v5174, %v5174 : tensor<32x192x28x28xf32>
    %v5176 = stablehlo.reduce(%v5175 init: %v5169) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5177 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5178 = stablehlo.divide %v5176, %v5177 : tensor<192xf32>
    %v5179 = stablehlo.reshape %v450 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5181 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5182 = stablehlo.reduce(%v5179 init: %v5180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5183 = stablehlo.divide %v5182, %v5181 : tensor<192xf32>
    %v5184 = stablehlo.reshape %v450 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5186 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v5187 = stablehlo.reduce(%v5184 init: %v5185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5188 = stablehlo.broadcast_in_dim %v5187, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v5189 = stablehlo.divide %v5188, %v5186 : tensor<32x192x28x28xf32>
    %v5190 = stablehlo.subtract %v5184, %v5189 : tensor<32x192x28x28xf32>
    %v5191 = stablehlo.multiply %v5190, %v5190 : tensor<32x192x28x28xf32>
    %v5192 = stablehlo.reduce(%v5191 init: %v5185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5193 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5194 = stablehlo.divide %v5192, %v5193 : tensor<192xf32>
    %v5195 = stablehlo.reshape %v479 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5196 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5197 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5198 = stablehlo.reduce(%v5195 init: %v5196) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5199 = stablehlo.divide %v5198, %v5197 : tensor<32xf32>
    %v5200 = stablehlo.reshape %v479 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5201 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5202 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v5203 = stablehlo.reduce(%v5200 init: %v5201) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5204 = stablehlo.broadcast_in_dim %v5203, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v5205 = stablehlo.divide %v5204, %v5202 : tensor<32x32x28x28xf32>
    %v5206 = stablehlo.subtract %v5200, %v5205 : tensor<32x32x28x28xf32>
    %v5207 = stablehlo.multiply %v5206, %v5206 : tensor<32x32x28x28xf32>
    %v5208 = stablehlo.reduce(%v5207 init: %v5201) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5209 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5210 = stablehlo.divide %v5208, %v5209 : tensor<32xf32>
    %v5211 = stablehlo.reshape %v505 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5212 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5213 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5214 = stablehlo.reduce(%v5211 init: %v5212) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5215 = stablehlo.divide %v5214, %v5213 : tensor<192xf32>
    %v5216 = stablehlo.reshape %v505 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5217 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5218 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v5219 = stablehlo.reduce(%v5216 init: %v5217) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5220 = stablehlo.broadcast_in_dim %v5219, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v5221 = stablehlo.divide %v5220, %v5218 : tensor<32x192x28x28xf32>
    %v5222 = stablehlo.subtract %v5216, %v5221 : tensor<32x192x28x28xf32>
    %v5223 = stablehlo.multiply %v5222, %v5222 : tensor<32x192x28x28xf32>
    %v5224 = stablehlo.reduce(%v5223 init: %v5217) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5225 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5226 = stablehlo.divide %v5224, %v5225 : tensor<192xf32>
    %v5227 = stablehlo.reshape %v534 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v5228 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5229 = stablehlo.constant dense<6272.0> : tensor<192xf32>
    %v5230 = stablehlo.reduce(%v5227 init: %v5228) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v5231 = stablehlo.divide %v5230, %v5229 : tensor<192xf32>
    %v5232 = stablehlo.reshape %v534 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v5233 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5234 = stablehlo.constant dense<6272.0> : tensor<32x192x14x14xf32>
    %v5235 = stablehlo.reduce(%v5232 init: %v5233) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v5236 = stablehlo.broadcast_in_dim %v5235, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v5237 = stablehlo.divide %v5236, %v5234 : tensor<32x192x14x14xf32>
    %v5238 = stablehlo.subtract %v5232, %v5237 : tensor<32x192x14x14xf32>
    %v5239 = stablehlo.multiply %v5238, %v5238 : tensor<32x192x14x14xf32>
    %v5240 = stablehlo.reduce(%v5239 init: %v5233) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v5241 = stablehlo.constant dense<6272.0> : tensor<192xf32>
    %v5242 = stablehlo.divide %v5240, %v5241 : tensor<192xf32>
    %v5243 = stablehlo.reshape %v563 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5245 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5246 = stablehlo.reduce(%v5243 init: %v5244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5247 = stablehlo.divide %v5246, %v5245 : tensor<64xf32>
    %v5248 = stablehlo.reshape %v563 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5249 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5250 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v5251 = stablehlo.reduce(%v5248 init: %v5249) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5252 = stablehlo.broadcast_in_dim %v5251, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v5253 = stablehlo.divide %v5252, %v5250 : tensor<32x64x14x14xf32>
    %v5254 = stablehlo.subtract %v5248, %v5253 : tensor<32x64x14x14xf32>
    %v5255 = stablehlo.multiply %v5254, %v5254 : tensor<32x64x14x14xf32>
    %v5256 = stablehlo.reduce(%v5255 init: %v5249) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5257 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5258 = stablehlo.divide %v5256, %v5257 : tensor<64xf32>
    %v5259 = stablehlo.reshape %v588 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5261 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5262 = stablehlo.reduce(%v5259 init: %v5260) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5263 = stablehlo.divide %v5262, %v5261 : tensor<384xf32>
    %v5264 = stablehlo.reshape %v588 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5266 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5267 = stablehlo.reduce(%v5264 init: %v5265) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5268 = stablehlo.broadcast_in_dim %v5267, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5269 = stablehlo.divide %v5268, %v5266 : tensor<32x384x14x14xf32>
    %v5270 = stablehlo.subtract %v5264, %v5269 : tensor<32x384x14x14xf32>
    %v5271 = stablehlo.multiply %v5270, %v5270 : tensor<32x384x14x14xf32>
    %v5272 = stablehlo.reduce(%v5271 init: %v5265) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5273 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5274 = stablehlo.divide %v5272, %v5273 : tensor<384xf32>
    %v5275 = stablehlo.reshape %v617 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5276 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5277 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5278 = stablehlo.reduce(%v5275 init: %v5276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5279 = stablehlo.divide %v5278, %v5277 : tensor<384xf32>
    %v5280 = stablehlo.reshape %v617 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5281 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5282 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5283 = stablehlo.reduce(%v5280 init: %v5281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5284 = stablehlo.broadcast_in_dim %v5283, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5285 = stablehlo.divide %v5284, %v5282 : tensor<32x384x14x14xf32>
    %v5286 = stablehlo.subtract %v5280, %v5285 : tensor<32x384x14x14xf32>
    %v5287 = stablehlo.multiply %v5286, %v5286 : tensor<32x384x14x14xf32>
    %v5288 = stablehlo.reduce(%v5287 init: %v5281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5289 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5290 = stablehlo.divide %v5288, %v5289 : tensor<384xf32>
    %v5291 = stablehlo.reshape %v646 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5292 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5293 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5294 = stablehlo.reduce(%v5291 init: %v5292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5295 = stablehlo.divide %v5294, %v5293 : tensor<64xf32>
    %v5296 = stablehlo.reshape %v646 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5298 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v5299 = stablehlo.reduce(%v5296 init: %v5297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5300 = stablehlo.broadcast_in_dim %v5299, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v5301 = stablehlo.divide %v5300, %v5298 : tensor<32x64x14x14xf32>
    %v5302 = stablehlo.subtract %v5296, %v5301 : tensor<32x64x14x14xf32>
    %v5303 = stablehlo.multiply %v5302, %v5302 : tensor<32x64x14x14xf32>
    %v5304 = stablehlo.reduce(%v5303 init: %v5297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5305 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5306 = stablehlo.divide %v5304, %v5305 : tensor<64xf32>
    %v5307 = stablehlo.reshape %v672 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5309 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5310 = stablehlo.reduce(%v5307 init: %v5308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5311 = stablehlo.divide %v5310, %v5309 : tensor<384xf32>
    %v5312 = stablehlo.reshape %v672 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5313 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5314 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5315 = stablehlo.reduce(%v5312 init: %v5313) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5316 = stablehlo.broadcast_in_dim %v5315, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5317 = stablehlo.divide %v5316, %v5314 : tensor<32x384x14x14xf32>
    %v5318 = stablehlo.subtract %v5312, %v5317 : tensor<32x384x14x14xf32>
    %v5319 = stablehlo.multiply %v5318, %v5318 : tensor<32x384x14x14xf32>
    %v5320 = stablehlo.reduce(%v5319 init: %v5313) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5321 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5322 = stablehlo.divide %v5320, %v5321 : tensor<384xf32>
    %v5323 = stablehlo.reshape %v701 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5324 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5325 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5326 = stablehlo.reduce(%v5323 init: %v5324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5327 = stablehlo.divide %v5326, %v5325 : tensor<384xf32>
    %v5328 = stablehlo.reshape %v701 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5329 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5330 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5331 = stablehlo.reduce(%v5328 init: %v5329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5332 = stablehlo.broadcast_in_dim %v5331, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5333 = stablehlo.divide %v5332, %v5330 : tensor<32x384x14x14xf32>
    %v5334 = stablehlo.subtract %v5328, %v5333 : tensor<32x384x14x14xf32>
    %v5335 = stablehlo.multiply %v5334, %v5334 : tensor<32x384x14x14xf32>
    %v5336 = stablehlo.reduce(%v5335 init: %v5329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5337 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5338 = stablehlo.divide %v5336, %v5337 : tensor<384xf32>
    %v5339 = stablehlo.reshape %v730 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5340 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5341 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5342 = stablehlo.reduce(%v5339 init: %v5340) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5343 = stablehlo.divide %v5342, %v5341 : tensor<64xf32>
    %v5344 = stablehlo.reshape %v730 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5345 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5346 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v5347 = stablehlo.reduce(%v5344 init: %v5345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5348 = stablehlo.broadcast_in_dim %v5347, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v5349 = stablehlo.divide %v5348, %v5346 : tensor<32x64x14x14xf32>
    %v5350 = stablehlo.subtract %v5344, %v5349 : tensor<32x64x14x14xf32>
    %v5351 = stablehlo.multiply %v5350, %v5350 : tensor<32x64x14x14xf32>
    %v5352 = stablehlo.reduce(%v5351 init: %v5345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5353 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5354 = stablehlo.divide %v5352, %v5353 : tensor<64xf32>
    %v5355 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5357 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5358 = stablehlo.reduce(%v5355 init: %v5356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5359 = stablehlo.divide %v5358, %v5357 : tensor<384xf32>
    %v5360 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5362 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5363 = stablehlo.reduce(%v5360 init: %v5361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5364 = stablehlo.broadcast_in_dim %v5363, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5365 = stablehlo.divide %v5364, %v5362 : tensor<32x384x14x14xf32>
    %v5366 = stablehlo.subtract %v5360, %v5365 : tensor<32x384x14x14xf32>
    %v5367 = stablehlo.multiply %v5366, %v5366 : tensor<32x384x14x14xf32>
    %v5368 = stablehlo.reduce(%v5367 init: %v5361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5369 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5370 = stablehlo.divide %v5368, %v5369 : tensor<384xf32>
    %v5371 = stablehlo.reshape %v785 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5372 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5373 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5374 = stablehlo.reduce(%v5371 init: %v5372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5375 = stablehlo.divide %v5374, %v5373 : tensor<384xf32>
    %v5376 = stablehlo.reshape %v785 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5378 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5379 = stablehlo.reduce(%v5376 init: %v5377) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5380 = stablehlo.broadcast_in_dim %v5379, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5381 = stablehlo.divide %v5380, %v5378 : tensor<32x384x14x14xf32>
    %v5382 = stablehlo.subtract %v5376, %v5381 : tensor<32x384x14x14xf32>
    %v5383 = stablehlo.multiply %v5382, %v5382 : tensor<32x384x14x14xf32>
    %v5384 = stablehlo.reduce(%v5383 init: %v5377) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5385 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5386 = stablehlo.divide %v5384, %v5385 : tensor<384xf32>
    %v5387 = stablehlo.reshape %v814 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5388 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5389 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5390 = stablehlo.reduce(%v5387 init: %v5388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5391 = stablehlo.divide %v5390, %v5389 : tensor<64xf32>
    %v5392 = stablehlo.reshape %v814 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5393 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5394 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v5395 = stablehlo.reduce(%v5392 init: %v5393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5396 = stablehlo.broadcast_in_dim %v5395, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v5397 = stablehlo.divide %v5396, %v5394 : tensor<32x64x14x14xf32>
    %v5398 = stablehlo.subtract %v5392, %v5397 : tensor<32x64x14x14xf32>
    %v5399 = stablehlo.multiply %v5398, %v5398 : tensor<32x64x14x14xf32>
    %v5400 = stablehlo.reduce(%v5399 init: %v5393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5401 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5402 = stablehlo.divide %v5400, %v5401 : tensor<64xf32>
    %v5403 = stablehlo.reshape %v840 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5405 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5406 = stablehlo.reduce(%v5403 init: %v5404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5407 = stablehlo.divide %v5406, %v5405 : tensor<384xf32>
    %v5408 = stablehlo.reshape %v840 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5409 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5410 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5411 = stablehlo.reduce(%v5408 init: %v5409) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5412 = stablehlo.broadcast_in_dim %v5411, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5413 = stablehlo.divide %v5412, %v5410 : tensor<32x384x14x14xf32>
    %v5414 = stablehlo.subtract %v5408, %v5413 : tensor<32x384x14x14xf32>
    %v5415 = stablehlo.multiply %v5414, %v5414 : tensor<32x384x14x14xf32>
    %v5416 = stablehlo.reduce(%v5415 init: %v5409) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5417 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5418 = stablehlo.divide %v5416, %v5417 : tensor<384xf32>
    %v5419 = stablehlo.reshape %v869 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5421 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5422 = stablehlo.reduce(%v5419 init: %v5420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5423 = stablehlo.divide %v5422, %v5421 : tensor<384xf32>
    %v5424 = stablehlo.reshape %v869 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5425 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5426 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5427 = stablehlo.reduce(%v5424 init: %v5425) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5428 = stablehlo.broadcast_in_dim %v5427, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5429 = stablehlo.divide %v5428, %v5426 : tensor<32x384x14x14xf32>
    %v5430 = stablehlo.subtract %v5424, %v5429 : tensor<32x384x14x14xf32>
    %v5431 = stablehlo.multiply %v5430, %v5430 : tensor<32x384x14x14xf32>
    %v5432 = stablehlo.reduce(%v5431 init: %v5425) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5433 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5434 = stablehlo.divide %v5432, %v5433 : tensor<384xf32>
    %v5435 = stablehlo.reshape %v898 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5436 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5437 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5438 = stablehlo.reduce(%v5435 init: %v5436) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5439 = stablehlo.divide %v5438, %v5437 : tensor<96xf32>
    %v5440 = stablehlo.reshape %v898 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5441 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5442 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v5443 = stablehlo.reduce(%v5440 init: %v5441) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5444 = stablehlo.broadcast_in_dim %v5443, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v5445 = stablehlo.divide %v5444, %v5442 : tensor<32x96x14x14xf32>
    %v5446 = stablehlo.subtract %v5440, %v5445 : tensor<32x96x14x14xf32>
    %v5447 = stablehlo.multiply %v5446, %v5446 : tensor<32x96x14x14xf32>
    %v5448 = stablehlo.reduce(%v5447 init: %v5441) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5449 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5450 = stablehlo.divide %v5448, %v5449 : tensor<96xf32>
    %v5451 = stablehlo.reshape %v923 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5453 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5454 = stablehlo.reduce(%v5451 init: %v5452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5455 = stablehlo.divide %v5454, %v5453 : tensor<576xf32>
    %v5456 = stablehlo.reshape %v923 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5457 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5458 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v5459 = stablehlo.reduce(%v5456 init: %v5457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5460 = stablehlo.broadcast_in_dim %v5459, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v5461 = stablehlo.divide %v5460, %v5458 : tensor<32x576x14x14xf32>
    %v5462 = stablehlo.subtract %v5456, %v5461 : tensor<32x576x14x14xf32>
    %v5463 = stablehlo.multiply %v5462, %v5462 : tensor<32x576x14x14xf32>
    %v5464 = stablehlo.reduce(%v5463 init: %v5457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5465 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5466 = stablehlo.divide %v5464, %v5465 : tensor<576xf32>
    %v5467 = stablehlo.reshape %v952 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5469 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5470 = stablehlo.reduce(%v5467 init: %v5468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5471 = stablehlo.divide %v5470, %v5469 : tensor<576xf32>
    %v5472 = stablehlo.reshape %v952 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5474 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v5475 = stablehlo.reduce(%v5472 init: %v5473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5476 = stablehlo.broadcast_in_dim %v5475, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v5477 = stablehlo.divide %v5476, %v5474 : tensor<32x576x14x14xf32>
    %v5478 = stablehlo.subtract %v5472, %v5477 : tensor<32x576x14x14xf32>
    %v5479 = stablehlo.multiply %v5478, %v5478 : tensor<32x576x14x14xf32>
    %v5480 = stablehlo.reduce(%v5479 init: %v5473) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5481 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5482 = stablehlo.divide %v5480, %v5481 : tensor<576xf32>
    %v5483 = stablehlo.reshape %v981 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5485 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5486 = stablehlo.reduce(%v5483 init: %v5484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5487 = stablehlo.divide %v5486, %v5485 : tensor<96xf32>
    %v5488 = stablehlo.reshape %v981 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5490 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v5491 = stablehlo.reduce(%v5488 init: %v5489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5492 = stablehlo.broadcast_in_dim %v5491, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v5493 = stablehlo.divide %v5492, %v5490 : tensor<32x96x14x14xf32>
    %v5494 = stablehlo.subtract %v5488, %v5493 : tensor<32x96x14x14xf32>
    %v5495 = stablehlo.multiply %v5494, %v5494 : tensor<32x96x14x14xf32>
    %v5496 = stablehlo.reduce(%v5495 init: %v5489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5497 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5498 = stablehlo.divide %v5496, %v5497 : tensor<96xf32>
    %v5499 = stablehlo.reshape %v1007 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5500 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5501 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5502 = stablehlo.reduce(%v5499 init: %v5500) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5503 = stablehlo.divide %v5502, %v5501 : tensor<576xf32>
    %v5504 = stablehlo.reshape %v1007 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5506 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v5507 = stablehlo.reduce(%v5504 init: %v5505) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5508 = stablehlo.broadcast_in_dim %v5507, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v5509 = stablehlo.divide %v5508, %v5506 : tensor<32x576x14x14xf32>
    %v5510 = stablehlo.subtract %v5504, %v5509 : tensor<32x576x14x14xf32>
    %v5511 = stablehlo.multiply %v5510, %v5510 : tensor<32x576x14x14xf32>
    %v5512 = stablehlo.reduce(%v5511 init: %v5505) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5513 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5514 = stablehlo.divide %v5512, %v5513 : tensor<576xf32>
    %v5515 = stablehlo.reshape %v1036 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5517 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5518 = stablehlo.reduce(%v5515 init: %v5516) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5519 = stablehlo.divide %v5518, %v5517 : tensor<576xf32>
    %v5520 = stablehlo.reshape %v1036 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5521 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5522 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v5523 = stablehlo.reduce(%v5520 init: %v5521) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5524 = stablehlo.broadcast_in_dim %v5523, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v5525 = stablehlo.divide %v5524, %v5522 : tensor<32x576x14x14xf32>
    %v5526 = stablehlo.subtract %v5520, %v5525 : tensor<32x576x14x14xf32>
    %v5527 = stablehlo.multiply %v5526, %v5526 : tensor<32x576x14x14xf32>
    %v5528 = stablehlo.reduce(%v5527 init: %v5521) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5529 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5530 = stablehlo.divide %v5528, %v5529 : tensor<576xf32>
    %v5531 = stablehlo.reshape %v1065 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5532 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5533 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5534 = stablehlo.reduce(%v5531 init: %v5532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5535 = stablehlo.divide %v5534, %v5533 : tensor<96xf32>
    %v5536 = stablehlo.reshape %v1065 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5537 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5538 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v5539 = stablehlo.reduce(%v5536 init: %v5537) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5540 = stablehlo.broadcast_in_dim %v5539, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v5541 = stablehlo.divide %v5540, %v5538 : tensor<32x96x14x14xf32>
    %v5542 = stablehlo.subtract %v5536, %v5541 : tensor<32x96x14x14xf32>
    %v5543 = stablehlo.multiply %v5542, %v5542 : tensor<32x96x14x14xf32>
    %v5544 = stablehlo.reduce(%v5543 init: %v5537) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5545 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5546 = stablehlo.divide %v5544, %v5545 : tensor<96xf32>
    %v5547 = stablehlo.reshape %v1091 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5548 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5549 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5550 = stablehlo.reduce(%v5547 init: %v5548) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5551 = stablehlo.divide %v5550, %v5549 : tensor<576xf32>
    %v5552 = stablehlo.reshape %v1091 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5553 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5554 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v5555 = stablehlo.reduce(%v5552 init: %v5553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5556 = stablehlo.broadcast_in_dim %v5555, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v5557 = stablehlo.divide %v5556, %v5554 : tensor<32x576x14x14xf32>
    %v5558 = stablehlo.subtract %v5552, %v5557 : tensor<32x576x14x14xf32>
    %v5559 = stablehlo.multiply %v5558, %v5558 : tensor<32x576x14x14xf32>
    %v5560 = stablehlo.reduce(%v5559 init: %v5553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5561 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5562 = stablehlo.divide %v5560, %v5561 : tensor<576xf32>
    %v5563 = stablehlo.reshape %v1120 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v5564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5565 = stablehlo.constant dense<1568.0> : tensor<576xf32>
    %v5566 = stablehlo.reduce(%v5563 init: %v5564) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v5567 = stablehlo.divide %v5566, %v5565 : tensor<576xf32>
    %v5568 = stablehlo.reshape %v1120 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v5569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5570 = stablehlo.constant dense<1568.0> : tensor<32x576x7x7xf32>
    %v5571 = stablehlo.reduce(%v5568 init: %v5569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v5572 = stablehlo.broadcast_in_dim %v5571, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v5573 = stablehlo.divide %v5572, %v5570 : tensor<32x576x7x7xf32>
    %v5574 = stablehlo.subtract %v5568, %v5573 : tensor<32x576x7x7xf32>
    %v5575 = stablehlo.multiply %v5574, %v5574 : tensor<32x576x7x7xf32>
    %v5576 = stablehlo.reduce(%v5575 init: %v5569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v5577 = stablehlo.constant dense<1568.0> : tensor<576xf32>
    %v5578 = stablehlo.divide %v5576, %v5577 : tensor<576xf32>
    %v5579 = stablehlo.reshape %v1149 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5580 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5581 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5582 = stablehlo.reduce(%v5579 init: %v5580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5583 = stablehlo.divide %v5582, %v5581 : tensor<160xf32>
    %v5584 = stablehlo.reshape %v1149 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5586 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v5587 = stablehlo.reduce(%v5584 init: %v5585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5588 = stablehlo.broadcast_in_dim %v5587, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v5589 = stablehlo.divide %v5588, %v5586 : tensor<32x160x7x7xf32>
    %v5590 = stablehlo.subtract %v5584, %v5589 : tensor<32x160x7x7xf32>
    %v5591 = stablehlo.multiply %v5590, %v5590 : tensor<32x160x7x7xf32>
    %v5592 = stablehlo.reduce(%v5591 init: %v5585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5593 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5594 = stablehlo.divide %v5592, %v5593 : tensor<160xf32>
    %v5595 = stablehlo.reshape %v1174 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5596 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5597 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5598 = stablehlo.reduce(%v5595 init: %v5596) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5599 = stablehlo.divide %v5598, %v5597 : tensor<960xf32>
    %v5600 = stablehlo.reshape %v1174 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5601 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5602 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5603 = stablehlo.reduce(%v5600 init: %v5601) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5604 = stablehlo.broadcast_in_dim %v5603, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5605 = stablehlo.divide %v5604, %v5602 : tensor<32x960x7x7xf32>
    %v5606 = stablehlo.subtract %v5600, %v5605 : tensor<32x960x7x7xf32>
    %v5607 = stablehlo.multiply %v5606, %v5606 : tensor<32x960x7x7xf32>
    %v5608 = stablehlo.reduce(%v5607 init: %v5601) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5609 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5610 = stablehlo.divide %v5608, %v5609 : tensor<960xf32>
    %v5611 = stablehlo.reshape %v1203 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5612 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5613 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5614 = stablehlo.reduce(%v5611 init: %v5612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5615 = stablehlo.divide %v5614, %v5613 : tensor<960xf32>
    %v5616 = stablehlo.reshape %v1203 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5618 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5619 = stablehlo.reduce(%v5616 init: %v5617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5620 = stablehlo.broadcast_in_dim %v5619, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5621 = stablehlo.divide %v5620, %v5618 : tensor<32x960x7x7xf32>
    %v5622 = stablehlo.subtract %v5616, %v5621 : tensor<32x960x7x7xf32>
    %v5623 = stablehlo.multiply %v5622, %v5622 : tensor<32x960x7x7xf32>
    %v5624 = stablehlo.reduce(%v5623 init: %v5617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5625 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5626 = stablehlo.divide %v5624, %v5625 : tensor<960xf32>
    %v5627 = stablehlo.reshape %v1232 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5628 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5629 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5630 = stablehlo.reduce(%v5627 init: %v5628) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5631 = stablehlo.divide %v5630, %v5629 : tensor<160xf32>
    %v5632 = stablehlo.reshape %v1232 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5633 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5634 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v5635 = stablehlo.reduce(%v5632 init: %v5633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5636 = stablehlo.broadcast_in_dim %v5635, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v5637 = stablehlo.divide %v5636, %v5634 : tensor<32x160x7x7xf32>
    %v5638 = stablehlo.subtract %v5632, %v5637 : tensor<32x160x7x7xf32>
    %v5639 = stablehlo.multiply %v5638, %v5638 : tensor<32x160x7x7xf32>
    %v5640 = stablehlo.reduce(%v5639 init: %v5633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5641 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5642 = stablehlo.divide %v5640, %v5641 : tensor<160xf32>
    %v5643 = stablehlo.reshape %v1258 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5644 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5645 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5646 = stablehlo.reduce(%v5643 init: %v5644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5647 = stablehlo.divide %v5646, %v5645 : tensor<960xf32>
    %v5648 = stablehlo.reshape %v1258 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5649 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5650 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5651 = stablehlo.reduce(%v5648 init: %v5649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5652 = stablehlo.broadcast_in_dim %v5651, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5653 = stablehlo.divide %v5652, %v5650 : tensor<32x960x7x7xf32>
    %v5654 = stablehlo.subtract %v5648, %v5653 : tensor<32x960x7x7xf32>
    %v5655 = stablehlo.multiply %v5654, %v5654 : tensor<32x960x7x7xf32>
    %v5656 = stablehlo.reduce(%v5655 init: %v5649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5657 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5658 = stablehlo.divide %v5656, %v5657 : tensor<960xf32>
    %v5659 = stablehlo.reshape %v1287 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5660 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5661 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5662 = stablehlo.reduce(%v5659 init: %v5660) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5663 = stablehlo.divide %v5662, %v5661 : tensor<960xf32>
    %v5664 = stablehlo.reshape %v1287 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5666 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5667 = stablehlo.reduce(%v5664 init: %v5665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5668 = stablehlo.broadcast_in_dim %v5667, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5669 = stablehlo.divide %v5668, %v5666 : tensor<32x960x7x7xf32>
    %v5670 = stablehlo.subtract %v5664, %v5669 : tensor<32x960x7x7xf32>
    %v5671 = stablehlo.multiply %v5670, %v5670 : tensor<32x960x7x7xf32>
    %v5672 = stablehlo.reduce(%v5671 init: %v5665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5673 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5674 = stablehlo.divide %v5672, %v5673 : tensor<960xf32>
    %v5675 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5677 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5678 = stablehlo.reduce(%v5675 init: %v5676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5679 = stablehlo.divide %v5678, %v5677 : tensor<160xf32>
    %v5680 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5681 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5682 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v5683 = stablehlo.reduce(%v5680 init: %v5681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5684 = stablehlo.broadcast_in_dim %v5683, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v5685 = stablehlo.divide %v5684, %v5682 : tensor<32x160x7x7xf32>
    %v5686 = stablehlo.subtract %v5680, %v5685 : tensor<32x160x7x7xf32>
    %v5687 = stablehlo.multiply %v5686, %v5686 : tensor<32x160x7x7xf32>
    %v5688 = stablehlo.reduce(%v5687 init: %v5681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5689 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5690 = stablehlo.divide %v5688, %v5689 : tensor<160xf32>
    %v5691 = stablehlo.reshape %v1342 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5693 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5694 = stablehlo.reduce(%v5691 init: %v5692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5695 = stablehlo.divide %v5694, %v5693 : tensor<960xf32>
    %v5696 = stablehlo.reshape %v1342 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5697 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5698 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5699 = stablehlo.reduce(%v5696 init: %v5697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5700 = stablehlo.broadcast_in_dim %v5699, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5701 = stablehlo.divide %v5700, %v5698 : tensor<32x960x7x7xf32>
    %v5702 = stablehlo.subtract %v5696, %v5701 : tensor<32x960x7x7xf32>
    %v5703 = stablehlo.multiply %v5702, %v5702 : tensor<32x960x7x7xf32>
    %v5704 = stablehlo.reduce(%v5703 init: %v5697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5705 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5706 = stablehlo.divide %v5704, %v5705 : tensor<960xf32>
    %v5707 = stablehlo.reshape %v1371 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5709 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5710 = stablehlo.reduce(%v5707 init: %v5708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5711 = stablehlo.divide %v5710, %v5709 : tensor<960xf32>
    %v5712 = stablehlo.reshape %v1371 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5714 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5715 = stablehlo.reduce(%v5712 init: %v5713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5716 = stablehlo.broadcast_in_dim %v5715, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5717 = stablehlo.divide %v5716, %v5714 : tensor<32x960x7x7xf32>
    %v5718 = stablehlo.subtract %v5712, %v5717 : tensor<32x960x7x7xf32>
    %v5719 = stablehlo.multiply %v5718, %v5718 : tensor<32x960x7x7xf32>
    %v5720 = stablehlo.reduce(%v5719 init: %v5713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5721 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5722 = stablehlo.divide %v5720, %v5721 : tensor<960xf32>
    %v5723 = stablehlo.reshape %v1400 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v5724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5725 = stablehlo.constant dense<1568.0> : tensor<320xf32>
    %v5726 = stablehlo.reduce(%v5723 init: %v5724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v5727 = stablehlo.divide %v5726, %v5725 : tensor<320xf32>
    %v5728 = stablehlo.reshape %v1400 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v5729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5730 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v5731 = stablehlo.reduce(%v5728 init: %v5729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v5732 = stablehlo.broadcast_in_dim %v5731, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v5733 = stablehlo.divide %v5732, %v5730 : tensor<32x320x7x7xf32>
    %v5734 = stablehlo.subtract %v5728, %v5733 : tensor<32x320x7x7xf32>
    %v5735 = stablehlo.multiply %v5734, %v5734 : tensor<32x320x7x7xf32>
    %v5736 = stablehlo.reduce(%v5735 init: %v5729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v5737 = stablehlo.constant dense<1568.0> : tensor<320xf32>
    %v5738 = stablehlo.divide %v5736, %v5737 : tensor<320xf32>
    %v5739 = stablehlo.reshape %v1425 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v5740 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5741 = stablehlo.constant dense<1568.0> : tensor<1280xf32>
    %v5742 = stablehlo.reduce(%v5739 init: %v5740) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v5743 = stablehlo.divide %v5742, %v5741 : tensor<1280xf32>
    %v5744 = stablehlo.reshape %v1425 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v5745 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5746 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v5747 = stablehlo.reduce(%v5744 init: %v5745) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v5748 = stablehlo.broadcast_in_dim %v5747, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v5749 = stablehlo.divide %v5748, %v5746 : tensor<32x1280x7x7xf32>
    %v5750 = stablehlo.subtract %v5744, %v5749 : tensor<32x1280x7x7xf32>
    %v5751 = stablehlo.multiply %v5750, %v5750 : tensor<32x1280x7x7xf32>
    %v5752 = stablehlo.reduce(%v5751 init: %v5745) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v5753 = stablehlo.constant dense<1568.0> : tensor<1280xf32>
    %v5754 = stablehlo.divide %v5752, %v5753 : tensor<1280xf32>
    %rho = stablehlo.constant dense<0.900000> : tensor<f32>
    %orho = stablehlo.constant dense<0.100000> : tensor<f32>
    %mu = stablehlo.constant dense<0.900000> : tensor<f32>
    %eps = stablehlo.constant dense<1.000000> : tensor<f32>
    %wd = stablehlo.constant dense<0.000040> : tensor<f32>
    %v5755 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5756 = stablehlo.multiply %v5755, %sW : tensor<32x3x3x3xf32>
    %v5757 = stablehlo.add %v5756, %v4901 : tensor<32x3x3x3xf32>
    %v5758 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5759 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5760 = stablehlo.multiply %v5758, %sWv : tensor<32x3x3x3xf32>
    %v5761 = stablehlo.multiply %v5757, %v5757 : tensor<32x3x3x3xf32>
    %v5762 = stablehlo.multiply %v5759, %v5761 : tensor<32x3x3x3xf32>
    %v5763 = stablehlo.add %v5760, %v5762 : tensor<32x3x3x3xf32>
    %v5764 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5765 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5766 = stablehlo.multiply %v5764, %sWv : tensor<32x3x3x3xf32>
    %v5767 = stablehlo.multiply %v5757, %v5757 : tensor<32x3x3x3xf32>
    %v5768 = stablehlo.multiply %v5765, %v5767 : tensor<32x3x3x3xf32>
    %v5769 = stablehlo.add %v5766, %v5768 : tensor<32x3x3x3xf32>
    %v5770 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5771 = stablehlo.add %v5769, %v5770 : tensor<32x3x3x3xf32>
    %v5772 = stablehlo.sqrt %v5771 : tensor<32x3x3x3xf32>
    %v5773 = stablehlo.divide %v5757, %v5772 : tensor<32x3x3x3xf32>
    %v5774 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5775 = stablehlo.multiply %v5774, %sWm : tensor<32x3x3x3xf32>
    %v5776 = stablehlo.add %v5775, %v5773 : tensor<32x3x3x3xf32>
    %v5777 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5778 = stablehlo.multiply %v5777, %v5776 : tensor<32x3x3x3xf32>
    %v5779 = stablehlo.subtract %sW, %v5778 : tensor<32x3x3x3xf32>
    %v5780 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5781 = stablehlo.multiply %v5780, %sg : tensor<32xf32>
    %v5782 = stablehlo.add %v5781, %v4919 : tensor<32xf32>
    %v5783 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5784 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5785 = stablehlo.multiply %v5783, %sgv : tensor<32xf32>
    %v5786 = stablehlo.multiply %v5782, %v5782 : tensor<32xf32>
    %v5787 = stablehlo.multiply %v5784, %v5786 : tensor<32xf32>
    %v5788 = stablehlo.add %v5785, %v5787 : tensor<32xf32>
    %v5789 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5790 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5791 = stablehlo.multiply %v5789, %sgv : tensor<32xf32>
    %v5792 = stablehlo.multiply %v5782, %v5782 : tensor<32xf32>
    %v5793 = stablehlo.multiply %v5790, %v5792 : tensor<32xf32>
    %v5794 = stablehlo.add %v5791, %v5793 : tensor<32xf32>
    %v5795 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5796 = stablehlo.add %v5794, %v5795 : tensor<32xf32>
    %v5797 = stablehlo.sqrt %v5796 : tensor<32xf32>
    %v5798 = stablehlo.divide %v5782, %v5797 : tensor<32xf32>
    %v5799 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5800 = stablehlo.multiply %v5799, %sgm : tensor<32xf32>
    %v5801 = stablehlo.add %v5800, %v5798 : tensor<32xf32>
    %v5802 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5803 = stablehlo.multiply %v5802, %v5801 : tensor<32xf32>
    %v5804 = stablehlo.subtract %sg, %v5803 : tensor<32xf32>
    %v5805 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5806 = stablehlo.multiply %v5805, %sbt : tensor<32xf32>
    %v5807 = stablehlo.add %v5806, %v4922 : tensor<32xf32>
    %v5808 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5809 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5810 = stablehlo.multiply %v5808, %sbtv : tensor<32xf32>
    %v5811 = stablehlo.multiply %v5807, %v5807 : tensor<32xf32>
    %v5812 = stablehlo.multiply %v5809, %v5811 : tensor<32xf32>
    %v5813 = stablehlo.add %v5810, %v5812 : tensor<32xf32>
    %v5814 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5815 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5816 = stablehlo.multiply %v5814, %sbtv : tensor<32xf32>
    %v5817 = stablehlo.multiply %v5807, %v5807 : tensor<32xf32>
    %v5818 = stablehlo.multiply %v5815, %v5817 : tensor<32xf32>
    %v5819 = stablehlo.add %v5816, %v5818 : tensor<32xf32>
    %v5820 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5821 = stablehlo.add %v5819, %v5820 : tensor<32xf32>
    %v5822 = stablehlo.sqrt %v5821 : tensor<32xf32>
    %v5823 = stablehlo.divide %v5807, %v5822 : tensor<32xf32>
    %v5824 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5825 = stablehlo.multiply %v5824, %sbtm : tensor<32xf32>
    %v5826 = stablehlo.add %v5825, %v5823 : tensor<32xf32>
    %v5827 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5828 = stablehlo.multiply %v5827, %v5826 : tensor<32xf32>
    %v5829 = stablehlo.subtract %sbt, %v5828 : tensor<32xf32>
    %v5830 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v5831 = stablehlo.multiply %v5830, %b1dW : tensor<32x1x3x3xf32>
    %v5832 = stablehlo.add %v5831, %v4809 : tensor<32x1x3x3xf32>
    %v5833 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v5834 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v5835 = stablehlo.multiply %v5833, %b1dWv : tensor<32x1x3x3xf32>
    %v5836 = stablehlo.multiply %v5832, %v5832 : tensor<32x1x3x3xf32>
    %v5837 = stablehlo.multiply %v5834, %v5836 : tensor<32x1x3x3xf32>
    %v5838 = stablehlo.add %v5835, %v5837 : tensor<32x1x3x3xf32>
    %v5839 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v5840 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v5841 = stablehlo.multiply %v5839, %b1dWv : tensor<32x1x3x3xf32>
    %v5842 = stablehlo.multiply %v5832, %v5832 : tensor<32x1x3x3xf32>
    %v5843 = stablehlo.multiply %v5840, %v5842 : tensor<32x1x3x3xf32>
    %v5844 = stablehlo.add %v5841, %v5843 : tensor<32x1x3x3xf32>
    %v5845 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v5846 = stablehlo.add %v5844, %v5845 : tensor<32x1x3x3xf32>
    %v5847 = stablehlo.sqrt %v5846 : tensor<32x1x3x3xf32>
    %v5848 = stablehlo.divide %v5832, %v5847 : tensor<32x1x3x3xf32>
    %v5849 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v5850 = stablehlo.multiply %v5849, %b1dWm : tensor<32x1x3x3xf32>
    %v5851 = stablehlo.add %v5850, %v5848 : tensor<32x1x3x3xf32>
    %v5852 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v5853 = stablehlo.multiply %v5852, %v5851 : tensor<32x1x3x3xf32>
    %v5854 = stablehlo.subtract %b1dW, %v5853 : tensor<32x1x3x3xf32>
    %v5855 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5856 = stablehlo.multiply %v5855, %b1dg : tensor<32xf32>
    %v5857 = stablehlo.add %v5856, %v4827 : tensor<32xf32>
    %v5858 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5859 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5860 = stablehlo.multiply %v5858, %b1dgv : tensor<32xf32>
    %v5861 = stablehlo.multiply %v5857, %v5857 : tensor<32xf32>
    %v5862 = stablehlo.multiply %v5859, %v5861 : tensor<32xf32>
    %v5863 = stablehlo.add %v5860, %v5862 : tensor<32xf32>
    %v5864 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5865 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5866 = stablehlo.multiply %v5864, %b1dgv : tensor<32xf32>
    %v5867 = stablehlo.multiply %v5857, %v5857 : tensor<32xf32>
    %v5868 = stablehlo.multiply %v5865, %v5867 : tensor<32xf32>
    %v5869 = stablehlo.add %v5866, %v5868 : tensor<32xf32>
    %v5870 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5871 = stablehlo.add %v5869, %v5870 : tensor<32xf32>
    %v5872 = stablehlo.sqrt %v5871 : tensor<32xf32>
    %v5873 = stablehlo.divide %v5857, %v5872 : tensor<32xf32>
    %v5874 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5875 = stablehlo.multiply %v5874, %b1dgm : tensor<32xf32>
    %v5876 = stablehlo.add %v5875, %v5873 : tensor<32xf32>
    %v5877 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5878 = stablehlo.multiply %v5877, %v5876 : tensor<32xf32>
    %v5879 = stablehlo.subtract %b1dg, %v5878 : tensor<32xf32>
    %v5880 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5881 = stablehlo.multiply %v5880, %b1dbt : tensor<32xf32>
    %v5882 = stablehlo.add %v5881, %v4830 : tensor<32xf32>
    %v5883 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5884 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5885 = stablehlo.multiply %v5883, %b1dbtv : tensor<32xf32>
    %v5886 = stablehlo.multiply %v5882, %v5882 : tensor<32xf32>
    %v5887 = stablehlo.multiply %v5884, %v5886 : tensor<32xf32>
    %v5888 = stablehlo.add %v5885, %v5887 : tensor<32xf32>
    %v5889 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5890 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5891 = stablehlo.multiply %v5889, %b1dbtv : tensor<32xf32>
    %v5892 = stablehlo.multiply %v5882, %v5882 : tensor<32xf32>
    %v5893 = stablehlo.multiply %v5890, %v5892 : tensor<32xf32>
    %v5894 = stablehlo.add %v5891, %v5893 : tensor<32xf32>
    %v5895 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5896 = stablehlo.add %v5894, %v5895 : tensor<32xf32>
    %v5897 = stablehlo.sqrt %v5896 : tensor<32xf32>
    %v5898 = stablehlo.divide %v5882, %v5897 : tensor<32xf32>
    %v5899 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5900 = stablehlo.multiply %v5899, %b1dbtm : tensor<32xf32>
    %v5901 = stablehlo.add %v5900, %v5898 : tensor<32xf32>
    %v5902 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v5903 = stablehlo.multiply %v5902, %v5901 : tensor<32xf32>
    %v5904 = stablehlo.subtract %b1dbt, %v5903 : tensor<32xf32>
    %v5905 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v5906 = stablehlo.multiply %v5905, %b1pW : tensor<16x32x1x1xf32>
    %v5907 = stablehlo.add %v5906, %v4836 : tensor<16x32x1x1xf32>
    %v5908 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v5909 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v5910 = stablehlo.multiply %v5908, %b1pWv : tensor<16x32x1x1xf32>
    %v5911 = stablehlo.multiply %v5907, %v5907 : tensor<16x32x1x1xf32>
    %v5912 = stablehlo.multiply %v5909, %v5911 : tensor<16x32x1x1xf32>
    %v5913 = stablehlo.add %v5910, %v5912 : tensor<16x32x1x1xf32>
    %v5914 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v5915 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v5916 = stablehlo.multiply %v5914, %b1pWv : tensor<16x32x1x1xf32>
    %v5917 = stablehlo.multiply %v5907, %v5907 : tensor<16x32x1x1xf32>
    %v5918 = stablehlo.multiply %v5915, %v5917 : tensor<16x32x1x1xf32>
    %v5919 = stablehlo.add %v5916, %v5918 : tensor<16x32x1x1xf32>
    %v5920 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v5921 = stablehlo.add %v5919, %v5920 : tensor<16x32x1x1xf32>
    %v5922 = stablehlo.sqrt %v5921 : tensor<16x32x1x1xf32>
    %v5923 = stablehlo.divide %v5907, %v5922 : tensor<16x32x1x1xf32>
    %v5924 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v5925 = stablehlo.multiply %v5924, %b1pWm : tensor<16x32x1x1xf32>
    %v5926 = stablehlo.add %v5925, %v5923 : tensor<16x32x1x1xf32>
    %v5927 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v5928 = stablehlo.multiply %v5927, %v5926 : tensor<16x32x1x1xf32>
    %v5929 = stablehlo.subtract %b1pW, %v5928 : tensor<16x32x1x1xf32>
    %v5930 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5931 = stablehlo.multiply %v5930, %b1pg : tensor<16xf32>
    %v5932 = stablehlo.add %v5931, %v4854 : tensor<16xf32>
    %v5933 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5934 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5935 = stablehlo.multiply %v5933, %b1pgv : tensor<16xf32>
    %v5936 = stablehlo.multiply %v5932, %v5932 : tensor<16xf32>
    %v5937 = stablehlo.multiply %v5934, %v5936 : tensor<16xf32>
    %v5938 = stablehlo.add %v5935, %v5937 : tensor<16xf32>
    %v5939 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5940 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5941 = stablehlo.multiply %v5939, %b1pgv : tensor<16xf32>
    %v5942 = stablehlo.multiply %v5932, %v5932 : tensor<16xf32>
    %v5943 = stablehlo.multiply %v5940, %v5942 : tensor<16xf32>
    %v5944 = stablehlo.add %v5941, %v5943 : tensor<16xf32>
    %v5945 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5946 = stablehlo.add %v5944, %v5945 : tensor<16xf32>
    %v5947 = stablehlo.sqrt %v5946 : tensor<16xf32>
    %v5948 = stablehlo.divide %v5932, %v5947 : tensor<16xf32>
    %v5949 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5950 = stablehlo.multiply %v5949, %b1pgm : tensor<16xf32>
    %v5951 = stablehlo.add %v5950, %v5948 : tensor<16xf32>
    %v5952 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5953 = stablehlo.multiply %v5952, %v5951 : tensor<16xf32>
    %v5954 = stablehlo.subtract %b1pg, %v5953 : tensor<16xf32>
    %v5955 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5956 = stablehlo.multiply %v5955, %b1pbt : tensor<16xf32>
    %v5957 = stablehlo.add %v5956, %v4857 : tensor<16xf32>
    %v5958 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5959 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5960 = stablehlo.multiply %v5958, %b1pbtv : tensor<16xf32>
    %v5961 = stablehlo.multiply %v5957, %v5957 : tensor<16xf32>
    %v5962 = stablehlo.multiply %v5959, %v5961 : tensor<16xf32>
    %v5963 = stablehlo.add %v5960, %v5962 : tensor<16xf32>
    %v5964 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5965 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5966 = stablehlo.multiply %v5964, %b1pbtv : tensor<16xf32>
    %v5967 = stablehlo.multiply %v5957, %v5957 : tensor<16xf32>
    %v5968 = stablehlo.multiply %v5965, %v5967 : tensor<16xf32>
    %v5969 = stablehlo.add %v5966, %v5968 : tensor<16xf32>
    %v5970 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5971 = stablehlo.add %v5969, %v5970 : tensor<16xf32>
    %v5972 = stablehlo.sqrt %v5971 : tensor<16xf32>
    %v5973 = stablehlo.divide %v5957, %v5972 : tensor<16xf32>
    %v5974 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5975 = stablehlo.multiply %v5974, %b1pbtm : tensor<16xf32>
    %v5976 = stablehlo.add %v5975, %v5973 : tensor<16xf32>
    %v5977 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v5978 = stablehlo.multiply %v5977, %v5976 : tensor<16xf32>
    %v5979 = stablehlo.subtract %b1pbt, %v5978 : tensor<16xf32>
    %v5980 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v5981 = stablehlo.multiply %v5980, %b2eW : tensor<96x16x1x1xf32>
    %v5982 = stablehlo.add %v5981, %v4651 : tensor<96x16x1x1xf32>
    %v5983 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v5984 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v5985 = stablehlo.multiply %v5983, %b2eWv : tensor<96x16x1x1xf32>
    %v5986 = stablehlo.multiply %v5982, %v5982 : tensor<96x16x1x1xf32>
    %v5987 = stablehlo.multiply %v5984, %v5986 : tensor<96x16x1x1xf32>
    %v5988 = stablehlo.add %v5985, %v5987 : tensor<96x16x1x1xf32>
    %v5989 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v5990 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v5991 = stablehlo.multiply %v5989, %b2eWv : tensor<96x16x1x1xf32>
    %v5992 = stablehlo.multiply %v5982, %v5982 : tensor<96x16x1x1xf32>
    %v5993 = stablehlo.multiply %v5990, %v5992 : tensor<96x16x1x1xf32>
    %v5994 = stablehlo.add %v5991, %v5993 : tensor<96x16x1x1xf32>
    %v5995 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v5996 = stablehlo.add %v5994, %v5995 : tensor<96x16x1x1xf32>
    %v5997 = stablehlo.sqrt %v5996 : tensor<96x16x1x1xf32>
    %v5998 = stablehlo.divide %v5982, %v5997 : tensor<96x16x1x1xf32>
    %v5999 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6000 = stablehlo.multiply %v5999, %b2eWm : tensor<96x16x1x1xf32>
    %v6001 = stablehlo.add %v6000, %v5998 : tensor<96x16x1x1xf32>
    %v6002 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6003 = stablehlo.multiply %v6002, %v6001 : tensor<96x16x1x1xf32>
    %v6004 = stablehlo.subtract %b2eW, %v6003 : tensor<96x16x1x1xf32>
    %v6005 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6006 = stablehlo.multiply %v6005, %b2eg : tensor<96xf32>
    %v6007 = stablehlo.add %v6006, %v4669 : tensor<96xf32>
    %v6008 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6009 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6010 = stablehlo.multiply %v6008, %b2egv : tensor<96xf32>
    %v6011 = stablehlo.multiply %v6007, %v6007 : tensor<96xf32>
    %v6012 = stablehlo.multiply %v6009, %v6011 : tensor<96xf32>
    %v6013 = stablehlo.add %v6010, %v6012 : tensor<96xf32>
    %v6014 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6015 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6016 = stablehlo.multiply %v6014, %b2egv : tensor<96xf32>
    %v6017 = stablehlo.multiply %v6007, %v6007 : tensor<96xf32>
    %v6018 = stablehlo.multiply %v6015, %v6017 : tensor<96xf32>
    %v6019 = stablehlo.add %v6016, %v6018 : tensor<96xf32>
    %v6020 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6021 = stablehlo.add %v6019, %v6020 : tensor<96xf32>
    %v6022 = stablehlo.sqrt %v6021 : tensor<96xf32>
    %v6023 = stablehlo.divide %v6007, %v6022 : tensor<96xf32>
    %v6024 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6025 = stablehlo.multiply %v6024, %b2egm : tensor<96xf32>
    %v6026 = stablehlo.add %v6025, %v6023 : tensor<96xf32>
    %v6027 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6028 = stablehlo.multiply %v6027, %v6026 : tensor<96xf32>
    %v6029 = stablehlo.subtract %b2eg, %v6028 : tensor<96xf32>
    %v6030 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6031 = stablehlo.multiply %v6030, %b2ebt : tensor<96xf32>
    %v6032 = stablehlo.add %v6031, %v4672 : tensor<96xf32>
    %v6033 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6034 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6035 = stablehlo.multiply %v6033, %b2ebtv : tensor<96xf32>
    %v6036 = stablehlo.multiply %v6032, %v6032 : tensor<96xf32>
    %v6037 = stablehlo.multiply %v6034, %v6036 : tensor<96xf32>
    %v6038 = stablehlo.add %v6035, %v6037 : tensor<96xf32>
    %v6039 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6040 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6041 = stablehlo.multiply %v6039, %b2ebtv : tensor<96xf32>
    %v6042 = stablehlo.multiply %v6032, %v6032 : tensor<96xf32>
    %v6043 = stablehlo.multiply %v6040, %v6042 : tensor<96xf32>
    %v6044 = stablehlo.add %v6041, %v6043 : tensor<96xf32>
    %v6045 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6046 = stablehlo.add %v6044, %v6045 : tensor<96xf32>
    %v6047 = stablehlo.sqrt %v6046 : tensor<96xf32>
    %v6048 = stablehlo.divide %v6032, %v6047 : tensor<96xf32>
    %v6049 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6050 = stablehlo.multiply %v6049, %b2ebtm : tensor<96xf32>
    %v6051 = stablehlo.add %v6050, %v6048 : tensor<96xf32>
    %v6052 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6053 = stablehlo.multiply %v6052, %v6051 : tensor<96xf32>
    %v6054 = stablehlo.subtract %b2ebt, %v6053 : tensor<96xf32>
    %v6055 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6056 = stablehlo.multiply %v6055, %b2dW : tensor<96x1x3x3xf32>
    %v6057 = stablehlo.add %v6056, %v4680 : tensor<96x1x3x3xf32>
    %v6058 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6059 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6060 = stablehlo.multiply %v6058, %b2dWv : tensor<96x1x3x3xf32>
    %v6061 = stablehlo.multiply %v6057, %v6057 : tensor<96x1x3x3xf32>
    %v6062 = stablehlo.multiply %v6059, %v6061 : tensor<96x1x3x3xf32>
    %v6063 = stablehlo.add %v6060, %v6062 : tensor<96x1x3x3xf32>
    %v6064 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6065 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6066 = stablehlo.multiply %v6064, %b2dWv : tensor<96x1x3x3xf32>
    %v6067 = stablehlo.multiply %v6057, %v6057 : tensor<96x1x3x3xf32>
    %v6068 = stablehlo.multiply %v6065, %v6067 : tensor<96x1x3x3xf32>
    %v6069 = stablehlo.add %v6066, %v6068 : tensor<96x1x3x3xf32>
    %v6070 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6071 = stablehlo.add %v6069, %v6070 : tensor<96x1x3x3xf32>
    %v6072 = stablehlo.sqrt %v6071 : tensor<96x1x3x3xf32>
    %v6073 = stablehlo.divide %v6057, %v6072 : tensor<96x1x3x3xf32>
    %v6074 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6075 = stablehlo.multiply %v6074, %b2dWm : tensor<96x1x3x3xf32>
    %v6076 = stablehlo.add %v6075, %v6073 : tensor<96x1x3x3xf32>
    %v6077 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6078 = stablehlo.multiply %v6077, %v6076 : tensor<96x1x3x3xf32>
    %v6079 = stablehlo.subtract %b2dW, %v6078 : tensor<96x1x3x3xf32>
    %v6080 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6081 = stablehlo.multiply %v6080, %b2dg : tensor<96xf32>
    %v6082 = stablehlo.add %v6081, %v4698 : tensor<96xf32>
    %v6083 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6084 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6085 = stablehlo.multiply %v6083, %b2dgv : tensor<96xf32>
    %v6086 = stablehlo.multiply %v6082, %v6082 : tensor<96xf32>
    %v6087 = stablehlo.multiply %v6084, %v6086 : tensor<96xf32>
    %v6088 = stablehlo.add %v6085, %v6087 : tensor<96xf32>
    %v6089 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6090 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6091 = stablehlo.multiply %v6089, %b2dgv : tensor<96xf32>
    %v6092 = stablehlo.multiply %v6082, %v6082 : tensor<96xf32>
    %v6093 = stablehlo.multiply %v6090, %v6092 : tensor<96xf32>
    %v6094 = stablehlo.add %v6091, %v6093 : tensor<96xf32>
    %v6095 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6096 = stablehlo.add %v6094, %v6095 : tensor<96xf32>
    %v6097 = stablehlo.sqrt %v6096 : tensor<96xf32>
    %v6098 = stablehlo.divide %v6082, %v6097 : tensor<96xf32>
    %v6099 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6100 = stablehlo.multiply %v6099, %b2dgm : tensor<96xf32>
    %v6101 = stablehlo.add %v6100, %v6098 : tensor<96xf32>
    %v6102 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6103 = stablehlo.multiply %v6102, %v6101 : tensor<96xf32>
    %v6104 = stablehlo.subtract %b2dg, %v6103 : tensor<96xf32>
    %v6105 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6106 = stablehlo.multiply %v6105, %b2dbt : tensor<96xf32>
    %v6107 = stablehlo.add %v6106, %v4701 : tensor<96xf32>
    %v6108 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6109 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6110 = stablehlo.multiply %v6108, %b2dbtv : tensor<96xf32>
    %v6111 = stablehlo.multiply %v6107, %v6107 : tensor<96xf32>
    %v6112 = stablehlo.multiply %v6109, %v6111 : tensor<96xf32>
    %v6113 = stablehlo.add %v6110, %v6112 : tensor<96xf32>
    %v6114 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6115 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6116 = stablehlo.multiply %v6114, %b2dbtv : tensor<96xf32>
    %v6117 = stablehlo.multiply %v6107, %v6107 : tensor<96xf32>
    %v6118 = stablehlo.multiply %v6115, %v6117 : tensor<96xf32>
    %v6119 = stablehlo.add %v6116, %v6118 : tensor<96xf32>
    %v6120 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6121 = stablehlo.add %v6119, %v6120 : tensor<96xf32>
    %v6122 = stablehlo.sqrt %v6121 : tensor<96xf32>
    %v6123 = stablehlo.divide %v6107, %v6122 : tensor<96xf32>
    %v6124 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6125 = stablehlo.multiply %v6124, %b2dbtm : tensor<96xf32>
    %v6126 = stablehlo.add %v6125, %v6123 : tensor<96xf32>
    %v6127 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6128 = stablehlo.multiply %v6127, %v6126 : tensor<96xf32>
    %v6129 = stablehlo.subtract %b2dbt, %v6128 : tensor<96xf32>
    %v6130 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6131 = stablehlo.multiply %v6130, %b2pW : tensor<24x96x1x1xf32>
    %v6132 = stablehlo.add %v6131, %v4707 : tensor<24x96x1x1xf32>
    %v6133 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6134 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6135 = stablehlo.multiply %v6133, %b2pWv : tensor<24x96x1x1xf32>
    %v6136 = stablehlo.multiply %v6132, %v6132 : tensor<24x96x1x1xf32>
    %v6137 = stablehlo.multiply %v6134, %v6136 : tensor<24x96x1x1xf32>
    %v6138 = stablehlo.add %v6135, %v6137 : tensor<24x96x1x1xf32>
    %v6139 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6140 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6141 = stablehlo.multiply %v6139, %b2pWv : tensor<24x96x1x1xf32>
    %v6142 = stablehlo.multiply %v6132, %v6132 : tensor<24x96x1x1xf32>
    %v6143 = stablehlo.multiply %v6140, %v6142 : tensor<24x96x1x1xf32>
    %v6144 = stablehlo.add %v6141, %v6143 : tensor<24x96x1x1xf32>
    %v6145 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6146 = stablehlo.add %v6144, %v6145 : tensor<24x96x1x1xf32>
    %v6147 = stablehlo.sqrt %v6146 : tensor<24x96x1x1xf32>
    %v6148 = stablehlo.divide %v6132, %v6147 : tensor<24x96x1x1xf32>
    %v6149 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6150 = stablehlo.multiply %v6149, %b2pWm : tensor<24x96x1x1xf32>
    %v6151 = stablehlo.add %v6150, %v6148 : tensor<24x96x1x1xf32>
    %v6152 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6153 = stablehlo.multiply %v6152, %v6151 : tensor<24x96x1x1xf32>
    %v6154 = stablehlo.subtract %b2pW, %v6153 : tensor<24x96x1x1xf32>
    %v6155 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6156 = stablehlo.multiply %v6155, %b2pg : tensor<24xf32>
    %v6157 = stablehlo.add %v6156, %v4725 : tensor<24xf32>
    %v6158 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6159 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6160 = stablehlo.multiply %v6158, %b2pgv : tensor<24xf32>
    %v6161 = stablehlo.multiply %v6157, %v6157 : tensor<24xf32>
    %v6162 = stablehlo.multiply %v6159, %v6161 : tensor<24xf32>
    %v6163 = stablehlo.add %v6160, %v6162 : tensor<24xf32>
    %v6164 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6165 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6166 = stablehlo.multiply %v6164, %b2pgv : tensor<24xf32>
    %v6167 = stablehlo.multiply %v6157, %v6157 : tensor<24xf32>
    %v6168 = stablehlo.multiply %v6165, %v6167 : tensor<24xf32>
    %v6169 = stablehlo.add %v6166, %v6168 : tensor<24xf32>
    %v6170 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6171 = stablehlo.add %v6169, %v6170 : tensor<24xf32>
    %v6172 = stablehlo.sqrt %v6171 : tensor<24xf32>
    %v6173 = stablehlo.divide %v6157, %v6172 : tensor<24xf32>
    %v6174 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6175 = stablehlo.multiply %v6174, %b2pgm : tensor<24xf32>
    %v6176 = stablehlo.add %v6175, %v6173 : tensor<24xf32>
    %v6177 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6178 = stablehlo.multiply %v6177, %v6176 : tensor<24xf32>
    %v6179 = stablehlo.subtract %b2pg, %v6178 : tensor<24xf32>
    %v6180 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6181 = stablehlo.multiply %v6180, %b2pbt : tensor<24xf32>
    %v6182 = stablehlo.add %v6181, %v4728 : tensor<24xf32>
    %v6183 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6184 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6185 = stablehlo.multiply %v6183, %b2pbtv : tensor<24xf32>
    %v6186 = stablehlo.multiply %v6182, %v6182 : tensor<24xf32>
    %v6187 = stablehlo.multiply %v6184, %v6186 : tensor<24xf32>
    %v6188 = stablehlo.add %v6185, %v6187 : tensor<24xf32>
    %v6189 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6190 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6191 = stablehlo.multiply %v6189, %b2pbtv : tensor<24xf32>
    %v6192 = stablehlo.multiply %v6182, %v6182 : tensor<24xf32>
    %v6193 = stablehlo.multiply %v6190, %v6192 : tensor<24xf32>
    %v6194 = stablehlo.add %v6191, %v6193 : tensor<24xf32>
    %v6195 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6196 = stablehlo.add %v6194, %v6195 : tensor<24xf32>
    %v6197 = stablehlo.sqrt %v6196 : tensor<24xf32>
    %v6198 = stablehlo.divide %v6182, %v6197 : tensor<24xf32>
    %v6199 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6200 = stablehlo.multiply %v6199, %b2pbtm : tensor<24xf32>
    %v6201 = stablehlo.add %v6200, %v6198 : tensor<24xf32>
    %v6202 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6203 = stablehlo.multiply %v6202, %v6201 : tensor<24xf32>
    %v6204 = stablehlo.subtract %b2pbt, %v6203 : tensor<24xf32>
    %v6205 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6206 = stablehlo.multiply %v6205, %b3eW : tensor<144x24x1x1xf32>
    %v6207 = stablehlo.add %v6206, %v4452 : tensor<144x24x1x1xf32>
    %v6208 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6209 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6210 = stablehlo.multiply %v6208, %b3eWv : tensor<144x24x1x1xf32>
    %v6211 = stablehlo.multiply %v6207, %v6207 : tensor<144x24x1x1xf32>
    %v6212 = stablehlo.multiply %v6209, %v6211 : tensor<144x24x1x1xf32>
    %v6213 = stablehlo.add %v6210, %v6212 : tensor<144x24x1x1xf32>
    %v6214 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6215 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6216 = stablehlo.multiply %v6214, %b3eWv : tensor<144x24x1x1xf32>
    %v6217 = stablehlo.multiply %v6207, %v6207 : tensor<144x24x1x1xf32>
    %v6218 = stablehlo.multiply %v6215, %v6217 : tensor<144x24x1x1xf32>
    %v6219 = stablehlo.add %v6216, %v6218 : tensor<144x24x1x1xf32>
    %v6220 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6221 = stablehlo.add %v6219, %v6220 : tensor<144x24x1x1xf32>
    %v6222 = stablehlo.sqrt %v6221 : tensor<144x24x1x1xf32>
    %v6223 = stablehlo.divide %v6207, %v6222 : tensor<144x24x1x1xf32>
    %v6224 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6225 = stablehlo.multiply %v6224, %b3eWm : tensor<144x24x1x1xf32>
    %v6226 = stablehlo.add %v6225, %v6223 : tensor<144x24x1x1xf32>
    %v6227 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6228 = stablehlo.multiply %v6227, %v6226 : tensor<144x24x1x1xf32>
    %v6229 = stablehlo.subtract %b3eW, %v6228 : tensor<144x24x1x1xf32>
    %v6230 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6231 = stablehlo.multiply %v6230, %b3eg : tensor<144xf32>
    %v6232 = stablehlo.add %v6231, %v4470 : tensor<144xf32>
    %v6233 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6234 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6235 = stablehlo.multiply %v6233, %b3egv : tensor<144xf32>
    %v6236 = stablehlo.multiply %v6232, %v6232 : tensor<144xf32>
    %v6237 = stablehlo.multiply %v6234, %v6236 : tensor<144xf32>
    %v6238 = stablehlo.add %v6235, %v6237 : tensor<144xf32>
    %v6239 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6240 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6241 = stablehlo.multiply %v6239, %b3egv : tensor<144xf32>
    %v6242 = stablehlo.multiply %v6232, %v6232 : tensor<144xf32>
    %v6243 = stablehlo.multiply %v6240, %v6242 : tensor<144xf32>
    %v6244 = stablehlo.add %v6241, %v6243 : tensor<144xf32>
    %v6245 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6246 = stablehlo.add %v6244, %v6245 : tensor<144xf32>
    %v6247 = stablehlo.sqrt %v6246 : tensor<144xf32>
    %v6248 = stablehlo.divide %v6232, %v6247 : tensor<144xf32>
    %v6249 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6250 = stablehlo.multiply %v6249, %b3egm : tensor<144xf32>
    %v6251 = stablehlo.add %v6250, %v6248 : tensor<144xf32>
    %v6252 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6253 = stablehlo.multiply %v6252, %v6251 : tensor<144xf32>
    %v6254 = stablehlo.subtract %b3eg, %v6253 : tensor<144xf32>
    %v6255 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6256 = stablehlo.multiply %v6255, %b3ebt : tensor<144xf32>
    %v6257 = stablehlo.add %v6256, %v4473 : tensor<144xf32>
    %v6258 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6259 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6260 = stablehlo.multiply %v6258, %b3ebtv : tensor<144xf32>
    %v6261 = stablehlo.multiply %v6257, %v6257 : tensor<144xf32>
    %v6262 = stablehlo.multiply %v6259, %v6261 : tensor<144xf32>
    %v6263 = stablehlo.add %v6260, %v6262 : tensor<144xf32>
    %v6264 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6265 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6266 = stablehlo.multiply %v6264, %b3ebtv : tensor<144xf32>
    %v6267 = stablehlo.multiply %v6257, %v6257 : tensor<144xf32>
    %v6268 = stablehlo.multiply %v6265, %v6267 : tensor<144xf32>
    %v6269 = stablehlo.add %v6266, %v6268 : tensor<144xf32>
    %v6270 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6271 = stablehlo.add %v6269, %v6270 : tensor<144xf32>
    %v6272 = stablehlo.sqrt %v6271 : tensor<144xf32>
    %v6273 = stablehlo.divide %v6257, %v6272 : tensor<144xf32>
    %v6274 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6275 = stablehlo.multiply %v6274, %b3ebtm : tensor<144xf32>
    %v6276 = stablehlo.add %v6275, %v6273 : tensor<144xf32>
    %v6277 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6278 = stablehlo.multiply %v6277, %v6276 : tensor<144xf32>
    %v6279 = stablehlo.subtract %b3ebt, %v6278 : tensor<144xf32>
    %v6280 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6281 = stablehlo.multiply %v6280, %b3dW : tensor<144x1x3x3xf32>
    %v6282 = stablehlo.add %v6281, %v4479 : tensor<144x1x3x3xf32>
    %v6283 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6284 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6285 = stablehlo.multiply %v6283, %b3dWv : tensor<144x1x3x3xf32>
    %v6286 = stablehlo.multiply %v6282, %v6282 : tensor<144x1x3x3xf32>
    %v6287 = stablehlo.multiply %v6284, %v6286 : tensor<144x1x3x3xf32>
    %v6288 = stablehlo.add %v6285, %v6287 : tensor<144x1x3x3xf32>
    %v6289 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6290 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6291 = stablehlo.multiply %v6289, %b3dWv : tensor<144x1x3x3xf32>
    %v6292 = stablehlo.multiply %v6282, %v6282 : tensor<144x1x3x3xf32>
    %v6293 = stablehlo.multiply %v6290, %v6292 : tensor<144x1x3x3xf32>
    %v6294 = stablehlo.add %v6291, %v6293 : tensor<144x1x3x3xf32>
    %v6295 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6296 = stablehlo.add %v6294, %v6295 : tensor<144x1x3x3xf32>
    %v6297 = stablehlo.sqrt %v6296 : tensor<144x1x3x3xf32>
    %v6298 = stablehlo.divide %v6282, %v6297 : tensor<144x1x3x3xf32>
    %v6299 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6300 = stablehlo.multiply %v6299, %b3dWm : tensor<144x1x3x3xf32>
    %v6301 = stablehlo.add %v6300, %v6298 : tensor<144x1x3x3xf32>
    %v6302 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6303 = stablehlo.multiply %v6302, %v6301 : tensor<144x1x3x3xf32>
    %v6304 = stablehlo.subtract %b3dW, %v6303 : tensor<144x1x3x3xf32>
    %v6305 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6306 = stablehlo.multiply %v6305, %b3dg : tensor<144xf32>
    %v6307 = stablehlo.add %v6306, %v4497 : tensor<144xf32>
    %v6308 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6309 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6310 = stablehlo.multiply %v6308, %b3dgv : tensor<144xf32>
    %v6311 = stablehlo.multiply %v6307, %v6307 : tensor<144xf32>
    %v6312 = stablehlo.multiply %v6309, %v6311 : tensor<144xf32>
    %v6313 = stablehlo.add %v6310, %v6312 : tensor<144xf32>
    %v6314 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6315 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6316 = stablehlo.multiply %v6314, %b3dgv : tensor<144xf32>
    %v6317 = stablehlo.multiply %v6307, %v6307 : tensor<144xf32>
    %v6318 = stablehlo.multiply %v6315, %v6317 : tensor<144xf32>
    %v6319 = stablehlo.add %v6316, %v6318 : tensor<144xf32>
    %v6320 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6321 = stablehlo.add %v6319, %v6320 : tensor<144xf32>
    %v6322 = stablehlo.sqrt %v6321 : tensor<144xf32>
    %v6323 = stablehlo.divide %v6307, %v6322 : tensor<144xf32>
    %v6324 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6325 = stablehlo.multiply %v6324, %b3dgm : tensor<144xf32>
    %v6326 = stablehlo.add %v6325, %v6323 : tensor<144xf32>
    %v6327 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6328 = stablehlo.multiply %v6327, %v6326 : tensor<144xf32>
    %v6329 = stablehlo.subtract %b3dg, %v6328 : tensor<144xf32>
    %v6330 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6331 = stablehlo.multiply %v6330, %b3dbt : tensor<144xf32>
    %v6332 = stablehlo.add %v6331, %v4500 : tensor<144xf32>
    %v6333 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6334 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6335 = stablehlo.multiply %v6333, %b3dbtv : tensor<144xf32>
    %v6336 = stablehlo.multiply %v6332, %v6332 : tensor<144xf32>
    %v6337 = stablehlo.multiply %v6334, %v6336 : tensor<144xf32>
    %v6338 = stablehlo.add %v6335, %v6337 : tensor<144xf32>
    %v6339 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6340 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6341 = stablehlo.multiply %v6339, %b3dbtv : tensor<144xf32>
    %v6342 = stablehlo.multiply %v6332, %v6332 : tensor<144xf32>
    %v6343 = stablehlo.multiply %v6340, %v6342 : tensor<144xf32>
    %v6344 = stablehlo.add %v6341, %v6343 : tensor<144xf32>
    %v6345 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6346 = stablehlo.add %v6344, %v6345 : tensor<144xf32>
    %v6347 = stablehlo.sqrt %v6346 : tensor<144xf32>
    %v6348 = stablehlo.divide %v6332, %v6347 : tensor<144xf32>
    %v6349 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6350 = stablehlo.multiply %v6349, %b3dbtm : tensor<144xf32>
    %v6351 = stablehlo.add %v6350, %v6348 : tensor<144xf32>
    %v6352 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6353 = stablehlo.multiply %v6352, %v6351 : tensor<144xf32>
    %v6354 = stablehlo.subtract %b3dbt, %v6353 : tensor<144xf32>
    %v6355 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6356 = stablehlo.multiply %v6355, %b3pW : tensor<24x144x1x1xf32>
    %v6357 = stablehlo.add %v6356, %v4506 : tensor<24x144x1x1xf32>
    %v6358 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6359 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6360 = stablehlo.multiply %v6358, %b3pWv : tensor<24x144x1x1xf32>
    %v6361 = stablehlo.multiply %v6357, %v6357 : tensor<24x144x1x1xf32>
    %v6362 = stablehlo.multiply %v6359, %v6361 : tensor<24x144x1x1xf32>
    %v6363 = stablehlo.add %v6360, %v6362 : tensor<24x144x1x1xf32>
    %v6364 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6365 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6366 = stablehlo.multiply %v6364, %b3pWv : tensor<24x144x1x1xf32>
    %v6367 = stablehlo.multiply %v6357, %v6357 : tensor<24x144x1x1xf32>
    %v6368 = stablehlo.multiply %v6365, %v6367 : tensor<24x144x1x1xf32>
    %v6369 = stablehlo.add %v6366, %v6368 : tensor<24x144x1x1xf32>
    %v6370 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6371 = stablehlo.add %v6369, %v6370 : tensor<24x144x1x1xf32>
    %v6372 = stablehlo.sqrt %v6371 : tensor<24x144x1x1xf32>
    %v6373 = stablehlo.divide %v6357, %v6372 : tensor<24x144x1x1xf32>
    %v6374 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6375 = stablehlo.multiply %v6374, %b3pWm : tensor<24x144x1x1xf32>
    %v6376 = stablehlo.add %v6375, %v6373 : tensor<24x144x1x1xf32>
    %v6377 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6378 = stablehlo.multiply %v6377, %v6376 : tensor<24x144x1x1xf32>
    %v6379 = stablehlo.subtract %b3pW, %v6378 : tensor<24x144x1x1xf32>
    %v6380 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6381 = stablehlo.multiply %v6380, %b3pg : tensor<24xf32>
    %v6382 = stablehlo.add %v6381, %v4524 : tensor<24xf32>
    %v6383 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6384 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6385 = stablehlo.multiply %v6383, %b3pgv : tensor<24xf32>
    %v6386 = stablehlo.multiply %v6382, %v6382 : tensor<24xf32>
    %v6387 = stablehlo.multiply %v6384, %v6386 : tensor<24xf32>
    %v6388 = stablehlo.add %v6385, %v6387 : tensor<24xf32>
    %v6389 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6390 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6391 = stablehlo.multiply %v6389, %b3pgv : tensor<24xf32>
    %v6392 = stablehlo.multiply %v6382, %v6382 : tensor<24xf32>
    %v6393 = stablehlo.multiply %v6390, %v6392 : tensor<24xf32>
    %v6394 = stablehlo.add %v6391, %v6393 : tensor<24xf32>
    %v6395 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6396 = stablehlo.add %v6394, %v6395 : tensor<24xf32>
    %v6397 = stablehlo.sqrt %v6396 : tensor<24xf32>
    %v6398 = stablehlo.divide %v6382, %v6397 : tensor<24xf32>
    %v6399 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6400 = stablehlo.multiply %v6399, %b3pgm : tensor<24xf32>
    %v6401 = stablehlo.add %v6400, %v6398 : tensor<24xf32>
    %v6402 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6403 = stablehlo.multiply %v6402, %v6401 : tensor<24xf32>
    %v6404 = stablehlo.subtract %b3pg, %v6403 : tensor<24xf32>
    %v6405 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6406 = stablehlo.multiply %v6405, %b3pbt : tensor<24xf32>
    %v6407 = stablehlo.add %v6406, %v4527 : tensor<24xf32>
    %v6408 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6409 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6410 = stablehlo.multiply %v6408, %b3pbtv : tensor<24xf32>
    %v6411 = stablehlo.multiply %v6407, %v6407 : tensor<24xf32>
    %v6412 = stablehlo.multiply %v6409, %v6411 : tensor<24xf32>
    %v6413 = stablehlo.add %v6410, %v6412 : tensor<24xf32>
    %v6414 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6415 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6416 = stablehlo.multiply %v6414, %b3pbtv : tensor<24xf32>
    %v6417 = stablehlo.multiply %v6407, %v6407 : tensor<24xf32>
    %v6418 = stablehlo.multiply %v6415, %v6417 : tensor<24xf32>
    %v6419 = stablehlo.add %v6416, %v6418 : tensor<24xf32>
    %v6420 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6421 = stablehlo.add %v6419, %v6420 : tensor<24xf32>
    %v6422 = stablehlo.sqrt %v6421 : tensor<24xf32>
    %v6423 = stablehlo.divide %v6407, %v6422 : tensor<24xf32>
    %v6424 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6425 = stablehlo.multiply %v6424, %b3pbtm : tensor<24xf32>
    %v6426 = stablehlo.add %v6425, %v6423 : tensor<24xf32>
    %v6427 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6428 = stablehlo.multiply %v6427, %v6426 : tensor<24xf32>
    %v6429 = stablehlo.subtract %b3pbt, %v6428 : tensor<24xf32>
    %v6430 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6431 = stablehlo.multiply %v6430, %b4eW : tensor<144x24x1x1xf32>
    %v6432 = stablehlo.add %v6431, %v4252 : tensor<144x24x1x1xf32>
    %v6433 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6434 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6435 = stablehlo.multiply %v6433, %b4eWv : tensor<144x24x1x1xf32>
    %v6436 = stablehlo.multiply %v6432, %v6432 : tensor<144x24x1x1xf32>
    %v6437 = stablehlo.multiply %v6434, %v6436 : tensor<144x24x1x1xf32>
    %v6438 = stablehlo.add %v6435, %v6437 : tensor<144x24x1x1xf32>
    %v6439 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6440 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6441 = stablehlo.multiply %v6439, %b4eWv : tensor<144x24x1x1xf32>
    %v6442 = stablehlo.multiply %v6432, %v6432 : tensor<144x24x1x1xf32>
    %v6443 = stablehlo.multiply %v6440, %v6442 : tensor<144x24x1x1xf32>
    %v6444 = stablehlo.add %v6441, %v6443 : tensor<144x24x1x1xf32>
    %v6445 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6446 = stablehlo.add %v6444, %v6445 : tensor<144x24x1x1xf32>
    %v6447 = stablehlo.sqrt %v6446 : tensor<144x24x1x1xf32>
    %v6448 = stablehlo.divide %v6432, %v6447 : tensor<144x24x1x1xf32>
    %v6449 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6450 = stablehlo.multiply %v6449, %b4eWm : tensor<144x24x1x1xf32>
    %v6451 = stablehlo.add %v6450, %v6448 : tensor<144x24x1x1xf32>
    %v6452 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6453 = stablehlo.multiply %v6452, %v6451 : tensor<144x24x1x1xf32>
    %v6454 = stablehlo.subtract %b4eW, %v6453 : tensor<144x24x1x1xf32>
    %v6455 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6456 = stablehlo.multiply %v6455, %b4eg : tensor<144xf32>
    %v6457 = stablehlo.add %v6456, %v4270 : tensor<144xf32>
    %v6458 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6459 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6460 = stablehlo.multiply %v6458, %b4egv : tensor<144xf32>
    %v6461 = stablehlo.multiply %v6457, %v6457 : tensor<144xf32>
    %v6462 = stablehlo.multiply %v6459, %v6461 : tensor<144xf32>
    %v6463 = stablehlo.add %v6460, %v6462 : tensor<144xf32>
    %v6464 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6465 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6466 = stablehlo.multiply %v6464, %b4egv : tensor<144xf32>
    %v6467 = stablehlo.multiply %v6457, %v6457 : tensor<144xf32>
    %v6468 = stablehlo.multiply %v6465, %v6467 : tensor<144xf32>
    %v6469 = stablehlo.add %v6466, %v6468 : tensor<144xf32>
    %v6470 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6471 = stablehlo.add %v6469, %v6470 : tensor<144xf32>
    %v6472 = stablehlo.sqrt %v6471 : tensor<144xf32>
    %v6473 = stablehlo.divide %v6457, %v6472 : tensor<144xf32>
    %v6474 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6475 = stablehlo.multiply %v6474, %b4egm : tensor<144xf32>
    %v6476 = stablehlo.add %v6475, %v6473 : tensor<144xf32>
    %v6477 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6478 = stablehlo.multiply %v6477, %v6476 : tensor<144xf32>
    %v6479 = stablehlo.subtract %b4eg, %v6478 : tensor<144xf32>
    %v6480 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6481 = stablehlo.multiply %v6480, %b4ebt : tensor<144xf32>
    %v6482 = stablehlo.add %v6481, %v4273 : tensor<144xf32>
    %v6483 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6484 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6485 = stablehlo.multiply %v6483, %b4ebtv : tensor<144xf32>
    %v6486 = stablehlo.multiply %v6482, %v6482 : tensor<144xf32>
    %v6487 = stablehlo.multiply %v6484, %v6486 : tensor<144xf32>
    %v6488 = stablehlo.add %v6485, %v6487 : tensor<144xf32>
    %v6489 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6490 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6491 = stablehlo.multiply %v6489, %b4ebtv : tensor<144xf32>
    %v6492 = stablehlo.multiply %v6482, %v6482 : tensor<144xf32>
    %v6493 = stablehlo.multiply %v6490, %v6492 : tensor<144xf32>
    %v6494 = stablehlo.add %v6491, %v6493 : tensor<144xf32>
    %v6495 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6496 = stablehlo.add %v6494, %v6495 : tensor<144xf32>
    %v6497 = stablehlo.sqrt %v6496 : tensor<144xf32>
    %v6498 = stablehlo.divide %v6482, %v6497 : tensor<144xf32>
    %v6499 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6500 = stablehlo.multiply %v6499, %b4ebtm : tensor<144xf32>
    %v6501 = stablehlo.add %v6500, %v6498 : tensor<144xf32>
    %v6502 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6503 = stablehlo.multiply %v6502, %v6501 : tensor<144xf32>
    %v6504 = stablehlo.subtract %b4ebt, %v6503 : tensor<144xf32>
    %v6505 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6506 = stablehlo.multiply %v6505, %b4dW : tensor<144x1x3x3xf32>
    %v6507 = stablehlo.add %v6506, %v4281 : tensor<144x1x3x3xf32>
    %v6508 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6509 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6510 = stablehlo.multiply %v6508, %b4dWv : tensor<144x1x3x3xf32>
    %v6511 = stablehlo.multiply %v6507, %v6507 : tensor<144x1x3x3xf32>
    %v6512 = stablehlo.multiply %v6509, %v6511 : tensor<144x1x3x3xf32>
    %v6513 = stablehlo.add %v6510, %v6512 : tensor<144x1x3x3xf32>
    %v6514 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6515 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6516 = stablehlo.multiply %v6514, %b4dWv : tensor<144x1x3x3xf32>
    %v6517 = stablehlo.multiply %v6507, %v6507 : tensor<144x1x3x3xf32>
    %v6518 = stablehlo.multiply %v6515, %v6517 : tensor<144x1x3x3xf32>
    %v6519 = stablehlo.add %v6516, %v6518 : tensor<144x1x3x3xf32>
    %v6520 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6521 = stablehlo.add %v6519, %v6520 : tensor<144x1x3x3xf32>
    %v6522 = stablehlo.sqrt %v6521 : tensor<144x1x3x3xf32>
    %v6523 = stablehlo.divide %v6507, %v6522 : tensor<144x1x3x3xf32>
    %v6524 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6525 = stablehlo.multiply %v6524, %b4dWm : tensor<144x1x3x3xf32>
    %v6526 = stablehlo.add %v6525, %v6523 : tensor<144x1x3x3xf32>
    %v6527 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6528 = stablehlo.multiply %v6527, %v6526 : tensor<144x1x3x3xf32>
    %v6529 = stablehlo.subtract %b4dW, %v6528 : tensor<144x1x3x3xf32>
    %v6530 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6531 = stablehlo.multiply %v6530, %b4dg : tensor<144xf32>
    %v6532 = stablehlo.add %v6531, %v4299 : tensor<144xf32>
    %v6533 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6534 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6535 = stablehlo.multiply %v6533, %b4dgv : tensor<144xf32>
    %v6536 = stablehlo.multiply %v6532, %v6532 : tensor<144xf32>
    %v6537 = stablehlo.multiply %v6534, %v6536 : tensor<144xf32>
    %v6538 = stablehlo.add %v6535, %v6537 : tensor<144xf32>
    %v6539 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6540 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6541 = stablehlo.multiply %v6539, %b4dgv : tensor<144xf32>
    %v6542 = stablehlo.multiply %v6532, %v6532 : tensor<144xf32>
    %v6543 = stablehlo.multiply %v6540, %v6542 : tensor<144xf32>
    %v6544 = stablehlo.add %v6541, %v6543 : tensor<144xf32>
    %v6545 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6546 = stablehlo.add %v6544, %v6545 : tensor<144xf32>
    %v6547 = stablehlo.sqrt %v6546 : tensor<144xf32>
    %v6548 = stablehlo.divide %v6532, %v6547 : tensor<144xf32>
    %v6549 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6550 = stablehlo.multiply %v6549, %b4dgm : tensor<144xf32>
    %v6551 = stablehlo.add %v6550, %v6548 : tensor<144xf32>
    %v6552 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6553 = stablehlo.multiply %v6552, %v6551 : tensor<144xf32>
    %v6554 = stablehlo.subtract %b4dg, %v6553 : tensor<144xf32>
    %v6555 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6556 = stablehlo.multiply %v6555, %b4dbt : tensor<144xf32>
    %v6557 = stablehlo.add %v6556, %v4302 : tensor<144xf32>
    %v6558 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6559 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6560 = stablehlo.multiply %v6558, %b4dbtv : tensor<144xf32>
    %v6561 = stablehlo.multiply %v6557, %v6557 : tensor<144xf32>
    %v6562 = stablehlo.multiply %v6559, %v6561 : tensor<144xf32>
    %v6563 = stablehlo.add %v6560, %v6562 : tensor<144xf32>
    %v6564 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6565 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6566 = stablehlo.multiply %v6564, %b4dbtv : tensor<144xf32>
    %v6567 = stablehlo.multiply %v6557, %v6557 : tensor<144xf32>
    %v6568 = stablehlo.multiply %v6565, %v6567 : tensor<144xf32>
    %v6569 = stablehlo.add %v6566, %v6568 : tensor<144xf32>
    %v6570 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6571 = stablehlo.add %v6569, %v6570 : tensor<144xf32>
    %v6572 = stablehlo.sqrt %v6571 : tensor<144xf32>
    %v6573 = stablehlo.divide %v6557, %v6572 : tensor<144xf32>
    %v6574 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6575 = stablehlo.multiply %v6574, %b4dbtm : tensor<144xf32>
    %v6576 = stablehlo.add %v6575, %v6573 : tensor<144xf32>
    %v6577 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6578 = stablehlo.multiply %v6577, %v6576 : tensor<144xf32>
    %v6579 = stablehlo.subtract %b4dbt, %v6578 : tensor<144xf32>
    %v6580 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6581 = stablehlo.multiply %v6580, %b4pW : tensor<32x144x1x1xf32>
    %v6582 = stablehlo.add %v6581, %v4308 : tensor<32x144x1x1xf32>
    %v6583 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6584 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6585 = stablehlo.multiply %v6583, %b4pWv : tensor<32x144x1x1xf32>
    %v6586 = stablehlo.multiply %v6582, %v6582 : tensor<32x144x1x1xf32>
    %v6587 = stablehlo.multiply %v6584, %v6586 : tensor<32x144x1x1xf32>
    %v6588 = stablehlo.add %v6585, %v6587 : tensor<32x144x1x1xf32>
    %v6589 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6590 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6591 = stablehlo.multiply %v6589, %b4pWv : tensor<32x144x1x1xf32>
    %v6592 = stablehlo.multiply %v6582, %v6582 : tensor<32x144x1x1xf32>
    %v6593 = stablehlo.multiply %v6590, %v6592 : tensor<32x144x1x1xf32>
    %v6594 = stablehlo.add %v6591, %v6593 : tensor<32x144x1x1xf32>
    %v6595 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6596 = stablehlo.add %v6594, %v6595 : tensor<32x144x1x1xf32>
    %v6597 = stablehlo.sqrt %v6596 : tensor<32x144x1x1xf32>
    %v6598 = stablehlo.divide %v6582, %v6597 : tensor<32x144x1x1xf32>
    %v6599 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6600 = stablehlo.multiply %v6599, %b4pWm : tensor<32x144x1x1xf32>
    %v6601 = stablehlo.add %v6600, %v6598 : tensor<32x144x1x1xf32>
    %v6602 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6603 = stablehlo.multiply %v6602, %v6601 : tensor<32x144x1x1xf32>
    %v6604 = stablehlo.subtract %b4pW, %v6603 : tensor<32x144x1x1xf32>
    %v6605 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6606 = stablehlo.multiply %v6605, %b4pg : tensor<32xf32>
    %v6607 = stablehlo.add %v6606, %v4326 : tensor<32xf32>
    %v6608 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6609 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6610 = stablehlo.multiply %v6608, %b4pgv : tensor<32xf32>
    %v6611 = stablehlo.multiply %v6607, %v6607 : tensor<32xf32>
    %v6612 = stablehlo.multiply %v6609, %v6611 : tensor<32xf32>
    %v6613 = stablehlo.add %v6610, %v6612 : tensor<32xf32>
    %v6614 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6615 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6616 = stablehlo.multiply %v6614, %b4pgv : tensor<32xf32>
    %v6617 = stablehlo.multiply %v6607, %v6607 : tensor<32xf32>
    %v6618 = stablehlo.multiply %v6615, %v6617 : tensor<32xf32>
    %v6619 = stablehlo.add %v6616, %v6618 : tensor<32xf32>
    %v6620 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6621 = stablehlo.add %v6619, %v6620 : tensor<32xf32>
    %v6622 = stablehlo.sqrt %v6621 : tensor<32xf32>
    %v6623 = stablehlo.divide %v6607, %v6622 : tensor<32xf32>
    %v6624 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6625 = stablehlo.multiply %v6624, %b4pgm : tensor<32xf32>
    %v6626 = stablehlo.add %v6625, %v6623 : tensor<32xf32>
    %v6627 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6628 = stablehlo.multiply %v6627, %v6626 : tensor<32xf32>
    %v6629 = stablehlo.subtract %b4pg, %v6628 : tensor<32xf32>
    %v6630 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6631 = stablehlo.multiply %v6630, %b4pbt : tensor<32xf32>
    %v6632 = stablehlo.add %v6631, %v4329 : tensor<32xf32>
    %v6633 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6634 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6635 = stablehlo.multiply %v6633, %b4pbtv : tensor<32xf32>
    %v6636 = stablehlo.multiply %v6632, %v6632 : tensor<32xf32>
    %v6637 = stablehlo.multiply %v6634, %v6636 : tensor<32xf32>
    %v6638 = stablehlo.add %v6635, %v6637 : tensor<32xf32>
    %v6639 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6640 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6641 = stablehlo.multiply %v6639, %b4pbtv : tensor<32xf32>
    %v6642 = stablehlo.multiply %v6632, %v6632 : tensor<32xf32>
    %v6643 = stablehlo.multiply %v6640, %v6642 : tensor<32xf32>
    %v6644 = stablehlo.add %v6641, %v6643 : tensor<32xf32>
    %v6645 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6646 = stablehlo.add %v6644, %v6645 : tensor<32xf32>
    %v6647 = stablehlo.sqrt %v6646 : tensor<32xf32>
    %v6648 = stablehlo.divide %v6632, %v6647 : tensor<32xf32>
    %v6649 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6650 = stablehlo.multiply %v6649, %b4pbtm : tensor<32xf32>
    %v6651 = stablehlo.add %v6650, %v6648 : tensor<32xf32>
    %v6652 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6653 = stablehlo.multiply %v6652, %v6651 : tensor<32xf32>
    %v6654 = stablehlo.subtract %b4pbt, %v6653 : tensor<32xf32>
    %v6655 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6656 = stablehlo.multiply %v6655, %b5eW : tensor<192x32x1x1xf32>
    %v6657 = stablehlo.add %v6656, %v4053 : tensor<192x32x1x1xf32>
    %v6658 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6659 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6660 = stablehlo.multiply %v6658, %b5eWv : tensor<192x32x1x1xf32>
    %v6661 = stablehlo.multiply %v6657, %v6657 : tensor<192x32x1x1xf32>
    %v6662 = stablehlo.multiply %v6659, %v6661 : tensor<192x32x1x1xf32>
    %v6663 = stablehlo.add %v6660, %v6662 : tensor<192x32x1x1xf32>
    %v6664 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6665 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6666 = stablehlo.multiply %v6664, %b5eWv : tensor<192x32x1x1xf32>
    %v6667 = stablehlo.multiply %v6657, %v6657 : tensor<192x32x1x1xf32>
    %v6668 = stablehlo.multiply %v6665, %v6667 : tensor<192x32x1x1xf32>
    %v6669 = stablehlo.add %v6666, %v6668 : tensor<192x32x1x1xf32>
    %v6670 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6671 = stablehlo.add %v6669, %v6670 : tensor<192x32x1x1xf32>
    %v6672 = stablehlo.sqrt %v6671 : tensor<192x32x1x1xf32>
    %v6673 = stablehlo.divide %v6657, %v6672 : tensor<192x32x1x1xf32>
    %v6674 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6675 = stablehlo.multiply %v6674, %b5eWm : tensor<192x32x1x1xf32>
    %v6676 = stablehlo.add %v6675, %v6673 : tensor<192x32x1x1xf32>
    %v6677 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6678 = stablehlo.multiply %v6677, %v6676 : tensor<192x32x1x1xf32>
    %v6679 = stablehlo.subtract %b5eW, %v6678 : tensor<192x32x1x1xf32>
    %v6680 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6681 = stablehlo.multiply %v6680, %b5eg : tensor<192xf32>
    %v6682 = stablehlo.add %v6681, %v4071 : tensor<192xf32>
    %v6683 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6684 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6685 = stablehlo.multiply %v6683, %b5egv : tensor<192xf32>
    %v6686 = stablehlo.multiply %v6682, %v6682 : tensor<192xf32>
    %v6687 = stablehlo.multiply %v6684, %v6686 : tensor<192xf32>
    %v6688 = stablehlo.add %v6685, %v6687 : tensor<192xf32>
    %v6689 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6690 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6691 = stablehlo.multiply %v6689, %b5egv : tensor<192xf32>
    %v6692 = stablehlo.multiply %v6682, %v6682 : tensor<192xf32>
    %v6693 = stablehlo.multiply %v6690, %v6692 : tensor<192xf32>
    %v6694 = stablehlo.add %v6691, %v6693 : tensor<192xf32>
    %v6695 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6696 = stablehlo.add %v6694, %v6695 : tensor<192xf32>
    %v6697 = stablehlo.sqrt %v6696 : tensor<192xf32>
    %v6698 = stablehlo.divide %v6682, %v6697 : tensor<192xf32>
    %v6699 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6700 = stablehlo.multiply %v6699, %b5egm : tensor<192xf32>
    %v6701 = stablehlo.add %v6700, %v6698 : tensor<192xf32>
    %v6702 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6703 = stablehlo.multiply %v6702, %v6701 : tensor<192xf32>
    %v6704 = stablehlo.subtract %b5eg, %v6703 : tensor<192xf32>
    %v6705 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6706 = stablehlo.multiply %v6705, %b5ebt : tensor<192xf32>
    %v6707 = stablehlo.add %v6706, %v4074 : tensor<192xf32>
    %v6708 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6709 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6710 = stablehlo.multiply %v6708, %b5ebtv : tensor<192xf32>
    %v6711 = stablehlo.multiply %v6707, %v6707 : tensor<192xf32>
    %v6712 = stablehlo.multiply %v6709, %v6711 : tensor<192xf32>
    %v6713 = stablehlo.add %v6710, %v6712 : tensor<192xf32>
    %v6714 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6715 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6716 = stablehlo.multiply %v6714, %b5ebtv : tensor<192xf32>
    %v6717 = stablehlo.multiply %v6707, %v6707 : tensor<192xf32>
    %v6718 = stablehlo.multiply %v6715, %v6717 : tensor<192xf32>
    %v6719 = stablehlo.add %v6716, %v6718 : tensor<192xf32>
    %v6720 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6721 = stablehlo.add %v6719, %v6720 : tensor<192xf32>
    %v6722 = stablehlo.sqrt %v6721 : tensor<192xf32>
    %v6723 = stablehlo.divide %v6707, %v6722 : tensor<192xf32>
    %v6724 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6725 = stablehlo.multiply %v6724, %b5ebtm : tensor<192xf32>
    %v6726 = stablehlo.add %v6725, %v6723 : tensor<192xf32>
    %v6727 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6728 = stablehlo.multiply %v6727, %v6726 : tensor<192xf32>
    %v6729 = stablehlo.subtract %b5ebt, %v6728 : tensor<192xf32>
    %v6730 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6731 = stablehlo.multiply %v6730, %b5dW : tensor<192x1x3x3xf32>
    %v6732 = stablehlo.add %v6731, %v4080 : tensor<192x1x3x3xf32>
    %v6733 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6734 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6735 = stablehlo.multiply %v6733, %b5dWv : tensor<192x1x3x3xf32>
    %v6736 = stablehlo.multiply %v6732, %v6732 : tensor<192x1x3x3xf32>
    %v6737 = stablehlo.multiply %v6734, %v6736 : tensor<192x1x3x3xf32>
    %v6738 = stablehlo.add %v6735, %v6737 : tensor<192x1x3x3xf32>
    %v6739 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6740 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6741 = stablehlo.multiply %v6739, %b5dWv : tensor<192x1x3x3xf32>
    %v6742 = stablehlo.multiply %v6732, %v6732 : tensor<192x1x3x3xf32>
    %v6743 = stablehlo.multiply %v6740, %v6742 : tensor<192x1x3x3xf32>
    %v6744 = stablehlo.add %v6741, %v6743 : tensor<192x1x3x3xf32>
    %v6745 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6746 = stablehlo.add %v6744, %v6745 : tensor<192x1x3x3xf32>
    %v6747 = stablehlo.sqrt %v6746 : tensor<192x1x3x3xf32>
    %v6748 = stablehlo.divide %v6732, %v6747 : tensor<192x1x3x3xf32>
    %v6749 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6750 = stablehlo.multiply %v6749, %b5dWm : tensor<192x1x3x3xf32>
    %v6751 = stablehlo.add %v6750, %v6748 : tensor<192x1x3x3xf32>
    %v6752 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6753 = stablehlo.multiply %v6752, %v6751 : tensor<192x1x3x3xf32>
    %v6754 = stablehlo.subtract %b5dW, %v6753 : tensor<192x1x3x3xf32>
    %v6755 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6756 = stablehlo.multiply %v6755, %b5dg : tensor<192xf32>
    %v6757 = stablehlo.add %v6756, %v4098 : tensor<192xf32>
    %v6758 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6759 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6760 = stablehlo.multiply %v6758, %b5dgv : tensor<192xf32>
    %v6761 = stablehlo.multiply %v6757, %v6757 : tensor<192xf32>
    %v6762 = stablehlo.multiply %v6759, %v6761 : tensor<192xf32>
    %v6763 = stablehlo.add %v6760, %v6762 : tensor<192xf32>
    %v6764 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6765 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6766 = stablehlo.multiply %v6764, %b5dgv : tensor<192xf32>
    %v6767 = stablehlo.multiply %v6757, %v6757 : tensor<192xf32>
    %v6768 = stablehlo.multiply %v6765, %v6767 : tensor<192xf32>
    %v6769 = stablehlo.add %v6766, %v6768 : tensor<192xf32>
    %v6770 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6771 = stablehlo.add %v6769, %v6770 : tensor<192xf32>
    %v6772 = stablehlo.sqrt %v6771 : tensor<192xf32>
    %v6773 = stablehlo.divide %v6757, %v6772 : tensor<192xf32>
    %v6774 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6775 = stablehlo.multiply %v6774, %b5dgm : tensor<192xf32>
    %v6776 = stablehlo.add %v6775, %v6773 : tensor<192xf32>
    %v6777 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6778 = stablehlo.multiply %v6777, %v6776 : tensor<192xf32>
    %v6779 = stablehlo.subtract %b5dg, %v6778 : tensor<192xf32>
    %v6780 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6781 = stablehlo.multiply %v6780, %b5dbt : tensor<192xf32>
    %v6782 = stablehlo.add %v6781, %v4101 : tensor<192xf32>
    %v6783 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6784 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6785 = stablehlo.multiply %v6783, %b5dbtv : tensor<192xf32>
    %v6786 = stablehlo.multiply %v6782, %v6782 : tensor<192xf32>
    %v6787 = stablehlo.multiply %v6784, %v6786 : tensor<192xf32>
    %v6788 = stablehlo.add %v6785, %v6787 : tensor<192xf32>
    %v6789 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6790 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6791 = stablehlo.multiply %v6789, %b5dbtv : tensor<192xf32>
    %v6792 = stablehlo.multiply %v6782, %v6782 : tensor<192xf32>
    %v6793 = stablehlo.multiply %v6790, %v6792 : tensor<192xf32>
    %v6794 = stablehlo.add %v6791, %v6793 : tensor<192xf32>
    %v6795 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6796 = stablehlo.add %v6794, %v6795 : tensor<192xf32>
    %v6797 = stablehlo.sqrt %v6796 : tensor<192xf32>
    %v6798 = stablehlo.divide %v6782, %v6797 : tensor<192xf32>
    %v6799 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6800 = stablehlo.multiply %v6799, %b5dbtm : tensor<192xf32>
    %v6801 = stablehlo.add %v6800, %v6798 : tensor<192xf32>
    %v6802 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6803 = stablehlo.multiply %v6802, %v6801 : tensor<192xf32>
    %v6804 = stablehlo.subtract %b5dbt, %v6803 : tensor<192xf32>
    %v6805 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v6806 = stablehlo.multiply %v6805, %b5pW : tensor<32x192x1x1xf32>
    %v6807 = stablehlo.add %v6806, %v4107 : tensor<32x192x1x1xf32>
    %v6808 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v6809 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v6810 = stablehlo.multiply %v6808, %b5pWv : tensor<32x192x1x1xf32>
    %v6811 = stablehlo.multiply %v6807, %v6807 : tensor<32x192x1x1xf32>
    %v6812 = stablehlo.multiply %v6809, %v6811 : tensor<32x192x1x1xf32>
    %v6813 = stablehlo.add %v6810, %v6812 : tensor<32x192x1x1xf32>
    %v6814 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v6815 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v6816 = stablehlo.multiply %v6814, %b5pWv : tensor<32x192x1x1xf32>
    %v6817 = stablehlo.multiply %v6807, %v6807 : tensor<32x192x1x1xf32>
    %v6818 = stablehlo.multiply %v6815, %v6817 : tensor<32x192x1x1xf32>
    %v6819 = stablehlo.add %v6816, %v6818 : tensor<32x192x1x1xf32>
    %v6820 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v6821 = stablehlo.add %v6819, %v6820 : tensor<32x192x1x1xf32>
    %v6822 = stablehlo.sqrt %v6821 : tensor<32x192x1x1xf32>
    %v6823 = stablehlo.divide %v6807, %v6822 : tensor<32x192x1x1xf32>
    %v6824 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v6825 = stablehlo.multiply %v6824, %b5pWm : tensor<32x192x1x1xf32>
    %v6826 = stablehlo.add %v6825, %v6823 : tensor<32x192x1x1xf32>
    %v6827 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v6828 = stablehlo.multiply %v6827, %v6826 : tensor<32x192x1x1xf32>
    %v6829 = stablehlo.subtract %b5pW, %v6828 : tensor<32x192x1x1xf32>
    %v6830 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6831 = stablehlo.multiply %v6830, %b5pg : tensor<32xf32>
    %v6832 = stablehlo.add %v6831, %v4125 : tensor<32xf32>
    %v6833 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6834 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6835 = stablehlo.multiply %v6833, %b5pgv : tensor<32xf32>
    %v6836 = stablehlo.multiply %v6832, %v6832 : tensor<32xf32>
    %v6837 = stablehlo.multiply %v6834, %v6836 : tensor<32xf32>
    %v6838 = stablehlo.add %v6835, %v6837 : tensor<32xf32>
    %v6839 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6840 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6841 = stablehlo.multiply %v6839, %b5pgv : tensor<32xf32>
    %v6842 = stablehlo.multiply %v6832, %v6832 : tensor<32xf32>
    %v6843 = stablehlo.multiply %v6840, %v6842 : tensor<32xf32>
    %v6844 = stablehlo.add %v6841, %v6843 : tensor<32xf32>
    %v6845 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6846 = stablehlo.add %v6844, %v6845 : tensor<32xf32>
    %v6847 = stablehlo.sqrt %v6846 : tensor<32xf32>
    %v6848 = stablehlo.divide %v6832, %v6847 : tensor<32xf32>
    %v6849 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6850 = stablehlo.multiply %v6849, %b5pgm : tensor<32xf32>
    %v6851 = stablehlo.add %v6850, %v6848 : tensor<32xf32>
    %v6852 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6853 = stablehlo.multiply %v6852, %v6851 : tensor<32xf32>
    %v6854 = stablehlo.subtract %b5pg, %v6853 : tensor<32xf32>
    %v6855 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6856 = stablehlo.multiply %v6855, %b5pbt : tensor<32xf32>
    %v6857 = stablehlo.add %v6856, %v4128 : tensor<32xf32>
    %v6858 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6859 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6860 = stablehlo.multiply %v6858, %b5pbtv : tensor<32xf32>
    %v6861 = stablehlo.multiply %v6857, %v6857 : tensor<32xf32>
    %v6862 = stablehlo.multiply %v6859, %v6861 : tensor<32xf32>
    %v6863 = stablehlo.add %v6860, %v6862 : tensor<32xf32>
    %v6864 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6865 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6866 = stablehlo.multiply %v6864, %b5pbtv : tensor<32xf32>
    %v6867 = stablehlo.multiply %v6857, %v6857 : tensor<32xf32>
    %v6868 = stablehlo.multiply %v6865, %v6867 : tensor<32xf32>
    %v6869 = stablehlo.add %v6866, %v6868 : tensor<32xf32>
    %v6870 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6871 = stablehlo.add %v6869, %v6870 : tensor<32xf32>
    %v6872 = stablehlo.sqrt %v6871 : tensor<32xf32>
    %v6873 = stablehlo.divide %v6857, %v6872 : tensor<32xf32>
    %v6874 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6875 = stablehlo.multiply %v6874, %b5pbtm : tensor<32xf32>
    %v6876 = stablehlo.add %v6875, %v6873 : tensor<32xf32>
    %v6877 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6878 = stablehlo.multiply %v6877, %v6876 : tensor<32xf32>
    %v6879 = stablehlo.subtract %b5pbt, %v6878 : tensor<32xf32>
    %v6880 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6881 = stablehlo.multiply %v6880, %b6eW : tensor<192x32x1x1xf32>
    %v6882 = stablehlo.add %v6881, %v3855 : tensor<192x32x1x1xf32>
    %v6883 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6884 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6885 = stablehlo.multiply %v6883, %b6eWv : tensor<192x32x1x1xf32>
    %v6886 = stablehlo.multiply %v6882, %v6882 : tensor<192x32x1x1xf32>
    %v6887 = stablehlo.multiply %v6884, %v6886 : tensor<192x32x1x1xf32>
    %v6888 = stablehlo.add %v6885, %v6887 : tensor<192x32x1x1xf32>
    %v6889 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6890 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6891 = stablehlo.multiply %v6889, %b6eWv : tensor<192x32x1x1xf32>
    %v6892 = stablehlo.multiply %v6882, %v6882 : tensor<192x32x1x1xf32>
    %v6893 = stablehlo.multiply %v6890, %v6892 : tensor<192x32x1x1xf32>
    %v6894 = stablehlo.add %v6891, %v6893 : tensor<192x32x1x1xf32>
    %v6895 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6896 = stablehlo.add %v6894, %v6895 : tensor<192x32x1x1xf32>
    %v6897 = stablehlo.sqrt %v6896 : tensor<192x32x1x1xf32>
    %v6898 = stablehlo.divide %v6882, %v6897 : tensor<192x32x1x1xf32>
    %v6899 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6900 = stablehlo.multiply %v6899, %b6eWm : tensor<192x32x1x1xf32>
    %v6901 = stablehlo.add %v6900, %v6898 : tensor<192x32x1x1xf32>
    %v6902 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6903 = stablehlo.multiply %v6902, %v6901 : tensor<192x32x1x1xf32>
    %v6904 = stablehlo.subtract %b6eW, %v6903 : tensor<192x32x1x1xf32>
    %v6905 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6906 = stablehlo.multiply %v6905, %b6eg : tensor<192xf32>
    %v6907 = stablehlo.add %v6906, %v3873 : tensor<192xf32>
    %v6908 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6909 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6910 = stablehlo.multiply %v6908, %b6egv : tensor<192xf32>
    %v6911 = stablehlo.multiply %v6907, %v6907 : tensor<192xf32>
    %v6912 = stablehlo.multiply %v6909, %v6911 : tensor<192xf32>
    %v6913 = stablehlo.add %v6910, %v6912 : tensor<192xf32>
    %v6914 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6915 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6916 = stablehlo.multiply %v6914, %b6egv : tensor<192xf32>
    %v6917 = stablehlo.multiply %v6907, %v6907 : tensor<192xf32>
    %v6918 = stablehlo.multiply %v6915, %v6917 : tensor<192xf32>
    %v6919 = stablehlo.add %v6916, %v6918 : tensor<192xf32>
    %v6920 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6921 = stablehlo.add %v6919, %v6920 : tensor<192xf32>
    %v6922 = stablehlo.sqrt %v6921 : tensor<192xf32>
    %v6923 = stablehlo.divide %v6907, %v6922 : tensor<192xf32>
    %v6924 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6925 = stablehlo.multiply %v6924, %b6egm : tensor<192xf32>
    %v6926 = stablehlo.add %v6925, %v6923 : tensor<192xf32>
    %v6927 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6928 = stablehlo.multiply %v6927, %v6926 : tensor<192xf32>
    %v6929 = stablehlo.subtract %b6eg, %v6928 : tensor<192xf32>
    %v6930 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6931 = stablehlo.multiply %v6930, %b6ebt : tensor<192xf32>
    %v6932 = stablehlo.add %v6931, %v3876 : tensor<192xf32>
    %v6933 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6934 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6935 = stablehlo.multiply %v6933, %b6ebtv : tensor<192xf32>
    %v6936 = stablehlo.multiply %v6932, %v6932 : tensor<192xf32>
    %v6937 = stablehlo.multiply %v6934, %v6936 : tensor<192xf32>
    %v6938 = stablehlo.add %v6935, %v6937 : tensor<192xf32>
    %v6939 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6940 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6941 = stablehlo.multiply %v6939, %b6ebtv : tensor<192xf32>
    %v6942 = stablehlo.multiply %v6932, %v6932 : tensor<192xf32>
    %v6943 = stablehlo.multiply %v6940, %v6942 : tensor<192xf32>
    %v6944 = stablehlo.add %v6941, %v6943 : tensor<192xf32>
    %v6945 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6946 = stablehlo.add %v6944, %v6945 : tensor<192xf32>
    %v6947 = stablehlo.sqrt %v6946 : tensor<192xf32>
    %v6948 = stablehlo.divide %v6932, %v6947 : tensor<192xf32>
    %v6949 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6950 = stablehlo.multiply %v6949, %b6ebtm : tensor<192xf32>
    %v6951 = stablehlo.add %v6950, %v6948 : tensor<192xf32>
    %v6952 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6953 = stablehlo.multiply %v6952, %v6951 : tensor<192xf32>
    %v6954 = stablehlo.subtract %b6ebt, %v6953 : tensor<192xf32>
    %v6955 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6956 = stablehlo.multiply %v6955, %b6dW : tensor<192x1x3x3xf32>
    %v6957 = stablehlo.add %v6956, %v3882 : tensor<192x1x3x3xf32>
    %v6958 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6959 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6960 = stablehlo.multiply %v6958, %b6dWv : tensor<192x1x3x3xf32>
    %v6961 = stablehlo.multiply %v6957, %v6957 : tensor<192x1x3x3xf32>
    %v6962 = stablehlo.multiply %v6959, %v6961 : tensor<192x1x3x3xf32>
    %v6963 = stablehlo.add %v6960, %v6962 : tensor<192x1x3x3xf32>
    %v6964 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6965 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6966 = stablehlo.multiply %v6964, %b6dWv : tensor<192x1x3x3xf32>
    %v6967 = stablehlo.multiply %v6957, %v6957 : tensor<192x1x3x3xf32>
    %v6968 = stablehlo.multiply %v6965, %v6967 : tensor<192x1x3x3xf32>
    %v6969 = stablehlo.add %v6966, %v6968 : tensor<192x1x3x3xf32>
    %v6970 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6971 = stablehlo.add %v6969, %v6970 : tensor<192x1x3x3xf32>
    %v6972 = stablehlo.sqrt %v6971 : tensor<192x1x3x3xf32>
    %v6973 = stablehlo.divide %v6957, %v6972 : tensor<192x1x3x3xf32>
    %v6974 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6975 = stablehlo.multiply %v6974, %b6dWm : tensor<192x1x3x3xf32>
    %v6976 = stablehlo.add %v6975, %v6973 : tensor<192x1x3x3xf32>
    %v6977 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6978 = stablehlo.multiply %v6977, %v6976 : tensor<192x1x3x3xf32>
    %v6979 = stablehlo.subtract %b6dW, %v6978 : tensor<192x1x3x3xf32>
    %v6980 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6981 = stablehlo.multiply %v6980, %b6dg : tensor<192xf32>
    %v6982 = stablehlo.add %v6981, %v3900 : tensor<192xf32>
    %v6983 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6984 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6985 = stablehlo.multiply %v6983, %b6dgv : tensor<192xf32>
    %v6986 = stablehlo.multiply %v6982, %v6982 : tensor<192xf32>
    %v6987 = stablehlo.multiply %v6984, %v6986 : tensor<192xf32>
    %v6988 = stablehlo.add %v6985, %v6987 : tensor<192xf32>
    %v6989 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6990 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6991 = stablehlo.multiply %v6989, %b6dgv : tensor<192xf32>
    %v6992 = stablehlo.multiply %v6982, %v6982 : tensor<192xf32>
    %v6993 = stablehlo.multiply %v6990, %v6992 : tensor<192xf32>
    %v6994 = stablehlo.add %v6991, %v6993 : tensor<192xf32>
    %v6995 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6996 = stablehlo.add %v6994, %v6995 : tensor<192xf32>
    %v6997 = stablehlo.sqrt %v6996 : tensor<192xf32>
    %v6998 = stablehlo.divide %v6982, %v6997 : tensor<192xf32>
    %v6999 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7000 = stablehlo.multiply %v6999, %b6dgm : tensor<192xf32>
    %v7001 = stablehlo.add %v7000, %v6998 : tensor<192xf32>
    %v7002 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7003 = stablehlo.multiply %v7002, %v7001 : tensor<192xf32>
    %v7004 = stablehlo.subtract %b6dg, %v7003 : tensor<192xf32>
    %v7005 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7006 = stablehlo.multiply %v7005, %b6dbt : tensor<192xf32>
    %v7007 = stablehlo.add %v7006, %v3903 : tensor<192xf32>
    %v7008 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7009 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7010 = stablehlo.multiply %v7008, %b6dbtv : tensor<192xf32>
    %v7011 = stablehlo.multiply %v7007, %v7007 : tensor<192xf32>
    %v7012 = stablehlo.multiply %v7009, %v7011 : tensor<192xf32>
    %v7013 = stablehlo.add %v7010, %v7012 : tensor<192xf32>
    %v7014 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7015 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7016 = stablehlo.multiply %v7014, %b6dbtv : tensor<192xf32>
    %v7017 = stablehlo.multiply %v7007, %v7007 : tensor<192xf32>
    %v7018 = stablehlo.multiply %v7015, %v7017 : tensor<192xf32>
    %v7019 = stablehlo.add %v7016, %v7018 : tensor<192xf32>
    %v7020 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7021 = stablehlo.add %v7019, %v7020 : tensor<192xf32>
    %v7022 = stablehlo.sqrt %v7021 : tensor<192xf32>
    %v7023 = stablehlo.divide %v7007, %v7022 : tensor<192xf32>
    %v7024 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7025 = stablehlo.multiply %v7024, %b6dbtm : tensor<192xf32>
    %v7026 = stablehlo.add %v7025, %v7023 : tensor<192xf32>
    %v7027 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7028 = stablehlo.multiply %v7027, %v7026 : tensor<192xf32>
    %v7029 = stablehlo.subtract %b6dbt, %v7028 : tensor<192xf32>
    %v7030 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7031 = stablehlo.multiply %v7030, %b6pW : tensor<32x192x1x1xf32>
    %v7032 = stablehlo.add %v7031, %v3909 : tensor<32x192x1x1xf32>
    %v7033 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7034 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7035 = stablehlo.multiply %v7033, %b6pWv : tensor<32x192x1x1xf32>
    %v7036 = stablehlo.multiply %v7032, %v7032 : tensor<32x192x1x1xf32>
    %v7037 = stablehlo.multiply %v7034, %v7036 : tensor<32x192x1x1xf32>
    %v7038 = stablehlo.add %v7035, %v7037 : tensor<32x192x1x1xf32>
    %v7039 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7040 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7041 = stablehlo.multiply %v7039, %b6pWv : tensor<32x192x1x1xf32>
    %v7042 = stablehlo.multiply %v7032, %v7032 : tensor<32x192x1x1xf32>
    %v7043 = stablehlo.multiply %v7040, %v7042 : tensor<32x192x1x1xf32>
    %v7044 = stablehlo.add %v7041, %v7043 : tensor<32x192x1x1xf32>
    %v7045 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7046 = stablehlo.add %v7044, %v7045 : tensor<32x192x1x1xf32>
    %v7047 = stablehlo.sqrt %v7046 : tensor<32x192x1x1xf32>
    %v7048 = stablehlo.divide %v7032, %v7047 : tensor<32x192x1x1xf32>
    %v7049 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7050 = stablehlo.multiply %v7049, %b6pWm : tensor<32x192x1x1xf32>
    %v7051 = stablehlo.add %v7050, %v7048 : tensor<32x192x1x1xf32>
    %v7052 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7053 = stablehlo.multiply %v7052, %v7051 : tensor<32x192x1x1xf32>
    %v7054 = stablehlo.subtract %b6pW, %v7053 : tensor<32x192x1x1xf32>
    %v7055 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7056 = stablehlo.multiply %v7055, %b6pg : tensor<32xf32>
    %v7057 = stablehlo.add %v7056, %v3927 : tensor<32xf32>
    %v7058 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7059 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7060 = stablehlo.multiply %v7058, %b6pgv : tensor<32xf32>
    %v7061 = stablehlo.multiply %v7057, %v7057 : tensor<32xf32>
    %v7062 = stablehlo.multiply %v7059, %v7061 : tensor<32xf32>
    %v7063 = stablehlo.add %v7060, %v7062 : tensor<32xf32>
    %v7064 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7065 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7066 = stablehlo.multiply %v7064, %b6pgv : tensor<32xf32>
    %v7067 = stablehlo.multiply %v7057, %v7057 : tensor<32xf32>
    %v7068 = stablehlo.multiply %v7065, %v7067 : tensor<32xf32>
    %v7069 = stablehlo.add %v7066, %v7068 : tensor<32xf32>
    %v7070 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7071 = stablehlo.add %v7069, %v7070 : tensor<32xf32>
    %v7072 = stablehlo.sqrt %v7071 : tensor<32xf32>
    %v7073 = stablehlo.divide %v7057, %v7072 : tensor<32xf32>
    %v7074 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7075 = stablehlo.multiply %v7074, %b6pgm : tensor<32xf32>
    %v7076 = stablehlo.add %v7075, %v7073 : tensor<32xf32>
    %v7077 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7078 = stablehlo.multiply %v7077, %v7076 : tensor<32xf32>
    %v7079 = stablehlo.subtract %b6pg, %v7078 : tensor<32xf32>
    %v7080 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7081 = stablehlo.multiply %v7080, %b6pbt : tensor<32xf32>
    %v7082 = stablehlo.add %v7081, %v3930 : tensor<32xf32>
    %v7083 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7084 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7085 = stablehlo.multiply %v7083, %b6pbtv : tensor<32xf32>
    %v7086 = stablehlo.multiply %v7082, %v7082 : tensor<32xf32>
    %v7087 = stablehlo.multiply %v7084, %v7086 : tensor<32xf32>
    %v7088 = stablehlo.add %v7085, %v7087 : tensor<32xf32>
    %v7089 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7090 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7091 = stablehlo.multiply %v7089, %b6pbtv : tensor<32xf32>
    %v7092 = stablehlo.multiply %v7082, %v7082 : tensor<32xf32>
    %v7093 = stablehlo.multiply %v7090, %v7092 : tensor<32xf32>
    %v7094 = stablehlo.add %v7091, %v7093 : tensor<32xf32>
    %v7095 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7096 = stablehlo.add %v7094, %v7095 : tensor<32xf32>
    %v7097 = stablehlo.sqrt %v7096 : tensor<32xf32>
    %v7098 = stablehlo.divide %v7082, %v7097 : tensor<32xf32>
    %v7099 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7100 = stablehlo.multiply %v7099, %b6pbtm : tensor<32xf32>
    %v7101 = stablehlo.add %v7100, %v7098 : tensor<32xf32>
    %v7102 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7103 = stablehlo.multiply %v7102, %v7101 : tensor<32xf32>
    %v7104 = stablehlo.subtract %b6pbt, %v7103 : tensor<32xf32>
    %v7105 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7106 = stablehlo.multiply %v7105, %b7eW : tensor<192x32x1x1xf32>
    %v7107 = stablehlo.add %v7106, %v3655 : tensor<192x32x1x1xf32>
    %v7108 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7109 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7110 = stablehlo.multiply %v7108, %b7eWv : tensor<192x32x1x1xf32>
    %v7111 = stablehlo.multiply %v7107, %v7107 : tensor<192x32x1x1xf32>
    %v7112 = stablehlo.multiply %v7109, %v7111 : tensor<192x32x1x1xf32>
    %v7113 = stablehlo.add %v7110, %v7112 : tensor<192x32x1x1xf32>
    %v7114 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7115 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7116 = stablehlo.multiply %v7114, %b7eWv : tensor<192x32x1x1xf32>
    %v7117 = stablehlo.multiply %v7107, %v7107 : tensor<192x32x1x1xf32>
    %v7118 = stablehlo.multiply %v7115, %v7117 : tensor<192x32x1x1xf32>
    %v7119 = stablehlo.add %v7116, %v7118 : tensor<192x32x1x1xf32>
    %v7120 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7121 = stablehlo.add %v7119, %v7120 : tensor<192x32x1x1xf32>
    %v7122 = stablehlo.sqrt %v7121 : tensor<192x32x1x1xf32>
    %v7123 = stablehlo.divide %v7107, %v7122 : tensor<192x32x1x1xf32>
    %v7124 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7125 = stablehlo.multiply %v7124, %b7eWm : tensor<192x32x1x1xf32>
    %v7126 = stablehlo.add %v7125, %v7123 : tensor<192x32x1x1xf32>
    %v7127 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7128 = stablehlo.multiply %v7127, %v7126 : tensor<192x32x1x1xf32>
    %v7129 = stablehlo.subtract %b7eW, %v7128 : tensor<192x32x1x1xf32>
    %v7130 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7131 = stablehlo.multiply %v7130, %b7eg : tensor<192xf32>
    %v7132 = stablehlo.add %v7131, %v3673 : tensor<192xf32>
    %v7133 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7134 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7135 = stablehlo.multiply %v7133, %b7egv : tensor<192xf32>
    %v7136 = stablehlo.multiply %v7132, %v7132 : tensor<192xf32>
    %v7137 = stablehlo.multiply %v7134, %v7136 : tensor<192xf32>
    %v7138 = stablehlo.add %v7135, %v7137 : tensor<192xf32>
    %v7139 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7140 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7141 = stablehlo.multiply %v7139, %b7egv : tensor<192xf32>
    %v7142 = stablehlo.multiply %v7132, %v7132 : tensor<192xf32>
    %v7143 = stablehlo.multiply %v7140, %v7142 : tensor<192xf32>
    %v7144 = stablehlo.add %v7141, %v7143 : tensor<192xf32>
    %v7145 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7146 = stablehlo.add %v7144, %v7145 : tensor<192xf32>
    %v7147 = stablehlo.sqrt %v7146 : tensor<192xf32>
    %v7148 = stablehlo.divide %v7132, %v7147 : tensor<192xf32>
    %v7149 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7150 = stablehlo.multiply %v7149, %b7egm : tensor<192xf32>
    %v7151 = stablehlo.add %v7150, %v7148 : tensor<192xf32>
    %v7152 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7153 = stablehlo.multiply %v7152, %v7151 : tensor<192xf32>
    %v7154 = stablehlo.subtract %b7eg, %v7153 : tensor<192xf32>
    %v7155 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7156 = stablehlo.multiply %v7155, %b7ebt : tensor<192xf32>
    %v7157 = stablehlo.add %v7156, %v3676 : tensor<192xf32>
    %v7158 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7159 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7160 = stablehlo.multiply %v7158, %b7ebtv : tensor<192xf32>
    %v7161 = stablehlo.multiply %v7157, %v7157 : tensor<192xf32>
    %v7162 = stablehlo.multiply %v7159, %v7161 : tensor<192xf32>
    %v7163 = stablehlo.add %v7160, %v7162 : tensor<192xf32>
    %v7164 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7165 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7166 = stablehlo.multiply %v7164, %b7ebtv : tensor<192xf32>
    %v7167 = stablehlo.multiply %v7157, %v7157 : tensor<192xf32>
    %v7168 = stablehlo.multiply %v7165, %v7167 : tensor<192xf32>
    %v7169 = stablehlo.add %v7166, %v7168 : tensor<192xf32>
    %v7170 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7171 = stablehlo.add %v7169, %v7170 : tensor<192xf32>
    %v7172 = stablehlo.sqrt %v7171 : tensor<192xf32>
    %v7173 = stablehlo.divide %v7157, %v7172 : tensor<192xf32>
    %v7174 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7175 = stablehlo.multiply %v7174, %b7ebtm : tensor<192xf32>
    %v7176 = stablehlo.add %v7175, %v7173 : tensor<192xf32>
    %v7177 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7178 = stablehlo.multiply %v7177, %v7176 : tensor<192xf32>
    %v7179 = stablehlo.subtract %b7ebt, %v7178 : tensor<192xf32>
    %v7180 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7181 = stablehlo.multiply %v7180, %b7dW : tensor<192x1x3x3xf32>
    %v7182 = stablehlo.add %v7181, %v3684 : tensor<192x1x3x3xf32>
    %v7183 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7184 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7185 = stablehlo.multiply %v7183, %b7dWv : tensor<192x1x3x3xf32>
    %v7186 = stablehlo.multiply %v7182, %v7182 : tensor<192x1x3x3xf32>
    %v7187 = stablehlo.multiply %v7184, %v7186 : tensor<192x1x3x3xf32>
    %v7188 = stablehlo.add %v7185, %v7187 : tensor<192x1x3x3xf32>
    %v7189 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7190 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7191 = stablehlo.multiply %v7189, %b7dWv : tensor<192x1x3x3xf32>
    %v7192 = stablehlo.multiply %v7182, %v7182 : tensor<192x1x3x3xf32>
    %v7193 = stablehlo.multiply %v7190, %v7192 : tensor<192x1x3x3xf32>
    %v7194 = stablehlo.add %v7191, %v7193 : tensor<192x1x3x3xf32>
    %v7195 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7196 = stablehlo.add %v7194, %v7195 : tensor<192x1x3x3xf32>
    %v7197 = stablehlo.sqrt %v7196 : tensor<192x1x3x3xf32>
    %v7198 = stablehlo.divide %v7182, %v7197 : tensor<192x1x3x3xf32>
    %v7199 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7200 = stablehlo.multiply %v7199, %b7dWm : tensor<192x1x3x3xf32>
    %v7201 = stablehlo.add %v7200, %v7198 : tensor<192x1x3x3xf32>
    %v7202 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7203 = stablehlo.multiply %v7202, %v7201 : tensor<192x1x3x3xf32>
    %v7204 = stablehlo.subtract %b7dW, %v7203 : tensor<192x1x3x3xf32>
    %v7205 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7206 = stablehlo.multiply %v7205, %b7dg : tensor<192xf32>
    %v7207 = stablehlo.add %v7206, %v3702 : tensor<192xf32>
    %v7208 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7209 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7210 = stablehlo.multiply %v7208, %b7dgv : tensor<192xf32>
    %v7211 = stablehlo.multiply %v7207, %v7207 : tensor<192xf32>
    %v7212 = stablehlo.multiply %v7209, %v7211 : tensor<192xf32>
    %v7213 = stablehlo.add %v7210, %v7212 : tensor<192xf32>
    %v7214 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7215 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7216 = stablehlo.multiply %v7214, %b7dgv : tensor<192xf32>
    %v7217 = stablehlo.multiply %v7207, %v7207 : tensor<192xf32>
    %v7218 = stablehlo.multiply %v7215, %v7217 : tensor<192xf32>
    %v7219 = stablehlo.add %v7216, %v7218 : tensor<192xf32>
    %v7220 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7221 = stablehlo.add %v7219, %v7220 : tensor<192xf32>
    %v7222 = stablehlo.sqrt %v7221 : tensor<192xf32>
    %v7223 = stablehlo.divide %v7207, %v7222 : tensor<192xf32>
    %v7224 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7225 = stablehlo.multiply %v7224, %b7dgm : tensor<192xf32>
    %v7226 = stablehlo.add %v7225, %v7223 : tensor<192xf32>
    %v7227 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7228 = stablehlo.multiply %v7227, %v7226 : tensor<192xf32>
    %v7229 = stablehlo.subtract %b7dg, %v7228 : tensor<192xf32>
    %v7230 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7231 = stablehlo.multiply %v7230, %b7dbt : tensor<192xf32>
    %v7232 = stablehlo.add %v7231, %v3705 : tensor<192xf32>
    %v7233 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7234 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7235 = stablehlo.multiply %v7233, %b7dbtv : tensor<192xf32>
    %v7236 = stablehlo.multiply %v7232, %v7232 : tensor<192xf32>
    %v7237 = stablehlo.multiply %v7234, %v7236 : tensor<192xf32>
    %v7238 = stablehlo.add %v7235, %v7237 : tensor<192xf32>
    %v7239 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7240 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7241 = stablehlo.multiply %v7239, %b7dbtv : tensor<192xf32>
    %v7242 = stablehlo.multiply %v7232, %v7232 : tensor<192xf32>
    %v7243 = stablehlo.multiply %v7240, %v7242 : tensor<192xf32>
    %v7244 = stablehlo.add %v7241, %v7243 : tensor<192xf32>
    %v7245 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7246 = stablehlo.add %v7244, %v7245 : tensor<192xf32>
    %v7247 = stablehlo.sqrt %v7246 : tensor<192xf32>
    %v7248 = stablehlo.divide %v7232, %v7247 : tensor<192xf32>
    %v7249 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7250 = stablehlo.multiply %v7249, %b7dbtm : tensor<192xf32>
    %v7251 = stablehlo.add %v7250, %v7248 : tensor<192xf32>
    %v7252 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7253 = stablehlo.multiply %v7252, %v7251 : tensor<192xf32>
    %v7254 = stablehlo.subtract %b7dbt, %v7253 : tensor<192xf32>
    %v7255 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7256 = stablehlo.multiply %v7255, %b7pW : tensor<64x192x1x1xf32>
    %v7257 = stablehlo.add %v7256, %v3711 : tensor<64x192x1x1xf32>
    %v7258 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7259 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7260 = stablehlo.multiply %v7258, %b7pWv : tensor<64x192x1x1xf32>
    %v7261 = stablehlo.multiply %v7257, %v7257 : tensor<64x192x1x1xf32>
    %v7262 = stablehlo.multiply %v7259, %v7261 : tensor<64x192x1x1xf32>
    %v7263 = stablehlo.add %v7260, %v7262 : tensor<64x192x1x1xf32>
    %v7264 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7265 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7266 = stablehlo.multiply %v7264, %b7pWv : tensor<64x192x1x1xf32>
    %v7267 = stablehlo.multiply %v7257, %v7257 : tensor<64x192x1x1xf32>
    %v7268 = stablehlo.multiply %v7265, %v7267 : tensor<64x192x1x1xf32>
    %v7269 = stablehlo.add %v7266, %v7268 : tensor<64x192x1x1xf32>
    %v7270 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7271 = stablehlo.add %v7269, %v7270 : tensor<64x192x1x1xf32>
    %v7272 = stablehlo.sqrt %v7271 : tensor<64x192x1x1xf32>
    %v7273 = stablehlo.divide %v7257, %v7272 : tensor<64x192x1x1xf32>
    %v7274 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7275 = stablehlo.multiply %v7274, %b7pWm : tensor<64x192x1x1xf32>
    %v7276 = stablehlo.add %v7275, %v7273 : tensor<64x192x1x1xf32>
    %v7277 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7278 = stablehlo.multiply %v7277, %v7276 : tensor<64x192x1x1xf32>
    %v7279 = stablehlo.subtract %b7pW, %v7278 : tensor<64x192x1x1xf32>
    %v7280 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7281 = stablehlo.multiply %v7280, %b7pg : tensor<64xf32>
    %v7282 = stablehlo.add %v7281, %v3729 : tensor<64xf32>
    %v7283 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7284 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7285 = stablehlo.multiply %v7283, %b7pgv : tensor<64xf32>
    %v7286 = stablehlo.multiply %v7282, %v7282 : tensor<64xf32>
    %v7287 = stablehlo.multiply %v7284, %v7286 : tensor<64xf32>
    %v7288 = stablehlo.add %v7285, %v7287 : tensor<64xf32>
    %v7289 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7290 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7291 = stablehlo.multiply %v7289, %b7pgv : tensor<64xf32>
    %v7292 = stablehlo.multiply %v7282, %v7282 : tensor<64xf32>
    %v7293 = stablehlo.multiply %v7290, %v7292 : tensor<64xf32>
    %v7294 = stablehlo.add %v7291, %v7293 : tensor<64xf32>
    %v7295 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7296 = stablehlo.add %v7294, %v7295 : tensor<64xf32>
    %v7297 = stablehlo.sqrt %v7296 : tensor<64xf32>
    %v7298 = stablehlo.divide %v7282, %v7297 : tensor<64xf32>
    %v7299 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7300 = stablehlo.multiply %v7299, %b7pgm : tensor<64xf32>
    %v7301 = stablehlo.add %v7300, %v7298 : tensor<64xf32>
    %v7302 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7303 = stablehlo.multiply %v7302, %v7301 : tensor<64xf32>
    %v7304 = stablehlo.subtract %b7pg, %v7303 : tensor<64xf32>
    %v7305 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7306 = stablehlo.multiply %v7305, %b7pbt : tensor<64xf32>
    %v7307 = stablehlo.add %v7306, %v3732 : tensor<64xf32>
    %v7308 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7309 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7310 = stablehlo.multiply %v7308, %b7pbtv : tensor<64xf32>
    %v7311 = stablehlo.multiply %v7307, %v7307 : tensor<64xf32>
    %v7312 = stablehlo.multiply %v7309, %v7311 : tensor<64xf32>
    %v7313 = stablehlo.add %v7310, %v7312 : tensor<64xf32>
    %v7314 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7315 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7316 = stablehlo.multiply %v7314, %b7pbtv : tensor<64xf32>
    %v7317 = stablehlo.multiply %v7307, %v7307 : tensor<64xf32>
    %v7318 = stablehlo.multiply %v7315, %v7317 : tensor<64xf32>
    %v7319 = stablehlo.add %v7316, %v7318 : tensor<64xf32>
    %v7320 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7321 = stablehlo.add %v7319, %v7320 : tensor<64xf32>
    %v7322 = stablehlo.sqrt %v7321 : tensor<64xf32>
    %v7323 = stablehlo.divide %v7307, %v7322 : tensor<64xf32>
    %v7324 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7325 = stablehlo.multiply %v7324, %b7pbtm : tensor<64xf32>
    %v7326 = stablehlo.add %v7325, %v7323 : tensor<64xf32>
    %v7327 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7328 = stablehlo.multiply %v7327, %v7326 : tensor<64xf32>
    %v7329 = stablehlo.subtract %b7pbt, %v7328 : tensor<64xf32>
    %v7330 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7331 = stablehlo.multiply %v7330, %b8eW : tensor<384x64x1x1xf32>
    %v7332 = stablehlo.add %v7331, %v3456 : tensor<384x64x1x1xf32>
    %v7333 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7334 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7335 = stablehlo.multiply %v7333, %b8eWv : tensor<384x64x1x1xf32>
    %v7336 = stablehlo.multiply %v7332, %v7332 : tensor<384x64x1x1xf32>
    %v7337 = stablehlo.multiply %v7334, %v7336 : tensor<384x64x1x1xf32>
    %v7338 = stablehlo.add %v7335, %v7337 : tensor<384x64x1x1xf32>
    %v7339 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7340 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7341 = stablehlo.multiply %v7339, %b8eWv : tensor<384x64x1x1xf32>
    %v7342 = stablehlo.multiply %v7332, %v7332 : tensor<384x64x1x1xf32>
    %v7343 = stablehlo.multiply %v7340, %v7342 : tensor<384x64x1x1xf32>
    %v7344 = stablehlo.add %v7341, %v7343 : tensor<384x64x1x1xf32>
    %v7345 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7346 = stablehlo.add %v7344, %v7345 : tensor<384x64x1x1xf32>
    %v7347 = stablehlo.sqrt %v7346 : tensor<384x64x1x1xf32>
    %v7348 = stablehlo.divide %v7332, %v7347 : tensor<384x64x1x1xf32>
    %v7349 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7350 = stablehlo.multiply %v7349, %b8eWm : tensor<384x64x1x1xf32>
    %v7351 = stablehlo.add %v7350, %v7348 : tensor<384x64x1x1xf32>
    %v7352 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7353 = stablehlo.multiply %v7352, %v7351 : tensor<384x64x1x1xf32>
    %v7354 = stablehlo.subtract %b8eW, %v7353 : tensor<384x64x1x1xf32>
    %v7355 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7356 = stablehlo.multiply %v7355, %b8eg : tensor<384xf32>
    %v7357 = stablehlo.add %v7356, %v3474 : tensor<384xf32>
    %v7358 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7359 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7360 = stablehlo.multiply %v7358, %b8egv : tensor<384xf32>
    %v7361 = stablehlo.multiply %v7357, %v7357 : tensor<384xf32>
    %v7362 = stablehlo.multiply %v7359, %v7361 : tensor<384xf32>
    %v7363 = stablehlo.add %v7360, %v7362 : tensor<384xf32>
    %v7364 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7365 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7366 = stablehlo.multiply %v7364, %b8egv : tensor<384xf32>
    %v7367 = stablehlo.multiply %v7357, %v7357 : tensor<384xf32>
    %v7368 = stablehlo.multiply %v7365, %v7367 : tensor<384xf32>
    %v7369 = stablehlo.add %v7366, %v7368 : tensor<384xf32>
    %v7370 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7371 = stablehlo.add %v7369, %v7370 : tensor<384xf32>
    %v7372 = stablehlo.sqrt %v7371 : tensor<384xf32>
    %v7373 = stablehlo.divide %v7357, %v7372 : tensor<384xf32>
    %v7374 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7375 = stablehlo.multiply %v7374, %b8egm : tensor<384xf32>
    %v7376 = stablehlo.add %v7375, %v7373 : tensor<384xf32>
    %v7377 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7378 = stablehlo.multiply %v7377, %v7376 : tensor<384xf32>
    %v7379 = stablehlo.subtract %b8eg, %v7378 : tensor<384xf32>
    %v7380 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7381 = stablehlo.multiply %v7380, %b8ebt : tensor<384xf32>
    %v7382 = stablehlo.add %v7381, %v3477 : tensor<384xf32>
    %v7383 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7384 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7385 = stablehlo.multiply %v7383, %b8ebtv : tensor<384xf32>
    %v7386 = stablehlo.multiply %v7382, %v7382 : tensor<384xf32>
    %v7387 = stablehlo.multiply %v7384, %v7386 : tensor<384xf32>
    %v7388 = stablehlo.add %v7385, %v7387 : tensor<384xf32>
    %v7389 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7390 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7391 = stablehlo.multiply %v7389, %b8ebtv : tensor<384xf32>
    %v7392 = stablehlo.multiply %v7382, %v7382 : tensor<384xf32>
    %v7393 = stablehlo.multiply %v7390, %v7392 : tensor<384xf32>
    %v7394 = stablehlo.add %v7391, %v7393 : tensor<384xf32>
    %v7395 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7396 = stablehlo.add %v7394, %v7395 : tensor<384xf32>
    %v7397 = stablehlo.sqrt %v7396 : tensor<384xf32>
    %v7398 = stablehlo.divide %v7382, %v7397 : tensor<384xf32>
    %v7399 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7400 = stablehlo.multiply %v7399, %b8ebtm : tensor<384xf32>
    %v7401 = stablehlo.add %v7400, %v7398 : tensor<384xf32>
    %v7402 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7403 = stablehlo.multiply %v7402, %v7401 : tensor<384xf32>
    %v7404 = stablehlo.subtract %b8ebt, %v7403 : tensor<384xf32>
    %v7405 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7406 = stablehlo.multiply %v7405, %b8dW : tensor<384x1x3x3xf32>
    %v7407 = stablehlo.add %v7406, %v3483 : tensor<384x1x3x3xf32>
    %v7408 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7409 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7410 = stablehlo.multiply %v7408, %b8dWv : tensor<384x1x3x3xf32>
    %v7411 = stablehlo.multiply %v7407, %v7407 : tensor<384x1x3x3xf32>
    %v7412 = stablehlo.multiply %v7409, %v7411 : tensor<384x1x3x3xf32>
    %v7413 = stablehlo.add %v7410, %v7412 : tensor<384x1x3x3xf32>
    %v7414 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7415 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7416 = stablehlo.multiply %v7414, %b8dWv : tensor<384x1x3x3xf32>
    %v7417 = stablehlo.multiply %v7407, %v7407 : tensor<384x1x3x3xf32>
    %v7418 = stablehlo.multiply %v7415, %v7417 : tensor<384x1x3x3xf32>
    %v7419 = stablehlo.add %v7416, %v7418 : tensor<384x1x3x3xf32>
    %v7420 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7421 = stablehlo.add %v7419, %v7420 : tensor<384x1x3x3xf32>
    %v7422 = stablehlo.sqrt %v7421 : tensor<384x1x3x3xf32>
    %v7423 = stablehlo.divide %v7407, %v7422 : tensor<384x1x3x3xf32>
    %v7424 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7425 = stablehlo.multiply %v7424, %b8dWm : tensor<384x1x3x3xf32>
    %v7426 = stablehlo.add %v7425, %v7423 : tensor<384x1x3x3xf32>
    %v7427 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7428 = stablehlo.multiply %v7427, %v7426 : tensor<384x1x3x3xf32>
    %v7429 = stablehlo.subtract %b8dW, %v7428 : tensor<384x1x3x3xf32>
    %v7430 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7431 = stablehlo.multiply %v7430, %b8dg : tensor<384xf32>
    %v7432 = stablehlo.add %v7431, %v3501 : tensor<384xf32>
    %v7433 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7434 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7435 = stablehlo.multiply %v7433, %b8dgv : tensor<384xf32>
    %v7436 = stablehlo.multiply %v7432, %v7432 : tensor<384xf32>
    %v7437 = stablehlo.multiply %v7434, %v7436 : tensor<384xf32>
    %v7438 = stablehlo.add %v7435, %v7437 : tensor<384xf32>
    %v7439 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7440 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7441 = stablehlo.multiply %v7439, %b8dgv : tensor<384xf32>
    %v7442 = stablehlo.multiply %v7432, %v7432 : tensor<384xf32>
    %v7443 = stablehlo.multiply %v7440, %v7442 : tensor<384xf32>
    %v7444 = stablehlo.add %v7441, %v7443 : tensor<384xf32>
    %v7445 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7446 = stablehlo.add %v7444, %v7445 : tensor<384xf32>
    %v7447 = stablehlo.sqrt %v7446 : tensor<384xf32>
    %v7448 = stablehlo.divide %v7432, %v7447 : tensor<384xf32>
    %v7449 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7450 = stablehlo.multiply %v7449, %b8dgm : tensor<384xf32>
    %v7451 = stablehlo.add %v7450, %v7448 : tensor<384xf32>
    %v7452 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7453 = stablehlo.multiply %v7452, %v7451 : tensor<384xf32>
    %v7454 = stablehlo.subtract %b8dg, %v7453 : tensor<384xf32>
    %v7455 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7456 = stablehlo.multiply %v7455, %b8dbt : tensor<384xf32>
    %v7457 = stablehlo.add %v7456, %v3504 : tensor<384xf32>
    %v7458 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7459 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7460 = stablehlo.multiply %v7458, %b8dbtv : tensor<384xf32>
    %v7461 = stablehlo.multiply %v7457, %v7457 : tensor<384xf32>
    %v7462 = stablehlo.multiply %v7459, %v7461 : tensor<384xf32>
    %v7463 = stablehlo.add %v7460, %v7462 : tensor<384xf32>
    %v7464 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7465 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7466 = stablehlo.multiply %v7464, %b8dbtv : tensor<384xf32>
    %v7467 = stablehlo.multiply %v7457, %v7457 : tensor<384xf32>
    %v7468 = stablehlo.multiply %v7465, %v7467 : tensor<384xf32>
    %v7469 = stablehlo.add %v7466, %v7468 : tensor<384xf32>
    %v7470 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7471 = stablehlo.add %v7469, %v7470 : tensor<384xf32>
    %v7472 = stablehlo.sqrt %v7471 : tensor<384xf32>
    %v7473 = stablehlo.divide %v7457, %v7472 : tensor<384xf32>
    %v7474 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7475 = stablehlo.multiply %v7474, %b8dbtm : tensor<384xf32>
    %v7476 = stablehlo.add %v7475, %v7473 : tensor<384xf32>
    %v7477 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7478 = stablehlo.multiply %v7477, %v7476 : tensor<384xf32>
    %v7479 = stablehlo.subtract %b8dbt, %v7478 : tensor<384xf32>
    %v7480 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7481 = stablehlo.multiply %v7480, %b8pW : tensor<64x384x1x1xf32>
    %v7482 = stablehlo.add %v7481, %v3510 : tensor<64x384x1x1xf32>
    %v7483 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7484 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7485 = stablehlo.multiply %v7483, %b8pWv : tensor<64x384x1x1xf32>
    %v7486 = stablehlo.multiply %v7482, %v7482 : tensor<64x384x1x1xf32>
    %v7487 = stablehlo.multiply %v7484, %v7486 : tensor<64x384x1x1xf32>
    %v7488 = stablehlo.add %v7485, %v7487 : tensor<64x384x1x1xf32>
    %v7489 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7490 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7491 = stablehlo.multiply %v7489, %b8pWv : tensor<64x384x1x1xf32>
    %v7492 = stablehlo.multiply %v7482, %v7482 : tensor<64x384x1x1xf32>
    %v7493 = stablehlo.multiply %v7490, %v7492 : tensor<64x384x1x1xf32>
    %v7494 = stablehlo.add %v7491, %v7493 : tensor<64x384x1x1xf32>
    %v7495 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7496 = stablehlo.add %v7494, %v7495 : tensor<64x384x1x1xf32>
    %v7497 = stablehlo.sqrt %v7496 : tensor<64x384x1x1xf32>
    %v7498 = stablehlo.divide %v7482, %v7497 : tensor<64x384x1x1xf32>
    %v7499 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7500 = stablehlo.multiply %v7499, %b8pWm : tensor<64x384x1x1xf32>
    %v7501 = stablehlo.add %v7500, %v7498 : tensor<64x384x1x1xf32>
    %v7502 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7503 = stablehlo.multiply %v7502, %v7501 : tensor<64x384x1x1xf32>
    %v7504 = stablehlo.subtract %b8pW, %v7503 : tensor<64x384x1x1xf32>
    %v7505 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7506 = stablehlo.multiply %v7505, %b8pg : tensor<64xf32>
    %v7507 = stablehlo.add %v7506, %v3528 : tensor<64xf32>
    %v7508 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7509 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7510 = stablehlo.multiply %v7508, %b8pgv : tensor<64xf32>
    %v7511 = stablehlo.multiply %v7507, %v7507 : tensor<64xf32>
    %v7512 = stablehlo.multiply %v7509, %v7511 : tensor<64xf32>
    %v7513 = stablehlo.add %v7510, %v7512 : tensor<64xf32>
    %v7514 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7515 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7516 = stablehlo.multiply %v7514, %b8pgv : tensor<64xf32>
    %v7517 = stablehlo.multiply %v7507, %v7507 : tensor<64xf32>
    %v7518 = stablehlo.multiply %v7515, %v7517 : tensor<64xf32>
    %v7519 = stablehlo.add %v7516, %v7518 : tensor<64xf32>
    %v7520 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7521 = stablehlo.add %v7519, %v7520 : tensor<64xf32>
    %v7522 = stablehlo.sqrt %v7521 : tensor<64xf32>
    %v7523 = stablehlo.divide %v7507, %v7522 : tensor<64xf32>
    %v7524 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7525 = stablehlo.multiply %v7524, %b8pgm : tensor<64xf32>
    %v7526 = stablehlo.add %v7525, %v7523 : tensor<64xf32>
    %v7527 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7528 = stablehlo.multiply %v7527, %v7526 : tensor<64xf32>
    %v7529 = stablehlo.subtract %b8pg, %v7528 : tensor<64xf32>
    %v7530 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7531 = stablehlo.multiply %v7530, %b8pbt : tensor<64xf32>
    %v7532 = stablehlo.add %v7531, %v3531 : tensor<64xf32>
    %v7533 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7534 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7535 = stablehlo.multiply %v7533, %b8pbtv : tensor<64xf32>
    %v7536 = stablehlo.multiply %v7532, %v7532 : tensor<64xf32>
    %v7537 = stablehlo.multiply %v7534, %v7536 : tensor<64xf32>
    %v7538 = stablehlo.add %v7535, %v7537 : tensor<64xf32>
    %v7539 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7540 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7541 = stablehlo.multiply %v7539, %b8pbtv : tensor<64xf32>
    %v7542 = stablehlo.multiply %v7532, %v7532 : tensor<64xf32>
    %v7543 = stablehlo.multiply %v7540, %v7542 : tensor<64xf32>
    %v7544 = stablehlo.add %v7541, %v7543 : tensor<64xf32>
    %v7545 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7546 = stablehlo.add %v7544, %v7545 : tensor<64xf32>
    %v7547 = stablehlo.sqrt %v7546 : tensor<64xf32>
    %v7548 = stablehlo.divide %v7532, %v7547 : tensor<64xf32>
    %v7549 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7550 = stablehlo.multiply %v7549, %b8pbtm : tensor<64xf32>
    %v7551 = stablehlo.add %v7550, %v7548 : tensor<64xf32>
    %v7552 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7553 = stablehlo.multiply %v7552, %v7551 : tensor<64xf32>
    %v7554 = stablehlo.subtract %b8pbt, %v7553 : tensor<64xf32>
    %v7555 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7556 = stablehlo.multiply %v7555, %b9eW : tensor<384x64x1x1xf32>
    %v7557 = stablehlo.add %v7556, %v3258 : tensor<384x64x1x1xf32>
    %v7558 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7559 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7560 = stablehlo.multiply %v7558, %b9eWv : tensor<384x64x1x1xf32>
    %v7561 = stablehlo.multiply %v7557, %v7557 : tensor<384x64x1x1xf32>
    %v7562 = stablehlo.multiply %v7559, %v7561 : tensor<384x64x1x1xf32>
    %v7563 = stablehlo.add %v7560, %v7562 : tensor<384x64x1x1xf32>
    %v7564 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7565 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7566 = stablehlo.multiply %v7564, %b9eWv : tensor<384x64x1x1xf32>
    %v7567 = stablehlo.multiply %v7557, %v7557 : tensor<384x64x1x1xf32>
    %v7568 = stablehlo.multiply %v7565, %v7567 : tensor<384x64x1x1xf32>
    %v7569 = stablehlo.add %v7566, %v7568 : tensor<384x64x1x1xf32>
    %v7570 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7571 = stablehlo.add %v7569, %v7570 : tensor<384x64x1x1xf32>
    %v7572 = stablehlo.sqrt %v7571 : tensor<384x64x1x1xf32>
    %v7573 = stablehlo.divide %v7557, %v7572 : tensor<384x64x1x1xf32>
    %v7574 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7575 = stablehlo.multiply %v7574, %b9eWm : tensor<384x64x1x1xf32>
    %v7576 = stablehlo.add %v7575, %v7573 : tensor<384x64x1x1xf32>
    %v7577 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7578 = stablehlo.multiply %v7577, %v7576 : tensor<384x64x1x1xf32>
    %v7579 = stablehlo.subtract %b9eW, %v7578 : tensor<384x64x1x1xf32>
    %v7580 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7581 = stablehlo.multiply %v7580, %b9eg : tensor<384xf32>
    %v7582 = stablehlo.add %v7581, %v3276 : tensor<384xf32>
    %v7583 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7584 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7585 = stablehlo.multiply %v7583, %b9egv : tensor<384xf32>
    %v7586 = stablehlo.multiply %v7582, %v7582 : tensor<384xf32>
    %v7587 = stablehlo.multiply %v7584, %v7586 : tensor<384xf32>
    %v7588 = stablehlo.add %v7585, %v7587 : tensor<384xf32>
    %v7589 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7590 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7591 = stablehlo.multiply %v7589, %b9egv : tensor<384xf32>
    %v7592 = stablehlo.multiply %v7582, %v7582 : tensor<384xf32>
    %v7593 = stablehlo.multiply %v7590, %v7592 : tensor<384xf32>
    %v7594 = stablehlo.add %v7591, %v7593 : tensor<384xf32>
    %v7595 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7596 = stablehlo.add %v7594, %v7595 : tensor<384xf32>
    %v7597 = stablehlo.sqrt %v7596 : tensor<384xf32>
    %v7598 = stablehlo.divide %v7582, %v7597 : tensor<384xf32>
    %v7599 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7600 = stablehlo.multiply %v7599, %b9egm : tensor<384xf32>
    %v7601 = stablehlo.add %v7600, %v7598 : tensor<384xf32>
    %v7602 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7603 = stablehlo.multiply %v7602, %v7601 : tensor<384xf32>
    %v7604 = stablehlo.subtract %b9eg, %v7603 : tensor<384xf32>
    %v7605 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7606 = stablehlo.multiply %v7605, %b9ebt : tensor<384xf32>
    %v7607 = stablehlo.add %v7606, %v3279 : tensor<384xf32>
    %v7608 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7609 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7610 = stablehlo.multiply %v7608, %b9ebtv : tensor<384xf32>
    %v7611 = stablehlo.multiply %v7607, %v7607 : tensor<384xf32>
    %v7612 = stablehlo.multiply %v7609, %v7611 : tensor<384xf32>
    %v7613 = stablehlo.add %v7610, %v7612 : tensor<384xf32>
    %v7614 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7615 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7616 = stablehlo.multiply %v7614, %b9ebtv : tensor<384xf32>
    %v7617 = stablehlo.multiply %v7607, %v7607 : tensor<384xf32>
    %v7618 = stablehlo.multiply %v7615, %v7617 : tensor<384xf32>
    %v7619 = stablehlo.add %v7616, %v7618 : tensor<384xf32>
    %v7620 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7621 = stablehlo.add %v7619, %v7620 : tensor<384xf32>
    %v7622 = stablehlo.sqrt %v7621 : tensor<384xf32>
    %v7623 = stablehlo.divide %v7607, %v7622 : tensor<384xf32>
    %v7624 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7625 = stablehlo.multiply %v7624, %b9ebtm : tensor<384xf32>
    %v7626 = stablehlo.add %v7625, %v7623 : tensor<384xf32>
    %v7627 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7628 = stablehlo.multiply %v7627, %v7626 : tensor<384xf32>
    %v7629 = stablehlo.subtract %b9ebt, %v7628 : tensor<384xf32>
    %v7630 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7631 = stablehlo.multiply %v7630, %b9dW : tensor<384x1x3x3xf32>
    %v7632 = stablehlo.add %v7631, %v3285 : tensor<384x1x3x3xf32>
    %v7633 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7634 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7635 = stablehlo.multiply %v7633, %b9dWv : tensor<384x1x3x3xf32>
    %v7636 = stablehlo.multiply %v7632, %v7632 : tensor<384x1x3x3xf32>
    %v7637 = stablehlo.multiply %v7634, %v7636 : tensor<384x1x3x3xf32>
    %v7638 = stablehlo.add %v7635, %v7637 : tensor<384x1x3x3xf32>
    %v7639 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7640 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7641 = stablehlo.multiply %v7639, %b9dWv : tensor<384x1x3x3xf32>
    %v7642 = stablehlo.multiply %v7632, %v7632 : tensor<384x1x3x3xf32>
    %v7643 = stablehlo.multiply %v7640, %v7642 : tensor<384x1x3x3xf32>
    %v7644 = stablehlo.add %v7641, %v7643 : tensor<384x1x3x3xf32>
    %v7645 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7646 = stablehlo.add %v7644, %v7645 : tensor<384x1x3x3xf32>
    %v7647 = stablehlo.sqrt %v7646 : tensor<384x1x3x3xf32>
    %v7648 = stablehlo.divide %v7632, %v7647 : tensor<384x1x3x3xf32>
    %v7649 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7650 = stablehlo.multiply %v7649, %b9dWm : tensor<384x1x3x3xf32>
    %v7651 = stablehlo.add %v7650, %v7648 : tensor<384x1x3x3xf32>
    %v7652 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7653 = stablehlo.multiply %v7652, %v7651 : tensor<384x1x3x3xf32>
    %v7654 = stablehlo.subtract %b9dW, %v7653 : tensor<384x1x3x3xf32>
    %v7655 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7656 = stablehlo.multiply %v7655, %b9dg : tensor<384xf32>
    %v7657 = stablehlo.add %v7656, %v3303 : tensor<384xf32>
    %v7658 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7659 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7660 = stablehlo.multiply %v7658, %b9dgv : tensor<384xf32>
    %v7661 = stablehlo.multiply %v7657, %v7657 : tensor<384xf32>
    %v7662 = stablehlo.multiply %v7659, %v7661 : tensor<384xf32>
    %v7663 = stablehlo.add %v7660, %v7662 : tensor<384xf32>
    %v7664 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7665 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7666 = stablehlo.multiply %v7664, %b9dgv : tensor<384xf32>
    %v7667 = stablehlo.multiply %v7657, %v7657 : tensor<384xf32>
    %v7668 = stablehlo.multiply %v7665, %v7667 : tensor<384xf32>
    %v7669 = stablehlo.add %v7666, %v7668 : tensor<384xf32>
    %v7670 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7671 = stablehlo.add %v7669, %v7670 : tensor<384xf32>
    %v7672 = stablehlo.sqrt %v7671 : tensor<384xf32>
    %v7673 = stablehlo.divide %v7657, %v7672 : tensor<384xf32>
    %v7674 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7675 = stablehlo.multiply %v7674, %b9dgm : tensor<384xf32>
    %v7676 = stablehlo.add %v7675, %v7673 : tensor<384xf32>
    %v7677 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7678 = stablehlo.multiply %v7677, %v7676 : tensor<384xf32>
    %v7679 = stablehlo.subtract %b9dg, %v7678 : tensor<384xf32>
    %v7680 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7681 = stablehlo.multiply %v7680, %b9dbt : tensor<384xf32>
    %v7682 = stablehlo.add %v7681, %v3306 : tensor<384xf32>
    %v7683 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7684 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7685 = stablehlo.multiply %v7683, %b9dbtv : tensor<384xf32>
    %v7686 = stablehlo.multiply %v7682, %v7682 : tensor<384xf32>
    %v7687 = stablehlo.multiply %v7684, %v7686 : tensor<384xf32>
    %v7688 = stablehlo.add %v7685, %v7687 : tensor<384xf32>
    %v7689 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7690 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7691 = stablehlo.multiply %v7689, %b9dbtv : tensor<384xf32>
    %v7692 = stablehlo.multiply %v7682, %v7682 : tensor<384xf32>
    %v7693 = stablehlo.multiply %v7690, %v7692 : tensor<384xf32>
    %v7694 = stablehlo.add %v7691, %v7693 : tensor<384xf32>
    %v7695 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7696 = stablehlo.add %v7694, %v7695 : tensor<384xf32>
    %v7697 = stablehlo.sqrt %v7696 : tensor<384xf32>
    %v7698 = stablehlo.divide %v7682, %v7697 : tensor<384xf32>
    %v7699 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7700 = stablehlo.multiply %v7699, %b9dbtm : tensor<384xf32>
    %v7701 = stablehlo.add %v7700, %v7698 : tensor<384xf32>
    %v7702 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7703 = stablehlo.multiply %v7702, %v7701 : tensor<384xf32>
    %v7704 = stablehlo.subtract %b9dbt, %v7703 : tensor<384xf32>
    %v7705 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7706 = stablehlo.multiply %v7705, %b9pW : tensor<64x384x1x1xf32>
    %v7707 = stablehlo.add %v7706, %v3312 : tensor<64x384x1x1xf32>
    %v7708 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7709 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7710 = stablehlo.multiply %v7708, %b9pWv : tensor<64x384x1x1xf32>
    %v7711 = stablehlo.multiply %v7707, %v7707 : tensor<64x384x1x1xf32>
    %v7712 = stablehlo.multiply %v7709, %v7711 : tensor<64x384x1x1xf32>
    %v7713 = stablehlo.add %v7710, %v7712 : tensor<64x384x1x1xf32>
    %v7714 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7715 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7716 = stablehlo.multiply %v7714, %b9pWv : tensor<64x384x1x1xf32>
    %v7717 = stablehlo.multiply %v7707, %v7707 : tensor<64x384x1x1xf32>
    %v7718 = stablehlo.multiply %v7715, %v7717 : tensor<64x384x1x1xf32>
    %v7719 = stablehlo.add %v7716, %v7718 : tensor<64x384x1x1xf32>
    %v7720 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7721 = stablehlo.add %v7719, %v7720 : tensor<64x384x1x1xf32>
    %v7722 = stablehlo.sqrt %v7721 : tensor<64x384x1x1xf32>
    %v7723 = stablehlo.divide %v7707, %v7722 : tensor<64x384x1x1xf32>
    %v7724 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7725 = stablehlo.multiply %v7724, %b9pWm : tensor<64x384x1x1xf32>
    %v7726 = stablehlo.add %v7725, %v7723 : tensor<64x384x1x1xf32>
    %v7727 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7728 = stablehlo.multiply %v7727, %v7726 : tensor<64x384x1x1xf32>
    %v7729 = stablehlo.subtract %b9pW, %v7728 : tensor<64x384x1x1xf32>
    %v7730 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7731 = stablehlo.multiply %v7730, %b9pg : tensor<64xf32>
    %v7732 = stablehlo.add %v7731, %v3330 : tensor<64xf32>
    %v7733 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7734 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7735 = stablehlo.multiply %v7733, %b9pgv : tensor<64xf32>
    %v7736 = stablehlo.multiply %v7732, %v7732 : tensor<64xf32>
    %v7737 = stablehlo.multiply %v7734, %v7736 : tensor<64xf32>
    %v7738 = stablehlo.add %v7735, %v7737 : tensor<64xf32>
    %v7739 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7740 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7741 = stablehlo.multiply %v7739, %b9pgv : tensor<64xf32>
    %v7742 = stablehlo.multiply %v7732, %v7732 : tensor<64xf32>
    %v7743 = stablehlo.multiply %v7740, %v7742 : tensor<64xf32>
    %v7744 = stablehlo.add %v7741, %v7743 : tensor<64xf32>
    %v7745 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7746 = stablehlo.add %v7744, %v7745 : tensor<64xf32>
    %v7747 = stablehlo.sqrt %v7746 : tensor<64xf32>
    %v7748 = stablehlo.divide %v7732, %v7747 : tensor<64xf32>
    %v7749 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7750 = stablehlo.multiply %v7749, %b9pgm : tensor<64xf32>
    %v7751 = stablehlo.add %v7750, %v7748 : tensor<64xf32>
    %v7752 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7753 = stablehlo.multiply %v7752, %v7751 : tensor<64xf32>
    %v7754 = stablehlo.subtract %b9pg, %v7753 : tensor<64xf32>
    %v7755 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7756 = stablehlo.multiply %v7755, %b9pbt : tensor<64xf32>
    %v7757 = stablehlo.add %v7756, %v3333 : tensor<64xf32>
    %v7758 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7759 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7760 = stablehlo.multiply %v7758, %b9pbtv : tensor<64xf32>
    %v7761 = stablehlo.multiply %v7757, %v7757 : tensor<64xf32>
    %v7762 = stablehlo.multiply %v7759, %v7761 : tensor<64xf32>
    %v7763 = stablehlo.add %v7760, %v7762 : tensor<64xf32>
    %v7764 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7765 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7766 = stablehlo.multiply %v7764, %b9pbtv : tensor<64xf32>
    %v7767 = stablehlo.multiply %v7757, %v7757 : tensor<64xf32>
    %v7768 = stablehlo.multiply %v7765, %v7767 : tensor<64xf32>
    %v7769 = stablehlo.add %v7766, %v7768 : tensor<64xf32>
    %v7770 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7771 = stablehlo.add %v7769, %v7770 : tensor<64xf32>
    %v7772 = stablehlo.sqrt %v7771 : tensor<64xf32>
    %v7773 = stablehlo.divide %v7757, %v7772 : tensor<64xf32>
    %v7774 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7775 = stablehlo.multiply %v7774, %b9pbtm : tensor<64xf32>
    %v7776 = stablehlo.add %v7775, %v7773 : tensor<64xf32>
    %v7777 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7778 = stablehlo.multiply %v7777, %v7776 : tensor<64xf32>
    %v7779 = stablehlo.subtract %b9pbt, %v7778 : tensor<64xf32>
    %v7780 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7781 = stablehlo.multiply %v7780, %b10eW : tensor<384x64x1x1xf32>
    %v7782 = stablehlo.add %v7781, %v3060 : tensor<384x64x1x1xf32>
    %v7783 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7784 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7785 = stablehlo.multiply %v7783, %b10eWv : tensor<384x64x1x1xf32>
    %v7786 = stablehlo.multiply %v7782, %v7782 : tensor<384x64x1x1xf32>
    %v7787 = stablehlo.multiply %v7784, %v7786 : tensor<384x64x1x1xf32>
    %v7788 = stablehlo.add %v7785, %v7787 : tensor<384x64x1x1xf32>
    %v7789 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7790 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7791 = stablehlo.multiply %v7789, %b10eWv : tensor<384x64x1x1xf32>
    %v7792 = stablehlo.multiply %v7782, %v7782 : tensor<384x64x1x1xf32>
    %v7793 = stablehlo.multiply %v7790, %v7792 : tensor<384x64x1x1xf32>
    %v7794 = stablehlo.add %v7791, %v7793 : tensor<384x64x1x1xf32>
    %v7795 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7796 = stablehlo.add %v7794, %v7795 : tensor<384x64x1x1xf32>
    %v7797 = stablehlo.sqrt %v7796 : tensor<384x64x1x1xf32>
    %v7798 = stablehlo.divide %v7782, %v7797 : tensor<384x64x1x1xf32>
    %v7799 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7800 = stablehlo.multiply %v7799, %b10eWm : tensor<384x64x1x1xf32>
    %v7801 = stablehlo.add %v7800, %v7798 : tensor<384x64x1x1xf32>
    %v7802 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7803 = stablehlo.multiply %v7802, %v7801 : tensor<384x64x1x1xf32>
    %v7804 = stablehlo.subtract %b10eW, %v7803 : tensor<384x64x1x1xf32>
    %v7805 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7806 = stablehlo.multiply %v7805, %b10eg : tensor<384xf32>
    %v7807 = stablehlo.add %v7806, %v3078 : tensor<384xf32>
    %v7808 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7809 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7810 = stablehlo.multiply %v7808, %b10egv : tensor<384xf32>
    %v7811 = stablehlo.multiply %v7807, %v7807 : tensor<384xf32>
    %v7812 = stablehlo.multiply %v7809, %v7811 : tensor<384xf32>
    %v7813 = stablehlo.add %v7810, %v7812 : tensor<384xf32>
    %v7814 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7815 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7816 = stablehlo.multiply %v7814, %b10egv : tensor<384xf32>
    %v7817 = stablehlo.multiply %v7807, %v7807 : tensor<384xf32>
    %v7818 = stablehlo.multiply %v7815, %v7817 : tensor<384xf32>
    %v7819 = stablehlo.add %v7816, %v7818 : tensor<384xf32>
    %v7820 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7821 = stablehlo.add %v7819, %v7820 : tensor<384xf32>
    %v7822 = stablehlo.sqrt %v7821 : tensor<384xf32>
    %v7823 = stablehlo.divide %v7807, %v7822 : tensor<384xf32>
    %v7824 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7825 = stablehlo.multiply %v7824, %b10egm : tensor<384xf32>
    %v7826 = stablehlo.add %v7825, %v7823 : tensor<384xf32>
    %v7827 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7828 = stablehlo.multiply %v7827, %v7826 : tensor<384xf32>
    %v7829 = stablehlo.subtract %b10eg, %v7828 : tensor<384xf32>
    %v7830 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7831 = stablehlo.multiply %v7830, %b10ebt : tensor<384xf32>
    %v7832 = stablehlo.add %v7831, %v3081 : tensor<384xf32>
    %v7833 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7834 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7835 = stablehlo.multiply %v7833, %b10ebtv : tensor<384xf32>
    %v7836 = stablehlo.multiply %v7832, %v7832 : tensor<384xf32>
    %v7837 = stablehlo.multiply %v7834, %v7836 : tensor<384xf32>
    %v7838 = stablehlo.add %v7835, %v7837 : tensor<384xf32>
    %v7839 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7840 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7841 = stablehlo.multiply %v7839, %b10ebtv : tensor<384xf32>
    %v7842 = stablehlo.multiply %v7832, %v7832 : tensor<384xf32>
    %v7843 = stablehlo.multiply %v7840, %v7842 : tensor<384xf32>
    %v7844 = stablehlo.add %v7841, %v7843 : tensor<384xf32>
    %v7845 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7846 = stablehlo.add %v7844, %v7845 : tensor<384xf32>
    %v7847 = stablehlo.sqrt %v7846 : tensor<384xf32>
    %v7848 = stablehlo.divide %v7832, %v7847 : tensor<384xf32>
    %v7849 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7850 = stablehlo.multiply %v7849, %b10ebtm : tensor<384xf32>
    %v7851 = stablehlo.add %v7850, %v7848 : tensor<384xf32>
    %v7852 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7853 = stablehlo.multiply %v7852, %v7851 : tensor<384xf32>
    %v7854 = stablehlo.subtract %b10ebt, %v7853 : tensor<384xf32>
    %v7855 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7856 = stablehlo.multiply %v7855, %b10dW : tensor<384x1x3x3xf32>
    %v7857 = stablehlo.add %v7856, %v3087 : tensor<384x1x3x3xf32>
    %v7858 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7859 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7860 = stablehlo.multiply %v7858, %b10dWv : tensor<384x1x3x3xf32>
    %v7861 = stablehlo.multiply %v7857, %v7857 : tensor<384x1x3x3xf32>
    %v7862 = stablehlo.multiply %v7859, %v7861 : tensor<384x1x3x3xf32>
    %v7863 = stablehlo.add %v7860, %v7862 : tensor<384x1x3x3xf32>
    %v7864 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7865 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7866 = stablehlo.multiply %v7864, %b10dWv : tensor<384x1x3x3xf32>
    %v7867 = stablehlo.multiply %v7857, %v7857 : tensor<384x1x3x3xf32>
    %v7868 = stablehlo.multiply %v7865, %v7867 : tensor<384x1x3x3xf32>
    %v7869 = stablehlo.add %v7866, %v7868 : tensor<384x1x3x3xf32>
    %v7870 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7871 = stablehlo.add %v7869, %v7870 : tensor<384x1x3x3xf32>
    %v7872 = stablehlo.sqrt %v7871 : tensor<384x1x3x3xf32>
    %v7873 = stablehlo.divide %v7857, %v7872 : tensor<384x1x3x3xf32>
    %v7874 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7875 = stablehlo.multiply %v7874, %b10dWm : tensor<384x1x3x3xf32>
    %v7876 = stablehlo.add %v7875, %v7873 : tensor<384x1x3x3xf32>
    %v7877 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7878 = stablehlo.multiply %v7877, %v7876 : tensor<384x1x3x3xf32>
    %v7879 = stablehlo.subtract %b10dW, %v7878 : tensor<384x1x3x3xf32>
    %v7880 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7881 = stablehlo.multiply %v7880, %b10dg : tensor<384xf32>
    %v7882 = stablehlo.add %v7881, %v3105 : tensor<384xf32>
    %v7883 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7884 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7885 = stablehlo.multiply %v7883, %b10dgv : tensor<384xf32>
    %v7886 = stablehlo.multiply %v7882, %v7882 : tensor<384xf32>
    %v7887 = stablehlo.multiply %v7884, %v7886 : tensor<384xf32>
    %v7888 = stablehlo.add %v7885, %v7887 : tensor<384xf32>
    %v7889 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7890 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7891 = stablehlo.multiply %v7889, %b10dgv : tensor<384xf32>
    %v7892 = stablehlo.multiply %v7882, %v7882 : tensor<384xf32>
    %v7893 = stablehlo.multiply %v7890, %v7892 : tensor<384xf32>
    %v7894 = stablehlo.add %v7891, %v7893 : tensor<384xf32>
    %v7895 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7896 = stablehlo.add %v7894, %v7895 : tensor<384xf32>
    %v7897 = stablehlo.sqrt %v7896 : tensor<384xf32>
    %v7898 = stablehlo.divide %v7882, %v7897 : tensor<384xf32>
    %v7899 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7900 = stablehlo.multiply %v7899, %b10dgm : tensor<384xf32>
    %v7901 = stablehlo.add %v7900, %v7898 : tensor<384xf32>
    %v7902 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7903 = stablehlo.multiply %v7902, %v7901 : tensor<384xf32>
    %v7904 = stablehlo.subtract %b10dg, %v7903 : tensor<384xf32>
    %v7905 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7906 = stablehlo.multiply %v7905, %b10dbt : tensor<384xf32>
    %v7907 = stablehlo.add %v7906, %v3108 : tensor<384xf32>
    %v7908 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7909 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7910 = stablehlo.multiply %v7908, %b10dbtv : tensor<384xf32>
    %v7911 = stablehlo.multiply %v7907, %v7907 : tensor<384xf32>
    %v7912 = stablehlo.multiply %v7909, %v7911 : tensor<384xf32>
    %v7913 = stablehlo.add %v7910, %v7912 : tensor<384xf32>
    %v7914 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7915 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7916 = stablehlo.multiply %v7914, %b10dbtv : tensor<384xf32>
    %v7917 = stablehlo.multiply %v7907, %v7907 : tensor<384xf32>
    %v7918 = stablehlo.multiply %v7915, %v7917 : tensor<384xf32>
    %v7919 = stablehlo.add %v7916, %v7918 : tensor<384xf32>
    %v7920 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7921 = stablehlo.add %v7919, %v7920 : tensor<384xf32>
    %v7922 = stablehlo.sqrt %v7921 : tensor<384xf32>
    %v7923 = stablehlo.divide %v7907, %v7922 : tensor<384xf32>
    %v7924 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7925 = stablehlo.multiply %v7924, %b10dbtm : tensor<384xf32>
    %v7926 = stablehlo.add %v7925, %v7923 : tensor<384xf32>
    %v7927 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7928 = stablehlo.multiply %v7927, %v7926 : tensor<384xf32>
    %v7929 = stablehlo.subtract %b10dbt, %v7928 : tensor<384xf32>
    %v7930 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7931 = stablehlo.multiply %v7930, %b10pW : tensor<64x384x1x1xf32>
    %v7932 = stablehlo.add %v7931, %v3114 : tensor<64x384x1x1xf32>
    %v7933 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7934 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7935 = stablehlo.multiply %v7933, %b10pWv : tensor<64x384x1x1xf32>
    %v7936 = stablehlo.multiply %v7932, %v7932 : tensor<64x384x1x1xf32>
    %v7937 = stablehlo.multiply %v7934, %v7936 : tensor<64x384x1x1xf32>
    %v7938 = stablehlo.add %v7935, %v7937 : tensor<64x384x1x1xf32>
    %v7939 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7940 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7941 = stablehlo.multiply %v7939, %b10pWv : tensor<64x384x1x1xf32>
    %v7942 = stablehlo.multiply %v7932, %v7932 : tensor<64x384x1x1xf32>
    %v7943 = stablehlo.multiply %v7940, %v7942 : tensor<64x384x1x1xf32>
    %v7944 = stablehlo.add %v7941, %v7943 : tensor<64x384x1x1xf32>
    %v7945 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7946 = stablehlo.add %v7944, %v7945 : tensor<64x384x1x1xf32>
    %v7947 = stablehlo.sqrt %v7946 : tensor<64x384x1x1xf32>
    %v7948 = stablehlo.divide %v7932, %v7947 : tensor<64x384x1x1xf32>
    %v7949 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7950 = stablehlo.multiply %v7949, %b10pWm : tensor<64x384x1x1xf32>
    %v7951 = stablehlo.add %v7950, %v7948 : tensor<64x384x1x1xf32>
    %v7952 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7953 = stablehlo.multiply %v7952, %v7951 : tensor<64x384x1x1xf32>
    %v7954 = stablehlo.subtract %b10pW, %v7953 : tensor<64x384x1x1xf32>
    %v7955 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7956 = stablehlo.multiply %v7955, %b10pg : tensor<64xf32>
    %v7957 = stablehlo.add %v7956, %v3132 : tensor<64xf32>
    %v7958 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7959 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7960 = stablehlo.multiply %v7958, %b10pgv : tensor<64xf32>
    %v7961 = stablehlo.multiply %v7957, %v7957 : tensor<64xf32>
    %v7962 = stablehlo.multiply %v7959, %v7961 : tensor<64xf32>
    %v7963 = stablehlo.add %v7960, %v7962 : tensor<64xf32>
    %v7964 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7965 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7966 = stablehlo.multiply %v7964, %b10pgv : tensor<64xf32>
    %v7967 = stablehlo.multiply %v7957, %v7957 : tensor<64xf32>
    %v7968 = stablehlo.multiply %v7965, %v7967 : tensor<64xf32>
    %v7969 = stablehlo.add %v7966, %v7968 : tensor<64xf32>
    %v7970 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7971 = stablehlo.add %v7969, %v7970 : tensor<64xf32>
    %v7972 = stablehlo.sqrt %v7971 : tensor<64xf32>
    %v7973 = stablehlo.divide %v7957, %v7972 : tensor<64xf32>
    %v7974 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7975 = stablehlo.multiply %v7974, %b10pgm : tensor<64xf32>
    %v7976 = stablehlo.add %v7975, %v7973 : tensor<64xf32>
    %v7977 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7978 = stablehlo.multiply %v7977, %v7976 : tensor<64xf32>
    %v7979 = stablehlo.subtract %b10pg, %v7978 : tensor<64xf32>
    %v7980 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7981 = stablehlo.multiply %v7980, %b10pbt : tensor<64xf32>
    %v7982 = stablehlo.add %v7981, %v3135 : tensor<64xf32>
    %v7983 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7984 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7985 = stablehlo.multiply %v7983, %b10pbtv : tensor<64xf32>
    %v7986 = stablehlo.multiply %v7982, %v7982 : tensor<64xf32>
    %v7987 = stablehlo.multiply %v7984, %v7986 : tensor<64xf32>
    %v7988 = stablehlo.add %v7985, %v7987 : tensor<64xf32>
    %v7989 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7990 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7991 = stablehlo.multiply %v7989, %b10pbtv : tensor<64xf32>
    %v7992 = stablehlo.multiply %v7982, %v7982 : tensor<64xf32>
    %v7993 = stablehlo.multiply %v7990, %v7992 : tensor<64xf32>
    %v7994 = stablehlo.add %v7991, %v7993 : tensor<64xf32>
    %v7995 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7996 = stablehlo.add %v7994, %v7995 : tensor<64xf32>
    %v7997 = stablehlo.sqrt %v7996 : tensor<64xf32>
    %v7998 = stablehlo.divide %v7982, %v7997 : tensor<64xf32>
    %v7999 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8000 = stablehlo.multiply %v7999, %b10pbtm : tensor<64xf32>
    %v8001 = stablehlo.add %v8000, %v7998 : tensor<64xf32>
    %v8002 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8003 = stablehlo.multiply %v8002, %v8001 : tensor<64xf32>
    %v8004 = stablehlo.subtract %b10pbt, %v8003 : tensor<64xf32>
    %v8005 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8006 = stablehlo.multiply %v8005, %b11eW : tensor<384x64x1x1xf32>
    %v8007 = stablehlo.add %v8006, %v2862 : tensor<384x64x1x1xf32>
    %v8008 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8009 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8010 = stablehlo.multiply %v8008, %b11eWv : tensor<384x64x1x1xf32>
    %v8011 = stablehlo.multiply %v8007, %v8007 : tensor<384x64x1x1xf32>
    %v8012 = stablehlo.multiply %v8009, %v8011 : tensor<384x64x1x1xf32>
    %v8013 = stablehlo.add %v8010, %v8012 : tensor<384x64x1x1xf32>
    %v8014 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8015 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8016 = stablehlo.multiply %v8014, %b11eWv : tensor<384x64x1x1xf32>
    %v8017 = stablehlo.multiply %v8007, %v8007 : tensor<384x64x1x1xf32>
    %v8018 = stablehlo.multiply %v8015, %v8017 : tensor<384x64x1x1xf32>
    %v8019 = stablehlo.add %v8016, %v8018 : tensor<384x64x1x1xf32>
    %v8020 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8021 = stablehlo.add %v8019, %v8020 : tensor<384x64x1x1xf32>
    %v8022 = stablehlo.sqrt %v8021 : tensor<384x64x1x1xf32>
    %v8023 = stablehlo.divide %v8007, %v8022 : tensor<384x64x1x1xf32>
    %v8024 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8025 = stablehlo.multiply %v8024, %b11eWm : tensor<384x64x1x1xf32>
    %v8026 = stablehlo.add %v8025, %v8023 : tensor<384x64x1x1xf32>
    %v8027 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8028 = stablehlo.multiply %v8027, %v8026 : tensor<384x64x1x1xf32>
    %v8029 = stablehlo.subtract %b11eW, %v8028 : tensor<384x64x1x1xf32>
    %v8030 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8031 = stablehlo.multiply %v8030, %b11eg : tensor<384xf32>
    %v8032 = stablehlo.add %v8031, %v2880 : tensor<384xf32>
    %v8033 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8034 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8035 = stablehlo.multiply %v8033, %b11egv : tensor<384xf32>
    %v8036 = stablehlo.multiply %v8032, %v8032 : tensor<384xf32>
    %v8037 = stablehlo.multiply %v8034, %v8036 : tensor<384xf32>
    %v8038 = stablehlo.add %v8035, %v8037 : tensor<384xf32>
    %v8039 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8040 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8041 = stablehlo.multiply %v8039, %b11egv : tensor<384xf32>
    %v8042 = stablehlo.multiply %v8032, %v8032 : tensor<384xf32>
    %v8043 = stablehlo.multiply %v8040, %v8042 : tensor<384xf32>
    %v8044 = stablehlo.add %v8041, %v8043 : tensor<384xf32>
    %v8045 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8046 = stablehlo.add %v8044, %v8045 : tensor<384xf32>
    %v8047 = stablehlo.sqrt %v8046 : tensor<384xf32>
    %v8048 = stablehlo.divide %v8032, %v8047 : tensor<384xf32>
    %v8049 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8050 = stablehlo.multiply %v8049, %b11egm : tensor<384xf32>
    %v8051 = stablehlo.add %v8050, %v8048 : tensor<384xf32>
    %v8052 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8053 = stablehlo.multiply %v8052, %v8051 : tensor<384xf32>
    %v8054 = stablehlo.subtract %b11eg, %v8053 : tensor<384xf32>
    %v8055 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8056 = stablehlo.multiply %v8055, %b11ebt : tensor<384xf32>
    %v8057 = stablehlo.add %v8056, %v2883 : tensor<384xf32>
    %v8058 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8059 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8060 = stablehlo.multiply %v8058, %b11ebtv : tensor<384xf32>
    %v8061 = stablehlo.multiply %v8057, %v8057 : tensor<384xf32>
    %v8062 = stablehlo.multiply %v8059, %v8061 : tensor<384xf32>
    %v8063 = stablehlo.add %v8060, %v8062 : tensor<384xf32>
    %v8064 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8065 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8066 = stablehlo.multiply %v8064, %b11ebtv : tensor<384xf32>
    %v8067 = stablehlo.multiply %v8057, %v8057 : tensor<384xf32>
    %v8068 = stablehlo.multiply %v8065, %v8067 : tensor<384xf32>
    %v8069 = stablehlo.add %v8066, %v8068 : tensor<384xf32>
    %v8070 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8071 = stablehlo.add %v8069, %v8070 : tensor<384xf32>
    %v8072 = stablehlo.sqrt %v8071 : tensor<384xf32>
    %v8073 = stablehlo.divide %v8057, %v8072 : tensor<384xf32>
    %v8074 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8075 = stablehlo.multiply %v8074, %b11ebtm : tensor<384xf32>
    %v8076 = stablehlo.add %v8075, %v8073 : tensor<384xf32>
    %v8077 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8078 = stablehlo.multiply %v8077, %v8076 : tensor<384xf32>
    %v8079 = stablehlo.subtract %b11ebt, %v8078 : tensor<384xf32>
    %v8080 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8081 = stablehlo.multiply %v8080, %b11dW : tensor<384x1x3x3xf32>
    %v8082 = stablehlo.add %v8081, %v2889 : tensor<384x1x3x3xf32>
    %v8083 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8084 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8085 = stablehlo.multiply %v8083, %b11dWv : tensor<384x1x3x3xf32>
    %v8086 = stablehlo.multiply %v8082, %v8082 : tensor<384x1x3x3xf32>
    %v8087 = stablehlo.multiply %v8084, %v8086 : tensor<384x1x3x3xf32>
    %v8088 = stablehlo.add %v8085, %v8087 : tensor<384x1x3x3xf32>
    %v8089 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8090 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8091 = stablehlo.multiply %v8089, %b11dWv : tensor<384x1x3x3xf32>
    %v8092 = stablehlo.multiply %v8082, %v8082 : tensor<384x1x3x3xf32>
    %v8093 = stablehlo.multiply %v8090, %v8092 : tensor<384x1x3x3xf32>
    %v8094 = stablehlo.add %v8091, %v8093 : tensor<384x1x3x3xf32>
    %v8095 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8096 = stablehlo.add %v8094, %v8095 : tensor<384x1x3x3xf32>
    %v8097 = stablehlo.sqrt %v8096 : tensor<384x1x3x3xf32>
    %v8098 = stablehlo.divide %v8082, %v8097 : tensor<384x1x3x3xf32>
    %v8099 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8100 = stablehlo.multiply %v8099, %b11dWm : tensor<384x1x3x3xf32>
    %v8101 = stablehlo.add %v8100, %v8098 : tensor<384x1x3x3xf32>
    %v8102 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8103 = stablehlo.multiply %v8102, %v8101 : tensor<384x1x3x3xf32>
    %v8104 = stablehlo.subtract %b11dW, %v8103 : tensor<384x1x3x3xf32>
    %v8105 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8106 = stablehlo.multiply %v8105, %b11dg : tensor<384xf32>
    %v8107 = stablehlo.add %v8106, %v2907 : tensor<384xf32>
    %v8108 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8109 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8110 = stablehlo.multiply %v8108, %b11dgv : tensor<384xf32>
    %v8111 = stablehlo.multiply %v8107, %v8107 : tensor<384xf32>
    %v8112 = stablehlo.multiply %v8109, %v8111 : tensor<384xf32>
    %v8113 = stablehlo.add %v8110, %v8112 : tensor<384xf32>
    %v8114 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8115 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8116 = stablehlo.multiply %v8114, %b11dgv : tensor<384xf32>
    %v8117 = stablehlo.multiply %v8107, %v8107 : tensor<384xf32>
    %v8118 = stablehlo.multiply %v8115, %v8117 : tensor<384xf32>
    %v8119 = stablehlo.add %v8116, %v8118 : tensor<384xf32>
    %v8120 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8121 = stablehlo.add %v8119, %v8120 : tensor<384xf32>
    %v8122 = stablehlo.sqrt %v8121 : tensor<384xf32>
    %v8123 = stablehlo.divide %v8107, %v8122 : tensor<384xf32>
    %v8124 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8125 = stablehlo.multiply %v8124, %b11dgm : tensor<384xf32>
    %v8126 = stablehlo.add %v8125, %v8123 : tensor<384xf32>
    %v8127 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8128 = stablehlo.multiply %v8127, %v8126 : tensor<384xf32>
    %v8129 = stablehlo.subtract %b11dg, %v8128 : tensor<384xf32>
    %v8130 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8131 = stablehlo.multiply %v8130, %b11dbt : tensor<384xf32>
    %v8132 = stablehlo.add %v8131, %v2910 : tensor<384xf32>
    %v8133 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8134 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8135 = stablehlo.multiply %v8133, %b11dbtv : tensor<384xf32>
    %v8136 = stablehlo.multiply %v8132, %v8132 : tensor<384xf32>
    %v8137 = stablehlo.multiply %v8134, %v8136 : tensor<384xf32>
    %v8138 = stablehlo.add %v8135, %v8137 : tensor<384xf32>
    %v8139 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8140 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8141 = stablehlo.multiply %v8139, %b11dbtv : tensor<384xf32>
    %v8142 = stablehlo.multiply %v8132, %v8132 : tensor<384xf32>
    %v8143 = stablehlo.multiply %v8140, %v8142 : tensor<384xf32>
    %v8144 = stablehlo.add %v8141, %v8143 : tensor<384xf32>
    %v8145 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8146 = stablehlo.add %v8144, %v8145 : tensor<384xf32>
    %v8147 = stablehlo.sqrt %v8146 : tensor<384xf32>
    %v8148 = stablehlo.divide %v8132, %v8147 : tensor<384xf32>
    %v8149 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8150 = stablehlo.multiply %v8149, %b11dbtm : tensor<384xf32>
    %v8151 = stablehlo.add %v8150, %v8148 : tensor<384xf32>
    %v8152 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8153 = stablehlo.multiply %v8152, %v8151 : tensor<384xf32>
    %v8154 = stablehlo.subtract %b11dbt, %v8153 : tensor<384xf32>
    %v8155 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8156 = stablehlo.multiply %v8155, %b11pW : tensor<96x384x1x1xf32>
    %v8157 = stablehlo.add %v8156, %v2916 : tensor<96x384x1x1xf32>
    %v8158 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8159 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8160 = stablehlo.multiply %v8158, %b11pWv : tensor<96x384x1x1xf32>
    %v8161 = stablehlo.multiply %v8157, %v8157 : tensor<96x384x1x1xf32>
    %v8162 = stablehlo.multiply %v8159, %v8161 : tensor<96x384x1x1xf32>
    %v8163 = stablehlo.add %v8160, %v8162 : tensor<96x384x1x1xf32>
    %v8164 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8165 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8166 = stablehlo.multiply %v8164, %b11pWv : tensor<96x384x1x1xf32>
    %v8167 = stablehlo.multiply %v8157, %v8157 : tensor<96x384x1x1xf32>
    %v8168 = stablehlo.multiply %v8165, %v8167 : tensor<96x384x1x1xf32>
    %v8169 = stablehlo.add %v8166, %v8168 : tensor<96x384x1x1xf32>
    %v8170 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8171 = stablehlo.add %v8169, %v8170 : tensor<96x384x1x1xf32>
    %v8172 = stablehlo.sqrt %v8171 : tensor<96x384x1x1xf32>
    %v8173 = stablehlo.divide %v8157, %v8172 : tensor<96x384x1x1xf32>
    %v8174 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8175 = stablehlo.multiply %v8174, %b11pWm : tensor<96x384x1x1xf32>
    %v8176 = stablehlo.add %v8175, %v8173 : tensor<96x384x1x1xf32>
    %v8177 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8178 = stablehlo.multiply %v8177, %v8176 : tensor<96x384x1x1xf32>
    %v8179 = stablehlo.subtract %b11pW, %v8178 : tensor<96x384x1x1xf32>
    %v8180 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8181 = stablehlo.multiply %v8180, %b11pg : tensor<96xf32>
    %v8182 = stablehlo.add %v8181, %v2934 : tensor<96xf32>
    %v8183 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8184 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8185 = stablehlo.multiply %v8183, %b11pgv : tensor<96xf32>
    %v8186 = stablehlo.multiply %v8182, %v8182 : tensor<96xf32>
    %v8187 = stablehlo.multiply %v8184, %v8186 : tensor<96xf32>
    %v8188 = stablehlo.add %v8185, %v8187 : tensor<96xf32>
    %v8189 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8190 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8191 = stablehlo.multiply %v8189, %b11pgv : tensor<96xf32>
    %v8192 = stablehlo.multiply %v8182, %v8182 : tensor<96xf32>
    %v8193 = stablehlo.multiply %v8190, %v8192 : tensor<96xf32>
    %v8194 = stablehlo.add %v8191, %v8193 : tensor<96xf32>
    %v8195 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8196 = stablehlo.add %v8194, %v8195 : tensor<96xf32>
    %v8197 = stablehlo.sqrt %v8196 : tensor<96xf32>
    %v8198 = stablehlo.divide %v8182, %v8197 : tensor<96xf32>
    %v8199 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8200 = stablehlo.multiply %v8199, %b11pgm : tensor<96xf32>
    %v8201 = stablehlo.add %v8200, %v8198 : tensor<96xf32>
    %v8202 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8203 = stablehlo.multiply %v8202, %v8201 : tensor<96xf32>
    %v8204 = stablehlo.subtract %b11pg, %v8203 : tensor<96xf32>
    %v8205 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8206 = stablehlo.multiply %v8205, %b11pbt : tensor<96xf32>
    %v8207 = stablehlo.add %v8206, %v2937 : tensor<96xf32>
    %v8208 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8209 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8210 = stablehlo.multiply %v8208, %b11pbtv : tensor<96xf32>
    %v8211 = stablehlo.multiply %v8207, %v8207 : tensor<96xf32>
    %v8212 = stablehlo.multiply %v8209, %v8211 : tensor<96xf32>
    %v8213 = stablehlo.add %v8210, %v8212 : tensor<96xf32>
    %v8214 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8215 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8216 = stablehlo.multiply %v8214, %b11pbtv : tensor<96xf32>
    %v8217 = stablehlo.multiply %v8207, %v8207 : tensor<96xf32>
    %v8218 = stablehlo.multiply %v8215, %v8217 : tensor<96xf32>
    %v8219 = stablehlo.add %v8216, %v8218 : tensor<96xf32>
    %v8220 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8221 = stablehlo.add %v8219, %v8220 : tensor<96xf32>
    %v8222 = stablehlo.sqrt %v8221 : tensor<96xf32>
    %v8223 = stablehlo.divide %v8207, %v8222 : tensor<96xf32>
    %v8224 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8225 = stablehlo.multiply %v8224, %b11pbtm : tensor<96xf32>
    %v8226 = stablehlo.add %v8225, %v8223 : tensor<96xf32>
    %v8227 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8228 = stablehlo.multiply %v8227, %v8226 : tensor<96xf32>
    %v8229 = stablehlo.subtract %b11pbt, %v8228 : tensor<96xf32>
    %v8230 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8231 = stablehlo.multiply %v8230, %b12eW : tensor<576x96x1x1xf32>
    %v8232 = stablehlo.add %v8231, %v2665 : tensor<576x96x1x1xf32>
    %v8233 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8234 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8235 = stablehlo.multiply %v8233, %b12eWv : tensor<576x96x1x1xf32>
    %v8236 = stablehlo.multiply %v8232, %v8232 : tensor<576x96x1x1xf32>
    %v8237 = stablehlo.multiply %v8234, %v8236 : tensor<576x96x1x1xf32>
    %v8238 = stablehlo.add %v8235, %v8237 : tensor<576x96x1x1xf32>
    %v8239 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8240 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8241 = stablehlo.multiply %v8239, %b12eWv : tensor<576x96x1x1xf32>
    %v8242 = stablehlo.multiply %v8232, %v8232 : tensor<576x96x1x1xf32>
    %v8243 = stablehlo.multiply %v8240, %v8242 : tensor<576x96x1x1xf32>
    %v8244 = stablehlo.add %v8241, %v8243 : tensor<576x96x1x1xf32>
    %v8245 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8246 = stablehlo.add %v8244, %v8245 : tensor<576x96x1x1xf32>
    %v8247 = stablehlo.sqrt %v8246 : tensor<576x96x1x1xf32>
    %v8248 = stablehlo.divide %v8232, %v8247 : tensor<576x96x1x1xf32>
    %v8249 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8250 = stablehlo.multiply %v8249, %b12eWm : tensor<576x96x1x1xf32>
    %v8251 = stablehlo.add %v8250, %v8248 : tensor<576x96x1x1xf32>
    %v8252 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8253 = stablehlo.multiply %v8252, %v8251 : tensor<576x96x1x1xf32>
    %v8254 = stablehlo.subtract %b12eW, %v8253 : tensor<576x96x1x1xf32>
    %v8255 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8256 = stablehlo.multiply %v8255, %b12eg : tensor<576xf32>
    %v8257 = stablehlo.add %v8256, %v2683 : tensor<576xf32>
    %v8258 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8259 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8260 = stablehlo.multiply %v8258, %b12egv : tensor<576xf32>
    %v8261 = stablehlo.multiply %v8257, %v8257 : tensor<576xf32>
    %v8262 = stablehlo.multiply %v8259, %v8261 : tensor<576xf32>
    %v8263 = stablehlo.add %v8260, %v8262 : tensor<576xf32>
    %v8264 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8265 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8266 = stablehlo.multiply %v8264, %b12egv : tensor<576xf32>
    %v8267 = stablehlo.multiply %v8257, %v8257 : tensor<576xf32>
    %v8268 = stablehlo.multiply %v8265, %v8267 : tensor<576xf32>
    %v8269 = stablehlo.add %v8266, %v8268 : tensor<576xf32>
    %v8270 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8271 = stablehlo.add %v8269, %v8270 : tensor<576xf32>
    %v8272 = stablehlo.sqrt %v8271 : tensor<576xf32>
    %v8273 = stablehlo.divide %v8257, %v8272 : tensor<576xf32>
    %v8274 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8275 = stablehlo.multiply %v8274, %b12egm : tensor<576xf32>
    %v8276 = stablehlo.add %v8275, %v8273 : tensor<576xf32>
    %v8277 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8278 = stablehlo.multiply %v8277, %v8276 : tensor<576xf32>
    %v8279 = stablehlo.subtract %b12eg, %v8278 : tensor<576xf32>
    %v8280 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8281 = stablehlo.multiply %v8280, %b12ebt : tensor<576xf32>
    %v8282 = stablehlo.add %v8281, %v2686 : tensor<576xf32>
    %v8283 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8284 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8285 = stablehlo.multiply %v8283, %b12ebtv : tensor<576xf32>
    %v8286 = stablehlo.multiply %v8282, %v8282 : tensor<576xf32>
    %v8287 = stablehlo.multiply %v8284, %v8286 : tensor<576xf32>
    %v8288 = stablehlo.add %v8285, %v8287 : tensor<576xf32>
    %v8289 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8290 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8291 = stablehlo.multiply %v8289, %b12ebtv : tensor<576xf32>
    %v8292 = stablehlo.multiply %v8282, %v8282 : tensor<576xf32>
    %v8293 = stablehlo.multiply %v8290, %v8292 : tensor<576xf32>
    %v8294 = stablehlo.add %v8291, %v8293 : tensor<576xf32>
    %v8295 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8296 = stablehlo.add %v8294, %v8295 : tensor<576xf32>
    %v8297 = stablehlo.sqrt %v8296 : tensor<576xf32>
    %v8298 = stablehlo.divide %v8282, %v8297 : tensor<576xf32>
    %v8299 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8300 = stablehlo.multiply %v8299, %b12ebtm : tensor<576xf32>
    %v8301 = stablehlo.add %v8300, %v8298 : tensor<576xf32>
    %v8302 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8303 = stablehlo.multiply %v8302, %v8301 : tensor<576xf32>
    %v8304 = stablehlo.subtract %b12ebt, %v8303 : tensor<576xf32>
    %v8305 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8306 = stablehlo.multiply %v8305, %b12dW : tensor<576x1x3x3xf32>
    %v8307 = stablehlo.add %v8306, %v2692 : tensor<576x1x3x3xf32>
    %v8308 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8309 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8310 = stablehlo.multiply %v8308, %b12dWv : tensor<576x1x3x3xf32>
    %v8311 = stablehlo.multiply %v8307, %v8307 : tensor<576x1x3x3xf32>
    %v8312 = stablehlo.multiply %v8309, %v8311 : tensor<576x1x3x3xf32>
    %v8313 = stablehlo.add %v8310, %v8312 : tensor<576x1x3x3xf32>
    %v8314 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8315 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8316 = stablehlo.multiply %v8314, %b12dWv : tensor<576x1x3x3xf32>
    %v8317 = stablehlo.multiply %v8307, %v8307 : tensor<576x1x3x3xf32>
    %v8318 = stablehlo.multiply %v8315, %v8317 : tensor<576x1x3x3xf32>
    %v8319 = stablehlo.add %v8316, %v8318 : tensor<576x1x3x3xf32>
    %v8320 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8321 = stablehlo.add %v8319, %v8320 : tensor<576x1x3x3xf32>
    %v8322 = stablehlo.sqrt %v8321 : tensor<576x1x3x3xf32>
    %v8323 = stablehlo.divide %v8307, %v8322 : tensor<576x1x3x3xf32>
    %v8324 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8325 = stablehlo.multiply %v8324, %b12dWm : tensor<576x1x3x3xf32>
    %v8326 = stablehlo.add %v8325, %v8323 : tensor<576x1x3x3xf32>
    %v8327 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8328 = stablehlo.multiply %v8327, %v8326 : tensor<576x1x3x3xf32>
    %v8329 = stablehlo.subtract %b12dW, %v8328 : tensor<576x1x3x3xf32>
    %v8330 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8331 = stablehlo.multiply %v8330, %b12dg : tensor<576xf32>
    %v8332 = stablehlo.add %v8331, %v2710 : tensor<576xf32>
    %v8333 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8334 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8335 = stablehlo.multiply %v8333, %b12dgv : tensor<576xf32>
    %v8336 = stablehlo.multiply %v8332, %v8332 : tensor<576xf32>
    %v8337 = stablehlo.multiply %v8334, %v8336 : tensor<576xf32>
    %v8338 = stablehlo.add %v8335, %v8337 : tensor<576xf32>
    %v8339 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8340 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8341 = stablehlo.multiply %v8339, %b12dgv : tensor<576xf32>
    %v8342 = stablehlo.multiply %v8332, %v8332 : tensor<576xf32>
    %v8343 = stablehlo.multiply %v8340, %v8342 : tensor<576xf32>
    %v8344 = stablehlo.add %v8341, %v8343 : tensor<576xf32>
    %v8345 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8346 = stablehlo.add %v8344, %v8345 : tensor<576xf32>
    %v8347 = stablehlo.sqrt %v8346 : tensor<576xf32>
    %v8348 = stablehlo.divide %v8332, %v8347 : tensor<576xf32>
    %v8349 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8350 = stablehlo.multiply %v8349, %b12dgm : tensor<576xf32>
    %v8351 = stablehlo.add %v8350, %v8348 : tensor<576xf32>
    %v8352 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8353 = stablehlo.multiply %v8352, %v8351 : tensor<576xf32>
    %v8354 = stablehlo.subtract %b12dg, %v8353 : tensor<576xf32>
    %v8355 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8356 = stablehlo.multiply %v8355, %b12dbt : tensor<576xf32>
    %v8357 = stablehlo.add %v8356, %v2713 : tensor<576xf32>
    %v8358 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8359 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8360 = stablehlo.multiply %v8358, %b12dbtv : tensor<576xf32>
    %v8361 = stablehlo.multiply %v8357, %v8357 : tensor<576xf32>
    %v8362 = stablehlo.multiply %v8359, %v8361 : tensor<576xf32>
    %v8363 = stablehlo.add %v8360, %v8362 : tensor<576xf32>
    %v8364 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8365 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8366 = stablehlo.multiply %v8364, %b12dbtv : tensor<576xf32>
    %v8367 = stablehlo.multiply %v8357, %v8357 : tensor<576xf32>
    %v8368 = stablehlo.multiply %v8365, %v8367 : tensor<576xf32>
    %v8369 = stablehlo.add %v8366, %v8368 : tensor<576xf32>
    %v8370 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8371 = stablehlo.add %v8369, %v8370 : tensor<576xf32>
    %v8372 = stablehlo.sqrt %v8371 : tensor<576xf32>
    %v8373 = stablehlo.divide %v8357, %v8372 : tensor<576xf32>
    %v8374 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8375 = stablehlo.multiply %v8374, %b12dbtm : tensor<576xf32>
    %v8376 = stablehlo.add %v8375, %v8373 : tensor<576xf32>
    %v8377 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8378 = stablehlo.multiply %v8377, %v8376 : tensor<576xf32>
    %v8379 = stablehlo.subtract %b12dbt, %v8378 : tensor<576xf32>
    %v8380 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8381 = stablehlo.multiply %v8380, %b12pW : tensor<96x576x1x1xf32>
    %v8382 = stablehlo.add %v8381, %v2719 : tensor<96x576x1x1xf32>
    %v8383 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8384 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8385 = stablehlo.multiply %v8383, %b12pWv : tensor<96x576x1x1xf32>
    %v8386 = stablehlo.multiply %v8382, %v8382 : tensor<96x576x1x1xf32>
    %v8387 = stablehlo.multiply %v8384, %v8386 : tensor<96x576x1x1xf32>
    %v8388 = stablehlo.add %v8385, %v8387 : tensor<96x576x1x1xf32>
    %v8389 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8390 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8391 = stablehlo.multiply %v8389, %b12pWv : tensor<96x576x1x1xf32>
    %v8392 = stablehlo.multiply %v8382, %v8382 : tensor<96x576x1x1xf32>
    %v8393 = stablehlo.multiply %v8390, %v8392 : tensor<96x576x1x1xf32>
    %v8394 = stablehlo.add %v8391, %v8393 : tensor<96x576x1x1xf32>
    %v8395 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8396 = stablehlo.add %v8394, %v8395 : tensor<96x576x1x1xf32>
    %v8397 = stablehlo.sqrt %v8396 : tensor<96x576x1x1xf32>
    %v8398 = stablehlo.divide %v8382, %v8397 : tensor<96x576x1x1xf32>
    %v8399 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8400 = stablehlo.multiply %v8399, %b12pWm : tensor<96x576x1x1xf32>
    %v8401 = stablehlo.add %v8400, %v8398 : tensor<96x576x1x1xf32>
    %v8402 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8403 = stablehlo.multiply %v8402, %v8401 : tensor<96x576x1x1xf32>
    %v8404 = stablehlo.subtract %b12pW, %v8403 : tensor<96x576x1x1xf32>
    %v8405 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8406 = stablehlo.multiply %v8405, %b12pg : tensor<96xf32>
    %v8407 = stablehlo.add %v8406, %v2737 : tensor<96xf32>
    %v8408 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8409 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8410 = stablehlo.multiply %v8408, %b12pgv : tensor<96xf32>
    %v8411 = stablehlo.multiply %v8407, %v8407 : tensor<96xf32>
    %v8412 = stablehlo.multiply %v8409, %v8411 : tensor<96xf32>
    %v8413 = stablehlo.add %v8410, %v8412 : tensor<96xf32>
    %v8414 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8415 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8416 = stablehlo.multiply %v8414, %b12pgv : tensor<96xf32>
    %v8417 = stablehlo.multiply %v8407, %v8407 : tensor<96xf32>
    %v8418 = stablehlo.multiply %v8415, %v8417 : tensor<96xf32>
    %v8419 = stablehlo.add %v8416, %v8418 : tensor<96xf32>
    %v8420 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8421 = stablehlo.add %v8419, %v8420 : tensor<96xf32>
    %v8422 = stablehlo.sqrt %v8421 : tensor<96xf32>
    %v8423 = stablehlo.divide %v8407, %v8422 : tensor<96xf32>
    %v8424 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8425 = stablehlo.multiply %v8424, %b12pgm : tensor<96xf32>
    %v8426 = stablehlo.add %v8425, %v8423 : tensor<96xf32>
    %v8427 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8428 = stablehlo.multiply %v8427, %v8426 : tensor<96xf32>
    %v8429 = stablehlo.subtract %b12pg, %v8428 : tensor<96xf32>
    %v8430 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8431 = stablehlo.multiply %v8430, %b12pbt : tensor<96xf32>
    %v8432 = stablehlo.add %v8431, %v2740 : tensor<96xf32>
    %v8433 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8434 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8435 = stablehlo.multiply %v8433, %b12pbtv : tensor<96xf32>
    %v8436 = stablehlo.multiply %v8432, %v8432 : tensor<96xf32>
    %v8437 = stablehlo.multiply %v8434, %v8436 : tensor<96xf32>
    %v8438 = stablehlo.add %v8435, %v8437 : tensor<96xf32>
    %v8439 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8440 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8441 = stablehlo.multiply %v8439, %b12pbtv : tensor<96xf32>
    %v8442 = stablehlo.multiply %v8432, %v8432 : tensor<96xf32>
    %v8443 = stablehlo.multiply %v8440, %v8442 : tensor<96xf32>
    %v8444 = stablehlo.add %v8441, %v8443 : tensor<96xf32>
    %v8445 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8446 = stablehlo.add %v8444, %v8445 : tensor<96xf32>
    %v8447 = stablehlo.sqrt %v8446 : tensor<96xf32>
    %v8448 = stablehlo.divide %v8432, %v8447 : tensor<96xf32>
    %v8449 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8450 = stablehlo.multiply %v8449, %b12pbtm : tensor<96xf32>
    %v8451 = stablehlo.add %v8450, %v8448 : tensor<96xf32>
    %v8452 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8453 = stablehlo.multiply %v8452, %v8451 : tensor<96xf32>
    %v8454 = stablehlo.subtract %b12pbt, %v8453 : tensor<96xf32>
    %v8455 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8456 = stablehlo.multiply %v8455, %b13eW : tensor<576x96x1x1xf32>
    %v8457 = stablehlo.add %v8456, %v2467 : tensor<576x96x1x1xf32>
    %v8458 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8459 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8460 = stablehlo.multiply %v8458, %b13eWv : tensor<576x96x1x1xf32>
    %v8461 = stablehlo.multiply %v8457, %v8457 : tensor<576x96x1x1xf32>
    %v8462 = stablehlo.multiply %v8459, %v8461 : tensor<576x96x1x1xf32>
    %v8463 = stablehlo.add %v8460, %v8462 : tensor<576x96x1x1xf32>
    %v8464 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8465 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8466 = stablehlo.multiply %v8464, %b13eWv : tensor<576x96x1x1xf32>
    %v8467 = stablehlo.multiply %v8457, %v8457 : tensor<576x96x1x1xf32>
    %v8468 = stablehlo.multiply %v8465, %v8467 : tensor<576x96x1x1xf32>
    %v8469 = stablehlo.add %v8466, %v8468 : tensor<576x96x1x1xf32>
    %v8470 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8471 = stablehlo.add %v8469, %v8470 : tensor<576x96x1x1xf32>
    %v8472 = stablehlo.sqrt %v8471 : tensor<576x96x1x1xf32>
    %v8473 = stablehlo.divide %v8457, %v8472 : tensor<576x96x1x1xf32>
    %v8474 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8475 = stablehlo.multiply %v8474, %b13eWm : tensor<576x96x1x1xf32>
    %v8476 = stablehlo.add %v8475, %v8473 : tensor<576x96x1x1xf32>
    %v8477 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8478 = stablehlo.multiply %v8477, %v8476 : tensor<576x96x1x1xf32>
    %v8479 = stablehlo.subtract %b13eW, %v8478 : tensor<576x96x1x1xf32>
    %v8480 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8481 = stablehlo.multiply %v8480, %b13eg : tensor<576xf32>
    %v8482 = stablehlo.add %v8481, %v2485 : tensor<576xf32>
    %v8483 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8484 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8485 = stablehlo.multiply %v8483, %b13egv : tensor<576xf32>
    %v8486 = stablehlo.multiply %v8482, %v8482 : tensor<576xf32>
    %v8487 = stablehlo.multiply %v8484, %v8486 : tensor<576xf32>
    %v8488 = stablehlo.add %v8485, %v8487 : tensor<576xf32>
    %v8489 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8490 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8491 = stablehlo.multiply %v8489, %b13egv : tensor<576xf32>
    %v8492 = stablehlo.multiply %v8482, %v8482 : tensor<576xf32>
    %v8493 = stablehlo.multiply %v8490, %v8492 : tensor<576xf32>
    %v8494 = stablehlo.add %v8491, %v8493 : tensor<576xf32>
    %v8495 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8496 = stablehlo.add %v8494, %v8495 : tensor<576xf32>
    %v8497 = stablehlo.sqrt %v8496 : tensor<576xf32>
    %v8498 = stablehlo.divide %v8482, %v8497 : tensor<576xf32>
    %v8499 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8500 = stablehlo.multiply %v8499, %b13egm : tensor<576xf32>
    %v8501 = stablehlo.add %v8500, %v8498 : tensor<576xf32>
    %v8502 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8503 = stablehlo.multiply %v8502, %v8501 : tensor<576xf32>
    %v8504 = stablehlo.subtract %b13eg, %v8503 : tensor<576xf32>
    %v8505 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8506 = stablehlo.multiply %v8505, %b13ebt : tensor<576xf32>
    %v8507 = stablehlo.add %v8506, %v2488 : tensor<576xf32>
    %v8508 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8509 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8510 = stablehlo.multiply %v8508, %b13ebtv : tensor<576xf32>
    %v8511 = stablehlo.multiply %v8507, %v8507 : tensor<576xf32>
    %v8512 = stablehlo.multiply %v8509, %v8511 : tensor<576xf32>
    %v8513 = stablehlo.add %v8510, %v8512 : tensor<576xf32>
    %v8514 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8515 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8516 = stablehlo.multiply %v8514, %b13ebtv : tensor<576xf32>
    %v8517 = stablehlo.multiply %v8507, %v8507 : tensor<576xf32>
    %v8518 = stablehlo.multiply %v8515, %v8517 : tensor<576xf32>
    %v8519 = stablehlo.add %v8516, %v8518 : tensor<576xf32>
    %v8520 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8521 = stablehlo.add %v8519, %v8520 : tensor<576xf32>
    %v8522 = stablehlo.sqrt %v8521 : tensor<576xf32>
    %v8523 = stablehlo.divide %v8507, %v8522 : tensor<576xf32>
    %v8524 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8525 = stablehlo.multiply %v8524, %b13ebtm : tensor<576xf32>
    %v8526 = stablehlo.add %v8525, %v8523 : tensor<576xf32>
    %v8527 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8528 = stablehlo.multiply %v8527, %v8526 : tensor<576xf32>
    %v8529 = stablehlo.subtract %b13ebt, %v8528 : tensor<576xf32>
    %v8530 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8531 = stablehlo.multiply %v8530, %b13dW : tensor<576x1x3x3xf32>
    %v8532 = stablehlo.add %v8531, %v2494 : tensor<576x1x3x3xf32>
    %v8533 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8534 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8535 = stablehlo.multiply %v8533, %b13dWv : tensor<576x1x3x3xf32>
    %v8536 = stablehlo.multiply %v8532, %v8532 : tensor<576x1x3x3xf32>
    %v8537 = stablehlo.multiply %v8534, %v8536 : tensor<576x1x3x3xf32>
    %v8538 = stablehlo.add %v8535, %v8537 : tensor<576x1x3x3xf32>
    %v8539 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8540 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8541 = stablehlo.multiply %v8539, %b13dWv : tensor<576x1x3x3xf32>
    %v8542 = stablehlo.multiply %v8532, %v8532 : tensor<576x1x3x3xf32>
    %v8543 = stablehlo.multiply %v8540, %v8542 : tensor<576x1x3x3xf32>
    %v8544 = stablehlo.add %v8541, %v8543 : tensor<576x1x3x3xf32>
    %v8545 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8546 = stablehlo.add %v8544, %v8545 : tensor<576x1x3x3xf32>
    %v8547 = stablehlo.sqrt %v8546 : tensor<576x1x3x3xf32>
    %v8548 = stablehlo.divide %v8532, %v8547 : tensor<576x1x3x3xf32>
    %v8549 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8550 = stablehlo.multiply %v8549, %b13dWm : tensor<576x1x3x3xf32>
    %v8551 = stablehlo.add %v8550, %v8548 : tensor<576x1x3x3xf32>
    %v8552 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8553 = stablehlo.multiply %v8552, %v8551 : tensor<576x1x3x3xf32>
    %v8554 = stablehlo.subtract %b13dW, %v8553 : tensor<576x1x3x3xf32>
    %v8555 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8556 = stablehlo.multiply %v8555, %b13dg : tensor<576xf32>
    %v8557 = stablehlo.add %v8556, %v2512 : tensor<576xf32>
    %v8558 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8559 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8560 = stablehlo.multiply %v8558, %b13dgv : tensor<576xf32>
    %v8561 = stablehlo.multiply %v8557, %v8557 : tensor<576xf32>
    %v8562 = stablehlo.multiply %v8559, %v8561 : tensor<576xf32>
    %v8563 = stablehlo.add %v8560, %v8562 : tensor<576xf32>
    %v8564 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8565 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8566 = stablehlo.multiply %v8564, %b13dgv : tensor<576xf32>
    %v8567 = stablehlo.multiply %v8557, %v8557 : tensor<576xf32>
    %v8568 = stablehlo.multiply %v8565, %v8567 : tensor<576xf32>
    %v8569 = stablehlo.add %v8566, %v8568 : tensor<576xf32>
    %v8570 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8571 = stablehlo.add %v8569, %v8570 : tensor<576xf32>
    %v8572 = stablehlo.sqrt %v8571 : tensor<576xf32>
    %v8573 = stablehlo.divide %v8557, %v8572 : tensor<576xf32>
    %v8574 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8575 = stablehlo.multiply %v8574, %b13dgm : tensor<576xf32>
    %v8576 = stablehlo.add %v8575, %v8573 : tensor<576xf32>
    %v8577 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8578 = stablehlo.multiply %v8577, %v8576 : tensor<576xf32>
    %v8579 = stablehlo.subtract %b13dg, %v8578 : tensor<576xf32>
    %v8580 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8581 = stablehlo.multiply %v8580, %b13dbt : tensor<576xf32>
    %v8582 = stablehlo.add %v8581, %v2515 : tensor<576xf32>
    %v8583 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8584 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8585 = stablehlo.multiply %v8583, %b13dbtv : tensor<576xf32>
    %v8586 = stablehlo.multiply %v8582, %v8582 : tensor<576xf32>
    %v8587 = stablehlo.multiply %v8584, %v8586 : tensor<576xf32>
    %v8588 = stablehlo.add %v8585, %v8587 : tensor<576xf32>
    %v8589 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8590 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8591 = stablehlo.multiply %v8589, %b13dbtv : tensor<576xf32>
    %v8592 = stablehlo.multiply %v8582, %v8582 : tensor<576xf32>
    %v8593 = stablehlo.multiply %v8590, %v8592 : tensor<576xf32>
    %v8594 = stablehlo.add %v8591, %v8593 : tensor<576xf32>
    %v8595 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8596 = stablehlo.add %v8594, %v8595 : tensor<576xf32>
    %v8597 = stablehlo.sqrt %v8596 : tensor<576xf32>
    %v8598 = stablehlo.divide %v8582, %v8597 : tensor<576xf32>
    %v8599 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8600 = stablehlo.multiply %v8599, %b13dbtm : tensor<576xf32>
    %v8601 = stablehlo.add %v8600, %v8598 : tensor<576xf32>
    %v8602 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8603 = stablehlo.multiply %v8602, %v8601 : tensor<576xf32>
    %v8604 = stablehlo.subtract %b13dbt, %v8603 : tensor<576xf32>
    %v8605 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8606 = stablehlo.multiply %v8605, %b13pW : tensor<96x576x1x1xf32>
    %v8607 = stablehlo.add %v8606, %v2521 : tensor<96x576x1x1xf32>
    %v8608 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8609 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8610 = stablehlo.multiply %v8608, %b13pWv : tensor<96x576x1x1xf32>
    %v8611 = stablehlo.multiply %v8607, %v8607 : tensor<96x576x1x1xf32>
    %v8612 = stablehlo.multiply %v8609, %v8611 : tensor<96x576x1x1xf32>
    %v8613 = stablehlo.add %v8610, %v8612 : tensor<96x576x1x1xf32>
    %v8614 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8615 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8616 = stablehlo.multiply %v8614, %b13pWv : tensor<96x576x1x1xf32>
    %v8617 = stablehlo.multiply %v8607, %v8607 : tensor<96x576x1x1xf32>
    %v8618 = stablehlo.multiply %v8615, %v8617 : tensor<96x576x1x1xf32>
    %v8619 = stablehlo.add %v8616, %v8618 : tensor<96x576x1x1xf32>
    %v8620 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8621 = stablehlo.add %v8619, %v8620 : tensor<96x576x1x1xf32>
    %v8622 = stablehlo.sqrt %v8621 : tensor<96x576x1x1xf32>
    %v8623 = stablehlo.divide %v8607, %v8622 : tensor<96x576x1x1xf32>
    %v8624 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8625 = stablehlo.multiply %v8624, %b13pWm : tensor<96x576x1x1xf32>
    %v8626 = stablehlo.add %v8625, %v8623 : tensor<96x576x1x1xf32>
    %v8627 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8628 = stablehlo.multiply %v8627, %v8626 : tensor<96x576x1x1xf32>
    %v8629 = stablehlo.subtract %b13pW, %v8628 : tensor<96x576x1x1xf32>
    %v8630 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8631 = stablehlo.multiply %v8630, %b13pg : tensor<96xf32>
    %v8632 = stablehlo.add %v8631, %v2539 : tensor<96xf32>
    %v8633 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8634 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8635 = stablehlo.multiply %v8633, %b13pgv : tensor<96xf32>
    %v8636 = stablehlo.multiply %v8632, %v8632 : tensor<96xf32>
    %v8637 = stablehlo.multiply %v8634, %v8636 : tensor<96xf32>
    %v8638 = stablehlo.add %v8635, %v8637 : tensor<96xf32>
    %v8639 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8640 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8641 = stablehlo.multiply %v8639, %b13pgv : tensor<96xf32>
    %v8642 = stablehlo.multiply %v8632, %v8632 : tensor<96xf32>
    %v8643 = stablehlo.multiply %v8640, %v8642 : tensor<96xf32>
    %v8644 = stablehlo.add %v8641, %v8643 : tensor<96xf32>
    %v8645 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8646 = stablehlo.add %v8644, %v8645 : tensor<96xf32>
    %v8647 = stablehlo.sqrt %v8646 : tensor<96xf32>
    %v8648 = stablehlo.divide %v8632, %v8647 : tensor<96xf32>
    %v8649 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8650 = stablehlo.multiply %v8649, %b13pgm : tensor<96xf32>
    %v8651 = stablehlo.add %v8650, %v8648 : tensor<96xf32>
    %v8652 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8653 = stablehlo.multiply %v8652, %v8651 : tensor<96xf32>
    %v8654 = stablehlo.subtract %b13pg, %v8653 : tensor<96xf32>
    %v8655 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8656 = stablehlo.multiply %v8655, %b13pbt : tensor<96xf32>
    %v8657 = stablehlo.add %v8656, %v2542 : tensor<96xf32>
    %v8658 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8659 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8660 = stablehlo.multiply %v8658, %b13pbtv : tensor<96xf32>
    %v8661 = stablehlo.multiply %v8657, %v8657 : tensor<96xf32>
    %v8662 = stablehlo.multiply %v8659, %v8661 : tensor<96xf32>
    %v8663 = stablehlo.add %v8660, %v8662 : tensor<96xf32>
    %v8664 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8665 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8666 = stablehlo.multiply %v8664, %b13pbtv : tensor<96xf32>
    %v8667 = stablehlo.multiply %v8657, %v8657 : tensor<96xf32>
    %v8668 = stablehlo.multiply %v8665, %v8667 : tensor<96xf32>
    %v8669 = stablehlo.add %v8666, %v8668 : tensor<96xf32>
    %v8670 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8671 = stablehlo.add %v8669, %v8670 : tensor<96xf32>
    %v8672 = stablehlo.sqrt %v8671 : tensor<96xf32>
    %v8673 = stablehlo.divide %v8657, %v8672 : tensor<96xf32>
    %v8674 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8675 = stablehlo.multiply %v8674, %b13pbtm : tensor<96xf32>
    %v8676 = stablehlo.add %v8675, %v8673 : tensor<96xf32>
    %v8677 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8678 = stablehlo.multiply %v8677, %v8676 : tensor<96xf32>
    %v8679 = stablehlo.subtract %b13pbt, %v8678 : tensor<96xf32>
    %v8680 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8681 = stablehlo.multiply %v8680, %b14eW : tensor<576x96x1x1xf32>
    %v8682 = stablehlo.add %v8681, %v2267 : tensor<576x96x1x1xf32>
    %v8683 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8684 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8685 = stablehlo.multiply %v8683, %b14eWv : tensor<576x96x1x1xf32>
    %v8686 = stablehlo.multiply %v8682, %v8682 : tensor<576x96x1x1xf32>
    %v8687 = stablehlo.multiply %v8684, %v8686 : tensor<576x96x1x1xf32>
    %v8688 = stablehlo.add %v8685, %v8687 : tensor<576x96x1x1xf32>
    %v8689 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8690 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8691 = stablehlo.multiply %v8689, %b14eWv : tensor<576x96x1x1xf32>
    %v8692 = stablehlo.multiply %v8682, %v8682 : tensor<576x96x1x1xf32>
    %v8693 = stablehlo.multiply %v8690, %v8692 : tensor<576x96x1x1xf32>
    %v8694 = stablehlo.add %v8691, %v8693 : tensor<576x96x1x1xf32>
    %v8695 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8696 = stablehlo.add %v8694, %v8695 : tensor<576x96x1x1xf32>
    %v8697 = stablehlo.sqrt %v8696 : tensor<576x96x1x1xf32>
    %v8698 = stablehlo.divide %v8682, %v8697 : tensor<576x96x1x1xf32>
    %v8699 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8700 = stablehlo.multiply %v8699, %b14eWm : tensor<576x96x1x1xf32>
    %v8701 = stablehlo.add %v8700, %v8698 : tensor<576x96x1x1xf32>
    %v8702 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8703 = stablehlo.multiply %v8702, %v8701 : tensor<576x96x1x1xf32>
    %v8704 = stablehlo.subtract %b14eW, %v8703 : tensor<576x96x1x1xf32>
    %v8705 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8706 = stablehlo.multiply %v8705, %b14eg : tensor<576xf32>
    %v8707 = stablehlo.add %v8706, %v2285 : tensor<576xf32>
    %v8708 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8709 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8710 = stablehlo.multiply %v8708, %b14egv : tensor<576xf32>
    %v8711 = stablehlo.multiply %v8707, %v8707 : tensor<576xf32>
    %v8712 = stablehlo.multiply %v8709, %v8711 : tensor<576xf32>
    %v8713 = stablehlo.add %v8710, %v8712 : tensor<576xf32>
    %v8714 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8715 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8716 = stablehlo.multiply %v8714, %b14egv : tensor<576xf32>
    %v8717 = stablehlo.multiply %v8707, %v8707 : tensor<576xf32>
    %v8718 = stablehlo.multiply %v8715, %v8717 : tensor<576xf32>
    %v8719 = stablehlo.add %v8716, %v8718 : tensor<576xf32>
    %v8720 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8721 = stablehlo.add %v8719, %v8720 : tensor<576xf32>
    %v8722 = stablehlo.sqrt %v8721 : tensor<576xf32>
    %v8723 = stablehlo.divide %v8707, %v8722 : tensor<576xf32>
    %v8724 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8725 = stablehlo.multiply %v8724, %b14egm : tensor<576xf32>
    %v8726 = stablehlo.add %v8725, %v8723 : tensor<576xf32>
    %v8727 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8728 = stablehlo.multiply %v8727, %v8726 : tensor<576xf32>
    %v8729 = stablehlo.subtract %b14eg, %v8728 : tensor<576xf32>
    %v8730 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8731 = stablehlo.multiply %v8730, %b14ebt : tensor<576xf32>
    %v8732 = stablehlo.add %v8731, %v2288 : tensor<576xf32>
    %v8733 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8734 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8735 = stablehlo.multiply %v8733, %b14ebtv : tensor<576xf32>
    %v8736 = stablehlo.multiply %v8732, %v8732 : tensor<576xf32>
    %v8737 = stablehlo.multiply %v8734, %v8736 : tensor<576xf32>
    %v8738 = stablehlo.add %v8735, %v8737 : tensor<576xf32>
    %v8739 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8740 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8741 = stablehlo.multiply %v8739, %b14ebtv : tensor<576xf32>
    %v8742 = stablehlo.multiply %v8732, %v8732 : tensor<576xf32>
    %v8743 = stablehlo.multiply %v8740, %v8742 : tensor<576xf32>
    %v8744 = stablehlo.add %v8741, %v8743 : tensor<576xf32>
    %v8745 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8746 = stablehlo.add %v8744, %v8745 : tensor<576xf32>
    %v8747 = stablehlo.sqrt %v8746 : tensor<576xf32>
    %v8748 = stablehlo.divide %v8732, %v8747 : tensor<576xf32>
    %v8749 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8750 = stablehlo.multiply %v8749, %b14ebtm : tensor<576xf32>
    %v8751 = stablehlo.add %v8750, %v8748 : tensor<576xf32>
    %v8752 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8753 = stablehlo.multiply %v8752, %v8751 : tensor<576xf32>
    %v8754 = stablehlo.subtract %b14ebt, %v8753 : tensor<576xf32>
    %v8755 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8756 = stablehlo.multiply %v8755, %b14dW : tensor<576x1x3x3xf32>
    %v8757 = stablehlo.add %v8756, %v2296 : tensor<576x1x3x3xf32>
    %v8758 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8759 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8760 = stablehlo.multiply %v8758, %b14dWv : tensor<576x1x3x3xf32>
    %v8761 = stablehlo.multiply %v8757, %v8757 : tensor<576x1x3x3xf32>
    %v8762 = stablehlo.multiply %v8759, %v8761 : tensor<576x1x3x3xf32>
    %v8763 = stablehlo.add %v8760, %v8762 : tensor<576x1x3x3xf32>
    %v8764 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8765 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8766 = stablehlo.multiply %v8764, %b14dWv : tensor<576x1x3x3xf32>
    %v8767 = stablehlo.multiply %v8757, %v8757 : tensor<576x1x3x3xf32>
    %v8768 = stablehlo.multiply %v8765, %v8767 : tensor<576x1x3x3xf32>
    %v8769 = stablehlo.add %v8766, %v8768 : tensor<576x1x3x3xf32>
    %v8770 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8771 = stablehlo.add %v8769, %v8770 : tensor<576x1x3x3xf32>
    %v8772 = stablehlo.sqrt %v8771 : tensor<576x1x3x3xf32>
    %v8773 = stablehlo.divide %v8757, %v8772 : tensor<576x1x3x3xf32>
    %v8774 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8775 = stablehlo.multiply %v8774, %b14dWm : tensor<576x1x3x3xf32>
    %v8776 = stablehlo.add %v8775, %v8773 : tensor<576x1x3x3xf32>
    %v8777 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8778 = stablehlo.multiply %v8777, %v8776 : tensor<576x1x3x3xf32>
    %v8779 = stablehlo.subtract %b14dW, %v8778 : tensor<576x1x3x3xf32>
    %v8780 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8781 = stablehlo.multiply %v8780, %b14dg : tensor<576xf32>
    %v8782 = stablehlo.add %v8781, %v2314 : tensor<576xf32>
    %v8783 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8784 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8785 = stablehlo.multiply %v8783, %b14dgv : tensor<576xf32>
    %v8786 = stablehlo.multiply %v8782, %v8782 : tensor<576xf32>
    %v8787 = stablehlo.multiply %v8784, %v8786 : tensor<576xf32>
    %v8788 = stablehlo.add %v8785, %v8787 : tensor<576xf32>
    %v8789 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8790 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8791 = stablehlo.multiply %v8789, %b14dgv : tensor<576xf32>
    %v8792 = stablehlo.multiply %v8782, %v8782 : tensor<576xf32>
    %v8793 = stablehlo.multiply %v8790, %v8792 : tensor<576xf32>
    %v8794 = stablehlo.add %v8791, %v8793 : tensor<576xf32>
    %v8795 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8796 = stablehlo.add %v8794, %v8795 : tensor<576xf32>
    %v8797 = stablehlo.sqrt %v8796 : tensor<576xf32>
    %v8798 = stablehlo.divide %v8782, %v8797 : tensor<576xf32>
    %v8799 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8800 = stablehlo.multiply %v8799, %b14dgm : tensor<576xf32>
    %v8801 = stablehlo.add %v8800, %v8798 : tensor<576xf32>
    %v8802 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8803 = stablehlo.multiply %v8802, %v8801 : tensor<576xf32>
    %v8804 = stablehlo.subtract %b14dg, %v8803 : tensor<576xf32>
    %v8805 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8806 = stablehlo.multiply %v8805, %b14dbt : tensor<576xf32>
    %v8807 = stablehlo.add %v8806, %v2317 : tensor<576xf32>
    %v8808 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8809 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8810 = stablehlo.multiply %v8808, %b14dbtv : tensor<576xf32>
    %v8811 = stablehlo.multiply %v8807, %v8807 : tensor<576xf32>
    %v8812 = stablehlo.multiply %v8809, %v8811 : tensor<576xf32>
    %v8813 = stablehlo.add %v8810, %v8812 : tensor<576xf32>
    %v8814 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8815 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8816 = stablehlo.multiply %v8814, %b14dbtv : tensor<576xf32>
    %v8817 = stablehlo.multiply %v8807, %v8807 : tensor<576xf32>
    %v8818 = stablehlo.multiply %v8815, %v8817 : tensor<576xf32>
    %v8819 = stablehlo.add %v8816, %v8818 : tensor<576xf32>
    %v8820 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8821 = stablehlo.add %v8819, %v8820 : tensor<576xf32>
    %v8822 = stablehlo.sqrt %v8821 : tensor<576xf32>
    %v8823 = stablehlo.divide %v8807, %v8822 : tensor<576xf32>
    %v8824 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8825 = stablehlo.multiply %v8824, %b14dbtm : tensor<576xf32>
    %v8826 = stablehlo.add %v8825, %v8823 : tensor<576xf32>
    %v8827 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8828 = stablehlo.multiply %v8827, %v8826 : tensor<576xf32>
    %v8829 = stablehlo.subtract %b14dbt, %v8828 : tensor<576xf32>
    %v8830 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v8831 = stablehlo.multiply %v8830, %b14pW : tensor<160x576x1x1xf32>
    %v8832 = stablehlo.add %v8831, %v2323 : tensor<160x576x1x1xf32>
    %v8833 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v8834 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v8835 = stablehlo.multiply %v8833, %b14pWv : tensor<160x576x1x1xf32>
    %v8836 = stablehlo.multiply %v8832, %v8832 : tensor<160x576x1x1xf32>
    %v8837 = stablehlo.multiply %v8834, %v8836 : tensor<160x576x1x1xf32>
    %v8838 = stablehlo.add %v8835, %v8837 : tensor<160x576x1x1xf32>
    %v8839 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v8840 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v8841 = stablehlo.multiply %v8839, %b14pWv : tensor<160x576x1x1xf32>
    %v8842 = stablehlo.multiply %v8832, %v8832 : tensor<160x576x1x1xf32>
    %v8843 = stablehlo.multiply %v8840, %v8842 : tensor<160x576x1x1xf32>
    %v8844 = stablehlo.add %v8841, %v8843 : tensor<160x576x1x1xf32>
    %v8845 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v8846 = stablehlo.add %v8844, %v8845 : tensor<160x576x1x1xf32>
    %v8847 = stablehlo.sqrt %v8846 : tensor<160x576x1x1xf32>
    %v8848 = stablehlo.divide %v8832, %v8847 : tensor<160x576x1x1xf32>
    %v8849 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v8850 = stablehlo.multiply %v8849, %b14pWm : tensor<160x576x1x1xf32>
    %v8851 = stablehlo.add %v8850, %v8848 : tensor<160x576x1x1xf32>
    %v8852 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v8853 = stablehlo.multiply %v8852, %v8851 : tensor<160x576x1x1xf32>
    %v8854 = stablehlo.subtract %b14pW, %v8853 : tensor<160x576x1x1xf32>
    %v8855 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8856 = stablehlo.multiply %v8855, %b14pg : tensor<160xf32>
    %v8857 = stablehlo.add %v8856, %v2341 : tensor<160xf32>
    %v8858 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8859 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8860 = stablehlo.multiply %v8858, %b14pgv : tensor<160xf32>
    %v8861 = stablehlo.multiply %v8857, %v8857 : tensor<160xf32>
    %v8862 = stablehlo.multiply %v8859, %v8861 : tensor<160xf32>
    %v8863 = stablehlo.add %v8860, %v8862 : tensor<160xf32>
    %v8864 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8865 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8866 = stablehlo.multiply %v8864, %b14pgv : tensor<160xf32>
    %v8867 = stablehlo.multiply %v8857, %v8857 : tensor<160xf32>
    %v8868 = stablehlo.multiply %v8865, %v8867 : tensor<160xf32>
    %v8869 = stablehlo.add %v8866, %v8868 : tensor<160xf32>
    %v8870 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8871 = stablehlo.add %v8869, %v8870 : tensor<160xf32>
    %v8872 = stablehlo.sqrt %v8871 : tensor<160xf32>
    %v8873 = stablehlo.divide %v8857, %v8872 : tensor<160xf32>
    %v8874 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8875 = stablehlo.multiply %v8874, %b14pgm : tensor<160xf32>
    %v8876 = stablehlo.add %v8875, %v8873 : tensor<160xf32>
    %v8877 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8878 = stablehlo.multiply %v8877, %v8876 : tensor<160xf32>
    %v8879 = stablehlo.subtract %b14pg, %v8878 : tensor<160xf32>
    %v8880 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8881 = stablehlo.multiply %v8880, %b14pbt : tensor<160xf32>
    %v8882 = stablehlo.add %v8881, %v2344 : tensor<160xf32>
    %v8883 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8884 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8885 = stablehlo.multiply %v8883, %b14pbtv : tensor<160xf32>
    %v8886 = stablehlo.multiply %v8882, %v8882 : tensor<160xf32>
    %v8887 = stablehlo.multiply %v8884, %v8886 : tensor<160xf32>
    %v8888 = stablehlo.add %v8885, %v8887 : tensor<160xf32>
    %v8889 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8890 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8891 = stablehlo.multiply %v8889, %b14pbtv : tensor<160xf32>
    %v8892 = stablehlo.multiply %v8882, %v8882 : tensor<160xf32>
    %v8893 = stablehlo.multiply %v8890, %v8892 : tensor<160xf32>
    %v8894 = stablehlo.add %v8891, %v8893 : tensor<160xf32>
    %v8895 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8896 = stablehlo.add %v8894, %v8895 : tensor<160xf32>
    %v8897 = stablehlo.sqrt %v8896 : tensor<160xf32>
    %v8898 = stablehlo.divide %v8882, %v8897 : tensor<160xf32>
    %v8899 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8900 = stablehlo.multiply %v8899, %b14pbtm : tensor<160xf32>
    %v8901 = stablehlo.add %v8900, %v8898 : tensor<160xf32>
    %v8902 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v8903 = stablehlo.multiply %v8902, %v8901 : tensor<160xf32>
    %v8904 = stablehlo.subtract %b14pbt, %v8903 : tensor<160xf32>
    %v8905 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v8906 = stablehlo.multiply %v8905, %b15eW : tensor<960x160x1x1xf32>
    %v8907 = stablehlo.add %v8906, %v2068 : tensor<960x160x1x1xf32>
    %v8908 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v8909 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v8910 = stablehlo.multiply %v8908, %b15eWv : tensor<960x160x1x1xf32>
    %v8911 = stablehlo.multiply %v8907, %v8907 : tensor<960x160x1x1xf32>
    %v8912 = stablehlo.multiply %v8909, %v8911 : tensor<960x160x1x1xf32>
    %v8913 = stablehlo.add %v8910, %v8912 : tensor<960x160x1x1xf32>
    %v8914 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v8915 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v8916 = stablehlo.multiply %v8914, %b15eWv : tensor<960x160x1x1xf32>
    %v8917 = stablehlo.multiply %v8907, %v8907 : tensor<960x160x1x1xf32>
    %v8918 = stablehlo.multiply %v8915, %v8917 : tensor<960x160x1x1xf32>
    %v8919 = stablehlo.add %v8916, %v8918 : tensor<960x160x1x1xf32>
    %v8920 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v8921 = stablehlo.add %v8919, %v8920 : tensor<960x160x1x1xf32>
    %v8922 = stablehlo.sqrt %v8921 : tensor<960x160x1x1xf32>
    %v8923 = stablehlo.divide %v8907, %v8922 : tensor<960x160x1x1xf32>
    %v8924 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v8925 = stablehlo.multiply %v8924, %b15eWm : tensor<960x160x1x1xf32>
    %v8926 = stablehlo.add %v8925, %v8923 : tensor<960x160x1x1xf32>
    %v8927 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v8928 = stablehlo.multiply %v8927, %v8926 : tensor<960x160x1x1xf32>
    %v8929 = stablehlo.subtract %b15eW, %v8928 : tensor<960x160x1x1xf32>
    %v8930 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8931 = stablehlo.multiply %v8930, %b15eg : tensor<960xf32>
    %v8932 = stablehlo.add %v8931, %v2086 : tensor<960xf32>
    %v8933 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8934 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8935 = stablehlo.multiply %v8933, %b15egv : tensor<960xf32>
    %v8936 = stablehlo.multiply %v8932, %v8932 : tensor<960xf32>
    %v8937 = stablehlo.multiply %v8934, %v8936 : tensor<960xf32>
    %v8938 = stablehlo.add %v8935, %v8937 : tensor<960xf32>
    %v8939 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8940 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8941 = stablehlo.multiply %v8939, %b15egv : tensor<960xf32>
    %v8942 = stablehlo.multiply %v8932, %v8932 : tensor<960xf32>
    %v8943 = stablehlo.multiply %v8940, %v8942 : tensor<960xf32>
    %v8944 = stablehlo.add %v8941, %v8943 : tensor<960xf32>
    %v8945 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8946 = stablehlo.add %v8944, %v8945 : tensor<960xf32>
    %v8947 = stablehlo.sqrt %v8946 : tensor<960xf32>
    %v8948 = stablehlo.divide %v8932, %v8947 : tensor<960xf32>
    %v8949 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8950 = stablehlo.multiply %v8949, %b15egm : tensor<960xf32>
    %v8951 = stablehlo.add %v8950, %v8948 : tensor<960xf32>
    %v8952 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8953 = stablehlo.multiply %v8952, %v8951 : tensor<960xf32>
    %v8954 = stablehlo.subtract %b15eg, %v8953 : tensor<960xf32>
    %v8955 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8956 = stablehlo.multiply %v8955, %b15ebt : tensor<960xf32>
    %v8957 = stablehlo.add %v8956, %v2089 : tensor<960xf32>
    %v8958 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8959 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8960 = stablehlo.multiply %v8958, %b15ebtv : tensor<960xf32>
    %v8961 = stablehlo.multiply %v8957, %v8957 : tensor<960xf32>
    %v8962 = stablehlo.multiply %v8959, %v8961 : tensor<960xf32>
    %v8963 = stablehlo.add %v8960, %v8962 : tensor<960xf32>
    %v8964 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8965 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8966 = stablehlo.multiply %v8964, %b15ebtv : tensor<960xf32>
    %v8967 = stablehlo.multiply %v8957, %v8957 : tensor<960xf32>
    %v8968 = stablehlo.multiply %v8965, %v8967 : tensor<960xf32>
    %v8969 = stablehlo.add %v8966, %v8968 : tensor<960xf32>
    %v8970 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8971 = stablehlo.add %v8969, %v8970 : tensor<960xf32>
    %v8972 = stablehlo.sqrt %v8971 : tensor<960xf32>
    %v8973 = stablehlo.divide %v8957, %v8972 : tensor<960xf32>
    %v8974 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8975 = stablehlo.multiply %v8974, %b15ebtm : tensor<960xf32>
    %v8976 = stablehlo.add %v8975, %v8973 : tensor<960xf32>
    %v8977 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v8978 = stablehlo.multiply %v8977, %v8976 : tensor<960xf32>
    %v8979 = stablehlo.subtract %b15ebt, %v8978 : tensor<960xf32>
    %v8980 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v8981 = stablehlo.multiply %v8980, %b15dW : tensor<960x1x3x3xf32>
    %v8982 = stablehlo.add %v8981, %v2095 : tensor<960x1x3x3xf32>
    %v8983 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v8984 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v8985 = stablehlo.multiply %v8983, %b15dWv : tensor<960x1x3x3xf32>
    %v8986 = stablehlo.multiply %v8982, %v8982 : tensor<960x1x3x3xf32>
    %v8987 = stablehlo.multiply %v8984, %v8986 : tensor<960x1x3x3xf32>
    %v8988 = stablehlo.add %v8985, %v8987 : tensor<960x1x3x3xf32>
    %v8989 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v8990 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v8991 = stablehlo.multiply %v8989, %b15dWv : tensor<960x1x3x3xf32>
    %v8992 = stablehlo.multiply %v8982, %v8982 : tensor<960x1x3x3xf32>
    %v8993 = stablehlo.multiply %v8990, %v8992 : tensor<960x1x3x3xf32>
    %v8994 = stablehlo.add %v8991, %v8993 : tensor<960x1x3x3xf32>
    %v8995 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v8996 = stablehlo.add %v8994, %v8995 : tensor<960x1x3x3xf32>
    %v8997 = stablehlo.sqrt %v8996 : tensor<960x1x3x3xf32>
    %v8998 = stablehlo.divide %v8982, %v8997 : tensor<960x1x3x3xf32>
    %v8999 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9000 = stablehlo.multiply %v8999, %b15dWm : tensor<960x1x3x3xf32>
    %v9001 = stablehlo.add %v9000, %v8998 : tensor<960x1x3x3xf32>
    %v9002 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9003 = stablehlo.multiply %v9002, %v9001 : tensor<960x1x3x3xf32>
    %v9004 = stablehlo.subtract %b15dW, %v9003 : tensor<960x1x3x3xf32>
    %v9005 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9006 = stablehlo.multiply %v9005, %b15dg : tensor<960xf32>
    %v9007 = stablehlo.add %v9006, %v2113 : tensor<960xf32>
    %v9008 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9009 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9010 = stablehlo.multiply %v9008, %b15dgv : tensor<960xf32>
    %v9011 = stablehlo.multiply %v9007, %v9007 : tensor<960xf32>
    %v9012 = stablehlo.multiply %v9009, %v9011 : tensor<960xf32>
    %v9013 = stablehlo.add %v9010, %v9012 : tensor<960xf32>
    %v9014 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9015 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9016 = stablehlo.multiply %v9014, %b15dgv : tensor<960xf32>
    %v9017 = stablehlo.multiply %v9007, %v9007 : tensor<960xf32>
    %v9018 = stablehlo.multiply %v9015, %v9017 : tensor<960xf32>
    %v9019 = stablehlo.add %v9016, %v9018 : tensor<960xf32>
    %v9020 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9021 = stablehlo.add %v9019, %v9020 : tensor<960xf32>
    %v9022 = stablehlo.sqrt %v9021 : tensor<960xf32>
    %v9023 = stablehlo.divide %v9007, %v9022 : tensor<960xf32>
    %v9024 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9025 = stablehlo.multiply %v9024, %b15dgm : tensor<960xf32>
    %v9026 = stablehlo.add %v9025, %v9023 : tensor<960xf32>
    %v9027 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9028 = stablehlo.multiply %v9027, %v9026 : tensor<960xf32>
    %v9029 = stablehlo.subtract %b15dg, %v9028 : tensor<960xf32>
    %v9030 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9031 = stablehlo.multiply %v9030, %b15dbt : tensor<960xf32>
    %v9032 = stablehlo.add %v9031, %v2116 : tensor<960xf32>
    %v9033 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9034 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9035 = stablehlo.multiply %v9033, %b15dbtv : tensor<960xf32>
    %v9036 = stablehlo.multiply %v9032, %v9032 : tensor<960xf32>
    %v9037 = stablehlo.multiply %v9034, %v9036 : tensor<960xf32>
    %v9038 = stablehlo.add %v9035, %v9037 : tensor<960xf32>
    %v9039 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9040 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9041 = stablehlo.multiply %v9039, %b15dbtv : tensor<960xf32>
    %v9042 = stablehlo.multiply %v9032, %v9032 : tensor<960xf32>
    %v9043 = stablehlo.multiply %v9040, %v9042 : tensor<960xf32>
    %v9044 = stablehlo.add %v9041, %v9043 : tensor<960xf32>
    %v9045 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9046 = stablehlo.add %v9044, %v9045 : tensor<960xf32>
    %v9047 = stablehlo.sqrt %v9046 : tensor<960xf32>
    %v9048 = stablehlo.divide %v9032, %v9047 : tensor<960xf32>
    %v9049 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9050 = stablehlo.multiply %v9049, %b15dbtm : tensor<960xf32>
    %v9051 = stablehlo.add %v9050, %v9048 : tensor<960xf32>
    %v9052 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9053 = stablehlo.multiply %v9052, %v9051 : tensor<960xf32>
    %v9054 = stablehlo.subtract %b15dbt, %v9053 : tensor<960xf32>
    %v9055 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9056 = stablehlo.multiply %v9055, %b15pW : tensor<160x960x1x1xf32>
    %v9057 = stablehlo.add %v9056, %v2122 : tensor<160x960x1x1xf32>
    %v9058 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9059 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9060 = stablehlo.multiply %v9058, %b15pWv : tensor<160x960x1x1xf32>
    %v9061 = stablehlo.multiply %v9057, %v9057 : tensor<160x960x1x1xf32>
    %v9062 = stablehlo.multiply %v9059, %v9061 : tensor<160x960x1x1xf32>
    %v9063 = stablehlo.add %v9060, %v9062 : tensor<160x960x1x1xf32>
    %v9064 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9065 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9066 = stablehlo.multiply %v9064, %b15pWv : tensor<160x960x1x1xf32>
    %v9067 = stablehlo.multiply %v9057, %v9057 : tensor<160x960x1x1xf32>
    %v9068 = stablehlo.multiply %v9065, %v9067 : tensor<160x960x1x1xf32>
    %v9069 = stablehlo.add %v9066, %v9068 : tensor<160x960x1x1xf32>
    %v9070 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9071 = stablehlo.add %v9069, %v9070 : tensor<160x960x1x1xf32>
    %v9072 = stablehlo.sqrt %v9071 : tensor<160x960x1x1xf32>
    %v9073 = stablehlo.divide %v9057, %v9072 : tensor<160x960x1x1xf32>
    %v9074 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9075 = stablehlo.multiply %v9074, %b15pWm : tensor<160x960x1x1xf32>
    %v9076 = stablehlo.add %v9075, %v9073 : tensor<160x960x1x1xf32>
    %v9077 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9078 = stablehlo.multiply %v9077, %v9076 : tensor<160x960x1x1xf32>
    %v9079 = stablehlo.subtract %b15pW, %v9078 : tensor<160x960x1x1xf32>
    %v9080 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9081 = stablehlo.multiply %v9080, %b15pg : tensor<160xf32>
    %v9082 = stablehlo.add %v9081, %v2140 : tensor<160xf32>
    %v9083 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9084 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9085 = stablehlo.multiply %v9083, %b15pgv : tensor<160xf32>
    %v9086 = stablehlo.multiply %v9082, %v9082 : tensor<160xf32>
    %v9087 = stablehlo.multiply %v9084, %v9086 : tensor<160xf32>
    %v9088 = stablehlo.add %v9085, %v9087 : tensor<160xf32>
    %v9089 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9090 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9091 = stablehlo.multiply %v9089, %b15pgv : tensor<160xf32>
    %v9092 = stablehlo.multiply %v9082, %v9082 : tensor<160xf32>
    %v9093 = stablehlo.multiply %v9090, %v9092 : tensor<160xf32>
    %v9094 = stablehlo.add %v9091, %v9093 : tensor<160xf32>
    %v9095 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9096 = stablehlo.add %v9094, %v9095 : tensor<160xf32>
    %v9097 = stablehlo.sqrt %v9096 : tensor<160xf32>
    %v9098 = stablehlo.divide %v9082, %v9097 : tensor<160xf32>
    %v9099 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9100 = stablehlo.multiply %v9099, %b15pgm : tensor<160xf32>
    %v9101 = stablehlo.add %v9100, %v9098 : tensor<160xf32>
    %v9102 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9103 = stablehlo.multiply %v9102, %v9101 : tensor<160xf32>
    %v9104 = stablehlo.subtract %b15pg, %v9103 : tensor<160xf32>
    %v9105 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9106 = stablehlo.multiply %v9105, %b15pbt : tensor<160xf32>
    %v9107 = stablehlo.add %v9106, %v2143 : tensor<160xf32>
    %v9108 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9109 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9110 = stablehlo.multiply %v9108, %b15pbtv : tensor<160xf32>
    %v9111 = stablehlo.multiply %v9107, %v9107 : tensor<160xf32>
    %v9112 = stablehlo.multiply %v9109, %v9111 : tensor<160xf32>
    %v9113 = stablehlo.add %v9110, %v9112 : tensor<160xf32>
    %v9114 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9115 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9116 = stablehlo.multiply %v9114, %b15pbtv : tensor<160xf32>
    %v9117 = stablehlo.multiply %v9107, %v9107 : tensor<160xf32>
    %v9118 = stablehlo.multiply %v9115, %v9117 : tensor<160xf32>
    %v9119 = stablehlo.add %v9116, %v9118 : tensor<160xf32>
    %v9120 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9121 = stablehlo.add %v9119, %v9120 : tensor<160xf32>
    %v9122 = stablehlo.sqrt %v9121 : tensor<160xf32>
    %v9123 = stablehlo.divide %v9107, %v9122 : tensor<160xf32>
    %v9124 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9125 = stablehlo.multiply %v9124, %b15pbtm : tensor<160xf32>
    %v9126 = stablehlo.add %v9125, %v9123 : tensor<160xf32>
    %v9127 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9128 = stablehlo.multiply %v9127, %v9126 : tensor<160xf32>
    %v9129 = stablehlo.subtract %b15pbt, %v9128 : tensor<160xf32>
    %v9130 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9131 = stablehlo.multiply %v9130, %b16eW : tensor<960x160x1x1xf32>
    %v9132 = stablehlo.add %v9131, %v1870 : tensor<960x160x1x1xf32>
    %v9133 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9134 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9135 = stablehlo.multiply %v9133, %b16eWv : tensor<960x160x1x1xf32>
    %v9136 = stablehlo.multiply %v9132, %v9132 : tensor<960x160x1x1xf32>
    %v9137 = stablehlo.multiply %v9134, %v9136 : tensor<960x160x1x1xf32>
    %v9138 = stablehlo.add %v9135, %v9137 : tensor<960x160x1x1xf32>
    %v9139 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9140 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9141 = stablehlo.multiply %v9139, %b16eWv : tensor<960x160x1x1xf32>
    %v9142 = stablehlo.multiply %v9132, %v9132 : tensor<960x160x1x1xf32>
    %v9143 = stablehlo.multiply %v9140, %v9142 : tensor<960x160x1x1xf32>
    %v9144 = stablehlo.add %v9141, %v9143 : tensor<960x160x1x1xf32>
    %v9145 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9146 = stablehlo.add %v9144, %v9145 : tensor<960x160x1x1xf32>
    %v9147 = stablehlo.sqrt %v9146 : tensor<960x160x1x1xf32>
    %v9148 = stablehlo.divide %v9132, %v9147 : tensor<960x160x1x1xf32>
    %v9149 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9150 = stablehlo.multiply %v9149, %b16eWm : tensor<960x160x1x1xf32>
    %v9151 = stablehlo.add %v9150, %v9148 : tensor<960x160x1x1xf32>
    %v9152 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9153 = stablehlo.multiply %v9152, %v9151 : tensor<960x160x1x1xf32>
    %v9154 = stablehlo.subtract %b16eW, %v9153 : tensor<960x160x1x1xf32>
    %v9155 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9156 = stablehlo.multiply %v9155, %b16eg : tensor<960xf32>
    %v9157 = stablehlo.add %v9156, %v1888 : tensor<960xf32>
    %v9158 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9159 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9160 = stablehlo.multiply %v9158, %b16egv : tensor<960xf32>
    %v9161 = stablehlo.multiply %v9157, %v9157 : tensor<960xf32>
    %v9162 = stablehlo.multiply %v9159, %v9161 : tensor<960xf32>
    %v9163 = stablehlo.add %v9160, %v9162 : tensor<960xf32>
    %v9164 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9165 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9166 = stablehlo.multiply %v9164, %b16egv : tensor<960xf32>
    %v9167 = stablehlo.multiply %v9157, %v9157 : tensor<960xf32>
    %v9168 = stablehlo.multiply %v9165, %v9167 : tensor<960xf32>
    %v9169 = stablehlo.add %v9166, %v9168 : tensor<960xf32>
    %v9170 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9171 = stablehlo.add %v9169, %v9170 : tensor<960xf32>
    %v9172 = stablehlo.sqrt %v9171 : tensor<960xf32>
    %v9173 = stablehlo.divide %v9157, %v9172 : tensor<960xf32>
    %v9174 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9175 = stablehlo.multiply %v9174, %b16egm : tensor<960xf32>
    %v9176 = stablehlo.add %v9175, %v9173 : tensor<960xf32>
    %v9177 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9178 = stablehlo.multiply %v9177, %v9176 : tensor<960xf32>
    %v9179 = stablehlo.subtract %b16eg, %v9178 : tensor<960xf32>
    %v9180 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9181 = stablehlo.multiply %v9180, %b16ebt : tensor<960xf32>
    %v9182 = stablehlo.add %v9181, %v1891 : tensor<960xf32>
    %v9183 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9184 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9185 = stablehlo.multiply %v9183, %b16ebtv : tensor<960xf32>
    %v9186 = stablehlo.multiply %v9182, %v9182 : tensor<960xf32>
    %v9187 = stablehlo.multiply %v9184, %v9186 : tensor<960xf32>
    %v9188 = stablehlo.add %v9185, %v9187 : tensor<960xf32>
    %v9189 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9190 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9191 = stablehlo.multiply %v9189, %b16ebtv : tensor<960xf32>
    %v9192 = stablehlo.multiply %v9182, %v9182 : tensor<960xf32>
    %v9193 = stablehlo.multiply %v9190, %v9192 : tensor<960xf32>
    %v9194 = stablehlo.add %v9191, %v9193 : tensor<960xf32>
    %v9195 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9196 = stablehlo.add %v9194, %v9195 : tensor<960xf32>
    %v9197 = stablehlo.sqrt %v9196 : tensor<960xf32>
    %v9198 = stablehlo.divide %v9182, %v9197 : tensor<960xf32>
    %v9199 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9200 = stablehlo.multiply %v9199, %b16ebtm : tensor<960xf32>
    %v9201 = stablehlo.add %v9200, %v9198 : tensor<960xf32>
    %v9202 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9203 = stablehlo.multiply %v9202, %v9201 : tensor<960xf32>
    %v9204 = stablehlo.subtract %b16ebt, %v9203 : tensor<960xf32>
    %v9205 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9206 = stablehlo.multiply %v9205, %b16dW : tensor<960x1x3x3xf32>
    %v9207 = stablehlo.add %v9206, %v1897 : tensor<960x1x3x3xf32>
    %v9208 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9209 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9210 = stablehlo.multiply %v9208, %b16dWv : tensor<960x1x3x3xf32>
    %v9211 = stablehlo.multiply %v9207, %v9207 : tensor<960x1x3x3xf32>
    %v9212 = stablehlo.multiply %v9209, %v9211 : tensor<960x1x3x3xf32>
    %v9213 = stablehlo.add %v9210, %v9212 : tensor<960x1x3x3xf32>
    %v9214 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9215 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9216 = stablehlo.multiply %v9214, %b16dWv : tensor<960x1x3x3xf32>
    %v9217 = stablehlo.multiply %v9207, %v9207 : tensor<960x1x3x3xf32>
    %v9218 = stablehlo.multiply %v9215, %v9217 : tensor<960x1x3x3xf32>
    %v9219 = stablehlo.add %v9216, %v9218 : tensor<960x1x3x3xf32>
    %v9220 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9221 = stablehlo.add %v9219, %v9220 : tensor<960x1x3x3xf32>
    %v9222 = stablehlo.sqrt %v9221 : tensor<960x1x3x3xf32>
    %v9223 = stablehlo.divide %v9207, %v9222 : tensor<960x1x3x3xf32>
    %v9224 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9225 = stablehlo.multiply %v9224, %b16dWm : tensor<960x1x3x3xf32>
    %v9226 = stablehlo.add %v9225, %v9223 : tensor<960x1x3x3xf32>
    %v9227 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9228 = stablehlo.multiply %v9227, %v9226 : tensor<960x1x3x3xf32>
    %v9229 = stablehlo.subtract %b16dW, %v9228 : tensor<960x1x3x3xf32>
    %v9230 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9231 = stablehlo.multiply %v9230, %b16dg : tensor<960xf32>
    %v9232 = stablehlo.add %v9231, %v1915 : tensor<960xf32>
    %v9233 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9234 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9235 = stablehlo.multiply %v9233, %b16dgv : tensor<960xf32>
    %v9236 = stablehlo.multiply %v9232, %v9232 : tensor<960xf32>
    %v9237 = stablehlo.multiply %v9234, %v9236 : tensor<960xf32>
    %v9238 = stablehlo.add %v9235, %v9237 : tensor<960xf32>
    %v9239 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9240 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9241 = stablehlo.multiply %v9239, %b16dgv : tensor<960xf32>
    %v9242 = stablehlo.multiply %v9232, %v9232 : tensor<960xf32>
    %v9243 = stablehlo.multiply %v9240, %v9242 : tensor<960xf32>
    %v9244 = stablehlo.add %v9241, %v9243 : tensor<960xf32>
    %v9245 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9246 = stablehlo.add %v9244, %v9245 : tensor<960xf32>
    %v9247 = stablehlo.sqrt %v9246 : tensor<960xf32>
    %v9248 = stablehlo.divide %v9232, %v9247 : tensor<960xf32>
    %v9249 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9250 = stablehlo.multiply %v9249, %b16dgm : tensor<960xf32>
    %v9251 = stablehlo.add %v9250, %v9248 : tensor<960xf32>
    %v9252 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9253 = stablehlo.multiply %v9252, %v9251 : tensor<960xf32>
    %v9254 = stablehlo.subtract %b16dg, %v9253 : tensor<960xf32>
    %v9255 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9256 = stablehlo.multiply %v9255, %b16dbt : tensor<960xf32>
    %v9257 = stablehlo.add %v9256, %v1918 : tensor<960xf32>
    %v9258 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9259 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9260 = stablehlo.multiply %v9258, %b16dbtv : tensor<960xf32>
    %v9261 = stablehlo.multiply %v9257, %v9257 : tensor<960xf32>
    %v9262 = stablehlo.multiply %v9259, %v9261 : tensor<960xf32>
    %v9263 = stablehlo.add %v9260, %v9262 : tensor<960xf32>
    %v9264 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9265 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9266 = stablehlo.multiply %v9264, %b16dbtv : tensor<960xf32>
    %v9267 = stablehlo.multiply %v9257, %v9257 : tensor<960xf32>
    %v9268 = stablehlo.multiply %v9265, %v9267 : tensor<960xf32>
    %v9269 = stablehlo.add %v9266, %v9268 : tensor<960xf32>
    %v9270 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9271 = stablehlo.add %v9269, %v9270 : tensor<960xf32>
    %v9272 = stablehlo.sqrt %v9271 : tensor<960xf32>
    %v9273 = stablehlo.divide %v9257, %v9272 : tensor<960xf32>
    %v9274 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9275 = stablehlo.multiply %v9274, %b16dbtm : tensor<960xf32>
    %v9276 = stablehlo.add %v9275, %v9273 : tensor<960xf32>
    %v9277 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9278 = stablehlo.multiply %v9277, %v9276 : tensor<960xf32>
    %v9279 = stablehlo.subtract %b16dbt, %v9278 : tensor<960xf32>
    %v9280 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9281 = stablehlo.multiply %v9280, %b16pW : tensor<160x960x1x1xf32>
    %v9282 = stablehlo.add %v9281, %v1924 : tensor<160x960x1x1xf32>
    %v9283 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9284 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9285 = stablehlo.multiply %v9283, %b16pWv : tensor<160x960x1x1xf32>
    %v9286 = stablehlo.multiply %v9282, %v9282 : tensor<160x960x1x1xf32>
    %v9287 = stablehlo.multiply %v9284, %v9286 : tensor<160x960x1x1xf32>
    %v9288 = stablehlo.add %v9285, %v9287 : tensor<160x960x1x1xf32>
    %v9289 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9290 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9291 = stablehlo.multiply %v9289, %b16pWv : tensor<160x960x1x1xf32>
    %v9292 = stablehlo.multiply %v9282, %v9282 : tensor<160x960x1x1xf32>
    %v9293 = stablehlo.multiply %v9290, %v9292 : tensor<160x960x1x1xf32>
    %v9294 = stablehlo.add %v9291, %v9293 : tensor<160x960x1x1xf32>
    %v9295 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9296 = stablehlo.add %v9294, %v9295 : tensor<160x960x1x1xf32>
    %v9297 = stablehlo.sqrt %v9296 : tensor<160x960x1x1xf32>
    %v9298 = stablehlo.divide %v9282, %v9297 : tensor<160x960x1x1xf32>
    %v9299 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9300 = stablehlo.multiply %v9299, %b16pWm : tensor<160x960x1x1xf32>
    %v9301 = stablehlo.add %v9300, %v9298 : tensor<160x960x1x1xf32>
    %v9302 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9303 = stablehlo.multiply %v9302, %v9301 : tensor<160x960x1x1xf32>
    %v9304 = stablehlo.subtract %b16pW, %v9303 : tensor<160x960x1x1xf32>
    %v9305 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9306 = stablehlo.multiply %v9305, %b16pg : tensor<160xf32>
    %v9307 = stablehlo.add %v9306, %v1942 : tensor<160xf32>
    %v9308 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9309 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9310 = stablehlo.multiply %v9308, %b16pgv : tensor<160xf32>
    %v9311 = stablehlo.multiply %v9307, %v9307 : tensor<160xf32>
    %v9312 = stablehlo.multiply %v9309, %v9311 : tensor<160xf32>
    %v9313 = stablehlo.add %v9310, %v9312 : tensor<160xf32>
    %v9314 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9315 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9316 = stablehlo.multiply %v9314, %b16pgv : tensor<160xf32>
    %v9317 = stablehlo.multiply %v9307, %v9307 : tensor<160xf32>
    %v9318 = stablehlo.multiply %v9315, %v9317 : tensor<160xf32>
    %v9319 = stablehlo.add %v9316, %v9318 : tensor<160xf32>
    %v9320 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9321 = stablehlo.add %v9319, %v9320 : tensor<160xf32>
    %v9322 = stablehlo.sqrt %v9321 : tensor<160xf32>
    %v9323 = stablehlo.divide %v9307, %v9322 : tensor<160xf32>
    %v9324 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9325 = stablehlo.multiply %v9324, %b16pgm : tensor<160xf32>
    %v9326 = stablehlo.add %v9325, %v9323 : tensor<160xf32>
    %v9327 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9328 = stablehlo.multiply %v9327, %v9326 : tensor<160xf32>
    %v9329 = stablehlo.subtract %b16pg, %v9328 : tensor<160xf32>
    %v9330 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9331 = stablehlo.multiply %v9330, %b16pbt : tensor<160xf32>
    %v9332 = stablehlo.add %v9331, %v1945 : tensor<160xf32>
    %v9333 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9334 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9335 = stablehlo.multiply %v9333, %b16pbtv : tensor<160xf32>
    %v9336 = stablehlo.multiply %v9332, %v9332 : tensor<160xf32>
    %v9337 = stablehlo.multiply %v9334, %v9336 : tensor<160xf32>
    %v9338 = stablehlo.add %v9335, %v9337 : tensor<160xf32>
    %v9339 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9340 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9341 = stablehlo.multiply %v9339, %b16pbtv : tensor<160xf32>
    %v9342 = stablehlo.multiply %v9332, %v9332 : tensor<160xf32>
    %v9343 = stablehlo.multiply %v9340, %v9342 : tensor<160xf32>
    %v9344 = stablehlo.add %v9341, %v9343 : tensor<160xf32>
    %v9345 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9346 = stablehlo.add %v9344, %v9345 : tensor<160xf32>
    %v9347 = stablehlo.sqrt %v9346 : tensor<160xf32>
    %v9348 = stablehlo.divide %v9332, %v9347 : tensor<160xf32>
    %v9349 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9350 = stablehlo.multiply %v9349, %b16pbtm : tensor<160xf32>
    %v9351 = stablehlo.add %v9350, %v9348 : tensor<160xf32>
    %v9352 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9353 = stablehlo.multiply %v9352, %v9351 : tensor<160xf32>
    %v9354 = stablehlo.subtract %b16pbt, %v9353 : tensor<160xf32>
    %v9355 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9356 = stablehlo.multiply %v9355, %b17eW : tensor<960x160x1x1xf32>
    %v9357 = stablehlo.add %v9356, %v1672 : tensor<960x160x1x1xf32>
    %v9358 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9359 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9360 = stablehlo.multiply %v9358, %b17eWv : tensor<960x160x1x1xf32>
    %v9361 = stablehlo.multiply %v9357, %v9357 : tensor<960x160x1x1xf32>
    %v9362 = stablehlo.multiply %v9359, %v9361 : tensor<960x160x1x1xf32>
    %v9363 = stablehlo.add %v9360, %v9362 : tensor<960x160x1x1xf32>
    %v9364 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9365 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9366 = stablehlo.multiply %v9364, %b17eWv : tensor<960x160x1x1xf32>
    %v9367 = stablehlo.multiply %v9357, %v9357 : tensor<960x160x1x1xf32>
    %v9368 = stablehlo.multiply %v9365, %v9367 : tensor<960x160x1x1xf32>
    %v9369 = stablehlo.add %v9366, %v9368 : tensor<960x160x1x1xf32>
    %v9370 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9371 = stablehlo.add %v9369, %v9370 : tensor<960x160x1x1xf32>
    %v9372 = stablehlo.sqrt %v9371 : tensor<960x160x1x1xf32>
    %v9373 = stablehlo.divide %v9357, %v9372 : tensor<960x160x1x1xf32>
    %v9374 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9375 = stablehlo.multiply %v9374, %b17eWm : tensor<960x160x1x1xf32>
    %v9376 = stablehlo.add %v9375, %v9373 : tensor<960x160x1x1xf32>
    %v9377 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9378 = stablehlo.multiply %v9377, %v9376 : tensor<960x160x1x1xf32>
    %v9379 = stablehlo.subtract %b17eW, %v9378 : tensor<960x160x1x1xf32>
    %v9380 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9381 = stablehlo.multiply %v9380, %b17eg : tensor<960xf32>
    %v9382 = stablehlo.add %v9381, %v1690 : tensor<960xf32>
    %v9383 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9384 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9385 = stablehlo.multiply %v9383, %b17egv : tensor<960xf32>
    %v9386 = stablehlo.multiply %v9382, %v9382 : tensor<960xf32>
    %v9387 = stablehlo.multiply %v9384, %v9386 : tensor<960xf32>
    %v9388 = stablehlo.add %v9385, %v9387 : tensor<960xf32>
    %v9389 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9390 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9391 = stablehlo.multiply %v9389, %b17egv : tensor<960xf32>
    %v9392 = stablehlo.multiply %v9382, %v9382 : tensor<960xf32>
    %v9393 = stablehlo.multiply %v9390, %v9392 : tensor<960xf32>
    %v9394 = stablehlo.add %v9391, %v9393 : tensor<960xf32>
    %v9395 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9396 = stablehlo.add %v9394, %v9395 : tensor<960xf32>
    %v9397 = stablehlo.sqrt %v9396 : tensor<960xf32>
    %v9398 = stablehlo.divide %v9382, %v9397 : tensor<960xf32>
    %v9399 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9400 = stablehlo.multiply %v9399, %b17egm : tensor<960xf32>
    %v9401 = stablehlo.add %v9400, %v9398 : tensor<960xf32>
    %v9402 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9403 = stablehlo.multiply %v9402, %v9401 : tensor<960xf32>
    %v9404 = stablehlo.subtract %b17eg, %v9403 : tensor<960xf32>
    %v9405 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9406 = stablehlo.multiply %v9405, %b17ebt : tensor<960xf32>
    %v9407 = stablehlo.add %v9406, %v1693 : tensor<960xf32>
    %v9408 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9409 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9410 = stablehlo.multiply %v9408, %b17ebtv : tensor<960xf32>
    %v9411 = stablehlo.multiply %v9407, %v9407 : tensor<960xf32>
    %v9412 = stablehlo.multiply %v9409, %v9411 : tensor<960xf32>
    %v9413 = stablehlo.add %v9410, %v9412 : tensor<960xf32>
    %v9414 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9415 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9416 = stablehlo.multiply %v9414, %b17ebtv : tensor<960xf32>
    %v9417 = stablehlo.multiply %v9407, %v9407 : tensor<960xf32>
    %v9418 = stablehlo.multiply %v9415, %v9417 : tensor<960xf32>
    %v9419 = stablehlo.add %v9416, %v9418 : tensor<960xf32>
    %v9420 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9421 = stablehlo.add %v9419, %v9420 : tensor<960xf32>
    %v9422 = stablehlo.sqrt %v9421 : tensor<960xf32>
    %v9423 = stablehlo.divide %v9407, %v9422 : tensor<960xf32>
    %v9424 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9425 = stablehlo.multiply %v9424, %b17ebtm : tensor<960xf32>
    %v9426 = stablehlo.add %v9425, %v9423 : tensor<960xf32>
    %v9427 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9428 = stablehlo.multiply %v9427, %v9426 : tensor<960xf32>
    %v9429 = stablehlo.subtract %b17ebt, %v9428 : tensor<960xf32>
    %v9430 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9431 = stablehlo.multiply %v9430, %b17dW : tensor<960x1x3x3xf32>
    %v9432 = stablehlo.add %v9431, %v1699 : tensor<960x1x3x3xf32>
    %v9433 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9434 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9435 = stablehlo.multiply %v9433, %b17dWv : tensor<960x1x3x3xf32>
    %v9436 = stablehlo.multiply %v9432, %v9432 : tensor<960x1x3x3xf32>
    %v9437 = stablehlo.multiply %v9434, %v9436 : tensor<960x1x3x3xf32>
    %v9438 = stablehlo.add %v9435, %v9437 : tensor<960x1x3x3xf32>
    %v9439 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9440 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9441 = stablehlo.multiply %v9439, %b17dWv : tensor<960x1x3x3xf32>
    %v9442 = stablehlo.multiply %v9432, %v9432 : tensor<960x1x3x3xf32>
    %v9443 = stablehlo.multiply %v9440, %v9442 : tensor<960x1x3x3xf32>
    %v9444 = stablehlo.add %v9441, %v9443 : tensor<960x1x3x3xf32>
    %v9445 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9446 = stablehlo.add %v9444, %v9445 : tensor<960x1x3x3xf32>
    %v9447 = stablehlo.sqrt %v9446 : tensor<960x1x3x3xf32>
    %v9448 = stablehlo.divide %v9432, %v9447 : tensor<960x1x3x3xf32>
    %v9449 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9450 = stablehlo.multiply %v9449, %b17dWm : tensor<960x1x3x3xf32>
    %v9451 = stablehlo.add %v9450, %v9448 : tensor<960x1x3x3xf32>
    %v9452 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9453 = stablehlo.multiply %v9452, %v9451 : tensor<960x1x3x3xf32>
    %v9454 = stablehlo.subtract %b17dW, %v9453 : tensor<960x1x3x3xf32>
    %v9455 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9456 = stablehlo.multiply %v9455, %b17dg : tensor<960xf32>
    %v9457 = stablehlo.add %v9456, %v1717 : tensor<960xf32>
    %v9458 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9459 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9460 = stablehlo.multiply %v9458, %b17dgv : tensor<960xf32>
    %v9461 = stablehlo.multiply %v9457, %v9457 : tensor<960xf32>
    %v9462 = stablehlo.multiply %v9459, %v9461 : tensor<960xf32>
    %v9463 = stablehlo.add %v9460, %v9462 : tensor<960xf32>
    %v9464 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9465 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9466 = stablehlo.multiply %v9464, %b17dgv : tensor<960xf32>
    %v9467 = stablehlo.multiply %v9457, %v9457 : tensor<960xf32>
    %v9468 = stablehlo.multiply %v9465, %v9467 : tensor<960xf32>
    %v9469 = stablehlo.add %v9466, %v9468 : tensor<960xf32>
    %v9470 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9471 = stablehlo.add %v9469, %v9470 : tensor<960xf32>
    %v9472 = stablehlo.sqrt %v9471 : tensor<960xf32>
    %v9473 = stablehlo.divide %v9457, %v9472 : tensor<960xf32>
    %v9474 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9475 = stablehlo.multiply %v9474, %b17dgm : tensor<960xf32>
    %v9476 = stablehlo.add %v9475, %v9473 : tensor<960xf32>
    %v9477 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9478 = stablehlo.multiply %v9477, %v9476 : tensor<960xf32>
    %v9479 = stablehlo.subtract %b17dg, %v9478 : tensor<960xf32>
    %v9480 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9481 = stablehlo.multiply %v9480, %b17dbt : tensor<960xf32>
    %v9482 = stablehlo.add %v9481, %v1720 : tensor<960xf32>
    %v9483 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9484 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9485 = stablehlo.multiply %v9483, %b17dbtv : tensor<960xf32>
    %v9486 = stablehlo.multiply %v9482, %v9482 : tensor<960xf32>
    %v9487 = stablehlo.multiply %v9484, %v9486 : tensor<960xf32>
    %v9488 = stablehlo.add %v9485, %v9487 : tensor<960xf32>
    %v9489 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9490 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9491 = stablehlo.multiply %v9489, %b17dbtv : tensor<960xf32>
    %v9492 = stablehlo.multiply %v9482, %v9482 : tensor<960xf32>
    %v9493 = stablehlo.multiply %v9490, %v9492 : tensor<960xf32>
    %v9494 = stablehlo.add %v9491, %v9493 : tensor<960xf32>
    %v9495 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9496 = stablehlo.add %v9494, %v9495 : tensor<960xf32>
    %v9497 = stablehlo.sqrt %v9496 : tensor<960xf32>
    %v9498 = stablehlo.divide %v9482, %v9497 : tensor<960xf32>
    %v9499 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9500 = stablehlo.multiply %v9499, %b17dbtm : tensor<960xf32>
    %v9501 = stablehlo.add %v9500, %v9498 : tensor<960xf32>
    %v9502 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9503 = stablehlo.multiply %v9502, %v9501 : tensor<960xf32>
    %v9504 = stablehlo.subtract %b17dbt, %v9503 : tensor<960xf32>
    %v9505 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9506 = stablehlo.multiply %v9505, %b17pW : tensor<320x960x1x1xf32>
    %v9507 = stablehlo.add %v9506, %v1726 : tensor<320x960x1x1xf32>
    %v9508 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9509 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9510 = stablehlo.multiply %v9508, %b17pWv : tensor<320x960x1x1xf32>
    %v9511 = stablehlo.multiply %v9507, %v9507 : tensor<320x960x1x1xf32>
    %v9512 = stablehlo.multiply %v9509, %v9511 : tensor<320x960x1x1xf32>
    %v9513 = stablehlo.add %v9510, %v9512 : tensor<320x960x1x1xf32>
    %v9514 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9515 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9516 = stablehlo.multiply %v9514, %b17pWv : tensor<320x960x1x1xf32>
    %v9517 = stablehlo.multiply %v9507, %v9507 : tensor<320x960x1x1xf32>
    %v9518 = stablehlo.multiply %v9515, %v9517 : tensor<320x960x1x1xf32>
    %v9519 = stablehlo.add %v9516, %v9518 : tensor<320x960x1x1xf32>
    %v9520 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9521 = stablehlo.add %v9519, %v9520 : tensor<320x960x1x1xf32>
    %v9522 = stablehlo.sqrt %v9521 : tensor<320x960x1x1xf32>
    %v9523 = stablehlo.divide %v9507, %v9522 : tensor<320x960x1x1xf32>
    %v9524 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9525 = stablehlo.multiply %v9524, %b17pWm : tensor<320x960x1x1xf32>
    %v9526 = stablehlo.add %v9525, %v9523 : tensor<320x960x1x1xf32>
    %v9527 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9528 = stablehlo.multiply %v9527, %v9526 : tensor<320x960x1x1xf32>
    %v9529 = stablehlo.subtract %b17pW, %v9528 : tensor<320x960x1x1xf32>
    %v9530 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9531 = stablehlo.multiply %v9530, %b17pg : tensor<320xf32>
    %v9532 = stablehlo.add %v9531, %v1744 : tensor<320xf32>
    %v9533 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9534 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9535 = stablehlo.multiply %v9533, %b17pgv : tensor<320xf32>
    %v9536 = stablehlo.multiply %v9532, %v9532 : tensor<320xf32>
    %v9537 = stablehlo.multiply %v9534, %v9536 : tensor<320xf32>
    %v9538 = stablehlo.add %v9535, %v9537 : tensor<320xf32>
    %v9539 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9540 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9541 = stablehlo.multiply %v9539, %b17pgv : tensor<320xf32>
    %v9542 = stablehlo.multiply %v9532, %v9532 : tensor<320xf32>
    %v9543 = stablehlo.multiply %v9540, %v9542 : tensor<320xf32>
    %v9544 = stablehlo.add %v9541, %v9543 : tensor<320xf32>
    %v9545 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9546 = stablehlo.add %v9544, %v9545 : tensor<320xf32>
    %v9547 = stablehlo.sqrt %v9546 : tensor<320xf32>
    %v9548 = stablehlo.divide %v9532, %v9547 : tensor<320xf32>
    %v9549 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9550 = stablehlo.multiply %v9549, %b17pgm : tensor<320xf32>
    %v9551 = stablehlo.add %v9550, %v9548 : tensor<320xf32>
    %v9552 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9553 = stablehlo.multiply %v9552, %v9551 : tensor<320xf32>
    %v9554 = stablehlo.subtract %b17pg, %v9553 : tensor<320xf32>
    %v9555 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9556 = stablehlo.multiply %v9555, %b17pbt : tensor<320xf32>
    %v9557 = stablehlo.add %v9556, %v1747 : tensor<320xf32>
    %v9558 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9559 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9560 = stablehlo.multiply %v9558, %b17pbtv : tensor<320xf32>
    %v9561 = stablehlo.multiply %v9557, %v9557 : tensor<320xf32>
    %v9562 = stablehlo.multiply %v9559, %v9561 : tensor<320xf32>
    %v9563 = stablehlo.add %v9560, %v9562 : tensor<320xf32>
    %v9564 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9565 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9566 = stablehlo.multiply %v9564, %b17pbtv : tensor<320xf32>
    %v9567 = stablehlo.multiply %v9557, %v9557 : tensor<320xf32>
    %v9568 = stablehlo.multiply %v9565, %v9567 : tensor<320xf32>
    %v9569 = stablehlo.add %v9566, %v9568 : tensor<320xf32>
    %v9570 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9571 = stablehlo.add %v9569, %v9570 : tensor<320xf32>
    %v9572 = stablehlo.sqrt %v9571 : tensor<320xf32>
    %v9573 = stablehlo.divide %v9557, %v9572 : tensor<320xf32>
    %v9574 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9575 = stablehlo.multiply %v9574, %b17pbtm : tensor<320xf32>
    %v9576 = stablehlo.add %v9575, %v9573 : tensor<320xf32>
    %v9577 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9578 = stablehlo.multiply %v9577, %v9576 : tensor<320xf32>
    %v9579 = stablehlo.subtract %b17pbt, %v9578 : tensor<320xf32>
    %v9580 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9581 = stablehlo.multiply %v9580, %hW : tensor<1280x320x1x1xf32>
    %v9582 = stablehlo.add %v9581, %v1529 : tensor<1280x320x1x1xf32>
    %v9583 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9584 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9585 = stablehlo.multiply %v9583, %hWv : tensor<1280x320x1x1xf32>
    %v9586 = stablehlo.multiply %v9582, %v9582 : tensor<1280x320x1x1xf32>
    %v9587 = stablehlo.multiply %v9584, %v9586 : tensor<1280x320x1x1xf32>
    %v9588 = stablehlo.add %v9585, %v9587 : tensor<1280x320x1x1xf32>
    %v9589 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9590 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9591 = stablehlo.multiply %v9589, %hWv : tensor<1280x320x1x1xf32>
    %v9592 = stablehlo.multiply %v9582, %v9582 : tensor<1280x320x1x1xf32>
    %v9593 = stablehlo.multiply %v9590, %v9592 : tensor<1280x320x1x1xf32>
    %v9594 = stablehlo.add %v9591, %v9593 : tensor<1280x320x1x1xf32>
    %v9595 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9596 = stablehlo.add %v9594, %v9595 : tensor<1280x320x1x1xf32>
    %v9597 = stablehlo.sqrt %v9596 : tensor<1280x320x1x1xf32>
    %v9598 = stablehlo.divide %v9582, %v9597 : tensor<1280x320x1x1xf32>
    %v9599 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9600 = stablehlo.multiply %v9599, %hWm : tensor<1280x320x1x1xf32>
    %v9601 = stablehlo.add %v9600, %v9598 : tensor<1280x320x1x1xf32>
    %v9602 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9603 = stablehlo.multiply %v9602, %v9601 : tensor<1280x320x1x1xf32>
    %v9604 = stablehlo.subtract %hW, %v9603 : tensor<1280x320x1x1xf32>
    %v9605 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9606 = stablehlo.multiply %v9605, %hg : tensor<1280xf32>
    %v9607 = stablehlo.add %v9606, %v1547 : tensor<1280xf32>
    %v9608 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9609 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9610 = stablehlo.multiply %v9608, %hgv : tensor<1280xf32>
    %v9611 = stablehlo.multiply %v9607, %v9607 : tensor<1280xf32>
    %v9612 = stablehlo.multiply %v9609, %v9611 : tensor<1280xf32>
    %v9613 = stablehlo.add %v9610, %v9612 : tensor<1280xf32>
    %v9614 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9615 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9616 = stablehlo.multiply %v9614, %hgv : tensor<1280xf32>
    %v9617 = stablehlo.multiply %v9607, %v9607 : tensor<1280xf32>
    %v9618 = stablehlo.multiply %v9615, %v9617 : tensor<1280xf32>
    %v9619 = stablehlo.add %v9616, %v9618 : tensor<1280xf32>
    %v9620 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9621 = stablehlo.add %v9619, %v9620 : tensor<1280xf32>
    %v9622 = stablehlo.sqrt %v9621 : tensor<1280xf32>
    %v9623 = stablehlo.divide %v9607, %v9622 : tensor<1280xf32>
    %v9624 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9625 = stablehlo.multiply %v9624, %hgm : tensor<1280xf32>
    %v9626 = stablehlo.add %v9625, %v9623 : tensor<1280xf32>
    %v9627 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9628 = stablehlo.multiply %v9627, %v9626 : tensor<1280xf32>
    %v9629 = stablehlo.subtract %hg, %v9628 : tensor<1280xf32>
    %v9630 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9631 = stablehlo.multiply %v9630, %hbt : tensor<1280xf32>
    %v9632 = stablehlo.add %v9631, %v1550 : tensor<1280xf32>
    %v9633 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9634 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9635 = stablehlo.multiply %v9633, %hbtv : tensor<1280xf32>
    %v9636 = stablehlo.multiply %v9632, %v9632 : tensor<1280xf32>
    %v9637 = stablehlo.multiply %v9634, %v9636 : tensor<1280xf32>
    %v9638 = stablehlo.add %v9635, %v9637 : tensor<1280xf32>
    %v9639 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9640 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9641 = stablehlo.multiply %v9639, %hbtv : tensor<1280xf32>
    %v9642 = stablehlo.multiply %v9632, %v9632 : tensor<1280xf32>
    %v9643 = stablehlo.multiply %v9640, %v9642 : tensor<1280xf32>
    %v9644 = stablehlo.add %v9641, %v9643 : tensor<1280xf32>
    %v9645 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9646 = stablehlo.add %v9644, %v9645 : tensor<1280xf32>
    %v9647 = stablehlo.sqrt %v9646 : tensor<1280xf32>
    %v9648 = stablehlo.divide %v9632, %v9647 : tensor<1280xf32>
    %v9649 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9650 = stablehlo.multiply %v9649, %hbtm : tensor<1280xf32>
    %v9651 = stablehlo.add %v9650, %v9648 : tensor<1280xf32>
    %v9652 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9653 = stablehlo.multiply %v9652, %v9651 : tensor<1280xf32>
    %v9654 = stablehlo.subtract %hbt, %v9653 : tensor<1280xf32>
    %v9655 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9656 = stablehlo.multiply %v9655, %Wd : tensor<1280x10xf32>
    %v9657 = stablehlo.add %v9656, %v1476 : tensor<1280x10xf32>
    %v9658 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9659 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9660 = stablehlo.multiply %v9658, %Wdv : tensor<1280x10xf32>
    %v9661 = stablehlo.multiply %v9657, %v9657 : tensor<1280x10xf32>
    %v9662 = stablehlo.multiply %v9659, %v9661 : tensor<1280x10xf32>
    %v9663 = stablehlo.add %v9660, %v9662 : tensor<1280x10xf32>
    %v9664 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9665 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9666 = stablehlo.multiply %v9664, %Wdv : tensor<1280x10xf32>
    %v9667 = stablehlo.multiply %v9657, %v9657 : tensor<1280x10xf32>
    %v9668 = stablehlo.multiply %v9665, %v9667 : tensor<1280x10xf32>
    %v9669 = stablehlo.add %v9666, %v9668 : tensor<1280x10xf32>
    %v9670 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9671 = stablehlo.add %v9669, %v9670 : tensor<1280x10xf32>
    %v9672 = stablehlo.sqrt %v9671 : tensor<1280x10xf32>
    %v9673 = stablehlo.divide %v9657, %v9672 : tensor<1280x10xf32>
    %v9674 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9675 = stablehlo.multiply %v9674, %Wdm : tensor<1280x10xf32>
    %v9676 = stablehlo.add %v9675, %v9673 : tensor<1280x10xf32>
    %v9677 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9678 = stablehlo.multiply %v9677, %v9676 : tensor<1280x10xf32>
    %v9679 = stablehlo.subtract %Wd, %v9678 : tensor<1280x10xf32>
    %v9680 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9681 = stablehlo.multiply %v9680, %bd : tensor<10xf32>
    %v9682 = stablehlo.add %v9681, %v1478 : tensor<10xf32>
    %v9683 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9684 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9685 = stablehlo.multiply %v9683, %bdv : tensor<10xf32>
    %v9686 = stablehlo.multiply %v9682, %v9682 : tensor<10xf32>
    %v9687 = stablehlo.multiply %v9684, %v9686 : tensor<10xf32>
    %v9688 = stablehlo.add %v9685, %v9687 : tensor<10xf32>
    %v9689 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9690 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9691 = stablehlo.multiply %v9689, %bdv : tensor<10xf32>
    %v9692 = stablehlo.multiply %v9682, %v9682 : tensor<10xf32>
    %v9693 = stablehlo.multiply %v9690, %v9692 : tensor<10xf32>
    %v9694 = stablehlo.add %v9691, %v9693 : tensor<10xf32>
    %v9695 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9696 = stablehlo.add %v9694, %v9695 : tensor<10xf32>
    %v9697 = stablehlo.sqrt %v9696 : tensor<10xf32>
    %v9698 = stablehlo.divide %v9682, %v9697 : tensor<10xf32>
    %v9699 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9700 = stablehlo.multiply %v9699, %bdm : tensor<10xf32>
    %v9701 = stablehlo.add %v9700, %v9698 : tensor<10xf32>
    %v9702 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9703 = stablehlo.multiply %v9702, %v9701 : tensor<10xf32>
    %v9704 = stablehlo.subtract %bd, %v9703 : tensor<10xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1464 : tensor<32x10xf32>
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
    return %v5779, %v5804, %v5829, %v5854, %v5879, %v5904, %v5929, %v5954, %v5979, %v6004, %v6029, %v6054, %v6079, %v6104, %v6129, %v6154, %v6179, %v6204, %v6229, %v6254, %v6279, %v6304, %v6329, %v6354, %v6379, %v6404, %v6429, %v6454, %v6479, %v6504, %v6529, %v6554, %v6579, %v6604, %v6629, %v6654, %v6679, %v6704, %v6729, %v6754, %v6779, %v6804, %v6829, %v6854, %v6879, %v6904, %v6929, %v6954, %v6979, %v7004, %v7029, %v7054, %v7079, %v7104, %v7129, %v7154, %v7179, %v7204, %v7229, %v7254, %v7279, %v7304, %v7329, %v7354, %v7379, %v7404, %v7429, %v7454, %v7479, %v7504, %v7529, %v7554, %v7579, %v7604, %v7629, %v7654, %v7679, %v7704, %v7729, %v7754, %v7779, %v7804, %v7829, %v7854, %v7879, %v7904, %v7929, %v7954, %v7979, %v8004, %v8029, %v8054, %v8079, %v8104, %v8129, %v8154, %v8179, %v8204, %v8229, %v8254, %v8279, %v8304, %v8329, %v8354, %v8379, %v8404, %v8429, %v8454, %v8479, %v8504, %v8529, %v8554, %v8579, %v8604, %v8629, %v8654, %v8679, %v8704, %v8729, %v8754, %v8779, %v8804, %v8829, %v8854, %v8879, %v8904, %v8929, %v8954, %v8979, %v9004, %v9029, %v9054, %v9079, %v9104, %v9129, %v9154, %v9179, %v9204, %v9229, %v9254, %v9279, %v9304, %v9329, %v9354, %v9379, %v9404, %v9429, %v9454, %v9479, %v9504, %v9529, %v9554, %v9579, %v9604, %v9629, %v9654, %v9679, %v9704, %v5776, %v5801, %v5826, %v5851, %v5876, %v5901, %v5926, %v5951, %v5976, %v6001, %v6026, %v6051, %v6076, %v6101, %v6126, %v6151, %v6176, %v6201, %v6226, %v6251, %v6276, %v6301, %v6326, %v6351, %v6376, %v6401, %v6426, %v6451, %v6476, %v6501, %v6526, %v6551, %v6576, %v6601, %v6626, %v6651, %v6676, %v6701, %v6726, %v6751, %v6776, %v6801, %v6826, %v6851, %v6876, %v6901, %v6926, %v6951, %v6976, %v7001, %v7026, %v7051, %v7076, %v7101, %v7126, %v7151, %v7176, %v7201, %v7226, %v7251, %v7276, %v7301, %v7326, %v7351, %v7376, %v7401, %v7426, %v7451, %v7476, %v7501, %v7526, %v7551, %v7576, %v7601, %v7626, %v7651, %v7676, %v7701, %v7726, %v7751, %v7776, %v7801, %v7826, %v7851, %v7876, %v7901, %v7926, %v7951, %v7976, %v8001, %v8026, %v8051, %v8076, %v8101, %v8126, %v8151, %v8176, %v8201, %v8226, %v8251, %v8276, %v8301, %v8326, %v8351, %v8376, %v8401, %v8426, %v8451, %v8476, %v8501, %v8526, %v8551, %v8576, %v8601, %v8626, %v8651, %v8676, %v8701, %v8726, %v8751, %v8776, %v8801, %v8826, %v8851, %v8876, %v8901, %v8926, %v8951, %v8976, %v9001, %v9026, %v9051, %v9076, %v9101, %v9126, %v9151, %v9176, %v9201, %v9226, %v9251, %v9276, %v9301, %v9326, %v9351, %v9376, %v9401, %v9426, %v9451, %v9476, %v9501, %v9526, %v9551, %v9576, %v9601, %v9626, %v9651, %v9676, %v9701, %v5763, %v5788, %v5813, %v5838, %v5863, %v5888, %v5913, %v5938, %v5963, %v5988, %v6013, %v6038, %v6063, %v6088, %v6113, %v6138, %v6163, %v6188, %v6213, %v6238, %v6263, %v6288, %v6313, %v6338, %v6363, %v6388, %v6413, %v6438, %v6463, %v6488, %v6513, %v6538, %v6563, %v6588, %v6613, %v6638, %v6663, %v6688, %v6713, %v6738, %v6763, %v6788, %v6813, %v6838, %v6863, %v6888, %v6913, %v6938, %v6963, %v6988, %v7013, %v7038, %v7063, %v7088, %v7113, %v7138, %v7163, %v7188, %v7213, %v7238, %v7263, %v7288, %v7313, %v7338, %v7363, %v7388, %v7413, %v7438, %v7463, %v7488, %v7513, %v7538, %v7563, %v7588, %v7613, %v7638, %v7663, %v7688, %v7713, %v7738, %v7763, %v7788, %v7813, %v7838, %v7863, %v7888, %v7913, %v7938, %v7963, %v7988, %v8013, %v8038, %v8063, %v8088, %v8113, %v8138, %v8163, %v8188, %v8213, %v8238, %v8263, %v8288, %v8313, %v8338, %v8363, %v8388, %v8413, %v8438, %v8463, %v8488, %v8513, %v8538, %v8563, %v8588, %v8613, %v8638, %v8663, %v8688, %v8713, %v8738, %v8763, %v8788, %v8813, %v8838, %v8863, %v8888, %v8913, %v8938, %v8963, %v8988, %v9013, %v9038, %v9063, %v9088, %v9113, %v9138, %v9163, %v9188, %v9213, %v9238, %v9263, %v9288, %v9313, %v9338, %v9363, %v9388, %v9413, %v9438, %v9463, %v9488, %v9513, %v9538, %v9563, %v9588, %v9613, %v9638, %v9663, %v9688, %loss, %bc1, %bc2, %v4927, %v4938, %v4943, %v4954, %v4959, %v4970, %v4975, %v4986, %v4991, %v5002, %v5007, %v5018, %v5023, %v5034, %v5039, %v5050, %v5055, %v5066, %v5071, %v5082, %v5087, %v5098, %v5103, %v5114, %v5119, %v5130, %v5135, %v5146, %v5151, %v5162, %v5167, %v5178, %v5183, %v5194, %v5199, %v5210, %v5215, %v5226, %v5231, %v5242, %v5247, %v5258, %v5263, %v5274, %v5279, %v5290, %v5295, %v5306, %v5311, %v5322, %v5327, %v5338, %v5343, %v5354, %v5359, %v5370, %v5375, %v5386, %v5391, %v5402, %v5407, %v5418, %v5423, %v5434, %v5439, %v5450, %v5455, %v5466, %v5471, %v5482, %v5487, %v5498, %v5503, %v5514, %v5519, %v5530, %v5535, %v5546, %v5551, %v5562, %v5567, %v5578, %v5583, %v5594, %v5599, %v5610, %v5615, %v5626, %v5631, %v5642, %v5647, %v5658, %v5663, %v5674, %v5679, %v5690, %v5695, %v5706, %v5711, %v5722, %v5727, %v5738, %v5743, %v5754 : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280xf32>, tensor<1280xf32>
  }
}

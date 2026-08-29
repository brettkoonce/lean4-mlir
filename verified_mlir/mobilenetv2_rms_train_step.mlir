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
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
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
    %v25 = stablehlo.reshape %v24 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %v27 = stablehlo.constant dense<6.0> : tensor<32x32x112x112xf32>
    %v28 = stablehlo.maximum %v25, %v26 : tensor<32x32x112x112xf32>
    %v29 = stablehlo.minimum %v28, %v27 : tensor<32x32x112x112xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v32 = stablehlo.convolution(%v31, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v33 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<32x32x112x112xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<f32>
    %v38 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v39 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v40 = stablehlo.reduce(%v36 init: %v37) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v41 = stablehlo.broadcast_in_dim %v40, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v42 = stablehlo.divide %v41, %v38 : tensor<32x32x112x112xf32>
    %v43 = stablehlo.subtract %v36, %v42 : tensor<32x32x112x112xf32>
    %v44 = stablehlo.multiply %v43, %v43 : tensor<32x32x112x112xf32>
    %v45 = stablehlo.reduce(%v44 init: %v37) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v46 = stablehlo.broadcast_in_dim %v45, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v47 = stablehlo.divide %v46, %v38 : tensor<32x32x112x112xf32>
    %v48 = stablehlo.add %v47, %v39 : tensor<32x32x112x112xf32>
    %v49 = stablehlo.rsqrt %v48 : tensor<32x32x112x112xf32>
    %v50 = stablehlo.multiply %v43, %v49 : tensor<32x32x112x112xf32>
    %v51 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v52 = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v53 = stablehlo.multiply %v50, %v51 : tensor<32x32x112x112xf32>
    %v54 = stablehlo.add %v53, %v52 : tensor<32x32x112x112xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v57 = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %v58 = stablehlo.constant dense<6.0> : tensor<32x32x112x112xf32>
    %v59 = stablehlo.maximum %v56, %v57 : tensor<32x32x112x112xf32>
    %v60 = stablehlo.minimum %v59, %v58 : tensor<32x32x112x112xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v63 = stablehlo.convolution(%v62, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v64 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<32x16x112x112xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v70 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v71 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v72 = stablehlo.broadcast_in_dim %v71, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v73 = stablehlo.divide %v72, %v69 : tensor<32x16x112x112xf32>
    %v74 = stablehlo.subtract %v67, %v73 : tensor<32x16x112x112xf32>
    %v75 = stablehlo.multiply %v74, %v74 : tensor<32x16x112x112xf32>
    %v76 = stablehlo.reduce(%v75 init: %v68) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v77 = stablehlo.broadcast_in_dim %v76, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v78 = stablehlo.divide %v77, %v69 : tensor<32x16x112x112xf32>
    %v79 = stablehlo.add %v78, %v70 : tensor<32x16x112x112xf32>
    %v80 = stablehlo.rsqrt %v79 : tensor<32x16x112x112xf32>
    %v81 = stablehlo.multiply %v74, %v80 : tensor<32x16x112x112xf32>
    %v82 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v83 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v84 = stablehlo.multiply %v81, %v82 : tensor<32x16x112x112xf32>
    %v85 = stablehlo.add %v84, %v83 : tensor<32x16x112x112xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v88 = stablehlo.convolution(%v87, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v89 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v90 = stablehlo.add %v88, %v89 : tensor<32x96x112x112xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v93 = stablehlo.constant dense<0.0> : tensor<f32>
    %v94 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v95 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v96 = stablehlo.reduce(%v92 init: %v93) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v97 = stablehlo.broadcast_in_dim %v96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v98 = stablehlo.divide %v97, %v94 : tensor<32x96x112x112xf32>
    %v99 = stablehlo.subtract %v92, %v98 : tensor<32x96x112x112xf32>
    %v100 = stablehlo.multiply %v99, %v99 : tensor<32x96x112x112xf32>
    %v101 = stablehlo.reduce(%v100 init: %v93) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v102 = stablehlo.broadcast_in_dim %v101, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v103 = stablehlo.divide %v102, %v94 : tensor<32x96x112x112xf32>
    %v104 = stablehlo.add %v103, %v95 : tensor<32x96x112x112xf32>
    %v105 = stablehlo.rsqrt %v104 : tensor<32x96x112x112xf32>
    %v106 = stablehlo.multiply %v99, %v105 : tensor<32x96x112x112xf32>
    %v107 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v108 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v109 = stablehlo.multiply %v106, %v107 : tensor<32x96x112x112xf32>
    %v110 = stablehlo.add %v109, %v108 : tensor<32x96x112x112xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v113 = stablehlo.constant dense<0.0> : tensor<32x96x112x112xf32>
    %v114 = stablehlo.constant dense<6.0> : tensor<32x96x112x112xf32>
    %v115 = stablehlo.maximum %v112, %v113 : tensor<32x96x112x112xf32>
    %v116 = stablehlo.minimum %v115, %v114 : tensor<32x96x112x112xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v119 = stablehlo.convolution(%v118, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v120 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v121 = stablehlo.add %v119, %v120 : tensor<32x96x56x56xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v125 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v126 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v127 = stablehlo.reduce(%v123 init: %v124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v128 = stablehlo.broadcast_in_dim %v127, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v129 = stablehlo.divide %v128, %v125 : tensor<32x96x56x56xf32>
    %v130 = stablehlo.subtract %v123, %v129 : tensor<32x96x56x56xf32>
    %v131 = stablehlo.multiply %v130, %v130 : tensor<32x96x56x56xf32>
    %v132 = stablehlo.reduce(%v131 init: %v124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v133 = stablehlo.broadcast_in_dim %v132, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v134 = stablehlo.divide %v133, %v125 : tensor<32x96x56x56xf32>
    %v135 = stablehlo.add %v134, %v126 : tensor<32x96x56x56xf32>
    %v136 = stablehlo.rsqrt %v135 : tensor<32x96x56x56xf32>
    %v137 = stablehlo.multiply %v130, %v136 : tensor<32x96x56x56xf32>
    %v138 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v139 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v140 = stablehlo.multiply %v137, %v138 : tensor<32x96x56x56xf32>
    %v141 = stablehlo.add %v140, %v139 : tensor<32x96x56x56xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v144 = stablehlo.constant dense<0.0> : tensor<32x96x56x56xf32>
    %v145 = stablehlo.constant dense<6.0> : tensor<32x96x56x56xf32>
    %v146 = stablehlo.maximum %v143, %v144 : tensor<32x96x56x56xf32>
    %v147 = stablehlo.minimum %v146, %v145 : tensor<32x96x56x56xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v150 = stablehlo.convolution(%v149, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v151 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v152 = stablehlo.add %v150, %v151 : tensor<32x24x56x56xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v156 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v157 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v158 = stablehlo.reduce(%v154 init: %v155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v159 = stablehlo.broadcast_in_dim %v158, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v160 = stablehlo.divide %v159, %v156 : tensor<32x24x56x56xf32>
    %v161 = stablehlo.subtract %v154, %v160 : tensor<32x24x56x56xf32>
    %v162 = stablehlo.multiply %v161, %v161 : tensor<32x24x56x56xf32>
    %v163 = stablehlo.reduce(%v162 init: %v155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v164 = stablehlo.broadcast_in_dim %v163, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v165 = stablehlo.divide %v164, %v156 : tensor<32x24x56x56xf32>
    %v166 = stablehlo.add %v165, %v157 : tensor<32x24x56x56xf32>
    %v167 = stablehlo.rsqrt %v166 : tensor<32x24x56x56xf32>
    %v168 = stablehlo.multiply %v161, %v167 : tensor<32x24x56x56xf32>
    %v169 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v170 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v171 = stablehlo.multiply %v168, %v169 : tensor<32x24x56x56xf32>
    %v172 = stablehlo.add %v171, %v170 : tensor<32x24x56x56xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v175 = stablehlo.convolution(%v174, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v176 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v177 = stablehlo.add %v175, %v176 : tensor<32x144x56x56xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v181 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v182 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v183 = stablehlo.reduce(%v179 init: %v180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v184 = stablehlo.broadcast_in_dim %v183, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v185 = stablehlo.divide %v184, %v181 : tensor<32x144x56x56xf32>
    %v186 = stablehlo.subtract %v179, %v185 : tensor<32x144x56x56xf32>
    %v187 = stablehlo.multiply %v186, %v186 : tensor<32x144x56x56xf32>
    %v188 = stablehlo.reduce(%v187 init: %v180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v189 = stablehlo.broadcast_in_dim %v188, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v190 = stablehlo.divide %v189, %v181 : tensor<32x144x56x56xf32>
    %v191 = stablehlo.add %v190, %v182 : tensor<32x144x56x56xf32>
    %v192 = stablehlo.rsqrt %v191 : tensor<32x144x56x56xf32>
    %v193 = stablehlo.multiply %v186, %v192 : tensor<32x144x56x56xf32>
    %v194 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v195 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v196 = stablehlo.multiply %v193, %v194 : tensor<32x144x56x56xf32>
    %v197 = stablehlo.add %v196, %v195 : tensor<32x144x56x56xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v200 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v201 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v202 = stablehlo.maximum %v199, %v200 : tensor<32x144x56x56xf32>
    %v203 = stablehlo.minimum %v202, %v201 : tensor<32x144x56x56xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v206 = stablehlo.convolution(%v205, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v207 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v208 = stablehlo.add %v206, %v207 : tensor<32x144x56x56xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v212 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v213 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v214 = stablehlo.reduce(%v210 init: %v211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v215 = stablehlo.broadcast_in_dim %v214, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v216 = stablehlo.divide %v215, %v212 : tensor<32x144x56x56xf32>
    %v217 = stablehlo.subtract %v210, %v216 : tensor<32x144x56x56xf32>
    %v218 = stablehlo.multiply %v217, %v217 : tensor<32x144x56x56xf32>
    %v219 = stablehlo.reduce(%v218 init: %v211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v220 = stablehlo.broadcast_in_dim %v219, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v221 = stablehlo.divide %v220, %v212 : tensor<32x144x56x56xf32>
    %v222 = stablehlo.add %v221, %v213 : tensor<32x144x56x56xf32>
    %v223 = stablehlo.rsqrt %v222 : tensor<32x144x56x56xf32>
    %v224 = stablehlo.multiply %v217, %v223 : tensor<32x144x56x56xf32>
    %v225 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v226 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v227 = stablehlo.multiply %v224, %v225 : tensor<32x144x56x56xf32>
    %v228 = stablehlo.add %v227, %v226 : tensor<32x144x56x56xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v231 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v232 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v233 = stablehlo.maximum %v230, %v231 : tensor<32x144x56x56xf32>
    %v234 = stablehlo.minimum %v233, %v232 : tensor<32x144x56x56xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v237 = stablehlo.convolution(%v236, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v238 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v239 = stablehlo.add %v237, %v238 : tensor<32x24x56x56xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v243 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v244 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v245 = stablehlo.reduce(%v241 init: %v242) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v246 = stablehlo.broadcast_in_dim %v245, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v247 = stablehlo.divide %v246, %v243 : tensor<32x24x56x56xf32>
    %v248 = stablehlo.subtract %v241, %v247 : tensor<32x24x56x56xf32>
    %v249 = stablehlo.multiply %v248, %v248 : tensor<32x24x56x56xf32>
    %v250 = stablehlo.reduce(%v249 init: %v242) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v251 = stablehlo.broadcast_in_dim %v250, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v252 = stablehlo.divide %v251, %v243 : tensor<32x24x56x56xf32>
    %v253 = stablehlo.add %v252, %v244 : tensor<32x24x56x56xf32>
    %v254 = stablehlo.rsqrt %v253 : tensor<32x24x56x56xf32>
    %v255 = stablehlo.multiply %v248, %v254 : tensor<32x24x56x56xf32>
    %v256 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v257 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v258 = stablehlo.multiply %v255, %v256 : tensor<32x24x56x56xf32>
    %v259 = stablehlo.add %v258, %v257 : tensor<32x24x56x56xf32>
    %v260 = stablehlo.reshape %v259 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v261 = stablehlo.reshape %v260 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v262 = stablehlo.reshape %v173 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v263 = stablehlo.add %v261, %v262 : tensor<32x24x56x56xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v265 = stablehlo.reshape %v264 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v266 = stablehlo.convolution(%v265, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v267 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v268 = stablehlo.add %v266, %v267 : tensor<32x144x56x56xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v272 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v273 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v274 = stablehlo.reduce(%v270 init: %v271) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v275 = stablehlo.broadcast_in_dim %v274, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v276 = stablehlo.divide %v275, %v272 : tensor<32x144x56x56xf32>
    %v277 = stablehlo.subtract %v270, %v276 : tensor<32x144x56x56xf32>
    %v278 = stablehlo.multiply %v277, %v277 : tensor<32x144x56x56xf32>
    %v279 = stablehlo.reduce(%v278 init: %v271) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v280 = stablehlo.broadcast_in_dim %v279, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v281 = stablehlo.divide %v280, %v272 : tensor<32x144x56x56xf32>
    %v282 = stablehlo.add %v281, %v273 : tensor<32x144x56x56xf32>
    %v283 = stablehlo.rsqrt %v282 : tensor<32x144x56x56xf32>
    %v284 = stablehlo.multiply %v277, %v283 : tensor<32x144x56x56xf32>
    %v285 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v286 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v287 = stablehlo.multiply %v284, %v285 : tensor<32x144x56x56xf32>
    %v288 = stablehlo.add %v287, %v286 : tensor<32x144x56x56xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v291 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v292 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v293 = stablehlo.maximum %v290, %v291 : tensor<32x144x56x56xf32>
    %v294 = stablehlo.minimum %v293, %v292 : tensor<32x144x56x56xf32>
    %v295 = stablehlo.reshape %v294 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v296 = stablehlo.reshape %v295 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v297 = stablehlo.convolution(%v296, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x28x28xf32>
    %v298 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v299 = stablehlo.add %v297, %v298 : tensor<32x144x28x28xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v302 = stablehlo.constant dense<0.0> : tensor<f32>
    %v303 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v304 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v305 = stablehlo.reduce(%v301 init: %v302) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v306 = stablehlo.broadcast_in_dim %v305, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v307 = stablehlo.divide %v306, %v303 : tensor<32x144x28x28xf32>
    %v308 = stablehlo.subtract %v301, %v307 : tensor<32x144x28x28xf32>
    %v309 = stablehlo.multiply %v308, %v308 : tensor<32x144x28x28xf32>
    %v310 = stablehlo.reduce(%v309 init: %v302) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v311 = stablehlo.broadcast_in_dim %v310, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v312 = stablehlo.divide %v311, %v303 : tensor<32x144x28x28xf32>
    %v313 = stablehlo.add %v312, %v304 : tensor<32x144x28x28xf32>
    %v314 = stablehlo.rsqrt %v313 : tensor<32x144x28x28xf32>
    %v315 = stablehlo.multiply %v308, %v314 : tensor<32x144x28x28xf32>
    %v316 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v317 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v318 = stablehlo.multiply %v315, %v316 : tensor<32x144x28x28xf32>
    %v319 = stablehlo.add %v318, %v317 : tensor<32x144x28x28xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v322 = stablehlo.constant dense<0.0> : tensor<32x144x28x28xf32>
    %v323 = stablehlo.constant dense<6.0> : tensor<32x144x28x28xf32>
    %v324 = stablehlo.maximum %v321, %v322 : tensor<32x144x28x28xf32>
    %v325 = stablehlo.minimum %v324, %v323 : tensor<32x144x28x28xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v328 = stablehlo.convolution(%v327, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<32x144x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v329 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v330 = stablehlo.add %v328, %v329 : tensor<32x32x28x28xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v333 = stablehlo.constant dense<0.0> : tensor<f32>
    %v334 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v335 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v336 = stablehlo.reduce(%v332 init: %v333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v337 = stablehlo.broadcast_in_dim %v336, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v338 = stablehlo.divide %v337, %v334 : tensor<32x32x28x28xf32>
    %v339 = stablehlo.subtract %v332, %v338 : tensor<32x32x28x28xf32>
    %v340 = stablehlo.multiply %v339, %v339 : tensor<32x32x28x28xf32>
    %v341 = stablehlo.reduce(%v340 init: %v333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v342 = stablehlo.broadcast_in_dim %v341, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v343 = stablehlo.divide %v342, %v334 : tensor<32x32x28x28xf32>
    %v344 = stablehlo.add %v343, %v335 : tensor<32x32x28x28xf32>
    %v345 = stablehlo.rsqrt %v344 : tensor<32x32x28x28xf32>
    %v346 = stablehlo.multiply %v339, %v345 : tensor<32x32x28x28xf32>
    %v347 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v349 = stablehlo.multiply %v346, %v347 : tensor<32x32x28x28xf32>
    %v350 = stablehlo.add %v349, %v348 : tensor<32x32x28x28xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v353 = stablehlo.convolution(%v352, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v354 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v355 = stablehlo.add %v353, %v354 : tensor<32x192x28x28xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v359 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v360 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v361 = stablehlo.reduce(%v357 init: %v358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v362 = stablehlo.broadcast_in_dim %v361, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v363 = stablehlo.divide %v362, %v359 : tensor<32x192x28x28xf32>
    %v364 = stablehlo.subtract %v357, %v363 : tensor<32x192x28x28xf32>
    %v365 = stablehlo.multiply %v364, %v364 : tensor<32x192x28x28xf32>
    %v366 = stablehlo.reduce(%v365 init: %v358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v367 = stablehlo.broadcast_in_dim %v366, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v368 = stablehlo.divide %v367, %v359 : tensor<32x192x28x28xf32>
    %v369 = stablehlo.add %v368, %v360 : tensor<32x192x28x28xf32>
    %v370 = stablehlo.rsqrt %v369 : tensor<32x192x28x28xf32>
    %v371 = stablehlo.multiply %v364, %v370 : tensor<32x192x28x28xf32>
    %v372 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v373 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v374 = stablehlo.multiply %v371, %v372 : tensor<32x192x28x28xf32>
    %v375 = stablehlo.add %v374, %v373 : tensor<32x192x28x28xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v378 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v379 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v380 = stablehlo.maximum %v377, %v378 : tensor<32x192x28x28xf32>
    %v381 = stablehlo.minimum %v380, %v379 : tensor<32x192x28x28xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v383 = stablehlo.reshape %v382 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v384 = stablehlo.convolution(%v383, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v385 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v386 = stablehlo.add %v384, %v385 : tensor<32x192x28x28xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v388 = stablehlo.reshape %v387 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v389 = stablehlo.constant dense<0.0> : tensor<f32>
    %v390 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v391 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v392 = stablehlo.reduce(%v388 init: %v389) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v393 = stablehlo.broadcast_in_dim %v392, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v394 = stablehlo.divide %v393, %v390 : tensor<32x192x28x28xf32>
    %v395 = stablehlo.subtract %v388, %v394 : tensor<32x192x28x28xf32>
    %v396 = stablehlo.multiply %v395, %v395 : tensor<32x192x28x28xf32>
    %v397 = stablehlo.reduce(%v396 init: %v389) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v398 = stablehlo.broadcast_in_dim %v397, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v399 = stablehlo.divide %v398, %v390 : tensor<32x192x28x28xf32>
    %v400 = stablehlo.add %v399, %v391 : tensor<32x192x28x28xf32>
    %v401 = stablehlo.rsqrt %v400 : tensor<32x192x28x28xf32>
    %v402 = stablehlo.multiply %v395, %v401 : tensor<32x192x28x28xf32>
    %v403 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v404 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v405 = stablehlo.multiply %v402, %v403 : tensor<32x192x28x28xf32>
    %v406 = stablehlo.add %v405, %v404 : tensor<32x192x28x28xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v409 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v410 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v411 = stablehlo.maximum %v408, %v409 : tensor<32x192x28x28xf32>
    %v412 = stablehlo.minimum %v411, %v410 : tensor<32x192x28x28xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v414 = stablehlo.reshape %v413 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v415 = stablehlo.convolution(%v414, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v416 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v417 = stablehlo.add %v415, %v416 : tensor<32x32x28x28xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v421 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v422 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v423 = stablehlo.reduce(%v419 init: %v420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v424 = stablehlo.broadcast_in_dim %v423, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v425 = stablehlo.divide %v424, %v421 : tensor<32x32x28x28xf32>
    %v426 = stablehlo.subtract %v419, %v425 : tensor<32x32x28x28xf32>
    %v427 = stablehlo.multiply %v426, %v426 : tensor<32x32x28x28xf32>
    %v428 = stablehlo.reduce(%v427 init: %v420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v429 = stablehlo.broadcast_in_dim %v428, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v430 = stablehlo.divide %v429, %v421 : tensor<32x32x28x28xf32>
    %v431 = stablehlo.add %v430, %v422 : tensor<32x32x28x28xf32>
    %v432 = stablehlo.rsqrt %v431 : tensor<32x32x28x28xf32>
    %v433 = stablehlo.multiply %v426, %v432 : tensor<32x32x28x28xf32>
    %v434 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v435 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v436 = stablehlo.multiply %v433, %v434 : tensor<32x32x28x28xf32>
    %v437 = stablehlo.add %v436, %v435 : tensor<32x32x28x28xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v439 = stablehlo.reshape %v438 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v440 = stablehlo.reshape %v351 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v441 = stablehlo.add %v439, %v440 : tensor<32x32x28x28xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v444 = stablehlo.convolution(%v443, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v445 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v446 = stablehlo.add %v444, %v445 : tensor<32x192x28x28xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v448 = stablehlo.reshape %v447 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v450 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v451 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v452 = stablehlo.reduce(%v448 init: %v449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v453 = stablehlo.broadcast_in_dim %v452, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v454 = stablehlo.divide %v453, %v450 : tensor<32x192x28x28xf32>
    %v455 = stablehlo.subtract %v448, %v454 : tensor<32x192x28x28xf32>
    %v456 = stablehlo.multiply %v455, %v455 : tensor<32x192x28x28xf32>
    %v457 = stablehlo.reduce(%v456 init: %v449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v458 = stablehlo.broadcast_in_dim %v457, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v459 = stablehlo.divide %v458, %v450 : tensor<32x192x28x28xf32>
    %v460 = stablehlo.add %v459, %v451 : tensor<32x192x28x28xf32>
    %v461 = stablehlo.rsqrt %v460 : tensor<32x192x28x28xf32>
    %v462 = stablehlo.multiply %v455, %v461 : tensor<32x192x28x28xf32>
    %v463 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v464 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v465 = stablehlo.multiply %v462, %v463 : tensor<32x192x28x28xf32>
    %v466 = stablehlo.add %v465, %v464 : tensor<32x192x28x28xf32>
    %v467 = stablehlo.reshape %v466 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v469 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v470 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v471 = stablehlo.maximum %v468, %v469 : tensor<32x192x28x28xf32>
    %v472 = stablehlo.minimum %v471, %v470 : tensor<32x192x28x28xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v475 = stablehlo.convolution(%v474, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v476 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v477 = stablehlo.add %v475, %v476 : tensor<32x192x28x28xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v481 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v482 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v483 = stablehlo.reduce(%v479 init: %v480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v484 = stablehlo.broadcast_in_dim %v483, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v485 = stablehlo.divide %v484, %v481 : tensor<32x192x28x28xf32>
    %v486 = stablehlo.subtract %v479, %v485 : tensor<32x192x28x28xf32>
    %v487 = stablehlo.multiply %v486, %v486 : tensor<32x192x28x28xf32>
    %v488 = stablehlo.reduce(%v487 init: %v480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v489 = stablehlo.broadcast_in_dim %v488, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v490 = stablehlo.divide %v489, %v481 : tensor<32x192x28x28xf32>
    %v491 = stablehlo.add %v490, %v482 : tensor<32x192x28x28xf32>
    %v492 = stablehlo.rsqrt %v491 : tensor<32x192x28x28xf32>
    %v493 = stablehlo.multiply %v486, %v492 : tensor<32x192x28x28xf32>
    %v494 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v495 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v496 = stablehlo.multiply %v493, %v494 : tensor<32x192x28x28xf32>
    %v497 = stablehlo.add %v496, %v495 : tensor<32x192x28x28xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v500 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v501 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v502 = stablehlo.maximum %v499, %v500 : tensor<32x192x28x28xf32>
    %v503 = stablehlo.minimum %v502, %v501 : tensor<32x192x28x28xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v506 = stablehlo.convolution(%v505, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v507 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v508 = stablehlo.add %v506, %v507 : tensor<32x32x28x28xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v512 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v513 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v514 = stablehlo.reduce(%v510 init: %v511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v515 = stablehlo.broadcast_in_dim %v514, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v516 = stablehlo.divide %v515, %v512 : tensor<32x32x28x28xf32>
    %v517 = stablehlo.subtract %v510, %v516 : tensor<32x32x28x28xf32>
    %v518 = stablehlo.multiply %v517, %v517 : tensor<32x32x28x28xf32>
    %v519 = stablehlo.reduce(%v518 init: %v511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v520 = stablehlo.broadcast_in_dim %v519, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v521 = stablehlo.divide %v520, %v512 : tensor<32x32x28x28xf32>
    %v522 = stablehlo.add %v521, %v513 : tensor<32x32x28x28xf32>
    %v523 = stablehlo.rsqrt %v522 : tensor<32x32x28x28xf32>
    %v524 = stablehlo.multiply %v517, %v523 : tensor<32x32x28x28xf32>
    %v525 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v526 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v527 = stablehlo.multiply %v524, %v525 : tensor<32x32x28x28xf32>
    %v528 = stablehlo.add %v527, %v526 : tensor<32x32x28x28xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v531 = stablehlo.reshape %v442 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v532 = stablehlo.add %v530, %v531 : tensor<32x32x28x28xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v535 = stablehlo.convolution(%v534, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v536 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v537 = stablehlo.add %v535, %v536 : tensor<32x192x28x28xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v539 = stablehlo.reshape %v538 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v541 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v542 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v543 = stablehlo.reduce(%v539 init: %v540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v544 = stablehlo.broadcast_in_dim %v543, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v545 = stablehlo.divide %v544, %v541 : tensor<32x192x28x28xf32>
    %v546 = stablehlo.subtract %v539, %v545 : tensor<32x192x28x28xf32>
    %v547 = stablehlo.multiply %v546, %v546 : tensor<32x192x28x28xf32>
    %v548 = stablehlo.reduce(%v547 init: %v540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v549 = stablehlo.broadcast_in_dim %v548, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v550 = stablehlo.divide %v549, %v541 : tensor<32x192x28x28xf32>
    %v551 = stablehlo.add %v550, %v542 : tensor<32x192x28x28xf32>
    %v552 = stablehlo.rsqrt %v551 : tensor<32x192x28x28xf32>
    %v553 = stablehlo.multiply %v546, %v552 : tensor<32x192x28x28xf32>
    %v554 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v555 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v556 = stablehlo.multiply %v553, %v554 : tensor<32x192x28x28xf32>
    %v557 = stablehlo.add %v556, %v555 : tensor<32x192x28x28xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v560 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v561 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v562 = stablehlo.maximum %v559, %v560 : tensor<32x192x28x28xf32>
    %v563 = stablehlo.minimum %v562, %v561 : tensor<32x192x28x28xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v566 = stablehlo.convolution(%v565, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x14x14xf32>
    %v567 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v568 = stablehlo.add %v566, %v567 : tensor<32x192x14x14xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v571 = stablehlo.constant dense<0.0> : tensor<f32>
    %v572 = stablehlo.constant dense<6272.0> : tensor<32x192x14x14xf32>
    %v573 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v574 = stablehlo.reduce(%v570 init: %v571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v575 = stablehlo.broadcast_in_dim %v574, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v576 = stablehlo.divide %v575, %v572 : tensor<32x192x14x14xf32>
    %v577 = stablehlo.subtract %v570, %v576 : tensor<32x192x14x14xf32>
    %v578 = stablehlo.multiply %v577, %v577 : tensor<32x192x14x14xf32>
    %v579 = stablehlo.reduce(%v578 init: %v571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v580 = stablehlo.broadcast_in_dim %v579, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v581 = stablehlo.divide %v580, %v572 : tensor<32x192x14x14xf32>
    %v582 = stablehlo.add %v581, %v573 : tensor<32x192x14x14xf32>
    %v583 = stablehlo.rsqrt %v582 : tensor<32x192x14x14xf32>
    %v584 = stablehlo.multiply %v577, %v583 : tensor<32x192x14x14xf32>
    %v585 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v586 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v587 = stablehlo.multiply %v584, %v585 : tensor<32x192x14x14xf32>
    %v588 = stablehlo.add %v587, %v586 : tensor<32x192x14x14xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v591 = stablehlo.constant dense<0.0> : tensor<32x192x14x14xf32>
    %v592 = stablehlo.constant dense<6.0> : tensor<32x192x14x14xf32>
    %v593 = stablehlo.maximum %v590, %v591 : tensor<32x192x14x14xf32>
    %v594 = stablehlo.minimum %v593, %v592 : tensor<32x192x14x14xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v596 = stablehlo.reshape %v595 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v597 = stablehlo.convolution(%v596, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x14x14xf32>, tensor<64x192x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v598 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v599 = stablehlo.add %v597, %v598 : tensor<32x64x14x14xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v603 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v604 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v605 = stablehlo.reduce(%v601 init: %v602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v606 = stablehlo.broadcast_in_dim %v605, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v607 = stablehlo.divide %v606, %v603 : tensor<32x64x14x14xf32>
    %v608 = stablehlo.subtract %v601, %v607 : tensor<32x64x14x14xf32>
    %v609 = stablehlo.multiply %v608, %v608 : tensor<32x64x14x14xf32>
    %v610 = stablehlo.reduce(%v609 init: %v602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v611 = stablehlo.broadcast_in_dim %v610, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v612 = stablehlo.divide %v611, %v603 : tensor<32x64x14x14xf32>
    %v613 = stablehlo.add %v612, %v604 : tensor<32x64x14x14xf32>
    %v614 = stablehlo.rsqrt %v613 : tensor<32x64x14x14xf32>
    %v615 = stablehlo.multiply %v608, %v614 : tensor<32x64x14x14xf32>
    %v616 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v617 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v618 = stablehlo.multiply %v615, %v616 : tensor<32x64x14x14xf32>
    %v619 = stablehlo.add %v618, %v617 : tensor<32x64x14x14xf32>
    %v620 = stablehlo.reshape %v619 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v622 = stablehlo.convolution(%v621, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v623 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v624 = stablehlo.add %v622, %v623 : tensor<32x384x14x14xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v627 = stablehlo.constant dense<0.0> : tensor<f32>
    %v628 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v629 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v630 = stablehlo.reduce(%v626 init: %v627) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v631 = stablehlo.broadcast_in_dim %v630, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v632 = stablehlo.divide %v631, %v628 : tensor<32x384x14x14xf32>
    %v633 = stablehlo.subtract %v626, %v632 : tensor<32x384x14x14xf32>
    %v634 = stablehlo.multiply %v633, %v633 : tensor<32x384x14x14xf32>
    %v635 = stablehlo.reduce(%v634 init: %v627) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v636 = stablehlo.broadcast_in_dim %v635, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v637 = stablehlo.divide %v636, %v628 : tensor<32x384x14x14xf32>
    %v638 = stablehlo.add %v637, %v629 : tensor<32x384x14x14xf32>
    %v639 = stablehlo.rsqrt %v638 : tensor<32x384x14x14xf32>
    %v640 = stablehlo.multiply %v633, %v639 : tensor<32x384x14x14xf32>
    %v641 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v642 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v643 = stablehlo.multiply %v640, %v641 : tensor<32x384x14x14xf32>
    %v644 = stablehlo.add %v643, %v642 : tensor<32x384x14x14xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v647 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v648 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v649 = stablehlo.maximum %v646, %v647 : tensor<32x384x14x14xf32>
    %v650 = stablehlo.minimum %v649, %v648 : tensor<32x384x14x14xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v653 = stablehlo.convolution(%v652, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v654 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v655 = stablehlo.add %v653, %v654 : tensor<32x384x14x14xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v658 = stablehlo.constant dense<0.0> : tensor<f32>
    %v659 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v660 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v661 = stablehlo.reduce(%v657 init: %v658) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v662 = stablehlo.broadcast_in_dim %v661, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v663 = stablehlo.divide %v662, %v659 : tensor<32x384x14x14xf32>
    %v664 = stablehlo.subtract %v657, %v663 : tensor<32x384x14x14xf32>
    %v665 = stablehlo.multiply %v664, %v664 : tensor<32x384x14x14xf32>
    %v666 = stablehlo.reduce(%v665 init: %v658) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v667 = stablehlo.broadcast_in_dim %v666, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v668 = stablehlo.divide %v667, %v659 : tensor<32x384x14x14xf32>
    %v669 = stablehlo.add %v668, %v660 : tensor<32x384x14x14xf32>
    %v670 = stablehlo.rsqrt %v669 : tensor<32x384x14x14xf32>
    %v671 = stablehlo.multiply %v664, %v670 : tensor<32x384x14x14xf32>
    %v672 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v673 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v674 = stablehlo.multiply %v671, %v672 : tensor<32x384x14x14xf32>
    %v675 = stablehlo.add %v674, %v673 : tensor<32x384x14x14xf32>
    %v676 = stablehlo.reshape %v675 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v678 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v679 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v680 = stablehlo.maximum %v677, %v678 : tensor<32x384x14x14xf32>
    %v681 = stablehlo.minimum %v680, %v679 : tensor<32x384x14x14xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v684 = stablehlo.convolution(%v683, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v685 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v686 = stablehlo.add %v684, %v685 : tensor<32x64x14x14xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v690 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v691 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v692 = stablehlo.reduce(%v688 init: %v689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v693 = stablehlo.broadcast_in_dim %v692, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v694 = stablehlo.divide %v693, %v690 : tensor<32x64x14x14xf32>
    %v695 = stablehlo.subtract %v688, %v694 : tensor<32x64x14x14xf32>
    %v696 = stablehlo.multiply %v695, %v695 : tensor<32x64x14x14xf32>
    %v697 = stablehlo.reduce(%v696 init: %v689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v698 = stablehlo.broadcast_in_dim %v697, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v699 = stablehlo.divide %v698, %v690 : tensor<32x64x14x14xf32>
    %v700 = stablehlo.add %v699, %v691 : tensor<32x64x14x14xf32>
    %v701 = stablehlo.rsqrt %v700 : tensor<32x64x14x14xf32>
    %v702 = stablehlo.multiply %v695, %v701 : tensor<32x64x14x14xf32>
    %v703 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v704 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v705 = stablehlo.multiply %v702, %v703 : tensor<32x64x14x14xf32>
    %v706 = stablehlo.add %v705, %v704 : tensor<32x64x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v709 = stablehlo.reshape %v620 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v710 = stablehlo.add %v708, %v709 : tensor<32x64x14x14xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v713 = stablehlo.convolution(%v712, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v714 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v715 = stablehlo.add %v713, %v714 : tensor<32x384x14x14xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v718 = stablehlo.constant dense<0.0> : tensor<f32>
    %v719 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v720 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v721 = stablehlo.reduce(%v717 init: %v718) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v722 = stablehlo.broadcast_in_dim %v721, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v723 = stablehlo.divide %v722, %v719 : tensor<32x384x14x14xf32>
    %v724 = stablehlo.subtract %v717, %v723 : tensor<32x384x14x14xf32>
    %v725 = stablehlo.multiply %v724, %v724 : tensor<32x384x14x14xf32>
    %v726 = stablehlo.reduce(%v725 init: %v718) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v727 = stablehlo.broadcast_in_dim %v726, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v728 = stablehlo.divide %v727, %v719 : tensor<32x384x14x14xf32>
    %v729 = stablehlo.add %v728, %v720 : tensor<32x384x14x14xf32>
    %v730 = stablehlo.rsqrt %v729 : tensor<32x384x14x14xf32>
    %v731 = stablehlo.multiply %v724, %v730 : tensor<32x384x14x14xf32>
    %v732 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v733 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v734 = stablehlo.multiply %v731, %v732 : tensor<32x384x14x14xf32>
    %v735 = stablehlo.add %v734, %v733 : tensor<32x384x14x14xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v738 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v739 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v740 = stablehlo.maximum %v737, %v738 : tensor<32x384x14x14xf32>
    %v741 = stablehlo.minimum %v740, %v739 : tensor<32x384x14x14xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v744 = stablehlo.convolution(%v743, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v745 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v746 = stablehlo.add %v744, %v745 : tensor<32x384x14x14xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v750 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v751 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v752 = stablehlo.reduce(%v748 init: %v749) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v753 = stablehlo.broadcast_in_dim %v752, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v754 = stablehlo.divide %v753, %v750 : tensor<32x384x14x14xf32>
    %v755 = stablehlo.subtract %v748, %v754 : tensor<32x384x14x14xf32>
    %v756 = stablehlo.multiply %v755, %v755 : tensor<32x384x14x14xf32>
    %v757 = stablehlo.reduce(%v756 init: %v749) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v758 = stablehlo.broadcast_in_dim %v757, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v759 = stablehlo.divide %v758, %v750 : tensor<32x384x14x14xf32>
    %v760 = stablehlo.add %v759, %v751 : tensor<32x384x14x14xf32>
    %v761 = stablehlo.rsqrt %v760 : tensor<32x384x14x14xf32>
    %v762 = stablehlo.multiply %v755, %v761 : tensor<32x384x14x14xf32>
    %v763 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v764 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v765 = stablehlo.multiply %v762, %v763 : tensor<32x384x14x14xf32>
    %v766 = stablehlo.add %v765, %v764 : tensor<32x384x14x14xf32>
    %v767 = stablehlo.reshape %v766 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v769 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v770 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v771 = stablehlo.maximum %v768, %v769 : tensor<32x384x14x14xf32>
    %v772 = stablehlo.minimum %v771, %v770 : tensor<32x384x14x14xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v775 = stablehlo.convolution(%v774, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v776 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v777 = stablehlo.add %v775, %v776 : tensor<32x64x14x14xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v780 = stablehlo.constant dense<0.0> : tensor<f32>
    %v781 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v782 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v783 = stablehlo.reduce(%v779 init: %v780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v784 = stablehlo.broadcast_in_dim %v783, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v785 = stablehlo.divide %v784, %v781 : tensor<32x64x14x14xf32>
    %v786 = stablehlo.subtract %v779, %v785 : tensor<32x64x14x14xf32>
    %v787 = stablehlo.multiply %v786, %v786 : tensor<32x64x14x14xf32>
    %v788 = stablehlo.reduce(%v787 init: %v780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v789 = stablehlo.broadcast_in_dim %v788, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v790 = stablehlo.divide %v789, %v781 : tensor<32x64x14x14xf32>
    %v791 = stablehlo.add %v790, %v782 : tensor<32x64x14x14xf32>
    %v792 = stablehlo.rsqrt %v791 : tensor<32x64x14x14xf32>
    %v793 = stablehlo.multiply %v786, %v792 : tensor<32x64x14x14xf32>
    %v794 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v795 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v796 = stablehlo.multiply %v793, %v794 : tensor<32x64x14x14xf32>
    %v797 = stablehlo.add %v796, %v795 : tensor<32x64x14x14xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v800 = stablehlo.reshape %v711 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v801 = stablehlo.add %v799, %v800 : tensor<32x64x14x14xf32>
    %v802 = stablehlo.reshape %v801 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v804 = stablehlo.convolution(%v803, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v805 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v806 = stablehlo.add %v804, %v805 : tensor<32x384x14x14xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v810 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v811 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v812 = stablehlo.reduce(%v808 init: %v809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v813 = stablehlo.broadcast_in_dim %v812, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v814 = stablehlo.divide %v813, %v810 : tensor<32x384x14x14xf32>
    %v815 = stablehlo.subtract %v808, %v814 : tensor<32x384x14x14xf32>
    %v816 = stablehlo.multiply %v815, %v815 : tensor<32x384x14x14xf32>
    %v817 = stablehlo.reduce(%v816 init: %v809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v818 = stablehlo.broadcast_in_dim %v817, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v819 = stablehlo.divide %v818, %v810 : tensor<32x384x14x14xf32>
    %v820 = stablehlo.add %v819, %v811 : tensor<32x384x14x14xf32>
    %v821 = stablehlo.rsqrt %v820 : tensor<32x384x14x14xf32>
    %v822 = stablehlo.multiply %v815, %v821 : tensor<32x384x14x14xf32>
    %v823 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v824 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v825 = stablehlo.multiply %v822, %v823 : tensor<32x384x14x14xf32>
    %v826 = stablehlo.add %v825, %v824 : tensor<32x384x14x14xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v829 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v830 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v831 = stablehlo.maximum %v828, %v829 : tensor<32x384x14x14xf32>
    %v832 = stablehlo.minimum %v831, %v830 : tensor<32x384x14x14xf32>
    %v833 = stablehlo.reshape %v832 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v835 = stablehlo.convolution(%v834, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v836 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v837 = stablehlo.add %v835, %v836 : tensor<32x384x14x14xf32>
    %v838 = stablehlo.reshape %v837 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v840 = stablehlo.constant dense<0.0> : tensor<f32>
    %v841 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v842 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v843 = stablehlo.reduce(%v839 init: %v840) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v844 = stablehlo.broadcast_in_dim %v843, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v845 = stablehlo.divide %v844, %v841 : tensor<32x384x14x14xf32>
    %v846 = stablehlo.subtract %v839, %v845 : tensor<32x384x14x14xf32>
    %v847 = stablehlo.multiply %v846, %v846 : tensor<32x384x14x14xf32>
    %v848 = stablehlo.reduce(%v847 init: %v840) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v849 = stablehlo.broadcast_in_dim %v848, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v850 = stablehlo.divide %v849, %v841 : tensor<32x384x14x14xf32>
    %v851 = stablehlo.add %v850, %v842 : tensor<32x384x14x14xf32>
    %v852 = stablehlo.rsqrt %v851 : tensor<32x384x14x14xf32>
    %v853 = stablehlo.multiply %v846, %v852 : tensor<32x384x14x14xf32>
    %v854 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v855 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v856 = stablehlo.multiply %v853, %v854 : tensor<32x384x14x14xf32>
    %v857 = stablehlo.add %v856, %v855 : tensor<32x384x14x14xf32>
    %v858 = stablehlo.reshape %v857 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v860 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v861 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v862 = stablehlo.maximum %v859, %v860 : tensor<32x384x14x14xf32>
    %v863 = stablehlo.minimum %v862, %v861 : tensor<32x384x14x14xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v866 = stablehlo.convolution(%v865, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v867 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v868 = stablehlo.add %v866, %v867 : tensor<32x64x14x14xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v871 = stablehlo.constant dense<0.0> : tensor<f32>
    %v872 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v873 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v874 = stablehlo.reduce(%v870 init: %v871) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v875 = stablehlo.broadcast_in_dim %v874, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v876 = stablehlo.divide %v875, %v872 : tensor<32x64x14x14xf32>
    %v877 = stablehlo.subtract %v870, %v876 : tensor<32x64x14x14xf32>
    %v878 = stablehlo.multiply %v877, %v877 : tensor<32x64x14x14xf32>
    %v879 = stablehlo.reduce(%v878 init: %v871) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v880 = stablehlo.broadcast_in_dim %v879, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v881 = stablehlo.divide %v880, %v872 : tensor<32x64x14x14xf32>
    %v882 = stablehlo.add %v881, %v873 : tensor<32x64x14x14xf32>
    %v883 = stablehlo.rsqrt %v882 : tensor<32x64x14x14xf32>
    %v884 = stablehlo.multiply %v877, %v883 : tensor<32x64x14x14xf32>
    %v885 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v886 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v887 = stablehlo.multiply %v884, %v885 : tensor<32x64x14x14xf32>
    %v888 = stablehlo.add %v887, %v886 : tensor<32x64x14x14xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v891 = stablehlo.reshape %v802 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v892 = stablehlo.add %v890, %v891 : tensor<32x64x14x14xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v895 = stablehlo.convolution(%v894, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v896 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v897 = stablehlo.add %v895, %v896 : tensor<32x384x14x14xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v899 = stablehlo.reshape %v898 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v901 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v902 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v903 = stablehlo.reduce(%v899 init: %v900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v904 = stablehlo.broadcast_in_dim %v903, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v905 = stablehlo.divide %v904, %v901 : tensor<32x384x14x14xf32>
    %v906 = stablehlo.subtract %v899, %v905 : tensor<32x384x14x14xf32>
    %v907 = stablehlo.multiply %v906, %v906 : tensor<32x384x14x14xf32>
    %v908 = stablehlo.reduce(%v907 init: %v900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v909 = stablehlo.broadcast_in_dim %v908, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v910 = stablehlo.divide %v909, %v901 : tensor<32x384x14x14xf32>
    %v911 = stablehlo.add %v910, %v902 : tensor<32x384x14x14xf32>
    %v912 = stablehlo.rsqrt %v911 : tensor<32x384x14x14xf32>
    %v913 = stablehlo.multiply %v906, %v912 : tensor<32x384x14x14xf32>
    %v914 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v915 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v916 = stablehlo.multiply %v913, %v914 : tensor<32x384x14x14xf32>
    %v917 = stablehlo.add %v916, %v915 : tensor<32x384x14x14xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v920 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v921 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v922 = stablehlo.maximum %v919, %v920 : tensor<32x384x14x14xf32>
    %v923 = stablehlo.minimum %v922, %v921 : tensor<32x384x14x14xf32>
    %v924 = stablehlo.reshape %v923 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v926 = stablehlo.convolution(%v925, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v927 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v928 = stablehlo.add %v926, %v927 : tensor<32x384x14x14xf32>
    %v929 = stablehlo.reshape %v928 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v930 = stablehlo.reshape %v929 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v932 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v933 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v934 = stablehlo.reduce(%v930 init: %v931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v935 = stablehlo.broadcast_in_dim %v934, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v936 = stablehlo.divide %v935, %v932 : tensor<32x384x14x14xf32>
    %v937 = stablehlo.subtract %v930, %v936 : tensor<32x384x14x14xf32>
    %v938 = stablehlo.multiply %v937, %v937 : tensor<32x384x14x14xf32>
    %v939 = stablehlo.reduce(%v938 init: %v931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v940 = stablehlo.broadcast_in_dim %v939, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v941 = stablehlo.divide %v940, %v932 : tensor<32x384x14x14xf32>
    %v942 = stablehlo.add %v941, %v933 : tensor<32x384x14x14xf32>
    %v943 = stablehlo.rsqrt %v942 : tensor<32x384x14x14xf32>
    %v944 = stablehlo.multiply %v937, %v943 : tensor<32x384x14x14xf32>
    %v945 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v946 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v947 = stablehlo.multiply %v944, %v945 : tensor<32x384x14x14xf32>
    %v948 = stablehlo.add %v947, %v946 : tensor<32x384x14x14xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v951 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v952 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v953 = stablehlo.maximum %v950, %v951 : tensor<32x384x14x14xf32>
    %v954 = stablehlo.minimum %v953, %v952 : tensor<32x384x14x14xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v957 = stablehlo.convolution(%v956, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v958 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v959 = stablehlo.add %v957, %v958 : tensor<32x96x14x14xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v963 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v964 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v965 = stablehlo.reduce(%v961 init: %v962) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v966 = stablehlo.broadcast_in_dim %v965, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v967 = stablehlo.divide %v966, %v963 : tensor<32x96x14x14xf32>
    %v968 = stablehlo.subtract %v961, %v967 : tensor<32x96x14x14xf32>
    %v969 = stablehlo.multiply %v968, %v968 : tensor<32x96x14x14xf32>
    %v970 = stablehlo.reduce(%v969 init: %v962) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v971 = stablehlo.broadcast_in_dim %v970, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v972 = stablehlo.divide %v971, %v963 : tensor<32x96x14x14xf32>
    %v973 = stablehlo.add %v972, %v964 : tensor<32x96x14x14xf32>
    %v974 = stablehlo.rsqrt %v973 : tensor<32x96x14x14xf32>
    %v975 = stablehlo.multiply %v968, %v974 : tensor<32x96x14x14xf32>
    %v976 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v977 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v978 = stablehlo.multiply %v975, %v976 : tensor<32x96x14x14xf32>
    %v979 = stablehlo.add %v978, %v977 : tensor<32x96x14x14xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v982 = stablehlo.convolution(%v981, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v983 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v984 = stablehlo.add %v982, %v983 : tensor<32x576x14x14xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v987 = stablehlo.constant dense<0.0> : tensor<f32>
    %v988 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v989 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v990 = stablehlo.reduce(%v986 init: %v987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v991 = stablehlo.broadcast_in_dim %v990, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v992 = stablehlo.divide %v991, %v988 : tensor<32x576x14x14xf32>
    %v993 = stablehlo.subtract %v986, %v992 : tensor<32x576x14x14xf32>
    %v994 = stablehlo.multiply %v993, %v993 : tensor<32x576x14x14xf32>
    %v995 = stablehlo.reduce(%v994 init: %v987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v996 = stablehlo.broadcast_in_dim %v995, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v997 = stablehlo.divide %v996, %v988 : tensor<32x576x14x14xf32>
    %v998 = stablehlo.add %v997, %v989 : tensor<32x576x14x14xf32>
    %v999 = stablehlo.rsqrt %v998 : tensor<32x576x14x14xf32>
    %v1000 = stablehlo.multiply %v993, %v999 : tensor<32x576x14x14xf32>
    %v1001 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1002 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1003 = stablehlo.multiply %v1000, %v1001 : tensor<32x576x14x14xf32>
    %v1004 = stablehlo.add %v1003, %v1002 : tensor<32x576x14x14xf32>
    %v1005 = stablehlo.reshape %v1004 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1007 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v1008 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v1009 = stablehlo.maximum %v1006, %v1007 : tensor<32x576x14x14xf32>
    %v1010 = stablehlo.minimum %v1009, %v1008 : tensor<32x576x14x14xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1013 = stablehlo.convolution(%v1012, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v1014 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1015 = stablehlo.add %v1013, %v1014 : tensor<32x576x14x14xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1019 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v1020 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v1021 = stablehlo.reduce(%v1017 init: %v1018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v1022 = stablehlo.broadcast_in_dim %v1021, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1023 = stablehlo.divide %v1022, %v1019 : tensor<32x576x14x14xf32>
    %v1024 = stablehlo.subtract %v1017, %v1023 : tensor<32x576x14x14xf32>
    %v1025 = stablehlo.multiply %v1024, %v1024 : tensor<32x576x14x14xf32>
    %v1026 = stablehlo.reduce(%v1025 init: %v1018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v1027 = stablehlo.broadcast_in_dim %v1026, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1028 = stablehlo.divide %v1027, %v1019 : tensor<32x576x14x14xf32>
    %v1029 = stablehlo.add %v1028, %v1020 : tensor<32x576x14x14xf32>
    %v1030 = stablehlo.rsqrt %v1029 : tensor<32x576x14x14xf32>
    %v1031 = stablehlo.multiply %v1024, %v1030 : tensor<32x576x14x14xf32>
    %v1032 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1033 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1034 = stablehlo.multiply %v1031, %v1032 : tensor<32x576x14x14xf32>
    %v1035 = stablehlo.add %v1034, %v1033 : tensor<32x576x14x14xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1038 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v1039 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v1040 = stablehlo.maximum %v1037, %v1038 : tensor<32x576x14x14xf32>
    %v1041 = stablehlo.minimum %v1040, %v1039 : tensor<32x576x14x14xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1044 = stablehlo.convolution(%v1043, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v1045 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1046 = stablehlo.add %v1044, %v1045 : tensor<32x96x14x14xf32>
    %v1047 = stablehlo.reshape %v1046 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1049 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1050 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v1051 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v1052 = stablehlo.reduce(%v1048 init: %v1049) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v1053 = stablehlo.broadcast_in_dim %v1052, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1054 = stablehlo.divide %v1053, %v1050 : tensor<32x96x14x14xf32>
    %v1055 = stablehlo.subtract %v1048, %v1054 : tensor<32x96x14x14xf32>
    %v1056 = stablehlo.multiply %v1055, %v1055 : tensor<32x96x14x14xf32>
    %v1057 = stablehlo.reduce(%v1056 init: %v1049) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v1058 = stablehlo.broadcast_in_dim %v1057, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1059 = stablehlo.divide %v1058, %v1050 : tensor<32x96x14x14xf32>
    %v1060 = stablehlo.add %v1059, %v1051 : tensor<32x96x14x14xf32>
    %v1061 = stablehlo.rsqrt %v1060 : tensor<32x96x14x14xf32>
    %v1062 = stablehlo.multiply %v1055, %v1061 : tensor<32x96x14x14xf32>
    %v1063 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1064 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1065 = stablehlo.multiply %v1062, %v1063 : tensor<32x96x14x14xf32>
    %v1066 = stablehlo.add %v1065, %v1064 : tensor<32x96x14x14xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1069 = stablehlo.reshape %v980 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1070 = stablehlo.add %v1068, %v1069 : tensor<32x96x14x14xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1073 = stablehlo.convolution(%v1072, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v1074 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1075 = stablehlo.add %v1073, %v1074 : tensor<32x576x14x14xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1077 = stablehlo.reshape %v1076 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1078 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1079 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v1080 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v1081 = stablehlo.reduce(%v1077 init: %v1078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v1082 = stablehlo.broadcast_in_dim %v1081, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1083 = stablehlo.divide %v1082, %v1079 : tensor<32x576x14x14xf32>
    %v1084 = stablehlo.subtract %v1077, %v1083 : tensor<32x576x14x14xf32>
    %v1085 = stablehlo.multiply %v1084, %v1084 : tensor<32x576x14x14xf32>
    %v1086 = stablehlo.reduce(%v1085 init: %v1078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v1087 = stablehlo.broadcast_in_dim %v1086, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1088 = stablehlo.divide %v1087, %v1079 : tensor<32x576x14x14xf32>
    %v1089 = stablehlo.add %v1088, %v1080 : tensor<32x576x14x14xf32>
    %v1090 = stablehlo.rsqrt %v1089 : tensor<32x576x14x14xf32>
    %v1091 = stablehlo.multiply %v1084, %v1090 : tensor<32x576x14x14xf32>
    %v1092 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1093 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1094 = stablehlo.multiply %v1091, %v1092 : tensor<32x576x14x14xf32>
    %v1095 = stablehlo.add %v1094, %v1093 : tensor<32x576x14x14xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1098 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v1099 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v1100 = stablehlo.maximum %v1097, %v1098 : tensor<32x576x14x14xf32>
    %v1101 = stablehlo.minimum %v1100, %v1099 : tensor<32x576x14x14xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1103 = stablehlo.reshape %v1102 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1104 = stablehlo.convolution(%v1103, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v1105 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1106 = stablehlo.add %v1104, %v1105 : tensor<32x576x14x14xf32>
    %v1107 = stablehlo.reshape %v1106 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1108 = stablehlo.reshape %v1107 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1110 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v1111 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v1112 = stablehlo.reduce(%v1108 init: %v1109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v1113 = stablehlo.broadcast_in_dim %v1112, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1114 = stablehlo.divide %v1113, %v1110 : tensor<32x576x14x14xf32>
    %v1115 = stablehlo.subtract %v1108, %v1114 : tensor<32x576x14x14xf32>
    %v1116 = stablehlo.multiply %v1115, %v1115 : tensor<32x576x14x14xf32>
    %v1117 = stablehlo.reduce(%v1116 init: %v1109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v1118 = stablehlo.broadcast_in_dim %v1117, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1119 = stablehlo.divide %v1118, %v1110 : tensor<32x576x14x14xf32>
    %v1120 = stablehlo.add %v1119, %v1111 : tensor<32x576x14x14xf32>
    %v1121 = stablehlo.rsqrt %v1120 : tensor<32x576x14x14xf32>
    %v1122 = stablehlo.multiply %v1115, %v1121 : tensor<32x576x14x14xf32>
    %v1123 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1124 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1125 = stablehlo.multiply %v1122, %v1123 : tensor<32x576x14x14xf32>
    %v1126 = stablehlo.add %v1125, %v1124 : tensor<32x576x14x14xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1128 = stablehlo.reshape %v1127 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1129 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v1130 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v1131 = stablehlo.maximum %v1128, %v1129 : tensor<32x576x14x14xf32>
    %v1132 = stablehlo.minimum %v1131, %v1130 : tensor<32x576x14x14xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1134 = stablehlo.reshape %v1133 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1135 = stablehlo.convolution(%v1134, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v1136 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1137 = stablehlo.add %v1135, %v1136 : tensor<32x96x14x14xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1139 = stablehlo.reshape %v1138 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1141 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v1142 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v1143 = stablehlo.reduce(%v1139 init: %v1140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v1144 = stablehlo.broadcast_in_dim %v1143, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1145 = stablehlo.divide %v1144, %v1141 : tensor<32x96x14x14xf32>
    %v1146 = stablehlo.subtract %v1139, %v1145 : tensor<32x96x14x14xf32>
    %v1147 = stablehlo.multiply %v1146, %v1146 : tensor<32x96x14x14xf32>
    %v1148 = stablehlo.reduce(%v1147 init: %v1140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v1149 = stablehlo.broadcast_in_dim %v1148, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1150 = stablehlo.divide %v1149, %v1141 : tensor<32x96x14x14xf32>
    %v1151 = stablehlo.add %v1150, %v1142 : tensor<32x96x14x14xf32>
    %v1152 = stablehlo.rsqrt %v1151 : tensor<32x96x14x14xf32>
    %v1153 = stablehlo.multiply %v1146, %v1152 : tensor<32x96x14x14xf32>
    %v1154 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1155 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1156 = stablehlo.multiply %v1153, %v1154 : tensor<32x96x14x14xf32>
    %v1157 = stablehlo.add %v1156, %v1155 : tensor<32x96x14x14xf32>
    %v1158 = stablehlo.reshape %v1157 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1160 = stablehlo.reshape %v1071 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1161 = stablehlo.add %v1159, %v1160 : tensor<32x96x14x14xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1164 = stablehlo.convolution(%v1163, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v1165 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1166 = stablehlo.add %v1164, %v1165 : tensor<32x576x14x14xf32>
    %v1167 = stablehlo.reshape %v1166 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1169 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1170 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v1171 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v1172 = stablehlo.reduce(%v1168 init: %v1169) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v1173 = stablehlo.broadcast_in_dim %v1172, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1174 = stablehlo.divide %v1173, %v1170 : tensor<32x576x14x14xf32>
    %v1175 = stablehlo.subtract %v1168, %v1174 : tensor<32x576x14x14xf32>
    %v1176 = stablehlo.multiply %v1175, %v1175 : tensor<32x576x14x14xf32>
    %v1177 = stablehlo.reduce(%v1176 init: %v1169) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v1178 = stablehlo.broadcast_in_dim %v1177, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1179 = stablehlo.divide %v1178, %v1170 : tensor<32x576x14x14xf32>
    %v1180 = stablehlo.add %v1179, %v1171 : tensor<32x576x14x14xf32>
    %v1181 = stablehlo.rsqrt %v1180 : tensor<32x576x14x14xf32>
    %v1182 = stablehlo.multiply %v1175, %v1181 : tensor<32x576x14x14xf32>
    %v1183 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1184 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1185 = stablehlo.multiply %v1182, %v1183 : tensor<32x576x14x14xf32>
    %v1186 = stablehlo.add %v1185, %v1184 : tensor<32x576x14x14xf32>
    %v1187 = stablehlo.reshape %v1186 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1189 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v1190 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v1191 = stablehlo.maximum %v1188, %v1189 : tensor<32x576x14x14xf32>
    %v1192 = stablehlo.minimum %v1191, %v1190 : tensor<32x576x14x14xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1194 = stablehlo.reshape %v1193 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1195 = stablehlo.convolution(%v1194, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x7x7xf32>
    %v1196 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1197 = stablehlo.add %v1195, %v1196 : tensor<32x576x7x7xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v1200 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1201 = stablehlo.constant dense<1568.0> : tensor<32x576x7x7xf32>
    %v1202 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v1203 = stablehlo.reduce(%v1199 init: %v1200) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v1204 = stablehlo.broadcast_in_dim %v1203, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1205 = stablehlo.divide %v1204, %v1201 : tensor<32x576x7x7xf32>
    %v1206 = stablehlo.subtract %v1199, %v1205 : tensor<32x576x7x7xf32>
    %v1207 = stablehlo.multiply %v1206, %v1206 : tensor<32x576x7x7xf32>
    %v1208 = stablehlo.reduce(%v1207 init: %v1200) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v1209 = stablehlo.broadcast_in_dim %v1208, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1210 = stablehlo.divide %v1209, %v1201 : tensor<32x576x7x7xf32>
    %v1211 = stablehlo.add %v1210, %v1202 : tensor<32x576x7x7xf32>
    %v1212 = stablehlo.rsqrt %v1211 : tensor<32x576x7x7xf32>
    %v1213 = stablehlo.multiply %v1206, %v1212 : tensor<32x576x7x7xf32>
    %v1214 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1215 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1216 = stablehlo.multiply %v1213, %v1214 : tensor<32x576x7x7xf32>
    %v1217 = stablehlo.add %v1216, %v1215 : tensor<32x576x7x7xf32>
    %v1218 = stablehlo.reshape %v1217 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v1220 = stablehlo.constant dense<0.0> : tensor<32x576x7x7xf32>
    %v1221 = stablehlo.constant dense<6.0> : tensor<32x576x7x7xf32>
    %v1222 = stablehlo.maximum %v1219, %v1220 : tensor<32x576x7x7xf32>
    %v1223 = stablehlo.minimum %v1222, %v1221 : tensor<32x576x7x7xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v1226 = stablehlo.convolution(%v1225, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x7x7xf32>, tensor<160x576x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1227 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1228 = stablehlo.add %v1226, %v1227 : tensor<32x160x7x7xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1232 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v1233 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1234 = stablehlo.reduce(%v1230 init: %v1231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1235 = stablehlo.broadcast_in_dim %v1234, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1236 = stablehlo.divide %v1235, %v1232 : tensor<32x160x7x7xf32>
    %v1237 = stablehlo.subtract %v1230, %v1236 : tensor<32x160x7x7xf32>
    %v1238 = stablehlo.multiply %v1237, %v1237 : tensor<32x160x7x7xf32>
    %v1239 = stablehlo.reduce(%v1238 init: %v1231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1240 = stablehlo.broadcast_in_dim %v1239, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1241 = stablehlo.divide %v1240, %v1232 : tensor<32x160x7x7xf32>
    %v1242 = stablehlo.add %v1241, %v1233 : tensor<32x160x7x7xf32>
    %v1243 = stablehlo.rsqrt %v1242 : tensor<32x160x7x7xf32>
    %v1244 = stablehlo.multiply %v1237, %v1243 : tensor<32x160x7x7xf32>
    %v1245 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1246 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1247 = stablehlo.multiply %v1244, %v1245 : tensor<32x160x7x7xf32>
    %v1248 = stablehlo.add %v1247, %v1246 : tensor<32x160x7x7xf32>
    %v1249 = stablehlo.reshape %v1248 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1251 = stablehlo.convolution(%v1250, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1252 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1253 = stablehlo.add %v1251, %v1252 : tensor<32x960x7x7xf32>
    %v1254 = stablehlo.reshape %v1253 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1257 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1258 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1259 = stablehlo.reduce(%v1255 init: %v1256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1260 = stablehlo.broadcast_in_dim %v1259, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1261 = stablehlo.divide %v1260, %v1257 : tensor<32x960x7x7xf32>
    %v1262 = stablehlo.subtract %v1255, %v1261 : tensor<32x960x7x7xf32>
    %v1263 = stablehlo.multiply %v1262, %v1262 : tensor<32x960x7x7xf32>
    %v1264 = stablehlo.reduce(%v1263 init: %v1256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1265 = stablehlo.broadcast_in_dim %v1264, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1266 = stablehlo.divide %v1265, %v1257 : tensor<32x960x7x7xf32>
    %v1267 = stablehlo.add %v1266, %v1258 : tensor<32x960x7x7xf32>
    %v1268 = stablehlo.rsqrt %v1267 : tensor<32x960x7x7xf32>
    %v1269 = stablehlo.multiply %v1262, %v1268 : tensor<32x960x7x7xf32>
    %v1270 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1271 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1272 = stablehlo.multiply %v1269, %v1270 : tensor<32x960x7x7xf32>
    %v1273 = stablehlo.add %v1272, %v1271 : tensor<32x960x7x7xf32>
    %v1274 = stablehlo.reshape %v1273 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1276 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1277 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1278 = stablehlo.maximum %v1275, %v1276 : tensor<32x960x7x7xf32>
    %v1279 = stablehlo.minimum %v1278, %v1277 : tensor<32x960x7x7xf32>
    %v1280 = stablehlo.reshape %v1279 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1282 = stablehlo.convolution(%v1281, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1283 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1284 = stablehlo.add %v1282, %v1283 : tensor<32x960x7x7xf32>
    %v1285 = stablehlo.reshape %v1284 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1286 = stablehlo.reshape %v1285 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1287 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1288 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1289 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1290 = stablehlo.reduce(%v1286 init: %v1287) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1291 = stablehlo.broadcast_in_dim %v1290, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1292 = stablehlo.divide %v1291, %v1288 : tensor<32x960x7x7xf32>
    %v1293 = stablehlo.subtract %v1286, %v1292 : tensor<32x960x7x7xf32>
    %v1294 = stablehlo.multiply %v1293, %v1293 : tensor<32x960x7x7xf32>
    %v1295 = stablehlo.reduce(%v1294 init: %v1287) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1296 = stablehlo.broadcast_in_dim %v1295, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1297 = stablehlo.divide %v1296, %v1288 : tensor<32x960x7x7xf32>
    %v1298 = stablehlo.add %v1297, %v1289 : tensor<32x960x7x7xf32>
    %v1299 = stablehlo.rsqrt %v1298 : tensor<32x960x7x7xf32>
    %v1300 = stablehlo.multiply %v1293, %v1299 : tensor<32x960x7x7xf32>
    %v1301 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1302 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1303 = stablehlo.multiply %v1300, %v1301 : tensor<32x960x7x7xf32>
    %v1304 = stablehlo.add %v1303, %v1302 : tensor<32x960x7x7xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1306 = stablehlo.reshape %v1305 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1307 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1308 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1309 = stablehlo.maximum %v1306, %v1307 : tensor<32x960x7x7xf32>
    %v1310 = stablehlo.minimum %v1309, %v1308 : tensor<32x960x7x7xf32>
    %v1311 = stablehlo.reshape %v1310 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1313 = stablehlo.convolution(%v1312, %b15pW)
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
    %v1332 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1333 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1334 = stablehlo.multiply %v1331, %v1332 : tensor<32x160x7x7xf32>
    %v1335 = stablehlo.add %v1334, %v1333 : tensor<32x160x7x7xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1338 = stablehlo.reshape %v1249 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1339 = stablehlo.add %v1337, %v1338 : tensor<32x160x7x7xf32>
    %v1340 = stablehlo.reshape %v1339 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1342 = stablehlo.convolution(%v1341, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1343 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1344 = stablehlo.add %v1342, %v1343 : tensor<32x960x7x7xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1347 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1348 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1349 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1350 = stablehlo.reduce(%v1346 init: %v1347) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1351 = stablehlo.broadcast_in_dim %v1350, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1352 = stablehlo.divide %v1351, %v1348 : tensor<32x960x7x7xf32>
    %v1353 = stablehlo.subtract %v1346, %v1352 : tensor<32x960x7x7xf32>
    %v1354 = stablehlo.multiply %v1353, %v1353 : tensor<32x960x7x7xf32>
    %v1355 = stablehlo.reduce(%v1354 init: %v1347) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1356 = stablehlo.broadcast_in_dim %v1355, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1357 = stablehlo.divide %v1356, %v1348 : tensor<32x960x7x7xf32>
    %v1358 = stablehlo.add %v1357, %v1349 : tensor<32x960x7x7xf32>
    %v1359 = stablehlo.rsqrt %v1358 : tensor<32x960x7x7xf32>
    %v1360 = stablehlo.multiply %v1353, %v1359 : tensor<32x960x7x7xf32>
    %v1361 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1362 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1363 = stablehlo.multiply %v1360, %v1361 : tensor<32x960x7x7xf32>
    %v1364 = stablehlo.add %v1363, %v1362 : tensor<32x960x7x7xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1366 = stablehlo.reshape %v1365 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1367 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1368 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1369 = stablehlo.maximum %v1366, %v1367 : tensor<32x960x7x7xf32>
    %v1370 = stablehlo.minimum %v1369, %v1368 : tensor<32x960x7x7xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1373 = stablehlo.convolution(%v1372, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1374 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1375 = stablehlo.add %v1373, %v1374 : tensor<32x960x7x7xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1377 = stablehlo.reshape %v1376 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1379 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1380 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1381 = stablehlo.reduce(%v1377 init: %v1378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1382 = stablehlo.broadcast_in_dim %v1381, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1383 = stablehlo.divide %v1382, %v1379 : tensor<32x960x7x7xf32>
    %v1384 = stablehlo.subtract %v1377, %v1383 : tensor<32x960x7x7xf32>
    %v1385 = stablehlo.multiply %v1384, %v1384 : tensor<32x960x7x7xf32>
    %v1386 = stablehlo.reduce(%v1385 init: %v1378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1387 = stablehlo.broadcast_in_dim %v1386, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1388 = stablehlo.divide %v1387, %v1379 : tensor<32x960x7x7xf32>
    %v1389 = stablehlo.add %v1388, %v1380 : tensor<32x960x7x7xf32>
    %v1390 = stablehlo.rsqrt %v1389 : tensor<32x960x7x7xf32>
    %v1391 = stablehlo.multiply %v1384, %v1390 : tensor<32x960x7x7xf32>
    %v1392 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1393 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1394 = stablehlo.multiply %v1391, %v1392 : tensor<32x960x7x7xf32>
    %v1395 = stablehlo.add %v1394, %v1393 : tensor<32x960x7x7xf32>
    %v1396 = stablehlo.reshape %v1395 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1397 = stablehlo.reshape %v1396 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1398 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1399 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1400 = stablehlo.maximum %v1397, %v1398 : tensor<32x960x7x7xf32>
    %v1401 = stablehlo.minimum %v1400, %v1399 : tensor<32x960x7x7xf32>
    %v1402 = stablehlo.reshape %v1401 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1404 = stablehlo.convolution(%v1403, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1405 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1406 = stablehlo.add %v1404, %v1405 : tensor<32x160x7x7xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1409 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1410 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v1411 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1412 = stablehlo.reduce(%v1408 init: %v1409) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1413 = stablehlo.broadcast_in_dim %v1412, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1414 = stablehlo.divide %v1413, %v1410 : tensor<32x160x7x7xf32>
    %v1415 = stablehlo.subtract %v1408, %v1414 : tensor<32x160x7x7xf32>
    %v1416 = stablehlo.multiply %v1415, %v1415 : tensor<32x160x7x7xf32>
    %v1417 = stablehlo.reduce(%v1416 init: %v1409) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1418 = stablehlo.broadcast_in_dim %v1417, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1419 = stablehlo.divide %v1418, %v1410 : tensor<32x160x7x7xf32>
    %v1420 = stablehlo.add %v1419, %v1411 : tensor<32x160x7x7xf32>
    %v1421 = stablehlo.rsqrt %v1420 : tensor<32x160x7x7xf32>
    %v1422 = stablehlo.multiply %v1415, %v1421 : tensor<32x160x7x7xf32>
    %v1423 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1424 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1425 = stablehlo.multiply %v1422, %v1423 : tensor<32x160x7x7xf32>
    %v1426 = stablehlo.add %v1425, %v1424 : tensor<32x160x7x7xf32>
    %v1427 = stablehlo.reshape %v1426 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1429 = stablehlo.reshape %v1340 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1430 = stablehlo.add %v1428, %v1429 : tensor<32x160x7x7xf32>
    %v1431 = stablehlo.reshape %v1430 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1433 = stablehlo.convolution(%v1432, %b17eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1434 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1435 = stablehlo.add %v1433, %v1434 : tensor<32x960x7x7xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1438 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1439 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1440 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1441 = stablehlo.reduce(%v1437 init: %v1438) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1442 = stablehlo.broadcast_in_dim %v1441, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1443 = stablehlo.divide %v1442, %v1439 : tensor<32x960x7x7xf32>
    %v1444 = stablehlo.subtract %v1437, %v1443 : tensor<32x960x7x7xf32>
    %v1445 = stablehlo.multiply %v1444, %v1444 : tensor<32x960x7x7xf32>
    %v1446 = stablehlo.reduce(%v1445 init: %v1438) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1447 = stablehlo.broadcast_in_dim %v1446, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1448 = stablehlo.divide %v1447, %v1439 : tensor<32x960x7x7xf32>
    %v1449 = stablehlo.add %v1448, %v1440 : tensor<32x960x7x7xf32>
    %v1450 = stablehlo.rsqrt %v1449 : tensor<32x960x7x7xf32>
    %v1451 = stablehlo.multiply %v1444, %v1450 : tensor<32x960x7x7xf32>
    %v1452 = stablehlo.broadcast_in_dim %b17eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1453 = stablehlo.broadcast_in_dim %b17ebt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1454 = stablehlo.multiply %v1451, %v1452 : tensor<32x960x7x7xf32>
    %v1455 = stablehlo.add %v1454, %v1453 : tensor<32x960x7x7xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1458 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1459 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1460 = stablehlo.maximum %v1457, %v1458 : tensor<32x960x7x7xf32>
    %v1461 = stablehlo.minimum %v1460, %v1459 : tensor<32x960x7x7xf32>
    %v1462 = stablehlo.reshape %v1461 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1463 = stablehlo.reshape %v1462 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1464 = stablehlo.convolution(%v1463, %b17dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1465 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1466 = stablehlo.add %v1464, %v1465 : tensor<32x960x7x7xf32>
    %v1467 = stablehlo.reshape %v1466 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1470 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1471 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1472 = stablehlo.reduce(%v1468 init: %v1469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1473 = stablehlo.broadcast_in_dim %v1472, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1474 = stablehlo.divide %v1473, %v1470 : tensor<32x960x7x7xf32>
    %v1475 = stablehlo.subtract %v1468, %v1474 : tensor<32x960x7x7xf32>
    %v1476 = stablehlo.multiply %v1475, %v1475 : tensor<32x960x7x7xf32>
    %v1477 = stablehlo.reduce(%v1476 init: %v1469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1478 = stablehlo.broadcast_in_dim %v1477, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1479 = stablehlo.divide %v1478, %v1470 : tensor<32x960x7x7xf32>
    %v1480 = stablehlo.add %v1479, %v1471 : tensor<32x960x7x7xf32>
    %v1481 = stablehlo.rsqrt %v1480 : tensor<32x960x7x7xf32>
    %v1482 = stablehlo.multiply %v1475, %v1481 : tensor<32x960x7x7xf32>
    %v1483 = stablehlo.broadcast_in_dim %b17dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1484 = stablehlo.broadcast_in_dim %b17dbt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1485 = stablehlo.multiply %v1482, %v1483 : tensor<32x960x7x7xf32>
    %v1486 = stablehlo.add %v1485, %v1484 : tensor<32x960x7x7xf32>
    %v1487 = stablehlo.reshape %v1486 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1488 = stablehlo.reshape %v1487 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1489 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1490 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1491 = stablehlo.maximum %v1488, %v1489 : tensor<32x960x7x7xf32>
    %v1492 = stablehlo.minimum %v1491, %v1490 : tensor<32x960x7x7xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1494 = stablehlo.reshape %v1493 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1495 = stablehlo.convolution(%v1494, %b17pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<320x960x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1496 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1497 = stablehlo.add %v1495, %v1496 : tensor<32x320x7x7xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1499 = stablehlo.reshape %v1498 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1500 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1501 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1502 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1503 = stablehlo.reduce(%v1499 init: %v1500) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1504 = stablehlo.broadcast_in_dim %v1503, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1505 = stablehlo.divide %v1504, %v1501 : tensor<32x320x7x7xf32>
    %v1506 = stablehlo.subtract %v1499, %v1505 : tensor<32x320x7x7xf32>
    %v1507 = stablehlo.multiply %v1506, %v1506 : tensor<32x320x7x7xf32>
    %v1508 = stablehlo.reduce(%v1507 init: %v1500) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1509 = stablehlo.broadcast_in_dim %v1508, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1510 = stablehlo.divide %v1509, %v1501 : tensor<32x320x7x7xf32>
    %v1511 = stablehlo.add %v1510, %v1502 : tensor<32x320x7x7xf32>
    %v1512 = stablehlo.rsqrt %v1511 : tensor<32x320x7x7xf32>
    %v1513 = stablehlo.multiply %v1506, %v1512 : tensor<32x320x7x7xf32>
    %v1514 = stablehlo.broadcast_in_dim %b17pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1515 = stablehlo.broadcast_in_dim %b17pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1516 = stablehlo.multiply %v1513, %v1514 : tensor<32x320x7x7xf32>
    %v1517 = stablehlo.add %v1516, %v1515 : tensor<32x320x7x7xf32>
    %v1518 = stablehlo.reshape %v1517 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1520 = stablehlo.convolution(%v1519, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1521 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1522 = stablehlo.add %v1520, %v1521 : tensor<32x1280x7x7xf32>
    %v1523 = stablehlo.reshape %v1522 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1526 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1527 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1528 = stablehlo.reduce(%v1524 init: %v1525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1529 = stablehlo.broadcast_in_dim %v1528, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1530 = stablehlo.divide %v1529, %v1526 : tensor<32x1280x7x7xf32>
    %v1531 = stablehlo.subtract %v1524, %v1530 : tensor<32x1280x7x7xf32>
    %v1532 = stablehlo.multiply %v1531, %v1531 : tensor<32x1280x7x7xf32>
    %v1533 = stablehlo.reduce(%v1532 init: %v1525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1534 = stablehlo.broadcast_in_dim %v1533, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1535 = stablehlo.divide %v1534, %v1526 : tensor<32x1280x7x7xf32>
    %v1536 = stablehlo.add %v1535, %v1527 : tensor<32x1280x7x7xf32>
    %v1537 = stablehlo.rsqrt %v1536 : tensor<32x1280x7x7xf32>
    %v1538 = stablehlo.multiply %v1531, %v1537 : tensor<32x1280x7x7xf32>
    %v1539 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1540 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1541 = stablehlo.multiply %v1538, %v1539 : tensor<32x1280x7x7xf32>
    %v1542 = stablehlo.add %v1541, %v1540 : tensor<32x1280x7x7xf32>
    %v1543 = stablehlo.reshape %v1542 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1544 = stablehlo.reshape %v1543 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1545 = stablehlo.constant dense<0.0> : tensor<32x1280x7x7xf32>
    %v1546 = stablehlo.constant dense<6.0> : tensor<32x1280x7x7xf32>
    %v1547 = stablehlo.maximum %v1544, %v1545 : tensor<32x1280x7x7xf32>
    %v1548 = stablehlo.minimum %v1547, %v1546 : tensor<32x1280x7x7xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1551 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1552 = stablehlo.reduce(%v1550 init: %v1551) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1553 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1554 = stablehlo.divide %v1552, %v1553 : tensor<32x1280xf32>
    %v1555 = stablehlo.dot_general %v1554, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1556 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1557 = stablehlo.add %v1555, %v1556 : tensor<32x10xf32>
    %v1558 = stablehlo.reshape %v1557 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1559 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1560 = stablehlo.exponential %v1558 : tensor<32x1x10xf32>
    %v1561 = stablehlo.reduce(%v1560 init: %v1559) applies stablehlo.add across dimensions = [2] : (tensor<32x1x10xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1562 = stablehlo.broadcast_in_dim %v1561, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x10xf32>
    %v1563 = stablehlo.divide %v1560, %v1562 : tensor<32x1x10xf32>
    %v1564 = stablehlo.reshape %v1563 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v1565 = stablehlo.subtract %v1564, %onehot : tensor<32x10xf32>
    %v1566 = stablehlo.constant dense<0.100000> : tensor<32x10xf32>
    %v1567 = stablehlo.multiply %onehot, %v1566 : tensor<32x10xf32>
    %v1568 = stablehlo.add %v1565, %v1567 : tensor<32x10xf32>
    %v1569 = stablehlo.constant dense<-0.010000> : tensor<32x10xf32>
    %v1570 = stablehlo.add %v1568, %v1569 : tensor<32x10xf32>
    %v1571 = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %v1572 = stablehlo.divide %v1570, %v1571 : tensor<32x10xf32>
    %v1573 = stablehlo.reshape %v1572 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1574 = stablehlo.dot_general %v1573, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x10xf32>, tensor<1280x10xf32>) -> tensor<32x1x1280xf32>
    %v1575 = stablehlo.reshape %v1574 : (tensor<32x1x1280xf32>) -> tensor<32x1280xf32>
    %v1576 = stablehlo.dot_general %v1554, %v1572, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<32x10xf32>) -> tensor<1280x10xf32>
    %v1577 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1578 = stablehlo.reduce(%v1572 init: %v1577) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1579 = stablehlo.broadcast_in_dim %v1575, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1580 = stablehlo.constant dense<49.0> : tensor<32x1280x7x7xf32>
    %v1581 = stablehlo.divide %v1579, %v1580 : tensor<32x1280x7x7xf32>
    %v1582 = stablehlo.reshape %v1581 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1583 = stablehlo.reshape %v1582 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1584 = stablehlo.reshape %v1543 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1585 = stablehlo.constant dense<0.0> : tensor<32x1280x7x7xf32>
    %v1586 = stablehlo.constant dense<6.0> : tensor<32x1280x7x7xf32>
    %v1587 = stablehlo.compare GT, %v1584, %v1585 : (tensor<32x1280x7x7xf32>, tensor<32x1280x7x7xf32>) -> tensor<32x1280x7x7xi1>
    %v1588 = stablehlo.compare LT, %v1584, %v1586 : (tensor<32x1280x7x7xf32>, tensor<32x1280x7x7xf32>) -> tensor<32x1280x7x7xi1>
    %v1589 = stablehlo.and %v1587, %v1588 : tensor<32x1280x7x7xi1>
    %v1590 = stablehlo.select %v1589, %v1583, %v1585 : tensor<32x1280x7x7xi1>, tensor<32x1280x7x7xf32>
    %v1591 = stablehlo.reshape %v1590 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1592 = stablehlo.reshape %v1523 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1594 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1595 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1596 = stablehlo.reduce(%v1592 init: %v1593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1597 = stablehlo.broadcast_in_dim %v1596, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1598 = stablehlo.divide %v1597, %v1594 : tensor<32x1280x7x7xf32>
    %v1599 = stablehlo.subtract %v1592, %v1598 : tensor<32x1280x7x7xf32>
    %v1600 = stablehlo.multiply %v1599, %v1599 : tensor<32x1280x7x7xf32>
    %v1601 = stablehlo.reduce(%v1600 init: %v1593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1602 = stablehlo.broadcast_in_dim %v1601, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1603 = stablehlo.divide %v1602, %v1594 : tensor<32x1280x7x7xf32>
    %v1604 = stablehlo.add %v1603, %v1595 : tensor<32x1280x7x7xf32>
    %v1605 = stablehlo.rsqrt %v1604 : tensor<32x1280x7x7xf32>
    %v1606 = stablehlo.multiply %v1599, %v1605 : tensor<32x1280x7x7xf32>
    %v1607 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1608 = stablehlo.reshape %v1591 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1609 = stablehlo.multiply %v1607, %v1608 : tensor<32x1280x7x7xf32>
    %v1610 = stablehlo.reduce(%v1609 init: %v1593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1611 = stablehlo.broadcast_in_dim %v1610, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1612 = stablehlo.multiply %v1606, %v1609 : tensor<32x1280x7x7xf32>
    %v1613 = stablehlo.reduce(%v1612 init: %v1593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1614 = stablehlo.broadcast_in_dim %v1613, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1615 = stablehlo.multiply %v1609, %v1594 : tensor<32x1280x7x7xf32>
    %v1616 = stablehlo.subtract %v1615, %v1611 : tensor<32x1280x7x7xf32>
    %v1617 = stablehlo.multiply %v1606, %v1614 : tensor<32x1280x7x7xf32>
    %v1618 = stablehlo.subtract %v1616, %v1617 : tensor<32x1280x7x7xf32>
    %v1619 = stablehlo.divide %v1605, %v1594 : tensor<32x1280x7x7xf32>
    %v1620 = stablehlo.multiply %v1619, %v1618 : tensor<32x1280x7x7xf32>
    %v1621 = stablehlo.reshape %v1620 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1622 = stablehlo.reshape %v1621 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1623 = stablehlo.reverse %hW, dims = [2, 3] : tensor<1280x320x1x1xf32>
    %v1624 = stablehlo.transpose %v1623, dims = [1, 0, 2, 3] : (tensor<1280x320x1x1xf32>) -> tensor<320x1280x1x1xf32>
    %v1625 = stablehlo.convolution(%v1622, %v1624)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1280x7x7xf32>, tensor<320x1280x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1626 = stablehlo.reshape %v1625 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1627 = stablehlo.reshape %v1518 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1628 = stablehlo.reshape %v1621 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1629 = stablehlo.transpose %v1627, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v1630 = stablehlo.transpose %v1628, dims = [1, 0, 2, 3] : (tensor<32x1280x7x7xf32>) -> tensor<1280x32x7x7xf32>
    %v1631 = stablehlo.convolution(%v1629, %v1630)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<320x32x7x7xf32>, tensor<1280x32x7x7xf32>) -> tensor<320x1280x1x1xf32>
    %v1632 = stablehlo.transpose %v1631, dims = [1, 0, 2, 3] : (tensor<320x1280x1x1xf32>) -> tensor<1280x320x1x1xf32>
    %v1633 = stablehlo.reshape %v1523 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1634 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1635 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1636 = stablehlo.reduce(%v1633 init: %v1634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1637 = stablehlo.broadcast_in_dim %v1636, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1638 = stablehlo.divide %v1637, %v1635 : tensor<32x1280x7x7xf32>
    %v1639 = stablehlo.subtract %v1633, %v1638 : tensor<32x1280x7x7xf32>
    %v1640 = stablehlo.multiply %v1639, %v1639 : tensor<32x1280x7x7xf32>
    %v1641 = stablehlo.reduce(%v1640 init: %v1634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1642 = stablehlo.broadcast_in_dim %v1641, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1643 = stablehlo.divide %v1642, %v1635 : tensor<32x1280x7x7xf32>
    %v1644 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1645 = stablehlo.add %v1643, %v1644 : tensor<32x1280x7x7xf32>
    %v1646 = stablehlo.rsqrt %v1645 : tensor<32x1280x7x7xf32>
    %v1647 = stablehlo.multiply %v1639, %v1646 : tensor<32x1280x7x7xf32>
    %v1648 = stablehlo.reshape %v1591 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1649 = stablehlo.multiply %v1648, %v1647 : tensor<32x1280x7x7xf32>
    %v1650 = stablehlo.reduce(%v1649 init: %v1634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1651 = stablehlo.reshape %v1591 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1652 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1653 = stablehlo.reduce(%v1651 init: %v1652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1654 = stablehlo.reshape %v1498 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1655 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1656 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1657 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1658 = stablehlo.reduce(%v1654 init: %v1655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1659 = stablehlo.broadcast_in_dim %v1658, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1660 = stablehlo.divide %v1659, %v1656 : tensor<32x320x7x7xf32>
    %v1661 = stablehlo.subtract %v1654, %v1660 : tensor<32x320x7x7xf32>
    %v1662 = stablehlo.multiply %v1661, %v1661 : tensor<32x320x7x7xf32>
    %v1663 = stablehlo.reduce(%v1662 init: %v1655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1664 = stablehlo.broadcast_in_dim %v1663, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1665 = stablehlo.divide %v1664, %v1656 : tensor<32x320x7x7xf32>
    %v1666 = stablehlo.add %v1665, %v1657 : tensor<32x320x7x7xf32>
    %v1667 = stablehlo.rsqrt %v1666 : tensor<32x320x7x7xf32>
    %v1668 = stablehlo.multiply %v1661, %v1667 : tensor<32x320x7x7xf32>
    %v1669 = stablehlo.broadcast_in_dim %b17pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1670 = stablehlo.reshape %v1626 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1671 = stablehlo.multiply %v1669, %v1670 : tensor<32x320x7x7xf32>
    %v1672 = stablehlo.reduce(%v1671 init: %v1655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1673 = stablehlo.broadcast_in_dim %v1672, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1674 = stablehlo.multiply %v1668, %v1671 : tensor<32x320x7x7xf32>
    %v1675 = stablehlo.reduce(%v1674 init: %v1655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1676 = stablehlo.broadcast_in_dim %v1675, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1677 = stablehlo.multiply %v1671, %v1656 : tensor<32x320x7x7xf32>
    %v1678 = stablehlo.subtract %v1677, %v1673 : tensor<32x320x7x7xf32>
    %v1679 = stablehlo.multiply %v1668, %v1676 : tensor<32x320x7x7xf32>
    %v1680 = stablehlo.subtract %v1678, %v1679 : tensor<32x320x7x7xf32>
    %v1681 = stablehlo.divide %v1667, %v1656 : tensor<32x320x7x7xf32>
    %v1682 = stablehlo.multiply %v1681, %v1680 : tensor<32x320x7x7xf32>
    %v1683 = stablehlo.reshape %v1682 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1684 = stablehlo.reshape %v1683 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1685 = stablehlo.reverse %b17pW, dims = [2, 3] : tensor<320x960x1x1xf32>
    %v1686 = stablehlo.transpose %v1685, dims = [1, 0, 2, 3] : (tensor<320x960x1x1xf32>) -> tensor<960x320x1x1xf32>
    %v1687 = stablehlo.convolution(%v1684, %v1686)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<960x320x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1688 = stablehlo.reshape %v1687 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1689 = stablehlo.reshape %v1688 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1690 = stablehlo.reshape %v1487 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1691 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1692 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1693 = stablehlo.compare GT, %v1690, %v1691 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1694 = stablehlo.compare LT, %v1690, %v1692 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1695 = stablehlo.and %v1693, %v1694 : tensor<32x960x7x7xi1>
    %v1696 = stablehlo.select %v1695, %v1689, %v1691 : tensor<32x960x7x7xi1>, tensor<32x960x7x7xf32>
    %v1697 = stablehlo.reshape %v1696 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1698 = stablehlo.reshape %v1467 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1699 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1700 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1701 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1702 = stablehlo.reduce(%v1698 init: %v1699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1703 = stablehlo.broadcast_in_dim %v1702, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1704 = stablehlo.divide %v1703, %v1700 : tensor<32x960x7x7xf32>
    %v1705 = stablehlo.subtract %v1698, %v1704 : tensor<32x960x7x7xf32>
    %v1706 = stablehlo.multiply %v1705, %v1705 : tensor<32x960x7x7xf32>
    %v1707 = stablehlo.reduce(%v1706 init: %v1699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1708 = stablehlo.broadcast_in_dim %v1707, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1709 = stablehlo.divide %v1708, %v1700 : tensor<32x960x7x7xf32>
    %v1710 = stablehlo.add %v1709, %v1701 : tensor<32x960x7x7xf32>
    %v1711 = stablehlo.rsqrt %v1710 : tensor<32x960x7x7xf32>
    %v1712 = stablehlo.multiply %v1705, %v1711 : tensor<32x960x7x7xf32>
    %v1713 = stablehlo.broadcast_in_dim %b17dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1714 = stablehlo.reshape %v1697 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1715 = stablehlo.multiply %v1713, %v1714 : tensor<32x960x7x7xf32>
    %v1716 = stablehlo.reduce(%v1715 init: %v1699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1717 = stablehlo.broadcast_in_dim %v1716, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1718 = stablehlo.multiply %v1712, %v1715 : tensor<32x960x7x7xf32>
    %v1719 = stablehlo.reduce(%v1718 init: %v1699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1720 = stablehlo.broadcast_in_dim %v1719, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1721 = stablehlo.multiply %v1715, %v1700 : tensor<32x960x7x7xf32>
    %v1722 = stablehlo.subtract %v1721, %v1717 : tensor<32x960x7x7xf32>
    %v1723 = stablehlo.multiply %v1712, %v1720 : tensor<32x960x7x7xf32>
    %v1724 = stablehlo.subtract %v1722, %v1723 : tensor<32x960x7x7xf32>
    %v1725 = stablehlo.divide %v1711, %v1700 : tensor<32x960x7x7xf32>
    %v1726 = stablehlo.multiply %v1725, %v1724 : tensor<32x960x7x7xf32>
    %v1727 = stablehlo.reshape %v1726 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1728 = stablehlo.reshape %v1727 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1729 = stablehlo.reverse %b17dW, dims = [2, 3] : tensor<960x1x3x3xf32>
    %v1730 = stablehlo.convolution(%v1728, %v1729)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1731 = stablehlo.reshape %v1730 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1732 = stablehlo.reshape %v1731 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1733 = stablehlo.reshape %v1456 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1734 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1735 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1736 = stablehlo.compare GT, %v1733, %v1734 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1737 = stablehlo.compare LT, %v1733, %v1735 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1738 = stablehlo.and %v1736, %v1737 : tensor<32x960x7x7xi1>
    %v1739 = stablehlo.select %v1738, %v1732, %v1734 : tensor<32x960x7x7xi1>, tensor<32x960x7x7xf32>
    %v1740 = stablehlo.reshape %v1739 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1741 = stablehlo.reshape %v1436 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1743 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1744 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1745 = stablehlo.reduce(%v1741 init: %v1742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1746 = stablehlo.broadcast_in_dim %v1745, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1747 = stablehlo.divide %v1746, %v1743 : tensor<32x960x7x7xf32>
    %v1748 = stablehlo.subtract %v1741, %v1747 : tensor<32x960x7x7xf32>
    %v1749 = stablehlo.multiply %v1748, %v1748 : tensor<32x960x7x7xf32>
    %v1750 = stablehlo.reduce(%v1749 init: %v1742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1751 = stablehlo.broadcast_in_dim %v1750, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1752 = stablehlo.divide %v1751, %v1743 : tensor<32x960x7x7xf32>
    %v1753 = stablehlo.add %v1752, %v1744 : tensor<32x960x7x7xf32>
    %v1754 = stablehlo.rsqrt %v1753 : tensor<32x960x7x7xf32>
    %v1755 = stablehlo.multiply %v1748, %v1754 : tensor<32x960x7x7xf32>
    %v1756 = stablehlo.broadcast_in_dim %b17eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1757 = stablehlo.reshape %v1740 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1758 = stablehlo.multiply %v1756, %v1757 : tensor<32x960x7x7xf32>
    %v1759 = stablehlo.reduce(%v1758 init: %v1742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1760 = stablehlo.broadcast_in_dim %v1759, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1761 = stablehlo.multiply %v1755, %v1758 : tensor<32x960x7x7xf32>
    %v1762 = stablehlo.reduce(%v1761 init: %v1742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1763 = stablehlo.broadcast_in_dim %v1762, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1764 = stablehlo.multiply %v1758, %v1743 : tensor<32x960x7x7xf32>
    %v1765 = stablehlo.subtract %v1764, %v1760 : tensor<32x960x7x7xf32>
    %v1766 = stablehlo.multiply %v1755, %v1763 : tensor<32x960x7x7xf32>
    %v1767 = stablehlo.subtract %v1765, %v1766 : tensor<32x960x7x7xf32>
    %v1768 = stablehlo.divide %v1754, %v1743 : tensor<32x960x7x7xf32>
    %v1769 = stablehlo.multiply %v1768, %v1767 : tensor<32x960x7x7xf32>
    %v1770 = stablehlo.reshape %v1769 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1771 = stablehlo.reshape %v1770 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1772 = stablehlo.reverse %b17eW, dims = [2, 3] : tensor<960x160x1x1xf32>
    %v1773 = stablehlo.transpose %v1772, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v1774 = stablehlo.convolution(%v1771, %v1773)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1775 = stablehlo.reshape %v1774 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1776 = stablehlo.reshape %v1431 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1777 = stablehlo.reshape %v1770 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1778 = stablehlo.transpose %v1776, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v1779 = stablehlo.transpose %v1777, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1780 = stablehlo.convolution(%v1778, %v1779)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<160x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<160x960x1x1xf32>
    %v1781 = stablehlo.transpose %v1780, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v1782 = stablehlo.reshape %v1436 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1784 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1785 = stablehlo.reduce(%v1782 init: %v1783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1786 = stablehlo.broadcast_in_dim %v1785, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1787 = stablehlo.divide %v1786, %v1784 : tensor<32x960x7x7xf32>
    %v1788 = stablehlo.subtract %v1782, %v1787 : tensor<32x960x7x7xf32>
    %v1789 = stablehlo.multiply %v1788, %v1788 : tensor<32x960x7x7xf32>
    %v1790 = stablehlo.reduce(%v1789 init: %v1783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1791 = stablehlo.broadcast_in_dim %v1790, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1792 = stablehlo.divide %v1791, %v1784 : tensor<32x960x7x7xf32>
    %v1793 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1794 = stablehlo.add %v1792, %v1793 : tensor<32x960x7x7xf32>
    %v1795 = stablehlo.rsqrt %v1794 : tensor<32x960x7x7xf32>
    %v1796 = stablehlo.multiply %v1788, %v1795 : tensor<32x960x7x7xf32>
    %v1797 = stablehlo.reshape %v1740 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1798 = stablehlo.multiply %v1797, %v1796 : tensor<32x960x7x7xf32>
    %v1799 = stablehlo.reduce(%v1798 init: %v1783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1800 = stablehlo.reshape %v1740 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1801 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1802 = stablehlo.reduce(%v1800 init: %v1801) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1803 = stablehlo.reshape %v1462 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1804 = stablehlo.reshape %v1727 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1805 = stablehlo.transpose %v1803, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1806 = stablehlo.transpose %v1804, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1807 = stablehlo.convolution(%v1805, %v1806)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 960 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<1x960x3x3xf32>
    %v1808 = stablehlo.reshape %v1807 : (tensor<1x960x3x3xf32>) -> tensor<960x1x3x3xf32>
    %v1809 = stablehlo.reshape %v1467 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1811 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1812 = stablehlo.reduce(%v1809 init: %v1810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1813 = stablehlo.broadcast_in_dim %v1812, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1814 = stablehlo.divide %v1813, %v1811 : tensor<32x960x7x7xf32>
    %v1815 = stablehlo.subtract %v1809, %v1814 : tensor<32x960x7x7xf32>
    %v1816 = stablehlo.multiply %v1815, %v1815 : tensor<32x960x7x7xf32>
    %v1817 = stablehlo.reduce(%v1816 init: %v1810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1818 = stablehlo.broadcast_in_dim %v1817, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1819 = stablehlo.divide %v1818, %v1811 : tensor<32x960x7x7xf32>
    %v1820 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1821 = stablehlo.add %v1819, %v1820 : tensor<32x960x7x7xf32>
    %v1822 = stablehlo.rsqrt %v1821 : tensor<32x960x7x7xf32>
    %v1823 = stablehlo.multiply %v1815, %v1822 : tensor<32x960x7x7xf32>
    %v1824 = stablehlo.reshape %v1697 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1825 = stablehlo.multiply %v1824, %v1823 : tensor<32x960x7x7xf32>
    %v1826 = stablehlo.reduce(%v1825 init: %v1810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1827 = stablehlo.reshape %v1697 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1828 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1829 = stablehlo.reduce(%v1827 init: %v1828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1830 = stablehlo.reshape %v1493 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1831 = stablehlo.reshape %v1683 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1832 = stablehlo.transpose %v1830, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1833 = stablehlo.transpose %v1831, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v1834 = stablehlo.convolution(%v1832, %v1833)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<320x32x7x7xf32>) -> tensor<960x320x1x1xf32>
    %v1835 = stablehlo.transpose %v1834, dims = [1, 0, 2, 3] : (tensor<960x320x1x1xf32>) -> tensor<320x960x1x1xf32>
    %v1836 = stablehlo.reshape %v1498 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1838 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1839 = stablehlo.reduce(%v1836 init: %v1837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1840 = stablehlo.broadcast_in_dim %v1839, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1841 = stablehlo.divide %v1840, %v1838 : tensor<32x320x7x7xf32>
    %v1842 = stablehlo.subtract %v1836, %v1841 : tensor<32x320x7x7xf32>
    %v1843 = stablehlo.multiply %v1842, %v1842 : tensor<32x320x7x7xf32>
    %v1844 = stablehlo.reduce(%v1843 init: %v1837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1845 = stablehlo.broadcast_in_dim %v1844, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1846 = stablehlo.divide %v1845, %v1838 : tensor<32x320x7x7xf32>
    %v1847 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1848 = stablehlo.add %v1846, %v1847 : tensor<32x320x7x7xf32>
    %v1849 = stablehlo.rsqrt %v1848 : tensor<32x320x7x7xf32>
    %v1850 = stablehlo.multiply %v1842, %v1849 : tensor<32x320x7x7xf32>
    %v1851 = stablehlo.reshape %v1626 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1852 = stablehlo.multiply %v1851, %v1850 : tensor<32x320x7x7xf32>
    %v1853 = stablehlo.reduce(%v1852 init: %v1837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1854 = stablehlo.reshape %v1626 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1855 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1856 = stablehlo.reduce(%v1854 init: %v1855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1857 = stablehlo.reshape %v1407 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1858 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1859 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v1860 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1861 = stablehlo.reduce(%v1857 init: %v1858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1862 = stablehlo.broadcast_in_dim %v1861, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1863 = stablehlo.divide %v1862, %v1859 : tensor<32x160x7x7xf32>
    %v1864 = stablehlo.subtract %v1857, %v1863 : tensor<32x160x7x7xf32>
    %v1865 = stablehlo.multiply %v1864, %v1864 : tensor<32x160x7x7xf32>
    %v1866 = stablehlo.reduce(%v1865 init: %v1858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1867 = stablehlo.broadcast_in_dim %v1866, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1868 = stablehlo.divide %v1867, %v1859 : tensor<32x160x7x7xf32>
    %v1869 = stablehlo.add %v1868, %v1860 : tensor<32x160x7x7xf32>
    %v1870 = stablehlo.rsqrt %v1869 : tensor<32x160x7x7xf32>
    %v1871 = stablehlo.multiply %v1864, %v1870 : tensor<32x160x7x7xf32>
    %v1872 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1873 = stablehlo.reshape %v1775 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1874 = stablehlo.multiply %v1872, %v1873 : tensor<32x160x7x7xf32>
    %v1875 = stablehlo.reduce(%v1874 init: %v1858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1876 = stablehlo.broadcast_in_dim %v1875, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1877 = stablehlo.multiply %v1871, %v1874 : tensor<32x160x7x7xf32>
    %v1878 = stablehlo.reduce(%v1877 init: %v1858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1879 = stablehlo.broadcast_in_dim %v1878, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1880 = stablehlo.multiply %v1874, %v1859 : tensor<32x160x7x7xf32>
    %v1881 = stablehlo.subtract %v1880, %v1876 : tensor<32x160x7x7xf32>
    %v1882 = stablehlo.multiply %v1871, %v1879 : tensor<32x160x7x7xf32>
    %v1883 = stablehlo.subtract %v1881, %v1882 : tensor<32x160x7x7xf32>
    %v1884 = stablehlo.divide %v1870, %v1859 : tensor<32x160x7x7xf32>
    %v1885 = stablehlo.multiply %v1884, %v1883 : tensor<32x160x7x7xf32>
    %v1886 = stablehlo.reshape %v1885 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1887 = stablehlo.reshape %v1886 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1888 = stablehlo.reverse %b16pW, dims = [2, 3] : tensor<160x960x1x1xf32>
    %v1889 = stablehlo.transpose %v1888, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v1890 = stablehlo.convolution(%v1887, %v1889)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1891 = stablehlo.reshape %v1890 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1892 = stablehlo.reshape %v1891 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1893 = stablehlo.reshape %v1396 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1894 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1895 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1896 = stablehlo.compare GT, %v1893, %v1894 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1897 = stablehlo.compare LT, %v1893, %v1895 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1898 = stablehlo.and %v1896, %v1897 : tensor<32x960x7x7xi1>
    %v1899 = stablehlo.select %v1898, %v1892, %v1894 : tensor<32x960x7x7xi1>, tensor<32x960x7x7xf32>
    %v1900 = stablehlo.reshape %v1899 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1901 = stablehlo.reshape %v1376 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1902 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1903 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1904 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1905 = stablehlo.reduce(%v1901 init: %v1902) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1906 = stablehlo.broadcast_in_dim %v1905, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1907 = stablehlo.divide %v1906, %v1903 : tensor<32x960x7x7xf32>
    %v1908 = stablehlo.subtract %v1901, %v1907 : tensor<32x960x7x7xf32>
    %v1909 = stablehlo.multiply %v1908, %v1908 : tensor<32x960x7x7xf32>
    %v1910 = stablehlo.reduce(%v1909 init: %v1902) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1911 = stablehlo.broadcast_in_dim %v1910, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1912 = stablehlo.divide %v1911, %v1903 : tensor<32x960x7x7xf32>
    %v1913 = stablehlo.add %v1912, %v1904 : tensor<32x960x7x7xf32>
    %v1914 = stablehlo.rsqrt %v1913 : tensor<32x960x7x7xf32>
    %v1915 = stablehlo.multiply %v1908, %v1914 : tensor<32x960x7x7xf32>
    %v1916 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1917 = stablehlo.reshape %v1900 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1918 = stablehlo.multiply %v1916, %v1917 : tensor<32x960x7x7xf32>
    %v1919 = stablehlo.reduce(%v1918 init: %v1902) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1920 = stablehlo.broadcast_in_dim %v1919, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1921 = stablehlo.multiply %v1915, %v1918 : tensor<32x960x7x7xf32>
    %v1922 = stablehlo.reduce(%v1921 init: %v1902) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1923 = stablehlo.broadcast_in_dim %v1922, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1924 = stablehlo.multiply %v1918, %v1903 : tensor<32x960x7x7xf32>
    %v1925 = stablehlo.subtract %v1924, %v1920 : tensor<32x960x7x7xf32>
    %v1926 = stablehlo.multiply %v1915, %v1923 : tensor<32x960x7x7xf32>
    %v1927 = stablehlo.subtract %v1925, %v1926 : tensor<32x960x7x7xf32>
    %v1928 = stablehlo.divide %v1914, %v1903 : tensor<32x960x7x7xf32>
    %v1929 = stablehlo.multiply %v1928, %v1927 : tensor<32x960x7x7xf32>
    %v1930 = stablehlo.reshape %v1929 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1931 = stablehlo.reshape %v1930 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1932 = stablehlo.reverse %b16dW, dims = [2, 3] : tensor<960x1x3x3xf32>
    %v1933 = stablehlo.convolution(%v1931, %v1932)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1934 = stablehlo.reshape %v1933 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1935 = stablehlo.reshape %v1934 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1936 = stablehlo.reshape %v1365 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1937 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1938 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1939 = stablehlo.compare GT, %v1936, %v1937 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1940 = stablehlo.compare LT, %v1936, %v1938 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1941 = stablehlo.and %v1939, %v1940 : tensor<32x960x7x7xi1>
    %v1942 = stablehlo.select %v1941, %v1935, %v1937 : tensor<32x960x7x7xi1>, tensor<32x960x7x7xf32>
    %v1943 = stablehlo.reshape %v1942 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1944 = stablehlo.reshape %v1345 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1946 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1947 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1948 = stablehlo.reduce(%v1944 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1949 = stablehlo.broadcast_in_dim %v1948, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1950 = stablehlo.divide %v1949, %v1946 : tensor<32x960x7x7xf32>
    %v1951 = stablehlo.subtract %v1944, %v1950 : tensor<32x960x7x7xf32>
    %v1952 = stablehlo.multiply %v1951, %v1951 : tensor<32x960x7x7xf32>
    %v1953 = stablehlo.reduce(%v1952 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1954 = stablehlo.broadcast_in_dim %v1953, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1955 = stablehlo.divide %v1954, %v1946 : tensor<32x960x7x7xf32>
    %v1956 = stablehlo.add %v1955, %v1947 : tensor<32x960x7x7xf32>
    %v1957 = stablehlo.rsqrt %v1956 : tensor<32x960x7x7xf32>
    %v1958 = stablehlo.multiply %v1951, %v1957 : tensor<32x960x7x7xf32>
    %v1959 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1960 = stablehlo.reshape %v1943 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1961 = stablehlo.multiply %v1959, %v1960 : tensor<32x960x7x7xf32>
    %v1962 = stablehlo.reduce(%v1961 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1963 = stablehlo.broadcast_in_dim %v1962, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1964 = stablehlo.multiply %v1958, %v1961 : tensor<32x960x7x7xf32>
    %v1965 = stablehlo.reduce(%v1964 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1966 = stablehlo.broadcast_in_dim %v1965, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1967 = stablehlo.multiply %v1961, %v1946 : tensor<32x960x7x7xf32>
    %v1968 = stablehlo.subtract %v1967, %v1963 : tensor<32x960x7x7xf32>
    %v1969 = stablehlo.multiply %v1958, %v1966 : tensor<32x960x7x7xf32>
    %v1970 = stablehlo.subtract %v1968, %v1969 : tensor<32x960x7x7xf32>
    %v1971 = stablehlo.divide %v1957, %v1946 : tensor<32x960x7x7xf32>
    %v1972 = stablehlo.multiply %v1971, %v1970 : tensor<32x960x7x7xf32>
    %v1973 = stablehlo.reshape %v1972 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1974 = stablehlo.reshape %v1973 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1975 = stablehlo.reverse %b16eW, dims = [2, 3] : tensor<960x160x1x1xf32>
    %v1976 = stablehlo.transpose %v1975, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v1977 = stablehlo.convolution(%v1974, %v1976)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1978 = stablehlo.reshape %v1977 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1979 = stablehlo.reshape %v1978 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1980 = stablehlo.reshape %v1775 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1981 = stablehlo.add %v1979, %v1980 : tensor<32x160x7x7xf32>
    %v1982 = stablehlo.reshape %v1981 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1983 = stablehlo.reshape %v1340 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1984 = stablehlo.reshape %v1973 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1985 = stablehlo.transpose %v1983, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v1986 = stablehlo.transpose %v1984, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1987 = stablehlo.convolution(%v1985, %v1986)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<160x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<160x960x1x1xf32>
    %v1988 = stablehlo.transpose %v1987, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v1989 = stablehlo.reshape %v1345 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1991 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1992 = stablehlo.reduce(%v1989 init: %v1990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1993 = stablehlo.broadcast_in_dim %v1992, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1994 = stablehlo.divide %v1993, %v1991 : tensor<32x960x7x7xf32>
    %v1995 = stablehlo.subtract %v1989, %v1994 : tensor<32x960x7x7xf32>
    %v1996 = stablehlo.multiply %v1995, %v1995 : tensor<32x960x7x7xf32>
    %v1997 = stablehlo.reduce(%v1996 init: %v1990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1998 = stablehlo.broadcast_in_dim %v1997, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1999 = stablehlo.divide %v1998, %v1991 : tensor<32x960x7x7xf32>
    %v2000 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2001 = stablehlo.add %v1999, %v2000 : tensor<32x960x7x7xf32>
    %v2002 = stablehlo.rsqrt %v2001 : tensor<32x960x7x7xf32>
    %v2003 = stablehlo.multiply %v1995, %v2002 : tensor<32x960x7x7xf32>
    %v2004 = stablehlo.reshape %v1943 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2005 = stablehlo.multiply %v2004, %v2003 : tensor<32x960x7x7xf32>
    %v2006 = stablehlo.reduce(%v2005 init: %v1990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2007 = stablehlo.reshape %v1943 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2008 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2009 = stablehlo.reduce(%v2007 init: %v2008) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2010 = stablehlo.reshape %v1371 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2011 = stablehlo.reshape %v1930 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2012 = stablehlo.transpose %v2010, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2013 = stablehlo.transpose %v2011, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2014 = stablehlo.convolution(%v2012, %v2013)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 960 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<1x960x3x3xf32>
    %v2015 = stablehlo.reshape %v2014 : (tensor<1x960x3x3xf32>) -> tensor<960x1x3x3xf32>
    %v2016 = stablehlo.reshape %v1376 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2017 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2018 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v2019 = stablehlo.reduce(%v2016 init: %v2017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2020 = stablehlo.broadcast_in_dim %v2019, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2021 = stablehlo.divide %v2020, %v2018 : tensor<32x960x7x7xf32>
    %v2022 = stablehlo.subtract %v2016, %v2021 : tensor<32x960x7x7xf32>
    %v2023 = stablehlo.multiply %v2022, %v2022 : tensor<32x960x7x7xf32>
    %v2024 = stablehlo.reduce(%v2023 init: %v2017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2025 = stablehlo.broadcast_in_dim %v2024, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2026 = stablehlo.divide %v2025, %v2018 : tensor<32x960x7x7xf32>
    %v2027 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2028 = stablehlo.add %v2026, %v2027 : tensor<32x960x7x7xf32>
    %v2029 = stablehlo.rsqrt %v2028 : tensor<32x960x7x7xf32>
    %v2030 = stablehlo.multiply %v2022, %v2029 : tensor<32x960x7x7xf32>
    %v2031 = stablehlo.reshape %v1900 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2032 = stablehlo.multiply %v2031, %v2030 : tensor<32x960x7x7xf32>
    %v2033 = stablehlo.reduce(%v2032 init: %v2017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2034 = stablehlo.reshape %v1900 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2035 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2036 = stablehlo.reduce(%v2034 init: %v2035) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2037 = stablehlo.reshape %v1402 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2038 = stablehlo.reshape %v1886 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2039 = stablehlo.transpose %v2037, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2040 = stablehlo.transpose %v2038, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2041 = stablehlo.convolution(%v2039, %v2040)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<960x160x1x1xf32>
    %v2042 = stablehlo.transpose %v2041, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v2043 = stablehlo.reshape %v1407 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2045 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v2046 = stablehlo.reduce(%v2043 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2047 = stablehlo.broadcast_in_dim %v2046, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2048 = stablehlo.divide %v2047, %v2045 : tensor<32x160x7x7xf32>
    %v2049 = stablehlo.subtract %v2043, %v2048 : tensor<32x160x7x7xf32>
    %v2050 = stablehlo.multiply %v2049, %v2049 : tensor<32x160x7x7xf32>
    %v2051 = stablehlo.reduce(%v2050 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2052 = stablehlo.broadcast_in_dim %v2051, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2053 = stablehlo.divide %v2052, %v2045 : tensor<32x160x7x7xf32>
    %v2054 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2055 = stablehlo.add %v2053, %v2054 : tensor<32x160x7x7xf32>
    %v2056 = stablehlo.rsqrt %v2055 : tensor<32x160x7x7xf32>
    %v2057 = stablehlo.multiply %v2049, %v2056 : tensor<32x160x7x7xf32>
    %v2058 = stablehlo.reshape %v1775 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2059 = stablehlo.multiply %v2058, %v2057 : tensor<32x160x7x7xf32>
    %v2060 = stablehlo.reduce(%v2059 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2061 = stablehlo.reshape %v1775 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2062 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2063 = stablehlo.reduce(%v2061 init: %v2062) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2064 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2065 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2066 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v2067 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2068 = stablehlo.reduce(%v2064 init: %v2065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2069 = stablehlo.broadcast_in_dim %v2068, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2070 = stablehlo.divide %v2069, %v2066 : tensor<32x160x7x7xf32>
    %v2071 = stablehlo.subtract %v2064, %v2070 : tensor<32x160x7x7xf32>
    %v2072 = stablehlo.multiply %v2071, %v2071 : tensor<32x160x7x7xf32>
    %v2073 = stablehlo.reduce(%v2072 init: %v2065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2074 = stablehlo.broadcast_in_dim %v2073, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2075 = stablehlo.divide %v2074, %v2066 : tensor<32x160x7x7xf32>
    %v2076 = stablehlo.add %v2075, %v2067 : tensor<32x160x7x7xf32>
    %v2077 = stablehlo.rsqrt %v2076 : tensor<32x160x7x7xf32>
    %v2078 = stablehlo.multiply %v2071, %v2077 : tensor<32x160x7x7xf32>
    %v2079 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2080 = stablehlo.reshape %v1982 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2081 = stablehlo.multiply %v2079, %v2080 : tensor<32x160x7x7xf32>
    %v2082 = stablehlo.reduce(%v2081 init: %v2065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2083 = stablehlo.broadcast_in_dim %v2082, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2084 = stablehlo.multiply %v2078, %v2081 : tensor<32x160x7x7xf32>
    %v2085 = stablehlo.reduce(%v2084 init: %v2065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2086 = stablehlo.broadcast_in_dim %v2085, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2087 = stablehlo.multiply %v2081, %v2066 : tensor<32x160x7x7xf32>
    %v2088 = stablehlo.subtract %v2087, %v2083 : tensor<32x160x7x7xf32>
    %v2089 = stablehlo.multiply %v2078, %v2086 : tensor<32x160x7x7xf32>
    %v2090 = stablehlo.subtract %v2088, %v2089 : tensor<32x160x7x7xf32>
    %v2091 = stablehlo.divide %v2077, %v2066 : tensor<32x160x7x7xf32>
    %v2092 = stablehlo.multiply %v2091, %v2090 : tensor<32x160x7x7xf32>
    %v2093 = stablehlo.reshape %v2092 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2094 = stablehlo.reshape %v2093 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2095 = stablehlo.reverse %b15pW, dims = [2, 3] : tensor<160x960x1x1xf32>
    %v2096 = stablehlo.transpose %v2095, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v2097 = stablehlo.convolution(%v2094, %v2096)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v2098 = stablehlo.reshape %v2097 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2099 = stablehlo.reshape %v2098 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2100 = stablehlo.reshape %v1305 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2101 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v2102 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v2103 = stablehlo.compare GT, %v2100, %v2101 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v2104 = stablehlo.compare LT, %v2100, %v2102 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v2105 = stablehlo.and %v2103, %v2104 : tensor<32x960x7x7xi1>
    %v2106 = stablehlo.select %v2105, %v2099, %v2101 : tensor<32x960x7x7xi1>, tensor<32x960x7x7xf32>
    %v2107 = stablehlo.reshape %v2106 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2108 = stablehlo.reshape %v1285 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2110 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v2111 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2112 = stablehlo.reduce(%v2108 init: %v2109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2113 = stablehlo.broadcast_in_dim %v2112, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2114 = stablehlo.divide %v2113, %v2110 : tensor<32x960x7x7xf32>
    %v2115 = stablehlo.subtract %v2108, %v2114 : tensor<32x960x7x7xf32>
    %v2116 = stablehlo.multiply %v2115, %v2115 : tensor<32x960x7x7xf32>
    %v2117 = stablehlo.reduce(%v2116 init: %v2109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2118 = stablehlo.broadcast_in_dim %v2117, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2119 = stablehlo.divide %v2118, %v2110 : tensor<32x960x7x7xf32>
    %v2120 = stablehlo.add %v2119, %v2111 : tensor<32x960x7x7xf32>
    %v2121 = stablehlo.rsqrt %v2120 : tensor<32x960x7x7xf32>
    %v2122 = stablehlo.multiply %v2115, %v2121 : tensor<32x960x7x7xf32>
    %v2123 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2124 = stablehlo.reshape %v2107 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2125 = stablehlo.multiply %v2123, %v2124 : tensor<32x960x7x7xf32>
    %v2126 = stablehlo.reduce(%v2125 init: %v2109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2127 = stablehlo.broadcast_in_dim %v2126, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2128 = stablehlo.multiply %v2122, %v2125 : tensor<32x960x7x7xf32>
    %v2129 = stablehlo.reduce(%v2128 init: %v2109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2130 = stablehlo.broadcast_in_dim %v2129, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2131 = stablehlo.multiply %v2125, %v2110 : tensor<32x960x7x7xf32>
    %v2132 = stablehlo.subtract %v2131, %v2127 : tensor<32x960x7x7xf32>
    %v2133 = stablehlo.multiply %v2122, %v2130 : tensor<32x960x7x7xf32>
    %v2134 = stablehlo.subtract %v2132, %v2133 : tensor<32x960x7x7xf32>
    %v2135 = stablehlo.divide %v2121, %v2110 : tensor<32x960x7x7xf32>
    %v2136 = stablehlo.multiply %v2135, %v2134 : tensor<32x960x7x7xf32>
    %v2137 = stablehlo.reshape %v2136 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2138 = stablehlo.reshape %v2137 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2139 = stablehlo.reverse %b15dW, dims = [2, 3] : tensor<960x1x3x3xf32>
    %v2140 = stablehlo.convolution(%v2138, %v2139)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v2141 = stablehlo.reshape %v2140 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2142 = stablehlo.reshape %v2141 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2143 = stablehlo.reshape %v1274 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2144 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v2145 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v2146 = stablehlo.compare GT, %v2143, %v2144 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v2147 = stablehlo.compare LT, %v2143, %v2145 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v2148 = stablehlo.and %v2146, %v2147 : tensor<32x960x7x7xi1>
    %v2149 = stablehlo.select %v2148, %v2142, %v2144 : tensor<32x960x7x7xi1>, tensor<32x960x7x7xf32>
    %v2150 = stablehlo.reshape %v2149 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2151 = stablehlo.reshape %v1254 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2152 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2153 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v2154 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2155 = stablehlo.reduce(%v2151 init: %v2152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2156 = stablehlo.broadcast_in_dim %v2155, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2157 = stablehlo.divide %v2156, %v2153 : tensor<32x960x7x7xf32>
    %v2158 = stablehlo.subtract %v2151, %v2157 : tensor<32x960x7x7xf32>
    %v2159 = stablehlo.multiply %v2158, %v2158 : tensor<32x960x7x7xf32>
    %v2160 = stablehlo.reduce(%v2159 init: %v2152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2161 = stablehlo.broadcast_in_dim %v2160, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2162 = stablehlo.divide %v2161, %v2153 : tensor<32x960x7x7xf32>
    %v2163 = stablehlo.add %v2162, %v2154 : tensor<32x960x7x7xf32>
    %v2164 = stablehlo.rsqrt %v2163 : tensor<32x960x7x7xf32>
    %v2165 = stablehlo.multiply %v2158, %v2164 : tensor<32x960x7x7xf32>
    %v2166 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2167 = stablehlo.reshape %v2150 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2168 = stablehlo.multiply %v2166, %v2167 : tensor<32x960x7x7xf32>
    %v2169 = stablehlo.reduce(%v2168 init: %v2152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2170 = stablehlo.broadcast_in_dim %v2169, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2171 = stablehlo.multiply %v2165, %v2168 : tensor<32x960x7x7xf32>
    %v2172 = stablehlo.reduce(%v2171 init: %v2152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2173 = stablehlo.broadcast_in_dim %v2172, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2174 = stablehlo.multiply %v2168, %v2153 : tensor<32x960x7x7xf32>
    %v2175 = stablehlo.subtract %v2174, %v2170 : tensor<32x960x7x7xf32>
    %v2176 = stablehlo.multiply %v2165, %v2173 : tensor<32x960x7x7xf32>
    %v2177 = stablehlo.subtract %v2175, %v2176 : tensor<32x960x7x7xf32>
    %v2178 = stablehlo.divide %v2164, %v2153 : tensor<32x960x7x7xf32>
    %v2179 = stablehlo.multiply %v2178, %v2177 : tensor<32x960x7x7xf32>
    %v2180 = stablehlo.reshape %v2179 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2181 = stablehlo.reshape %v2180 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2182 = stablehlo.reverse %b15eW, dims = [2, 3] : tensor<960x160x1x1xf32>
    %v2183 = stablehlo.transpose %v2182, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v2184 = stablehlo.convolution(%v2181, %v2183)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v2185 = stablehlo.reshape %v2184 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2186 = stablehlo.reshape %v2185 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2187 = stablehlo.reshape %v1982 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2188 = stablehlo.add %v2186, %v2187 : tensor<32x160x7x7xf32>
    %v2189 = stablehlo.reshape %v2188 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2190 = stablehlo.reshape %v1249 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2191 = stablehlo.reshape %v2180 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2192 = stablehlo.transpose %v2190, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2193 = stablehlo.transpose %v2191, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2194 = stablehlo.convolution(%v2192, %v2193)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<160x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<160x960x1x1xf32>
    %v2195 = stablehlo.transpose %v2194, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v2196 = stablehlo.reshape %v1254 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2198 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v2199 = stablehlo.reduce(%v2196 init: %v2197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2200 = stablehlo.broadcast_in_dim %v2199, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2201 = stablehlo.divide %v2200, %v2198 : tensor<32x960x7x7xf32>
    %v2202 = stablehlo.subtract %v2196, %v2201 : tensor<32x960x7x7xf32>
    %v2203 = stablehlo.multiply %v2202, %v2202 : tensor<32x960x7x7xf32>
    %v2204 = stablehlo.reduce(%v2203 init: %v2197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2205 = stablehlo.broadcast_in_dim %v2204, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2206 = stablehlo.divide %v2205, %v2198 : tensor<32x960x7x7xf32>
    %v2207 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2208 = stablehlo.add %v2206, %v2207 : tensor<32x960x7x7xf32>
    %v2209 = stablehlo.rsqrt %v2208 : tensor<32x960x7x7xf32>
    %v2210 = stablehlo.multiply %v2202, %v2209 : tensor<32x960x7x7xf32>
    %v2211 = stablehlo.reshape %v2150 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2212 = stablehlo.multiply %v2211, %v2210 : tensor<32x960x7x7xf32>
    %v2213 = stablehlo.reduce(%v2212 init: %v2197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2214 = stablehlo.reshape %v2150 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2216 = stablehlo.reduce(%v2214 init: %v2215) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2217 = stablehlo.reshape %v1280 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2218 = stablehlo.reshape %v2137 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2219 = stablehlo.transpose %v2217, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2220 = stablehlo.transpose %v2218, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2221 = stablehlo.convolution(%v2219, %v2220)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 960 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<1x960x3x3xf32>
    %v2222 = stablehlo.reshape %v2221 : (tensor<1x960x3x3xf32>) -> tensor<960x1x3x3xf32>
    %v2223 = stablehlo.reshape %v1285 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2224 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2225 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v2226 = stablehlo.reduce(%v2223 init: %v2224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2227 = stablehlo.broadcast_in_dim %v2226, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2228 = stablehlo.divide %v2227, %v2225 : tensor<32x960x7x7xf32>
    %v2229 = stablehlo.subtract %v2223, %v2228 : tensor<32x960x7x7xf32>
    %v2230 = stablehlo.multiply %v2229, %v2229 : tensor<32x960x7x7xf32>
    %v2231 = stablehlo.reduce(%v2230 init: %v2224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2232 = stablehlo.broadcast_in_dim %v2231, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2233 = stablehlo.divide %v2232, %v2225 : tensor<32x960x7x7xf32>
    %v2234 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2235 = stablehlo.add %v2233, %v2234 : tensor<32x960x7x7xf32>
    %v2236 = stablehlo.rsqrt %v2235 : tensor<32x960x7x7xf32>
    %v2237 = stablehlo.multiply %v2229, %v2236 : tensor<32x960x7x7xf32>
    %v2238 = stablehlo.reshape %v2107 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2239 = stablehlo.multiply %v2238, %v2237 : tensor<32x960x7x7xf32>
    %v2240 = stablehlo.reduce(%v2239 init: %v2224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2241 = stablehlo.reshape %v2107 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2243 = stablehlo.reduce(%v2241 init: %v2242) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2244 = stablehlo.reshape %v1311 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2245 = stablehlo.reshape %v2093 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2246 = stablehlo.transpose %v2244, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2247 = stablehlo.transpose %v2245, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2248 = stablehlo.convolution(%v2246, %v2247)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<960x160x1x1xf32>
    %v2249 = stablehlo.transpose %v2248, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v2250 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2252 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v2253 = stablehlo.reduce(%v2250 init: %v2251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2254 = stablehlo.broadcast_in_dim %v2253, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2255 = stablehlo.divide %v2254, %v2252 : tensor<32x160x7x7xf32>
    %v2256 = stablehlo.subtract %v2250, %v2255 : tensor<32x160x7x7xf32>
    %v2257 = stablehlo.multiply %v2256, %v2256 : tensor<32x160x7x7xf32>
    %v2258 = stablehlo.reduce(%v2257 init: %v2251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2259 = stablehlo.broadcast_in_dim %v2258, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2260 = stablehlo.divide %v2259, %v2252 : tensor<32x160x7x7xf32>
    %v2261 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2262 = stablehlo.add %v2260, %v2261 : tensor<32x160x7x7xf32>
    %v2263 = stablehlo.rsqrt %v2262 : tensor<32x160x7x7xf32>
    %v2264 = stablehlo.multiply %v2256, %v2263 : tensor<32x160x7x7xf32>
    %v2265 = stablehlo.reshape %v1982 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2266 = stablehlo.multiply %v2265, %v2264 : tensor<32x160x7x7xf32>
    %v2267 = stablehlo.reduce(%v2266 init: %v2251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2268 = stablehlo.reshape %v1982 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2269 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2270 = stablehlo.reduce(%v2268 init: %v2269) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2271 = stablehlo.reshape %v1229 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2273 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v2274 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2275 = stablehlo.reduce(%v2271 init: %v2272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2276 = stablehlo.broadcast_in_dim %v2275, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2277 = stablehlo.divide %v2276, %v2273 : tensor<32x160x7x7xf32>
    %v2278 = stablehlo.subtract %v2271, %v2277 : tensor<32x160x7x7xf32>
    %v2279 = stablehlo.multiply %v2278, %v2278 : tensor<32x160x7x7xf32>
    %v2280 = stablehlo.reduce(%v2279 init: %v2272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2281 = stablehlo.broadcast_in_dim %v2280, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2282 = stablehlo.divide %v2281, %v2273 : tensor<32x160x7x7xf32>
    %v2283 = stablehlo.add %v2282, %v2274 : tensor<32x160x7x7xf32>
    %v2284 = stablehlo.rsqrt %v2283 : tensor<32x160x7x7xf32>
    %v2285 = stablehlo.multiply %v2278, %v2284 : tensor<32x160x7x7xf32>
    %v2286 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2287 = stablehlo.reshape %v2189 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2288 = stablehlo.multiply %v2286, %v2287 : tensor<32x160x7x7xf32>
    %v2289 = stablehlo.reduce(%v2288 init: %v2272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2290 = stablehlo.broadcast_in_dim %v2289, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2291 = stablehlo.multiply %v2285, %v2288 : tensor<32x160x7x7xf32>
    %v2292 = stablehlo.reduce(%v2291 init: %v2272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2293 = stablehlo.broadcast_in_dim %v2292, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2294 = stablehlo.multiply %v2288, %v2273 : tensor<32x160x7x7xf32>
    %v2295 = stablehlo.subtract %v2294, %v2290 : tensor<32x160x7x7xf32>
    %v2296 = stablehlo.multiply %v2285, %v2293 : tensor<32x160x7x7xf32>
    %v2297 = stablehlo.subtract %v2295, %v2296 : tensor<32x160x7x7xf32>
    %v2298 = stablehlo.divide %v2284, %v2273 : tensor<32x160x7x7xf32>
    %v2299 = stablehlo.multiply %v2298, %v2297 : tensor<32x160x7x7xf32>
    %v2300 = stablehlo.reshape %v2299 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2301 = stablehlo.reshape %v2300 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2302 = stablehlo.reverse %b14pW, dims = [2, 3] : tensor<160x576x1x1xf32>
    %v2303 = stablehlo.transpose %v2302, dims = [1, 0, 2, 3] : (tensor<160x576x1x1xf32>) -> tensor<576x160x1x1xf32>
    %v2304 = stablehlo.convolution(%v2301, %v2303)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<576x160x1x1xf32>) -> tensor<32x576x7x7xf32>
    %v2305 = stablehlo.reshape %v2304 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v2306 = stablehlo.reshape %v2305 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2307 = stablehlo.reshape %v1218 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2308 = stablehlo.constant dense<0.0> : tensor<32x576x7x7xf32>
    %v2309 = stablehlo.constant dense<6.0> : tensor<32x576x7x7xf32>
    %v2310 = stablehlo.compare GT, %v2307, %v2308 : (tensor<32x576x7x7xf32>, tensor<32x576x7x7xf32>) -> tensor<32x576x7x7xi1>
    %v2311 = stablehlo.compare LT, %v2307, %v2309 : (tensor<32x576x7x7xf32>, tensor<32x576x7x7xf32>) -> tensor<32x576x7x7xi1>
    %v2312 = stablehlo.and %v2310, %v2311 : tensor<32x576x7x7xi1>
    %v2313 = stablehlo.select %v2312, %v2306, %v2308 : tensor<32x576x7x7xi1>, tensor<32x576x7x7xf32>
    %v2314 = stablehlo.reshape %v2313 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v2315 = stablehlo.reshape %v1198 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2317 = stablehlo.constant dense<1568.0> : tensor<32x576x7x7xf32>
    %v2318 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v2319 = stablehlo.reduce(%v2315 init: %v2316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2320 = stablehlo.broadcast_in_dim %v2319, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2321 = stablehlo.divide %v2320, %v2317 : tensor<32x576x7x7xf32>
    %v2322 = stablehlo.subtract %v2315, %v2321 : tensor<32x576x7x7xf32>
    %v2323 = stablehlo.multiply %v2322, %v2322 : tensor<32x576x7x7xf32>
    %v2324 = stablehlo.reduce(%v2323 init: %v2316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2325 = stablehlo.broadcast_in_dim %v2324, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2326 = stablehlo.divide %v2325, %v2317 : tensor<32x576x7x7xf32>
    %v2327 = stablehlo.add %v2326, %v2318 : tensor<32x576x7x7xf32>
    %v2328 = stablehlo.rsqrt %v2327 : tensor<32x576x7x7xf32>
    %v2329 = stablehlo.multiply %v2322, %v2328 : tensor<32x576x7x7xf32>
    %v2330 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2331 = stablehlo.reshape %v2314 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2332 = stablehlo.multiply %v2330, %v2331 : tensor<32x576x7x7xf32>
    %v2333 = stablehlo.reduce(%v2332 init: %v2316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2334 = stablehlo.broadcast_in_dim %v2333, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2335 = stablehlo.multiply %v2329, %v2332 : tensor<32x576x7x7xf32>
    %v2336 = stablehlo.reduce(%v2335 init: %v2316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2337 = stablehlo.broadcast_in_dim %v2336, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2338 = stablehlo.multiply %v2332, %v2317 : tensor<32x576x7x7xf32>
    %v2339 = stablehlo.subtract %v2338, %v2334 : tensor<32x576x7x7xf32>
    %v2340 = stablehlo.multiply %v2329, %v2337 : tensor<32x576x7x7xf32>
    %v2341 = stablehlo.subtract %v2339, %v2340 : tensor<32x576x7x7xf32>
    %v2342 = stablehlo.divide %v2328, %v2317 : tensor<32x576x7x7xf32>
    %v2343 = stablehlo.multiply %v2342, %v2341 : tensor<32x576x7x7xf32>
    %v2344 = stablehlo.reshape %v2343 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v2345 = stablehlo.reshape %v2344 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2346 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2347 = stablehlo.pad %v2345, %v2346, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576x14x14xf32>
    %v2348 = stablehlo.reverse %b14dW, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2349 = stablehlo.convolution(%v2347, %v2348)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 0], [2, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2350 = stablehlo.reshape %v2349 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2351 = stablehlo.reshape %v2350 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2352 = stablehlo.reshape %v1187 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2353 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v2354 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v2355 = stablehlo.compare GT, %v2352, %v2353 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2356 = stablehlo.compare LT, %v2352, %v2354 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2357 = stablehlo.and %v2355, %v2356 : tensor<32x576x14x14xi1>
    %v2358 = stablehlo.select %v2357, %v2351, %v2353 : tensor<32x576x14x14xi1>, tensor<32x576x14x14xf32>
    %v2359 = stablehlo.reshape %v2358 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2360 = stablehlo.reshape %v1167 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2362 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2363 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2364 = stablehlo.reduce(%v2360 init: %v2361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2365 = stablehlo.broadcast_in_dim %v2364, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2366 = stablehlo.divide %v2365, %v2362 : tensor<32x576x14x14xf32>
    %v2367 = stablehlo.subtract %v2360, %v2366 : tensor<32x576x14x14xf32>
    %v2368 = stablehlo.multiply %v2367, %v2367 : tensor<32x576x14x14xf32>
    %v2369 = stablehlo.reduce(%v2368 init: %v2361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2370 = stablehlo.broadcast_in_dim %v2369, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2371 = stablehlo.divide %v2370, %v2362 : tensor<32x576x14x14xf32>
    %v2372 = stablehlo.add %v2371, %v2363 : tensor<32x576x14x14xf32>
    %v2373 = stablehlo.rsqrt %v2372 : tensor<32x576x14x14xf32>
    %v2374 = stablehlo.multiply %v2367, %v2373 : tensor<32x576x14x14xf32>
    %v2375 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2376 = stablehlo.reshape %v2359 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2377 = stablehlo.multiply %v2375, %v2376 : tensor<32x576x14x14xf32>
    %v2378 = stablehlo.reduce(%v2377 init: %v2361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2379 = stablehlo.broadcast_in_dim %v2378, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2380 = stablehlo.multiply %v2374, %v2377 : tensor<32x576x14x14xf32>
    %v2381 = stablehlo.reduce(%v2380 init: %v2361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2382 = stablehlo.broadcast_in_dim %v2381, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2383 = stablehlo.multiply %v2377, %v2362 : tensor<32x576x14x14xf32>
    %v2384 = stablehlo.subtract %v2383, %v2379 : tensor<32x576x14x14xf32>
    %v2385 = stablehlo.multiply %v2374, %v2382 : tensor<32x576x14x14xf32>
    %v2386 = stablehlo.subtract %v2384, %v2385 : tensor<32x576x14x14xf32>
    %v2387 = stablehlo.divide %v2373, %v2362 : tensor<32x576x14x14xf32>
    %v2388 = stablehlo.multiply %v2387, %v2386 : tensor<32x576x14x14xf32>
    %v2389 = stablehlo.reshape %v2388 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2390 = stablehlo.reshape %v2389 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2391 = stablehlo.reverse %b14eW, dims = [2, 3] : tensor<576x96x1x1xf32>
    %v2392 = stablehlo.transpose %v2391, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2393 = stablehlo.convolution(%v2390, %v2392)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2394 = stablehlo.reshape %v2393 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2395 = stablehlo.reshape %v1162 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2396 = stablehlo.reshape %v2389 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2397 = stablehlo.transpose %v2395, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2398 = stablehlo.transpose %v2396, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2399 = stablehlo.convolution(%v2397, %v2398)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2400 = stablehlo.transpose %v2399, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2401 = stablehlo.reshape %v1167 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2403 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2404 = stablehlo.reduce(%v2401 init: %v2402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2405 = stablehlo.broadcast_in_dim %v2404, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2406 = stablehlo.divide %v2405, %v2403 : tensor<32x576x14x14xf32>
    %v2407 = stablehlo.subtract %v2401, %v2406 : tensor<32x576x14x14xf32>
    %v2408 = stablehlo.multiply %v2407, %v2407 : tensor<32x576x14x14xf32>
    %v2409 = stablehlo.reduce(%v2408 init: %v2402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2410 = stablehlo.broadcast_in_dim %v2409, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2411 = stablehlo.divide %v2410, %v2403 : tensor<32x576x14x14xf32>
    %v2412 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2413 = stablehlo.add %v2411, %v2412 : tensor<32x576x14x14xf32>
    %v2414 = stablehlo.rsqrt %v2413 : tensor<32x576x14x14xf32>
    %v2415 = stablehlo.multiply %v2407, %v2414 : tensor<32x576x14x14xf32>
    %v2416 = stablehlo.reshape %v2359 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2417 = stablehlo.multiply %v2416, %v2415 : tensor<32x576x14x14xf32>
    %v2418 = stablehlo.reduce(%v2417 init: %v2402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2419 = stablehlo.reshape %v2359 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2421 = stablehlo.reduce(%v2419 init: %v2420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2422 = stablehlo.reshape %v1193 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2423 = stablehlo.reshape %v2344 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2424 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2425 = stablehlo.pad %v2423, %v2424, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576x14x14xf32>
    %v2426 = stablehlo.transpose %v2422, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2427 = stablehlo.transpose %v2425, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2428 = stablehlo.convolution(%v2426, %v2427)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 2], [0, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2429 = stablehlo.reshape %v2428 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2430 = stablehlo.reshape %v1198 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2432 = stablehlo.constant dense<1568.0> : tensor<32x576x7x7xf32>
    %v2433 = stablehlo.reduce(%v2430 init: %v2431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2434 = stablehlo.broadcast_in_dim %v2433, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2435 = stablehlo.divide %v2434, %v2432 : tensor<32x576x7x7xf32>
    %v2436 = stablehlo.subtract %v2430, %v2435 : tensor<32x576x7x7xf32>
    %v2437 = stablehlo.multiply %v2436, %v2436 : tensor<32x576x7x7xf32>
    %v2438 = stablehlo.reduce(%v2437 init: %v2431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2439 = stablehlo.broadcast_in_dim %v2438, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2440 = stablehlo.divide %v2439, %v2432 : tensor<32x576x7x7xf32>
    %v2441 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v2442 = stablehlo.add %v2440, %v2441 : tensor<32x576x7x7xf32>
    %v2443 = stablehlo.rsqrt %v2442 : tensor<32x576x7x7xf32>
    %v2444 = stablehlo.multiply %v2436, %v2443 : tensor<32x576x7x7xf32>
    %v2445 = stablehlo.reshape %v2314 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2446 = stablehlo.multiply %v2445, %v2444 : tensor<32x576x7x7xf32>
    %v2447 = stablehlo.reduce(%v2446 init: %v2431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2448 = stablehlo.reshape %v2314 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2450 = stablehlo.reduce(%v2448 init: %v2449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2451 = stablehlo.reshape %v1224 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2452 = stablehlo.reshape %v2300 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2453 = stablehlo.transpose %v2451, dims = [1, 0, 2, 3] : (tensor<32x576x7x7xf32>) -> tensor<576x32x7x7xf32>
    %v2454 = stablehlo.transpose %v2452, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2455 = stablehlo.convolution(%v2453, %v2454)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<576x160x1x1xf32>
    %v2456 = stablehlo.transpose %v2455, dims = [1, 0, 2, 3] : (tensor<576x160x1x1xf32>) -> tensor<160x576x1x1xf32>
    %v2457 = stablehlo.reshape %v1229 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2459 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v2460 = stablehlo.reduce(%v2457 init: %v2458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2461 = stablehlo.broadcast_in_dim %v2460, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2462 = stablehlo.divide %v2461, %v2459 : tensor<32x160x7x7xf32>
    %v2463 = stablehlo.subtract %v2457, %v2462 : tensor<32x160x7x7xf32>
    %v2464 = stablehlo.multiply %v2463, %v2463 : tensor<32x160x7x7xf32>
    %v2465 = stablehlo.reduce(%v2464 init: %v2458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2466 = stablehlo.broadcast_in_dim %v2465, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2467 = stablehlo.divide %v2466, %v2459 : tensor<32x160x7x7xf32>
    %v2468 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2469 = stablehlo.add %v2467, %v2468 : tensor<32x160x7x7xf32>
    %v2470 = stablehlo.rsqrt %v2469 : tensor<32x160x7x7xf32>
    %v2471 = stablehlo.multiply %v2463, %v2470 : tensor<32x160x7x7xf32>
    %v2472 = stablehlo.reshape %v2189 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2473 = stablehlo.multiply %v2472, %v2471 : tensor<32x160x7x7xf32>
    %v2474 = stablehlo.reduce(%v2473 init: %v2458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2475 = stablehlo.reshape %v2189 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2476 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2477 = stablehlo.reduce(%v2475 init: %v2476) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2478 = stablehlo.reshape %v1138 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2479 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2480 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2481 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2482 = stablehlo.reduce(%v2478 init: %v2479) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2483 = stablehlo.broadcast_in_dim %v2482, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2484 = stablehlo.divide %v2483, %v2480 : tensor<32x96x14x14xf32>
    %v2485 = stablehlo.subtract %v2478, %v2484 : tensor<32x96x14x14xf32>
    %v2486 = stablehlo.multiply %v2485, %v2485 : tensor<32x96x14x14xf32>
    %v2487 = stablehlo.reduce(%v2486 init: %v2479) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2488 = stablehlo.broadcast_in_dim %v2487, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2489 = stablehlo.divide %v2488, %v2480 : tensor<32x96x14x14xf32>
    %v2490 = stablehlo.add %v2489, %v2481 : tensor<32x96x14x14xf32>
    %v2491 = stablehlo.rsqrt %v2490 : tensor<32x96x14x14xf32>
    %v2492 = stablehlo.multiply %v2485, %v2491 : tensor<32x96x14x14xf32>
    %v2493 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2494 = stablehlo.reshape %v2394 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2495 = stablehlo.multiply %v2493, %v2494 : tensor<32x96x14x14xf32>
    %v2496 = stablehlo.reduce(%v2495 init: %v2479) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2497 = stablehlo.broadcast_in_dim %v2496, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2498 = stablehlo.multiply %v2492, %v2495 : tensor<32x96x14x14xf32>
    %v2499 = stablehlo.reduce(%v2498 init: %v2479) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2500 = stablehlo.broadcast_in_dim %v2499, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2501 = stablehlo.multiply %v2495, %v2480 : tensor<32x96x14x14xf32>
    %v2502 = stablehlo.subtract %v2501, %v2497 : tensor<32x96x14x14xf32>
    %v2503 = stablehlo.multiply %v2492, %v2500 : tensor<32x96x14x14xf32>
    %v2504 = stablehlo.subtract %v2502, %v2503 : tensor<32x96x14x14xf32>
    %v2505 = stablehlo.divide %v2491, %v2480 : tensor<32x96x14x14xf32>
    %v2506 = stablehlo.multiply %v2505, %v2504 : tensor<32x96x14x14xf32>
    %v2507 = stablehlo.reshape %v2506 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2508 = stablehlo.reshape %v2507 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2509 = stablehlo.reverse %b13pW, dims = [2, 3] : tensor<96x576x1x1xf32>
    %v2510 = stablehlo.transpose %v2509, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2511 = stablehlo.convolution(%v2508, %v2510)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v2512 = stablehlo.reshape %v2511 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2513 = stablehlo.reshape %v2512 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2514 = stablehlo.reshape %v1127 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2515 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v2516 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v2517 = stablehlo.compare GT, %v2514, %v2515 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2518 = stablehlo.compare LT, %v2514, %v2516 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2519 = stablehlo.and %v2517, %v2518 : tensor<32x576x14x14xi1>
    %v2520 = stablehlo.select %v2519, %v2513, %v2515 : tensor<32x576x14x14xi1>, tensor<32x576x14x14xf32>
    %v2521 = stablehlo.reshape %v2520 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2522 = stablehlo.reshape %v1107 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2523 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2524 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2525 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2526 = stablehlo.reduce(%v2522 init: %v2523) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2527 = stablehlo.broadcast_in_dim %v2526, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2528 = stablehlo.divide %v2527, %v2524 : tensor<32x576x14x14xf32>
    %v2529 = stablehlo.subtract %v2522, %v2528 : tensor<32x576x14x14xf32>
    %v2530 = stablehlo.multiply %v2529, %v2529 : tensor<32x576x14x14xf32>
    %v2531 = stablehlo.reduce(%v2530 init: %v2523) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2532 = stablehlo.broadcast_in_dim %v2531, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2533 = stablehlo.divide %v2532, %v2524 : tensor<32x576x14x14xf32>
    %v2534 = stablehlo.add %v2533, %v2525 : tensor<32x576x14x14xf32>
    %v2535 = stablehlo.rsqrt %v2534 : tensor<32x576x14x14xf32>
    %v2536 = stablehlo.multiply %v2529, %v2535 : tensor<32x576x14x14xf32>
    %v2537 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2538 = stablehlo.reshape %v2521 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2539 = stablehlo.multiply %v2537, %v2538 : tensor<32x576x14x14xf32>
    %v2540 = stablehlo.reduce(%v2539 init: %v2523) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2541 = stablehlo.broadcast_in_dim %v2540, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2542 = stablehlo.multiply %v2536, %v2539 : tensor<32x576x14x14xf32>
    %v2543 = stablehlo.reduce(%v2542 init: %v2523) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2544 = stablehlo.broadcast_in_dim %v2543, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2545 = stablehlo.multiply %v2539, %v2524 : tensor<32x576x14x14xf32>
    %v2546 = stablehlo.subtract %v2545, %v2541 : tensor<32x576x14x14xf32>
    %v2547 = stablehlo.multiply %v2536, %v2544 : tensor<32x576x14x14xf32>
    %v2548 = stablehlo.subtract %v2546, %v2547 : tensor<32x576x14x14xf32>
    %v2549 = stablehlo.divide %v2535, %v2524 : tensor<32x576x14x14xf32>
    %v2550 = stablehlo.multiply %v2549, %v2548 : tensor<32x576x14x14xf32>
    %v2551 = stablehlo.reshape %v2550 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2552 = stablehlo.reshape %v2551 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2553 = stablehlo.reverse %b13dW, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2554 = stablehlo.convolution(%v2552, %v2553)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2555 = stablehlo.reshape %v2554 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2556 = stablehlo.reshape %v2555 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2557 = stablehlo.reshape %v1096 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2558 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v2559 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v2560 = stablehlo.compare GT, %v2557, %v2558 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2561 = stablehlo.compare LT, %v2557, %v2559 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2562 = stablehlo.and %v2560, %v2561 : tensor<32x576x14x14xi1>
    %v2563 = stablehlo.select %v2562, %v2556, %v2558 : tensor<32x576x14x14xi1>, tensor<32x576x14x14xf32>
    %v2564 = stablehlo.reshape %v2563 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2565 = stablehlo.reshape %v1076 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2567 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2568 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2569 = stablehlo.reduce(%v2565 init: %v2566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2570 = stablehlo.broadcast_in_dim %v2569, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2571 = stablehlo.divide %v2570, %v2567 : tensor<32x576x14x14xf32>
    %v2572 = stablehlo.subtract %v2565, %v2571 : tensor<32x576x14x14xf32>
    %v2573 = stablehlo.multiply %v2572, %v2572 : tensor<32x576x14x14xf32>
    %v2574 = stablehlo.reduce(%v2573 init: %v2566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2575 = stablehlo.broadcast_in_dim %v2574, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2576 = stablehlo.divide %v2575, %v2567 : tensor<32x576x14x14xf32>
    %v2577 = stablehlo.add %v2576, %v2568 : tensor<32x576x14x14xf32>
    %v2578 = stablehlo.rsqrt %v2577 : tensor<32x576x14x14xf32>
    %v2579 = stablehlo.multiply %v2572, %v2578 : tensor<32x576x14x14xf32>
    %v2580 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2581 = stablehlo.reshape %v2564 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2582 = stablehlo.multiply %v2580, %v2581 : tensor<32x576x14x14xf32>
    %v2583 = stablehlo.reduce(%v2582 init: %v2566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2584 = stablehlo.broadcast_in_dim %v2583, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2585 = stablehlo.multiply %v2579, %v2582 : tensor<32x576x14x14xf32>
    %v2586 = stablehlo.reduce(%v2585 init: %v2566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2587 = stablehlo.broadcast_in_dim %v2586, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2588 = stablehlo.multiply %v2582, %v2567 : tensor<32x576x14x14xf32>
    %v2589 = stablehlo.subtract %v2588, %v2584 : tensor<32x576x14x14xf32>
    %v2590 = stablehlo.multiply %v2579, %v2587 : tensor<32x576x14x14xf32>
    %v2591 = stablehlo.subtract %v2589, %v2590 : tensor<32x576x14x14xf32>
    %v2592 = stablehlo.divide %v2578, %v2567 : tensor<32x576x14x14xf32>
    %v2593 = stablehlo.multiply %v2592, %v2591 : tensor<32x576x14x14xf32>
    %v2594 = stablehlo.reshape %v2593 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2595 = stablehlo.reshape %v2594 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2596 = stablehlo.reverse %b13eW, dims = [2, 3] : tensor<576x96x1x1xf32>
    %v2597 = stablehlo.transpose %v2596, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2598 = stablehlo.convolution(%v2595, %v2597)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2599 = stablehlo.reshape %v2598 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2600 = stablehlo.reshape %v2599 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2601 = stablehlo.reshape %v2394 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2602 = stablehlo.add %v2600, %v2601 : tensor<32x96x14x14xf32>
    %v2603 = stablehlo.reshape %v2602 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2604 = stablehlo.reshape %v1071 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2605 = stablehlo.reshape %v2594 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2606 = stablehlo.transpose %v2604, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2607 = stablehlo.transpose %v2605, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2608 = stablehlo.convolution(%v2606, %v2607)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2609 = stablehlo.transpose %v2608, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2610 = stablehlo.reshape %v1076 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2611 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2612 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2613 = stablehlo.reduce(%v2610 init: %v2611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2614 = stablehlo.broadcast_in_dim %v2613, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2615 = stablehlo.divide %v2614, %v2612 : tensor<32x576x14x14xf32>
    %v2616 = stablehlo.subtract %v2610, %v2615 : tensor<32x576x14x14xf32>
    %v2617 = stablehlo.multiply %v2616, %v2616 : tensor<32x576x14x14xf32>
    %v2618 = stablehlo.reduce(%v2617 init: %v2611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2619 = stablehlo.broadcast_in_dim %v2618, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2620 = stablehlo.divide %v2619, %v2612 : tensor<32x576x14x14xf32>
    %v2621 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2622 = stablehlo.add %v2620, %v2621 : tensor<32x576x14x14xf32>
    %v2623 = stablehlo.rsqrt %v2622 : tensor<32x576x14x14xf32>
    %v2624 = stablehlo.multiply %v2616, %v2623 : tensor<32x576x14x14xf32>
    %v2625 = stablehlo.reshape %v2564 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2626 = stablehlo.multiply %v2625, %v2624 : tensor<32x576x14x14xf32>
    %v2627 = stablehlo.reduce(%v2626 init: %v2611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2628 = stablehlo.reshape %v2564 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2630 = stablehlo.reduce(%v2628 init: %v2629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2631 = stablehlo.reshape %v1102 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2632 = stablehlo.reshape %v2551 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2633 = stablehlo.transpose %v2631, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2634 = stablehlo.transpose %v2632, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2635 = stablehlo.convolution(%v2633, %v2634)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2636 = stablehlo.reshape %v2635 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2637 = stablehlo.reshape %v1107 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2638 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2639 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2640 = stablehlo.reduce(%v2637 init: %v2638) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2641 = stablehlo.broadcast_in_dim %v2640, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2642 = stablehlo.divide %v2641, %v2639 : tensor<32x576x14x14xf32>
    %v2643 = stablehlo.subtract %v2637, %v2642 : tensor<32x576x14x14xf32>
    %v2644 = stablehlo.multiply %v2643, %v2643 : tensor<32x576x14x14xf32>
    %v2645 = stablehlo.reduce(%v2644 init: %v2638) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2646 = stablehlo.broadcast_in_dim %v2645, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2647 = stablehlo.divide %v2646, %v2639 : tensor<32x576x14x14xf32>
    %v2648 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2649 = stablehlo.add %v2647, %v2648 : tensor<32x576x14x14xf32>
    %v2650 = stablehlo.rsqrt %v2649 : tensor<32x576x14x14xf32>
    %v2651 = stablehlo.multiply %v2643, %v2650 : tensor<32x576x14x14xf32>
    %v2652 = stablehlo.reshape %v2521 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2653 = stablehlo.multiply %v2652, %v2651 : tensor<32x576x14x14xf32>
    %v2654 = stablehlo.reduce(%v2653 init: %v2638) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2655 = stablehlo.reshape %v2521 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2657 = stablehlo.reduce(%v2655 init: %v2656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2658 = stablehlo.reshape %v1133 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2659 = stablehlo.reshape %v2507 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2660 = stablehlo.transpose %v2658, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2661 = stablehlo.transpose %v2659, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2662 = stablehlo.convolution(%v2660, %v2661)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<576x96x1x1xf32>
    %v2663 = stablehlo.transpose %v2662, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2664 = stablehlo.reshape %v1138 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2666 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2667 = stablehlo.reduce(%v2664 init: %v2665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2668 = stablehlo.broadcast_in_dim %v2667, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2669 = stablehlo.divide %v2668, %v2666 : tensor<32x96x14x14xf32>
    %v2670 = stablehlo.subtract %v2664, %v2669 : tensor<32x96x14x14xf32>
    %v2671 = stablehlo.multiply %v2670, %v2670 : tensor<32x96x14x14xf32>
    %v2672 = stablehlo.reduce(%v2671 init: %v2665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2673 = stablehlo.broadcast_in_dim %v2672, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2674 = stablehlo.divide %v2673, %v2666 : tensor<32x96x14x14xf32>
    %v2675 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2676 = stablehlo.add %v2674, %v2675 : tensor<32x96x14x14xf32>
    %v2677 = stablehlo.rsqrt %v2676 : tensor<32x96x14x14xf32>
    %v2678 = stablehlo.multiply %v2670, %v2677 : tensor<32x96x14x14xf32>
    %v2679 = stablehlo.reshape %v2394 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2680 = stablehlo.multiply %v2679, %v2678 : tensor<32x96x14x14xf32>
    %v2681 = stablehlo.reduce(%v2680 init: %v2665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2682 = stablehlo.reshape %v2394 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2683 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2684 = stablehlo.reduce(%v2682 init: %v2683) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2685 = stablehlo.reshape %v1047 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2687 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2688 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2689 = stablehlo.reduce(%v2685 init: %v2686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2690 = stablehlo.broadcast_in_dim %v2689, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2691 = stablehlo.divide %v2690, %v2687 : tensor<32x96x14x14xf32>
    %v2692 = stablehlo.subtract %v2685, %v2691 : tensor<32x96x14x14xf32>
    %v2693 = stablehlo.multiply %v2692, %v2692 : tensor<32x96x14x14xf32>
    %v2694 = stablehlo.reduce(%v2693 init: %v2686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2695 = stablehlo.broadcast_in_dim %v2694, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2696 = stablehlo.divide %v2695, %v2687 : tensor<32x96x14x14xf32>
    %v2697 = stablehlo.add %v2696, %v2688 : tensor<32x96x14x14xf32>
    %v2698 = stablehlo.rsqrt %v2697 : tensor<32x96x14x14xf32>
    %v2699 = stablehlo.multiply %v2692, %v2698 : tensor<32x96x14x14xf32>
    %v2700 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2701 = stablehlo.reshape %v2603 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2702 = stablehlo.multiply %v2700, %v2701 : tensor<32x96x14x14xf32>
    %v2703 = stablehlo.reduce(%v2702 init: %v2686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2704 = stablehlo.broadcast_in_dim %v2703, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2705 = stablehlo.multiply %v2699, %v2702 : tensor<32x96x14x14xf32>
    %v2706 = stablehlo.reduce(%v2705 init: %v2686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2707 = stablehlo.broadcast_in_dim %v2706, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2708 = stablehlo.multiply %v2702, %v2687 : tensor<32x96x14x14xf32>
    %v2709 = stablehlo.subtract %v2708, %v2704 : tensor<32x96x14x14xf32>
    %v2710 = stablehlo.multiply %v2699, %v2707 : tensor<32x96x14x14xf32>
    %v2711 = stablehlo.subtract %v2709, %v2710 : tensor<32x96x14x14xf32>
    %v2712 = stablehlo.divide %v2698, %v2687 : tensor<32x96x14x14xf32>
    %v2713 = stablehlo.multiply %v2712, %v2711 : tensor<32x96x14x14xf32>
    %v2714 = stablehlo.reshape %v2713 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2715 = stablehlo.reshape %v2714 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2716 = stablehlo.reverse %b12pW, dims = [2, 3] : tensor<96x576x1x1xf32>
    %v2717 = stablehlo.transpose %v2716, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2718 = stablehlo.convolution(%v2715, %v2717)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v2719 = stablehlo.reshape %v2718 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2720 = stablehlo.reshape %v2719 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2721 = stablehlo.reshape %v1036 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2722 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v2723 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v2724 = stablehlo.compare GT, %v2721, %v2722 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2725 = stablehlo.compare LT, %v2721, %v2723 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2726 = stablehlo.and %v2724, %v2725 : tensor<32x576x14x14xi1>
    %v2727 = stablehlo.select %v2726, %v2720, %v2722 : tensor<32x576x14x14xi1>, tensor<32x576x14x14xf32>
    %v2728 = stablehlo.reshape %v2727 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2729 = stablehlo.reshape %v1016 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2731 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2732 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2733 = stablehlo.reduce(%v2729 init: %v2730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2734 = stablehlo.broadcast_in_dim %v2733, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2735 = stablehlo.divide %v2734, %v2731 : tensor<32x576x14x14xf32>
    %v2736 = stablehlo.subtract %v2729, %v2735 : tensor<32x576x14x14xf32>
    %v2737 = stablehlo.multiply %v2736, %v2736 : tensor<32x576x14x14xf32>
    %v2738 = stablehlo.reduce(%v2737 init: %v2730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2739 = stablehlo.broadcast_in_dim %v2738, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2740 = stablehlo.divide %v2739, %v2731 : tensor<32x576x14x14xf32>
    %v2741 = stablehlo.add %v2740, %v2732 : tensor<32x576x14x14xf32>
    %v2742 = stablehlo.rsqrt %v2741 : tensor<32x576x14x14xf32>
    %v2743 = stablehlo.multiply %v2736, %v2742 : tensor<32x576x14x14xf32>
    %v2744 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2745 = stablehlo.reshape %v2728 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2746 = stablehlo.multiply %v2744, %v2745 : tensor<32x576x14x14xf32>
    %v2747 = stablehlo.reduce(%v2746 init: %v2730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2748 = stablehlo.broadcast_in_dim %v2747, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2749 = stablehlo.multiply %v2743, %v2746 : tensor<32x576x14x14xf32>
    %v2750 = stablehlo.reduce(%v2749 init: %v2730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2751 = stablehlo.broadcast_in_dim %v2750, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2752 = stablehlo.multiply %v2746, %v2731 : tensor<32x576x14x14xf32>
    %v2753 = stablehlo.subtract %v2752, %v2748 : tensor<32x576x14x14xf32>
    %v2754 = stablehlo.multiply %v2743, %v2751 : tensor<32x576x14x14xf32>
    %v2755 = stablehlo.subtract %v2753, %v2754 : tensor<32x576x14x14xf32>
    %v2756 = stablehlo.divide %v2742, %v2731 : tensor<32x576x14x14xf32>
    %v2757 = stablehlo.multiply %v2756, %v2755 : tensor<32x576x14x14xf32>
    %v2758 = stablehlo.reshape %v2757 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2759 = stablehlo.reshape %v2758 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2760 = stablehlo.reverse %b12dW, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2761 = stablehlo.convolution(%v2759, %v2760)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2762 = stablehlo.reshape %v2761 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2763 = stablehlo.reshape %v2762 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2764 = stablehlo.reshape %v1005 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2765 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v2766 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v2767 = stablehlo.compare GT, %v2764, %v2765 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2768 = stablehlo.compare LT, %v2764, %v2766 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2769 = stablehlo.and %v2767, %v2768 : tensor<32x576x14x14xi1>
    %v2770 = stablehlo.select %v2769, %v2763, %v2765 : tensor<32x576x14x14xi1>, tensor<32x576x14x14xf32>
    %v2771 = stablehlo.reshape %v2770 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2772 = stablehlo.reshape %v985 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2773 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2774 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2775 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2776 = stablehlo.reduce(%v2772 init: %v2773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2777 = stablehlo.broadcast_in_dim %v2776, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2778 = stablehlo.divide %v2777, %v2774 : tensor<32x576x14x14xf32>
    %v2779 = stablehlo.subtract %v2772, %v2778 : tensor<32x576x14x14xf32>
    %v2780 = stablehlo.multiply %v2779, %v2779 : tensor<32x576x14x14xf32>
    %v2781 = stablehlo.reduce(%v2780 init: %v2773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2782 = stablehlo.broadcast_in_dim %v2781, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2783 = stablehlo.divide %v2782, %v2774 : tensor<32x576x14x14xf32>
    %v2784 = stablehlo.add %v2783, %v2775 : tensor<32x576x14x14xf32>
    %v2785 = stablehlo.rsqrt %v2784 : tensor<32x576x14x14xf32>
    %v2786 = stablehlo.multiply %v2779, %v2785 : tensor<32x576x14x14xf32>
    %v2787 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2788 = stablehlo.reshape %v2771 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2789 = stablehlo.multiply %v2787, %v2788 : tensor<32x576x14x14xf32>
    %v2790 = stablehlo.reduce(%v2789 init: %v2773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2791 = stablehlo.broadcast_in_dim %v2790, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2792 = stablehlo.multiply %v2786, %v2789 : tensor<32x576x14x14xf32>
    %v2793 = stablehlo.reduce(%v2792 init: %v2773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2794 = stablehlo.broadcast_in_dim %v2793, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2795 = stablehlo.multiply %v2789, %v2774 : tensor<32x576x14x14xf32>
    %v2796 = stablehlo.subtract %v2795, %v2791 : tensor<32x576x14x14xf32>
    %v2797 = stablehlo.multiply %v2786, %v2794 : tensor<32x576x14x14xf32>
    %v2798 = stablehlo.subtract %v2796, %v2797 : tensor<32x576x14x14xf32>
    %v2799 = stablehlo.divide %v2785, %v2774 : tensor<32x576x14x14xf32>
    %v2800 = stablehlo.multiply %v2799, %v2798 : tensor<32x576x14x14xf32>
    %v2801 = stablehlo.reshape %v2800 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2802 = stablehlo.reshape %v2801 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2803 = stablehlo.reverse %b12eW, dims = [2, 3] : tensor<576x96x1x1xf32>
    %v2804 = stablehlo.transpose %v2803, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2805 = stablehlo.convolution(%v2802, %v2804)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2806 = stablehlo.reshape %v2805 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2807 = stablehlo.reshape %v2806 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2808 = stablehlo.reshape %v2603 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2809 = stablehlo.add %v2807, %v2808 : tensor<32x96x14x14xf32>
    %v2810 = stablehlo.reshape %v2809 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2811 = stablehlo.reshape %v980 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2812 = stablehlo.reshape %v2801 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2813 = stablehlo.transpose %v2811, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2814 = stablehlo.transpose %v2812, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2815 = stablehlo.convolution(%v2813, %v2814)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2816 = stablehlo.transpose %v2815, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2817 = stablehlo.reshape %v985 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2818 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2819 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2820 = stablehlo.reduce(%v2817 init: %v2818) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2821 = stablehlo.broadcast_in_dim %v2820, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2822 = stablehlo.divide %v2821, %v2819 : tensor<32x576x14x14xf32>
    %v2823 = stablehlo.subtract %v2817, %v2822 : tensor<32x576x14x14xf32>
    %v2824 = stablehlo.multiply %v2823, %v2823 : tensor<32x576x14x14xf32>
    %v2825 = stablehlo.reduce(%v2824 init: %v2818) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2826 = stablehlo.broadcast_in_dim %v2825, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2827 = stablehlo.divide %v2826, %v2819 : tensor<32x576x14x14xf32>
    %v2828 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2829 = stablehlo.add %v2827, %v2828 : tensor<32x576x14x14xf32>
    %v2830 = stablehlo.rsqrt %v2829 : tensor<32x576x14x14xf32>
    %v2831 = stablehlo.multiply %v2823, %v2830 : tensor<32x576x14x14xf32>
    %v2832 = stablehlo.reshape %v2771 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2833 = stablehlo.multiply %v2832, %v2831 : tensor<32x576x14x14xf32>
    %v2834 = stablehlo.reduce(%v2833 init: %v2818) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2835 = stablehlo.reshape %v2771 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2836 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2837 = stablehlo.reduce(%v2835 init: %v2836) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2838 = stablehlo.reshape %v1011 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2839 = stablehlo.reshape %v2758 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2840 = stablehlo.transpose %v2838, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2841 = stablehlo.transpose %v2839, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2842 = stablehlo.convolution(%v2840, %v2841)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2843 = stablehlo.reshape %v2842 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2844 = stablehlo.reshape %v1016 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2845 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2846 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v2847 = stablehlo.reduce(%v2844 init: %v2845) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2848 = stablehlo.broadcast_in_dim %v2847, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2849 = stablehlo.divide %v2848, %v2846 : tensor<32x576x14x14xf32>
    %v2850 = stablehlo.subtract %v2844, %v2849 : tensor<32x576x14x14xf32>
    %v2851 = stablehlo.multiply %v2850, %v2850 : tensor<32x576x14x14xf32>
    %v2852 = stablehlo.reduce(%v2851 init: %v2845) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2853 = stablehlo.broadcast_in_dim %v2852, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2854 = stablehlo.divide %v2853, %v2846 : tensor<32x576x14x14xf32>
    %v2855 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2856 = stablehlo.add %v2854, %v2855 : tensor<32x576x14x14xf32>
    %v2857 = stablehlo.rsqrt %v2856 : tensor<32x576x14x14xf32>
    %v2858 = stablehlo.multiply %v2850, %v2857 : tensor<32x576x14x14xf32>
    %v2859 = stablehlo.reshape %v2728 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2860 = stablehlo.multiply %v2859, %v2858 : tensor<32x576x14x14xf32>
    %v2861 = stablehlo.reduce(%v2860 init: %v2845) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2862 = stablehlo.reshape %v2728 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2864 = stablehlo.reduce(%v2862 init: %v2863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2865 = stablehlo.reshape %v1042 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2866 = stablehlo.reshape %v2714 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2867 = stablehlo.transpose %v2865, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2868 = stablehlo.transpose %v2866, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2869 = stablehlo.convolution(%v2867, %v2868)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<576x96x1x1xf32>
    %v2870 = stablehlo.transpose %v2869, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2871 = stablehlo.reshape %v1047 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2872 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2873 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2874 = stablehlo.reduce(%v2871 init: %v2872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2875 = stablehlo.broadcast_in_dim %v2874, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2876 = stablehlo.divide %v2875, %v2873 : tensor<32x96x14x14xf32>
    %v2877 = stablehlo.subtract %v2871, %v2876 : tensor<32x96x14x14xf32>
    %v2878 = stablehlo.multiply %v2877, %v2877 : tensor<32x96x14x14xf32>
    %v2879 = stablehlo.reduce(%v2878 init: %v2872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2880 = stablehlo.broadcast_in_dim %v2879, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2881 = stablehlo.divide %v2880, %v2873 : tensor<32x96x14x14xf32>
    %v2882 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2883 = stablehlo.add %v2881, %v2882 : tensor<32x96x14x14xf32>
    %v2884 = stablehlo.rsqrt %v2883 : tensor<32x96x14x14xf32>
    %v2885 = stablehlo.multiply %v2877, %v2884 : tensor<32x96x14x14xf32>
    %v2886 = stablehlo.reshape %v2603 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2887 = stablehlo.multiply %v2886, %v2885 : tensor<32x96x14x14xf32>
    %v2888 = stablehlo.reduce(%v2887 init: %v2872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2889 = stablehlo.reshape %v2603 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2891 = stablehlo.reduce(%v2889 init: %v2890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2892 = stablehlo.reshape %v960 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2894 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v2895 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2896 = stablehlo.reduce(%v2892 init: %v2893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2897 = stablehlo.broadcast_in_dim %v2896, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2898 = stablehlo.divide %v2897, %v2894 : tensor<32x96x14x14xf32>
    %v2899 = stablehlo.subtract %v2892, %v2898 : tensor<32x96x14x14xf32>
    %v2900 = stablehlo.multiply %v2899, %v2899 : tensor<32x96x14x14xf32>
    %v2901 = stablehlo.reduce(%v2900 init: %v2893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2902 = stablehlo.broadcast_in_dim %v2901, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2903 = stablehlo.divide %v2902, %v2894 : tensor<32x96x14x14xf32>
    %v2904 = stablehlo.add %v2903, %v2895 : tensor<32x96x14x14xf32>
    %v2905 = stablehlo.rsqrt %v2904 : tensor<32x96x14x14xf32>
    %v2906 = stablehlo.multiply %v2899, %v2905 : tensor<32x96x14x14xf32>
    %v2907 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2908 = stablehlo.reshape %v2810 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2909 = stablehlo.multiply %v2907, %v2908 : tensor<32x96x14x14xf32>
    %v2910 = stablehlo.reduce(%v2909 init: %v2893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2911 = stablehlo.broadcast_in_dim %v2910, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2912 = stablehlo.multiply %v2906, %v2909 : tensor<32x96x14x14xf32>
    %v2913 = stablehlo.reduce(%v2912 init: %v2893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2914 = stablehlo.broadcast_in_dim %v2913, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2915 = stablehlo.multiply %v2909, %v2894 : tensor<32x96x14x14xf32>
    %v2916 = stablehlo.subtract %v2915, %v2911 : tensor<32x96x14x14xf32>
    %v2917 = stablehlo.multiply %v2906, %v2914 : tensor<32x96x14x14xf32>
    %v2918 = stablehlo.subtract %v2916, %v2917 : tensor<32x96x14x14xf32>
    %v2919 = stablehlo.divide %v2905, %v2894 : tensor<32x96x14x14xf32>
    %v2920 = stablehlo.multiply %v2919, %v2918 : tensor<32x96x14x14xf32>
    %v2921 = stablehlo.reshape %v2920 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2922 = stablehlo.reshape %v2921 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2923 = stablehlo.reverse %b11pW, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v2924 = stablehlo.transpose %v2923, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v2925 = stablehlo.convolution(%v2922, %v2924)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2926 = stablehlo.reshape %v2925 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2927 = stablehlo.reshape %v2926 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2928 = stablehlo.reshape %v949 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2929 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v2930 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v2931 = stablehlo.compare GT, %v2928, %v2929 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v2932 = stablehlo.compare LT, %v2928, %v2930 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v2933 = stablehlo.and %v2931, %v2932 : tensor<32x384x14x14xi1>
    %v2934 = stablehlo.select %v2933, %v2927, %v2929 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v2935 = stablehlo.reshape %v2934 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2936 = stablehlo.reshape %v929 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2938 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v2939 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v2940 = stablehlo.reduce(%v2936 init: %v2937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2941 = stablehlo.broadcast_in_dim %v2940, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2942 = stablehlo.divide %v2941, %v2938 : tensor<32x384x14x14xf32>
    %v2943 = stablehlo.subtract %v2936, %v2942 : tensor<32x384x14x14xf32>
    %v2944 = stablehlo.multiply %v2943, %v2943 : tensor<32x384x14x14xf32>
    %v2945 = stablehlo.reduce(%v2944 init: %v2937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2946 = stablehlo.broadcast_in_dim %v2945, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2947 = stablehlo.divide %v2946, %v2938 : tensor<32x384x14x14xf32>
    %v2948 = stablehlo.add %v2947, %v2939 : tensor<32x384x14x14xf32>
    %v2949 = stablehlo.rsqrt %v2948 : tensor<32x384x14x14xf32>
    %v2950 = stablehlo.multiply %v2943, %v2949 : tensor<32x384x14x14xf32>
    %v2951 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2952 = stablehlo.reshape %v2935 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2953 = stablehlo.multiply %v2951, %v2952 : tensor<32x384x14x14xf32>
    %v2954 = stablehlo.reduce(%v2953 init: %v2937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2955 = stablehlo.broadcast_in_dim %v2954, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2956 = stablehlo.multiply %v2950, %v2953 : tensor<32x384x14x14xf32>
    %v2957 = stablehlo.reduce(%v2956 init: %v2937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2958 = stablehlo.broadcast_in_dim %v2957, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2959 = stablehlo.multiply %v2953, %v2938 : tensor<32x384x14x14xf32>
    %v2960 = stablehlo.subtract %v2959, %v2955 : tensor<32x384x14x14xf32>
    %v2961 = stablehlo.multiply %v2950, %v2958 : tensor<32x384x14x14xf32>
    %v2962 = stablehlo.subtract %v2960, %v2961 : tensor<32x384x14x14xf32>
    %v2963 = stablehlo.divide %v2949, %v2938 : tensor<32x384x14x14xf32>
    %v2964 = stablehlo.multiply %v2963, %v2962 : tensor<32x384x14x14xf32>
    %v2965 = stablehlo.reshape %v2964 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2966 = stablehlo.reshape %v2965 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2967 = stablehlo.reverse %b11dW, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v2968 = stablehlo.convolution(%v2966, %v2967)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v2969 = stablehlo.reshape %v2968 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2970 = stablehlo.reshape %v2969 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2971 = stablehlo.reshape %v918 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2972 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v2973 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v2974 = stablehlo.compare GT, %v2971, %v2972 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v2975 = stablehlo.compare LT, %v2971, %v2973 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v2976 = stablehlo.and %v2974, %v2975 : tensor<32x384x14x14xi1>
    %v2977 = stablehlo.select %v2976, %v2970, %v2972 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v2978 = stablehlo.reshape %v2977 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2979 = stablehlo.reshape %v898 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
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
    %v2994 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
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
    %v3010 = stablehlo.reverse %b11eW, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3011 = stablehlo.transpose %v3010, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3012 = stablehlo.convolution(%v3009, %v3011)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3013 = stablehlo.reshape %v3012 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3014 = stablehlo.reshape %v893 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3015 = stablehlo.reshape %v3008 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3016 = stablehlo.transpose %v3014, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3017 = stablehlo.transpose %v3015, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3018 = stablehlo.convolution(%v3016, %v3017)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3019 = stablehlo.transpose %v3018, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3020 = stablehlo.reshape %v898 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3021 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3022 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3023 = stablehlo.reduce(%v3020 init: %v3021) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3024 = stablehlo.broadcast_in_dim %v3023, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3025 = stablehlo.divide %v3024, %v3022 : tensor<32x384x14x14xf32>
    %v3026 = stablehlo.subtract %v3020, %v3025 : tensor<32x384x14x14xf32>
    %v3027 = stablehlo.multiply %v3026, %v3026 : tensor<32x384x14x14xf32>
    %v3028 = stablehlo.reduce(%v3027 init: %v3021) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3029 = stablehlo.broadcast_in_dim %v3028, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3030 = stablehlo.divide %v3029, %v3022 : tensor<32x384x14x14xf32>
    %v3031 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3032 = stablehlo.add %v3030, %v3031 : tensor<32x384x14x14xf32>
    %v3033 = stablehlo.rsqrt %v3032 : tensor<32x384x14x14xf32>
    %v3034 = stablehlo.multiply %v3026, %v3033 : tensor<32x384x14x14xf32>
    %v3035 = stablehlo.reshape %v2978 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3036 = stablehlo.multiply %v3035, %v3034 : tensor<32x384x14x14xf32>
    %v3037 = stablehlo.reduce(%v3036 init: %v3021) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3038 = stablehlo.reshape %v2978 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3040 = stablehlo.reduce(%v3038 init: %v3039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3041 = stablehlo.reshape %v924 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3042 = stablehlo.reshape %v2965 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3043 = stablehlo.transpose %v3041, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3044 = stablehlo.transpose %v3042, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3045 = stablehlo.convolution(%v3043, %v3044)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3046 = stablehlo.reshape %v3045 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3047 = stablehlo.reshape %v929 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3048 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3049 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3050 = stablehlo.reduce(%v3047 init: %v3048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3051 = stablehlo.broadcast_in_dim %v3050, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3052 = stablehlo.divide %v3051, %v3049 : tensor<32x384x14x14xf32>
    %v3053 = stablehlo.subtract %v3047, %v3052 : tensor<32x384x14x14xf32>
    %v3054 = stablehlo.multiply %v3053, %v3053 : tensor<32x384x14x14xf32>
    %v3055 = stablehlo.reduce(%v3054 init: %v3048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3056 = stablehlo.broadcast_in_dim %v3055, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3057 = stablehlo.divide %v3056, %v3049 : tensor<32x384x14x14xf32>
    %v3058 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3059 = stablehlo.add %v3057, %v3058 : tensor<32x384x14x14xf32>
    %v3060 = stablehlo.rsqrt %v3059 : tensor<32x384x14x14xf32>
    %v3061 = stablehlo.multiply %v3053, %v3060 : tensor<32x384x14x14xf32>
    %v3062 = stablehlo.reshape %v2935 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3063 = stablehlo.multiply %v3062, %v3061 : tensor<32x384x14x14xf32>
    %v3064 = stablehlo.reduce(%v3063 init: %v3048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3065 = stablehlo.reshape %v2935 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3066 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3067 = stablehlo.reduce(%v3065 init: %v3066) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3068 = stablehlo.reshape %v955 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3069 = stablehlo.reshape %v2921 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3070 = stablehlo.transpose %v3068, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3071 = stablehlo.transpose %v3069, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v3072 = stablehlo.convolution(%v3070, %v3071)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<384x96x1x1xf32>
    %v3073 = stablehlo.transpose %v3072, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3074 = stablehlo.reshape %v960 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3075 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3076 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v3077 = stablehlo.reduce(%v3074 init: %v3075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3078 = stablehlo.broadcast_in_dim %v3077, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v3079 = stablehlo.divide %v3078, %v3076 : tensor<32x96x14x14xf32>
    %v3080 = stablehlo.subtract %v3074, %v3079 : tensor<32x96x14x14xf32>
    %v3081 = stablehlo.multiply %v3080, %v3080 : tensor<32x96x14x14xf32>
    %v3082 = stablehlo.reduce(%v3081 init: %v3075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3083 = stablehlo.broadcast_in_dim %v3082, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v3084 = stablehlo.divide %v3083, %v3076 : tensor<32x96x14x14xf32>
    %v3085 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v3086 = stablehlo.add %v3084, %v3085 : tensor<32x96x14x14xf32>
    %v3087 = stablehlo.rsqrt %v3086 : tensor<32x96x14x14xf32>
    %v3088 = stablehlo.multiply %v3080, %v3087 : tensor<32x96x14x14xf32>
    %v3089 = stablehlo.reshape %v2810 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3090 = stablehlo.multiply %v3089, %v3088 : tensor<32x96x14x14xf32>
    %v3091 = stablehlo.reduce(%v3090 init: %v3075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3092 = stablehlo.reshape %v2810 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3094 = stablehlo.reduce(%v3092 init: %v3093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3095 = stablehlo.reshape %v869 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3096 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3097 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3098 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3099 = stablehlo.reduce(%v3095 init: %v3096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3100 = stablehlo.broadcast_in_dim %v3099, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3101 = stablehlo.divide %v3100, %v3097 : tensor<32x64x14x14xf32>
    %v3102 = stablehlo.subtract %v3095, %v3101 : tensor<32x64x14x14xf32>
    %v3103 = stablehlo.multiply %v3102, %v3102 : tensor<32x64x14x14xf32>
    %v3104 = stablehlo.reduce(%v3103 init: %v3096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3105 = stablehlo.broadcast_in_dim %v3104, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3106 = stablehlo.divide %v3105, %v3097 : tensor<32x64x14x14xf32>
    %v3107 = stablehlo.add %v3106, %v3098 : tensor<32x64x14x14xf32>
    %v3108 = stablehlo.rsqrt %v3107 : tensor<32x64x14x14xf32>
    %v3109 = stablehlo.multiply %v3102, %v3108 : tensor<32x64x14x14xf32>
    %v3110 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3111 = stablehlo.reshape %v3013 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3112 = stablehlo.multiply %v3110, %v3111 : tensor<32x64x14x14xf32>
    %v3113 = stablehlo.reduce(%v3112 init: %v3096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3114 = stablehlo.broadcast_in_dim %v3113, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3115 = stablehlo.multiply %v3109, %v3112 : tensor<32x64x14x14xf32>
    %v3116 = stablehlo.reduce(%v3115 init: %v3096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3117 = stablehlo.broadcast_in_dim %v3116, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3118 = stablehlo.multiply %v3112, %v3097 : tensor<32x64x14x14xf32>
    %v3119 = stablehlo.subtract %v3118, %v3114 : tensor<32x64x14x14xf32>
    %v3120 = stablehlo.multiply %v3109, %v3117 : tensor<32x64x14x14xf32>
    %v3121 = stablehlo.subtract %v3119, %v3120 : tensor<32x64x14x14xf32>
    %v3122 = stablehlo.divide %v3108, %v3097 : tensor<32x64x14x14xf32>
    %v3123 = stablehlo.multiply %v3122, %v3121 : tensor<32x64x14x14xf32>
    %v3124 = stablehlo.reshape %v3123 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3125 = stablehlo.reshape %v3124 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3126 = stablehlo.reverse %b10pW, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3127 = stablehlo.transpose %v3126, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3128 = stablehlo.convolution(%v3125, %v3127)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3129 = stablehlo.reshape %v3128 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3130 = stablehlo.reshape %v3129 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3131 = stablehlo.reshape %v858 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3132 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3133 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3134 = stablehlo.compare GT, %v3131, %v3132 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3135 = stablehlo.compare LT, %v3131, %v3133 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3136 = stablehlo.and %v3134, %v3135 : tensor<32x384x14x14xi1>
    %v3137 = stablehlo.select %v3136, %v3130, %v3132 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3138 = stablehlo.reshape %v3137 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3139 = stablehlo.reshape %v838 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3141 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3142 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3143 = stablehlo.reduce(%v3139 init: %v3140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3144 = stablehlo.broadcast_in_dim %v3143, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3145 = stablehlo.divide %v3144, %v3141 : tensor<32x384x14x14xf32>
    %v3146 = stablehlo.subtract %v3139, %v3145 : tensor<32x384x14x14xf32>
    %v3147 = stablehlo.multiply %v3146, %v3146 : tensor<32x384x14x14xf32>
    %v3148 = stablehlo.reduce(%v3147 init: %v3140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3149 = stablehlo.broadcast_in_dim %v3148, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3150 = stablehlo.divide %v3149, %v3141 : tensor<32x384x14x14xf32>
    %v3151 = stablehlo.add %v3150, %v3142 : tensor<32x384x14x14xf32>
    %v3152 = stablehlo.rsqrt %v3151 : tensor<32x384x14x14xf32>
    %v3153 = stablehlo.multiply %v3146, %v3152 : tensor<32x384x14x14xf32>
    %v3154 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3155 = stablehlo.reshape %v3138 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3156 = stablehlo.multiply %v3154, %v3155 : tensor<32x384x14x14xf32>
    %v3157 = stablehlo.reduce(%v3156 init: %v3140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3158 = stablehlo.broadcast_in_dim %v3157, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3159 = stablehlo.multiply %v3153, %v3156 : tensor<32x384x14x14xf32>
    %v3160 = stablehlo.reduce(%v3159 init: %v3140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3161 = stablehlo.broadcast_in_dim %v3160, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3162 = stablehlo.multiply %v3156, %v3141 : tensor<32x384x14x14xf32>
    %v3163 = stablehlo.subtract %v3162, %v3158 : tensor<32x384x14x14xf32>
    %v3164 = stablehlo.multiply %v3153, %v3161 : tensor<32x384x14x14xf32>
    %v3165 = stablehlo.subtract %v3163, %v3164 : tensor<32x384x14x14xf32>
    %v3166 = stablehlo.divide %v3152, %v3141 : tensor<32x384x14x14xf32>
    %v3167 = stablehlo.multiply %v3166, %v3165 : tensor<32x384x14x14xf32>
    %v3168 = stablehlo.reshape %v3167 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3169 = stablehlo.reshape %v3168 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3170 = stablehlo.reverse %b10dW, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3171 = stablehlo.convolution(%v3169, %v3170)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3172 = stablehlo.reshape %v3171 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3173 = stablehlo.reshape %v3172 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3174 = stablehlo.reshape %v827 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3175 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3176 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3177 = stablehlo.compare GT, %v3174, %v3175 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3178 = stablehlo.compare LT, %v3174, %v3176 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3179 = stablehlo.and %v3177, %v3178 : tensor<32x384x14x14xi1>
    %v3180 = stablehlo.select %v3179, %v3173, %v3175 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3181 = stablehlo.reshape %v3180 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3182 = stablehlo.reshape %v807 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3183 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3184 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3185 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3186 = stablehlo.reduce(%v3182 init: %v3183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3187 = stablehlo.broadcast_in_dim %v3186, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3188 = stablehlo.divide %v3187, %v3184 : tensor<32x384x14x14xf32>
    %v3189 = stablehlo.subtract %v3182, %v3188 : tensor<32x384x14x14xf32>
    %v3190 = stablehlo.multiply %v3189, %v3189 : tensor<32x384x14x14xf32>
    %v3191 = stablehlo.reduce(%v3190 init: %v3183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3192 = stablehlo.broadcast_in_dim %v3191, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3193 = stablehlo.divide %v3192, %v3184 : tensor<32x384x14x14xf32>
    %v3194 = stablehlo.add %v3193, %v3185 : tensor<32x384x14x14xf32>
    %v3195 = stablehlo.rsqrt %v3194 : tensor<32x384x14x14xf32>
    %v3196 = stablehlo.multiply %v3189, %v3195 : tensor<32x384x14x14xf32>
    %v3197 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3198 = stablehlo.reshape %v3181 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3199 = stablehlo.multiply %v3197, %v3198 : tensor<32x384x14x14xf32>
    %v3200 = stablehlo.reduce(%v3199 init: %v3183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3201 = stablehlo.broadcast_in_dim %v3200, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3202 = stablehlo.multiply %v3196, %v3199 : tensor<32x384x14x14xf32>
    %v3203 = stablehlo.reduce(%v3202 init: %v3183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3204 = stablehlo.broadcast_in_dim %v3203, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3205 = stablehlo.multiply %v3199, %v3184 : tensor<32x384x14x14xf32>
    %v3206 = stablehlo.subtract %v3205, %v3201 : tensor<32x384x14x14xf32>
    %v3207 = stablehlo.multiply %v3196, %v3204 : tensor<32x384x14x14xf32>
    %v3208 = stablehlo.subtract %v3206, %v3207 : tensor<32x384x14x14xf32>
    %v3209 = stablehlo.divide %v3195, %v3184 : tensor<32x384x14x14xf32>
    %v3210 = stablehlo.multiply %v3209, %v3208 : tensor<32x384x14x14xf32>
    %v3211 = stablehlo.reshape %v3210 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3212 = stablehlo.reshape %v3211 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3213 = stablehlo.reverse %b10eW, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3214 = stablehlo.transpose %v3213, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3215 = stablehlo.convolution(%v3212, %v3214)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3216 = stablehlo.reshape %v3215 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3217 = stablehlo.reshape %v3216 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3218 = stablehlo.reshape %v3013 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3219 = stablehlo.add %v3217, %v3218 : tensor<32x64x14x14xf32>
    %v3220 = stablehlo.reshape %v3219 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3221 = stablehlo.reshape %v802 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3222 = stablehlo.reshape %v3211 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3223 = stablehlo.transpose %v3221, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3224 = stablehlo.transpose %v3222, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3225 = stablehlo.convolution(%v3223, %v3224)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3226 = stablehlo.transpose %v3225, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3227 = stablehlo.reshape %v807 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3228 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3229 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3230 = stablehlo.reduce(%v3227 init: %v3228) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3231 = stablehlo.broadcast_in_dim %v3230, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3232 = stablehlo.divide %v3231, %v3229 : tensor<32x384x14x14xf32>
    %v3233 = stablehlo.subtract %v3227, %v3232 : tensor<32x384x14x14xf32>
    %v3234 = stablehlo.multiply %v3233, %v3233 : tensor<32x384x14x14xf32>
    %v3235 = stablehlo.reduce(%v3234 init: %v3228) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3236 = stablehlo.broadcast_in_dim %v3235, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3237 = stablehlo.divide %v3236, %v3229 : tensor<32x384x14x14xf32>
    %v3238 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3239 = stablehlo.add %v3237, %v3238 : tensor<32x384x14x14xf32>
    %v3240 = stablehlo.rsqrt %v3239 : tensor<32x384x14x14xf32>
    %v3241 = stablehlo.multiply %v3233, %v3240 : tensor<32x384x14x14xf32>
    %v3242 = stablehlo.reshape %v3181 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3243 = stablehlo.multiply %v3242, %v3241 : tensor<32x384x14x14xf32>
    %v3244 = stablehlo.reduce(%v3243 init: %v3228) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3245 = stablehlo.reshape %v3181 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3247 = stablehlo.reduce(%v3245 init: %v3246) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3248 = stablehlo.reshape %v833 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3249 = stablehlo.reshape %v3168 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3250 = stablehlo.transpose %v3248, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3251 = stablehlo.transpose %v3249, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3252 = stablehlo.convolution(%v3250, %v3251)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3253 = stablehlo.reshape %v3252 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3254 = stablehlo.reshape %v838 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3255 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3256 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3257 = stablehlo.reduce(%v3254 init: %v3255) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3258 = stablehlo.broadcast_in_dim %v3257, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3259 = stablehlo.divide %v3258, %v3256 : tensor<32x384x14x14xf32>
    %v3260 = stablehlo.subtract %v3254, %v3259 : tensor<32x384x14x14xf32>
    %v3261 = stablehlo.multiply %v3260, %v3260 : tensor<32x384x14x14xf32>
    %v3262 = stablehlo.reduce(%v3261 init: %v3255) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3263 = stablehlo.broadcast_in_dim %v3262, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3264 = stablehlo.divide %v3263, %v3256 : tensor<32x384x14x14xf32>
    %v3265 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3266 = stablehlo.add %v3264, %v3265 : tensor<32x384x14x14xf32>
    %v3267 = stablehlo.rsqrt %v3266 : tensor<32x384x14x14xf32>
    %v3268 = stablehlo.multiply %v3260, %v3267 : tensor<32x384x14x14xf32>
    %v3269 = stablehlo.reshape %v3138 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3270 = stablehlo.multiply %v3269, %v3268 : tensor<32x384x14x14xf32>
    %v3271 = stablehlo.reduce(%v3270 init: %v3255) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3272 = stablehlo.reshape %v3138 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3274 = stablehlo.reduce(%v3272 init: %v3273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3275 = stablehlo.reshape %v864 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3276 = stablehlo.reshape %v3124 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3277 = stablehlo.transpose %v3275, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3278 = stablehlo.transpose %v3276, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3279 = stablehlo.convolution(%v3277, %v3278)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3280 = stablehlo.transpose %v3279, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3281 = stablehlo.reshape %v869 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3282 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3283 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3284 = stablehlo.reduce(%v3281 init: %v3282) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3285 = stablehlo.broadcast_in_dim %v3284, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3286 = stablehlo.divide %v3285, %v3283 : tensor<32x64x14x14xf32>
    %v3287 = stablehlo.subtract %v3281, %v3286 : tensor<32x64x14x14xf32>
    %v3288 = stablehlo.multiply %v3287, %v3287 : tensor<32x64x14x14xf32>
    %v3289 = stablehlo.reduce(%v3288 init: %v3282) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3290 = stablehlo.broadcast_in_dim %v3289, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3291 = stablehlo.divide %v3290, %v3283 : tensor<32x64x14x14xf32>
    %v3292 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3293 = stablehlo.add %v3291, %v3292 : tensor<32x64x14x14xf32>
    %v3294 = stablehlo.rsqrt %v3293 : tensor<32x64x14x14xf32>
    %v3295 = stablehlo.multiply %v3287, %v3294 : tensor<32x64x14x14xf32>
    %v3296 = stablehlo.reshape %v3013 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3297 = stablehlo.multiply %v3296, %v3295 : tensor<32x64x14x14xf32>
    %v3298 = stablehlo.reduce(%v3297 init: %v3282) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3299 = stablehlo.reshape %v3013 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3300 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3301 = stablehlo.reduce(%v3299 init: %v3300) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3302 = stablehlo.reshape %v778 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3304 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3305 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3306 = stablehlo.reduce(%v3302 init: %v3303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3307 = stablehlo.broadcast_in_dim %v3306, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3308 = stablehlo.divide %v3307, %v3304 : tensor<32x64x14x14xf32>
    %v3309 = stablehlo.subtract %v3302, %v3308 : tensor<32x64x14x14xf32>
    %v3310 = stablehlo.multiply %v3309, %v3309 : tensor<32x64x14x14xf32>
    %v3311 = stablehlo.reduce(%v3310 init: %v3303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3312 = stablehlo.broadcast_in_dim %v3311, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3313 = stablehlo.divide %v3312, %v3304 : tensor<32x64x14x14xf32>
    %v3314 = stablehlo.add %v3313, %v3305 : tensor<32x64x14x14xf32>
    %v3315 = stablehlo.rsqrt %v3314 : tensor<32x64x14x14xf32>
    %v3316 = stablehlo.multiply %v3309, %v3315 : tensor<32x64x14x14xf32>
    %v3317 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3318 = stablehlo.reshape %v3220 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3319 = stablehlo.multiply %v3317, %v3318 : tensor<32x64x14x14xf32>
    %v3320 = stablehlo.reduce(%v3319 init: %v3303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3321 = stablehlo.broadcast_in_dim %v3320, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3322 = stablehlo.multiply %v3316, %v3319 : tensor<32x64x14x14xf32>
    %v3323 = stablehlo.reduce(%v3322 init: %v3303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3324 = stablehlo.broadcast_in_dim %v3323, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3325 = stablehlo.multiply %v3319, %v3304 : tensor<32x64x14x14xf32>
    %v3326 = stablehlo.subtract %v3325, %v3321 : tensor<32x64x14x14xf32>
    %v3327 = stablehlo.multiply %v3316, %v3324 : tensor<32x64x14x14xf32>
    %v3328 = stablehlo.subtract %v3326, %v3327 : tensor<32x64x14x14xf32>
    %v3329 = stablehlo.divide %v3315, %v3304 : tensor<32x64x14x14xf32>
    %v3330 = stablehlo.multiply %v3329, %v3328 : tensor<32x64x14x14xf32>
    %v3331 = stablehlo.reshape %v3330 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3332 = stablehlo.reshape %v3331 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3333 = stablehlo.reverse %b9pW, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3334 = stablehlo.transpose %v3333, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3335 = stablehlo.convolution(%v3332, %v3334)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3336 = stablehlo.reshape %v3335 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3337 = stablehlo.reshape %v3336 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3338 = stablehlo.reshape %v767 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3339 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3340 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3341 = stablehlo.compare GT, %v3338, %v3339 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3342 = stablehlo.compare LT, %v3338, %v3340 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3343 = stablehlo.and %v3341, %v3342 : tensor<32x384x14x14xi1>
    %v3344 = stablehlo.select %v3343, %v3337, %v3339 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3345 = stablehlo.reshape %v3344 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3346 = stablehlo.reshape %v747 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3347 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3348 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3349 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3350 = stablehlo.reduce(%v3346 init: %v3347) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3351 = stablehlo.broadcast_in_dim %v3350, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3352 = stablehlo.divide %v3351, %v3348 : tensor<32x384x14x14xf32>
    %v3353 = stablehlo.subtract %v3346, %v3352 : tensor<32x384x14x14xf32>
    %v3354 = stablehlo.multiply %v3353, %v3353 : tensor<32x384x14x14xf32>
    %v3355 = stablehlo.reduce(%v3354 init: %v3347) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3356 = stablehlo.broadcast_in_dim %v3355, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3357 = stablehlo.divide %v3356, %v3348 : tensor<32x384x14x14xf32>
    %v3358 = stablehlo.add %v3357, %v3349 : tensor<32x384x14x14xf32>
    %v3359 = stablehlo.rsqrt %v3358 : tensor<32x384x14x14xf32>
    %v3360 = stablehlo.multiply %v3353, %v3359 : tensor<32x384x14x14xf32>
    %v3361 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3362 = stablehlo.reshape %v3345 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3363 = stablehlo.multiply %v3361, %v3362 : tensor<32x384x14x14xf32>
    %v3364 = stablehlo.reduce(%v3363 init: %v3347) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3365 = stablehlo.broadcast_in_dim %v3364, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3366 = stablehlo.multiply %v3360, %v3363 : tensor<32x384x14x14xf32>
    %v3367 = stablehlo.reduce(%v3366 init: %v3347) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3368 = stablehlo.broadcast_in_dim %v3367, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3369 = stablehlo.multiply %v3363, %v3348 : tensor<32x384x14x14xf32>
    %v3370 = stablehlo.subtract %v3369, %v3365 : tensor<32x384x14x14xf32>
    %v3371 = stablehlo.multiply %v3360, %v3368 : tensor<32x384x14x14xf32>
    %v3372 = stablehlo.subtract %v3370, %v3371 : tensor<32x384x14x14xf32>
    %v3373 = stablehlo.divide %v3359, %v3348 : tensor<32x384x14x14xf32>
    %v3374 = stablehlo.multiply %v3373, %v3372 : tensor<32x384x14x14xf32>
    %v3375 = stablehlo.reshape %v3374 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3376 = stablehlo.reshape %v3375 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3377 = stablehlo.reverse %b9dW, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3378 = stablehlo.convolution(%v3376, %v3377)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3379 = stablehlo.reshape %v3378 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3380 = stablehlo.reshape %v3379 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3381 = stablehlo.reshape %v736 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3382 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3383 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3384 = stablehlo.compare GT, %v3381, %v3382 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3385 = stablehlo.compare LT, %v3381, %v3383 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3386 = stablehlo.and %v3384, %v3385 : tensor<32x384x14x14xi1>
    %v3387 = stablehlo.select %v3386, %v3380, %v3382 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3388 = stablehlo.reshape %v3387 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3389 = stablehlo.reshape %v716 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3391 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3392 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3393 = stablehlo.reduce(%v3389 init: %v3390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3394 = stablehlo.broadcast_in_dim %v3393, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3395 = stablehlo.divide %v3394, %v3391 : tensor<32x384x14x14xf32>
    %v3396 = stablehlo.subtract %v3389, %v3395 : tensor<32x384x14x14xf32>
    %v3397 = stablehlo.multiply %v3396, %v3396 : tensor<32x384x14x14xf32>
    %v3398 = stablehlo.reduce(%v3397 init: %v3390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3399 = stablehlo.broadcast_in_dim %v3398, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3400 = stablehlo.divide %v3399, %v3391 : tensor<32x384x14x14xf32>
    %v3401 = stablehlo.add %v3400, %v3392 : tensor<32x384x14x14xf32>
    %v3402 = stablehlo.rsqrt %v3401 : tensor<32x384x14x14xf32>
    %v3403 = stablehlo.multiply %v3396, %v3402 : tensor<32x384x14x14xf32>
    %v3404 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3405 = stablehlo.reshape %v3388 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3406 = stablehlo.multiply %v3404, %v3405 : tensor<32x384x14x14xf32>
    %v3407 = stablehlo.reduce(%v3406 init: %v3390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3408 = stablehlo.broadcast_in_dim %v3407, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3409 = stablehlo.multiply %v3403, %v3406 : tensor<32x384x14x14xf32>
    %v3410 = stablehlo.reduce(%v3409 init: %v3390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3411 = stablehlo.broadcast_in_dim %v3410, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3412 = stablehlo.multiply %v3406, %v3391 : tensor<32x384x14x14xf32>
    %v3413 = stablehlo.subtract %v3412, %v3408 : tensor<32x384x14x14xf32>
    %v3414 = stablehlo.multiply %v3403, %v3411 : tensor<32x384x14x14xf32>
    %v3415 = stablehlo.subtract %v3413, %v3414 : tensor<32x384x14x14xf32>
    %v3416 = stablehlo.divide %v3402, %v3391 : tensor<32x384x14x14xf32>
    %v3417 = stablehlo.multiply %v3416, %v3415 : tensor<32x384x14x14xf32>
    %v3418 = stablehlo.reshape %v3417 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3419 = stablehlo.reshape %v3418 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3420 = stablehlo.reverse %b9eW, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3421 = stablehlo.transpose %v3420, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3422 = stablehlo.convolution(%v3419, %v3421)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3423 = stablehlo.reshape %v3422 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3424 = stablehlo.reshape %v3423 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3425 = stablehlo.reshape %v3220 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3426 = stablehlo.add %v3424, %v3425 : tensor<32x64x14x14xf32>
    %v3427 = stablehlo.reshape %v3426 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3428 = stablehlo.reshape %v711 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3429 = stablehlo.reshape %v3418 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3430 = stablehlo.transpose %v3428, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3431 = stablehlo.transpose %v3429, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3432 = stablehlo.convolution(%v3430, %v3431)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3433 = stablehlo.transpose %v3432, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3434 = stablehlo.reshape %v716 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3435 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3436 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3437 = stablehlo.reduce(%v3434 init: %v3435) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3438 = stablehlo.broadcast_in_dim %v3437, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3439 = stablehlo.divide %v3438, %v3436 : tensor<32x384x14x14xf32>
    %v3440 = stablehlo.subtract %v3434, %v3439 : tensor<32x384x14x14xf32>
    %v3441 = stablehlo.multiply %v3440, %v3440 : tensor<32x384x14x14xf32>
    %v3442 = stablehlo.reduce(%v3441 init: %v3435) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3443 = stablehlo.broadcast_in_dim %v3442, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3444 = stablehlo.divide %v3443, %v3436 : tensor<32x384x14x14xf32>
    %v3445 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3446 = stablehlo.add %v3444, %v3445 : tensor<32x384x14x14xf32>
    %v3447 = stablehlo.rsqrt %v3446 : tensor<32x384x14x14xf32>
    %v3448 = stablehlo.multiply %v3440, %v3447 : tensor<32x384x14x14xf32>
    %v3449 = stablehlo.reshape %v3388 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3450 = stablehlo.multiply %v3449, %v3448 : tensor<32x384x14x14xf32>
    %v3451 = stablehlo.reduce(%v3450 init: %v3435) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3452 = stablehlo.reshape %v3388 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3453 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3454 = stablehlo.reduce(%v3452 init: %v3453) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3455 = stablehlo.reshape %v742 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3456 = stablehlo.reshape %v3375 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3457 = stablehlo.transpose %v3455, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3458 = stablehlo.transpose %v3456, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3459 = stablehlo.convolution(%v3457, %v3458)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3460 = stablehlo.reshape %v3459 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3461 = stablehlo.reshape %v747 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3463 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3464 = stablehlo.reduce(%v3461 init: %v3462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3465 = stablehlo.broadcast_in_dim %v3464, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3466 = stablehlo.divide %v3465, %v3463 : tensor<32x384x14x14xf32>
    %v3467 = stablehlo.subtract %v3461, %v3466 : tensor<32x384x14x14xf32>
    %v3468 = stablehlo.multiply %v3467, %v3467 : tensor<32x384x14x14xf32>
    %v3469 = stablehlo.reduce(%v3468 init: %v3462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3470 = stablehlo.broadcast_in_dim %v3469, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3471 = stablehlo.divide %v3470, %v3463 : tensor<32x384x14x14xf32>
    %v3472 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3473 = stablehlo.add %v3471, %v3472 : tensor<32x384x14x14xf32>
    %v3474 = stablehlo.rsqrt %v3473 : tensor<32x384x14x14xf32>
    %v3475 = stablehlo.multiply %v3467, %v3474 : tensor<32x384x14x14xf32>
    %v3476 = stablehlo.reshape %v3345 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3477 = stablehlo.multiply %v3476, %v3475 : tensor<32x384x14x14xf32>
    %v3478 = stablehlo.reduce(%v3477 init: %v3462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3479 = stablehlo.reshape %v3345 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3481 = stablehlo.reduce(%v3479 init: %v3480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3482 = stablehlo.reshape %v773 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3483 = stablehlo.reshape %v3331 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3484 = stablehlo.transpose %v3482, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3485 = stablehlo.transpose %v3483, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3486 = stablehlo.convolution(%v3484, %v3485)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3487 = stablehlo.transpose %v3486, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3488 = stablehlo.reshape %v778 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3490 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3491 = stablehlo.reduce(%v3488 init: %v3489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3492 = stablehlo.broadcast_in_dim %v3491, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3493 = stablehlo.divide %v3492, %v3490 : tensor<32x64x14x14xf32>
    %v3494 = stablehlo.subtract %v3488, %v3493 : tensor<32x64x14x14xf32>
    %v3495 = stablehlo.multiply %v3494, %v3494 : tensor<32x64x14x14xf32>
    %v3496 = stablehlo.reduce(%v3495 init: %v3489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3497 = stablehlo.broadcast_in_dim %v3496, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3498 = stablehlo.divide %v3497, %v3490 : tensor<32x64x14x14xf32>
    %v3499 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3500 = stablehlo.add %v3498, %v3499 : tensor<32x64x14x14xf32>
    %v3501 = stablehlo.rsqrt %v3500 : tensor<32x64x14x14xf32>
    %v3502 = stablehlo.multiply %v3494, %v3501 : tensor<32x64x14x14xf32>
    %v3503 = stablehlo.reshape %v3220 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3504 = stablehlo.multiply %v3503, %v3502 : tensor<32x64x14x14xf32>
    %v3505 = stablehlo.reduce(%v3504 init: %v3489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3506 = stablehlo.reshape %v3220 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3507 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3508 = stablehlo.reduce(%v3506 init: %v3507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3509 = stablehlo.reshape %v687 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3510 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3511 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3512 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3513 = stablehlo.reduce(%v3509 init: %v3510) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3514 = stablehlo.broadcast_in_dim %v3513, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3515 = stablehlo.divide %v3514, %v3511 : tensor<32x64x14x14xf32>
    %v3516 = stablehlo.subtract %v3509, %v3515 : tensor<32x64x14x14xf32>
    %v3517 = stablehlo.multiply %v3516, %v3516 : tensor<32x64x14x14xf32>
    %v3518 = stablehlo.reduce(%v3517 init: %v3510) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3519 = stablehlo.broadcast_in_dim %v3518, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3520 = stablehlo.divide %v3519, %v3511 : tensor<32x64x14x14xf32>
    %v3521 = stablehlo.add %v3520, %v3512 : tensor<32x64x14x14xf32>
    %v3522 = stablehlo.rsqrt %v3521 : tensor<32x64x14x14xf32>
    %v3523 = stablehlo.multiply %v3516, %v3522 : tensor<32x64x14x14xf32>
    %v3524 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3525 = stablehlo.reshape %v3427 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3526 = stablehlo.multiply %v3524, %v3525 : tensor<32x64x14x14xf32>
    %v3527 = stablehlo.reduce(%v3526 init: %v3510) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3528 = stablehlo.broadcast_in_dim %v3527, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3529 = stablehlo.multiply %v3523, %v3526 : tensor<32x64x14x14xf32>
    %v3530 = stablehlo.reduce(%v3529 init: %v3510) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3531 = stablehlo.broadcast_in_dim %v3530, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3532 = stablehlo.multiply %v3526, %v3511 : tensor<32x64x14x14xf32>
    %v3533 = stablehlo.subtract %v3532, %v3528 : tensor<32x64x14x14xf32>
    %v3534 = stablehlo.multiply %v3523, %v3531 : tensor<32x64x14x14xf32>
    %v3535 = stablehlo.subtract %v3533, %v3534 : tensor<32x64x14x14xf32>
    %v3536 = stablehlo.divide %v3522, %v3511 : tensor<32x64x14x14xf32>
    %v3537 = stablehlo.multiply %v3536, %v3535 : tensor<32x64x14x14xf32>
    %v3538 = stablehlo.reshape %v3537 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3539 = stablehlo.reshape %v3538 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3540 = stablehlo.reverse %b8pW, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3541 = stablehlo.transpose %v3540, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3542 = stablehlo.convolution(%v3539, %v3541)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3543 = stablehlo.reshape %v3542 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3544 = stablehlo.reshape %v3543 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3545 = stablehlo.reshape %v676 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3546 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3547 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3548 = stablehlo.compare GT, %v3545, %v3546 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3549 = stablehlo.compare LT, %v3545, %v3547 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3550 = stablehlo.and %v3548, %v3549 : tensor<32x384x14x14xi1>
    %v3551 = stablehlo.select %v3550, %v3544, %v3546 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3552 = stablehlo.reshape %v3551 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3553 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3555 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3556 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3557 = stablehlo.reduce(%v3553 init: %v3554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3558 = stablehlo.broadcast_in_dim %v3557, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3559 = stablehlo.divide %v3558, %v3555 : tensor<32x384x14x14xf32>
    %v3560 = stablehlo.subtract %v3553, %v3559 : tensor<32x384x14x14xf32>
    %v3561 = stablehlo.multiply %v3560, %v3560 : tensor<32x384x14x14xf32>
    %v3562 = stablehlo.reduce(%v3561 init: %v3554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3563 = stablehlo.broadcast_in_dim %v3562, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3564 = stablehlo.divide %v3563, %v3555 : tensor<32x384x14x14xf32>
    %v3565 = stablehlo.add %v3564, %v3556 : tensor<32x384x14x14xf32>
    %v3566 = stablehlo.rsqrt %v3565 : tensor<32x384x14x14xf32>
    %v3567 = stablehlo.multiply %v3560, %v3566 : tensor<32x384x14x14xf32>
    %v3568 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3569 = stablehlo.reshape %v3552 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3570 = stablehlo.multiply %v3568, %v3569 : tensor<32x384x14x14xf32>
    %v3571 = stablehlo.reduce(%v3570 init: %v3554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3572 = stablehlo.broadcast_in_dim %v3571, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3573 = stablehlo.multiply %v3567, %v3570 : tensor<32x384x14x14xf32>
    %v3574 = stablehlo.reduce(%v3573 init: %v3554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3575 = stablehlo.broadcast_in_dim %v3574, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3576 = stablehlo.multiply %v3570, %v3555 : tensor<32x384x14x14xf32>
    %v3577 = stablehlo.subtract %v3576, %v3572 : tensor<32x384x14x14xf32>
    %v3578 = stablehlo.multiply %v3567, %v3575 : tensor<32x384x14x14xf32>
    %v3579 = stablehlo.subtract %v3577, %v3578 : tensor<32x384x14x14xf32>
    %v3580 = stablehlo.divide %v3566, %v3555 : tensor<32x384x14x14xf32>
    %v3581 = stablehlo.multiply %v3580, %v3579 : tensor<32x384x14x14xf32>
    %v3582 = stablehlo.reshape %v3581 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3583 = stablehlo.reshape %v3582 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3584 = stablehlo.reverse %b8dW, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3585 = stablehlo.convolution(%v3583, %v3584)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3586 = stablehlo.reshape %v3585 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3587 = stablehlo.reshape %v3586 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3588 = stablehlo.reshape %v645 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3589 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3590 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3591 = stablehlo.compare GT, %v3588, %v3589 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3592 = stablehlo.compare LT, %v3588, %v3590 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3593 = stablehlo.and %v3591, %v3592 : tensor<32x384x14x14xi1>
    %v3594 = stablehlo.select %v3593, %v3587, %v3589 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3595 = stablehlo.reshape %v3594 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3596 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3598 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3599 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3600 = stablehlo.reduce(%v3596 init: %v3597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3601 = stablehlo.broadcast_in_dim %v3600, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3602 = stablehlo.divide %v3601, %v3598 : tensor<32x384x14x14xf32>
    %v3603 = stablehlo.subtract %v3596, %v3602 : tensor<32x384x14x14xf32>
    %v3604 = stablehlo.multiply %v3603, %v3603 : tensor<32x384x14x14xf32>
    %v3605 = stablehlo.reduce(%v3604 init: %v3597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3606 = stablehlo.broadcast_in_dim %v3605, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3607 = stablehlo.divide %v3606, %v3598 : tensor<32x384x14x14xf32>
    %v3608 = stablehlo.add %v3607, %v3599 : tensor<32x384x14x14xf32>
    %v3609 = stablehlo.rsqrt %v3608 : tensor<32x384x14x14xf32>
    %v3610 = stablehlo.multiply %v3603, %v3609 : tensor<32x384x14x14xf32>
    %v3611 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3612 = stablehlo.reshape %v3595 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3613 = stablehlo.multiply %v3611, %v3612 : tensor<32x384x14x14xf32>
    %v3614 = stablehlo.reduce(%v3613 init: %v3597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3615 = stablehlo.broadcast_in_dim %v3614, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3616 = stablehlo.multiply %v3610, %v3613 : tensor<32x384x14x14xf32>
    %v3617 = stablehlo.reduce(%v3616 init: %v3597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3618 = stablehlo.broadcast_in_dim %v3617, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3619 = stablehlo.multiply %v3613, %v3598 : tensor<32x384x14x14xf32>
    %v3620 = stablehlo.subtract %v3619, %v3615 : tensor<32x384x14x14xf32>
    %v3621 = stablehlo.multiply %v3610, %v3618 : tensor<32x384x14x14xf32>
    %v3622 = stablehlo.subtract %v3620, %v3621 : tensor<32x384x14x14xf32>
    %v3623 = stablehlo.divide %v3609, %v3598 : tensor<32x384x14x14xf32>
    %v3624 = stablehlo.multiply %v3623, %v3622 : tensor<32x384x14x14xf32>
    %v3625 = stablehlo.reshape %v3624 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3626 = stablehlo.reshape %v3625 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3627 = stablehlo.reverse %b8eW, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3628 = stablehlo.transpose %v3627, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3629 = stablehlo.convolution(%v3626, %v3628)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3630 = stablehlo.reshape %v3629 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3631 = stablehlo.reshape %v3630 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3632 = stablehlo.reshape %v3427 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3633 = stablehlo.add %v3631, %v3632 : tensor<32x64x14x14xf32>
    %v3634 = stablehlo.reshape %v3633 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3635 = stablehlo.reshape %v620 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3636 = stablehlo.reshape %v3625 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3637 = stablehlo.transpose %v3635, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3638 = stablehlo.transpose %v3636, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3639 = stablehlo.convolution(%v3637, %v3638)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3640 = stablehlo.transpose %v3639, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3641 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3642 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3643 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3644 = stablehlo.reduce(%v3641 init: %v3642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3645 = stablehlo.broadcast_in_dim %v3644, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3646 = stablehlo.divide %v3645, %v3643 : tensor<32x384x14x14xf32>
    %v3647 = stablehlo.subtract %v3641, %v3646 : tensor<32x384x14x14xf32>
    %v3648 = stablehlo.multiply %v3647, %v3647 : tensor<32x384x14x14xf32>
    %v3649 = stablehlo.reduce(%v3648 init: %v3642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3650 = stablehlo.broadcast_in_dim %v3649, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3651 = stablehlo.divide %v3650, %v3643 : tensor<32x384x14x14xf32>
    %v3652 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3653 = stablehlo.add %v3651, %v3652 : tensor<32x384x14x14xf32>
    %v3654 = stablehlo.rsqrt %v3653 : tensor<32x384x14x14xf32>
    %v3655 = stablehlo.multiply %v3647, %v3654 : tensor<32x384x14x14xf32>
    %v3656 = stablehlo.reshape %v3595 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3657 = stablehlo.multiply %v3656, %v3655 : tensor<32x384x14x14xf32>
    %v3658 = stablehlo.reduce(%v3657 init: %v3642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3659 = stablehlo.reshape %v3595 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3660 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3661 = stablehlo.reduce(%v3659 init: %v3660) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3662 = stablehlo.reshape %v651 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3663 = stablehlo.reshape %v3582 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3664 = stablehlo.transpose %v3662, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3665 = stablehlo.transpose %v3663, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3666 = stablehlo.convolution(%v3664, %v3665)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3667 = stablehlo.reshape %v3666 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3668 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3669 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3670 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v3671 = stablehlo.reduce(%v3668 init: %v3669) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3672 = stablehlo.broadcast_in_dim %v3671, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3673 = stablehlo.divide %v3672, %v3670 : tensor<32x384x14x14xf32>
    %v3674 = stablehlo.subtract %v3668, %v3673 : tensor<32x384x14x14xf32>
    %v3675 = stablehlo.multiply %v3674, %v3674 : tensor<32x384x14x14xf32>
    %v3676 = stablehlo.reduce(%v3675 init: %v3669) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3677 = stablehlo.broadcast_in_dim %v3676, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3678 = stablehlo.divide %v3677, %v3670 : tensor<32x384x14x14xf32>
    %v3679 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3680 = stablehlo.add %v3678, %v3679 : tensor<32x384x14x14xf32>
    %v3681 = stablehlo.rsqrt %v3680 : tensor<32x384x14x14xf32>
    %v3682 = stablehlo.multiply %v3674, %v3681 : tensor<32x384x14x14xf32>
    %v3683 = stablehlo.reshape %v3552 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3684 = stablehlo.multiply %v3683, %v3682 : tensor<32x384x14x14xf32>
    %v3685 = stablehlo.reduce(%v3684 init: %v3669) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3686 = stablehlo.reshape %v3552 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3687 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3688 = stablehlo.reduce(%v3686 init: %v3687) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3689 = stablehlo.reshape %v682 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3690 = stablehlo.reshape %v3538 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3691 = stablehlo.transpose %v3689, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3692 = stablehlo.transpose %v3690, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3693 = stablehlo.convolution(%v3691, %v3692)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3694 = stablehlo.transpose %v3693, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3695 = stablehlo.reshape %v687 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3696 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3697 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3698 = stablehlo.reduce(%v3695 init: %v3696) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3699 = stablehlo.broadcast_in_dim %v3698, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3700 = stablehlo.divide %v3699, %v3697 : tensor<32x64x14x14xf32>
    %v3701 = stablehlo.subtract %v3695, %v3700 : tensor<32x64x14x14xf32>
    %v3702 = stablehlo.multiply %v3701, %v3701 : tensor<32x64x14x14xf32>
    %v3703 = stablehlo.reduce(%v3702 init: %v3696) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3704 = stablehlo.broadcast_in_dim %v3703, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3705 = stablehlo.divide %v3704, %v3697 : tensor<32x64x14x14xf32>
    %v3706 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3707 = stablehlo.add %v3705, %v3706 : tensor<32x64x14x14xf32>
    %v3708 = stablehlo.rsqrt %v3707 : tensor<32x64x14x14xf32>
    %v3709 = stablehlo.multiply %v3701, %v3708 : tensor<32x64x14x14xf32>
    %v3710 = stablehlo.reshape %v3427 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3711 = stablehlo.multiply %v3710, %v3709 : tensor<32x64x14x14xf32>
    %v3712 = stablehlo.reduce(%v3711 init: %v3696) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3713 = stablehlo.reshape %v3427 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3715 = stablehlo.reduce(%v3713 init: %v3714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3716 = stablehlo.reshape %v600 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3717 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3718 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3719 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3720 = stablehlo.reduce(%v3716 init: %v3717) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3721 = stablehlo.broadcast_in_dim %v3720, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3722 = stablehlo.divide %v3721, %v3718 : tensor<32x64x14x14xf32>
    %v3723 = stablehlo.subtract %v3716, %v3722 : tensor<32x64x14x14xf32>
    %v3724 = stablehlo.multiply %v3723, %v3723 : tensor<32x64x14x14xf32>
    %v3725 = stablehlo.reduce(%v3724 init: %v3717) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3726 = stablehlo.broadcast_in_dim %v3725, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3727 = stablehlo.divide %v3726, %v3718 : tensor<32x64x14x14xf32>
    %v3728 = stablehlo.add %v3727, %v3719 : tensor<32x64x14x14xf32>
    %v3729 = stablehlo.rsqrt %v3728 : tensor<32x64x14x14xf32>
    %v3730 = stablehlo.multiply %v3723, %v3729 : tensor<32x64x14x14xf32>
    %v3731 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3732 = stablehlo.reshape %v3634 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3733 = stablehlo.multiply %v3731, %v3732 : tensor<32x64x14x14xf32>
    %v3734 = stablehlo.reduce(%v3733 init: %v3717) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3735 = stablehlo.broadcast_in_dim %v3734, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3736 = stablehlo.multiply %v3730, %v3733 : tensor<32x64x14x14xf32>
    %v3737 = stablehlo.reduce(%v3736 init: %v3717) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3738 = stablehlo.broadcast_in_dim %v3737, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3739 = stablehlo.multiply %v3733, %v3718 : tensor<32x64x14x14xf32>
    %v3740 = stablehlo.subtract %v3739, %v3735 : tensor<32x64x14x14xf32>
    %v3741 = stablehlo.multiply %v3730, %v3738 : tensor<32x64x14x14xf32>
    %v3742 = stablehlo.subtract %v3740, %v3741 : tensor<32x64x14x14xf32>
    %v3743 = stablehlo.divide %v3729, %v3718 : tensor<32x64x14x14xf32>
    %v3744 = stablehlo.multiply %v3743, %v3742 : tensor<32x64x14x14xf32>
    %v3745 = stablehlo.reshape %v3744 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3746 = stablehlo.reshape %v3745 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3747 = stablehlo.reverse %b7pW, dims = [2, 3] : tensor<64x192x1x1xf32>
    %v3748 = stablehlo.transpose %v3747, dims = [1, 0, 2, 3] : (tensor<64x192x1x1xf32>) -> tensor<192x64x1x1xf32>
    %v3749 = stablehlo.convolution(%v3746, %v3748)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<192x64x1x1xf32>) -> tensor<32x192x14x14xf32>
    %v3750 = stablehlo.reshape %v3749 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v3751 = stablehlo.reshape %v3750 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3752 = stablehlo.reshape %v589 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3753 = stablehlo.constant dense<0.0> : tensor<32x192x14x14xf32>
    %v3754 = stablehlo.constant dense<6.0> : tensor<32x192x14x14xf32>
    %v3755 = stablehlo.compare GT, %v3752, %v3753 : (tensor<32x192x14x14xf32>, tensor<32x192x14x14xf32>) -> tensor<32x192x14x14xi1>
    %v3756 = stablehlo.compare LT, %v3752, %v3754 : (tensor<32x192x14x14xf32>, tensor<32x192x14x14xf32>) -> tensor<32x192x14x14xi1>
    %v3757 = stablehlo.and %v3755, %v3756 : tensor<32x192x14x14xi1>
    %v3758 = stablehlo.select %v3757, %v3751, %v3753 : tensor<32x192x14x14xi1>, tensor<32x192x14x14xf32>
    %v3759 = stablehlo.reshape %v3758 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v3760 = stablehlo.reshape %v569 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3761 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3762 = stablehlo.constant dense<6272.0> : tensor<32x192x14x14xf32>
    %v3763 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v3764 = stablehlo.reduce(%v3760 init: %v3761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3765 = stablehlo.broadcast_in_dim %v3764, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3766 = stablehlo.divide %v3765, %v3762 : tensor<32x192x14x14xf32>
    %v3767 = stablehlo.subtract %v3760, %v3766 : tensor<32x192x14x14xf32>
    %v3768 = stablehlo.multiply %v3767, %v3767 : tensor<32x192x14x14xf32>
    %v3769 = stablehlo.reduce(%v3768 init: %v3761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3770 = stablehlo.broadcast_in_dim %v3769, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3771 = stablehlo.divide %v3770, %v3762 : tensor<32x192x14x14xf32>
    %v3772 = stablehlo.add %v3771, %v3763 : tensor<32x192x14x14xf32>
    %v3773 = stablehlo.rsqrt %v3772 : tensor<32x192x14x14xf32>
    %v3774 = stablehlo.multiply %v3767, %v3773 : tensor<32x192x14x14xf32>
    %v3775 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3776 = stablehlo.reshape %v3759 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3777 = stablehlo.multiply %v3775, %v3776 : tensor<32x192x14x14xf32>
    %v3778 = stablehlo.reduce(%v3777 init: %v3761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3779 = stablehlo.broadcast_in_dim %v3778, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3780 = stablehlo.multiply %v3774, %v3777 : tensor<32x192x14x14xf32>
    %v3781 = stablehlo.reduce(%v3780 init: %v3761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3782 = stablehlo.broadcast_in_dim %v3781, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3783 = stablehlo.multiply %v3777, %v3762 : tensor<32x192x14x14xf32>
    %v3784 = stablehlo.subtract %v3783, %v3779 : tensor<32x192x14x14xf32>
    %v3785 = stablehlo.multiply %v3774, %v3782 : tensor<32x192x14x14xf32>
    %v3786 = stablehlo.subtract %v3784, %v3785 : tensor<32x192x14x14xf32>
    %v3787 = stablehlo.divide %v3773, %v3762 : tensor<32x192x14x14xf32>
    %v3788 = stablehlo.multiply %v3787, %v3786 : tensor<32x192x14x14xf32>
    %v3789 = stablehlo.reshape %v3788 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v3790 = stablehlo.reshape %v3789 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3791 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3792 = stablehlo.pad %v3790, %v3791, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192x28x28xf32>
    %v3793 = stablehlo.reverse %b7dW, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v3794 = stablehlo.convolution(%v3792, %v3793)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 0], [2, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v3795 = stablehlo.reshape %v3794 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3796 = stablehlo.reshape %v3795 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3797 = stablehlo.reshape %v558 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3798 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v3799 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v3800 = stablehlo.compare GT, %v3797, %v3798 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v3801 = stablehlo.compare LT, %v3797, %v3799 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v3802 = stablehlo.and %v3800, %v3801 : tensor<32x192x28x28xi1>
    %v3803 = stablehlo.select %v3802, %v3796, %v3798 : tensor<32x192x28x28xi1>, tensor<32x192x28x28xf32>
    %v3804 = stablehlo.reshape %v3803 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3805 = stablehlo.reshape %v538 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3806 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3807 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v3808 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v3809 = stablehlo.reduce(%v3805 init: %v3806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3810 = stablehlo.broadcast_in_dim %v3809, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3811 = stablehlo.divide %v3810, %v3807 : tensor<32x192x28x28xf32>
    %v3812 = stablehlo.subtract %v3805, %v3811 : tensor<32x192x28x28xf32>
    %v3813 = stablehlo.multiply %v3812, %v3812 : tensor<32x192x28x28xf32>
    %v3814 = stablehlo.reduce(%v3813 init: %v3806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3815 = stablehlo.broadcast_in_dim %v3814, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3816 = stablehlo.divide %v3815, %v3807 : tensor<32x192x28x28xf32>
    %v3817 = stablehlo.add %v3816, %v3808 : tensor<32x192x28x28xf32>
    %v3818 = stablehlo.rsqrt %v3817 : tensor<32x192x28x28xf32>
    %v3819 = stablehlo.multiply %v3812, %v3818 : tensor<32x192x28x28xf32>
    %v3820 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3821 = stablehlo.reshape %v3804 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3822 = stablehlo.multiply %v3820, %v3821 : tensor<32x192x28x28xf32>
    %v3823 = stablehlo.reduce(%v3822 init: %v3806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3824 = stablehlo.broadcast_in_dim %v3823, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3825 = stablehlo.multiply %v3819, %v3822 : tensor<32x192x28x28xf32>
    %v3826 = stablehlo.reduce(%v3825 init: %v3806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3827 = stablehlo.broadcast_in_dim %v3826, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3828 = stablehlo.multiply %v3822, %v3807 : tensor<32x192x28x28xf32>
    %v3829 = stablehlo.subtract %v3828, %v3824 : tensor<32x192x28x28xf32>
    %v3830 = stablehlo.multiply %v3819, %v3827 : tensor<32x192x28x28xf32>
    %v3831 = stablehlo.subtract %v3829, %v3830 : tensor<32x192x28x28xf32>
    %v3832 = stablehlo.divide %v3818, %v3807 : tensor<32x192x28x28xf32>
    %v3833 = stablehlo.multiply %v3832, %v3831 : tensor<32x192x28x28xf32>
    %v3834 = stablehlo.reshape %v3833 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3835 = stablehlo.reshape %v3834 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3836 = stablehlo.reverse %b7eW, dims = [2, 3] : tensor<192x32x1x1xf32>
    %v3837 = stablehlo.transpose %v3836, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v3838 = stablehlo.convolution(%v3835, %v3837)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v3839 = stablehlo.reshape %v3838 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v3840 = stablehlo.reshape %v533 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3841 = stablehlo.reshape %v3834 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3842 = stablehlo.transpose %v3840, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v3843 = stablehlo.transpose %v3841, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3844 = stablehlo.convolution(%v3842, %v3843)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v3845 = stablehlo.transpose %v3844, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v3846 = stablehlo.reshape %v538 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3847 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3848 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v3849 = stablehlo.reduce(%v3846 init: %v3847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3850 = stablehlo.broadcast_in_dim %v3849, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3851 = stablehlo.divide %v3850, %v3848 : tensor<32x192x28x28xf32>
    %v3852 = stablehlo.subtract %v3846, %v3851 : tensor<32x192x28x28xf32>
    %v3853 = stablehlo.multiply %v3852, %v3852 : tensor<32x192x28x28xf32>
    %v3854 = stablehlo.reduce(%v3853 init: %v3847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3855 = stablehlo.broadcast_in_dim %v3854, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3856 = stablehlo.divide %v3855, %v3848 : tensor<32x192x28x28xf32>
    %v3857 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v3858 = stablehlo.add %v3856, %v3857 : tensor<32x192x28x28xf32>
    %v3859 = stablehlo.rsqrt %v3858 : tensor<32x192x28x28xf32>
    %v3860 = stablehlo.multiply %v3852, %v3859 : tensor<32x192x28x28xf32>
    %v3861 = stablehlo.reshape %v3804 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3862 = stablehlo.multiply %v3861, %v3860 : tensor<32x192x28x28xf32>
    %v3863 = stablehlo.reduce(%v3862 init: %v3847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3864 = stablehlo.reshape %v3804 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3865 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3866 = stablehlo.reduce(%v3864 init: %v3865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3867 = stablehlo.reshape %v564 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3868 = stablehlo.reshape %v3789 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3870 = stablehlo.pad %v3868, %v3869, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192x28x28xf32>
    %v3871 = stablehlo.transpose %v3867, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3872 = stablehlo.transpose %v3870, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3873 = stablehlo.convolution(%v3871, %v3872)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 2], [0, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v3874 = stablehlo.reshape %v3873 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v3875 = stablehlo.reshape %v569 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3876 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3877 = stablehlo.constant dense<6272.0> : tensor<32x192x14x14xf32>
    %v3878 = stablehlo.reduce(%v3875 init: %v3876) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3879 = stablehlo.broadcast_in_dim %v3878, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3880 = stablehlo.divide %v3879, %v3877 : tensor<32x192x14x14xf32>
    %v3881 = stablehlo.subtract %v3875, %v3880 : tensor<32x192x14x14xf32>
    %v3882 = stablehlo.multiply %v3881, %v3881 : tensor<32x192x14x14xf32>
    %v3883 = stablehlo.reduce(%v3882 init: %v3876) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3884 = stablehlo.broadcast_in_dim %v3883, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3885 = stablehlo.divide %v3884, %v3877 : tensor<32x192x14x14xf32>
    %v3886 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v3887 = stablehlo.add %v3885, %v3886 : tensor<32x192x14x14xf32>
    %v3888 = stablehlo.rsqrt %v3887 : tensor<32x192x14x14xf32>
    %v3889 = stablehlo.multiply %v3881, %v3888 : tensor<32x192x14x14xf32>
    %v3890 = stablehlo.reshape %v3759 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3891 = stablehlo.multiply %v3890, %v3889 : tensor<32x192x14x14xf32>
    %v3892 = stablehlo.reduce(%v3891 init: %v3876) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3893 = stablehlo.reshape %v3759 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3895 = stablehlo.reduce(%v3893 init: %v3894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3896 = stablehlo.reshape %v595 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3897 = stablehlo.reshape %v3745 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3898 = stablehlo.transpose %v3896, dims = [1, 0, 2, 3] : (tensor<32x192x14x14xf32>) -> tensor<192x32x14x14xf32>
    %v3899 = stablehlo.transpose %v3897, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3900 = stablehlo.convolution(%v3898, %v3899)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<192x64x1x1xf32>
    %v3901 = stablehlo.transpose %v3900, dims = [1, 0, 2, 3] : (tensor<192x64x1x1xf32>) -> tensor<64x192x1x1xf32>
    %v3902 = stablehlo.reshape %v600 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3903 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3904 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v3905 = stablehlo.reduce(%v3902 init: %v3903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3906 = stablehlo.broadcast_in_dim %v3905, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3907 = stablehlo.divide %v3906, %v3904 : tensor<32x64x14x14xf32>
    %v3908 = stablehlo.subtract %v3902, %v3907 : tensor<32x64x14x14xf32>
    %v3909 = stablehlo.multiply %v3908, %v3908 : tensor<32x64x14x14xf32>
    %v3910 = stablehlo.reduce(%v3909 init: %v3903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3911 = stablehlo.broadcast_in_dim %v3910, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3912 = stablehlo.divide %v3911, %v3904 : tensor<32x64x14x14xf32>
    %v3913 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3914 = stablehlo.add %v3912, %v3913 : tensor<32x64x14x14xf32>
    %v3915 = stablehlo.rsqrt %v3914 : tensor<32x64x14x14xf32>
    %v3916 = stablehlo.multiply %v3908, %v3915 : tensor<32x64x14x14xf32>
    %v3917 = stablehlo.reshape %v3634 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3918 = stablehlo.multiply %v3917, %v3916 : tensor<32x64x14x14xf32>
    %v3919 = stablehlo.reduce(%v3918 init: %v3903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3920 = stablehlo.reshape %v3634 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3922 = stablehlo.reduce(%v3920 init: %v3921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3923 = stablehlo.reshape %v509 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3925 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v3926 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v3927 = stablehlo.reduce(%v3923 init: %v3924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3928 = stablehlo.broadcast_in_dim %v3927, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3929 = stablehlo.divide %v3928, %v3925 : tensor<32x32x28x28xf32>
    %v3930 = stablehlo.subtract %v3923, %v3929 : tensor<32x32x28x28xf32>
    %v3931 = stablehlo.multiply %v3930, %v3930 : tensor<32x32x28x28xf32>
    %v3932 = stablehlo.reduce(%v3931 init: %v3924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3933 = stablehlo.broadcast_in_dim %v3932, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3934 = stablehlo.divide %v3933, %v3925 : tensor<32x32x28x28xf32>
    %v3935 = stablehlo.add %v3934, %v3926 : tensor<32x32x28x28xf32>
    %v3936 = stablehlo.rsqrt %v3935 : tensor<32x32x28x28xf32>
    %v3937 = stablehlo.multiply %v3930, %v3936 : tensor<32x32x28x28xf32>
    %v3938 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3939 = stablehlo.reshape %v3839 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3940 = stablehlo.multiply %v3938, %v3939 : tensor<32x32x28x28xf32>
    %v3941 = stablehlo.reduce(%v3940 init: %v3924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3942 = stablehlo.broadcast_in_dim %v3941, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3943 = stablehlo.multiply %v3937, %v3940 : tensor<32x32x28x28xf32>
    %v3944 = stablehlo.reduce(%v3943 init: %v3924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v3945 = stablehlo.broadcast_in_dim %v3944, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v3946 = stablehlo.multiply %v3940, %v3925 : tensor<32x32x28x28xf32>
    %v3947 = stablehlo.subtract %v3946, %v3942 : tensor<32x32x28x28xf32>
    %v3948 = stablehlo.multiply %v3937, %v3945 : tensor<32x32x28x28xf32>
    %v3949 = stablehlo.subtract %v3947, %v3948 : tensor<32x32x28x28xf32>
    %v3950 = stablehlo.divide %v3936, %v3925 : tensor<32x32x28x28xf32>
    %v3951 = stablehlo.multiply %v3950, %v3949 : tensor<32x32x28x28xf32>
    %v3952 = stablehlo.reshape %v3951 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v3953 = stablehlo.reshape %v3952 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3954 = stablehlo.reverse %b6pW, dims = [2, 3] : tensor<32x192x1x1xf32>
    %v3955 = stablehlo.transpose %v3954, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v3956 = stablehlo.convolution(%v3953, %v3955)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3957 = stablehlo.reshape %v3956 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3958 = stablehlo.reshape %v3957 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3959 = stablehlo.reshape %v498 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3960 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v3961 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v3962 = stablehlo.compare GT, %v3959, %v3960 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v3963 = stablehlo.compare LT, %v3959, %v3961 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v3964 = stablehlo.and %v3962, %v3963 : tensor<32x192x28x28xi1>
    %v3965 = stablehlo.select %v3964, %v3958, %v3960 : tensor<32x192x28x28xi1>, tensor<32x192x28x28xf32>
    %v3966 = stablehlo.reshape %v3965 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3967 = stablehlo.reshape %v478 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3968 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3969 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v3970 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v3971 = stablehlo.reduce(%v3967 init: %v3968) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3972 = stablehlo.broadcast_in_dim %v3971, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3973 = stablehlo.divide %v3972, %v3969 : tensor<32x192x28x28xf32>
    %v3974 = stablehlo.subtract %v3967, %v3973 : tensor<32x192x28x28xf32>
    %v3975 = stablehlo.multiply %v3974, %v3974 : tensor<32x192x28x28xf32>
    %v3976 = stablehlo.reduce(%v3975 init: %v3968) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3977 = stablehlo.broadcast_in_dim %v3976, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3978 = stablehlo.divide %v3977, %v3969 : tensor<32x192x28x28xf32>
    %v3979 = stablehlo.add %v3978, %v3970 : tensor<32x192x28x28xf32>
    %v3980 = stablehlo.rsqrt %v3979 : tensor<32x192x28x28xf32>
    %v3981 = stablehlo.multiply %v3974, %v3980 : tensor<32x192x28x28xf32>
    %v3982 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3983 = stablehlo.reshape %v3966 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3984 = stablehlo.multiply %v3982, %v3983 : tensor<32x192x28x28xf32>
    %v3985 = stablehlo.reduce(%v3984 init: %v3968) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3986 = stablehlo.broadcast_in_dim %v3985, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3987 = stablehlo.multiply %v3981, %v3984 : tensor<32x192x28x28xf32>
    %v3988 = stablehlo.reduce(%v3987 init: %v3968) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3989 = stablehlo.broadcast_in_dim %v3988, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3990 = stablehlo.multiply %v3984, %v3969 : tensor<32x192x28x28xf32>
    %v3991 = stablehlo.subtract %v3990, %v3986 : tensor<32x192x28x28xf32>
    %v3992 = stablehlo.multiply %v3981, %v3989 : tensor<32x192x28x28xf32>
    %v3993 = stablehlo.subtract %v3991, %v3992 : tensor<32x192x28x28xf32>
    %v3994 = stablehlo.divide %v3980, %v3969 : tensor<32x192x28x28xf32>
    %v3995 = stablehlo.multiply %v3994, %v3993 : tensor<32x192x28x28xf32>
    %v3996 = stablehlo.reshape %v3995 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3997 = stablehlo.reshape %v3996 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3998 = stablehlo.reverse %b6dW, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v3999 = stablehlo.convolution(%v3997, %v3998)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v4000 = stablehlo.reshape %v3999 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4001 = stablehlo.reshape %v4000 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4002 = stablehlo.reshape %v467 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4003 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v4004 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v4005 = stablehlo.compare GT, %v4002, %v4003 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4006 = stablehlo.compare LT, %v4002, %v4004 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4007 = stablehlo.and %v4005, %v4006 : tensor<32x192x28x28xi1>
    %v4008 = stablehlo.select %v4007, %v4001, %v4003 : tensor<32x192x28x28xi1>, tensor<32x192x28x28xf32>
    %v4009 = stablehlo.reshape %v4008 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4010 = stablehlo.reshape %v447 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4012 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4013 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4014 = stablehlo.reduce(%v4010 init: %v4011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4015 = stablehlo.broadcast_in_dim %v4014, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4016 = stablehlo.divide %v4015, %v4012 : tensor<32x192x28x28xf32>
    %v4017 = stablehlo.subtract %v4010, %v4016 : tensor<32x192x28x28xf32>
    %v4018 = stablehlo.multiply %v4017, %v4017 : tensor<32x192x28x28xf32>
    %v4019 = stablehlo.reduce(%v4018 init: %v4011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4020 = stablehlo.broadcast_in_dim %v4019, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4021 = stablehlo.divide %v4020, %v4012 : tensor<32x192x28x28xf32>
    %v4022 = stablehlo.add %v4021, %v4013 : tensor<32x192x28x28xf32>
    %v4023 = stablehlo.rsqrt %v4022 : tensor<32x192x28x28xf32>
    %v4024 = stablehlo.multiply %v4017, %v4023 : tensor<32x192x28x28xf32>
    %v4025 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4026 = stablehlo.reshape %v4009 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4027 = stablehlo.multiply %v4025, %v4026 : tensor<32x192x28x28xf32>
    %v4028 = stablehlo.reduce(%v4027 init: %v4011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4029 = stablehlo.broadcast_in_dim %v4028, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4030 = stablehlo.multiply %v4024, %v4027 : tensor<32x192x28x28xf32>
    %v4031 = stablehlo.reduce(%v4030 init: %v4011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4032 = stablehlo.broadcast_in_dim %v4031, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4033 = stablehlo.multiply %v4027, %v4012 : tensor<32x192x28x28xf32>
    %v4034 = stablehlo.subtract %v4033, %v4029 : tensor<32x192x28x28xf32>
    %v4035 = stablehlo.multiply %v4024, %v4032 : tensor<32x192x28x28xf32>
    %v4036 = stablehlo.subtract %v4034, %v4035 : tensor<32x192x28x28xf32>
    %v4037 = stablehlo.divide %v4023, %v4012 : tensor<32x192x28x28xf32>
    %v4038 = stablehlo.multiply %v4037, %v4036 : tensor<32x192x28x28xf32>
    %v4039 = stablehlo.reshape %v4038 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4040 = stablehlo.reshape %v4039 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4041 = stablehlo.reverse %b6eW, dims = [2, 3] : tensor<192x32x1x1xf32>
    %v4042 = stablehlo.transpose %v4041, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4043 = stablehlo.convolution(%v4040, %v4042)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v4044 = stablehlo.reshape %v4043 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4045 = stablehlo.reshape %v4044 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4046 = stablehlo.reshape %v3839 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4047 = stablehlo.add %v4045, %v4046 : tensor<32x32x28x28xf32>
    %v4048 = stablehlo.reshape %v4047 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4049 = stablehlo.reshape %v442 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4050 = stablehlo.reshape %v4039 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4051 = stablehlo.transpose %v4049, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4052 = stablehlo.transpose %v4050, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4053 = stablehlo.convolution(%v4051, %v4052)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v4054 = stablehlo.transpose %v4053, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4055 = stablehlo.reshape %v447 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4056 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4057 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4058 = stablehlo.reduce(%v4055 init: %v4056) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4059 = stablehlo.broadcast_in_dim %v4058, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4060 = stablehlo.divide %v4059, %v4057 : tensor<32x192x28x28xf32>
    %v4061 = stablehlo.subtract %v4055, %v4060 : tensor<32x192x28x28xf32>
    %v4062 = stablehlo.multiply %v4061, %v4061 : tensor<32x192x28x28xf32>
    %v4063 = stablehlo.reduce(%v4062 init: %v4056) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4064 = stablehlo.broadcast_in_dim %v4063, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4065 = stablehlo.divide %v4064, %v4057 : tensor<32x192x28x28xf32>
    %v4066 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4067 = stablehlo.add %v4065, %v4066 : tensor<32x192x28x28xf32>
    %v4068 = stablehlo.rsqrt %v4067 : tensor<32x192x28x28xf32>
    %v4069 = stablehlo.multiply %v4061, %v4068 : tensor<32x192x28x28xf32>
    %v4070 = stablehlo.reshape %v4009 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4071 = stablehlo.multiply %v4070, %v4069 : tensor<32x192x28x28xf32>
    %v4072 = stablehlo.reduce(%v4071 init: %v4056) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4073 = stablehlo.reshape %v4009 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4074 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4075 = stablehlo.reduce(%v4073 init: %v4074) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4076 = stablehlo.reshape %v473 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4077 = stablehlo.reshape %v3996 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4078 = stablehlo.transpose %v4076, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4079 = stablehlo.transpose %v4077, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4080 = stablehlo.convolution(%v4078, %v4079)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v4081 = stablehlo.reshape %v4080 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v4082 = stablehlo.reshape %v478 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4083 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4084 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4085 = stablehlo.reduce(%v4082 init: %v4083) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4086 = stablehlo.broadcast_in_dim %v4085, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4087 = stablehlo.divide %v4086, %v4084 : tensor<32x192x28x28xf32>
    %v4088 = stablehlo.subtract %v4082, %v4087 : tensor<32x192x28x28xf32>
    %v4089 = stablehlo.multiply %v4088, %v4088 : tensor<32x192x28x28xf32>
    %v4090 = stablehlo.reduce(%v4089 init: %v4083) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4091 = stablehlo.broadcast_in_dim %v4090, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4092 = stablehlo.divide %v4091, %v4084 : tensor<32x192x28x28xf32>
    %v4093 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4094 = stablehlo.add %v4092, %v4093 : tensor<32x192x28x28xf32>
    %v4095 = stablehlo.rsqrt %v4094 : tensor<32x192x28x28xf32>
    %v4096 = stablehlo.multiply %v4088, %v4095 : tensor<32x192x28x28xf32>
    %v4097 = stablehlo.reshape %v3966 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4098 = stablehlo.multiply %v4097, %v4096 : tensor<32x192x28x28xf32>
    %v4099 = stablehlo.reduce(%v4098 init: %v4083) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4100 = stablehlo.reshape %v3966 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4102 = stablehlo.reduce(%v4100 init: %v4101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4103 = stablehlo.reshape %v504 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4104 = stablehlo.reshape %v3952 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4105 = stablehlo.transpose %v4103, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4106 = stablehlo.transpose %v4104, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4107 = stablehlo.convolution(%v4105, %v4106)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<192x32x1x1xf32>
    %v4108 = stablehlo.transpose %v4107, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4109 = stablehlo.reshape %v509 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4110 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4111 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v4112 = stablehlo.reduce(%v4109 init: %v4110) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4113 = stablehlo.broadcast_in_dim %v4112, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4114 = stablehlo.divide %v4113, %v4111 : tensor<32x32x28x28xf32>
    %v4115 = stablehlo.subtract %v4109, %v4114 : tensor<32x32x28x28xf32>
    %v4116 = stablehlo.multiply %v4115, %v4115 : tensor<32x32x28x28xf32>
    %v4117 = stablehlo.reduce(%v4116 init: %v4110) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4118 = stablehlo.broadcast_in_dim %v4117, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4119 = stablehlo.divide %v4118, %v4111 : tensor<32x32x28x28xf32>
    %v4120 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4121 = stablehlo.add %v4119, %v4120 : tensor<32x32x28x28xf32>
    %v4122 = stablehlo.rsqrt %v4121 : tensor<32x32x28x28xf32>
    %v4123 = stablehlo.multiply %v4115, %v4122 : tensor<32x32x28x28xf32>
    %v4124 = stablehlo.reshape %v3839 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4125 = stablehlo.multiply %v4124, %v4123 : tensor<32x32x28x28xf32>
    %v4126 = stablehlo.reduce(%v4125 init: %v4110) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4127 = stablehlo.reshape %v3839 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4129 = stablehlo.reduce(%v4127 init: %v4128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4130 = stablehlo.reshape %v418 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4132 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v4133 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4134 = stablehlo.reduce(%v4130 init: %v4131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4135 = stablehlo.broadcast_in_dim %v4134, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4136 = stablehlo.divide %v4135, %v4132 : tensor<32x32x28x28xf32>
    %v4137 = stablehlo.subtract %v4130, %v4136 : tensor<32x32x28x28xf32>
    %v4138 = stablehlo.multiply %v4137, %v4137 : tensor<32x32x28x28xf32>
    %v4139 = stablehlo.reduce(%v4138 init: %v4131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4140 = stablehlo.broadcast_in_dim %v4139, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4141 = stablehlo.divide %v4140, %v4132 : tensor<32x32x28x28xf32>
    %v4142 = stablehlo.add %v4141, %v4133 : tensor<32x32x28x28xf32>
    %v4143 = stablehlo.rsqrt %v4142 : tensor<32x32x28x28xf32>
    %v4144 = stablehlo.multiply %v4137, %v4143 : tensor<32x32x28x28xf32>
    %v4145 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4146 = stablehlo.reshape %v4048 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4147 = stablehlo.multiply %v4145, %v4146 : tensor<32x32x28x28xf32>
    %v4148 = stablehlo.reduce(%v4147 init: %v4131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4149 = stablehlo.broadcast_in_dim %v4148, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4150 = stablehlo.multiply %v4144, %v4147 : tensor<32x32x28x28xf32>
    %v4151 = stablehlo.reduce(%v4150 init: %v4131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4152 = stablehlo.broadcast_in_dim %v4151, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4153 = stablehlo.multiply %v4147, %v4132 : tensor<32x32x28x28xf32>
    %v4154 = stablehlo.subtract %v4153, %v4149 : tensor<32x32x28x28xf32>
    %v4155 = stablehlo.multiply %v4144, %v4152 : tensor<32x32x28x28xf32>
    %v4156 = stablehlo.subtract %v4154, %v4155 : tensor<32x32x28x28xf32>
    %v4157 = stablehlo.divide %v4143, %v4132 : tensor<32x32x28x28xf32>
    %v4158 = stablehlo.multiply %v4157, %v4156 : tensor<32x32x28x28xf32>
    %v4159 = stablehlo.reshape %v4158 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4160 = stablehlo.reshape %v4159 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4161 = stablehlo.reverse %b5pW, dims = [2, 3] : tensor<32x192x1x1xf32>
    %v4162 = stablehlo.transpose %v4161, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4163 = stablehlo.convolution(%v4160, %v4162)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v4164 = stablehlo.reshape %v4163 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4165 = stablehlo.reshape %v4164 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4166 = stablehlo.reshape %v407 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4167 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v4168 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v4169 = stablehlo.compare GT, %v4166, %v4167 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4170 = stablehlo.compare LT, %v4166, %v4168 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4171 = stablehlo.and %v4169, %v4170 : tensor<32x192x28x28xi1>
    %v4172 = stablehlo.select %v4171, %v4165, %v4167 : tensor<32x192x28x28xi1>, tensor<32x192x28x28xf32>
    %v4173 = stablehlo.reshape %v4172 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4174 = stablehlo.reshape %v387 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4175 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4176 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4177 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4178 = stablehlo.reduce(%v4174 init: %v4175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4179 = stablehlo.broadcast_in_dim %v4178, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4180 = stablehlo.divide %v4179, %v4176 : tensor<32x192x28x28xf32>
    %v4181 = stablehlo.subtract %v4174, %v4180 : tensor<32x192x28x28xf32>
    %v4182 = stablehlo.multiply %v4181, %v4181 : tensor<32x192x28x28xf32>
    %v4183 = stablehlo.reduce(%v4182 init: %v4175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4184 = stablehlo.broadcast_in_dim %v4183, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4185 = stablehlo.divide %v4184, %v4176 : tensor<32x192x28x28xf32>
    %v4186 = stablehlo.add %v4185, %v4177 : tensor<32x192x28x28xf32>
    %v4187 = stablehlo.rsqrt %v4186 : tensor<32x192x28x28xf32>
    %v4188 = stablehlo.multiply %v4181, %v4187 : tensor<32x192x28x28xf32>
    %v4189 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4190 = stablehlo.reshape %v4173 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4191 = stablehlo.multiply %v4189, %v4190 : tensor<32x192x28x28xf32>
    %v4192 = stablehlo.reduce(%v4191 init: %v4175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4193 = stablehlo.broadcast_in_dim %v4192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4194 = stablehlo.multiply %v4188, %v4191 : tensor<32x192x28x28xf32>
    %v4195 = stablehlo.reduce(%v4194 init: %v4175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4196 = stablehlo.broadcast_in_dim %v4195, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4197 = stablehlo.multiply %v4191, %v4176 : tensor<32x192x28x28xf32>
    %v4198 = stablehlo.subtract %v4197, %v4193 : tensor<32x192x28x28xf32>
    %v4199 = stablehlo.multiply %v4188, %v4196 : tensor<32x192x28x28xf32>
    %v4200 = stablehlo.subtract %v4198, %v4199 : tensor<32x192x28x28xf32>
    %v4201 = stablehlo.divide %v4187, %v4176 : tensor<32x192x28x28xf32>
    %v4202 = stablehlo.multiply %v4201, %v4200 : tensor<32x192x28x28xf32>
    %v4203 = stablehlo.reshape %v4202 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4204 = stablehlo.reshape %v4203 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4205 = stablehlo.reverse %b5dW, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v4206 = stablehlo.convolution(%v4204, %v4205)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v4207 = stablehlo.reshape %v4206 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4208 = stablehlo.reshape %v4207 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4209 = stablehlo.reshape %v376 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4210 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v4211 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v4212 = stablehlo.compare GT, %v4209, %v4210 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4213 = stablehlo.compare LT, %v4209, %v4211 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4214 = stablehlo.and %v4212, %v4213 : tensor<32x192x28x28xi1>
    %v4215 = stablehlo.select %v4214, %v4208, %v4210 : tensor<32x192x28x28xi1>, tensor<32x192x28x28xf32>
    %v4216 = stablehlo.reshape %v4215 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4217 = stablehlo.reshape %v356 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4219 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4220 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4221 = stablehlo.reduce(%v4217 init: %v4218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4222 = stablehlo.broadcast_in_dim %v4221, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4223 = stablehlo.divide %v4222, %v4219 : tensor<32x192x28x28xf32>
    %v4224 = stablehlo.subtract %v4217, %v4223 : tensor<32x192x28x28xf32>
    %v4225 = stablehlo.multiply %v4224, %v4224 : tensor<32x192x28x28xf32>
    %v4226 = stablehlo.reduce(%v4225 init: %v4218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4227 = stablehlo.broadcast_in_dim %v4226, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4228 = stablehlo.divide %v4227, %v4219 : tensor<32x192x28x28xf32>
    %v4229 = stablehlo.add %v4228, %v4220 : tensor<32x192x28x28xf32>
    %v4230 = stablehlo.rsqrt %v4229 : tensor<32x192x28x28xf32>
    %v4231 = stablehlo.multiply %v4224, %v4230 : tensor<32x192x28x28xf32>
    %v4232 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4233 = stablehlo.reshape %v4216 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4234 = stablehlo.multiply %v4232, %v4233 : tensor<32x192x28x28xf32>
    %v4235 = stablehlo.reduce(%v4234 init: %v4218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4236 = stablehlo.broadcast_in_dim %v4235, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4237 = stablehlo.multiply %v4231, %v4234 : tensor<32x192x28x28xf32>
    %v4238 = stablehlo.reduce(%v4237 init: %v4218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4239 = stablehlo.broadcast_in_dim %v4238, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4240 = stablehlo.multiply %v4234, %v4219 : tensor<32x192x28x28xf32>
    %v4241 = stablehlo.subtract %v4240, %v4236 : tensor<32x192x28x28xf32>
    %v4242 = stablehlo.multiply %v4231, %v4239 : tensor<32x192x28x28xf32>
    %v4243 = stablehlo.subtract %v4241, %v4242 : tensor<32x192x28x28xf32>
    %v4244 = stablehlo.divide %v4230, %v4219 : tensor<32x192x28x28xf32>
    %v4245 = stablehlo.multiply %v4244, %v4243 : tensor<32x192x28x28xf32>
    %v4246 = stablehlo.reshape %v4245 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4247 = stablehlo.reshape %v4246 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4248 = stablehlo.reverse %b5eW, dims = [2, 3] : tensor<192x32x1x1xf32>
    %v4249 = stablehlo.transpose %v4248, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4250 = stablehlo.convolution(%v4247, %v4249)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v4251 = stablehlo.reshape %v4250 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4252 = stablehlo.reshape %v4251 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4253 = stablehlo.reshape %v4048 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4254 = stablehlo.add %v4252, %v4253 : tensor<32x32x28x28xf32>
    %v4255 = stablehlo.reshape %v4254 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4256 = stablehlo.reshape %v351 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4257 = stablehlo.reshape %v4246 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4258 = stablehlo.transpose %v4256, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4259 = stablehlo.transpose %v4257, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4260 = stablehlo.convolution(%v4258, %v4259)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v4261 = stablehlo.transpose %v4260, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4262 = stablehlo.reshape %v356 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4263 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4264 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4265 = stablehlo.reduce(%v4262 init: %v4263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4266 = stablehlo.broadcast_in_dim %v4265, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4267 = stablehlo.divide %v4266, %v4264 : tensor<32x192x28x28xf32>
    %v4268 = stablehlo.subtract %v4262, %v4267 : tensor<32x192x28x28xf32>
    %v4269 = stablehlo.multiply %v4268, %v4268 : tensor<32x192x28x28xf32>
    %v4270 = stablehlo.reduce(%v4269 init: %v4263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4271 = stablehlo.broadcast_in_dim %v4270, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4272 = stablehlo.divide %v4271, %v4264 : tensor<32x192x28x28xf32>
    %v4273 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4274 = stablehlo.add %v4272, %v4273 : tensor<32x192x28x28xf32>
    %v4275 = stablehlo.rsqrt %v4274 : tensor<32x192x28x28xf32>
    %v4276 = stablehlo.multiply %v4268, %v4275 : tensor<32x192x28x28xf32>
    %v4277 = stablehlo.reshape %v4216 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4278 = stablehlo.multiply %v4277, %v4276 : tensor<32x192x28x28xf32>
    %v4279 = stablehlo.reduce(%v4278 init: %v4263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4280 = stablehlo.reshape %v4216 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4281 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4282 = stablehlo.reduce(%v4280 init: %v4281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4283 = stablehlo.reshape %v382 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4284 = stablehlo.reshape %v4203 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4285 = stablehlo.transpose %v4283, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4286 = stablehlo.transpose %v4284, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4287 = stablehlo.convolution(%v4285, %v4286)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v4288 = stablehlo.reshape %v4287 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v4289 = stablehlo.reshape %v387 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4290 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4291 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v4292 = stablehlo.reduce(%v4289 init: %v4290) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4293 = stablehlo.broadcast_in_dim %v4292, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4294 = stablehlo.divide %v4293, %v4291 : tensor<32x192x28x28xf32>
    %v4295 = stablehlo.subtract %v4289, %v4294 : tensor<32x192x28x28xf32>
    %v4296 = stablehlo.multiply %v4295, %v4295 : tensor<32x192x28x28xf32>
    %v4297 = stablehlo.reduce(%v4296 init: %v4290) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4298 = stablehlo.broadcast_in_dim %v4297, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4299 = stablehlo.divide %v4298, %v4291 : tensor<32x192x28x28xf32>
    %v4300 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4301 = stablehlo.add %v4299, %v4300 : tensor<32x192x28x28xf32>
    %v4302 = stablehlo.rsqrt %v4301 : tensor<32x192x28x28xf32>
    %v4303 = stablehlo.multiply %v4295, %v4302 : tensor<32x192x28x28xf32>
    %v4304 = stablehlo.reshape %v4173 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4305 = stablehlo.multiply %v4304, %v4303 : tensor<32x192x28x28xf32>
    %v4306 = stablehlo.reduce(%v4305 init: %v4290) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4307 = stablehlo.reshape %v4173 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4309 = stablehlo.reduce(%v4307 init: %v4308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4310 = stablehlo.reshape %v413 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4311 = stablehlo.reshape %v4159 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4312 = stablehlo.transpose %v4310, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4313 = stablehlo.transpose %v4311, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4314 = stablehlo.convolution(%v4312, %v4313)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<192x32x1x1xf32>
    %v4315 = stablehlo.transpose %v4314, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4316 = stablehlo.reshape %v418 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4317 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4318 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v4319 = stablehlo.reduce(%v4316 init: %v4317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4320 = stablehlo.broadcast_in_dim %v4319, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4321 = stablehlo.divide %v4320, %v4318 : tensor<32x32x28x28xf32>
    %v4322 = stablehlo.subtract %v4316, %v4321 : tensor<32x32x28x28xf32>
    %v4323 = stablehlo.multiply %v4322, %v4322 : tensor<32x32x28x28xf32>
    %v4324 = stablehlo.reduce(%v4323 init: %v4317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4325 = stablehlo.broadcast_in_dim %v4324, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4326 = stablehlo.divide %v4325, %v4318 : tensor<32x32x28x28xf32>
    %v4327 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4328 = stablehlo.add %v4326, %v4327 : tensor<32x32x28x28xf32>
    %v4329 = stablehlo.rsqrt %v4328 : tensor<32x32x28x28xf32>
    %v4330 = stablehlo.multiply %v4322, %v4329 : tensor<32x32x28x28xf32>
    %v4331 = stablehlo.reshape %v4048 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4332 = stablehlo.multiply %v4331, %v4330 : tensor<32x32x28x28xf32>
    %v4333 = stablehlo.reduce(%v4332 init: %v4317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4334 = stablehlo.reshape %v4048 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4336 = stablehlo.reduce(%v4334 init: %v4335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4337 = stablehlo.reshape %v331 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4338 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4339 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v4340 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4341 = stablehlo.reduce(%v4337 init: %v4338) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4342 = stablehlo.broadcast_in_dim %v4341, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4343 = stablehlo.divide %v4342, %v4339 : tensor<32x32x28x28xf32>
    %v4344 = stablehlo.subtract %v4337, %v4343 : tensor<32x32x28x28xf32>
    %v4345 = stablehlo.multiply %v4344, %v4344 : tensor<32x32x28x28xf32>
    %v4346 = stablehlo.reduce(%v4345 init: %v4338) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4347 = stablehlo.broadcast_in_dim %v4346, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4348 = stablehlo.divide %v4347, %v4339 : tensor<32x32x28x28xf32>
    %v4349 = stablehlo.add %v4348, %v4340 : tensor<32x32x28x28xf32>
    %v4350 = stablehlo.rsqrt %v4349 : tensor<32x32x28x28xf32>
    %v4351 = stablehlo.multiply %v4344, %v4350 : tensor<32x32x28x28xf32>
    %v4352 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4353 = stablehlo.reshape %v4255 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4354 = stablehlo.multiply %v4352, %v4353 : tensor<32x32x28x28xf32>
    %v4355 = stablehlo.reduce(%v4354 init: %v4338) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4356 = stablehlo.broadcast_in_dim %v4355, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4357 = stablehlo.multiply %v4351, %v4354 : tensor<32x32x28x28xf32>
    %v4358 = stablehlo.reduce(%v4357 init: %v4338) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4359 = stablehlo.broadcast_in_dim %v4358, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4360 = stablehlo.multiply %v4354, %v4339 : tensor<32x32x28x28xf32>
    %v4361 = stablehlo.subtract %v4360, %v4356 : tensor<32x32x28x28xf32>
    %v4362 = stablehlo.multiply %v4351, %v4359 : tensor<32x32x28x28xf32>
    %v4363 = stablehlo.subtract %v4361, %v4362 : tensor<32x32x28x28xf32>
    %v4364 = stablehlo.divide %v4350, %v4339 : tensor<32x32x28x28xf32>
    %v4365 = stablehlo.multiply %v4364, %v4363 : tensor<32x32x28x28xf32>
    %v4366 = stablehlo.reshape %v4365 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4367 = stablehlo.reshape %v4366 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4368 = stablehlo.reverse %b4pW, dims = [2, 3] : tensor<32x144x1x1xf32>
    %v4369 = stablehlo.transpose %v4368, dims = [1, 0, 2, 3] : (tensor<32x144x1x1xf32>) -> tensor<144x32x1x1xf32>
    %v4370 = stablehlo.convolution(%v4367, %v4369)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<144x32x1x1xf32>) -> tensor<32x144x28x28xf32>
    %v4371 = stablehlo.reshape %v4370 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v4372 = stablehlo.reshape %v4371 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4373 = stablehlo.reshape %v320 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4374 = stablehlo.constant dense<0.0> : tensor<32x144x28x28xf32>
    %v4375 = stablehlo.constant dense<6.0> : tensor<32x144x28x28xf32>
    %v4376 = stablehlo.compare GT, %v4373, %v4374 : (tensor<32x144x28x28xf32>, tensor<32x144x28x28xf32>) -> tensor<32x144x28x28xi1>
    %v4377 = stablehlo.compare LT, %v4373, %v4375 : (tensor<32x144x28x28xf32>, tensor<32x144x28x28xf32>) -> tensor<32x144x28x28xi1>
    %v4378 = stablehlo.and %v4376, %v4377 : tensor<32x144x28x28xi1>
    %v4379 = stablehlo.select %v4378, %v4372, %v4374 : tensor<32x144x28x28xi1>, tensor<32x144x28x28xf32>
    %v4380 = stablehlo.reshape %v4379 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v4381 = stablehlo.reshape %v300 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4383 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v4384 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v4385 = stablehlo.reduce(%v4381 init: %v4382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4386 = stablehlo.broadcast_in_dim %v4385, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4387 = stablehlo.divide %v4386, %v4383 : tensor<32x144x28x28xf32>
    %v4388 = stablehlo.subtract %v4381, %v4387 : tensor<32x144x28x28xf32>
    %v4389 = stablehlo.multiply %v4388, %v4388 : tensor<32x144x28x28xf32>
    %v4390 = stablehlo.reduce(%v4389 init: %v4382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4391 = stablehlo.broadcast_in_dim %v4390, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4392 = stablehlo.divide %v4391, %v4383 : tensor<32x144x28x28xf32>
    %v4393 = stablehlo.add %v4392, %v4384 : tensor<32x144x28x28xf32>
    %v4394 = stablehlo.rsqrt %v4393 : tensor<32x144x28x28xf32>
    %v4395 = stablehlo.multiply %v4388, %v4394 : tensor<32x144x28x28xf32>
    %v4396 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4397 = stablehlo.reshape %v4380 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4398 = stablehlo.multiply %v4396, %v4397 : tensor<32x144x28x28xf32>
    %v4399 = stablehlo.reduce(%v4398 init: %v4382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4400 = stablehlo.broadcast_in_dim %v4399, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4401 = stablehlo.multiply %v4395, %v4398 : tensor<32x144x28x28xf32>
    %v4402 = stablehlo.reduce(%v4401 init: %v4382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4403 = stablehlo.broadcast_in_dim %v4402, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4404 = stablehlo.multiply %v4398, %v4383 : tensor<32x144x28x28xf32>
    %v4405 = stablehlo.subtract %v4404, %v4400 : tensor<32x144x28x28xf32>
    %v4406 = stablehlo.multiply %v4395, %v4403 : tensor<32x144x28x28xf32>
    %v4407 = stablehlo.subtract %v4405, %v4406 : tensor<32x144x28x28xf32>
    %v4408 = stablehlo.divide %v4394, %v4383 : tensor<32x144x28x28xf32>
    %v4409 = stablehlo.multiply %v4408, %v4407 : tensor<32x144x28x28xf32>
    %v4410 = stablehlo.reshape %v4409 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v4411 = stablehlo.reshape %v4410 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4412 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4413 = stablehlo.pad %v4411, %v4412, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v4414 = stablehlo.reverse %b4dW, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v4415 = stablehlo.convolution(%v4413, %v4414)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 0], [2, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v4416 = stablehlo.reshape %v4415 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4417 = stablehlo.reshape %v4416 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4418 = stablehlo.reshape %v289 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4419 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v4420 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v4421 = stablehlo.compare GT, %v4418, %v4419 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4422 = stablehlo.compare LT, %v4418, %v4420 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4423 = stablehlo.and %v4421, %v4422 : tensor<32x144x56x56xi1>
    %v4424 = stablehlo.select %v4423, %v4417, %v4419 : tensor<32x144x56x56xi1>, tensor<32x144x56x56xf32>
    %v4425 = stablehlo.reshape %v4424 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4426 = stablehlo.reshape %v269 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4427 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4428 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4429 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4430 = stablehlo.reduce(%v4426 init: %v4427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4431 = stablehlo.broadcast_in_dim %v4430, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4432 = stablehlo.divide %v4431, %v4428 : tensor<32x144x56x56xf32>
    %v4433 = stablehlo.subtract %v4426, %v4432 : tensor<32x144x56x56xf32>
    %v4434 = stablehlo.multiply %v4433, %v4433 : tensor<32x144x56x56xf32>
    %v4435 = stablehlo.reduce(%v4434 init: %v4427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4436 = stablehlo.broadcast_in_dim %v4435, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4437 = stablehlo.divide %v4436, %v4428 : tensor<32x144x56x56xf32>
    %v4438 = stablehlo.add %v4437, %v4429 : tensor<32x144x56x56xf32>
    %v4439 = stablehlo.rsqrt %v4438 : tensor<32x144x56x56xf32>
    %v4440 = stablehlo.multiply %v4433, %v4439 : tensor<32x144x56x56xf32>
    %v4441 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4442 = stablehlo.reshape %v4425 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4443 = stablehlo.multiply %v4441, %v4442 : tensor<32x144x56x56xf32>
    %v4444 = stablehlo.reduce(%v4443 init: %v4427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4445 = stablehlo.broadcast_in_dim %v4444, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4446 = stablehlo.multiply %v4440, %v4443 : tensor<32x144x56x56xf32>
    %v4447 = stablehlo.reduce(%v4446 init: %v4427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4448 = stablehlo.broadcast_in_dim %v4447, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4449 = stablehlo.multiply %v4443, %v4428 : tensor<32x144x56x56xf32>
    %v4450 = stablehlo.subtract %v4449, %v4445 : tensor<32x144x56x56xf32>
    %v4451 = stablehlo.multiply %v4440, %v4448 : tensor<32x144x56x56xf32>
    %v4452 = stablehlo.subtract %v4450, %v4451 : tensor<32x144x56x56xf32>
    %v4453 = stablehlo.divide %v4439, %v4428 : tensor<32x144x56x56xf32>
    %v4454 = stablehlo.multiply %v4453, %v4452 : tensor<32x144x56x56xf32>
    %v4455 = stablehlo.reshape %v4454 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4456 = stablehlo.reshape %v4455 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4457 = stablehlo.reverse %b4eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v4458 = stablehlo.transpose %v4457, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v4459 = stablehlo.convolution(%v4456, %v4458)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v4460 = stablehlo.reshape %v4459 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4461 = stablehlo.reshape %v264 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4462 = stablehlo.reshape %v4455 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4463 = stablehlo.transpose %v4461, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4464 = stablehlo.transpose %v4462, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4465 = stablehlo.convolution(%v4463, %v4464)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v4466 = stablehlo.transpose %v4465, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v4467 = stablehlo.reshape %v269 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4469 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4470 = stablehlo.reduce(%v4467 init: %v4468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4471 = stablehlo.broadcast_in_dim %v4470, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4472 = stablehlo.divide %v4471, %v4469 : tensor<32x144x56x56xf32>
    %v4473 = stablehlo.subtract %v4467, %v4472 : tensor<32x144x56x56xf32>
    %v4474 = stablehlo.multiply %v4473, %v4473 : tensor<32x144x56x56xf32>
    %v4475 = stablehlo.reduce(%v4474 init: %v4468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4476 = stablehlo.broadcast_in_dim %v4475, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4477 = stablehlo.divide %v4476, %v4469 : tensor<32x144x56x56xf32>
    %v4478 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4479 = stablehlo.add %v4477, %v4478 : tensor<32x144x56x56xf32>
    %v4480 = stablehlo.rsqrt %v4479 : tensor<32x144x56x56xf32>
    %v4481 = stablehlo.multiply %v4473, %v4480 : tensor<32x144x56x56xf32>
    %v4482 = stablehlo.reshape %v4425 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4483 = stablehlo.multiply %v4482, %v4481 : tensor<32x144x56x56xf32>
    %v4484 = stablehlo.reduce(%v4483 init: %v4468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4485 = stablehlo.reshape %v4425 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4486 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4487 = stablehlo.reduce(%v4485 init: %v4486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4488 = stablehlo.reshape %v295 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4489 = stablehlo.reshape %v4410 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4490 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4491 = stablehlo.pad %v4489, %v4490, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v4492 = stablehlo.transpose %v4488, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4493 = stablehlo.transpose %v4491, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4494 = stablehlo.convolution(%v4492, %v4493)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 2], [0, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v4495 = stablehlo.reshape %v4494 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v4496 = stablehlo.reshape %v300 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4497 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4498 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v4499 = stablehlo.reduce(%v4496 init: %v4497) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4500 = stablehlo.broadcast_in_dim %v4499, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4501 = stablehlo.divide %v4500, %v4498 : tensor<32x144x28x28xf32>
    %v4502 = stablehlo.subtract %v4496, %v4501 : tensor<32x144x28x28xf32>
    %v4503 = stablehlo.multiply %v4502, %v4502 : tensor<32x144x28x28xf32>
    %v4504 = stablehlo.reduce(%v4503 init: %v4497) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4505 = stablehlo.broadcast_in_dim %v4504, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4506 = stablehlo.divide %v4505, %v4498 : tensor<32x144x28x28xf32>
    %v4507 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v4508 = stablehlo.add %v4506, %v4507 : tensor<32x144x28x28xf32>
    %v4509 = stablehlo.rsqrt %v4508 : tensor<32x144x28x28xf32>
    %v4510 = stablehlo.multiply %v4502, %v4509 : tensor<32x144x28x28xf32>
    %v4511 = stablehlo.reshape %v4380 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4512 = stablehlo.multiply %v4511, %v4510 : tensor<32x144x28x28xf32>
    %v4513 = stablehlo.reduce(%v4512 init: %v4497) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4514 = stablehlo.reshape %v4380 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4516 = stablehlo.reduce(%v4514 init: %v4515) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4517 = stablehlo.reshape %v326 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4518 = stablehlo.reshape %v4366 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4519 = stablehlo.transpose %v4517, dims = [1, 0, 2, 3] : (tensor<32x144x28x28xf32>) -> tensor<144x32x28x28xf32>
    %v4520 = stablehlo.transpose %v4518, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4521 = stablehlo.convolution(%v4519, %v4520)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<144x32x1x1xf32>
    %v4522 = stablehlo.transpose %v4521, dims = [1, 0, 2, 3] : (tensor<144x32x1x1xf32>) -> tensor<32x144x1x1xf32>
    %v4523 = stablehlo.reshape %v331 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4524 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4525 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v4526 = stablehlo.reduce(%v4523 init: %v4524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4527 = stablehlo.broadcast_in_dim %v4526, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4528 = stablehlo.divide %v4527, %v4525 : tensor<32x32x28x28xf32>
    %v4529 = stablehlo.subtract %v4523, %v4528 : tensor<32x32x28x28xf32>
    %v4530 = stablehlo.multiply %v4529, %v4529 : tensor<32x32x28x28xf32>
    %v4531 = stablehlo.reduce(%v4530 init: %v4524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4532 = stablehlo.broadcast_in_dim %v4531, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4533 = stablehlo.divide %v4532, %v4525 : tensor<32x32x28x28xf32>
    %v4534 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4535 = stablehlo.add %v4533, %v4534 : tensor<32x32x28x28xf32>
    %v4536 = stablehlo.rsqrt %v4535 : tensor<32x32x28x28xf32>
    %v4537 = stablehlo.multiply %v4529, %v4536 : tensor<32x32x28x28xf32>
    %v4538 = stablehlo.reshape %v4255 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4539 = stablehlo.multiply %v4538, %v4537 : tensor<32x32x28x28xf32>
    %v4540 = stablehlo.reduce(%v4539 init: %v4524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4541 = stablehlo.reshape %v4255 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4543 = stablehlo.reduce(%v4541 init: %v4542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4544 = stablehlo.reshape %v240 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4546 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v4547 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4548 = stablehlo.reduce(%v4544 init: %v4545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4549 = stablehlo.broadcast_in_dim %v4548, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4550 = stablehlo.divide %v4549, %v4546 : tensor<32x24x56x56xf32>
    %v4551 = stablehlo.subtract %v4544, %v4550 : tensor<32x24x56x56xf32>
    %v4552 = stablehlo.multiply %v4551, %v4551 : tensor<32x24x56x56xf32>
    %v4553 = stablehlo.reduce(%v4552 init: %v4545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4554 = stablehlo.broadcast_in_dim %v4553, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4555 = stablehlo.divide %v4554, %v4546 : tensor<32x24x56x56xf32>
    %v4556 = stablehlo.add %v4555, %v4547 : tensor<32x24x56x56xf32>
    %v4557 = stablehlo.rsqrt %v4556 : tensor<32x24x56x56xf32>
    %v4558 = stablehlo.multiply %v4551, %v4557 : tensor<32x24x56x56xf32>
    %v4559 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4560 = stablehlo.reshape %v4460 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4561 = stablehlo.multiply %v4559, %v4560 : tensor<32x24x56x56xf32>
    %v4562 = stablehlo.reduce(%v4561 init: %v4545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4563 = stablehlo.broadcast_in_dim %v4562, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4564 = stablehlo.multiply %v4558, %v4561 : tensor<32x24x56x56xf32>
    %v4565 = stablehlo.reduce(%v4564 init: %v4545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4566 = stablehlo.broadcast_in_dim %v4565, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4567 = stablehlo.multiply %v4561, %v4546 : tensor<32x24x56x56xf32>
    %v4568 = stablehlo.subtract %v4567, %v4563 : tensor<32x24x56x56xf32>
    %v4569 = stablehlo.multiply %v4558, %v4566 : tensor<32x24x56x56xf32>
    %v4570 = stablehlo.subtract %v4568, %v4569 : tensor<32x24x56x56xf32>
    %v4571 = stablehlo.divide %v4557, %v4546 : tensor<32x24x56x56xf32>
    %v4572 = stablehlo.multiply %v4571, %v4570 : tensor<32x24x56x56xf32>
    %v4573 = stablehlo.reshape %v4572 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4574 = stablehlo.reshape %v4573 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4575 = stablehlo.reverse %b3pW, dims = [2, 3] : tensor<24x144x1x1xf32>
    %v4576 = stablehlo.transpose %v4575, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v4577 = stablehlo.convolution(%v4574, %v4576)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v4578 = stablehlo.reshape %v4577 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4579 = stablehlo.reshape %v4578 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4580 = stablehlo.reshape %v229 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4581 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v4582 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v4583 = stablehlo.compare GT, %v4580, %v4581 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4584 = stablehlo.compare LT, %v4580, %v4582 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4585 = stablehlo.and %v4583, %v4584 : tensor<32x144x56x56xi1>
    %v4586 = stablehlo.select %v4585, %v4579, %v4581 : tensor<32x144x56x56xi1>, tensor<32x144x56x56xf32>
    %v4587 = stablehlo.reshape %v4586 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4588 = stablehlo.reshape %v209 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4589 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4590 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4591 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4592 = stablehlo.reduce(%v4588 init: %v4589) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4593 = stablehlo.broadcast_in_dim %v4592, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4594 = stablehlo.divide %v4593, %v4590 : tensor<32x144x56x56xf32>
    %v4595 = stablehlo.subtract %v4588, %v4594 : tensor<32x144x56x56xf32>
    %v4596 = stablehlo.multiply %v4595, %v4595 : tensor<32x144x56x56xf32>
    %v4597 = stablehlo.reduce(%v4596 init: %v4589) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4598 = stablehlo.broadcast_in_dim %v4597, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4599 = stablehlo.divide %v4598, %v4590 : tensor<32x144x56x56xf32>
    %v4600 = stablehlo.add %v4599, %v4591 : tensor<32x144x56x56xf32>
    %v4601 = stablehlo.rsqrt %v4600 : tensor<32x144x56x56xf32>
    %v4602 = stablehlo.multiply %v4595, %v4601 : tensor<32x144x56x56xf32>
    %v4603 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4604 = stablehlo.reshape %v4587 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4605 = stablehlo.multiply %v4603, %v4604 : tensor<32x144x56x56xf32>
    %v4606 = stablehlo.reduce(%v4605 init: %v4589) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4607 = stablehlo.broadcast_in_dim %v4606, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4608 = stablehlo.multiply %v4602, %v4605 : tensor<32x144x56x56xf32>
    %v4609 = stablehlo.reduce(%v4608 init: %v4589) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4610 = stablehlo.broadcast_in_dim %v4609, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4611 = stablehlo.multiply %v4605, %v4590 : tensor<32x144x56x56xf32>
    %v4612 = stablehlo.subtract %v4611, %v4607 : tensor<32x144x56x56xf32>
    %v4613 = stablehlo.multiply %v4602, %v4610 : tensor<32x144x56x56xf32>
    %v4614 = stablehlo.subtract %v4612, %v4613 : tensor<32x144x56x56xf32>
    %v4615 = stablehlo.divide %v4601, %v4590 : tensor<32x144x56x56xf32>
    %v4616 = stablehlo.multiply %v4615, %v4614 : tensor<32x144x56x56xf32>
    %v4617 = stablehlo.reshape %v4616 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4618 = stablehlo.reshape %v4617 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4619 = stablehlo.reverse %b3dW, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v4620 = stablehlo.convolution(%v4618, %v4619)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v4621 = stablehlo.reshape %v4620 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4622 = stablehlo.reshape %v4621 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4623 = stablehlo.reshape %v198 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4624 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v4625 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v4626 = stablehlo.compare GT, %v4623, %v4624 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4627 = stablehlo.compare LT, %v4623, %v4625 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4628 = stablehlo.and %v4626, %v4627 : tensor<32x144x56x56xi1>
    %v4629 = stablehlo.select %v4628, %v4622, %v4624 : tensor<32x144x56x56xi1>, tensor<32x144x56x56xf32>
    %v4630 = stablehlo.reshape %v4629 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4631 = stablehlo.reshape %v178 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4632 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4633 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4634 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4635 = stablehlo.reduce(%v4631 init: %v4632) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4636 = stablehlo.broadcast_in_dim %v4635, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4637 = stablehlo.divide %v4636, %v4633 : tensor<32x144x56x56xf32>
    %v4638 = stablehlo.subtract %v4631, %v4637 : tensor<32x144x56x56xf32>
    %v4639 = stablehlo.multiply %v4638, %v4638 : tensor<32x144x56x56xf32>
    %v4640 = stablehlo.reduce(%v4639 init: %v4632) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4641 = stablehlo.broadcast_in_dim %v4640, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4642 = stablehlo.divide %v4641, %v4633 : tensor<32x144x56x56xf32>
    %v4643 = stablehlo.add %v4642, %v4634 : tensor<32x144x56x56xf32>
    %v4644 = stablehlo.rsqrt %v4643 : tensor<32x144x56x56xf32>
    %v4645 = stablehlo.multiply %v4638, %v4644 : tensor<32x144x56x56xf32>
    %v4646 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4647 = stablehlo.reshape %v4630 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4648 = stablehlo.multiply %v4646, %v4647 : tensor<32x144x56x56xf32>
    %v4649 = stablehlo.reduce(%v4648 init: %v4632) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4650 = stablehlo.broadcast_in_dim %v4649, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4651 = stablehlo.multiply %v4645, %v4648 : tensor<32x144x56x56xf32>
    %v4652 = stablehlo.reduce(%v4651 init: %v4632) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4653 = stablehlo.broadcast_in_dim %v4652, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4654 = stablehlo.multiply %v4648, %v4633 : tensor<32x144x56x56xf32>
    %v4655 = stablehlo.subtract %v4654, %v4650 : tensor<32x144x56x56xf32>
    %v4656 = stablehlo.multiply %v4645, %v4653 : tensor<32x144x56x56xf32>
    %v4657 = stablehlo.subtract %v4655, %v4656 : tensor<32x144x56x56xf32>
    %v4658 = stablehlo.divide %v4644, %v4633 : tensor<32x144x56x56xf32>
    %v4659 = stablehlo.multiply %v4658, %v4657 : tensor<32x144x56x56xf32>
    %v4660 = stablehlo.reshape %v4659 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4661 = stablehlo.reshape %v4660 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4662 = stablehlo.reverse %b3eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v4663 = stablehlo.transpose %v4662, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v4664 = stablehlo.convolution(%v4661, %v4663)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v4665 = stablehlo.reshape %v4664 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4666 = stablehlo.reshape %v4665 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4667 = stablehlo.reshape %v4460 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4668 = stablehlo.add %v4666, %v4667 : tensor<32x24x56x56xf32>
    %v4669 = stablehlo.reshape %v4668 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4670 = stablehlo.reshape %v173 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4671 = stablehlo.reshape %v4660 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4672 = stablehlo.transpose %v4670, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4673 = stablehlo.transpose %v4671, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4674 = stablehlo.convolution(%v4672, %v4673)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v4675 = stablehlo.transpose %v4674, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v4676 = stablehlo.reshape %v178 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4677 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4678 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4679 = stablehlo.reduce(%v4676 init: %v4677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4680 = stablehlo.broadcast_in_dim %v4679, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4681 = stablehlo.divide %v4680, %v4678 : tensor<32x144x56x56xf32>
    %v4682 = stablehlo.subtract %v4676, %v4681 : tensor<32x144x56x56xf32>
    %v4683 = stablehlo.multiply %v4682, %v4682 : tensor<32x144x56x56xf32>
    %v4684 = stablehlo.reduce(%v4683 init: %v4677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4685 = stablehlo.broadcast_in_dim %v4684, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4686 = stablehlo.divide %v4685, %v4678 : tensor<32x144x56x56xf32>
    %v4687 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4688 = stablehlo.add %v4686, %v4687 : tensor<32x144x56x56xf32>
    %v4689 = stablehlo.rsqrt %v4688 : tensor<32x144x56x56xf32>
    %v4690 = stablehlo.multiply %v4682, %v4689 : tensor<32x144x56x56xf32>
    %v4691 = stablehlo.reshape %v4630 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4692 = stablehlo.multiply %v4691, %v4690 : tensor<32x144x56x56xf32>
    %v4693 = stablehlo.reduce(%v4692 init: %v4677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4694 = stablehlo.reshape %v4630 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4695 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4696 = stablehlo.reduce(%v4694 init: %v4695) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4697 = stablehlo.reshape %v204 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4698 = stablehlo.reshape %v4617 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4699 = stablehlo.transpose %v4697, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4700 = stablehlo.transpose %v4698, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4701 = stablehlo.convolution(%v4699, %v4700)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v4702 = stablehlo.reshape %v4701 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v4703 = stablehlo.reshape %v209 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4704 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4705 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v4706 = stablehlo.reduce(%v4703 init: %v4704) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4707 = stablehlo.broadcast_in_dim %v4706, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4708 = stablehlo.divide %v4707, %v4705 : tensor<32x144x56x56xf32>
    %v4709 = stablehlo.subtract %v4703, %v4708 : tensor<32x144x56x56xf32>
    %v4710 = stablehlo.multiply %v4709, %v4709 : tensor<32x144x56x56xf32>
    %v4711 = stablehlo.reduce(%v4710 init: %v4704) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4712 = stablehlo.broadcast_in_dim %v4711, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4713 = stablehlo.divide %v4712, %v4705 : tensor<32x144x56x56xf32>
    %v4714 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4715 = stablehlo.add %v4713, %v4714 : tensor<32x144x56x56xf32>
    %v4716 = stablehlo.rsqrt %v4715 : tensor<32x144x56x56xf32>
    %v4717 = stablehlo.multiply %v4709, %v4716 : tensor<32x144x56x56xf32>
    %v4718 = stablehlo.reshape %v4587 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4719 = stablehlo.multiply %v4718, %v4717 : tensor<32x144x56x56xf32>
    %v4720 = stablehlo.reduce(%v4719 init: %v4704) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4721 = stablehlo.reshape %v4587 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4722 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4723 = stablehlo.reduce(%v4721 init: %v4722) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4724 = stablehlo.reshape %v235 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4725 = stablehlo.reshape %v4573 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4726 = stablehlo.transpose %v4724, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4727 = stablehlo.transpose %v4725, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4728 = stablehlo.convolution(%v4726, %v4727)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<144x24x1x1xf32>
    %v4729 = stablehlo.transpose %v4728, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v4730 = stablehlo.reshape %v240 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4731 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4732 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v4733 = stablehlo.reduce(%v4730 init: %v4731) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4734 = stablehlo.broadcast_in_dim %v4733, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4735 = stablehlo.divide %v4734, %v4732 : tensor<32x24x56x56xf32>
    %v4736 = stablehlo.subtract %v4730, %v4735 : tensor<32x24x56x56xf32>
    %v4737 = stablehlo.multiply %v4736, %v4736 : tensor<32x24x56x56xf32>
    %v4738 = stablehlo.reduce(%v4737 init: %v4731) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4739 = stablehlo.broadcast_in_dim %v4738, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4740 = stablehlo.divide %v4739, %v4732 : tensor<32x24x56x56xf32>
    %v4741 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4742 = stablehlo.add %v4740, %v4741 : tensor<32x24x56x56xf32>
    %v4743 = stablehlo.rsqrt %v4742 : tensor<32x24x56x56xf32>
    %v4744 = stablehlo.multiply %v4736, %v4743 : tensor<32x24x56x56xf32>
    %v4745 = stablehlo.reshape %v4460 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4746 = stablehlo.multiply %v4745, %v4744 : tensor<32x24x56x56xf32>
    %v4747 = stablehlo.reduce(%v4746 init: %v4731) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4748 = stablehlo.reshape %v4460 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4750 = stablehlo.reduce(%v4748 init: %v4749) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4751 = stablehlo.reshape %v153 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4752 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4753 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v4754 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4755 = stablehlo.reduce(%v4751 init: %v4752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4756 = stablehlo.broadcast_in_dim %v4755, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4757 = stablehlo.divide %v4756, %v4753 : tensor<32x24x56x56xf32>
    %v4758 = stablehlo.subtract %v4751, %v4757 : tensor<32x24x56x56xf32>
    %v4759 = stablehlo.multiply %v4758, %v4758 : tensor<32x24x56x56xf32>
    %v4760 = stablehlo.reduce(%v4759 init: %v4752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4761 = stablehlo.broadcast_in_dim %v4760, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4762 = stablehlo.divide %v4761, %v4753 : tensor<32x24x56x56xf32>
    %v4763 = stablehlo.add %v4762, %v4754 : tensor<32x24x56x56xf32>
    %v4764 = stablehlo.rsqrt %v4763 : tensor<32x24x56x56xf32>
    %v4765 = stablehlo.multiply %v4758, %v4764 : tensor<32x24x56x56xf32>
    %v4766 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4767 = stablehlo.reshape %v4669 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4768 = stablehlo.multiply %v4766, %v4767 : tensor<32x24x56x56xf32>
    %v4769 = stablehlo.reduce(%v4768 init: %v4752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4770 = stablehlo.broadcast_in_dim %v4769, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4771 = stablehlo.multiply %v4765, %v4768 : tensor<32x24x56x56xf32>
    %v4772 = stablehlo.reduce(%v4771 init: %v4752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4773 = stablehlo.broadcast_in_dim %v4772, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4774 = stablehlo.multiply %v4768, %v4753 : tensor<32x24x56x56xf32>
    %v4775 = stablehlo.subtract %v4774, %v4770 : tensor<32x24x56x56xf32>
    %v4776 = stablehlo.multiply %v4765, %v4773 : tensor<32x24x56x56xf32>
    %v4777 = stablehlo.subtract %v4775, %v4776 : tensor<32x24x56x56xf32>
    %v4778 = stablehlo.divide %v4764, %v4753 : tensor<32x24x56x56xf32>
    %v4779 = stablehlo.multiply %v4778, %v4777 : tensor<32x24x56x56xf32>
    %v4780 = stablehlo.reshape %v4779 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4781 = stablehlo.reshape %v4780 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4782 = stablehlo.reverse %b2pW, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v4783 = stablehlo.transpose %v4782, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v4784 = stablehlo.convolution(%v4781, %v4783)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v4785 = stablehlo.reshape %v4784 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4786 = stablehlo.reshape %v4785 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4787 = stablehlo.reshape %v142 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4788 = stablehlo.constant dense<0.0> : tensor<32x96x56x56xf32>
    %v4789 = stablehlo.constant dense<6.0> : tensor<32x96x56x56xf32>
    %v4790 = stablehlo.compare GT, %v4787, %v4788 : (tensor<32x96x56x56xf32>, tensor<32x96x56x56xf32>) -> tensor<32x96x56x56xi1>
    %v4791 = stablehlo.compare LT, %v4787, %v4789 : (tensor<32x96x56x56xf32>, tensor<32x96x56x56xf32>) -> tensor<32x96x56x56xi1>
    %v4792 = stablehlo.and %v4790, %v4791 : tensor<32x96x56x56xi1>
    %v4793 = stablehlo.select %v4792, %v4786, %v4788 : tensor<32x96x56x56xi1>, tensor<32x96x56x56xf32>
    %v4794 = stablehlo.reshape %v4793 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4795 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4796 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4797 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v4798 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v4799 = stablehlo.reduce(%v4795 init: %v4796) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4800 = stablehlo.broadcast_in_dim %v4799, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4801 = stablehlo.divide %v4800, %v4797 : tensor<32x96x56x56xf32>
    %v4802 = stablehlo.subtract %v4795, %v4801 : tensor<32x96x56x56xf32>
    %v4803 = stablehlo.multiply %v4802, %v4802 : tensor<32x96x56x56xf32>
    %v4804 = stablehlo.reduce(%v4803 init: %v4796) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4805 = stablehlo.broadcast_in_dim %v4804, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4806 = stablehlo.divide %v4805, %v4797 : tensor<32x96x56x56xf32>
    %v4807 = stablehlo.add %v4806, %v4798 : tensor<32x96x56x56xf32>
    %v4808 = stablehlo.rsqrt %v4807 : tensor<32x96x56x56xf32>
    %v4809 = stablehlo.multiply %v4802, %v4808 : tensor<32x96x56x56xf32>
    %v4810 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4811 = stablehlo.reshape %v4794 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4812 = stablehlo.multiply %v4810, %v4811 : tensor<32x96x56x56xf32>
    %v4813 = stablehlo.reduce(%v4812 init: %v4796) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4814 = stablehlo.broadcast_in_dim %v4813, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4815 = stablehlo.multiply %v4809, %v4812 : tensor<32x96x56x56xf32>
    %v4816 = stablehlo.reduce(%v4815 init: %v4796) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4817 = stablehlo.broadcast_in_dim %v4816, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4818 = stablehlo.multiply %v4812, %v4797 : tensor<32x96x56x56xf32>
    %v4819 = stablehlo.subtract %v4818, %v4814 : tensor<32x96x56x56xf32>
    %v4820 = stablehlo.multiply %v4809, %v4817 : tensor<32x96x56x56xf32>
    %v4821 = stablehlo.subtract %v4819, %v4820 : tensor<32x96x56x56xf32>
    %v4822 = stablehlo.divide %v4808, %v4797 : tensor<32x96x56x56xf32>
    %v4823 = stablehlo.multiply %v4822, %v4821 : tensor<32x96x56x56xf32>
    %v4824 = stablehlo.reshape %v4823 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4825 = stablehlo.reshape %v4824 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4827 = stablehlo.pad %v4825, %v4826, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v4828 = stablehlo.reverse %b2dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v4829 = stablehlo.convolution(%v4827, %v4828)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 0], [2, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x112x112xf32>
    %v4830 = stablehlo.reshape %v4829 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v4831 = stablehlo.reshape %v4830 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4832 = stablehlo.reshape %v111 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4833 = stablehlo.constant dense<0.0> : tensor<32x96x112x112xf32>
    %v4834 = stablehlo.constant dense<6.0> : tensor<32x96x112x112xf32>
    %v4835 = stablehlo.compare GT, %v4832, %v4833 : (tensor<32x96x112x112xf32>, tensor<32x96x112x112xf32>) -> tensor<32x96x112x112xi1>
    %v4836 = stablehlo.compare LT, %v4832, %v4834 : (tensor<32x96x112x112xf32>, tensor<32x96x112x112xf32>) -> tensor<32x96x112x112xi1>
    %v4837 = stablehlo.and %v4835, %v4836 : tensor<32x96x112x112xi1>
    %v4838 = stablehlo.select %v4837, %v4831, %v4833 : tensor<32x96x112x112xi1>, tensor<32x96x112x112xf32>
    %v4839 = stablehlo.reshape %v4838 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v4840 = stablehlo.reshape %v91 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4841 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4842 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v4843 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v4844 = stablehlo.reduce(%v4840 init: %v4841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4845 = stablehlo.broadcast_in_dim %v4844, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4846 = stablehlo.divide %v4845, %v4842 : tensor<32x96x112x112xf32>
    %v4847 = stablehlo.subtract %v4840, %v4846 : tensor<32x96x112x112xf32>
    %v4848 = stablehlo.multiply %v4847, %v4847 : tensor<32x96x112x112xf32>
    %v4849 = stablehlo.reduce(%v4848 init: %v4841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4850 = stablehlo.broadcast_in_dim %v4849, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4851 = stablehlo.divide %v4850, %v4842 : tensor<32x96x112x112xf32>
    %v4852 = stablehlo.add %v4851, %v4843 : tensor<32x96x112x112xf32>
    %v4853 = stablehlo.rsqrt %v4852 : tensor<32x96x112x112xf32>
    %v4854 = stablehlo.multiply %v4847, %v4853 : tensor<32x96x112x112xf32>
    %v4855 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4856 = stablehlo.reshape %v4839 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4857 = stablehlo.multiply %v4855, %v4856 : tensor<32x96x112x112xf32>
    %v4858 = stablehlo.reduce(%v4857 init: %v4841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4859 = stablehlo.broadcast_in_dim %v4858, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4860 = stablehlo.multiply %v4854, %v4857 : tensor<32x96x112x112xf32>
    %v4861 = stablehlo.reduce(%v4860 init: %v4841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4862 = stablehlo.broadcast_in_dim %v4861, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4863 = stablehlo.multiply %v4857, %v4842 : tensor<32x96x112x112xf32>
    %v4864 = stablehlo.subtract %v4863, %v4859 : tensor<32x96x112x112xf32>
    %v4865 = stablehlo.multiply %v4854, %v4862 : tensor<32x96x112x112xf32>
    %v4866 = stablehlo.subtract %v4864, %v4865 : tensor<32x96x112x112xf32>
    %v4867 = stablehlo.divide %v4853, %v4842 : tensor<32x96x112x112xf32>
    %v4868 = stablehlo.multiply %v4867, %v4866 : tensor<32x96x112x112xf32>
    %v4869 = stablehlo.reshape %v4868 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v4870 = stablehlo.reshape %v4869 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4871 = stablehlo.reverse %b2eW, dims = [2, 3] : tensor<96x16x1x1xf32>
    %v4872 = stablehlo.transpose %v4871, dims = [1, 0, 2, 3] : (tensor<96x16x1x1xf32>) -> tensor<16x96x1x1xf32>
    %v4873 = stablehlo.convolution(%v4870, %v4872)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x112x112xf32>, tensor<16x96x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v4874 = stablehlo.reshape %v4873 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v4875 = stablehlo.reshape %v86 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4876 = stablehlo.reshape %v4869 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4877 = stablehlo.transpose %v4875, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v4878 = stablehlo.transpose %v4876, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v4879 = stablehlo.convolution(%v4877, %v4878)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<16x96x1x1xf32>
    %v4880 = stablehlo.transpose %v4879, dims = [1, 0, 2, 3] : (tensor<16x96x1x1xf32>) -> tensor<96x16x1x1xf32>
    %v4881 = stablehlo.reshape %v91 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4883 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v4884 = stablehlo.reduce(%v4881 init: %v4882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4885 = stablehlo.broadcast_in_dim %v4884, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4886 = stablehlo.divide %v4885, %v4883 : tensor<32x96x112x112xf32>
    %v4887 = stablehlo.subtract %v4881, %v4886 : tensor<32x96x112x112xf32>
    %v4888 = stablehlo.multiply %v4887, %v4887 : tensor<32x96x112x112xf32>
    %v4889 = stablehlo.reduce(%v4888 init: %v4882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4890 = stablehlo.broadcast_in_dim %v4889, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v4891 = stablehlo.divide %v4890, %v4883 : tensor<32x96x112x112xf32>
    %v4892 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v4893 = stablehlo.add %v4891, %v4892 : tensor<32x96x112x112xf32>
    %v4894 = stablehlo.rsqrt %v4893 : tensor<32x96x112x112xf32>
    %v4895 = stablehlo.multiply %v4887, %v4894 : tensor<32x96x112x112xf32>
    %v4896 = stablehlo.reshape %v4839 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4897 = stablehlo.multiply %v4896, %v4895 : tensor<32x96x112x112xf32>
    %v4898 = stablehlo.reduce(%v4897 init: %v4882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4899 = stablehlo.reshape %v4839 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4901 = stablehlo.reduce(%v4899 init: %v4900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v4902 = stablehlo.reshape %v117 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v4903 = stablehlo.reshape %v4824 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4904 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4905 = stablehlo.pad %v4903, %v4904, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v4906 = stablehlo.transpose %v4902, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v4907 = stablehlo.transpose %v4905, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v4908 = stablehlo.convolution(%v4906, %v4907)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 2], [0, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<1x96x3x3xf32>
    %v4909 = stablehlo.reshape %v4908 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v4910 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4912 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v4913 = stablehlo.reduce(%v4910 init: %v4911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4914 = stablehlo.broadcast_in_dim %v4913, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4915 = stablehlo.divide %v4914, %v4912 : tensor<32x96x56x56xf32>
    %v4916 = stablehlo.subtract %v4910, %v4915 : tensor<32x96x56x56xf32>
    %v4917 = stablehlo.multiply %v4916, %v4916 : tensor<32x96x56x56xf32>
    %v4918 = stablehlo.reduce(%v4917 init: %v4911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4919 = stablehlo.broadcast_in_dim %v4918, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4920 = stablehlo.divide %v4919, %v4912 : tensor<32x96x56x56xf32>
    %v4921 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v4922 = stablehlo.add %v4920, %v4921 : tensor<32x96x56x56xf32>
    %v4923 = stablehlo.rsqrt %v4922 : tensor<32x96x56x56xf32>
    %v4924 = stablehlo.multiply %v4916, %v4923 : tensor<32x96x56x56xf32>
    %v4925 = stablehlo.reshape %v4794 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4926 = stablehlo.multiply %v4925, %v4924 : tensor<32x96x56x56xf32>
    %v4927 = stablehlo.reduce(%v4926 init: %v4911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4928 = stablehlo.reshape %v4794 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4929 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4930 = stablehlo.reduce(%v4928 init: %v4929) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4931 = stablehlo.reshape %v148 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4932 = stablehlo.reshape %v4780 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4933 = stablehlo.transpose %v4931, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4934 = stablehlo.transpose %v4932, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4935 = stablehlo.convolution(%v4933, %v4934)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %v4936 = stablehlo.transpose %v4935, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v4937 = stablehlo.reshape %v153 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4939 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v4940 = stablehlo.reduce(%v4937 init: %v4938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4941 = stablehlo.broadcast_in_dim %v4940, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4942 = stablehlo.divide %v4941, %v4939 : tensor<32x24x56x56xf32>
    %v4943 = stablehlo.subtract %v4937, %v4942 : tensor<32x24x56x56xf32>
    %v4944 = stablehlo.multiply %v4943, %v4943 : tensor<32x24x56x56xf32>
    %v4945 = stablehlo.reduce(%v4944 init: %v4938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4946 = stablehlo.broadcast_in_dim %v4945, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4947 = stablehlo.divide %v4946, %v4939 : tensor<32x24x56x56xf32>
    %v4948 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4949 = stablehlo.add %v4947, %v4948 : tensor<32x24x56x56xf32>
    %v4950 = stablehlo.rsqrt %v4949 : tensor<32x24x56x56xf32>
    %v4951 = stablehlo.multiply %v4943, %v4950 : tensor<32x24x56x56xf32>
    %v4952 = stablehlo.reshape %v4669 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4953 = stablehlo.multiply %v4952, %v4951 : tensor<32x24x56x56xf32>
    %v4954 = stablehlo.reduce(%v4953 init: %v4938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4955 = stablehlo.reshape %v4669 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4956 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4957 = stablehlo.reduce(%v4955 init: %v4956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4958 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4960 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v4961 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v4962 = stablehlo.reduce(%v4958 init: %v4959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4963 = stablehlo.broadcast_in_dim %v4962, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4964 = stablehlo.divide %v4963, %v4960 : tensor<32x16x112x112xf32>
    %v4965 = stablehlo.subtract %v4958, %v4964 : tensor<32x16x112x112xf32>
    %v4966 = stablehlo.multiply %v4965, %v4965 : tensor<32x16x112x112xf32>
    %v4967 = stablehlo.reduce(%v4966 init: %v4959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4968 = stablehlo.broadcast_in_dim %v4967, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4969 = stablehlo.divide %v4968, %v4960 : tensor<32x16x112x112xf32>
    %v4970 = stablehlo.add %v4969, %v4961 : tensor<32x16x112x112xf32>
    %v4971 = stablehlo.rsqrt %v4970 : tensor<32x16x112x112xf32>
    %v4972 = stablehlo.multiply %v4965, %v4971 : tensor<32x16x112x112xf32>
    %v4973 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4974 = stablehlo.reshape %v4874 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4975 = stablehlo.multiply %v4973, %v4974 : tensor<32x16x112x112xf32>
    %v4976 = stablehlo.reduce(%v4975 init: %v4959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4977 = stablehlo.broadcast_in_dim %v4976, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4978 = stablehlo.multiply %v4972, %v4975 : tensor<32x16x112x112xf32>
    %v4979 = stablehlo.reduce(%v4978 init: %v4959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v4980 = stablehlo.broadcast_in_dim %v4979, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v4981 = stablehlo.multiply %v4975, %v4960 : tensor<32x16x112x112xf32>
    %v4982 = stablehlo.subtract %v4981, %v4977 : tensor<32x16x112x112xf32>
    %v4983 = stablehlo.multiply %v4972, %v4980 : tensor<32x16x112x112xf32>
    %v4984 = stablehlo.subtract %v4982, %v4983 : tensor<32x16x112x112xf32>
    %v4985 = stablehlo.divide %v4971, %v4960 : tensor<32x16x112x112xf32>
    %v4986 = stablehlo.multiply %v4985, %v4984 : tensor<32x16x112x112xf32>
    %v4987 = stablehlo.reshape %v4986 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v4988 = stablehlo.reshape %v4987 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v4989 = stablehlo.reverse %b1pW, dims = [2, 3] : tensor<16x32x1x1xf32>
    %v4990 = stablehlo.transpose %v4989, dims = [1, 0, 2, 3] : (tensor<16x32x1x1xf32>) -> tensor<32x16x1x1xf32>
    %v4991 = stablehlo.convolution(%v4988, %v4990)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<32x16x1x1xf32>) -> tensor<32x32x112x112xf32>
    %v4992 = stablehlo.reshape %v4991 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v4993 = stablehlo.reshape %v4992 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4994 = stablehlo.reshape %v55 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v4995 = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %v4996 = stablehlo.constant dense<6.0> : tensor<32x32x112x112xf32>
    %v4997 = stablehlo.compare GT, %v4994, %v4995 : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xi1>
    %v4998 = stablehlo.compare LT, %v4994, %v4996 : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xi1>
    %v4999 = stablehlo.and %v4997, %v4998 : tensor<32x32x112x112xi1>
    %v5000 = stablehlo.select %v4999, %v4993, %v4995 : tensor<32x32x112x112xi1>, tensor<32x32x112x112xf32>
    %v5001 = stablehlo.reshape %v5000 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5002 = stablehlo.reshape %v35 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5003 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5004 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v5005 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5006 = stablehlo.reduce(%v5002 init: %v5003) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5007 = stablehlo.broadcast_in_dim %v5006, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5008 = stablehlo.divide %v5007, %v5004 : tensor<32x32x112x112xf32>
    %v5009 = stablehlo.subtract %v5002, %v5008 : tensor<32x32x112x112xf32>
    %v5010 = stablehlo.multiply %v5009, %v5009 : tensor<32x32x112x112xf32>
    %v5011 = stablehlo.reduce(%v5010 init: %v5003) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5012 = stablehlo.broadcast_in_dim %v5011, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5013 = stablehlo.divide %v5012, %v5004 : tensor<32x32x112x112xf32>
    %v5014 = stablehlo.add %v5013, %v5005 : tensor<32x32x112x112xf32>
    %v5015 = stablehlo.rsqrt %v5014 : tensor<32x32x112x112xf32>
    %v5016 = stablehlo.multiply %v5009, %v5015 : tensor<32x32x112x112xf32>
    %v5017 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5018 = stablehlo.reshape %v5001 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5019 = stablehlo.multiply %v5017, %v5018 : tensor<32x32x112x112xf32>
    %v5020 = stablehlo.reduce(%v5019 init: %v5003) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5021 = stablehlo.broadcast_in_dim %v5020, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5022 = stablehlo.multiply %v5016, %v5019 : tensor<32x32x112x112xf32>
    %v5023 = stablehlo.reduce(%v5022 init: %v5003) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5024 = stablehlo.broadcast_in_dim %v5023, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5025 = stablehlo.multiply %v5019, %v5004 : tensor<32x32x112x112xf32>
    %v5026 = stablehlo.subtract %v5025, %v5021 : tensor<32x32x112x112xf32>
    %v5027 = stablehlo.multiply %v5016, %v5024 : tensor<32x32x112x112xf32>
    %v5028 = stablehlo.subtract %v5026, %v5027 : tensor<32x32x112x112xf32>
    %v5029 = stablehlo.divide %v5015, %v5004 : tensor<32x32x112x112xf32>
    %v5030 = stablehlo.multiply %v5029, %v5028 : tensor<32x32x112x112xf32>
    %v5031 = stablehlo.reshape %v5030 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5032 = stablehlo.reshape %v5031 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5033 = stablehlo.reverse %b1dW, dims = [2, 3] : tensor<32x1x3x3xf32>
    %v5034 = stablehlo.convolution(%v5032, %v5033)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v5035 = stablehlo.reshape %v5034 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5036 = stablehlo.reshape %v30 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5037 = stablehlo.reshape %v5031 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5038 = stablehlo.transpose %v5036, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5039 = stablehlo.transpose %v5037, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5040 = stablehlo.convolution(%v5038, %v5039)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 32 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<1x32x3x3xf32>
    %v5041 = stablehlo.reshape %v5040 : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %v5042 = stablehlo.reshape %v35 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5044 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v5045 = stablehlo.reduce(%v5042 init: %v5043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5046 = stablehlo.broadcast_in_dim %v5045, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5047 = stablehlo.divide %v5046, %v5044 : tensor<32x32x112x112xf32>
    %v5048 = stablehlo.subtract %v5042, %v5047 : tensor<32x32x112x112xf32>
    %v5049 = stablehlo.multiply %v5048, %v5048 : tensor<32x32x112x112xf32>
    %v5050 = stablehlo.reduce(%v5049 init: %v5043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5051 = stablehlo.broadcast_in_dim %v5050, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5052 = stablehlo.divide %v5051, %v5044 : tensor<32x32x112x112xf32>
    %v5053 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5054 = stablehlo.add %v5052, %v5053 : tensor<32x32x112x112xf32>
    %v5055 = stablehlo.rsqrt %v5054 : tensor<32x32x112x112xf32>
    %v5056 = stablehlo.multiply %v5048, %v5055 : tensor<32x32x112x112xf32>
    %v5057 = stablehlo.reshape %v5001 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5058 = stablehlo.multiply %v5057, %v5056 : tensor<32x32x112x112xf32>
    %v5059 = stablehlo.reduce(%v5058 init: %v5043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5060 = stablehlo.reshape %v5001 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5062 = stablehlo.reduce(%v5060 init: %v5061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5063 = stablehlo.reshape %v61 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5064 = stablehlo.reshape %v4987 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5065 = stablehlo.transpose %v5063, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5066 = stablehlo.transpose %v5064, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v5067 = stablehlo.convolution(%v5065, %v5066)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x112x112xf32>) -> tensor<32x16x1x1xf32>
    %v5068 = stablehlo.transpose %v5067, dims = [1, 0, 2, 3] : (tensor<32x16x1x1xf32>) -> tensor<16x32x1x1xf32>
    %v5069 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5070 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5071 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v5072 = stablehlo.reduce(%v5069 init: %v5070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5073 = stablehlo.broadcast_in_dim %v5072, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v5074 = stablehlo.divide %v5073, %v5071 : tensor<32x16x112x112xf32>
    %v5075 = stablehlo.subtract %v5069, %v5074 : tensor<32x16x112x112xf32>
    %v5076 = stablehlo.multiply %v5075, %v5075 : tensor<32x16x112x112xf32>
    %v5077 = stablehlo.reduce(%v5076 init: %v5070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5078 = stablehlo.broadcast_in_dim %v5077, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v5079 = stablehlo.divide %v5078, %v5071 : tensor<32x16x112x112xf32>
    %v5080 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v5081 = stablehlo.add %v5079, %v5080 : tensor<32x16x112x112xf32>
    %v5082 = stablehlo.rsqrt %v5081 : tensor<32x16x112x112xf32>
    %v5083 = stablehlo.multiply %v5075, %v5082 : tensor<32x16x112x112xf32>
    %v5084 = stablehlo.reshape %v4874 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5085 = stablehlo.multiply %v5084, %v5083 : tensor<32x16x112x112xf32>
    %v5086 = stablehlo.reduce(%v5085 init: %v5070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5087 = stablehlo.reshape %v4874 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5088 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5089 = stablehlo.reduce(%v5087 init: %v5088) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5090 = stablehlo.reshape %v5035 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5091 = stablehlo.reshape %v24 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5092 = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %v5093 = stablehlo.constant dense<6.0> : tensor<32x32x112x112xf32>
    %v5094 = stablehlo.compare GT, %v5091, %v5092 : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xi1>
    %v5095 = stablehlo.compare LT, %v5091, %v5093 : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xi1>
    %v5096 = stablehlo.and %v5094, %v5095 : tensor<32x32x112x112xi1>
    %v5097 = stablehlo.select %v5096, %v5090, %v5092 : tensor<32x32x112x112xi1>, tensor<32x32x112x112xf32>
    %v5098 = stablehlo.reshape %v5097 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5099 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5100 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5101 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v5102 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5103 = stablehlo.reduce(%v5099 init: %v5100) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5104 = stablehlo.broadcast_in_dim %v5103, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5105 = stablehlo.divide %v5104, %v5101 : tensor<32x32x112x112xf32>
    %v5106 = stablehlo.subtract %v5099, %v5105 : tensor<32x32x112x112xf32>
    %v5107 = stablehlo.multiply %v5106, %v5106 : tensor<32x32x112x112xf32>
    %v5108 = stablehlo.reduce(%v5107 init: %v5100) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5109 = stablehlo.broadcast_in_dim %v5108, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5110 = stablehlo.divide %v5109, %v5101 : tensor<32x32x112x112xf32>
    %v5111 = stablehlo.add %v5110, %v5102 : tensor<32x32x112x112xf32>
    %v5112 = stablehlo.rsqrt %v5111 : tensor<32x32x112x112xf32>
    %v5113 = stablehlo.multiply %v5106, %v5112 : tensor<32x32x112x112xf32>
    %v5114 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5115 = stablehlo.reshape %v5098 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5116 = stablehlo.multiply %v5114, %v5115 : tensor<32x32x112x112xf32>
    %v5117 = stablehlo.reduce(%v5116 init: %v5100) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5118 = stablehlo.broadcast_in_dim %v5117, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5119 = stablehlo.multiply %v5113, %v5116 : tensor<32x32x112x112xf32>
    %v5120 = stablehlo.reduce(%v5119 init: %v5100) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5121 = stablehlo.broadcast_in_dim %v5120, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5122 = stablehlo.multiply %v5116, %v5101 : tensor<32x32x112x112xf32>
    %v5123 = stablehlo.subtract %v5122, %v5118 : tensor<32x32x112x112xf32>
    %v5124 = stablehlo.multiply %v5113, %v5121 : tensor<32x32x112x112xf32>
    %v5125 = stablehlo.subtract %v5123, %v5124 : tensor<32x32x112x112xf32>
    %v5126 = stablehlo.divide %v5112, %v5101 : tensor<32x32x112x112xf32>
    %v5127 = stablehlo.multiply %v5126, %v5125 : tensor<32x32x112x112xf32>
    %v5128 = stablehlo.reshape %v5127 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5129 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v5130 = stablehlo.reshape %v5128 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5132 = stablehlo.pad %v5130, %v5131, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32x224x224xf32>
    %v5133 = stablehlo.transpose %v5129, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v5134 = stablehlo.transpose %v5132, dims = [1, 0, 2, 3] : (tensor<32x32x224x224xf32>) -> tensor<32x32x224x224xf32>
    %v5135 = stablehlo.convolution(%v5133, %v5134)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 2], [0, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<32x32x224x224xf32>) -> tensor<3x32x3x3xf32>
    %v5136 = stablehlo.transpose %v5135, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %v5137 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5139 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v5140 = stablehlo.reduce(%v5137 init: %v5138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5141 = stablehlo.broadcast_in_dim %v5140, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5142 = stablehlo.divide %v5141, %v5139 : tensor<32x32x112x112xf32>
    %v5143 = stablehlo.subtract %v5137, %v5142 : tensor<32x32x112x112xf32>
    %v5144 = stablehlo.multiply %v5143, %v5143 : tensor<32x32x112x112xf32>
    %v5145 = stablehlo.reduce(%v5144 init: %v5138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5146 = stablehlo.broadcast_in_dim %v5145, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5147 = stablehlo.divide %v5146, %v5139 : tensor<32x32x112x112xf32>
    %v5148 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5149 = stablehlo.add %v5147, %v5148 : tensor<32x32x112x112xf32>
    %v5150 = stablehlo.rsqrt %v5149 : tensor<32x32x112x112xf32>
    %v5151 = stablehlo.multiply %v5143, %v5150 : tensor<32x32x112x112xf32>
    %v5152 = stablehlo.reshape %v5098 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5153 = stablehlo.multiply %v5152, %v5151 : tensor<32x32x112x112xf32>
    %v5154 = stablehlo.reduce(%v5153 init: %v5138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5155 = stablehlo.reshape %v5098 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5157 = stablehlo.reduce(%v5155 init: %v5156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5158 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5160 = stablehlo.constant dense<401408.0> : tensor<32xf32>
    %v5161 = stablehlo.reduce(%v5158 init: %v5159) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5162 = stablehlo.divide %v5161, %v5160 : tensor<32xf32>
    %v5163 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5165 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v5166 = stablehlo.reduce(%v5163 init: %v5164) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5167 = stablehlo.broadcast_in_dim %v5166, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5168 = stablehlo.divide %v5167, %v5165 : tensor<32x32x112x112xf32>
    %v5169 = stablehlo.subtract %v5163, %v5168 : tensor<32x32x112x112xf32>
    %v5170 = stablehlo.multiply %v5169, %v5169 : tensor<32x32x112x112xf32>
    %v5171 = stablehlo.reduce(%v5170 init: %v5164) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5172 = stablehlo.constant dense<401408.0> : tensor<32xf32>
    %v5173 = stablehlo.divide %v5171, %v5172 : tensor<32xf32>
    %v5174 = stablehlo.reshape %v35 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5175 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5176 = stablehlo.constant dense<401408.0> : tensor<32xf32>
    %v5177 = stablehlo.reduce(%v5174 init: %v5175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5178 = stablehlo.divide %v5177, %v5176 : tensor<32xf32>
    %v5179 = stablehlo.reshape %v35 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5181 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v5182 = stablehlo.reduce(%v5179 init: %v5180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5183 = stablehlo.broadcast_in_dim %v5182, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5184 = stablehlo.divide %v5183, %v5181 : tensor<32x32x112x112xf32>
    %v5185 = stablehlo.subtract %v5179, %v5184 : tensor<32x32x112x112xf32>
    %v5186 = stablehlo.multiply %v5185, %v5185 : tensor<32x32x112x112xf32>
    %v5187 = stablehlo.reduce(%v5186 init: %v5180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5188 = stablehlo.constant dense<401408.0> : tensor<32xf32>
    %v5189 = stablehlo.divide %v5187, %v5188 : tensor<32xf32>
    %v5190 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5191 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5192 = stablehlo.constant dense<401408.0> : tensor<16xf32>
    %v5193 = stablehlo.reduce(%v5190 init: %v5191) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5194 = stablehlo.divide %v5193, %v5192 : tensor<16xf32>
    %v5195 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5196 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5197 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v5198 = stablehlo.reduce(%v5195 init: %v5196) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5199 = stablehlo.broadcast_in_dim %v5198, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v5200 = stablehlo.divide %v5199, %v5197 : tensor<32x16x112x112xf32>
    %v5201 = stablehlo.subtract %v5195, %v5200 : tensor<32x16x112x112xf32>
    %v5202 = stablehlo.multiply %v5201, %v5201 : tensor<32x16x112x112xf32>
    %v5203 = stablehlo.reduce(%v5202 init: %v5196) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5204 = stablehlo.constant dense<401408.0> : tensor<16xf32>
    %v5205 = stablehlo.divide %v5203, %v5204 : tensor<16xf32>
    %v5206 = stablehlo.reshape %v91 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5208 = stablehlo.constant dense<401408.0> : tensor<96xf32>
    %v5209 = stablehlo.reduce(%v5206 init: %v5207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v5210 = stablehlo.divide %v5209, %v5208 : tensor<96xf32>
    %v5211 = stablehlo.reshape %v91 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5212 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5213 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v5214 = stablehlo.reduce(%v5211 init: %v5212) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v5215 = stablehlo.broadcast_in_dim %v5214, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v5216 = stablehlo.divide %v5215, %v5213 : tensor<32x96x112x112xf32>
    %v5217 = stablehlo.subtract %v5211, %v5216 : tensor<32x96x112x112xf32>
    %v5218 = stablehlo.multiply %v5217, %v5217 : tensor<32x96x112x112xf32>
    %v5219 = stablehlo.reduce(%v5218 init: %v5212) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v5220 = stablehlo.constant dense<401408.0> : tensor<96xf32>
    %v5221 = stablehlo.divide %v5219, %v5220 : tensor<96xf32>
    %v5222 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5223 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5224 = stablehlo.constant dense<100352.0> : tensor<96xf32>
    %v5225 = stablehlo.reduce(%v5222 init: %v5223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5226 = stablehlo.divide %v5225, %v5224 : tensor<96xf32>
    %v5227 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5228 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5229 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v5230 = stablehlo.reduce(%v5227 init: %v5228) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5231 = stablehlo.broadcast_in_dim %v5230, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v5232 = stablehlo.divide %v5231, %v5229 : tensor<32x96x56x56xf32>
    %v5233 = stablehlo.subtract %v5227, %v5232 : tensor<32x96x56x56xf32>
    %v5234 = stablehlo.multiply %v5233, %v5233 : tensor<32x96x56x56xf32>
    %v5235 = stablehlo.reduce(%v5234 init: %v5228) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5236 = stablehlo.constant dense<100352.0> : tensor<96xf32>
    %v5237 = stablehlo.divide %v5235, %v5236 : tensor<96xf32>
    %v5238 = stablehlo.reshape %v153 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5239 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5240 = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %v5241 = stablehlo.reduce(%v5238 init: %v5239) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5242 = stablehlo.divide %v5241, %v5240 : tensor<24xf32>
    %v5243 = stablehlo.reshape %v153 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5245 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v5246 = stablehlo.reduce(%v5243 init: %v5244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5247 = stablehlo.broadcast_in_dim %v5246, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5248 = stablehlo.divide %v5247, %v5245 : tensor<32x24x56x56xf32>
    %v5249 = stablehlo.subtract %v5243, %v5248 : tensor<32x24x56x56xf32>
    %v5250 = stablehlo.multiply %v5249, %v5249 : tensor<32x24x56x56xf32>
    %v5251 = stablehlo.reduce(%v5250 init: %v5244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5252 = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %v5253 = stablehlo.divide %v5251, %v5252 : tensor<24xf32>
    %v5254 = stablehlo.reshape %v178 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5255 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5256 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5257 = stablehlo.reduce(%v5254 init: %v5255) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5258 = stablehlo.divide %v5257, %v5256 : tensor<144xf32>
    %v5259 = stablehlo.reshape %v178 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5261 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5262 = stablehlo.reduce(%v5259 init: %v5260) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5263 = stablehlo.broadcast_in_dim %v5262, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5264 = stablehlo.divide %v5263, %v5261 : tensor<32x144x56x56xf32>
    %v5265 = stablehlo.subtract %v5259, %v5264 : tensor<32x144x56x56xf32>
    %v5266 = stablehlo.multiply %v5265, %v5265 : tensor<32x144x56x56xf32>
    %v5267 = stablehlo.reduce(%v5266 init: %v5260) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5268 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5269 = stablehlo.divide %v5267, %v5268 : tensor<144xf32>
    %v5270 = stablehlo.reshape %v209 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5272 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5273 = stablehlo.reduce(%v5270 init: %v5271) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5274 = stablehlo.divide %v5273, %v5272 : tensor<144xf32>
    %v5275 = stablehlo.reshape %v209 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5276 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5277 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5278 = stablehlo.reduce(%v5275 init: %v5276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5279 = stablehlo.broadcast_in_dim %v5278, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5280 = stablehlo.divide %v5279, %v5277 : tensor<32x144x56x56xf32>
    %v5281 = stablehlo.subtract %v5275, %v5280 : tensor<32x144x56x56xf32>
    %v5282 = stablehlo.multiply %v5281, %v5281 : tensor<32x144x56x56xf32>
    %v5283 = stablehlo.reduce(%v5282 init: %v5276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5284 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5285 = stablehlo.divide %v5283, %v5284 : tensor<144xf32>
    %v5286 = stablehlo.reshape %v240 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5287 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5288 = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %v5289 = stablehlo.reduce(%v5286 init: %v5287) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5290 = stablehlo.divide %v5289, %v5288 : tensor<24xf32>
    %v5291 = stablehlo.reshape %v240 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5292 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5293 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v5294 = stablehlo.reduce(%v5291 init: %v5292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5295 = stablehlo.broadcast_in_dim %v5294, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5296 = stablehlo.divide %v5295, %v5293 : tensor<32x24x56x56xf32>
    %v5297 = stablehlo.subtract %v5291, %v5296 : tensor<32x24x56x56xf32>
    %v5298 = stablehlo.multiply %v5297, %v5297 : tensor<32x24x56x56xf32>
    %v5299 = stablehlo.reduce(%v5298 init: %v5292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5300 = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %v5301 = stablehlo.divide %v5299, %v5300 : tensor<24xf32>
    %v5302 = stablehlo.reshape %v269 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5304 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5305 = stablehlo.reduce(%v5302 init: %v5303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5306 = stablehlo.divide %v5305, %v5304 : tensor<144xf32>
    %v5307 = stablehlo.reshape %v269 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5309 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5310 = stablehlo.reduce(%v5307 init: %v5308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5311 = stablehlo.broadcast_in_dim %v5310, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5312 = stablehlo.divide %v5311, %v5309 : tensor<32x144x56x56xf32>
    %v5313 = stablehlo.subtract %v5307, %v5312 : tensor<32x144x56x56xf32>
    %v5314 = stablehlo.multiply %v5313, %v5313 : tensor<32x144x56x56xf32>
    %v5315 = stablehlo.reduce(%v5314 init: %v5308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5316 = stablehlo.constant dense<100352.0> : tensor<144xf32>
    %v5317 = stablehlo.divide %v5315, %v5316 : tensor<144xf32>
    %v5318 = stablehlo.reshape %v300 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5319 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5320 = stablehlo.constant dense<25088.0> : tensor<144xf32>
    %v5321 = stablehlo.reduce(%v5318 init: %v5319) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5322 = stablehlo.divide %v5321, %v5320 : tensor<144xf32>
    %v5323 = stablehlo.reshape %v300 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5324 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5325 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v5326 = stablehlo.reduce(%v5323 init: %v5324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5327 = stablehlo.broadcast_in_dim %v5326, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5328 = stablehlo.divide %v5327, %v5325 : tensor<32x144x28x28xf32>
    %v5329 = stablehlo.subtract %v5323, %v5328 : tensor<32x144x28x28xf32>
    %v5330 = stablehlo.multiply %v5329, %v5329 : tensor<32x144x28x28xf32>
    %v5331 = stablehlo.reduce(%v5330 init: %v5324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5332 = stablehlo.constant dense<25088.0> : tensor<144xf32>
    %v5333 = stablehlo.divide %v5331, %v5332 : tensor<144xf32>
    %v5334 = stablehlo.reshape %v331 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5336 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5337 = stablehlo.reduce(%v5334 init: %v5335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5338 = stablehlo.divide %v5337, %v5336 : tensor<32xf32>
    %v5339 = stablehlo.reshape %v331 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5340 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5341 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v5342 = stablehlo.reduce(%v5339 init: %v5340) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5343 = stablehlo.broadcast_in_dim %v5342, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v5344 = stablehlo.divide %v5343, %v5341 : tensor<32x32x28x28xf32>
    %v5345 = stablehlo.subtract %v5339, %v5344 : tensor<32x32x28x28xf32>
    %v5346 = stablehlo.multiply %v5345, %v5345 : tensor<32x32x28x28xf32>
    %v5347 = stablehlo.reduce(%v5346 init: %v5340) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5348 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5349 = stablehlo.divide %v5347, %v5348 : tensor<32xf32>
    %v5350 = stablehlo.reshape %v356 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5351 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5352 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5353 = stablehlo.reduce(%v5350 init: %v5351) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5354 = stablehlo.divide %v5353, %v5352 : tensor<192xf32>
    %v5355 = stablehlo.reshape %v356 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5357 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v5358 = stablehlo.reduce(%v5355 init: %v5356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5359 = stablehlo.broadcast_in_dim %v5358, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v5360 = stablehlo.divide %v5359, %v5357 : tensor<32x192x28x28xf32>
    %v5361 = stablehlo.subtract %v5355, %v5360 : tensor<32x192x28x28xf32>
    %v5362 = stablehlo.multiply %v5361, %v5361 : tensor<32x192x28x28xf32>
    %v5363 = stablehlo.reduce(%v5362 init: %v5356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5364 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5365 = stablehlo.divide %v5363, %v5364 : tensor<192xf32>
    %v5366 = stablehlo.reshape %v387 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5368 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5369 = stablehlo.reduce(%v5366 init: %v5367) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5370 = stablehlo.divide %v5369, %v5368 : tensor<192xf32>
    %v5371 = stablehlo.reshape %v387 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5372 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5373 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v5374 = stablehlo.reduce(%v5371 init: %v5372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5375 = stablehlo.broadcast_in_dim %v5374, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v5376 = stablehlo.divide %v5375, %v5373 : tensor<32x192x28x28xf32>
    %v5377 = stablehlo.subtract %v5371, %v5376 : tensor<32x192x28x28xf32>
    %v5378 = stablehlo.multiply %v5377, %v5377 : tensor<32x192x28x28xf32>
    %v5379 = stablehlo.reduce(%v5378 init: %v5372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5380 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5381 = stablehlo.divide %v5379, %v5380 : tensor<192xf32>
    %v5382 = stablehlo.reshape %v418 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5384 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5385 = stablehlo.reduce(%v5382 init: %v5383) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5386 = stablehlo.divide %v5385, %v5384 : tensor<32xf32>
    %v5387 = stablehlo.reshape %v418 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5388 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5389 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v5390 = stablehlo.reduce(%v5387 init: %v5388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5391 = stablehlo.broadcast_in_dim %v5390, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v5392 = stablehlo.divide %v5391, %v5389 : tensor<32x32x28x28xf32>
    %v5393 = stablehlo.subtract %v5387, %v5392 : tensor<32x32x28x28xf32>
    %v5394 = stablehlo.multiply %v5393, %v5393 : tensor<32x32x28x28xf32>
    %v5395 = stablehlo.reduce(%v5394 init: %v5388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5396 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5397 = stablehlo.divide %v5395, %v5396 : tensor<32xf32>
    %v5398 = stablehlo.reshape %v447 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5400 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5401 = stablehlo.reduce(%v5398 init: %v5399) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5402 = stablehlo.divide %v5401, %v5400 : tensor<192xf32>
    %v5403 = stablehlo.reshape %v447 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5405 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v5406 = stablehlo.reduce(%v5403 init: %v5404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5407 = stablehlo.broadcast_in_dim %v5406, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v5408 = stablehlo.divide %v5407, %v5405 : tensor<32x192x28x28xf32>
    %v5409 = stablehlo.subtract %v5403, %v5408 : tensor<32x192x28x28xf32>
    %v5410 = stablehlo.multiply %v5409, %v5409 : tensor<32x192x28x28xf32>
    %v5411 = stablehlo.reduce(%v5410 init: %v5404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5412 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5413 = stablehlo.divide %v5411, %v5412 : tensor<192xf32>
    %v5414 = stablehlo.reshape %v478 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5416 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5417 = stablehlo.reduce(%v5414 init: %v5415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5418 = stablehlo.divide %v5417, %v5416 : tensor<192xf32>
    %v5419 = stablehlo.reshape %v478 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5421 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v5422 = stablehlo.reduce(%v5419 init: %v5420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5423 = stablehlo.broadcast_in_dim %v5422, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v5424 = stablehlo.divide %v5423, %v5421 : tensor<32x192x28x28xf32>
    %v5425 = stablehlo.subtract %v5419, %v5424 : tensor<32x192x28x28xf32>
    %v5426 = stablehlo.multiply %v5425, %v5425 : tensor<32x192x28x28xf32>
    %v5427 = stablehlo.reduce(%v5426 init: %v5420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5428 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5429 = stablehlo.divide %v5427, %v5428 : tensor<192xf32>
    %v5430 = stablehlo.reshape %v509 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5432 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5433 = stablehlo.reduce(%v5430 init: %v5431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5434 = stablehlo.divide %v5433, %v5432 : tensor<32xf32>
    %v5435 = stablehlo.reshape %v509 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v5436 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5437 = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %v5438 = stablehlo.reduce(%v5435 init: %v5436) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5439 = stablehlo.broadcast_in_dim %v5438, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v5440 = stablehlo.divide %v5439, %v5437 : tensor<32x32x28x28xf32>
    %v5441 = stablehlo.subtract %v5435, %v5440 : tensor<32x32x28x28xf32>
    %v5442 = stablehlo.multiply %v5441, %v5441 : tensor<32x32x28x28xf32>
    %v5443 = stablehlo.reduce(%v5442 init: %v5436) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v5444 = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %v5445 = stablehlo.divide %v5443, %v5444 : tensor<32xf32>
    %v5446 = stablehlo.reshape %v538 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5448 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5449 = stablehlo.reduce(%v5446 init: %v5447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5450 = stablehlo.divide %v5449, %v5448 : tensor<192xf32>
    %v5451 = stablehlo.reshape %v538 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v5452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5453 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v5454 = stablehlo.reduce(%v5451 init: %v5452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5455 = stablehlo.broadcast_in_dim %v5454, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v5456 = stablehlo.divide %v5455, %v5453 : tensor<32x192x28x28xf32>
    %v5457 = stablehlo.subtract %v5451, %v5456 : tensor<32x192x28x28xf32>
    %v5458 = stablehlo.multiply %v5457, %v5457 : tensor<32x192x28x28xf32>
    %v5459 = stablehlo.reduce(%v5458 init: %v5452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v5460 = stablehlo.constant dense<25088.0> : tensor<192xf32>
    %v5461 = stablehlo.divide %v5459, %v5460 : tensor<192xf32>
    %v5462 = stablehlo.reshape %v569 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v5463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5464 = stablehlo.constant dense<6272.0> : tensor<192xf32>
    %v5465 = stablehlo.reduce(%v5462 init: %v5463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v5466 = stablehlo.divide %v5465, %v5464 : tensor<192xf32>
    %v5467 = stablehlo.reshape %v569 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v5468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5469 = stablehlo.constant dense<6272.0> : tensor<32x192x14x14xf32>
    %v5470 = stablehlo.reduce(%v5467 init: %v5468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v5471 = stablehlo.broadcast_in_dim %v5470, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v5472 = stablehlo.divide %v5471, %v5469 : tensor<32x192x14x14xf32>
    %v5473 = stablehlo.subtract %v5467, %v5472 : tensor<32x192x14x14xf32>
    %v5474 = stablehlo.multiply %v5473, %v5473 : tensor<32x192x14x14xf32>
    %v5475 = stablehlo.reduce(%v5474 init: %v5468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v5476 = stablehlo.constant dense<6272.0> : tensor<192xf32>
    %v5477 = stablehlo.divide %v5475, %v5476 : tensor<192xf32>
    %v5478 = stablehlo.reshape %v600 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5479 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5480 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5481 = stablehlo.reduce(%v5478 init: %v5479) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5482 = stablehlo.divide %v5481, %v5480 : tensor<64xf32>
    %v5483 = stablehlo.reshape %v600 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5485 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v5486 = stablehlo.reduce(%v5483 init: %v5484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5487 = stablehlo.broadcast_in_dim %v5486, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v5488 = stablehlo.divide %v5487, %v5485 : tensor<32x64x14x14xf32>
    %v5489 = stablehlo.subtract %v5483, %v5488 : tensor<32x64x14x14xf32>
    %v5490 = stablehlo.multiply %v5489, %v5489 : tensor<32x64x14x14xf32>
    %v5491 = stablehlo.reduce(%v5490 init: %v5484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5492 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5493 = stablehlo.divide %v5491, %v5492 : tensor<64xf32>
    %v5494 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5495 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5496 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5497 = stablehlo.reduce(%v5494 init: %v5495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5498 = stablehlo.divide %v5497, %v5496 : tensor<384xf32>
    %v5499 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5500 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5501 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5502 = stablehlo.reduce(%v5499 init: %v5500) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5503 = stablehlo.broadcast_in_dim %v5502, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5504 = stablehlo.divide %v5503, %v5501 : tensor<32x384x14x14xf32>
    %v5505 = stablehlo.subtract %v5499, %v5504 : tensor<32x384x14x14xf32>
    %v5506 = stablehlo.multiply %v5505, %v5505 : tensor<32x384x14x14xf32>
    %v5507 = stablehlo.reduce(%v5506 init: %v5500) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5508 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5509 = stablehlo.divide %v5507, %v5508 : tensor<384xf32>
    %v5510 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5512 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5513 = stablehlo.reduce(%v5510 init: %v5511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5514 = stablehlo.divide %v5513, %v5512 : tensor<384xf32>
    %v5515 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5517 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5518 = stablehlo.reduce(%v5515 init: %v5516) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5519 = stablehlo.broadcast_in_dim %v5518, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5520 = stablehlo.divide %v5519, %v5517 : tensor<32x384x14x14xf32>
    %v5521 = stablehlo.subtract %v5515, %v5520 : tensor<32x384x14x14xf32>
    %v5522 = stablehlo.multiply %v5521, %v5521 : tensor<32x384x14x14xf32>
    %v5523 = stablehlo.reduce(%v5522 init: %v5516) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5524 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5525 = stablehlo.divide %v5523, %v5524 : tensor<384xf32>
    %v5526 = stablehlo.reshape %v687 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5528 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5529 = stablehlo.reduce(%v5526 init: %v5527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5530 = stablehlo.divide %v5529, %v5528 : tensor<64xf32>
    %v5531 = stablehlo.reshape %v687 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5532 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5533 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v5534 = stablehlo.reduce(%v5531 init: %v5532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5535 = stablehlo.broadcast_in_dim %v5534, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v5536 = stablehlo.divide %v5535, %v5533 : tensor<32x64x14x14xf32>
    %v5537 = stablehlo.subtract %v5531, %v5536 : tensor<32x64x14x14xf32>
    %v5538 = stablehlo.multiply %v5537, %v5537 : tensor<32x64x14x14xf32>
    %v5539 = stablehlo.reduce(%v5538 init: %v5532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5540 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5541 = stablehlo.divide %v5539, %v5540 : tensor<64xf32>
    %v5542 = stablehlo.reshape %v716 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5543 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5544 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5545 = stablehlo.reduce(%v5542 init: %v5543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5546 = stablehlo.divide %v5545, %v5544 : tensor<384xf32>
    %v5547 = stablehlo.reshape %v716 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5548 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5549 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5550 = stablehlo.reduce(%v5547 init: %v5548) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5551 = stablehlo.broadcast_in_dim %v5550, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5552 = stablehlo.divide %v5551, %v5549 : tensor<32x384x14x14xf32>
    %v5553 = stablehlo.subtract %v5547, %v5552 : tensor<32x384x14x14xf32>
    %v5554 = stablehlo.multiply %v5553, %v5553 : tensor<32x384x14x14xf32>
    %v5555 = stablehlo.reduce(%v5554 init: %v5548) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5556 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5557 = stablehlo.divide %v5555, %v5556 : tensor<384xf32>
    %v5558 = stablehlo.reshape %v747 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5559 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5560 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5561 = stablehlo.reduce(%v5558 init: %v5559) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5562 = stablehlo.divide %v5561, %v5560 : tensor<384xf32>
    %v5563 = stablehlo.reshape %v747 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5565 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5566 = stablehlo.reduce(%v5563 init: %v5564) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5567 = stablehlo.broadcast_in_dim %v5566, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5568 = stablehlo.divide %v5567, %v5565 : tensor<32x384x14x14xf32>
    %v5569 = stablehlo.subtract %v5563, %v5568 : tensor<32x384x14x14xf32>
    %v5570 = stablehlo.multiply %v5569, %v5569 : tensor<32x384x14x14xf32>
    %v5571 = stablehlo.reduce(%v5570 init: %v5564) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5572 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5573 = stablehlo.divide %v5571, %v5572 : tensor<384xf32>
    %v5574 = stablehlo.reshape %v778 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5575 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5576 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5577 = stablehlo.reduce(%v5574 init: %v5575) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5578 = stablehlo.divide %v5577, %v5576 : tensor<64xf32>
    %v5579 = stablehlo.reshape %v778 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5580 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5581 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v5582 = stablehlo.reduce(%v5579 init: %v5580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5583 = stablehlo.broadcast_in_dim %v5582, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v5584 = stablehlo.divide %v5583, %v5581 : tensor<32x64x14x14xf32>
    %v5585 = stablehlo.subtract %v5579, %v5584 : tensor<32x64x14x14xf32>
    %v5586 = stablehlo.multiply %v5585, %v5585 : tensor<32x64x14x14xf32>
    %v5587 = stablehlo.reduce(%v5586 init: %v5580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5588 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5589 = stablehlo.divide %v5587, %v5588 : tensor<64xf32>
    %v5590 = stablehlo.reshape %v807 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5591 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5592 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5593 = stablehlo.reduce(%v5590 init: %v5591) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5594 = stablehlo.divide %v5593, %v5592 : tensor<384xf32>
    %v5595 = stablehlo.reshape %v807 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5596 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5597 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5598 = stablehlo.reduce(%v5595 init: %v5596) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5599 = stablehlo.broadcast_in_dim %v5598, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5600 = stablehlo.divide %v5599, %v5597 : tensor<32x384x14x14xf32>
    %v5601 = stablehlo.subtract %v5595, %v5600 : tensor<32x384x14x14xf32>
    %v5602 = stablehlo.multiply %v5601, %v5601 : tensor<32x384x14x14xf32>
    %v5603 = stablehlo.reduce(%v5602 init: %v5596) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5604 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5605 = stablehlo.divide %v5603, %v5604 : tensor<384xf32>
    %v5606 = stablehlo.reshape %v838 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5608 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5609 = stablehlo.reduce(%v5606 init: %v5607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5610 = stablehlo.divide %v5609, %v5608 : tensor<384xf32>
    %v5611 = stablehlo.reshape %v838 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5612 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5613 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5614 = stablehlo.reduce(%v5611 init: %v5612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5615 = stablehlo.broadcast_in_dim %v5614, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5616 = stablehlo.divide %v5615, %v5613 : tensor<32x384x14x14xf32>
    %v5617 = stablehlo.subtract %v5611, %v5616 : tensor<32x384x14x14xf32>
    %v5618 = stablehlo.multiply %v5617, %v5617 : tensor<32x384x14x14xf32>
    %v5619 = stablehlo.reduce(%v5618 init: %v5612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5620 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5621 = stablehlo.divide %v5619, %v5620 : tensor<384xf32>
    %v5622 = stablehlo.reshape %v869 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5623 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5624 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5625 = stablehlo.reduce(%v5622 init: %v5623) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5626 = stablehlo.divide %v5625, %v5624 : tensor<64xf32>
    %v5627 = stablehlo.reshape %v869 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v5628 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5629 = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %v5630 = stablehlo.reduce(%v5627 init: %v5628) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5631 = stablehlo.broadcast_in_dim %v5630, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v5632 = stablehlo.divide %v5631, %v5629 : tensor<32x64x14x14xf32>
    %v5633 = stablehlo.subtract %v5627, %v5632 : tensor<32x64x14x14xf32>
    %v5634 = stablehlo.multiply %v5633, %v5633 : tensor<32x64x14x14xf32>
    %v5635 = stablehlo.reduce(%v5634 init: %v5628) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v5636 = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %v5637 = stablehlo.divide %v5635, %v5636 : tensor<64xf32>
    %v5638 = stablehlo.reshape %v898 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5639 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5640 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5641 = stablehlo.reduce(%v5638 init: %v5639) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5642 = stablehlo.divide %v5641, %v5640 : tensor<384xf32>
    %v5643 = stablehlo.reshape %v898 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5644 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5645 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5646 = stablehlo.reduce(%v5643 init: %v5644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5647 = stablehlo.broadcast_in_dim %v5646, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5648 = stablehlo.divide %v5647, %v5645 : tensor<32x384x14x14xf32>
    %v5649 = stablehlo.subtract %v5643, %v5648 : tensor<32x384x14x14xf32>
    %v5650 = stablehlo.multiply %v5649, %v5649 : tensor<32x384x14x14xf32>
    %v5651 = stablehlo.reduce(%v5650 init: %v5644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5652 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5653 = stablehlo.divide %v5651, %v5652 : tensor<384xf32>
    %v5654 = stablehlo.reshape %v929 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5655 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5656 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5657 = stablehlo.reduce(%v5654 init: %v5655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5658 = stablehlo.divide %v5657, %v5656 : tensor<384xf32>
    %v5659 = stablehlo.reshape %v929 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v5660 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5661 = stablehlo.constant dense<6272.0> : tensor<32x384x14x14xf32>
    %v5662 = stablehlo.reduce(%v5659 init: %v5660) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5663 = stablehlo.broadcast_in_dim %v5662, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v5664 = stablehlo.divide %v5663, %v5661 : tensor<32x384x14x14xf32>
    %v5665 = stablehlo.subtract %v5659, %v5664 : tensor<32x384x14x14xf32>
    %v5666 = stablehlo.multiply %v5665, %v5665 : tensor<32x384x14x14xf32>
    %v5667 = stablehlo.reduce(%v5666 init: %v5660) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v5668 = stablehlo.constant dense<6272.0> : tensor<384xf32>
    %v5669 = stablehlo.divide %v5667, %v5668 : tensor<384xf32>
    %v5670 = stablehlo.reshape %v960 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5672 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5673 = stablehlo.reduce(%v5670 init: %v5671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5674 = stablehlo.divide %v5673, %v5672 : tensor<96xf32>
    %v5675 = stablehlo.reshape %v960 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5677 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v5678 = stablehlo.reduce(%v5675 init: %v5676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5679 = stablehlo.broadcast_in_dim %v5678, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v5680 = stablehlo.divide %v5679, %v5677 : tensor<32x96x14x14xf32>
    %v5681 = stablehlo.subtract %v5675, %v5680 : tensor<32x96x14x14xf32>
    %v5682 = stablehlo.multiply %v5681, %v5681 : tensor<32x96x14x14xf32>
    %v5683 = stablehlo.reduce(%v5682 init: %v5676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5684 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5685 = stablehlo.divide %v5683, %v5684 : tensor<96xf32>
    %v5686 = stablehlo.reshape %v985 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5687 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5688 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5689 = stablehlo.reduce(%v5686 init: %v5687) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5690 = stablehlo.divide %v5689, %v5688 : tensor<576xf32>
    %v5691 = stablehlo.reshape %v985 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5693 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v5694 = stablehlo.reduce(%v5691 init: %v5692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5695 = stablehlo.broadcast_in_dim %v5694, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v5696 = stablehlo.divide %v5695, %v5693 : tensor<32x576x14x14xf32>
    %v5697 = stablehlo.subtract %v5691, %v5696 : tensor<32x576x14x14xf32>
    %v5698 = stablehlo.multiply %v5697, %v5697 : tensor<32x576x14x14xf32>
    %v5699 = stablehlo.reduce(%v5698 init: %v5692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5700 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5701 = stablehlo.divide %v5699, %v5700 : tensor<576xf32>
    %v5702 = stablehlo.reshape %v1016 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5703 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5704 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5705 = stablehlo.reduce(%v5702 init: %v5703) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5706 = stablehlo.divide %v5705, %v5704 : tensor<576xf32>
    %v5707 = stablehlo.reshape %v1016 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5709 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v5710 = stablehlo.reduce(%v5707 init: %v5708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5711 = stablehlo.broadcast_in_dim %v5710, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v5712 = stablehlo.divide %v5711, %v5709 : tensor<32x576x14x14xf32>
    %v5713 = stablehlo.subtract %v5707, %v5712 : tensor<32x576x14x14xf32>
    %v5714 = stablehlo.multiply %v5713, %v5713 : tensor<32x576x14x14xf32>
    %v5715 = stablehlo.reduce(%v5714 init: %v5708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5716 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5717 = stablehlo.divide %v5715, %v5716 : tensor<576xf32>
    %v5718 = stablehlo.reshape %v1047 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5720 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5721 = stablehlo.reduce(%v5718 init: %v5719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5722 = stablehlo.divide %v5721, %v5720 : tensor<96xf32>
    %v5723 = stablehlo.reshape %v1047 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5725 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v5726 = stablehlo.reduce(%v5723 init: %v5724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5727 = stablehlo.broadcast_in_dim %v5726, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v5728 = stablehlo.divide %v5727, %v5725 : tensor<32x96x14x14xf32>
    %v5729 = stablehlo.subtract %v5723, %v5728 : tensor<32x96x14x14xf32>
    %v5730 = stablehlo.multiply %v5729, %v5729 : tensor<32x96x14x14xf32>
    %v5731 = stablehlo.reduce(%v5730 init: %v5724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5732 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5733 = stablehlo.divide %v5731, %v5732 : tensor<96xf32>
    %v5734 = stablehlo.reshape %v1076 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5736 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5737 = stablehlo.reduce(%v5734 init: %v5735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5738 = stablehlo.divide %v5737, %v5736 : tensor<576xf32>
    %v5739 = stablehlo.reshape %v1076 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5740 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5741 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v5742 = stablehlo.reduce(%v5739 init: %v5740) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5743 = stablehlo.broadcast_in_dim %v5742, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v5744 = stablehlo.divide %v5743, %v5741 : tensor<32x576x14x14xf32>
    %v5745 = stablehlo.subtract %v5739, %v5744 : tensor<32x576x14x14xf32>
    %v5746 = stablehlo.multiply %v5745, %v5745 : tensor<32x576x14x14xf32>
    %v5747 = stablehlo.reduce(%v5746 init: %v5740) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5748 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5749 = stablehlo.divide %v5747, %v5748 : tensor<576xf32>
    %v5750 = stablehlo.reshape %v1107 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5751 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5752 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5753 = stablehlo.reduce(%v5750 init: %v5751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5754 = stablehlo.divide %v5753, %v5752 : tensor<576xf32>
    %v5755 = stablehlo.reshape %v1107 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5757 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v5758 = stablehlo.reduce(%v5755 init: %v5756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5759 = stablehlo.broadcast_in_dim %v5758, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v5760 = stablehlo.divide %v5759, %v5757 : tensor<32x576x14x14xf32>
    %v5761 = stablehlo.subtract %v5755, %v5760 : tensor<32x576x14x14xf32>
    %v5762 = stablehlo.multiply %v5761, %v5761 : tensor<32x576x14x14xf32>
    %v5763 = stablehlo.reduce(%v5762 init: %v5756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5764 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5765 = stablehlo.divide %v5763, %v5764 : tensor<576xf32>
    %v5766 = stablehlo.reshape %v1138 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5768 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5769 = stablehlo.reduce(%v5766 init: %v5767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5770 = stablehlo.divide %v5769, %v5768 : tensor<96xf32>
    %v5771 = stablehlo.reshape %v1138 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v5772 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5773 = stablehlo.constant dense<6272.0> : tensor<32x96x14x14xf32>
    %v5774 = stablehlo.reduce(%v5771 init: %v5772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5775 = stablehlo.broadcast_in_dim %v5774, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v5776 = stablehlo.divide %v5775, %v5773 : tensor<32x96x14x14xf32>
    %v5777 = stablehlo.subtract %v5771, %v5776 : tensor<32x96x14x14xf32>
    %v5778 = stablehlo.multiply %v5777, %v5777 : tensor<32x96x14x14xf32>
    %v5779 = stablehlo.reduce(%v5778 init: %v5772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v5780 = stablehlo.constant dense<6272.0> : tensor<96xf32>
    %v5781 = stablehlo.divide %v5779, %v5780 : tensor<96xf32>
    %v5782 = stablehlo.reshape %v1167 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5784 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5785 = stablehlo.reduce(%v5782 init: %v5783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5786 = stablehlo.divide %v5785, %v5784 : tensor<576xf32>
    %v5787 = stablehlo.reshape %v1167 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v5788 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5789 = stablehlo.constant dense<6272.0> : tensor<32x576x14x14xf32>
    %v5790 = stablehlo.reduce(%v5787 init: %v5788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5791 = stablehlo.broadcast_in_dim %v5790, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v5792 = stablehlo.divide %v5791, %v5789 : tensor<32x576x14x14xf32>
    %v5793 = stablehlo.subtract %v5787, %v5792 : tensor<32x576x14x14xf32>
    %v5794 = stablehlo.multiply %v5793, %v5793 : tensor<32x576x14x14xf32>
    %v5795 = stablehlo.reduce(%v5794 init: %v5788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v5796 = stablehlo.constant dense<6272.0> : tensor<576xf32>
    %v5797 = stablehlo.divide %v5795, %v5796 : tensor<576xf32>
    %v5798 = stablehlo.reshape %v1198 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v5799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5800 = stablehlo.constant dense<1568.0> : tensor<576xf32>
    %v5801 = stablehlo.reduce(%v5798 init: %v5799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v5802 = stablehlo.divide %v5801, %v5800 : tensor<576xf32>
    %v5803 = stablehlo.reshape %v1198 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v5804 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5805 = stablehlo.constant dense<1568.0> : tensor<32x576x7x7xf32>
    %v5806 = stablehlo.reduce(%v5803 init: %v5804) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v5807 = stablehlo.broadcast_in_dim %v5806, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v5808 = stablehlo.divide %v5807, %v5805 : tensor<32x576x7x7xf32>
    %v5809 = stablehlo.subtract %v5803, %v5808 : tensor<32x576x7x7xf32>
    %v5810 = stablehlo.multiply %v5809, %v5809 : tensor<32x576x7x7xf32>
    %v5811 = stablehlo.reduce(%v5810 init: %v5804) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v5812 = stablehlo.constant dense<1568.0> : tensor<576xf32>
    %v5813 = stablehlo.divide %v5811, %v5812 : tensor<576xf32>
    %v5814 = stablehlo.reshape %v1229 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5815 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5816 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5817 = stablehlo.reduce(%v5814 init: %v5815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5818 = stablehlo.divide %v5817, %v5816 : tensor<160xf32>
    %v5819 = stablehlo.reshape %v1229 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5820 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5821 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v5822 = stablehlo.reduce(%v5819 init: %v5820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5823 = stablehlo.broadcast_in_dim %v5822, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v5824 = stablehlo.divide %v5823, %v5821 : tensor<32x160x7x7xf32>
    %v5825 = stablehlo.subtract %v5819, %v5824 : tensor<32x160x7x7xf32>
    %v5826 = stablehlo.multiply %v5825, %v5825 : tensor<32x160x7x7xf32>
    %v5827 = stablehlo.reduce(%v5826 init: %v5820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5828 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5829 = stablehlo.divide %v5827, %v5828 : tensor<160xf32>
    %v5830 = stablehlo.reshape %v1254 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5831 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5832 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5833 = stablehlo.reduce(%v5830 init: %v5831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5834 = stablehlo.divide %v5833, %v5832 : tensor<960xf32>
    %v5835 = stablehlo.reshape %v1254 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5836 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5837 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5838 = stablehlo.reduce(%v5835 init: %v5836) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5839 = stablehlo.broadcast_in_dim %v5838, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5840 = stablehlo.divide %v5839, %v5837 : tensor<32x960x7x7xf32>
    %v5841 = stablehlo.subtract %v5835, %v5840 : tensor<32x960x7x7xf32>
    %v5842 = stablehlo.multiply %v5841, %v5841 : tensor<32x960x7x7xf32>
    %v5843 = stablehlo.reduce(%v5842 init: %v5836) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5844 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5845 = stablehlo.divide %v5843, %v5844 : tensor<960xf32>
    %v5846 = stablehlo.reshape %v1285 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5847 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5848 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5849 = stablehlo.reduce(%v5846 init: %v5847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5850 = stablehlo.divide %v5849, %v5848 : tensor<960xf32>
    %v5851 = stablehlo.reshape %v1285 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5852 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5853 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5854 = stablehlo.reduce(%v5851 init: %v5852) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5855 = stablehlo.broadcast_in_dim %v5854, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5856 = stablehlo.divide %v5855, %v5853 : tensor<32x960x7x7xf32>
    %v5857 = stablehlo.subtract %v5851, %v5856 : tensor<32x960x7x7xf32>
    %v5858 = stablehlo.multiply %v5857, %v5857 : tensor<32x960x7x7xf32>
    %v5859 = stablehlo.reduce(%v5858 init: %v5852) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5860 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5861 = stablehlo.divide %v5859, %v5860 : tensor<960xf32>
    %v5862 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5864 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5865 = stablehlo.reduce(%v5862 init: %v5863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5866 = stablehlo.divide %v5865, %v5864 : tensor<160xf32>
    %v5867 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5868 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5869 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v5870 = stablehlo.reduce(%v5867 init: %v5868) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5871 = stablehlo.broadcast_in_dim %v5870, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v5872 = stablehlo.divide %v5871, %v5869 : tensor<32x160x7x7xf32>
    %v5873 = stablehlo.subtract %v5867, %v5872 : tensor<32x160x7x7xf32>
    %v5874 = stablehlo.multiply %v5873, %v5873 : tensor<32x160x7x7xf32>
    %v5875 = stablehlo.reduce(%v5874 init: %v5868) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5876 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5877 = stablehlo.divide %v5875, %v5876 : tensor<160xf32>
    %v5878 = stablehlo.reshape %v1345 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5879 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5880 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5881 = stablehlo.reduce(%v5878 init: %v5879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5882 = stablehlo.divide %v5881, %v5880 : tensor<960xf32>
    %v5883 = stablehlo.reshape %v1345 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5884 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5885 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5886 = stablehlo.reduce(%v5883 init: %v5884) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5887 = stablehlo.broadcast_in_dim %v5886, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5888 = stablehlo.divide %v5887, %v5885 : tensor<32x960x7x7xf32>
    %v5889 = stablehlo.subtract %v5883, %v5888 : tensor<32x960x7x7xf32>
    %v5890 = stablehlo.multiply %v5889, %v5889 : tensor<32x960x7x7xf32>
    %v5891 = stablehlo.reduce(%v5890 init: %v5884) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5892 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5893 = stablehlo.divide %v5891, %v5892 : tensor<960xf32>
    %v5894 = stablehlo.reshape %v1376 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5896 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5897 = stablehlo.reduce(%v5894 init: %v5895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5898 = stablehlo.divide %v5897, %v5896 : tensor<960xf32>
    %v5899 = stablehlo.reshape %v1376 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5901 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5902 = stablehlo.reduce(%v5899 init: %v5900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5903 = stablehlo.broadcast_in_dim %v5902, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5904 = stablehlo.divide %v5903, %v5901 : tensor<32x960x7x7xf32>
    %v5905 = stablehlo.subtract %v5899, %v5904 : tensor<32x960x7x7xf32>
    %v5906 = stablehlo.multiply %v5905, %v5905 : tensor<32x960x7x7xf32>
    %v5907 = stablehlo.reduce(%v5906 init: %v5900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5908 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5909 = stablehlo.divide %v5907, %v5908 : tensor<960xf32>
    %v5910 = stablehlo.reshape %v1407 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5912 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5913 = stablehlo.reduce(%v5910 init: %v5911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5914 = stablehlo.divide %v5913, %v5912 : tensor<160xf32>
    %v5915 = stablehlo.reshape %v1407 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v5916 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5917 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v5918 = stablehlo.reduce(%v5915 init: %v5916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5919 = stablehlo.broadcast_in_dim %v5918, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v5920 = stablehlo.divide %v5919, %v5917 : tensor<32x160x7x7xf32>
    %v5921 = stablehlo.subtract %v5915, %v5920 : tensor<32x160x7x7xf32>
    %v5922 = stablehlo.multiply %v5921, %v5921 : tensor<32x160x7x7xf32>
    %v5923 = stablehlo.reduce(%v5922 init: %v5916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v5924 = stablehlo.constant dense<1568.0> : tensor<160xf32>
    %v5925 = stablehlo.divide %v5923, %v5924 : tensor<160xf32>
    %v5926 = stablehlo.reshape %v1436 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5928 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5929 = stablehlo.reduce(%v5926 init: %v5927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5930 = stablehlo.divide %v5929, %v5928 : tensor<960xf32>
    %v5931 = stablehlo.reshape %v1436 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5933 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5934 = stablehlo.reduce(%v5931 init: %v5932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5935 = stablehlo.broadcast_in_dim %v5934, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5936 = stablehlo.divide %v5935, %v5933 : tensor<32x960x7x7xf32>
    %v5937 = stablehlo.subtract %v5931, %v5936 : tensor<32x960x7x7xf32>
    %v5938 = stablehlo.multiply %v5937, %v5937 : tensor<32x960x7x7xf32>
    %v5939 = stablehlo.reduce(%v5938 init: %v5932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5940 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5941 = stablehlo.divide %v5939, %v5940 : tensor<960xf32>
    %v5942 = stablehlo.reshape %v1467 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5943 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5944 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5945 = stablehlo.reduce(%v5942 init: %v5943) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5946 = stablehlo.divide %v5945, %v5944 : tensor<960xf32>
    %v5947 = stablehlo.reshape %v1467 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v5948 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5949 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v5950 = stablehlo.reduce(%v5947 init: %v5948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5951 = stablehlo.broadcast_in_dim %v5950, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v5952 = stablehlo.divide %v5951, %v5949 : tensor<32x960x7x7xf32>
    %v5953 = stablehlo.subtract %v5947, %v5952 : tensor<32x960x7x7xf32>
    %v5954 = stablehlo.multiply %v5953, %v5953 : tensor<32x960x7x7xf32>
    %v5955 = stablehlo.reduce(%v5954 init: %v5948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v5956 = stablehlo.constant dense<1568.0> : tensor<960xf32>
    %v5957 = stablehlo.divide %v5955, %v5956 : tensor<960xf32>
    %v5958 = stablehlo.reshape %v1498 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v5959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5960 = stablehlo.constant dense<1568.0> : tensor<320xf32>
    %v5961 = stablehlo.reduce(%v5958 init: %v5959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v5962 = stablehlo.divide %v5961, %v5960 : tensor<320xf32>
    %v5963 = stablehlo.reshape %v1498 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v5964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5965 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v5966 = stablehlo.reduce(%v5963 init: %v5964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v5967 = stablehlo.broadcast_in_dim %v5966, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v5968 = stablehlo.divide %v5967, %v5965 : tensor<32x320x7x7xf32>
    %v5969 = stablehlo.subtract %v5963, %v5968 : tensor<32x320x7x7xf32>
    %v5970 = stablehlo.multiply %v5969, %v5969 : tensor<32x320x7x7xf32>
    %v5971 = stablehlo.reduce(%v5970 init: %v5964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v5972 = stablehlo.constant dense<1568.0> : tensor<320xf32>
    %v5973 = stablehlo.divide %v5971, %v5972 : tensor<320xf32>
    %v5974 = stablehlo.reshape %v1523 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v5975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5976 = stablehlo.constant dense<1568.0> : tensor<1280xf32>
    %v5977 = stablehlo.reduce(%v5974 init: %v5975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v5978 = stablehlo.divide %v5977, %v5976 : tensor<1280xf32>
    %v5979 = stablehlo.reshape %v1523 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v5980 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5981 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v5982 = stablehlo.reduce(%v5979 init: %v5980) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v5983 = stablehlo.broadcast_in_dim %v5982, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v5984 = stablehlo.divide %v5983, %v5981 : tensor<32x1280x7x7xf32>
    %v5985 = stablehlo.subtract %v5979, %v5984 : tensor<32x1280x7x7xf32>
    %v5986 = stablehlo.multiply %v5985, %v5985 : tensor<32x1280x7x7xf32>
    %v5987 = stablehlo.reduce(%v5986 init: %v5980) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v5988 = stablehlo.constant dense<1568.0> : tensor<1280xf32>
    %v5989 = stablehlo.divide %v5987, %v5988 : tensor<1280xf32>
    %rho = stablehlo.constant dense<0.900000> : tensor<f32>
    %orho = stablehlo.constant dense<0.100000> : tensor<f32>
    %mu = stablehlo.constant dense<0.900000> : tensor<f32>
    %eps = stablehlo.constant dense<1.000000> : tensor<f32>
    %wd = stablehlo.constant dense<0.000040> : tensor<f32>
    %v5990 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5991 = stablehlo.multiply %v5990, %sW : tensor<32x3x3x3xf32>
    %v5992 = stablehlo.add %v5991, %v5136 : tensor<32x3x3x3xf32>
    %v5993 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5994 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v5995 = stablehlo.multiply %v5993, %sWv : tensor<32x3x3x3xf32>
    %v5996 = stablehlo.multiply %v5992, %v5992 : tensor<32x3x3x3xf32>
    %v5997 = stablehlo.multiply %v5994, %v5996 : tensor<32x3x3x3xf32>
    %v5998 = stablehlo.add %v5995, %v5997 : tensor<32x3x3x3xf32>
    %v5999 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v6000 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v6001 = stablehlo.multiply %v5999, %sWv : tensor<32x3x3x3xf32>
    %v6002 = stablehlo.multiply %v5992, %v5992 : tensor<32x3x3x3xf32>
    %v6003 = stablehlo.multiply %v6000, %v6002 : tensor<32x3x3x3xf32>
    %v6004 = stablehlo.add %v6001, %v6003 : tensor<32x3x3x3xf32>
    %v6005 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v6006 = stablehlo.add %v6004, %v6005 : tensor<32x3x3x3xf32>
    %v6007 = stablehlo.sqrt %v6006 : tensor<32x3x3x3xf32>
    %v6008 = stablehlo.divide %v5992, %v6007 : tensor<32x3x3x3xf32>
    %v6009 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v6010 = stablehlo.multiply %v6009, %sWm : tensor<32x3x3x3xf32>
    %v6011 = stablehlo.add %v6010, %v6008 : tensor<32x3x3x3xf32>
    %v6012 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x3x3x3xf32>
    %v6013 = stablehlo.multiply %v6012, %v6011 : tensor<32x3x3x3xf32>
    %v6014 = stablehlo.subtract %sW, %v6013 : tensor<32x3x3x3xf32>
    %v6015 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6016 = stablehlo.multiply %v6015, %sg : tensor<32xf32>
    %v6017 = stablehlo.add %v6016, %v5154 : tensor<32xf32>
    %v6018 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6019 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6020 = stablehlo.multiply %v6018, %sgv : tensor<32xf32>
    %v6021 = stablehlo.multiply %v6017, %v6017 : tensor<32xf32>
    %v6022 = stablehlo.multiply %v6019, %v6021 : tensor<32xf32>
    %v6023 = stablehlo.add %v6020, %v6022 : tensor<32xf32>
    %v6024 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6025 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6026 = stablehlo.multiply %v6024, %sgv : tensor<32xf32>
    %v6027 = stablehlo.multiply %v6017, %v6017 : tensor<32xf32>
    %v6028 = stablehlo.multiply %v6025, %v6027 : tensor<32xf32>
    %v6029 = stablehlo.add %v6026, %v6028 : tensor<32xf32>
    %v6030 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6031 = stablehlo.add %v6029, %v6030 : tensor<32xf32>
    %v6032 = stablehlo.sqrt %v6031 : tensor<32xf32>
    %v6033 = stablehlo.divide %v6017, %v6032 : tensor<32xf32>
    %v6034 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6035 = stablehlo.multiply %v6034, %sgm : tensor<32xf32>
    %v6036 = stablehlo.add %v6035, %v6033 : tensor<32xf32>
    %v6037 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6038 = stablehlo.multiply %v6037, %v6036 : tensor<32xf32>
    %v6039 = stablehlo.subtract %sg, %v6038 : tensor<32xf32>
    %v6040 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6041 = stablehlo.multiply %v6040, %sbt : tensor<32xf32>
    %v6042 = stablehlo.add %v6041, %v5157 : tensor<32xf32>
    %v6043 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6044 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6045 = stablehlo.multiply %v6043, %sbtv : tensor<32xf32>
    %v6046 = stablehlo.multiply %v6042, %v6042 : tensor<32xf32>
    %v6047 = stablehlo.multiply %v6044, %v6046 : tensor<32xf32>
    %v6048 = stablehlo.add %v6045, %v6047 : tensor<32xf32>
    %v6049 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6050 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6051 = stablehlo.multiply %v6049, %sbtv : tensor<32xf32>
    %v6052 = stablehlo.multiply %v6042, %v6042 : tensor<32xf32>
    %v6053 = stablehlo.multiply %v6050, %v6052 : tensor<32xf32>
    %v6054 = stablehlo.add %v6051, %v6053 : tensor<32xf32>
    %v6055 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6056 = stablehlo.add %v6054, %v6055 : tensor<32xf32>
    %v6057 = stablehlo.sqrt %v6056 : tensor<32xf32>
    %v6058 = stablehlo.divide %v6042, %v6057 : tensor<32xf32>
    %v6059 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6060 = stablehlo.multiply %v6059, %sbtm : tensor<32xf32>
    %v6061 = stablehlo.add %v6060, %v6058 : tensor<32xf32>
    %v6062 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6063 = stablehlo.multiply %v6062, %v6061 : tensor<32xf32>
    %v6064 = stablehlo.subtract %sbt, %v6063 : tensor<32xf32>
    %v6065 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6066 = stablehlo.multiply %v6065, %b1dW : tensor<32x1x3x3xf32>
    %v6067 = stablehlo.add %v6066, %v5041 : tensor<32x1x3x3xf32>
    %v6068 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6069 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6070 = stablehlo.multiply %v6068, %b1dWv : tensor<32x1x3x3xf32>
    %v6071 = stablehlo.multiply %v6067, %v6067 : tensor<32x1x3x3xf32>
    %v6072 = stablehlo.multiply %v6069, %v6071 : tensor<32x1x3x3xf32>
    %v6073 = stablehlo.add %v6070, %v6072 : tensor<32x1x3x3xf32>
    %v6074 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6075 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6076 = stablehlo.multiply %v6074, %b1dWv : tensor<32x1x3x3xf32>
    %v6077 = stablehlo.multiply %v6067, %v6067 : tensor<32x1x3x3xf32>
    %v6078 = stablehlo.multiply %v6075, %v6077 : tensor<32x1x3x3xf32>
    %v6079 = stablehlo.add %v6076, %v6078 : tensor<32x1x3x3xf32>
    %v6080 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6081 = stablehlo.add %v6079, %v6080 : tensor<32x1x3x3xf32>
    %v6082 = stablehlo.sqrt %v6081 : tensor<32x1x3x3xf32>
    %v6083 = stablehlo.divide %v6067, %v6082 : tensor<32x1x3x3xf32>
    %v6084 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6085 = stablehlo.multiply %v6084, %b1dWm : tensor<32x1x3x3xf32>
    %v6086 = stablehlo.add %v6085, %v6083 : tensor<32x1x3x3xf32>
    %v6087 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %v6088 = stablehlo.multiply %v6087, %v6086 : tensor<32x1x3x3xf32>
    %v6089 = stablehlo.subtract %b1dW, %v6088 : tensor<32x1x3x3xf32>
    %v6090 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6091 = stablehlo.multiply %v6090, %b1dg : tensor<32xf32>
    %v6092 = stablehlo.add %v6091, %v5059 : tensor<32xf32>
    %v6093 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6094 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6095 = stablehlo.multiply %v6093, %b1dgv : tensor<32xf32>
    %v6096 = stablehlo.multiply %v6092, %v6092 : tensor<32xf32>
    %v6097 = stablehlo.multiply %v6094, %v6096 : tensor<32xf32>
    %v6098 = stablehlo.add %v6095, %v6097 : tensor<32xf32>
    %v6099 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6100 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6101 = stablehlo.multiply %v6099, %b1dgv : tensor<32xf32>
    %v6102 = stablehlo.multiply %v6092, %v6092 : tensor<32xf32>
    %v6103 = stablehlo.multiply %v6100, %v6102 : tensor<32xf32>
    %v6104 = stablehlo.add %v6101, %v6103 : tensor<32xf32>
    %v6105 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6106 = stablehlo.add %v6104, %v6105 : tensor<32xf32>
    %v6107 = stablehlo.sqrt %v6106 : tensor<32xf32>
    %v6108 = stablehlo.divide %v6092, %v6107 : tensor<32xf32>
    %v6109 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6110 = stablehlo.multiply %v6109, %b1dgm : tensor<32xf32>
    %v6111 = stablehlo.add %v6110, %v6108 : tensor<32xf32>
    %v6112 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6113 = stablehlo.multiply %v6112, %v6111 : tensor<32xf32>
    %v6114 = stablehlo.subtract %b1dg, %v6113 : tensor<32xf32>
    %v6115 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6116 = stablehlo.multiply %v6115, %b1dbt : tensor<32xf32>
    %v6117 = stablehlo.add %v6116, %v5062 : tensor<32xf32>
    %v6118 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6119 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6120 = stablehlo.multiply %v6118, %b1dbtv : tensor<32xf32>
    %v6121 = stablehlo.multiply %v6117, %v6117 : tensor<32xf32>
    %v6122 = stablehlo.multiply %v6119, %v6121 : tensor<32xf32>
    %v6123 = stablehlo.add %v6120, %v6122 : tensor<32xf32>
    %v6124 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6125 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6126 = stablehlo.multiply %v6124, %b1dbtv : tensor<32xf32>
    %v6127 = stablehlo.multiply %v6117, %v6117 : tensor<32xf32>
    %v6128 = stablehlo.multiply %v6125, %v6127 : tensor<32xf32>
    %v6129 = stablehlo.add %v6126, %v6128 : tensor<32xf32>
    %v6130 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6131 = stablehlo.add %v6129, %v6130 : tensor<32xf32>
    %v6132 = stablehlo.sqrt %v6131 : tensor<32xf32>
    %v6133 = stablehlo.divide %v6117, %v6132 : tensor<32xf32>
    %v6134 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6135 = stablehlo.multiply %v6134, %b1dbtm : tensor<32xf32>
    %v6136 = stablehlo.add %v6135, %v6133 : tensor<32xf32>
    %v6137 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6138 = stablehlo.multiply %v6137, %v6136 : tensor<32xf32>
    %v6139 = stablehlo.subtract %b1dbt, %v6138 : tensor<32xf32>
    %v6140 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6141 = stablehlo.multiply %v6140, %b1pW : tensor<16x32x1x1xf32>
    %v6142 = stablehlo.add %v6141, %v5068 : tensor<16x32x1x1xf32>
    %v6143 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6144 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6145 = stablehlo.multiply %v6143, %b1pWv : tensor<16x32x1x1xf32>
    %v6146 = stablehlo.multiply %v6142, %v6142 : tensor<16x32x1x1xf32>
    %v6147 = stablehlo.multiply %v6144, %v6146 : tensor<16x32x1x1xf32>
    %v6148 = stablehlo.add %v6145, %v6147 : tensor<16x32x1x1xf32>
    %v6149 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6150 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6151 = stablehlo.multiply %v6149, %b1pWv : tensor<16x32x1x1xf32>
    %v6152 = stablehlo.multiply %v6142, %v6142 : tensor<16x32x1x1xf32>
    %v6153 = stablehlo.multiply %v6150, %v6152 : tensor<16x32x1x1xf32>
    %v6154 = stablehlo.add %v6151, %v6153 : tensor<16x32x1x1xf32>
    %v6155 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6156 = stablehlo.add %v6154, %v6155 : tensor<16x32x1x1xf32>
    %v6157 = stablehlo.sqrt %v6156 : tensor<16x32x1x1xf32>
    %v6158 = stablehlo.divide %v6142, %v6157 : tensor<16x32x1x1xf32>
    %v6159 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6160 = stablehlo.multiply %v6159, %b1pWm : tensor<16x32x1x1xf32>
    %v6161 = stablehlo.add %v6160, %v6158 : tensor<16x32x1x1xf32>
    %v6162 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x32x1x1xf32>
    %v6163 = stablehlo.multiply %v6162, %v6161 : tensor<16x32x1x1xf32>
    %v6164 = stablehlo.subtract %b1pW, %v6163 : tensor<16x32x1x1xf32>
    %v6165 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6166 = stablehlo.multiply %v6165, %b1pg : tensor<16xf32>
    %v6167 = stablehlo.add %v6166, %v5086 : tensor<16xf32>
    %v6168 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6169 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6170 = stablehlo.multiply %v6168, %b1pgv : tensor<16xf32>
    %v6171 = stablehlo.multiply %v6167, %v6167 : tensor<16xf32>
    %v6172 = stablehlo.multiply %v6169, %v6171 : tensor<16xf32>
    %v6173 = stablehlo.add %v6170, %v6172 : tensor<16xf32>
    %v6174 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6175 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6176 = stablehlo.multiply %v6174, %b1pgv : tensor<16xf32>
    %v6177 = stablehlo.multiply %v6167, %v6167 : tensor<16xf32>
    %v6178 = stablehlo.multiply %v6175, %v6177 : tensor<16xf32>
    %v6179 = stablehlo.add %v6176, %v6178 : tensor<16xf32>
    %v6180 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6181 = stablehlo.add %v6179, %v6180 : tensor<16xf32>
    %v6182 = stablehlo.sqrt %v6181 : tensor<16xf32>
    %v6183 = stablehlo.divide %v6167, %v6182 : tensor<16xf32>
    %v6184 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6185 = stablehlo.multiply %v6184, %b1pgm : tensor<16xf32>
    %v6186 = stablehlo.add %v6185, %v6183 : tensor<16xf32>
    %v6187 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6188 = stablehlo.multiply %v6187, %v6186 : tensor<16xf32>
    %v6189 = stablehlo.subtract %b1pg, %v6188 : tensor<16xf32>
    %v6190 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6191 = stablehlo.multiply %v6190, %b1pbt : tensor<16xf32>
    %v6192 = stablehlo.add %v6191, %v5089 : tensor<16xf32>
    %v6193 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6194 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6195 = stablehlo.multiply %v6193, %b1pbtv : tensor<16xf32>
    %v6196 = stablehlo.multiply %v6192, %v6192 : tensor<16xf32>
    %v6197 = stablehlo.multiply %v6194, %v6196 : tensor<16xf32>
    %v6198 = stablehlo.add %v6195, %v6197 : tensor<16xf32>
    %v6199 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6200 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6201 = stablehlo.multiply %v6199, %b1pbtv : tensor<16xf32>
    %v6202 = stablehlo.multiply %v6192, %v6192 : tensor<16xf32>
    %v6203 = stablehlo.multiply %v6200, %v6202 : tensor<16xf32>
    %v6204 = stablehlo.add %v6201, %v6203 : tensor<16xf32>
    %v6205 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6206 = stablehlo.add %v6204, %v6205 : tensor<16xf32>
    %v6207 = stablehlo.sqrt %v6206 : tensor<16xf32>
    %v6208 = stablehlo.divide %v6192, %v6207 : tensor<16xf32>
    %v6209 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6210 = stablehlo.multiply %v6209, %b1pbtm : tensor<16xf32>
    %v6211 = stablehlo.add %v6210, %v6208 : tensor<16xf32>
    %v6212 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v6213 = stablehlo.multiply %v6212, %v6211 : tensor<16xf32>
    %v6214 = stablehlo.subtract %b1pbt, %v6213 : tensor<16xf32>
    %v6215 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6216 = stablehlo.multiply %v6215, %b2eW : tensor<96x16x1x1xf32>
    %v6217 = stablehlo.add %v6216, %v4880 : tensor<96x16x1x1xf32>
    %v6218 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6219 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6220 = stablehlo.multiply %v6218, %b2eWv : tensor<96x16x1x1xf32>
    %v6221 = stablehlo.multiply %v6217, %v6217 : tensor<96x16x1x1xf32>
    %v6222 = stablehlo.multiply %v6219, %v6221 : tensor<96x16x1x1xf32>
    %v6223 = stablehlo.add %v6220, %v6222 : tensor<96x16x1x1xf32>
    %v6224 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6225 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6226 = stablehlo.multiply %v6224, %b2eWv : tensor<96x16x1x1xf32>
    %v6227 = stablehlo.multiply %v6217, %v6217 : tensor<96x16x1x1xf32>
    %v6228 = stablehlo.multiply %v6225, %v6227 : tensor<96x16x1x1xf32>
    %v6229 = stablehlo.add %v6226, %v6228 : tensor<96x16x1x1xf32>
    %v6230 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6231 = stablehlo.add %v6229, %v6230 : tensor<96x16x1x1xf32>
    %v6232 = stablehlo.sqrt %v6231 : tensor<96x16x1x1xf32>
    %v6233 = stablehlo.divide %v6217, %v6232 : tensor<96x16x1x1xf32>
    %v6234 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6235 = stablehlo.multiply %v6234, %b2eWm : tensor<96x16x1x1xf32>
    %v6236 = stablehlo.add %v6235, %v6233 : tensor<96x16x1x1xf32>
    %v6237 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x16x1x1xf32>
    %v6238 = stablehlo.multiply %v6237, %v6236 : tensor<96x16x1x1xf32>
    %v6239 = stablehlo.subtract %b2eW, %v6238 : tensor<96x16x1x1xf32>
    %v6240 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6241 = stablehlo.multiply %v6240, %b2eg : tensor<96xf32>
    %v6242 = stablehlo.add %v6241, %v4898 : tensor<96xf32>
    %v6243 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6244 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6245 = stablehlo.multiply %v6243, %b2egv : tensor<96xf32>
    %v6246 = stablehlo.multiply %v6242, %v6242 : tensor<96xf32>
    %v6247 = stablehlo.multiply %v6244, %v6246 : tensor<96xf32>
    %v6248 = stablehlo.add %v6245, %v6247 : tensor<96xf32>
    %v6249 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6250 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6251 = stablehlo.multiply %v6249, %b2egv : tensor<96xf32>
    %v6252 = stablehlo.multiply %v6242, %v6242 : tensor<96xf32>
    %v6253 = stablehlo.multiply %v6250, %v6252 : tensor<96xf32>
    %v6254 = stablehlo.add %v6251, %v6253 : tensor<96xf32>
    %v6255 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6256 = stablehlo.add %v6254, %v6255 : tensor<96xf32>
    %v6257 = stablehlo.sqrt %v6256 : tensor<96xf32>
    %v6258 = stablehlo.divide %v6242, %v6257 : tensor<96xf32>
    %v6259 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6260 = stablehlo.multiply %v6259, %b2egm : tensor<96xf32>
    %v6261 = stablehlo.add %v6260, %v6258 : tensor<96xf32>
    %v6262 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6263 = stablehlo.multiply %v6262, %v6261 : tensor<96xf32>
    %v6264 = stablehlo.subtract %b2eg, %v6263 : tensor<96xf32>
    %v6265 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6266 = stablehlo.multiply %v6265, %b2ebt : tensor<96xf32>
    %v6267 = stablehlo.add %v6266, %v4901 : tensor<96xf32>
    %v6268 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6269 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6270 = stablehlo.multiply %v6268, %b2ebtv : tensor<96xf32>
    %v6271 = stablehlo.multiply %v6267, %v6267 : tensor<96xf32>
    %v6272 = stablehlo.multiply %v6269, %v6271 : tensor<96xf32>
    %v6273 = stablehlo.add %v6270, %v6272 : tensor<96xf32>
    %v6274 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6275 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6276 = stablehlo.multiply %v6274, %b2ebtv : tensor<96xf32>
    %v6277 = stablehlo.multiply %v6267, %v6267 : tensor<96xf32>
    %v6278 = stablehlo.multiply %v6275, %v6277 : tensor<96xf32>
    %v6279 = stablehlo.add %v6276, %v6278 : tensor<96xf32>
    %v6280 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6281 = stablehlo.add %v6279, %v6280 : tensor<96xf32>
    %v6282 = stablehlo.sqrt %v6281 : tensor<96xf32>
    %v6283 = stablehlo.divide %v6267, %v6282 : tensor<96xf32>
    %v6284 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6285 = stablehlo.multiply %v6284, %b2ebtm : tensor<96xf32>
    %v6286 = stablehlo.add %v6285, %v6283 : tensor<96xf32>
    %v6287 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6288 = stablehlo.multiply %v6287, %v6286 : tensor<96xf32>
    %v6289 = stablehlo.subtract %b2ebt, %v6288 : tensor<96xf32>
    %v6290 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6291 = stablehlo.multiply %v6290, %b2dW : tensor<96x1x3x3xf32>
    %v6292 = stablehlo.add %v6291, %v4909 : tensor<96x1x3x3xf32>
    %v6293 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6294 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6295 = stablehlo.multiply %v6293, %b2dWv : tensor<96x1x3x3xf32>
    %v6296 = stablehlo.multiply %v6292, %v6292 : tensor<96x1x3x3xf32>
    %v6297 = stablehlo.multiply %v6294, %v6296 : tensor<96x1x3x3xf32>
    %v6298 = stablehlo.add %v6295, %v6297 : tensor<96x1x3x3xf32>
    %v6299 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6300 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6301 = stablehlo.multiply %v6299, %b2dWv : tensor<96x1x3x3xf32>
    %v6302 = stablehlo.multiply %v6292, %v6292 : tensor<96x1x3x3xf32>
    %v6303 = stablehlo.multiply %v6300, %v6302 : tensor<96x1x3x3xf32>
    %v6304 = stablehlo.add %v6301, %v6303 : tensor<96x1x3x3xf32>
    %v6305 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6306 = stablehlo.add %v6304, %v6305 : tensor<96x1x3x3xf32>
    %v6307 = stablehlo.sqrt %v6306 : tensor<96x1x3x3xf32>
    %v6308 = stablehlo.divide %v6292, %v6307 : tensor<96x1x3x3xf32>
    %v6309 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6310 = stablehlo.multiply %v6309, %b2dWm : tensor<96x1x3x3xf32>
    %v6311 = stablehlo.add %v6310, %v6308 : tensor<96x1x3x3xf32>
    %v6312 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %v6313 = stablehlo.multiply %v6312, %v6311 : tensor<96x1x3x3xf32>
    %v6314 = stablehlo.subtract %b2dW, %v6313 : tensor<96x1x3x3xf32>
    %v6315 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6316 = stablehlo.multiply %v6315, %b2dg : tensor<96xf32>
    %v6317 = stablehlo.add %v6316, %v4927 : tensor<96xf32>
    %v6318 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6319 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6320 = stablehlo.multiply %v6318, %b2dgv : tensor<96xf32>
    %v6321 = stablehlo.multiply %v6317, %v6317 : tensor<96xf32>
    %v6322 = stablehlo.multiply %v6319, %v6321 : tensor<96xf32>
    %v6323 = stablehlo.add %v6320, %v6322 : tensor<96xf32>
    %v6324 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6325 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6326 = stablehlo.multiply %v6324, %b2dgv : tensor<96xf32>
    %v6327 = stablehlo.multiply %v6317, %v6317 : tensor<96xf32>
    %v6328 = stablehlo.multiply %v6325, %v6327 : tensor<96xf32>
    %v6329 = stablehlo.add %v6326, %v6328 : tensor<96xf32>
    %v6330 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6331 = stablehlo.add %v6329, %v6330 : tensor<96xf32>
    %v6332 = stablehlo.sqrt %v6331 : tensor<96xf32>
    %v6333 = stablehlo.divide %v6317, %v6332 : tensor<96xf32>
    %v6334 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6335 = stablehlo.multiply %v6334, %b2dgm : tensor<96xf32>
    %v6336 = stablehlo.add %v6335, %v6333 : tensor<96xf32>
    %v6337 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6338 = stablehlo.multiply %v6337, %v6336 : tensor<96xf32>
    %v6339 = stablehlo.subtract %b2dg, %v6338 : tensor<96xf32>
    %v6340 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6341 = stablehlo.multiply %v6340, %b2dbt : tensor<96xf32>
    %v6342 = stablehlo.add %v6341, %v4930 : tensor<96xf32>
    %v6343 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6344 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6345 = stablehlo.multiply %v6343, %b2dbtv : tensor<96xf32>
    %v6346 = stablehlo.multiply %v6342, %v6342 : tensor<96xf32>
    %v6347 = stablehlo.multiply %v6344, %v6346 : tensor<96xf32>
    %v6348 = stablehlo.add %v6345, %v6347 : tensor<96xf32>
    %v6349 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6350 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6351 = stablehlo.multiply %v6349, %b2dbtv : tensor<96xf32>
    %v6352 = stablehlo.multiply %v6342, %v6342 : tensor<96xf32>
    %v6353 = stablehlo.multiply %v6350, %v6352 : tensor<96xf32>
    %v6354 = stablehlo.add %v6351, %v6353 : tensor<96xf32>
    %v6355 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6356 = stablehlo.add %v6354, %v6355 : tensor<96xf32>
    %v6357 = stablehlo.sqrt %v6356 : tensor<96xf32>
    %v6358 = stablehlo.divide %v6342, %v6357 : tensor<96xf32>
    %v6359 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6360 = stablehlo.multiply %v6359, %b2dbtm : tensor<96xf32>
    %v6361 = stablehlo.add %v6360, %v6358 : tensor<96xf32>
    %v6362 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v6363 = stablehlo.multiply %v6362, %v6361 : tensor<96xf32>
    %v6364 = stablehlo.subtract %b2dbt, %v6363 : tensor<96xf32>
    %v6365 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6366 = stablehlo.multiply %v6365, %b2pW : tensor<24x96x1x1xf32>
    %v6367 = stablehlo.add %v6366, %v4936 : tensor<24x96x1x1xf32>
    %v6368 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6369 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6370 = stablehlo.multiply %v6368, %b2pWv : tensor<24x96x1x1xf32>
    %v6371 = stablehlo.multiply %v6367, %v6367 : tensor<24x96x1x1xf32>
    %v6372 = stablehlo.multiply %v6369, %v6371 : tensor<24x96x1x1xf32>
    %v6373 = stablehlo.add %v6370, %v6372 : tensor<24x96x1x1xf32>
    %v6374 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6375 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6376 = stablehlo.multiply %v6374, %b2pWv : tensor<24x96x1x1xf32>
    %v6377 = stablehlo.multiply %v6367, %v6367 : tensor<24x96x1x1xf32>
    %v6378 = stablehlo.multiply %v6375, %v6377 : tensor<24x96x1x1xf32>
    %v6379 = stablehlo.add %v6376, %v6378 : tensor<24x96x1x1xf32>
    %v6380 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6381 = stablehlo.add %v6379, %v6380 : tensor<24x96x1x1xf32>
    %v6382 = stablehlo.sqrt %v6381 : tensor<24x96x1x1xf32>
    %v6383 = stablehlo.divide %v6367, %v6382 : tensor<24x96x1x1xf32>
    %v6384 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6385 = stablehlo.multiply %v6384, %b2pWm : tensor<24x96x1x1xf32>
    %v6386 = stablehlo.add %v6385, %v6383 : tensor<24x96x1x1xf32>
    %v6387 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %v6388 = stablehlo.multiply %v6387, %v6386 : tensor<24x96x1x1xf32>
    %v6389 = stablehlo.subtract %b2pW, %v6388 : tensor<24x96x1x1xf32>
    %v6390 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6391 = stablehlo.multiply %v6390, %b2pg : tensor<24xf32>
    %v6392 = stablehlo.add %v6391, %v4954 : tensor<24xf32>
    %v6393 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6394 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6395 = stablehlo.multiply %v6393, %b2pgv : tensor<24xf32>
    %v6396 = stablehlo.multiply %v6392, %v6392 : tensor<24xf32>
    %v6397 = stablehlo.multiply %v6394, %v6396 : tensor<24xf32>
    %v6398 = stablehlo.add %v6395, %v6397 : tensor<24xf32>
    %v6399 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6400 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6401 = stablehlo.multiply %v6399, %b2pgv : tensor<24xf32>
    %v6402 = stablehlo.multiply %v6392, %v6392 : tensor<24xf32>
    %v6403 = stablehlo.multiply %v6400, %v6402 : tensor<24xf32>
    %v6404 = stablehlo.add %v6401, %v6403 : tensor<24xf32>
    %v6405 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6406 = stablehlo.add %v6404, %v6405 : tensor<24xf32>
    %v6407 = stablehlo.sqrt %v6406 : tensor<24xf32>
    %v6408 = stablehlo.divide %v6392, %v6407 : tensor<24xf32>
    %v6409 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6410 = stablehlo.multiply %v6409, %b2pgm : tensor<24xf32>
    %v6411 = stablehlo.add %v6410, %v6408 : tensor<24xf32>
    %v6412 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6413 = stablehlo.multiply %v6412, %v6411 : tensor<24xf32>
    %v6414 = stablehlo.subtract %b2pg, %v6413 : tensor<24xf32>
    %v6415 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6416 = stablehlo.multiply %v6415, %b2pbt : tensor<24xf32>
    %v6417 = stablehlo.add %v6416, %v4957 : tensor<24xf32>
    %v6418 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6419 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6420 = stablehlo.multiply %v6418, %b2pbtv : tensor<24xf32>
    %v6421 = stablehlo.multiply %v6417, %v6417 : tensor<24xf32>
    %v6422 = stablehlo.multiply %v6419, %v6421 : tensor<24xf32>
    %v6423 = stablehlo.add %v6420, %v6422 : tensor<24xf32>
    %v6424 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6425 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6426 = stablehlo.multiply %v6424, %b2pbtv : tensor<24xf32>
    %v6427 = stablehlo.multiply %v6417, %v6417 : tensor<24xf32>
    %v6428 = stablehlo.multiply %v6425, %v6427 : tensor<24xf32>
    %v6429 = stablehlo.add %v6426, %v6428 : tensor<24xf32>
    %v6430 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6431 = stablehlo.add %v6429, %v6430 : tensor<24xf32>
    %v6432 = stablehlo.sqrt %v6431 : tensor<24xf32>
    %v6433 = stablehlo.divide %v6417, %v6432 : tensor<24xf32>
    %v6434 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6435 = stablehlo.multiply %v6434, %b2pbtm : tensor<24xf32>
    %v6436 = stablehlo.add %v6435, %v6433 : tensor<24xf32>
    %v6437 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6438 = stablehlo.multiply %v6437, %v6436 : tensor<24xf32>
    %v6439 = stablehlo.subtract %b2pbt, %v6438 : tensor<24xf32>
    %v6440 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6441 = stablehlo.multiply %v6440, %b3eW : tensor<144x24x1x1xf32>
    %v6442 = stablehlo.add %v6441, %v4675 : tensor<144x24x1x1xf32>
    %v6443 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6444 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6445 = stablehlo.multiply %v6443, %b3eWv : tensor<144x24x1x1xf32>
    %v6446 = stablehlo.multiply %v6442, %v6442 : tensor<144x24x1x1xf32>
    %v6447 = stablehlo.multiply %v6444, %v6446 : tensor<144x24x1x1xf32>
    %v6448 = stablehlo.add %v6445, %v6447 : tensor<144x24x1x1xf32>
    %v6449 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6450 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6451 = stablehlo.multiply %v6449, %b3eWv : tensor<144x24x1x1xf32>
    %v6452 = stablehlo.multiply %v6442, %v6442 : tensor<144x24x1x1xf32>
    %v6453 = stablehlo.multiply %v6450, %v6452 : tensor<144x24x1x1xf32>
    %v6454 = stablehlo.add %v6451, %v6453 : tensor<144x24x1x1xf32>
    %v6455 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6456 = stablehlo.add %v6454, %v6455 : tensor<144x24x1x1xf32>
    %v6457 = stablehlo.sqrt %v6456 : tensor<144x24x1x1xf32>
    %v6458 = stablehlo.divide %v6442, %v6457 : tensor<144x24x1x1xf32>
    %v6459 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6460 = stablehlo.multiply %v6459, %b3eWm : tensor<144x24x1x1xf32>
    %v6461 = stablehlo.add %v6460, %v6458 : tensor<144x24x1x1xf32>
    %v6462 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6463 = stablehlo.multiply %v6462, %v6461 : tensor<144x24x1x1xf32>
    %v6464 = stablehlo.subtract %b3eW, %v6463 : tensor<144x24x1x1xf32>
    %v6465 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6466 = stablehlo.multiply %v6465, %b3eg : tensor<144xf32>
    %v6467 = stablehlo.add %v6466, %v4693 : tensor<144xf32>
    %v6468 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6469 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6470 = stablehlo.multiply %v6468, %b3egv : tensor<144xf32>
    %v6471 = stablehlo.multiply %v6467, %v6467 : tensor<144xf32>
    %v6472 = stablehlo.multiply %v6469, %v6471 : tensor<144xf32>
    %v6473 = stablehlo.add %v6470, %v6472 : tensor<144xf32>
    %v6474 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6475 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6476 = stablehlo.multiply %v6474, %b3egv : tensor<144xf32>
    %v6477 = stablehlo.multiply %v6467, %v6467 : tensor<144xf32>
    %v6478 = stablehlo.multiply %v6475, %v6477 : tensor<144xf32>
    %v6479 = stablehlo.add %v6476, %v6478 : tensor<144xf32>
    %v6480 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6481 = stablehlo.add %v6479, %v6480 : tensor<144xf32>
    %v6482 = stablehlo.sqrt %v6481 : tensor<144xf32>
    %v6483 = stablehlo.divide %v6467, %v6482 : tensor<144xf32>
    %v6484 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6485 = stablehlo.multiply %v6484, %b3egm : tensor<144xf32>
    %v6486 = stablehlo.add %v6485, %v6483 : tensor<144xf32>
    %v6487 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6488 = stablehlo.multiply %v6487, %v6486 : tensor<144xf32>
    %v6489 = stablehlo.subtract %b3eg, %v6488 : tensor<144xf32>
    %v6490 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6491 = stablehlo.multiply %v6490, %b3ebt : tensor<144xf32>
    %v6492 = stablehlo.add %v6491, %v4696 : tensor<144xf32>
    %v6493 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6494 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6495 = stablehlo.multiply %v6493, %b3ebtv : tensor<144xf32>
    %v6496 = stablehlo.multiply %v6492, %v6492 : tensor<144xf32>
    %v6497 = stablehlo.multiply %v6494, %v6496 : tensor<144xf32>
    %v6498 = stablehlo.add %v6495, %v6497 : tensor<144xf32>
    %v6499 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6500 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6501 = stablehlo.multiply %v6499, %b3ebtv : tensor<144xf32>
    %v6502 = stablehlo.multiply %v6492, %v6492 : tensor<144xf32>
    %v6503 = stablehlo.multiply %v6500, %v6502 : tensor<144xf32>
    %v6504 = stablehlo.add %v6501, %v6503 : tensor<144xf32>
    %v6505 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6506 = stablehlo.add %v6504, %v6505 : tensor<144xf32>
    %v6507 = stablehlo.sqrt %v6506 : tensor<144xf32>
    %v6508 = stablehlo.divide %v6492, %v6507 : tensor<144xf32>
    %v6509 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6510 = stablehlo.multiply %v6509, %b3ebtm : tensor<144xf32>
    %v6511 = stablehlo.add %v6510, %v6508 : tensor<144xf32>
    %v6512 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6513 = stablehlo.multiply %v6512, %v6511 : tensor<144xf32>
    %v6514 = stablehlo.subtract %b3ebt, %v6513 : tensor<144xf32>
    %v6515 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6516 = stablehlo.multiply %v6515, %b3dW : tensor<144x1x3x3xf32>
    %v6517 = stablehlo.add %v6516, %v4702 : tensor<144x1x3x3xf32>
    %v6518 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6519 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6520 = stablehlo.multiply %v6518, %b3dWv : tensor<144x1x3x3xf32>
    %v6521 = stablehlo.multiply %v6517, %v6517 : tensor<144x1x3x3xf32>
    %v6522 = stablehlo.multiply %v6519, %v6521 : tensor<144x1x3x3xf32>
    %v6523 = stablehlo.add %v6520, %v6522 : tensor<144x1x3x3xf32>
    %v6524 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6525 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6526 = stablehlo.multiply %v6524, %b3dWv : tensor<144x1x3x3xf32>
    %v6527 = stablehlo.multiply %v6517, %v6517 : tensor<144x1x3x3xf32>
    %v6528 = stablehlo.multiply %v6525, %v6527 : tensor<144x1x3x3xf32>
    %v6529 = stablehlo.add %v6526, %v6528 : tensor<144x1x3x3xf32>
    %v6530 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6531 = stablehlo.add %v6529, %v6530 : tensor<144x1x3x3xf32>
    %v6532 = stablehlo.sqrt %v6531 : tensor<144x1x3x3xf32>
    %v6533 = stablehlo.divide %v6517, %v6532 : tensor<144x1x3x3xf32>
    %v6534 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6535 = stablehlo.multiply %v6534, %b3dWm : tensor<144x1x3x3xf32>
    %v6536 = stablehlo.add %v6535, %v6533 : tensor<144x1x3x3xf32>
    %v6537 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6538 = stablehlo.multiply %v6537, %v6536 : tensor<144x1x3x3xf32>
    %v6539 = stablehlo.subtract %b3dW, %v6538 : tensor<144x1x3x3xf32>
    %v6540 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6541 = stablehlo.multiply %v6540, %b3dg : tensor<144xf32>
    %v6542 = stablehlo.add %v6541, %v4720 : tensor<144xf32>
    %v6543 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6544 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6545 = stablehlo.multiply %v6543, %b3dgv : tensor<144xf32>
    %v6546 = stablehlo.multiply %v6542, %v6542 : tensor<144xf32>
    %v6547 = stablehlo.multiply %v6544, %v6546 : tensor<144xf32>
    %v6548 = stablehlo.add %v6545, %v6547 : tensor<144xf32>
    %v6549 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6550 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6551 = stablehlo.multiply %v6549, %b3dgv : tensor<144xf32>
    %v6552 = stablehlo.multiply %v6542, %v6542 : tensor<144xf32>
    %v6553 = stablehlo.multiply %v6550, %v6552 : tensor<144xf32>
    %v6554 = stablehlo.add %v6551, %v6553 : tensor<144xf32>
    %v6555 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6556 = stablehlo.add %v6554, %v6555 : tensor<144xf32>
    %v6557 = stablehlo.sqrt %v6556 : tensor<144xf32>
    %v6558 = stablehlo.divide %v6542, %v6557 : tensor<144xf32>
    %v6559 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6560 = stablehlo.multiply %v6559, %b3dgm : tensor<144xf32>
    %v6561 = stablehlo.add %v6560, %v6558 : tensor<144xf32>
    %v6562 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6563 = stablehlo.multiply %v6562, %v6561 : tensor<144xf32>
    %v6564 = stablehlo.subtract %b3dg, %v6563 : tensor<144xf32>
    %v6565 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6566 = stablehlo.multiply %v6565, %b3dbt : tensor<144xf32>
    %v6567 = stablehlo.add %v6566, %v4723 : tensor<144xf32>
    %v6568 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6569 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6570 = stablehlo.multiply %v6568, %b3dbtv : tensor<144xf32>
    %v6571 = stablehlo.multiply %v6567, %v6567 : tensor<144xf32>
    %v6572 = stablehlo.multiply %v6569, %v6571 : tensor<144xf32>
    %v6573 = stablehlo.add %v6570, %v6572 : tensor<144xf32>
    %v6574 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6575 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6576 = stablehlo.multiply %v6574, %b3dbtv : tensor<144xf32>
    %v6577 = stablehlo.multiply %v6567, %v6567 : tensor<144xf32>
    %v6578 = stablehlo.multiply %v6575, %v6577 : tensor<144xf32>
    %v6579 = stablehlo.add %v6576, %v6578 : tensor<144xf32>
    %v6580 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6581 = stablehlo.add %v6579, %v6580 : tensor<144xf32>
    %v6582 = stablehlo.sqrt %v6581 : tensor<144xf32>
    %v6583 = stablehlo.divide %v6567, %v6582 : tensor<144xf32>
    %v6584 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6585 = stablehlo.multiply %v6584, %b3dbtm : tensor<144xf32>
    %v6586 = stablehlo.add %v6585, %v6583 : tensor<144xf32>
    %v6587 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6588 = stablehlo.multiply %v6587, %v6586 : tensor<144xf32>
    %v6589 = stablehlo.subtract %b3dbt, %v6588 : tensor<144xf32>
    %v6590 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6591 = stablehlo.multiply %v6590, %b3pW : tensor<24x144x1x1xf32>
    %v6592 = stablehlo.add %v6591, %v4729 : tensor<24x144x1x1xf32>
    %v6593 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6594 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6595 = stablehlo.multiply %v6593, %b3pWv : tensor<24x144x1x1xf32>
    %v6596 = stablehlo.multiply %v6592, %v6592 : tensor<24x144x1x1xf32>
    %v6597 = stablehlo.multiply %v6594, %v6596 : tensor<24x144x1x1xf32>
    %v6598 = stablehlo.add %v6595, %v6597 : tensor<24x144x1x1xf32>
    %v6599 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6600 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6601 = stablehlo.multiply %v6599, %b3pWv : tensor<24x144x1x1xf32>
    %v6602 = stablehlo.multiply %v6592, %v6592 : tensor<24x144x1x1xf32>
    %v6603 = stablehlo.multiply %v6600, %v6602 : tensor<24x144x1x1xf32>
    %v6604 = stablehlo.add %v6601, %v6603 : tensor<24x144x1x1xf32>
    %v6605 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6606 = stablehlo.add %v6604, %v6605 : tensor<24x144x1x1xf32>
    %v6607 = stablehlo.sqrt %v6606 : tensor<24x144x1x1xf32>
    %v6608 = stablehlo.divide %v6592, %v6607 : tensor<24x144x1x1xf32>
    %v6609 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6610 = stablehlo.multiply %v6609, %b3pWm : tensor<24x144x1x1xf32>
    %v6611 = stablehlo.add %v6610, %v6608 : tensor<24x144x1x1xf32>
    %v6612 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24x144x1x1xf32>
    %v6613 = stablehlo.multiply %v6612, %v6611 : tensor<24x144x1x1xf32>
    %v6614 = stablehlo.subtract %b3pW, %v6613 : tensor<24x144x1x1xf32>
    %v6615 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6616 = stablehlo.multiply %v6615, %b3pg : tensor<24xf32>
    %v6617 = stablehlo.add %v6616, %v4747 : tensor<24xf32>
    %v6618 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6619 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6620 = stablehlo.multiply %v6618, %b3pgv : tensor<24xf32>
    %v6621 = stablehlo.multiply %v6617, %v6617 : tensor<24xf32>
    %v6622 = stablehlo.multiply %v6619, %v6621 : tensor<24xf32>
    %v6623 = stablehlo.add %v6620, %v6622 : tensor<24xf32>
    %v6624 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6625 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6626 = stablehlo.multiply %v6624, %b3pgv : tensor<24xf32>
    %v6627 = stablehlo.multiply %v6617, %v6617 : tensor<24xf32>
    %v6628 = stablehlo.multiply %v6625, %v6627 : tensor<24xf32>
    %v6629 = stablehlo.add %v6626, %v6628 : tensor<24xf32>
    %v6630 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6631 = stablehlo.add %v6629, %v6630 : tensor<24xf32>
    %v6632 = stablehlo.sqrt %v6631 : tensor<24xf32>
    %v6633 = stablehlo.divide %v6617, %v6632 : tensor<24xf32>
    %v6634 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6635 = stablehlo.multiply %v6634, %b3pgm : tensor<24xf32>
    %v6636 = stablehlo.add %v6635, %v6633 : tensor<24xf32>
    %v6637 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6638 = stablehlo.multiply %v6637, %v6636 : tensor<24xf32>
    %v6639 = stablehlo.subtract %b3pg, %v6638 : tensor<24xf32>
    %v6640 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6641 = stablehlo.multiply %v6640, %b3pbt : tensor<24xf32>
    %v6642 = stablehlo.add %v6641, %v4750 : tensor<24xf32>
    %v6643 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6644 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6645 = stablehlo.multiply %v6643, %b3pbtv : tensor<24xf32>
    %v6646 = stablehlo.multiply %v6642, %v6642 : tensor<24xf32>
    %v6647 = stablehlo.multiply %v6644, %v6646 : tensor<24xf32>
    %v6648 = stablehlo.add %v6645, %v6647 : tensor<24xf32>
    %v6649 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6650 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6651 = stablehlo.multiply %v6649, %b3pbtv : tensor<24xf32>
    %v6652 = stablehlo.multiply %v6642, %v6642 : tensor<24xf32>
    %v6653 = stablehlo.multiply %v6650, %v6652 : tensor<24xf32>
    %v6654 = stablehlo.add %v6651, %v6653 : tensor<24xf32>
    %v6655 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6656 = stablehlo.add %v6654, %v6655 : tensor<24xf32>
    %v6657 = stablehlo.sqrt %v6656 : tensor<24xf32>
    %v6658 = stablehlo.divide %v6642, %v6657 : tensor<24xf32>
    %v6659 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6660 = stablehlo.multiply %v6659, %b3pbtm : tensor<24xf32>
    %v6661 = stablehlo.add %v6660, %v6658 : tensor<24xf32>
    %v6662 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %v6663 = stablehlo.multiply %v6662, %v6661 : tensor<24xf32>
    %v6664 = stablehlo.subtract %b3pbt, %v6663 : tensor<24xf32>
    %v6665 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6666 = stablehlo.multiply %v6665, %b4eW : tensor<144x24x1x1xf32>
    %v6667 = stablehlo.add %v6666, %v4466 : tensor<144x24x1x1xf32>
    %v6668 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6669 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6670 = stablehlo.multiply %v6668, %b4eWv : tensor<144x24x1x1xf32>
    %v6671 = stablehlo.multiply %v6667, %v6667 : tensor<144x24x1x1xf32>
    %v6672 = stablehlo.multiply %v6669, %v6671 : tensor<144x24x1x1xf32>
    %v6673 = stablehlo.add %v6670, %v6672 : tensor<144x24x1x1xf32>
    %v6674 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6675 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6676 = stablehlo.multiply %v6674, %b4eWv : tensor<144x24x1x1xf32>
    %v6677 = stablehlo.multiply %v6667, %v6667 : tensor<144x24x1x1xf32>
    %v6678 = stablehlo.multiply %v6675, %v6677 : tensor<144x24x1x1xf32>
    %v6679 = stablehlo.add %v6676, %v6678 : tensor<144x24x1x1xf32>
    %v6680 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6681 = stablehlo.add %v6679, %v6680 : tensor<144x24x1x1xf32>
    %v6682 = stablehlo.sqrt %v6681 : tensor<144x24x1x1xf32>
    %v6683 = stablehlo.divide %v6667, %v6682 : tensor<144x24x1x1xf32>
    %v6684 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6685 = stablehlo.multiply %v6684, %b4eWm : tensor<144x24x1x1xf32>
    %v6686 = stablehlo.add %v6685, %v6683 : tensor<144x24x1x1xf32>
    %v6687 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x24x1x1xf32>
    %v6688 = stablehlo.multiply %v6687, %v6686 : tensor<144x24x1x1xf32>
    %v6689 = stablehlo.subtract %b4eW, %v6688 : tensor<144x24x1x1xf32>
    %v6690 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6691 = stablehlo.multiply %v6690, %b4eg : tensor<144xf32>
    %v6692 = stablehlo.add %v6691, %v4484 : tensor<144xf32>
    %v6693 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6694 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6695 = stablehlo.multiply %v6693, %b4egv : tensor<144xf32>
    %v6696 = stablehlo.multiply %v6692, %v6692 : tensor<144xf32>
    %v6697 = stablehlo.multiply %v6694, %v6696 : tensor<144xf32>
    %v6698 = stablehlo.add %v6695, %v6697 : tensor<144xf32>
    %v6699 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6700 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6701 = stablehlo.multiply %v6699, %b4egv : tensor<144xf32>
    %v6702 = stablehlo.multiply %v6692, %v6692 : tensor<144xf32>
    %v6703 = stablehlo.multiply %v6700, %v6702 : tensor<144xf32>
    %v6704 = stablehlo.add %v6701, %v6703 : tensor<144xf32>
    %v6705 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6706 = stablehlo.add %v6704, %v6705 : tensor<144xf32>
    %v6707 = stablehlo.sqrt %v6706 : tensor<144xf32>
    %v6708 = stablehlo.divide %v6692, %v6707 : tensor<144xf32>
    %v6709 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6710 = stablehlo.multiply %v6709, %b4egm : tensor<144xf32>
    %v6711 = stablehlo.add %v6710, %v6708 : tensor<144xf32>
    %v6712 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6713 = stablehlo.multiply %v6712, %v6711 : tensor<144xf32>
    %v6714 = stablehlo.subtract %b4eg, %v6713 : tensor<144xf32>
    %v6715 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6716 = stablehlo.multiply %v6715, %b4ebt : tensor<144xf32>
    %v6717 = stablehlo.add %v6716, %v4487 : tensor<144xf32>
    %v6718 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6719 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6720 = stablehlo.multiply %v6718, %b4ebtv : tensor<144xf32>
    %v6721 = stablehlo.multiply %v6717, %v6717 : tensor<144xf32>
    %v6722 = stablehlo.multiply %v6719, %v6721 : tensor<144xf32>
    %v6723 = stablehlo.add %v6720, %v6722 : tensor<144xf32>
    %v6724 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6725 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6726 = stablehlo.multiply %v6724, %b4ebtv : tensor<144xf32>
    %v6727 = stablehlo.multiply %v6717, %v6717 : tensor<144xf32>
    %v6728 = stablehlo.multiply %v6725, %v6727 : tensor<144xf32>
    %v6729 = stablehlo.add %v6726, %v6728 : tensor<144xf32>
    %v6730 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6731 = stablehlo.add %v6729, %v6730 : tensor<144xf32>
    %v6732 = stablehlo.sqrt %v6731 : tensor<144xf32>
    %v6733 = stablehlo.divide %v6717, %v6732 : tensor<144xf32>
    %v6734 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6735 = stablehlo.multiply %v6734, %b4ebtm : tensor<144xf32>
    %v6736 = stablehlo.add %v6735, %v6733 : tensor<144xf32>
    %v6737 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6738 = stablehlo.multiply %v6737, %v6736 : tensor<144xf32>
    %v6739 = stablehlo.subtract %b4ebt, %v6738 : tensor<144xf32>
    %v6740 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6741 = stablehlo.multiply %v6740, %b4dW : tensor<144x1x3x3xf32>
    %v6742 = stablehlo.add %v6741, %v4495 : tensor<144x1x3x3xf32>
    %v6743 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6744 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6745 = stablehlo.multiply %v6743, %b4dWv : tensor<144x1x3x3xf32>
    %v6746 = stablehlo.multiply %v6742, %v6742 : tensor<144x1x3x3xf32>
    %v6747 = stablehlo.multiply %v6744, %v6746 : tensor<144x1x3x3xf32>
    %v6748 = stablehlo.add %v6745, %v6747 : tensor<144x1x3x3xf32>
    %v6749 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6750 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6751 = stablehlo.multiply %v6749, %b4dWv : tensor<144x1x3x3xf32>
    %v6752 = stablehlo.multiply %v6742, %v6742 : tensor<144x1x3x3xf32>
    %v6753 = stablehlo.multiply %v6750, %v6752 : tensor<144x1x3x3xf32>
    %v6754 = stablehlo.add %v6751, %v6753 : tensor<144x1x3x3xf32>
    %v6755 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6756 = stablehlo.add %v6754, %v6755 : tensor<144x1x3x3xf32>
    %v6757 = stablehlo.sqrt %v6756 : tensor<144x1x3x3xf32>
    %v6758 = stablehlo.divide %v6742, %v6757 : tensor<144x1x3x3xf32>
    %v6759 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6760 = stablehlo.multiply %v6759, %b4dWm : tensor<144x1x3x3xf32>
    %v6761 = stablehlo.add %v6760, %v6758 : tensor<144x1x3x3xf32>
    %v6762 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144x1x3x3xf32>
    %v6763 = stablehlo.multiply %v6762, %v6761 : tensor<144x1x3x3xf32>
    %v6764 = stablehlo.subtract %b4dW, %v6763 : tensor<144x1x3x3xf32>
    %v6765 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6766 = stablehlo.multiply %v6765, %b4dg : tensor<144xf32>
    %v6767 = stablehlo.add %v6766, %v4513 : tensor<144xf32>
    %v6768 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6769 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6770 = stablehlo.multiply %v6768, %b4dgv : tensor<144xf32>
    %v6771 = stablehlo.multiply %v6767, %v6767 : tensor<144xf32>
    %v6772 = stablehlo.multiply %v6769, %v6771 : tensor<144xf32>
    %v6773 = stablehlo.add %v6770, %v6772 : tensor<144xf32>
    %v6774 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6775 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6776 = stablehlo.multiply %v6774, %b4dgv : tensor<144xf32>
    %v6777 = stablehlo.multiply %v6767, %v6767 : tensor<144xf32>
    %v6778 = stablehlo.multiply %v6775, %v6777 : tensor<144xf32>
    %v6779 = stablehlo.add %v6776, %v6778 : tensor<144xf32>
    %v6780 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6781 = stablehlo.add %v6779, %v6780 : tensor<144xf32>
    %v6782 = stablehlo.sqrt %v6781 : tensor<144xf32>
    %v6783 = stablehlo.divide %v6767, %v6782 : tensor<144xf32>
    %v6784 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6785 = stablehlo.multiply %v6784, %b4dgm : tensor<144xf32>
    %v6786 = stablehlo.add %v6785, %v6783 : tensor<144xf32>
    %v6787 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6788 = stablehlo.multiply %v6787, %v6786 : tensor<144xf32>
    %v6789 = stablehlo.subtract %b4dg, %v6788 : tensor<144xf32>
    %v6790 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6791 = stablehlo.multiply %v6790, %b4dbt : tensor<144xf32>
    %v6792 = stablehlo.add %v6791, %v4516 : tensor<144xf32>
    %v6793 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6794 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6795 = stablehlo.multiply %v6793, %b4dbtv : tensor<144xf32>
    %v6796 = stablehlo.multiply %v6792, %v6792 : tensor<144xf32>
    %v6797 = stablehlo.multiply %v6794, %v6796 : tensor<144xf32>
    %v6798 = stablehlo.add %v6795, %v6797 : tensor<144xf32>
    %v6799 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6800 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6801 = stablehlo.multiply %v6799, %b4dbtv : tensor<144xf32>
    %v6802 = stablehlo.multiply %v6792, %v6792 : tensor<144xf32>
    %v6803 = stablehlo.multiply %v6800, %v6802 : tensor<144xf32>
    %v6804 = stablehlo.add %v6801, %v6803 : tensor<144xf32>
    %v6805 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6806 = stablehlo.add %v6804, %v6805 : tensor<144xf32>
    %v6807 = stablehlo.sqrt %v6806 : tensor<144xf32>
    %v6808 = stablehlo.divide %v6792, %v6807 : tensor<144xf32>
    %v6809 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6810 = stablehlo.multiply %v6809, %b4dbtm : tensor<144xf32>
    %v6811 = stablehlo.add %v6810, %v6808 : tensor<144xf32>
    %v6812 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<144xf32>
    %v6813 = stablehlo.multiply %v6812, %v6811 : tensor<144xf32>
    %v6814 = stablehlo.subtract %b4dbt, %v6813 : tensor<144xf32>
    %v6815 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6816 = stablehlo.multiply %v6815, %b4pW : tensor<32x144x1x1xf32>
    %v6817 = stablehlo.add %v6816, %v4522 : tensor<32x144x1x1xf32>
    %v6818 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6819 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6820 = stablehlo.multiply %v6818, %b4pWv : tensor<32x144x1x1xf32>
    %v6821 = stablehlo.multiply %v6817, %v6817 : tensor<32x144x1x1xf32>
    %v6822 = stablehlo.multiply %v6819, %v6821 : tensor<32x144x1x1xf32>
    %v6823 = stablehlo.add %v6820, %v6822 : tensor<32x144x1x1xf32>
    %v6824 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6825 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6826 = stablehlo.multiply %v6824, %b4pWv : tensor<32x144x1x1xf32>
    %v6827 = stablehlo.multiply %v6817, %v6817 : tensor<32x144x1x1xf32>
    %v6828 = stablehlo.multiply %v6825, %v6827 : tensor<32x144x1x1xf32>
    %v6829 = stablehlo.add %v6826, %v6828 : tensor<32x144x1x1xf32>
    %v6830 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6831 = stablehlo.add %v6829, %v6830 : tensor<32x144x1x1xf32>
    %v6832 = stablehlo.sqrt %v6831 : tensor<32x144x1x1xf32>
    %v6833 = stablehlo.divide %v6817, %v6832 : tensor<32x144x1x1xf32>
    %v6834 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6835 = stablehlo.multiply %v6834, %b4pWm : tensor<32x144x1x1xf32>
    %v6836 = stablehlo.add %v6835, %v6833 : tensor<32x144x1x1xf32>
    %v6837 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x144x1x1xf32>
    %v6838 = stablehlo.multiply %v6837, %v6836 : tensor<32x144x1x1xf32>
    %v6839 = stablehlo.subtract %b4pW, %v6838 : tensor<32x144x1x1xf32>
    %v6840 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6841 = stablehlo.multiply %v6840, %b4pg : tensor<32xf32>
    %v6842 = stablehlo.add %v6841, %v4540 : tensor<32xf32>
    %v6843 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6844 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6845 = stablehlo.multiply %v6843, %b4pgv : tensor<32xf32>
    %v6846 = stablehlo.multiply %v6842, %v6842 : tensor<32xf32>
    %v6847 = stablehlo.multiply %v6844, %v6846 : tensor<32xf32>
    %v6848 = stablehlo.add %v6845, %v6847 : tensor<32xf32>
    %v6849 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6850 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6851 = stablehlo.multiply %v6849, %b4pgv : tensor<32xf32>
    %v6852 = stablehlo.multiply %v6842, %v6842 : tensor<32xf32>
    %v6853 = stablehlo.multiply %v6850, %v6852 : tensor<32xf32>
    %v6854 = stablehlo.add %v6851, %v6853 : tensor<32xf32>
    %v6855 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6856 = stablehlo.add %v6854, %v6855 : tensor<32xf32>
    %v6857 = stablehlo.sqrt %v6856 : tensor<32xf32>
    %v6858 = stablehlo.divide %v6842, %v6857 : tensor<32xf32>
    %v6859 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6860 = stablehlo.multiply %v6859, %b4pgm : tensor<32xf32>
    %v6861 = stablehlo.add %v6860, %v6858 : tensor<32xf32>
    %v6862 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6863 = stablehlo.multiply %v6862, %v6861 : tensor<32xf32>
    %v6864 = stablehlo.subtract %b4pg, %v6863 : tensor<32xf32>
    %v6865 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6866 = stablehlo.multiply %v6865, %b4pbt : tensor<32xf32>
    %v6867 = stablehlo.add %v6866, %v4543 : tensor<32xf32>
    %v6868 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6869 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6870 = stablehlo.multiply %v6868, %b4pbtv : tensor<32xf32>
    %v6871 = stablehlo.multiply %v6867, %v6867 : tensor<32xf32>
    %v6872 = stablehlo.multiply %v6869, %v6871 : tensor<32xf32>
    %v6873 = stablehlo.add %v6870, %v6872 : tensor<32xf32>
    %v6874 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6875 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6876 = stablehlo.multiply %v6874, %b4pbtv : tensor<32xf32>
    %v6877 = stablehlo.multiply %v6867, %v6867 : tensor<32xf32>
    %v6878 = stablehlo.multiply %v6875, %v6877 : tensor<32xf32>
    %v6879 = stablehlo.add %v6876, %v6878 : tensor<32xf32>
    %v6880 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6881 = stablehlo.add %v6879, %v6880 : tensor<32xf32>
    %v6882 = stablehlo.sqrt %v6881 : tensor<32xf32>
    %v6883 = stablehlo.divide %v6867, %v6882 : tensor<32xf32>
    %v6884 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6885 = stablehlo.multiply %v6884, %b4pbtm : tensor<32xf32>
    %v6886 = stablehlo.add %v6885, %v6883 : tensor<32xf32>
    %v6887 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v6888 = stablehlo.multiply %v6887, %v6886 : tensor<32xf32>
    %v6889 = stablehlo.subtract %b4pbt, %v6888 : tensor<32xf32>
    %v6890 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6891 = stablehlo.multiply %v6890, %b5eW : tensor<192x32x1x1xf32>
    %v6892 = stablehlo.add %v6891, %v4261 : tensor<192x32x1x1xf32>
    %v6893 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6894 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6895 = stablehlo.multiply %v6893, %b5eWv : tensor<192x32x1x1xf32>
    %v6896 = stablehlo.multiply %v6892, %v6892 : tensor<192x32x1x1xf32>
    %v6897 = stablehlo.multiply %v6894, %v6896 : tensor<192x32x1x1xf32>
    %v6898 = stablehlo.add %v6895, %v6897 : tensor<192x32x1x1xf32>
    %v6899 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6900 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6901 = stablehlo.multiply %v6899, %b5eWv : tensor<192x32x1x1xf32>
    %v6902 = stablehlo.multiply %v6892, %v6892 : tensor<192x32x1x1xf32>
    %v6903 = stablehlo.multiply %v6900, %v6902 : tensor<192x32x1x1xf32>
    %v6904 = stablehlo.add %v6901, %v6903 : tensor<192x32x1x1xf32>
    %v6905 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6906 = stablehlo.add %v6904, %v6905 : tensor<192x32x1x1xf32>
    %v6907 = stablehlo.sqrt %v6906 : tensor<192x32x1x1xf32>
    %v6908 = stablehlo.divide %v6892, %v6907 : tensor<192x32x1x1xf32>
    %v6909 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6910 = stablehlo.multiply %v6909, %b5eWm : tensor<192x32x1x1xf32>
    %v6911 = stablehlo.add %v6910, %v6908 : tensor<192x32x1x1xf32>
    %v6912 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v6913 = stablehlo.multiply %v6912, %v6911 : tensor<192x32x1x1xf32>
    %v6914 = stablehlo.subtract %b5eW, %v6913 : tensor<192x32x1x1xf32>
    %v6915 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6916 = stablehlo.multiply %v6915, %b5eg : tensor<192xf32>
    %v6917 = stablehlo.add %v6916, %v4279 : tensor<192xf32>
    %v6918 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6919 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6920 = stablehlo.multiply %v6918, %b5egv : tensor<192xf32>
    %v6921 = stablehlo.multiply %v6917, %v6917 : tensor<192xf32>
    %v6922 = stablehlo.multiply %v6919, %v6921 : tensor<192xf32>
    %v6923 = stablehlo.add %v6920, %v6922 : tensor<192xf32>
    %v6924 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6925 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6926 = stablehlo.multiply %v6924, %b5egv : tensor<192xf32>
    %v6927 = stablehlo.multiply %v6917, %v6917 : tensor<192xf32>
    %v6928 = stablehlo.multiply %v6925, %v6927 : tensor<192xf32>
    %v6929 = stablehlo.add %v6926, %v6928 : tensor<192xf32>
    %v6930 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6931 = stablehlo.add %v6929, %v6930 : tensor<192xf32>
    %v6932 = stablehlo.sqrt %v6931 : tensor<192xf32>
    %v6933 = stablehlo.divide %v6917, %v6932 : tensor<192xf32>
    %v6934 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6935 = stablehlo.multiply %v6934, %b5egm : tensor<192xf32>
    %v6936 = stablehlo.add %v6935, %v6933 : tensor<192xf32>
    %v6937 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6938 = stablehlo.multiply %v6937, %v6936 : tensor<192xf32>
    %v6939 = stablehlo.subtract %b5eg, %v6938 : tensor<192xf32>
    %v6940 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6941 = stablehlo.multiply %v6940, %b5ebt : tensor<192xf32>
    %v6942 = stablehlo.add %v6941, %v4282 : tensor<192xf32>
    %v6943 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6944 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6945 = stablehlo.multiply %v6943, %b5ebtv : tensor<192xf32>
    %v6946 = stablehlo.multiply %v6942, %v6942 : tensor<192xf32>
    %v6947 = stablehlo.multiply %v6944, %v6946 : tensor<192xf32>
    %v6948 = stablehlo.add %v6945, %v6947 : tensor<192xf32>
    %v6949 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6950 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6951 = stablehlo.multiply %v6949, %b5ebtv : tensor<192xf32>
    %v6952 = stablehlo.multiply %v6942, %v6942 : tensor<192xf32>
    %v6953 = stablehlo.multiply %v6950, %v6952 : tensor<192xf32>
    %v6954 = stablehlo.add %v6951, %v6953 : tensor<192xf32>
    %v6955 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6956 = stablehlo.add %v6954, %v6955 : tensor<192xf32>
    %v6957 = stablehlo.sqrt %v6956 : tensor<192xf32>
    %v6958 = stablehlo.divide %v6942, %v6957 : tensor<192xf32>
    %v6959 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6960 = stablehlo.multiply %v6959, %b5ebtm : tensor<192xf32>
    %v6961 = stablehlo.add %v6960, %v6958 : tensor<192xf32>
    %v6962 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6963 = stablehlo.multiply %v6962, %v6961 : tensor<192xf32>
    %v6964 = stablehlo.subtract %b5ebt, %v6963 : tensor<192xf32>
    %v6965 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6966 = stablehlo.multiply %v6965, %b5dW : tensor<192x1x3x3xf32>
    %v6967 = stablehlo.add %v6966, %v4288 : tensor<192x1x3x3xf32>
    %v6968 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6969 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6970 = stablehlo.multiply %v6968, %b5dWv : tensor<192x1x3x3xf32>
    %v6971 = stablehlo.multiply %v6967, %v6967 : tensor<192x1x3x3xf32>
    %v6972 = stablehlo.multiply %v6969, %v6971 : tensor<192x1x3x3xf32>
    %v6973 = stablehlo.add %v6970, %v6972 : tensor<192x1x3x3xf32>
    %v6974 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6975 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6976 = stablehlo.multiply %v6974, %b5dWv : tensor<192x1x3x3xf32>
    %v6977 = stablehlo.multiply %v6967, %v6967 : tensor<192x1x3x3xf32>
    %v6978 = stablehlo.multiply %v6975, %v6977 : tensor<192x1x3x3xf32>
    %v6979 = stablehlo.add %v6976, %v6978 : tensor<192x1x3x3xf32>
    %v6980 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6981 = stablehlo.add %v6979, %v6980 : tensor<192x1x3x3xf32>
    %v6982 = stablehlo.sqrt %v6981 : tensor<192x1x3x3xf32>
    %v6983 = stablehlo.divide %v6967, %v6982 : tensor<192x1x3x3xf32>
    %v6984 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6985 = stablehlo.multiply %v6984, %b5dWm : tensor<192x1x3x3xf32>
    %v6986 = stablehlo.add %v6985, %v6983 : tensor<192x1x3x3xf32>
    %v6987 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v6988 = stablehlo.multiply %v6987, %v6986 : tensor<192x1x3x3xf32>
    %v6989 = stablehlo.subtract %b5dW, %v6988 : tensor<192x1x3x3xf32>
    %v6990 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6991 = stablehlo.multiply %v6990, %b5dg : tensor<192xf32>
    %v6992 = stablehlo.add %v6991, %v4306 : tensor<192xf32>
    %v6993 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6994 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6995 = stablehlo.multiply %v6993, %b5dgv : tensor<192xf32>
    %v6996 = stablehlo.multiply %v6992, %v6992 : tensor<192xf32>
    %v6997 = stablehlo.multiply %v6994, %v6996 : tensor<192xf32>
    %v6998 = stablehlo.add %v6995, %v6997 : tensor<192xf32>
    %v6999 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7000 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7001 = stablehlo.multiply %v6999, %b5dgv : tensor<192xf32>
    %v7002 = stablehlo.multiply %v6992, %v6992 : tensor<192xf32>
    %v7003 = stablehlo.multiply %v7000, %v7002 : tensor<192xf32>
    %v7004 = stablehlo.add %v7001, %v7003 : tensor<192xf32>
    %v7005 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7006 = stablehlo.add %v7004, %v7005 : tensor<192xf32>
    %v7007 = stablehlo.sqrt %v7006 : tensor<192xf32>
    %v7008 = stablehlo.divide %v6992, %v7007 : tensor<192xf32>
    %v7009 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7010 = stablehlo.multiply %v7009, %b5dgm : tensor<192xf32>
    %v7011 = stablehlo.add %v7010, %v7008 : tensor<192xf32>
    %v7012 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7013 = stablehlo.multiply %v7012, %v7011 : tensor<192xf32>
    %v7014 = stablehlo.subtract %b5dg, %v7013 : tensor<192xf32>
    %v7015 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7016 = stablehlo.multiply %v7015, %b5dbt : tensor<192xf32>
    %v7017 = stablehlo.add %v7016, %v4309 : tensor<192xf32>
    %v7018 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7019 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7020 = stablehlo.multiply %v7018, %b5dbtv : tensor<192xf32>
    %v7021 = stablehlo.multiply %v7017, %v7017 : tensor<192xf32>
    %v7022 = stablehlo.multiply %v7019, %v7021 : tensor<192xf32>
    %v7023 = stablehlo.add %v7020, %v7022 : tensor<192xf32>
    %v7024 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7025 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7026 = stablehlo.multiply %v7024, %b5dbtv : tensor<192xf32>
    %v7027 = stablehlo.multiply %v7017, %v7017 : tensor<192xf32>
    %v7028 = stablehlo.multiply %v7025, %v7027 : tensor<192xf32>
    %v7029 = stablehlo.add %v7026, %v7028 : tensor<192xf32>
    %v7030 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7031 = stablehlo.add %v7029, %v7030 : tensor<192xf32>
    %v7032 = stablehlo.sqrt %v7031 : tensor<192xf32>
    %v7033 = stablehlo.divide %v7017, %v7032 : tensor<192xf32>
    %v7034 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7035 = stablehlo.multiply %v7034, %b5dbtm : tensor<192xf32>
    %v7036 = stablehlo.add %v7035, %v7033 : tensor<192xf32>
    %v7037 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7038 = stablehlo.multiply %v7037, %v7036 : tensor<192xf32>
    %v7039 = stablehlo.subtract %b5dbt, %v7038 : tensor<192xf32>
    %v7040 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7041 = stablehlo.multiply %v7040, %b5pW : tensor<32x192x1x1xf32>
    %v7042 = stablehlo.add %v7041, %v4315 : tensor<32x192x1x1xf32>
    %v7043 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7044 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7045 = stablehlo.multiply %v7043, %b5pWv : tensor<32x192x1x1xf32>
    %v7046 = stablehlo.multiply %v7042, %v7042 : tensor<32x192x1x1xf32>
    %v7047 = stablehlo.multiply %v7044, %v7046 : tensor<32x192x1x1xf32>
    %v7048 = stablehlo.add %v7045, %v7047 : tensor<32x192x1x1xf32>
    %v7049 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7050 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7051 = stablehlo.multiply %v7049, %b5pWv : tensor<32x192x1x1xf32>
    %v7052 = stablehlo.multiply %v7042, %v7042 : tensor<32x192x1x1xf32>
    %v7053 = stablehlo.multiply %v7050, %v7052 : tensor<32x192x1x1xf32>
    %v7054 = stablehlo.add %v7051, %v7053 : tensor<32x192x1x1xf32>
    %v7055 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7056 = stablehlo.add %v7054, %v7055 : tensor<32x192x1x1xf32>
    %v7057 = stablehlo.sqrt %v7056 : tensor<32x192x1x1xf32>
    %v7058 = stablehlo.divide %v7042, %v7057 : tensor<32x192x1x1xf32>
    %v7059 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7060 = stablehlo.multiply %v7059, %b5pWm : tensor<32x192x1x1xf32>
    %v7061 = stablehlo.add %v7060, %v7058 : tensor<32x192x1x1xf32>
    %v7062 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7063 = stablehlo.multiply %v7062, %v7061 : tensor<32x192x1x1xf32>
    %v7064 = stablehlo.subtract %b5pW, %v7063 : tensor<32x192x1x1xf32>
    %v7065 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7066 = stablehlo.multiply %v7065, %b5pg : tensor<32xf32>
    %v7067 = stablehlo.add %v7066, %v4333 : tensor<32xf32>
    %v7068 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7069 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7070 = stablehlo.multiply %v7068, %b5pgv : tensor<32xf32>
    %v7071 = stablehlo.multiply %v7067, %v7067 : tensor<32xf32>
    %v7072 = stablehlo.multiply %v7069, %v7071 : tensor<32xf32>
    %v7073 = stablehlo.add %v7070, %v7072 : tensor<32xf32>
    %v7074 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7075 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7076 = stablehlo.multiply %v7074, %b5pgv : tensor<32xf32>
    %v7077 = stablehlo.multiply %v7067, %v7067 : tensor<32xf32>
    %v7078 = stablehlo.multiply %v7075, %v7077 : tensor<32xf32>
    %v7079 = stablehlo.add %v7076, %v7078 : tensor<32xf32>
    %v7080 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7081 = stablehlo.add %v7079, %v7080 : tensor<32xf32>
    %v7082 = stablehlo.sqrt %v7081 : tensor<32xf32>
    %v7083 = stablehlo.divide %v7067, %v7082 : tensor<32xf32>
    %v7084 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7085 = stablehlo.multiply %v7084, %b5pgm : tensor<32xf32>
    %v7086 = stablehlo.add %v7085, %v7083 : tensor<32xf32>
    %v7087 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7088 = stablehlo.multiply %v7087, %v7086 : tensor<32xf32>
    %v7089 = stablehlo.subtract %b5pg, %v7088 : tensor<32xf32>
    %v7090 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7091 = stablehlo.multiply %v7090, %b5pbt : tensor<32xf32>
    %v7092 = stablehlo.add %v7091, %v4336 : tensor<32xf32>
    %v7093 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7094 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7095 = stablehlo.multiply %v7093, %b5pbtv : tensor<32xf32>
    %v7096 = stablehlo.multiply %v7092, %v7092 : tensor<32xf32>
    %v7097 = stablehlo.multiply %v7094, %v7096 : tensor<32xf32>
    %v7098 = stablehlo.add %v7095, %v7097 : tensor<32xf32>
    %v7099 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7100 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7101 = stablehlo.multiply %v7099, %b5pbtv : tensor<32xf32>
    %v7102 = stablehlo.multiply %v7092, %v7092 : tensor<32xf32>
    %v7103 = stablehlo.multiply %v7100, %v7102 : tensor<32xf32>
    %v7104 = stablehlo.add %v7101, %v7103 : tensor<32xf32>
    %v7105 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7106 = stablehlo.add %v7104, %v7105 : tensor<32xf32>
    %v7107 = stablehlo.sqrt %v7106 : tensor<32xf32>
    %v7108 = stablehlo.divide %v7092, %v7107 : tensor<32xf32>
    %v7109 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7110 = stablehlo.multiply %v7109, %b5pbtm : tensor<32xf32>
    %v7111 = stablehlo.add %v7110, %v7108 : tensor<32xf32>
    %v7112 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7113 = stablehlo.multiply %v7112, %v7111 : tensor<32xf32>
    %v7114 = stablehlo.subtract %b5pbt, %v7113 : tensor<32xf32>
    %v7115 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7116 = stablehlo.multiply %v7115, %b6eW : tensor<192x32x1x1xf32>
    %v7117 = stablehlo.add %v7116, %v4054 : tensor<192x32x1x1xf32>
    %v7118 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7119 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7120 = stablehlo.multiply %v7118, %b6eWv : tensor<192x32x1x1xf32>
    %v7121 = stablehlo.multiply %v7117, %v7117 : tensor<192x32x1x1xf32>
    %v7122 = stablehlo.multiply %v7119, %v7121 : tensor<192x32x1x1xf32>
    %v7123 = stablehlo.add %v7120, %v7122 : tensor<192x32x1x1xf32>
    %v7124 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7125 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7126 = stablehlo.multiply %v7124, %b6eWv : tensor<192x32x1x1xf32>
    %v7127 = stablehlo.multiply %v7117, %v7117 : tensor<192x32x1x1xf32>
    %v7128 = stablehlo.multiply %v7125, %v7127 : tensor<192x32x1x1xf32>
    %v7129 = stablehlo.add %v7126, %v7128 : tensor<192x32x1x1xf32>
    %v7130 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7131 = stablehlo.add %v7129, %v7130 : tensor<192x32x1x1xf32>
    %v7132 = stablehlo.sqrt %v7131 : tensor<192x32x1x1xf32>
    %v7133 = stablehlo.divide %v7117, %v7132 : tensor<192x32x1x1xf32>
    %v7134 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7135 = stablehlo.multiply %v7134, %b6eWm : tensor<192x32x1x1xf32>
    %v7136 = stablehlo.add %v7135, %v7133 : tensor<192x32x1x1xf32>
    %v7137 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7138 = stablehlo.multiply %v7137, %v7136 : tensor<192x32x1x1xf32>
    %v7139 = stablehlo.subtract %b6eW, %v7138 : tensor<192x32x1x1xf32>
    %v7140 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7141 = stablehlo.multiply %v7140, %b6eg : tensor<192xf32>
    %v7142 = stablehlo.add %v7141, %v4072 : tensor<192xf32>
    %v7143 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7144 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7145 = stablehlo.multiply %v7143, %b6egv : tensor<192xf32>
    %v7146 = stablehlo.multiply %v7142, %v7142 : tensor<192xf32>
    %v7147 = stablehlo.multiply %v7144, %v7146 : tensor<192xf32>
    %v7148 = stablehlo.add %v7145, %v7147 : tensor<192xf32>
    %v7149 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7150 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7151 = stablehlo.multiply %v7149, %b6egv : tensor<192xf32>
    %v7152 = stablehlo.multiply %v7142, %v7142 : tensor<192xf32>
    %v7153 = stablehlo.multiply %v7150, %v7152 : tensor<192xf32>
    %v7154 = stablehlo.add %v7151, %v7153 : tensor<192xf32>
    %v7155 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7156 = stablehlo.add %v7154, %v7155 : tensor<192xf32>
    %v7157 = stablehlo.sqrt %v7156 : tensor<192xf32>
    %v7158 = stablehlo.divide %v7142, %v7157 : tensor<192xf32>
    %v7159 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7160 = stablehlo.multiply %v7159, %b6egm : tensor<192xf32>
    %v7161 = stablehlo.add %v7160, %v7158 : tensor<192xf32>
    %v7162 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7163 = stablehlo.multiply %v7162, %v7161 : tensor<192xf32>
    %v7164 = stablehlo.subtract %b6eg, %v7163 : tensor<192xf32>
    %v7165 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7166 = stablehlo.multiply %v7165, %b6ebt : tensor<192xf32>
    %v7167 = stablehlo.add %v7166, %v4075 : tensor<192xf32>
    %v7168 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7169 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7170 = stablehlo.multiply %v7168, %b6ebtv : tensor<192xf32>
    %v7171 = stablehlo.multiply %v7167, %v7167 : tensor<192xf32>
    %v7172 = stablehlo.multiply %v7169, %v7171 : tensor<192xf32>
    %v7173 = stablehlo.add %v7170, %v7172 : tensor<192xf32>
    %v7174 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7175 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7176 = stablehlo.multiply %v7174, %b6ebtv : tensor<192xf32>
    %v7177 = stablehlo.multiply %v7167, %v7167 : tensor<192xf32>
    %v7178 = stablehlo.multiply %v7175, %v7177 : tensor<192xf32>
    %v7179 = stablehlo.add %v7176, %v7178 : tensor<192xf32>
    %v7180 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7181 = stablehlo.add %v7179, %v7180 : tensor<192xf32>
    %v7182 = stablehlo.sqrt %v7181 : tensor<192xf32>
    %v7183 = stablehlo.divide %v7167, %v7182 : tensor<192xf32>
    %v7184 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7185 = stablehlo.multiply %v7184, %b6ebtm : tensor<192xf32>
    %v7186 = stablehlo.add %v7185, %v7183 : tensor<192xf32>
    %v7187 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7188 = stablehlo.multiply %v7187, %v7186 : tensor<192xf32>
    %v7189 = stablehlo.subtract %b6ebt, %v7188 : tensor<192xf32>
    %v7190 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7191 = stablehlo.multiply %v7190, %b6dW : tensor<192x1x3x3xf32>
    %v7192 = stablehlo.add %v7191, %v4081 : tensor<192x1x3x3xf32>
    %v7193 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7194 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7195 = stablehlo.multiply %v7193, %b6dWv : tensor<192x1x3x3xf32>
    %v7196 = stablehlo.multiply %v7192, %v7192 : tensor<192x1x3x3xf32>
    %v7197 = stablehlo.multiply %v7194, %v7196 : tensor<192x1x3x3xf32>
    %v7198 = stablehlo.add %v7195, %v7197 : tensor<192x1x3x3xf32>
    %v7199 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7200 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7201 = stablehlo.multiply %v7199, %b6dWv : tensor<192x1x3x3xf32>
    %v7202 = stablehlo.multiply %v7192, %v7192 : tensor<192x1x3x3xf32>
    %v7203 = stablehlo.multiply %v7200, %v7202 : tensor<192x1x3x3xf32>
    %v7204 = stablehlo.add %v7201, %v7203 : tensor<192x1x3x3xf32>
    %v7205 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7206 = stablehlo.add %v7204, %v7205 : tensor<192x1x3x3xf32>
    %v7207 = stablehlo.sqrt %v7206 : tensor<192x1x3x3xf32>
    %v7208 = stablehlo.divide %v7192, %v7207 : tensor<192x1x3x3xf32>
    %v7209 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7210 = stablehlo.multiply %v7209, %b6dWm : tensor<192x1x3x3xf32>
    %v7211 = stablehlo.add %v7210, %v7208 : tensor<192x1x3x3xf32>
    %v7212 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7213 = stablehlo.multiply %v7212, %v7211 : tensor<192x1x3x3xf32>
    %v7214 = stablehlo.subtract %b6dW, %v7213 : tensor<192x1x3x3xf32>
    %v7215 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7216 = stablehlo.multiply %v7215, %b6dg : tensor<192xf32>
    %v7217 = stablehlo.add %v7216, %v4099 : tensor<192xf32>
    %v7218 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7219 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7220 = stablehlo.multiply %v7218, %b6dgv : tensor<192xf32>
    %v7221 = stablehlo.multiply %v7217, %v7217 : tensor<192xf32>
    %v7222 = stablehlo.multiply %v7219, %v7221 : tensor<192xf32>
    %v7223 = stablehlo.add %v7220, %v7222 : tensor<192xf32>
    %v7224 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7225 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7226 = stablehlo.multiply %v7224, %b6dgv : tensor<192xf32>
    %v7227 = stablehlo.multiply %v7217, %v7217 : tensor<192xf32>
    %v7228 = stablehlo.multiply %v7225, %v7227 : tensor<192xf32>
    %v7229 = stablehlo.add %v7226, %v7228 : tensor<192xf32>
    %v7230 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7231 = stablehlo.add %v7229, %v7230 : tensor<192xf32>
    %v7232 = stablehlo.sqrt %v7231 : tensor<192xf32>
    %v7233 = stablehlo.divide %v7217, %v7232 : tensor<192xf32>
    %v7234 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7235 = stablehlo.multiply %v7234, %b6dgm : tensor<192xf32>
    %v7236 = stablehlo.add %v7235, %v7233 : tensor<192xf32>
    %v7237 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7238 = stablehlo.multiply %v7237, %v7236 : tensor<192xf32>
    %v7239 = stablehlo.subtract %b6dg, %v7238 : tensor<192xf32>
    %v7240 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7241 = stablehlo.multiply %v7240, %b6dbt : tensor<192xf32>
    %v7242 = stablehlo.add %v7241, %v4102 : tensor<192xf32>
    %v7243 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7244 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7245 = stablehlo.multiply %v7243, %b6dbtv : tensor<192xf32>
    %v7246 = stablehlo.multiply %v7242, %v7242 : tensor<192xf32>
    %v7247 = stablehlo.multiply %v7244, %v7246 : tensor<192xf32>
    %v7248 = stablehlo.add %v7245, %v7247 : tensor<192xf32>
    %v7249 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7250 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7251 = stablehlo.multiply %v7249, %b6dbtv : tensor<192xf32>
    %v7252 = stablehlo.multiply %v7242, %v7242 : tensor<192xf32>
    %v7253 = stablehlo.multiply %v7250, %v7252 : tensor<192xf32>
    %v7254 = stablehlo.add %v7251, %v7253 : tensor<192xf32>
    %v7255 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7256 = stablehlo.add %v7254, %v7255 : tensor<192xf32>
    %v7257 = stablehlo.sqrt %v7256 : tensor<192xf32>
    %v7258 = stablehlo.divide %v7242, %v7257 : tensor<192xf32>
    %v7259 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7260 = stablehlo.multiply %v7259, %b6dbtm : tensor<192xf32>
    %v7261 = stablehlo.add %v7260, %v7258 : tensor<192xf32>
    %v7262 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7263 = stablehlo.multiply %v7262, %v7261 : tensor<192xf32>
    %v7264 = stablehlo.subtract %b6dbt, %v7263 : tensor<192xf32>
    %v7265 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7266 = stablehlo.multiply %v7265, %b6pW : tensor<32x192x1x1xf32>
    %v7267 = stablehlo.add %v7266, %v4108 : tensor<32x192x1x1xf32>
    %v7268 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7269 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7270 = stablehlo.multiply %v7268, %b6pWv : tensor<32x192x1x1xf32>
    %v7271 = stablehlo.multiply %v7267, %v7267 : tensor<32x192x1x1xf32>
    %v7272 = stablehlo.multiply %v7269, %v7271 : tensor<32x192x1x1xf32>
    %v7273 = stablehlo.add %v7270, %v7272 : tensor<32x192x1x1xf32>
    %v7274 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7275 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7276 = stablehlo.multiply %v7274, %b6pWv : tensor<32x192x1x1xf32>
    %v7277 = stablehlo.multiply %v7267, %v7267 : tensor<32x192x1x1xf32>
    %v7278 = stablehlo.multiply %v7275, %v7277 : tensor<32x192x1x1xf32>
    %v7279 = stablehlo.add %v7276, %v7278 : tensor<32x192x1x1xf32>
    %v7280 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7281 = stablehlo.add %v7279, %v7280 : tensor<32x192x1x1xf32>
    %v7282 = stablehlo.sqrt %v7281 : tensor<32x192x1x1xf32>
    %v7283 = stablehlo.divide %v7267, %v7282 : tensor<32x192x1x1xf32>
    %v7284 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7285 = stablehlo.multiply %v7284, %b6pWm : tensor<32x192x1x1xf32>
    %v7286 = stablehlo.add %v7285, %v7283 : tensor<32x192x1x1xf32>
    %v7287 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x192x1x1xf32>
    %v7288 = stablehlo.multiply %v7287, %v7286 : tensor<32x192x1x1xf32>
    %v7289 = stablehlo.subtract %b6pW, %v7288 : tensor<32x192x1x1xf32>
    %v7290 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7291 = stablehlo.multiply %v7290, %b6pg : tensor<32xf32>
    %v7292 = stablehlo.add %v7291, %v4126 : tensor<32xf32>
    %v7293 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7294 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7295 = stablehlo.multiply %v7293, %b6pgv : tensor<32xf32>
    %v7296 = stablehlo.multiply %v7292, %v7292 : tensor<32xf32>
    %v7297 = stablehlo.multiply %v7294, %v7296 : tensor<32xf32>
    %v7298 = stablehlo.add %v7295, %v7297 : tensor<32xf32>
    %v7299 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7300 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7301 = stablehlo.multiply %v7299, %b6pgv : tensor<32xf32>
    %v7302 = stablehlo.multiply %v7292, %v7292 : tensor<32xf32>
    %v7303 = stablehlo.multiply %v7300, %v7302 : tensor<32xf32>
    %v7304 = stablehlo.add %v7301, %v7303 : tensor<32xf32>
    %v7305 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7306 = stablehlo.add %v7304, %v7305 : tensor<32xf32>
    %v7307 = stablehlo.sqrt %v7306 : tensor<32xf32>
    %v7308 = stablehlo.divide %v7292, %v7307 : tensor<32xf32>
    %v7309 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7310 = stablehlo.multiply %v7309, %b6pgm : tensor<32xf32>
    %v7311 = stablehlo.add %v7310, %v7308 : tensor<32xf32>
    %v7312 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7313 = stablehlo.multiply %v7312, %v7311 : tensor<32xf32>
    %v7314 = stablehlo.subtract %b6pg, %v7313 : tensor<32xf32>
    %v7315 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7316 = stablehlo.multiply %v7315, %b6pbt : tensor<32xf32>
    %v7317 = stablehlo.add %v7316, %v4129 : tensor<32xf32>
    %v7318 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7319 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7320 = stablehlo.multiply %v7318, %b6pbtv : tensor<32xf32>
    %v7321 = stablehlo.multiply %v7317, %v7317 : tensor<32xf32>
    %v7322 = stablehlo.multiply %v7319, %v7321 : tensor<32xf32>
    %v7323 = stablehlo.add %v7320, %v7322 : tensor<32xf32>
    %v7324 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7325 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7326 = stablehlo.multiply %v7324, %b6pbtv : tensor<32xf32>
    %v7327 = stablehlo.multiply %v7317, %v7317 : tensor<32xf32>
    %v7328 = stablehlo.multiply %v7325, %v7327 : tensor<32xf32>
    %v7329 = stablehlo.add %v7326, %v7328 : tensor<32xf32>
    %v7330 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7331 = stablehlo.add %v7329, %v7330 : tensor<32xf32>
    %v7332 = stablehlo.sqrt %v7331 : tensor<32xf32>
    %v7333 = stablehlo.divide %v7317, %v7332 : tensor<32xf32>
    %v7334 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7335 = stablehlo.multiply %v7334, %b6pbtm : tensor<32xf32>
    %v7336 = stablehlo.add %v7335, %v7333 : tensor<32xf32>
    %v7337 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v7338 = stablehlo.multiply %v7337, %v7336 : tensor<32xf32>
    %v7339 = stablehlo.subtract %b6pbt, %v7338 : tensor<32xf32>
    %v7340 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7341 = stablehlo.multiply %v7340, %b7eW : tensor<192x32x1x1xf32>
    %v7342 = stablehlo.add %v7341, %v3845 : tensor<192x32x1x1xf32>
    %v7343 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7344 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7345 = stablehlo.multiply %v7343, %b7eWv : tensor<192x32x1x1xf32>
    %v7346 = stablehlo.multiply %v7342, %v7342 : tensor<192x32x1x1xf32>
    %v7347 = stablehlo.multiply %v7344, %v7346 : tensor<192x32x1x1xf32>
    %v7348 = stablehlo.add %v7345, %v7347 : tensor<192x32x1x1xf32>
    %v7349 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7350 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7351 = stablehlo.multiply %v7349, %b7eWv : tensor<192x32x1x1xf32>
    %v7352 = stablehlo.multiply %v7342, %v7342 : tensor<192x32x1x1xf32>
    %v7353 = stablehlo.multiply %v7350, %v7352 : tensor<192x32x1x1xf32>
    %v7354 = stablehlo.add %v7351, %v7353 : tensor<192x32x1x1xf32>
    %v7355 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7356 = stablehlo.add %v7354, %v7355 : tensor<192x32x1x1xf32>
    %v7357 = stablehlo.sqrt %v7356 : tensor<192x32x1x1xf32>
    %v7358 = stablehlo.divide %v7342, %v7357 : tensor<192x32x1x1xf32>
    %v7359 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7360 = stablehlo.multiply %v7359, %b7eWm : tensor<192x32x1x1xf32>
    %v7361 = stablehlo.add %v7360, %v7358 : tensor<192x32x1x1xf32>
    %v7362 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x32x1x1xf32>
    %v7363 = stablehlo.multiply %v7362, %v7361 : tensor<192x32x1x1xf32>
    %v7364 = stablehlo.subtract %b7eW, %v7363 : tensor<192x32x1x1xf32>
    %v7365 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7366 = stablehlo.multiply %v7365, %b7eg : tensor<192xf32>
    %v7367 = stablehlo.add %v7366, %v3863 : tensor<192xf32>
    %v7368 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7369 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7370 = stablehlo.multiply %v7368, %b7egv : tensor<192xf32>
    %v7371 = stablehlo.multiply %v7367, %v7367 : tensor<192xf32>
    %v7372 = stablehlo.multiply %v7369, %v7371 : tensor<192xf32>
    %v7373 = stablehlo.add %v7370, %v7372 : tensor<192xf32>
    %v7374 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7375 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7376 = stablehlo.multiply %v7374, %b7egv : tensor<192xf32>
    %v7377 = stablehlo.multiply %v7367, %v7367 : tensor<192xf32>
    %v7378 = stablehlo.multiply %v7375, %v7377 : tensor<192xf32>
    %v7379 = stablehlo.add %v7376, %v7378 : tensor<192xf32>
    %v7380 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7381 = stablehlo.add %v7379, %v7380 : tensor<192xf32>
    %v7382 = stablehlo.sqrt %v7381 : tensor<192xf32>
    %v7383 = stablehlo.divide %v7367, %v7382 : tensor<192xf32>
    %v7384 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7385 = stablehlo.multiply %v7384, %b7egm : tensor<192xf32>
    %v7386 = stablehlo.add %v7385, %v7383 : tensor<192xf32>
    %v7387 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7388 = stablehlo.multiply %v7387, %v7386 : tensor<192xf32>
    %v7389 = stablehlo.subtract %b7eg, %v7388 : tensor<192xf32>
    %v7390 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7391 = stablehlo.multiply %v7390, %b7ebt : tensor<192xf32>
    %v7392 = stablehlo.add %v7391, %v3866 : tensor<192xf32>
    %v7393 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7394 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7395 = stablehlo.multiply %v7393, %b7ebtv : tensor<192xf32>
    %v7396 = stablehlo.multiply %v7392, %v7392 : tensor<192xf32>
    %v7397 = stablehlo.multiply %v7394, %v7396 : tensor<192xf32>
    %v7398 = stablehlo.add %v7395, %v7397 : tensor<192xf32>
    %v7399 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7400 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7401 = stablehlo.multiply %v7399, %b7ebtv : tensor<192xf32>
    %v7402 = stablehlo.multiply %v7392, %v7392 : tensor<192xf32>
    %v7403 = stablehlo.multiply %v7400, %v7402 : tensor<192xf32>
    %v7404 = stablehlo.add %v7401, %v7403 : tensor<192xf32>
    %v7405 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7406 = stablehlo.add %v7404, %v7405 : tensor<192xf32>
    %v7407 = stablehlo.sqrt %v7406 : tensor<192xf32>
    %v7408 = stablehlo.divide %v7392, %v7407 : tensor<192xf32>
    %v7409 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7410 = stablehlo.multiply %v7409, %b7ebtm : tensor<192xf32>
    %v7411 = stablehlo.add %v7410, %v7408 : tensor<192xf32>
    %v7412 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7413 = stablehlo.multiply %v7412, %v7411 : tensor<192xf32>
    %v7414 = stablehlo.subtract %b7ebt, %v7413 : tensor<192xf32>
    %v7415 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7416 = stablehlo.multiply %v7415, %b7dW : tensor<192x1x3x3xf32>
    %v7417 = stablehlo.add %v7416, %v3874 : tensor<192x1x3x3xf32>
    %v7418 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7419 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7420 = stablehlo.multiply %v7418, %b7dWv : tensor<192x1x3x3xf32>
    %v7421 = stablehlo.multiply %v7417, %v7417 : tensor<192x1x3x3xf32>
    %v7422 = stablehlo.multiply %v7419, %v7421 : tensor<192x1x3x3xf32>
    %v7423 = stablehlo.add %v7420, %v7422 : tensor<192x1x3x3xf32>
    %v7424 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7425 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7426 = stablehlo.multiply %v7424, %b7dWv : tensor<192x1x3x3xf32>
    %v7427 = stablehlo.multiply %v7417, %v7417 : tensor<192x1x3x3xf32>
    %v7428 = stablehlo.multiply %v7425, %v7427 : tensor<192x1x3x3xf32>
    %v7429 = stablehlo.add %v7426, %v7428 : tensor<192x1x3x3xf32>
    %v7430 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7431 = stablehlo.add %v7429, %v7430 : tensor<192x1x3x3xf32>
    %v7432 = stablehlo.sqrt %v7431 : tensor<192x1x3x3xf32>
    %v7433 = stablehlo.divide %v7417, %v7432 : tensor<192x1x3x3xf32>
    %v7434 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7435 = stablehlo.multiply %v7434, %b7dWm : tensor<192x1x3x3xf32>
    %v7436 = stablehlo.add %v7435, %v7433 : tensor<192x1x3x3xf32>
    %v7437 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x3x3xf32>
    %v7438 = stablehlo.multiply %v7437, %v7436 : tensor<192x1x3x3xf32>
    %v7439 = stablehlo.subtract %b7dW, %v7438 : tensor<192x1x3x3xf32>
    %v7440 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7441 = stablehlo.multiply %v7440, %b7dg : tensor<192xf32>
    %v7442 = stablehlo.add %v7441, %v3892 : tensor<192xf32>
    %v7443 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7444 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7445 = stablehlo.multiply %v7443, %b7dgv : tensor<192xf32>
    %v7446 = stablehlo.multiply %v7442, %v7442 : tensor<192xf32>
    %v7447 = stablehlo.multiply %v7444, %v7446 : tensor<192xf32>
    %v7448 = stablehlo.add %v7445, %v7447 : tensor<192xf32>
    %v7449 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7450 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7451 = stablehlo.multiply %v7449, %b7dgv : tensor<192xf32>
    %v7452 = stablehlo.multiply %v7442, %v7442 : tensor<192xf32>
    %v7453 = stablehlo.multiply %v7450, %v7452 : tensor<192xf32>
    %v7454 = stablehlo.add %v7451, %v7453 : tensor<192xf32>
    %v7455 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7456 = stablehlo.add %v7454, %v7455 : tensor<192xf32>
    %v7457 = stablehlo.sqrt %v7456 : tensor<192xf32>
    %v7458 = stablehlo.divide %v7442, %v7457 : tensor<192xf32>
    %v7459 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7460 = stablehlo.multiply %v7459, %b7dgm : tensor<192xf32>
    %v7461 = stablehlo.add %v7460, %v7458 : tensor<192xf32>
    %v7462 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7463 = stablehlo.multiply %v7462, %v7461 : tensor<192xf32>
    %v7464 = stablehlo.subtract %b7dg, %v7463 : tensor<192xf32>
    %v7465 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7466 = stablehlo.multiply %v7465, %b7dbt : tensor<192xf32>
    %v7467 = stablehlo.add %v7466, %v3895 : tensor<192xf32>
    %v7468 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7469 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7470 = stablehlo.multiply %v7468, %b7dbtv : tensor<192xf32>
    %v7471 = stablehlo.multiply %v7467, %v7467 : tensor<192xf32>
    %v7472 = stablehlo.multiply %v7469, %v7471 : tensor<192xf32>
    %v7473 = stablehlo.add %v7470, %v7472 : tensor<192xf32>
    %v7474 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7475 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7476 = stablehlo.multiply %v7474, %b7dbtv : tensor<192xf32>
    %v7477 = stablehlo.multiply %v7467, %v7467 : tensor<192xf32>
    %v7478 = stablehlo.multiply %v7475, %v7477 : tensor<192xf32>
    %v7479 = stablehlo.add %v7476, %v7478 : tensor<192xf32>
    %v7480 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7481 = stablehlo.add %v7479, %v7480 : tensor<192xf32>
    %v7482 = stablehlo.sqrt %v7481 : tensor<192xf32>
    %v7483 = stablehlo.divide %v7467, %v7482 : tensor<192xf32>
    %v7484 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7485 = stablehlo.multiply %v7484, %b7dbtm : tensor<192xf32>
    %v7486 = stablehlo.add %v7485, %v7483 : tensor<192xf32>
    %v7487 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v7488 = stablehlo.multiply %v7487, %v7486 : tensor<192xf32>
    %v7489 = stablehlo.subtract %b7dbt, %v7488 : tensor<192xf32>
    %v7490 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7491 = stablehlo.multiply %v7490, %b7pW : tensor<64x192x1x1xf32>
    %v7492 = stablehlo.add %v7491, %v3901 : tensor<64x192x1x1xf32>
    %v7493 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7494 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7495 = stablehlo.multiply %v7493, %b7pWv : tensor<64x192x1x1xf32>
    %v7496 = stablehlo.multiply %v7492, %v7492 : tensor<64x192x1x1xf32>
    %v7497 = stablehlo.multiply %v7494, %v7496 : tensor<64x192x1x1xf32>
    %v7498 = stablehlo.add %v7495, %v7497 : tensor<64x192x1x1xf32>
    %v7499 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7500 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7501 = stablehlo.multiply %v7499, %b7pWv : tensor<64x192x1x1xf32>
    %v7502 = stablehlo.multiply %v7492, %v7492 : tensor<64x192x1x1xf32>
    %v7503 = stablehlo.multiply %v7500, %v7502 : tensor<64x192x1x1xf32>
    %v7504 = stablehlo.add %v7501, %v7503 : tensor<64x192x1x1xf32>
    %v7505 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7506 = stablehlo.add %v7504, %v7505 : tensor<64x192x1x1xf32>
    %v7507 = stablehlo.sqrt %v7506 : tensor<64x192x1x1xf32>
    %v7508 = stablehlo.divide %v7492, %v7507 : tensor<64x192x1x1xf32>
    %v7509 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7510 = stablehlo.multiply %v7509, %b7pWm : tensor<64x192x1x1xf32>
    %v7511 = stablehlo.add %v7510, %v7508 : tensor<64x192x1x1xf32>
    %v7512 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x192x1x1xf32>
    %v7513 = stablehlo.multiply %v7512, %v7511 : tensor<64x192x1x1xf32>
    %v7514 = stablehlo.subtract %b7pW, %v7513 : tensor<64x192x1x1xf32>
    %v7515 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7516 = stablehlo.multiply %v7515, %b7pg : tensor<64xf32>
    %v7517 = stablehlo.add %v7516, %v3919 : tensor<64xf32>
    %v7518 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7519 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7520 = stablehlo.multiply %v7518, %b7pgv : tensor<64xf32>
    %v7521 = stablehlo.multiply %v7517, %v7517 : tensor<64xf32>
    %v7522 = stablehlo.multiply %v7519, %v7521 : tensor<64xf32>
    %v7523 = stablehlo.add %v7520, %v7522 : tensor<64xf32>
    %v7524 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7525 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7526 = stablehlo.multiply %v7524, %b7pgv : tensor<64xf32>
    %v7527 = stablehlo.multiply %v7517, %v7517 : tensor<64xf32>
    %v7528 = stablehlo.multiply %v7525, %v7527 : tensor<64xf32>
    %v7529 = stablehlo.add %v7526, %v7528 : tensor<64xf32>
    %v7530 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7531 = stablehlo.add %v7529, %v7530 : tensor<64xf32>
    %v7532 = stablehlo.sqrt %v7531 : tensor<64xf32>
    %v7533 = stablehlo.divide %v7517, %v7532 : tensor<64xf32>
    %v7534 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7535 = stablehlo.multiply %v7534, %b7pgm : tensor<64xf32>
    %v7536 = stablehlo.add %v7535, %v7533 : tensor<64xf32>
    %v7537 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7538 = stablehlo.multiply %v7537, %v7536 : tensor<64xf32>
    %v7539 = stablehlo.subtract %b7pg, %v7538 : tensor<64xf32>
    %v7540 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7541 = stablehlo.multiply %v7540, %b7pbt : tensor<64xf32>
    %v7542 = stablehlo.add %v7541, %v3922 : tensor<64xf32>
    %v7543 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7544 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7545 = stablehlo.multiply %v7543, %b7pbtv : tensor<64xf32>
    %v7546 = stablehlo.multiply %v7542, %v7542 : tensor<64xf32>
    %v7547 = stablehlo.multiply %v7544, %v7546 : tensor<64xf32>
    %v7548 = stablehlo.add %v7545, %v7547 : tensor<64xf32>
    %v7549 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7550 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7551 = stablehlo.multiply %v7549, %b7pbtv : tensor<64xf32>
    %v7552 = stablehlo.multiply %v7542, %v7542 : tensor<64xf32>
    %v7553 = stablehlo.multiply %v7550, %v7552 : tensor<64xf32>
    %v7554 = stablehlo.add %v7551, %v7553 : tensor<64xf32>
    %v7555 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7556 = stablehlo.add %v7554, %v7555 : tensor<64xf32>
    %v7557 = stablehlo.sqrt %v7556 : tensor<64xf32>
    %v7558 = stablehlo.divide %v7542, %v7557 : tensor<64xf32>
    %v7559 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7560 = stablehlo.multiply %v7559, %b7pbtm : tensor<64xf32>
    %v7561 = stablehlo.add %v7560, %v7558 : tensor<64xf32>
    %v7562 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7563 = stablehlo.multiply %v7562, %v7561 : tensor<64xf32>
    %v7564 = stablehlo.subtract %b7pbt, %v7563 : tensor<64xf32>
    %v7565 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7566 = stablehlo.multiply %v7565, %b8eW : tensor<384x64x1x1xf32>
    %v7567 = stablehlo.add %v7566, %v3640 : tensor<384x64x1x1xf32>
    %v7568 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7569 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7570 = stablehlo.multiply %v7568, %b8eWv : tensor<384x64x1x1xf32>
    %v7571 = stablehlo.multiply %v7567, %v7567 : tensor<384x64x1x1xf32>
    %v7572 = stablehlo.multiply %v7569, %v7571 : tensor<384x64x1x1xf32>
    %v7573 = stablehlo.add %v7570, %v7572 : tensor<384x64x1x1xf32>
    %v7574 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7575 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7576 = stablehlo.multiply %v7574, %b8eWv : tensor<384x64x1x1xf32>
    %v7577 = stablehlo.multiply %v7567, %v7567 : tensor<384x64x1x1xf32>
    %v7578 = stablehlo.multiply %v7575, %v7577 : tensor<384x64x1x1xf32>
    %v7579 = stablehlo.add %v7576, %v7578 : tensor<384x64x1x1xf32>
    %v7580 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7581 = stablehlo.add %v7579, %v7580 : tensor<384x64x1x1xf32>
    %v7582 = stablehlo.sqrt %v7581 : tensor<384x64x1x1xf32>
    %v7583 = stablehlo.divide %v7567, %v7582 : tensor<384x64x1x1xf32>
    %v7584 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7585 = stablehlo.multiply %v7584, %b8eWm : tensor<384x64x1x1xf32>
    %v7586 = stablehlo.add %v7585, %v7583 : tensor<384x64x1x1xf32>
    %v7587 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7588 = stablehlo.multiply %v7587, %v7586 : tensor<384x64x1x1xf32>
    %v7589 = stablehlo.subtract %b8eW, %v7588 : tensor<384x64x1x1xf32>
    %v7590 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7591 = stablehlo.multiply %v7590, %b8eg : tensor<384xf32>
    %v7592 = stablehlo.add %v7591, %v3658 : tensor<384xf32>
    %v7593 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7594 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7595 = stablehlo.multiply %v7593, %b8egv : tensor<384xf32>
    %v7596 = stablehlo.multiply %v7592, %v7592 : tensor<384xf32>
    %v7597 = stablehlo.multiply %v7594, %v7596 : tensor<384xf32>
    %v7598 = stablehlo.add %v7595, %v7597 : tensor<384xf32>
    %v7599 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7600 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7601 = stablehlo.multiply %v7599, %b8egv : tensor<384xf32>
    %v7602 = stablehlo.multiply %v7592, %v7592 : tensor<384xf32>
    %v7603 = stablehlo.multiply %v7600, %v7602 : tensor<384xf32>
    %v7604 = stablehlo.add %v7601, %v7603 : tensor<384xf32>
    %v7605 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7606 = stablehlo.add %v7604, %v7605 : tensor<384xf32>
    %v7607 = stablehlo.sqrt %v7606 : tensor<384xf32>
    %v7608 = stablehlo.divide %v7592, %v7607 : tensor<384xf32>
    %v7609 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7610 = stablehlo.multiply %v7609, %b8egm : tensor<384xf32>
    %v7611 = stablehlo.add %v7610, %v7608 : tensor<384xf32>
    %v7612 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7613 = stablehlo.multiply %v7612, %v7611 : tensor<384xf32>
    %v7614 = stablehlo.subtract %b8eg, %v7613 : tensor<384xf32>
    %v7615 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7616 = stablehlo.multiply %v7615, %b8ebt : tensor<384xf32>
    %v7617 = stablehlo.add %v7616, %v3661 : tensor<384xf32>
    %v7618 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7619 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7620 = stablehlo.multiply %v7618, %b8ebtv : tensor<384xf32>
    %v7621 = stablehlo.multiply %v7617, %v7617 : tensor<384xf32>
    %v7622 = stablehlo.multiply %v7619, %v7621 : tensor<384xf32>
    %v7623 = stablehlo.add %v7620, %v7622 : tensor<384xf32>
    %v7624 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7625 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7626 = stablehlo.multiply %v7624, %b8ebtv : tensor<384xf32>
    %v7627 = stablehlo.multiply %v7617, %v7617 : tensor<384xf32>
    %v7628 = stablehlo.multiply %v7625, %v7627 : tensor<384xf32>
    %v7629 = stablehlo.add %v7626, %v7628 : tensor<384xf32>
    %v7630 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7631 = stablehlo.add %v7629, %v7630 : tensor<384xf32>
    %v7632 = stablehlo.sqrt %v7631 : tensor<384xf32>
    %v7633 = stablehlo.divide %v7617, %v7632 : tensor<384xf32>
    %v7634 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7635 = stablehlo.multiply %v7634, %b8ebtm : tensor<384xf32>
    %v7636 = stablehlo.add %v7635, %v7633 : tensor<384xf32>
    %v7637 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7638 = stablehlo.multiply %v7637, %v7636 : tensor<384xf32>
    %v7639 = stablehlo.subtract %b8ebt, %v7638 : tensor<384xf32>
    %v7640 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7641 = stablehlo.multiply %v7640, %b8dW : tensor<384x1x3x3xf32>
    %v7642 = stablehlo.add %v7641, %v3667 : tensor<384x1x3x3xf32>
    %v7643 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7644 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7645 = stablehlo.multiply %v7643, %b8dWv : tensor<384x1x3x3xf32>
    %v7646 = stablehlo.multiply %v7642, %v7642 : tensor<384x1x3x3xf32>
    %v7647 = stablehlo.multiply %v7644, %v7646 : tensor<384x1x3x3xf32>
    %v7648 = stablehlo.add %v7645, %v7647 : tensor<384x1x3x3xf32>
    %v7649 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7650 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7651 = stablehlo.multiply %v7649, %b8dWv : tensor<384x1x3x3xf32>
    %v7652 = stablehlo.multiply %v7642, %v7642 : tensor<384x1x3x3xf32>
    %v7653 = stablehlo.multiply %v7650, %v7652 : tensor<384x1x3x3xf32>
    %v7654 = stablehlo.add %v7651, %v7653 : tensor<384x1x3x3xf32>
    %v7655 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7656 = stablehlo.add %v7654, %v7655 : tensor<384x1x3x3xf32>
    %v7657 = stablehlo.sqrt %v7656 : tensor<384x1x3x3xf32>
    %v7658 = stablehlo.divide %v7642, %v7657 : tensor<384x1x3x3xf32>
    %v7659 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7660 = stablehlo.multiply %v7659, %b8dWm : tensor<384x1x3x3xf32>
    %v7661 = stablehlo.add %v7660, %v7658 : tensor<384x1x3x3xf32>
    %v7662 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7663 = stablehlo.multiply %v7662, %v7661 : tensor<384x1x3x3xf32>
    %v7664 = stablehlo.subtract %b8dW, %v7663 : tensor<384x1x3x3xf32>
    %v7665 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7666 = stablehlo.multiply %v7665, %b8dg : tensor<384xf32>
    %v7667 = stablehlo.add %v7666, %v3685 : tensor<384xf32>
    %v7668 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7669 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7670 = stablehlo.multiply %v7668, %b8dgv : tensor<384xf32>
    %v7671 = stablehlo.multiply %v7667, %v7667 : tensor<384xf32>
    %v7672 = stablehlo.multiply %v7669, %v7671 : tensor<384xf32>
    %v7673 = stablehlo.add %v7670, %v7672 : tensor<384xf32>
    %v7674 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7675 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7676 = stablehlo.multiply %v7674, %b8dgv : tensor<384xf32>
    %v7677 = stablehlo.multiply %v7667, %v7667 : tensor<384xf32>
    %v7678 = stablehlo.multiply %v7675, %v7677 : tensor<384xf32>
    %v7679 = stablehlo.add %v7676, %v7678 : tensor<384xf32>
    %v7680 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7681 = stablehlo.add %v7679, %v7680 : tensor<384xf32>
    %v7682 = stablehlo.sqrt %v7681 : tensor<384xf32>
    %v7683 = stablehlo.divide %v7667, %v7682 : tensor<384xf32>
    %v7684 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7685 = stablehlo.multiply %v7684, %b8dgm : tensor<384xf32>
    %v7686 = stablehlo.add %v7685, %v7683 : tensor<384xf32>
    %v7687 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7688 = stablehlo.multiply %v7687, %v7686 : tensor<384xf32>
    %v7689 = stablehlo.subtract %b8dg, %v7688 : tensor<384xf32>
    %v7690 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7691 = stablehlo.multiply %v7690, %b8dbt : tensor<384xf32>
    %v7692 = stablehlo.add %v7691, %v3688 : tensor<384xf32>
    %v7693 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7694 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7695 = stablehlo.multiply %v7693, %b8dbtv : tensor<384xf32>
    %v7696 = stablehlo.multiply %v7692, %v7692 : tensor<384xf32>
    %v7697 = stablehlo.multiply %v7694, %v7696 : tensor<384xf32>
    %v7698 = stablehlo.add %v7695, %v7697 : tensor<384xf32>
    %v7699 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7700 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7701 = stablehlo.multiply %v7699, %b8dbtv : tensor<384xf32>
    %v7702 = stablehlo.multiply %v7692, %v7692 : tensor<384xf32>
    %v7703 = stablehlo.multiply %v7700, %v7702 : tensor<384xf32>
    %v7704 = stablehlo.add %v7701, %v7703 : tensor<384xf32>
    %v7705 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7706 = stablehlo.add %v7704, %v7705 : tensor<384xf32>
    %v7707 = stablehlo.sqrt %v7706 : tensor<384xf32>
    %v7708 = stablehlo.divide %v7692, %v7707 : tensor<384xf32>
    %v7709 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7710 = stablehlo.multiply %v7709, %b8dbtm : tensor<384xf32>
    %v7711 = stablehlo.add %v7710, %v7708 : tensor<384xf32>
    %v7712 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7713 = stablehlo.multiply %v7712, %v7711 : tensor<384xf32>
    %v7714 = stablehlo.subtract %b8dbt, %v7713 : tensor<384xf32>
    %v7715 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7716 = stablehlo.multiply %v7715, %b8pW : tensor<64x384x1x1xf32>
    %v7717 = stablehlo.add %v7716, %v3694 : tensor<64x384x1x1xf32>
    %v7718 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7719 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7720 = stablehlo.multiply %v7718, %b8pWv : tensor<64x384x1x1xf32>
    %v7721 = stablehlo.multiply %v7717, %v7717 : tensor<64x384x1x1xf32>
    %v7722 = stablehlo.multiply %v7719, %v7721 : tensor<64x384x1x1xf32>
    %v7723 = stablehlo.add %v7720, %v7722 : tensor<64x384x1x1xf32>
    %v7724 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7725 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7726 = stablehlo.multiply %v7724, %b8pWv : tensor<64x384x1x1xf32>
    %v7727 = stablehlo.multiply %v7717, %v7717 : tensor<64x384x1x1xf32>
    %v7728 = stablehlo.multiply %v7725, %v7727 : tensor<64x384x1x1xf32>
    %v7729 = stablehlo.add %v7726, %v7728 : tensor<64x384x1x1xf32>
    %v7730 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7731 = stablehlo.add %v7729, %v7730 : tensor<64x384x1x1xf32>
    %v7732 = stablehlo.sqrt %v7731 : tensor<64x384x1x1xf32>
    %v7733 = stablehlo.divide %v7717, %v7732 : tensor<64x384x1x1xf32>
    %v7734 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7735 = stablehlo.multiply %v7734, %b8pWm : tensor<64x384x1x1xf32>
    %v7736 = stablehlo.add %v7735, %v7733 : tensor<64x384x1x1xf32>
    %v7737 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7738 = stablehlo.multiply %v7737, %v7736 : tensor<64x384x1x1xf32>
    %v7739 = stablehlo.subtract %b8pW, %v7738 : tensor<64x384x1x1xf32>
    %v7740 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7741 = stablehlo.multiply %v7740, %b8pg : tensor<64xf32>
    %v7742 = stablehlo.add %v7741, %v3712 : tensor<64xf32>
    %v7743 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7744 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7745 = stablehlo.multiply %v7743, %b8pgv : tensor<64xf32>
    %v7746 = stablehlo.multiply %v7742, %v7742 : tensor<64xf32>
    %v7747 = stablehlo.multiply %v7744, %v7746 : tensor<64xf32>
    %v7748 = stablehlo.add %v7745, %v7747 : tensor<64xf32>
    %v7749 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7750 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7751 = stablehlo.multiply %v7749, %b8pgv : tensor<64xf32>
    %v7752 = stablehlo.multiply %v7742, %v7742 : tensor<64xf32>
    %v7753 = stablehlo.multiply %v7750, %v7752 : tensor<64xf32>
    %v7754 = stablehlo.add %v7751, %v7753 : tensor<64xf32>
    %v7755 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7756 = stablehlo.add %v7754, %v7755 : tensor<64xf32>
    %v7757 = stablehlo.sqrt %v7756 : tensor<64xf32>
    %v7758 = stablehlo.divide %v7742, %v7757 : tensor<64xf32>
    %v7759 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7760 = stablehlo.multiply %v7759, %b8pgm : tensor<64xf32>
    %v7761 = stablehlo.add %v7760, %v7758 : tensor<64xf32>
    %v7762 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7763 = stablehlo.multiply %v7762, %v7761 : tensor<64xf32>
    %v7764 = stablehlo.subtract %b8pg, %v7763 : tensor<64xf32>
    %v7765 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7766 = stablehlo.multiply %v7765, %b8pbt : tensor<64xf32>
    %v7767 = stablehlo.add %v7766, %v3715 : tensor<64xf32>
    %v7768 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7769 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7770 = stablehlo.multiply %v7768, %b8pbtv : tensor<64xf32>
    %v7771 = stablehlo.multiply %v7767, %v7767 : tensor<64xf32>
    %v7772 = stablehlo.multiply %v7769, %v7771 : tensor<64xf32>
    %v7773 = stablehlo.add %v7770, %v7772 : tensor<64xf32>
    %v7774 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7775 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7776 = stablehlo.multiply %v7774, %b8pbtv : tensor<64xf32>
    %v7777 = stablehlo.multiply %v7767, %v7767 : tensor<64xf32>
    %v7778 = stablehlo.multiply %v7775, %v7777 : tensor<64xf32>
    %v7779 = stablehlo.add %v7776, %v7778 : tensor<64xf32>
    %v7780 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7781 = stablehlo.add %v7779, %v7780 : tensor<64xf32>
    %v7782 = stablehlo.sqrt %v7781 : tensor<64xf32>
    %v7783 = stablehlo.divide %v7767, %v7782 : tensor<64xf32>
    %v7784 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7785 = stablehlo.multiply %v7784, %b8pbtm : tensor<64xf32>
    %v7786 = stablehlo.add %v7785, %v7783 : tensor<64xf32>
    %v7787 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7788 = stablehlo.multiply %v7787, %v7786 : tensor<64xf32>
    %v7789 = stablehlo.subtract %b8pbt, %v7788 : tensor<64xf32>
    %v7790 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7791 = stablehlo.multiply %v7790, %b9eW : tensor<384x64x1x1xf32>
    %v7792 = stablehlo.add %v7791, %v3433 : tensor<384x64x1x1xf32>
    %v7793 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7794 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7795 = stablehlo.multiply %v7793, %b9eWv : tensor<384x64x1x1xf32>
    %v7796 = stablehlo.multiply %v7792, %v7792 : tensor<384x64x1x1xf32>
    %v7797 = stablehlo.multiply %v7794, %v7796 : tensor<384x64x1x1xf32>
    %v7798 = stablehlo.add %v7795, %v7797 : tensor<384x64x1x1xf32>
    %v7799 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7800 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7801 = stablehlo.multiply %v7799, %b9eWv : tensor<384x64x1x1xf32>
    %v7802 = stablehlo.multiply %v7792, %v7792 : tensor<384x64x1x1xf32>
    %v7803 = stablehlo.multiply %v7800, %v7802 : tensor<384x64x1x1xf32>
    %v7804 = stablehlo.add %v7801, %v7803 : tensor<384x64x1x1xf32>
    %v7805 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7806 = stablehlo.add %v7804, %v7805 : tensor<384x64x1x1xf32>
    %v7807 = stablehlo.sqrt %v7806 : tensor<384x64x1x1xf32>
    %v7808 = stablehlo.divide %v7792, %v7807 : tensor<384x64x1x1xf32>
    %v7809 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7810 = stablehlo.multiply %v7809, %b9eWm : tensor<384x64x1x1xf32>
    %v7811 = stablehlo.add %v7810, %v7808 : tensor<384x64x1x1xf32>
    %v7812 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v7813 = stablehlo.multiply %v7812, %v7811 : tensor<384x64x1x1xf32>
    %v7814 = stablehlo.subtract %b9eW, %v7813 : tensor<384x64x1x1xf32>
    %v7815 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7816 = stablehlo.multiply %v7815, %b9eg : tensor<384xf32>
    %v7817 = stablehlo.add %v7816, %v3451 : tensor<384xf32>
    %v7818 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7819 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7820 = stablehlo.multiply %v7818, %b9egv : tensor<384xf32>
    %v7821 = stablehlo.multiply %v7817, %v7817 : tensor<384xf32>
    %v7822 = stablehlo.multiply %v7819, %v7821 : tensor<384xf32>
    %v7823 = stablehlo.add %v7820, %v7822 : tensor<384xf32>
    %v7824 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7825 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7826 = stablehlo.multiply %v7824, %b9egv : tensor<384xf32>
    %v7827 = stablehlo.multiply %v7817, %v7817 : tensor<384xf32>
    %v7828 = stablehlo.multiply %v7825, %v7827 : tensor<384xf32>
    %v7829 = stablehlo.add %v7826, %v7828 : tensor<384xf32>
    %v7830 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7831 = stablehlo.add %v7829, %v7830 : tensor<384xf32>
    %v7832 = stablehlo.sqrt %v7831 : tensor<384xf32>
    %v7833 = stablehlo.divide %v7817, %v7832 : tensor<384xf32>
    %v7834 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7835 = stablehlo.multiply %v7834, %b9egm : tensor<384xf32>
    %v7836 = stablehlo.add %v7835, %v7833 : tensor<384xf32>
    %v7837 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7838 = stablehlo.multiply %v7837, %v7836 : tensor<384xf32>
    %v7839 = stablehlo.subtract %b9eg, %v7838 : tensor<384xf32>
    %v7840 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7841 = stablehlo.multiply %v7840, %b9ebt : tensor<384xf32>
    %v7842 = stablehlo.add %v7841, %v3454 : tensor<384xf32>
    %v7843 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7844 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7845 = stablehlo.multiply %v7843, %b9ebtv : tensor<384xf32>
    %v7846 = stablehlo.multiply %v7842, %v7842 : tensor<384xf32>
    %v7847 = stablehlo.multiply %v7844, %v7846 : tensor<384xf32>
    %v7848 = stablehlo.add %v7845, %v7847 : tensor<384xf32>
    %v7849 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7850 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7851 = stablehlo.multiply %v7849, %b9ebtv : tensor<384xf32>
    %v7852 = stablehlo.multiply %v7842, %v7842 : tensor<384xf32>
    %v7853 = stablehlo.multiply %v7850, %v7852 : tensor<384xf32>
    %v7854 = stablehlo.add %v7851, %v7853 : tensor<384xf32>
    %v7855 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7856 = stablehlo.add %v7854, %v7855 : tensor<384xf32>
    %v7857 = stablehlo.sqrt %v7856 : tensor<384xf32>
    %v7858 = stablehlo.divide %v7842, %v7857 : tensor<384xf32>
    %v7859 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7860 = stablehlo.multiply %v7859, %b9ebtm : tensor<384xf32>
    %v7861 = stablehlo.add %v7860, %v7858 : tensor<384xf32>
    %v7862 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7863 = stablehlo.multiply %v7862, %v7861 : tensor<384xf32>
    %v7864 = stablehlo.subtract %b9ebt, %v7863 : tensor<384xf32>
    %v7865 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7866 = stablehlo.multiply %v7865, %b9dW : tensor<384x1x3x3xf32>
    %v7867 = stablehlo.add %v7866, %v3460 : tensor<384x1x3x3xf32>
    %v7868 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7869 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7870 = stablehlo.multiply %v7868, %b9dWv : tensor<384x1x3x3xf32>
    %v7871 = stablehlo.multiply %v7867, %v7867 : tensor<384x1x3x3xf32>
    %v7872 = stablehlo.multiply %v7869, %v7871 : tensor<384x1x3x3xf32>
    %v7873 = stablehlo.add %v7870, %v7872 : tensor<384x1x3x3xf32>
    %v7874 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7875 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7876 = stablehlo.multiply %v7874, %b9dWv : tensor<384x1x3x3xf32>
    %v7877 = stablehlo.multiply %v7867, %v7867 : tensor<384x1x3x3xf32>
    %v7878 = stablehlo.multiply %v7875, %v7877 : tensor<384x1x3x3xf32>
    %v7879 = stablehlo.add %v7876, %v7878 : tensor<384x1x3x3xf32>
    %v7880 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7881 = stablehlo.add %v7879, %v7880 : tensor<384x1x3x3xf32>
    %v7882 = stablehlo.sqrt %v7881 : tensor<384x1x3x3xf32>
    %v7883 = stablehlo.divide %v7867, %v7882 : tensor<384x1x3x3xf32>
    %v7884 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7885 = stablehlo.multiply %v7884, %b9dWm : tensor<384x1x3x3xf32>
    %v7886 = stablehlo.add %v7885, %v7883 : tensor<384x1x3x3xf32>
    %v7887 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v7888 = stablehlo.multiply %v7887, %v7886 : tensor<384x1x3x3xf32>
    %v7889 = stablehlo.subtract %b9dW, %v7888 : tensor<384x1x3x3xf32>
    %v7890 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7891 = stablehlo.multiply %v7890, %b9dg : tensor<384xf32>
    %v7892 = stablehlo.add %v7891, %v3478 : tensor<384xf32>
    %v7893 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7894 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7895 = stablehlo.multiply %v7893, %b9dgv : tensor<384xf32>
    %v7896 = stablehlo.multiply %v7892, %v7892 : tensor<384xf32>
    %v7897 = stablehlo.multiply %v7894, %v7896 : tensor<384xf32>
    %v7898 = stablehlo.add %v7895, %v7897 : tensor<384xf32>
    %v7899 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7900 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7901 = stablehlo.multiply %v7899, %b9dgv : tensor<384xf32>
    %v7902 = stablehlo.multiply %v7892, %v7892 : tensor<384xf32>
    %v7903 = stablehlo.multiply %v7900, %v7902 : tensor<384xf32>
    %v7904 = stablehlo.add %v7901, %v7903 : tensor<384xf32>
    %v7905 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7906 = stablehlo.add %v7904, %v7905 : tensor<384xf32>
    %v7907 = stablehlo.sqrt %v7906 : tensor<384xf32>
    %v7908 = stablehlo.divide %v7892, %v7907 : tensor<384xf32>
    %v7909 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7910 = stablehlo.multiply %v7909, %b9dgm : tensor<384xf32>
    %v7911 = stablehlo.add %v7910, %v7908 : tensor<384xf32>
    %v7912 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7913 = stablehlo.multiply %v7912, %v7911 : tensor<384xf32>
    %v7914 = stablehlo.subtract %b9dg, %v7913 : tensor<384xf32>
    %v7915 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7916 = stablehlo.multiply %v7915, %b9dbt : tensor<384xf32>
    %v7917 = stablehlo.add %v7916, %v3481 : tensor<384xf32>
    %v7918 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7919 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7920 = stablehlo.multiply %v7918, %b9dbtv : tensor<384xf32>
    %v7921 = stablehlo.multiply %v7917, %v7917 : tensor<384xf32>
    %v7922 = stablehlo.multiply %v7919, %v7921 : tensor<384xf32>
    %v7923 = stablehlo.add %v7920, %v7922 : tensor<384xf32>
    %v7924 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7925 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7926 = stablehlo.multiply %v7924, %b9dbtv : tensor<384xf32>
    %v7927 = stablehlo.multiply %v7917, %v7917 : tensor<384xf32>
    %v7928 = stablehlo.multiply %v7925, %v7927 : tensor<384xf32>
    %v7929 = stablehlo.add %v7926, %v7928 : tensor<384xf32>
    %v7930 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7931 = stablehlo.add %v7929, %v7930 : tensor<384xf32>
    %v7932 = stablehlo.sqrt %v7931 : tensor<384xf32>
    %v7933 = stablehlo.divide %v7917, %v7932 : tensor<384xf32>
    %v7934 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7935 = stablehlo.multiply %v7934, %b9dbtm : tensor<384xf32>
    %v7936 = stablehlo.add %v7935, %v7933 : tensor<384xf32>
    %v7937 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7938 = stablehlo.multiply %v7937, %v7936 : tensor<384xf32>
    %v7939 = stablehlo.subtract %b9dbt, %v7938 : tensor<384xf32>
    %v7940 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7941 = stablehlo.multiply %v7940, %b9pW : tensor<64x384x1x1xf32>
    %v7942 = stablehlo.add %v7941, %v3487 : tensor<64x384x1x1xf32>
    %v7943 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7944 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7945 = stablehlo.multiply %v7943, %b9pWv : tensor<64x384x1x1xf32>
    %v7946 = stablehlo.multiply %v7942, %v7942 : tensor<64x384x1x1xf32>
    %v7947 = stablehlo.multiply %v7944, %v7946 : tensor<64x384x1x1xf32>
    %v7948 = stablehlo.add %v7945, %v7947 : tensor<64x384x1x1xf32>
    %v7949 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7950 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7951 = stablehlo.multiply %v7949, %b9pWv : tensor<64x384x1x1xf32>
    %v7952 = stablehlo.multiply %v7942, %v7942 : tensor<64x384x1x1xf32>
    %v7953 = stablehlo.multiply %v7950, %v7952 : tensor<64x384x1x1xf32>
    %v7954 = stablehlo.add %v7951, %v7953 : tensor<64x384x1x1xf32>
    %v7955 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7956 = stablehlo.add %v7954, %v7955 : tensor<64x384x1x1xf32>
    %v7957 = stablehlo.sqrt %v7956 : tensor<64x384x1x1xf32>
    %v7958 = stablehlo.divide %v7942, %v7957 : tensor<64x384x1x1xf32>
    %v7959 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7960 = stablehlo.multiply %v7959, %b9pWm : tensor<64x384x1x1xf32>
    %v7961 = stablehlo.add %v7960, %v7958 : tensor<64x384x1x1xf32>
    %v7962 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v7963 = stablehlo.multiply %v7962, %v7961 : tensor<64x384x1x1xf32>
    %v7964 = stablehlo.subtract %b9pW, %v7963 : tensor<64x384x1x1xf32>
    %v7965 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7966 = stablehlo.multiply %v7965, %b9pg : tensor<64xf32>
    %v7967 = stablehlo.add %v7966, %v3505 : tensor<64xf32>
    %v7968 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7969 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7970 = stablehlo.multiply %v7968, %b9pgv : tensor<64xf32>
    %v7971 = stablehlo.multiply %v7967, %v7967 : tensor<64xf32>
    %v7972 = stablehlo.multiply %v7969, %v7971 : tensor<64xf32>
    %v7973 = stablehlo.add %v7970, %v7972 : tensor<64xf32>
    %v7974 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7975 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7976 = stablehlo.multiply %v7974, %b9pgv : tensor<64xf32>
    %v7977 = stablehlo.multiply %v7967, %v7967 : tensor<64xf32>
    %v7978 = stablehlo.multiply %v7975, %v7977 : tensor<64xf32>
    %v7979 = stablehlo.add %v7976, %v7978 : tensor<64xf32>
    %v7980 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7981 = stablehlo.add %v7979, %v7980 : tensor<64xf32>
    %v7982 = stablehlo.sqrt %v7981 : tensor<64xf32>
    %v7983 = stablehlo.divide %v7967, %v7982 : tensor<64xf32>
    %v7984 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7985 = stablehlo.multiply %v7984, %b9pgm : tensor<64xf32>
    %v7986 = stablehlo.add %v7985, %v7983 : tensor<64xf32>
    %v7987 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7988 = stablehlo.multiply %v7987, %v7986 : tensor<64xf32>
    %v7989 = stablehlo.subtract %b9pg, %v7988 : tensor<64xf32>
    %v7990 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7991 = stablehlo.multiply %v7990, %b9pbt : tensor<64xf32>
    %v7992 = stablehlo.add %v7991, %v3508 : tensor<64xf32>
    %v7993 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7994 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7995 = stablehlo.multiply %v7993, %b9pbtv : tensor<64xf32>
    %v7996 = stablehlo.multiply %v7992, %v7992 : tensor<64xf32>
    %v7997 = stablehlo.multiply %v7994, %v7996 : tensor<64xf32>
    %v7998 = stablehlo.add %v7995, %v7997 : tensor<64xf32>
    %v7999 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8000 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8001 = stablehlo.multiply %v7999, %b9pbtv : tensor<64xf32>
    %v8002 = stablehlo.multiply %v7992, %v7992 : tensor<64xf32>
    %v8003 = stablehlo.multiply %v8000, %v8002 : tensor<64xf32>
    %v8004 = stablehlo.add %v8001, %v8003 : tensor<64xf32>
    %v8005 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8006 = stablehlo.add %v8004, %v8005 : tensor<64xf32>
    %v8007 = stablehlo.sqrt %v8006 : tensor<64xf32>
    %v8008 = stablehlo.divide %v7992, %v8007 : tensor<64xf32>
    %v8009 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8010 = stablehlo.multiply %v8009, %b9pbtm : tensor<64xf32>
    %v8011 = stablehlo.add %v8010, %v8008 : tensor<64xf32>
    %v8012 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8013 = stablehlo.multiply %v8012, %v8011 : tensor<64xf32>
    %v8014 = stablehlo.subtract %b9pbt, %v8013 : tensor<64xf32>
    %v8015 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8016 = stablehlo.multiply %v8015, %b10eW : tensor<384x64x1x1xf32>
    %v8017 = stablehlo.add %v8016, %v3226 : tensor<384x64x1x1xf32>
    %v8018 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8019 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8020 = stablehlo.multiply %v8018, %b10eWv : tensor<384x64x1x1xf32>
    %v8021 = stablehlo.multiply %v8017, %v8017 : tensor<384x64x1x1xf32>
    %v8022 = stablehlo.multiply %v8019, %v8021 : tensor<384x64x1x1xf32>
    %v8023 = stablehlo.add %v8020, %v8022 : tensor<384x64x1x1xf32>
    %v8024 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8025 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8026 = stablehlo.multiply %v8024, %b10eWv : tensor<384x64x1x1xf32>
    %v8027 = stablehlo.multiply %v8017, %v8017 : tensor<384x64x1x1xf32>
    %v8028 = stablehlo.multiply %v8025, %v8027 : tensor<384x64x1x1xf32>
    %v8029 = stablehlo.add %v8026, %v8028 : tensor<384x64x1x1xf32>
    %v8030 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8031 = stablehlo.add %v8029, %v8030 : tensor<384x64x1x1xf32>
    %v8032 = stablehlo.sqrt %v8031 : tensor<384x64x1x1xf32>
    %v8033 = stablehlo.divide %v8017, %v8032 : tensor<384x64x1x1xf32>
    %v8034 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8035 = stablehlo.multiply %v8034, %b10eWm : tensor<384x64x1x1xf32>
    %v8036 = stablehlo.add %v8035, %v8033 : tensor<384x64x1x1xf32>
    %v8037 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8038 = stablehlo.multiply %v8037, %v8036 : tensor<384x64x1x1xf32>
    %v8039 = stablehlo.subtract %b10eW, %v8038 : tensor<384x64x1x1xf32>
    %v8040 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8041 = stablehlo.multiply %v8040, %b10eg : tensor<384xf32>
    %v8042 = stablehlo.add %v8041, %v3244 : tensor<384xf32>
    %v8043 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8044 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8045 = stablehlo.multiply %v8043, %b10egv : tensor<384xf32>
    %v8046 = stablehlo.multiply %v8042, %v8042 : tensor<384xf32>
    %v8047 = stablehlo.multiply %v8044, %v8046 : tensor<384xf32>
    %v8048 = stablehlo.add %v8045, %v8047 : tensor<384xf32>
    %v8049 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8050 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8051 = stablehlo.multiply %v8049, %b10egv : tensor<384xf32>
    %v8052 = stablehlo.multiply %v8042, %v8042 : tensor<384xf32>
    %v8053 = stablehlo.multiply %v8050, %v8052 : tensor<384xf32>
    %v8054 = stablehlo.add %v8051, %v8053 : tensor<384xf32>
    %v8055 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8056 = stablehlo.add %v8054, %v8055 : tensor<384xf32>
    %v8057 = stablehlo.sqrt %v8056 : tensor<384xf32>
    %v8058 = stablehlo.divide %v8042, %v8057 : tensor<384xf32>
    %v8059 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8060 = stablehlo.multiply %v8059, %b10egm : tensor<384xf32>
    %v8061 = stablehlo.add %v8060, %v8058 : tensor<384xf32>
    %v8062 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8063 = stablehlo.multiply %v8062, %v8061 : tensor<384xf32>
    %v8064 = stablehlo.subtract %b10eg, %v8063 : tensor<384xf32>
    %v8065 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8066 = stablehlo.multiply %v8065, %b10ebt : tensor<384xf32>
    %v8067 = stablehlo.add %v8066, %v3247 : tensor<384xf32>
    %v8068 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8069 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8070 = stablehlo.multiply %v8068, %b10ebtv : tensor<384xf32>
    %v8071 = stablehlo.multiply %v8067, %v8067 : tensor<384xf32>
    %v8072 = stablehlo.multiply %v8069, %v8071 : tensor<384xf32>
    %v8073 = stablehlo.add %v8070, %v8072 : tensor<384xf32>
    %v8074 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8075 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8076 = stablehlo.multiply %v8074, %b10ebtv : tensor<384xf32>
    %v8077 = stablehlo.multiply %v8067, %v8067 : tensor<384xf32>
    %v8078 = stablehlo.multiply %v8075, %v8077 : tensor<384xf32>
    %v8079 = stablehlo.add %v8076, %v8078 : tensor<384xf32>
    %v8080 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8081 = stablehlo.add %v8079, %v8080 : tensor<384xf32>
    %v8082 = stablehlo.sqrt %v8081 : tensor<384xf32>
    %v8083 = stablehlo.divide %v8067, %v8082 : tensor<384xf32>
    %v8084 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8085 = stablehlo.multiply %v8084, %b10ebtm : tensor<384xf32>
    %v8086 = stablehlo.add %v8085, %v8083 : tensor<384xf32>
    %v8087 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8088 = stablehlo.multiply %v8087, %v8086 : tensor<384xf32>
    %v8089 = stablehlo.subtract %b10ebt, %v8088 : tensor<384xf32>
    %v8090 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8091 = stablehlo.multiply %v8090, %b10dW : tensor<384x1x3x3xf32>
    %v8092 = stablehlo.add %v8091, %v3253 : tensor<384x1x3x3xf32>
    %v8093 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8094 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8095 = stablehlo.multiply %v8093, %b10dWv : tensor<384x1x3x3xf32>
    %v8096 = stablehlo.multiply %v8092, %v8092 : tensor<384x1x3x3xf32>
    %v8097 = stablehlo.multiply %v8094, %v8096 : tensor<384x1x3x3xf32>
    %v8098 = stablehlo.add %v8095, %v8097 : tensor<384x1x3x3xf32>
    %v8099 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8100 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8101 = stablehlo.multiply %v8099, %b10dWv : tensor<384x1x3x3xf32>
    %v8102 = stablehlo.multiply %v8092, %v8092 : tensor<384x1x3x3xf32>
    %v8103 = stablehlo.multiply %v8100, %v8102 : tensor<384x1x3x3xf32>
    %v8104 = stablehlo.add %v8101, %v8103 : tensor<384x1x3x3xf32>
    %v8105 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8106 = stablehlo.add %v8104, %v8105 : tensor<384x1x3x3xf32>
    %v8107 = stablehlo.sqrt %v8106 : tensor<384x1x3x3xf32>
    %v8108 = stablehlo.divide %v8092, %v8107 : tensor<384x1x3x3xf32>
    %v8109 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8110 = stablehlo.multiply %v8109, %b10dWm : tensor<384x1x3x3xf32>
    %v8111 = stablehlo.add %v8110, %v8108 : tensor<384x1x3x3xf32>
    %v8112 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8113 = stablehlo.multiply %v8112, %v8111 : tensor<384x1x3x3xf32>
    %v8114 = stablehlo.subtract %b10dW, %v8113 : tensor<384x1x3x3xf32>
    %v8115 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8116 = stablehlo.multiply %v8115, %b10dg : tensor<384xf32>
    %v8117 = stablehlo.add %v8116, %v3271 : tensor<384xf32>
    %v8118 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8119 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8120 = stablehlo.multiply %v8118, %b10dgv : tensor<384xf32>
    %v8121 = stablehlo.multiply %v8117, %v8117 : tensor<384xf32>
    %v8122 = stablehlo.multiply %v8119, %v8121 : tensor<384xf32>
    %v8123 = stablehlo.add %v8120, %v8122 : tensor<384xf32>
    %v8124 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8125 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8126 = stablehlo.multiply %v8124, %b10dgv : tensor<384xf32>
    %v8127 = stablehlo.multiply %v8117, %v8117 : tensor<384xf32>
    %v8128 = stablehlo.multiply %v8125, %v8127 : tensor<384xf32>
    %v8129 = stablehlo.add %v8126, %v8128 : tensor<384xf32>
    %v8130 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8131 = stablehlo.add %v8129, %v8130 : tensor<384xf32>
    %v8132 = stablehlo.sqrt %v8131 : tensor<384xf32>
    %v8133 = stablehlo.divide %v8117, %v8132 : tensor<384xf32>
    %v8134 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8135 = stablehlo.multiply %v8134, %b10dgm : tensor<384xf32>
    %v8136 = stablehlo.add %v8135, %v8133 : tensor<384xf32>
    %v8137 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8138 = stablehlo.multiply %v8137, %v8136 : tensor<384xf32>
    %v8139 = stablehlo.subtract %b10dg, %v8138 : tensor<384xf32>
    %v8140 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8141 = stablehlo.multiply %v8140, %b10dbt : tensor<384xf32>
    %v8142 = stablehlo.add %v8141, %v3274 : tensor<384xf32>
    %v8143 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8144 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8145 = stablehlo.multiply %v8143, %b10dbtv : tensor<384xf32>
    %v8146 = stablehlo.multiply %v8142, %v8142 : tensor<384xf32>
    %v8147 = stablehlo.multiply %v8144, %v8146 : tensor<384xf32>
    %v8148 = stablehlo.add %v8145, %v8147 : tensor<384xf32>
    %v8149 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8150 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8151 = stablehlo.multiply %v8149, %b10dbtv : tensor<384xf32>
    %v8152 = stablehlo.multiply %v8142, %v8142 : tensor<384xf32>
    %v8153 = stablehlo.multiply %v8150, %v8152 : tensor<384xf32>
    %v8154 = stablehlo.add %v8151, %v8153 : tensor<384xf32>
    %v8155 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8156 = stablehlo.add %v8154, %v8155 : tensor<384xf32>
    %v8157 = stablehlo.sqrt %v8156 : tensor<384xf32>
    %v8158 = stablehlo.divide %v8142, %v8157 : tensor<384xf32>
    %v8159 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8160 = stablehlo.multiply %v8159, %b10dbtm : tensor<384xf32>
    %v8161 = stablehlo.add %v8160, %v8158 : tensor<384xf32>
    %v8162 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8163 = stablehlo.multiply %v8162, %v8161 : tensor<384xf32>
    %v8164 = stablehlo.subtract %b10dbt, %v8163 : tensor<384xf32>
    %v8165 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8166 = stablehlo.multiply %v8165, %b10pW : tensor<64x384x1x1xf32>
    %v8167 = stablehlo.add %v8166, %v3280 : tensor<64x384x1x1xf32>
    %v8168 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8169 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8170 = stablehlo.multiply %v8168, %b10pWv : tensor<64x384x1x1xf32>
    %v8171 = stablehlo.multiply %v8167, %v8167 : tensor<64x384x1x1xf32>
    %v8172 = stablehlo.multiply %v8169, %v8171 : tensor<64x384x1x1xf32>
    %v8173 = stablehlo.add %v8170, %v8172 : tensor<64x384x1x1xf32>
    %v8174 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8175 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8176 = stablehlo.multiply %v8174, %b10pWv : tensor<64x384x1x1xf32>
    %v8177 = stablehlo.multiply %v8167, %v8167 : tensor<64x384x1x1xf32>
    %v8178 = stablehlo.multiply %v8175, %v8177 : tensor<64x384x1x1xf32>
    %v8179 = stablehlo.add %v8176, %v8178 : tensor<64x384x1x1xf32>
    %v8180 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8181 = stablehlo.add %v8179, %v8180 : tensor<64x384x1x1xf32>
    %v8182 = stablehlo.sqrt %v8181 : tensor<64x384x1x1xf32>
    %v8183 = stablehlo.divide %v8167, %v8182 : tensor<64x384x1x1xf32>
    %v8184 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8185 = stablehlo.multiply %v8184, %b10pWm : tensor<64x384x1x1xf32>
    %v8186 = stablehlo.add %v8185, %v8183 : tensor<64x384x1x1xf32>
    %v8187 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x384x1x1xf32>
    %v8188 = stablehlo.multiply %v8187, %v8186 : tensor<64x384x1x1xf32>
    %v8189 = stablehlo.subtract %b10pW, %v8188 : tensor<64x384x1x1xf32>
    %v8190 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8191 = stablehlo.multiply %v8190, %b10pg : tensor<64xf32>
    %v8192 = stablehlo.add %v8191, %v3298 : tensor<64xf32>
    %v8193 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8194 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8195 = stablehlo.multiply %v8193, %b10pgv : tensor<64xf32>
    %v8196 = stablehlo.multiply %v8192, %v8192 : tensor<64xf32>
    %v8197 = stablehlo.multiply %v8194, %v8196 : tensor<64xf32>
    %v8198 = stablehlo.add %v8195, %v8197 : tensor<64xf32>
    %v8199 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8200 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8201 = stablehlo.multiply %v8199, %b10pgv : tensor<64xf32>
    %v8202 = stablehlo.multiply %v8192, %v8192 : tensor<64xf32>
    %v8203 = stablehlo.multiply %v8200, %v8202 : tensor<64xf32>
    %v8204 = stablehlo.add %v8201, %v8203 : tensor<64xf32>
    %v8205 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8206 = stablehlo.add %v8204, %v8205 : tensor<64xf32>
    %v8207 = stablehlo.sqrt %v8206 : tensor<64xf32>
    %v8208 = stablehlo.divide %v8192, %v8207 : tensor<64xf32>
    %v8209 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8210 = stablehlo.multiply %v8209, %b10pgm : tensor<64xf32>
    %v8211 = stablehlo.add %v8210, %v8208 : tensor<64xf32>
    %v8212 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8213 = stablehlo.multiply %v8212, %v8211 : tensor<64xf32>
    %v8214 = stablehlo.subtract %b10pg, %v8213 : tensor<64xf32>
    %v8215 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8216 = stablehlo.multiply %v8215, %b10pbt : tensor<64xf32>
    %v8217 = stablehlo.add %v8216, %v3301 : tensor<64xf32>
    %v8218 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8219 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8220 = stablehlo.multiply %v8218, %b10pbtv : tensor<64xf32>
    %v8221 = stablehlo.multiply %v8217, %v8217 : tensor<64xf32>
    %v8222 = stablehlo.multiply %v8219, %v8221 : tensor<64xf32>
    %v8223 = stablehlo.add %v8220, %v8222 : tensor<64xf32>
    %v8224 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8225 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8226 = stablehlo.multiply %v8224, %b10pbtv : tensor<64xf32>
    %v8227 = stablehlo.multiply %v8217, %v8217 : tensor<64xf32>
    %v8228 = stablehlo.multiply %v8225, %v8227 : tensor<64xf32>
    %v8229 = stablehlo.add %v8226, %v8228 : tensor<64xf32>
    %v8230 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8231 = stablehlo.add %v8229, %v8230 : tensor<64xf32>
    %v8232 = stablehlo.sqrt %v8231 : tensor<64xf32>
    %v8233 = stablehlo.divide %v8217, %v8232 : tensor<64xf32>
    %v8234 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8235 = stablehlo.multiply %v8234, %b10pbtm : tensor<64xf32>
    %v8236 = stablehlo.add %v8235, %v8233 : tensor<64xf32>
    %v8237 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v8238 = stablehlo.multiply %v8237, %v8236 : tensor<64xf32>
    %v8239 = stablehlo.subtract %b10pbt, %v8238 : tensor<64xf32>
    %v8240 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8241 = stablehlo.multiply %v8240, %b11eW : tensor<384x64x1x1xf32>
    %v8242 = stablehlo.add %v8241, %v3019 : tensor<384x64x1x1xf32>
    %v8243 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8244 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8245 = stablehlo.multiply %v8243, %b11eWv : tensor<384x64x1x1xf32>
    %v8246 = stablehlo.multiply %v8242, %v8242 : tensor<384x64x1x1xf32>
    %v8247 = stablehlo.multiply %v8244, %v8246 : tensor<384x64x1x1xf32>
    %v8248 = stablehlo.add %v8245, %v8247 : tensor<384x64x1x1xf32>
    %v8249 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8250 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8251 = stablehlo.multiply %v8249, %b11eWv : tensor<384x64x1x1xf32>
    %v8252 = stablehlo.multiply %v8242, %v8242 : tensor<384x64x1x1xf32>
    %v8253 = stablehlo.multiply %v8250, %v8252 : tensor<384x64x1x1xf32>
    %v8254 = stablehlo.add %v8251, %v8253 : tensor<384x64x1x1xf32>
    %v8255 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8256 = stablehlo.add %v8254, %v8255 : tensor<384x64x1x1xf32>
    %v8257 = stablehlo.sqrt %v8256 : tensor<384x64x1x1xf32>
    %v8258 = stablehlo.divide %v8242, %v8257 : tensor<384x64x1x1xf32>
    %v8259 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8260 = stablehlo.multiply %v8259, %b11eWm : tensor<384x64x1x1xf32>
    %v8261 = stablehlo.add %v8260, %v8258 : tensor<384x64x1x1xf32>
    %v8262 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x64x1x1xf32>
    %v8263 = stablehlo.multiply %v8262, %v8261 : tensor<384x64x1x1xf32>
    %v8264 = stablehlo.subtract %b11eW, %v8263 : tensor<384x64x1x1xf32>
    %v8265 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8266 = stablehlo.multiply %v8265, %b11eg : tensor<384xf32>
    %v8267 = stablehlo.add %v8266, %v3037 : tensor<384xf32>
    %v8268 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8269 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8270 = stablehlo.multiply %v8268, %b11egv : tensor<384xf32>
    %v8271 = stablehlo.multiply %v8267, %v8267 : tensor<384xf32>
    %v8272 = stablehlo.multiply %v8269, %v8271 : tensor<384xf32>
    %v8273 = stablehlo.add %v8270, %v8272 : tensor<384xf32>
    %v8274 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8275 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8276 = stablehlo.multiply %v8274, %b11egv : tensor<384xf32>
    %v8277 = stablehlo.multiply %v8267, %v8267 : tensor<384xf32>
    %v8278 = stablehlo.multiply %v8275, %v8277 : tensor<384xf32>
    %v8279 = stablehlo.add %v8276, %v8278 : tensor<384xf32>
    %v8280 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8281 = stablehlo.add %v8279, %v8280 : tensor<384xf32>
    %v8282 = stablehlo.sqrt %v8281 : tensor<384xf32>
    %v8283 = stablehlo.divide %v8267, %v8282 : tensor<384xf32>
    %v8284 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8285 = stablehlo.multiply %v8284, %b11egm : tensor<384xf32>
    %v8286 = stablehlo.add %v8285, %v8283 : tensor<384xf32>
    %v8287 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8288 = stablehlo.multiply %v8287, %v8286 : tensor<384xf32>
    %v8289 = stablehlo.subtract %b11eg, %v8288 : tensor<384xf32>
    %v8290 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8291 = stablehlo.multiply %v8290, %b11ebt : tensor<384xf32>
    %v8292 = stablehlo.add %v8291, %v3040 : tensor<384xf32>
    %v8293 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8294 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8295 = stablehlo.multiply %v8293, %b11ebtv : tensor<384xf32>
    %v8296 = stablehlo.multiply %v8292, %v8292 : tensor<384xf32>
    %v8297 = stablehlo.multiply %v8294, %v8296 : tensor<384xf32>
    %v8298 = stablehlo.add %v8295, %v8297 : tensor<384xf32>
    %v8299 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8300 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8301 = stablehlo.multiply %v8299, %b11ebtv : tensor<384xf32>
    %v8302 = stablehlo.multiply %v8292, %v8292 : tensor<384xf32>
    %v8303 = stablehlo.multiply %v8300, %v8302 : tensor<384xf32>
    %v8304 = stablehlo.add %v8301, %v8303 : tensor<384xf32>
    %v8305 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8306 = stablehlo.add %v8304, %v8305 : tensor<384xf32>
    %v8307 = stablehlo.sqrt %v8306 : tensor<384xf32>
    %v8308 = stablehlo.divide %v8292, %v8307 : tensor<384xf32>
    %v8309 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8310 = stablehlo.multiply %v8309, %b11ebtm : tensor<384xf32>
    %v8311 = stablehlo.add %v8310, %v8308 : tensor<384xf32>
    %v8312 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8313 = stablehlo.multiply %v8312, %v8311 : tensor<384xf32>
    %v8314 = stablehlo.subtract %b11ebt, %v8313 : tensor<384xf32>
    %v8315 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8316 = stablehlo.multiply %v8315, %b11dW : tensor<384x1x3x3xf32>
    %v8317 = stablehlo.add %v8316, %v3046 : tensor<384x1x3x3xf32>
    %v8318 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8319 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8320 = stablehlo.multiply %v8318, %b11dWv : tensor<384x1x3x3xf32>
    %v8321 = stablehlo.multiply %v8317, %v8317 : tensor<384x1x3x3xf32>
    %v8322 = stablehlo.multiply %v8319, %v8321 : tensor<384x1x3x3xf32>
    %v8323 = stablehlo.add %v8320, %v8322 : tensor<384x1x3x3xf32>
    %v8324 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8325 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8326 = stablehlo.multiply %v8324, %b11dWv : tensor<384x1x3x3xf32>
    %v8327 = stablehlo.multiply %v8317, %v8317 : tensor<384x1x3x3xf32>
    %v8328 = stablehlo.multiply %v8325, %v8327 : tensor<384x1x3x3xf32>
    %v8329 = stablehlo.add %v8326, %v8328 : tensor<384x1x3x3xf32>
    %v8330 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8331 = stablehlo.add %v8329, %v8330 : tensor<384x1x3x3xf32>
    %v8332 = stablehlo.sqrt %v8331 : tensor<384x1x3x3xf32>
    %v8333 = stablehlo.divide %v8317, %v8332 : tensor<384x1x3x3xf32>
    %v8334 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8335 = stablehlo.multiply %v8334, %b11dWm : tensor<384x1x3x3xf32>
    %v8336 = stablehlo.add %v8335, %v8333 : tensor<384x1x3x3xf32>
    %v8337 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x3x3xf32>
    %v8338 = stablehlo.multiply %v8337, %v8336 : tensor<384x1x3x3xf32>
    %v8339 = stablehlo.subtract %b11dW, %v8338 : tensor<384x1x3x3xf32>
    %v8340 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8341 = stablehlo.multiply %v8340, %b11dg : tensor<384xf32>
    %v8342 = stablehlo.add %v8341, %v3064 : tensor<384xf32>
    %v8343 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8344 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8345 = stablehlo.multiply %v8343, %b11dgv : tensor<384xf32>
    %v8346 = stablehlo.multiply %v8342, %v8342 : tensor<384xf32>
    %v8347 = stablehlo.multiply %v8344, %v8346 : tensor<384xf32>
    %v8348 = stablehlo.add %v8345, %v8347 : tensor<384xf32>
    %v8349 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8350 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8351 = stablehlo.multiply %v8349, %b11dgv : tensor<384xf32>
    %v8352 = stablehlo.multiply %v8342, %v8342 : tensor<384xf32>
    %v8353 = stablehlo.multiply %v8350, %v8352 : tensor<384xf32>
    %v8354 = stablehlo.add %v8351, %v8353 : tensor<384xf32>
    %v8355 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8356 = stablehlo.add %v8354, %v8355 : tensor<384xf32>
    %v8357 = stablehlo.sqrt %v8356 : tensor<384xf32>
    %v8358 = stablehlo.divide %v8342, %v8357 : tensor<384xf32>
    %v8359 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8360 = stablehlo.multiply %v8359, %b11dgm : tensor<384xf32>
    %v8361 = stablehlo.add %v8360, %v8358 : tensor<384xf32>
    %v8362 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8363 = stablehlo.multiply %v8362, %v8361 : tensor<384xf32>
    %v8364 = stablehlo.subtract %b11dg, %v8363 : tensor<384xf32>
    %v8365 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8366 = stablehlo.multiply %v8365, %b11dbt : tensor<384xf32>
    %v8367 = stablehlo.add %v8366, %v3067 : tensor<384xf32>
    %v8368 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8369 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8370 = stablehlo.multiply %v8368, %b11dbtv : tensor<384xf32>
    %v8371 = stablehlo.multiply %v8367, %v8367 : tensor<384xf32>
    %v8372 = stablehlo.multiply %v8369, %v8371 : tensor<384xf32>
    %v8373 = stablehlo.add %v8370, %v8372 : tensor<384xf32>
    %v8374 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8375 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8376 = stablehlo.multiply %v8374, %b11dbtv : tensor<384xf32>
    %v8377 = stablehlo.multiply %v8367, %v8367 : tensor<384xf32>
    %v8378 = stablehlo.multiply %v8375, %v8377 : tensor<384xf32>
    %v8379 = stablehlo.add %v8376, %v8378 : tensor<384xf32>
    %v8380 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8381 = stablehlo.add %v8379, %v8380 : tensor<384xf32>
    %v8382 = stablehlo.sqrt %v8381 : tensor<384xf32>
    %v8383 = stablehlo.divide %v8367, %v8382 : tensor<384xf32>
    %v8384 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8385 = stablehlo.multiply %v8384, %b11dbtm : tensor<384xf32>
    %v8386 = stablehlo.add %v8385, %v8383 : tensor<384xf32>
    %v8387 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8388 = stablehlo.multiply %v8387, %v8386 : tensor<384xf32>
    %v8389 = stablehlo.subtract %b11dbt, %v8388 : tensor<384xf32>
    %v8390 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8391 = stablehlo.multiply %v8390, %b11pW : tensor<96x384x1x1xf32>
    %v8392 = stablehlo.add %v8391, %v3073 : tensor<96x384x1x1xf32>
    %v8393 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8394 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8395 = stablehlo.multiply %v8393, %b11pWv : tensor<96x384x1x1xf32>
    %v8396 = stablehlo.multiply %v8392, %v8392 : tensor<96x384x1x1xf32>
    %v8397 = stablehlo.multiply %v8394, %v8396 : tensor<96x384x1x1xf32>
    %v8398 = stablehlo.add %v8395, %v8397 : tensor<96x384x1x1xf32>
    %v8399 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8400 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8401 = stablehlo.multiply %v8399, %b11pWv : tensor<96x384x1x1xf32>
    %v8402 = stablehlo.multiply %v8392, %v8392 : tensor<96x384x1x1xf32>
    %v8403 = stablehlo.multiply %v8400, %v8402 : tensor<96x384x1x1xf32>
    %v8404 = stablehlo.add %v8401, %v8403 : tensor<96x384x1x1xf32>
    %v8405 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8406 = stablehlo.add %v8404, %v8405 : tensor<96x384x1x1xf32>
    %v8407 = stablehlo.sqrt %v8406 : tensor<96x384x1x1xf32>
    %v8408 = stablehlo.divide %v8392, %v8407 : tensor<96x384x1x1xf32>
    %v8409 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8410 = stablehlo.multiply %v8409, %b11pWm : tensor<96x384x1x1xf32>
    %v8411 = stablehlo.add %v8410, %v8408 : tensor<96x384x1x1xf32>
    %v8412 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v8413 = stablehlo.multiply %v8412, %v8411 : tensor<96x384x1x1xf32>
    %v8414 = stablehlo.subtract %b11pW, %v8413 : tensor<96x384x1x1xf32>
    %v8415 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8416 = stablehlo.multiply %v8415, %b11pg : tensor<96xf32>
    %v8417 = stablehlo.add %v8416, %v3091 : tensor<96xf32>
    %v8418 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8419 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8420 = stablehlo.multiply %v8418, %b11pgv : tensor<96xf32>
    %v8421 = stablehlo.multiply %v8417, %v8417 : tensor<96xf32>
    %v8422 = stablehlo.multiply %v8419, %v8421 : tensor<96xf32>
    %v8423 = stablehlo.add %v8420, %v8422 : tensor<96xf32>
    %v8424 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8425 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8426 = stablehlo.multiply %v8424, %b11pgv : tensor<96xf32>
    %v8427 = stablehlo.multiply %v8417, %v8417 : tensor<96xf32>
    %v8428 = stablehlo.multiply %v8425, %v8427 : tensor<96xf32>
    %v8429 = stablehlo.add %v8426, %v8428 : tensor<96xf32>
    %v8430 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8431 = stablehlo.add %v8429, %v8430 : tensor<96xf32>
    %v8432 = stablehlo.sqrt %v8431 : tensor<96xf32>
    %v8433 = stablehlo.divide %v8417, %v8432 : tensor<96xf32>
    %v8434 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8435 = stablehlo.multiply %v8434, %b11pgm : tensor<96xf32>
    %v8436 = stablehlo.add %v8435, %v8433 : tensor<96xf32>
    %v8437 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8438 = stablehlo.multiply %v8437, %v8436 : tensor<96xf32>
    %v8439 = stablehlo.subtract %b11pg, %v8438 : tensor<96xf32>
    %v8440 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8441 = stablehlo.multiply %v8440, %b11pbt : tensor<96xf32>
    %v8442 = stablehlo.add %v8441, %v3094 : tensor<96xf32>
    %v8443 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8444 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8445 = stablehlo.multiply %v8443, %b11pbtv : tensor<96xf32>
    %v8446 = stablehlo.multiply %v8442, %v8442 : tensor<96xf32>
    %v8447 = stablehlo.multiply %v8444, %v8446 : tensor<96xf32>
    %v8448 = stablehlo.add %v8445, %v8447 : tensor<96xf32>
    %v8449 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8450 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8451 = stablehlo.multiply %v8449, %b11pbtv : tensor<96xf32>
    %v8452 = stablehlo.multiply %v8442, %v8442 : tensor<96xf32>
    %v8453 = stablehlo.multiply %v8450, %v8452 : tensor<96xf32>
    %v8454 = stablehlo.add %v8451, %v8453 : tensor<96xf32>
    %v8455 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8456 = stablehlo.add %v8454, %v8455 : tensor<96xf32>
    %v8457 = stablehlo.sqrt %v8456 : tensor<96xf32>
    %v8458 = stablehlo.divide %v8442, %v8457 : tensor<96xf32>
    %v8459 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8460 = stablehlo.multiply %v8459, %b11pbtm : tensor<96xf32>
    %v8461 = stablehlo.add %v8460, %v8458 : tensor<96xf32>
    %v8462 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8463 = stablehlo.multiply %v8462, %v8461 : tensor<96xf32>
    %v8464 = stablehlo.subtract %b11pbt, %v8463 : tensor<96xf32>
    %v8465 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8466 = stablehlo.multiply %v8465, %b12eW : tensor<576x96x1x1xf32>
    %v8467 = stablehlo.add %v8466, %v2816 : tensor<576x96x1x1xf32>
    %v8468 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8469 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8470 = stablehlo.multiply %v8468, %b12eWv : tensor<576x96x1x1xf32>
    %v8471 = stablehlo.multiply %v8467, %v8467 : tensor<576x96x1x1xf32>
    %v8472 = stablehlo.multiply %v8469, %v8471 : tensor<576x96x1x1xf32>
    %v8473 = stablehlo.add %v8470, %v8472 : tensor<576x96x1x1xf32>
    %v8474 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8475 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8476 = stablehlo.multiply %v8474, %b12eWv : tensor<576x96x1x1xf32>
    %v8477 = stablehlo.multiply %v8467, %v8467 : tensor<576x96x1x1xf32>
    %v8478 = stablehlo.multiply %v8475, %v8477 : tensor<576x96x1x1xf32>
    %v8479 = stablehlo.add %v8476, %v8478 : tensor<576x96x1x1xf32>
    %v8480 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8481 = stablehlo.add %v8479, %v8480 : tensor<576x96x1x1xf32>
    %v8482 = stablehlo.sqrt %v8481 : tensor<576x96x1x1xf32>
    %v8483 = stablehlo.divide %v8467, %v8482 : tensor<576x96x1x1xf32>
    %v8484 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8485 = stablehlo.multiply %v8484, %b12eWm : tensor<576x96x1x1xf32>
    %v8486 = stablehlo.add %v8485, %v8483 : tensor<576x96x1x1xf32>
    %v8487 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8488 = stablehlo.multiply %v8487, %v8486 : tensor<576x96x1x1xf32>
    %v8489 = stablehlo.subtract %b12eW, %v8488 : tensor<576x96x1x1xf32>
    %v8490 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8491 = stablehlo.multiply %v8490, %b12eg : tensor<576xf32>
    %v8492 = stablehlo.add %v8491, %v2834 : tensor<576xf32>
    %v8493 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8494 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8495 = stablehlo.multiply %v8493, %b12egv : tensor<576xf32>
    %v8496 = stablehlo.multiply %v8492, %v8492 : tensor<576xf32>
    %v8497 = stablehlo.multiply %v8494, %v8496 : tensor<576xf32>
    %v8498 = stablehlo.add %v8495, %v8497 : tensor<576xf32>
    %v8499 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8500 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8501 = stablehlo.multiply %v8499, %b12egv : tensor<576xf32>
    %v8502 = stablehlo.multiply %v8492, %v8492 : tensor<576xf32>
    %v8503 = stablehlo.multiply %v8500, %v8502 : tensor<576xf32>
    %v8504 = stablehlo.add %v8501, %v8503 : tensor<576xf32>
    %v8505 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8506 = stablehlo.add %v8504, %v8505 : tensor<576xf32>
    %v8507 = stablehlo.sqrt %v8506 : tensor<576xf32>
    %v8508 = stablehlo.divide %v8492, %v8507 : tensor<576xf32>
    %v8509 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8510 = stablehlo.multiply %v8509, %b12egm : tensor<576xf32>
    %v8511 = stablehlo.add %v8510, %v8508 : tensor<576xf32>
    %v8512 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8513 = stablehlo.multiply %v8512, %v8511 : tensor<576xf32>
    %v8514 = stablehlo.subtract %b12eg, %v8513 : tensor<576xf32>
    %v8515 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8516 = stablehlo.multiply %v8515, %b12ebt : tensor<576xf32>
    %v8517 = stablehlo.add %v8516, %v2837 : tensor<576xf32>
    %v8518 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8519 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8520 = stablehlo.multiply %v8518, %b12ebtv : tensor<576xf32>
    %v8521 = stablehlo.multiply %v8517, %v8517 : tensor<576xf32>
    %v8522 = stablehlo.multiply %v8519, %v8521 : tensor<576xf32>
    %v8523 = stablehlo.add %v8520, %v8522 : tensor<576xf32>
    %v8524 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8525 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8526 = stablehlo.multiply %v8524, %b12ebtv : tensor<576xf32>
    %v8527 = stablehlo.multiply %v8517, %v8517 : tensor<576xf32>
    %v8528 = stablehlo.multiply %v8525, %v8527 : tensor<576xf32>
    %v8529 = stablehlo.add %v8526, %v8528 : tensor<576xf32>
    %v8530 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8531 = stablehlo.add %v8529, %v8530 : tensor<576xf32>
    %v8532 = stablehlo.sqrt %v8531 : tensor<576xf32>
    %v8533 = stablehlo.divide %v8517, %v8532 : tensor<576xf32>
    %v8534 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8535 = stablehlo.multiply %v8534, %b12ebtm : tensor<576xf32>
    %v8536 = stablehlo.add %v8535, %v8533 : tensor<576xf32>
    %v8537 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8538 = stablehlo.multiply %v8537, %v8536 : tensor<576xf32>
    %v8539 = stablehlo.subtract %b12ebt, %v8538 : tensor<576xf32>
    %v8540 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8541 = stablehlo.multiply %v8540, %b12dW : tensor<576x1x3x3xf32>
    %v8542 = stablehlo.add %v8541, %v2843 : tensor<576x1x3x3xf32>
    %v8543 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8544 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8545 = stablehlo.multiply %v8543, %b12dWv : tensor<576x1x3x3xf32>
    %v8546 = stablehlo.multiply %v8542, %v8542 : tensor<576x1x3x3xf32>
    %v8547 = stablehlo.multiply %v8544, %v8546 : tensor<576x1x3x3xf32>
    %v8548 = stablehlo.add %v8545, %v8547 : tensor<576x1x3x3xf32>
    %v8549 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8550 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8551 = stablehlo.multiply %v8549, %b12dWv : tensor<576x1x3x3xf32>
    %v8552 = stablehlo.multiply %v8542, %v8542 : tensor<576x1x3x3xf32>
    %v8553 = stablehlo.multiply %v8550, %v8552 : tensor<576x1x3x3xf32>
    %v8554 = stablehlo.add %v8551, %v8553 : tensor<576x1x3x3xf32>
    %v8555 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8556 = stablehlo.add %v8554, %v8555 : tensor<576x1x3x3xf32>
    %v8557 = stablehlo.sqrt %v8556 : tensor<576x1x3x3xf32>
    %v8558 = stablehlo.divide %v8542, %v8557 : tensor<576x1x3x3xf32>
    %v8559 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8560 = stablehlo.multiply %v8559, %b12dWm : tensor<576x1x3x3xf32>
    %v8561 = stablehlo.add %v8560, %v8558 : tensor<576x1x3x3xf32>
    %v8562 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8563 = stablehlo.multiply %v8562, %v8561 : tensor<576x1x3x3xf32>
    %v8564 = stablehlo.subtract %b12dW, %v8563 : tensor<576x1x3x3xf32>
    %v8565 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8566 = stablehlo.multiply %v8565, %b12dg : tensor<576xf32>
    %v8567 = stablehlo.add %v8566, %v2861 : tensor<576xf32>
    %v8568 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8569 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8570 = stablehlo.multiply %v8568, %b12dgv : tensor<576xf32>
    %v8571 = stablehlo.multiply %v8567, %v8567 : tensor<576xf32>
    %v8572 = stablehlo.multiply %v8569, %v8571 : tensor<576xf32>
    %v8573 = stablehlo.add %v8570, %v8572 : tensor<576xf32>
    %v8574 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8575 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8576 = stablehlo.multiply %v8574, %b12dgv : tensor<576xf32>
    %v8577 = stablehlo.multiply %v8567, %v8567 : tensor<576xf32>
    %v8578 = stablehlo.multiply %v8575, %v8577 : tensor<576xf32>
    %v8579 = stablehlo.add %v8576, %v8578 : tensor<576xf32>
    %v8580 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8581 = stablehlo.add %v8579, %v8580 : tensor<576xf32>
    %v8582 = stablehlo.sqrt %v8581 : tensor<576xf32>
    %v8583 = stablehlo.divide %v8567, %v8582 : tensor<576xf32>
    %v8584 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8585 = stablehlo.multiply %v8584, %b12dgm : tensor<576xf32>
    %v8586 = stablehlo.add %v8585, %v8583 : tensor<576xf32>
    %v8587 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8588 = stablehlo.multiply %v8587, %v8586 : tensor<576xf32>
    %v8589 = stablehlo.subtract %b12dg, %v8588 : tensor<576xf32>
    %v8590 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8591 = stablehlo.multiply %v8590, %b12dbt : tensor<576xf32>
    %v8592 = stablehlo.add %v8591, %v2864 : tensor<576xf32>
    %v8593 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8594 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8595 = stablehlo.multiply %v8593, %b12dbtv : tensor<576xf32>
    %v8596 = stablehlo.multiply %v8592, %v8592 : tensor<576xf32>
    %v8597 = stablehlo.multiply %v8594, %v8596 : tensor<576xf32>
    %v8598 = stablehlo.add %v8595, %v8597 : tensor<576xf32>
    %v8599 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8600 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8601 = stablehlo.multiply %v8599, %b12dbtv : tensor<576xf32>
    %v8602 = stablehlo.multiply %v8592, %v8592 : tensor<576xf32>
    %v8603 = stablehlo.multiply %v8600, %v8602 : tensor<576xf32>
    %v8604 = stablehlo.add %v8601, %v8603 : tensor<576xf32>
    %v8605 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8606 = stablehlo.add %v8604, %v8605 : tensor<576xf32>
    %v8607 = stablehlo.sqrt %v8606 : tensor<576xf32>
    %v8608 = stablehlo.divide %v8592, %v8607 : tensor<576xf32>
    %v8609 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8610 = stablehlo.multiply %v8609, %b12dbtm : tensor<576xf32>
    %v8611 = stablehlo.add %v8610, %v8608 : tensor<576xf32>
    %v8612 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8613 = stablehlo.multiply %v8612, %v8611 : tensor<576xf32>
    %v8614 = stablehlo.subtract %b12dbt, %v8613 : tensor<576xf32>
    %v8615 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8616 = stablehlo.multiply %v8615, %b12pW : tensor<96x576x1x1xf32>
    %v8617 = stablehlo.add %v8616, %v2870 : tensor<96x576x1x1xf32>
    %v8618 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8619 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8620 = stablehlo.multiply %v8618, %b12pWv : tensor<96x576x1x1xf32>
    %v8621 = stablehlo.multiply %v8617, %v8617 : tensor<96x576x1x1xf32>
    %v8622 = stablehlo.multiply %v8619, %v8621 : tensor<96x576x1x1xf32>
    %v8623 = stablehlo.add %v8620, %v8622 : tensor<96x576x1x1xf32>
    %v8624 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8625 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8626 = stablehlo.multiply %v8624, %b12pWv : tensor<96x576x1x1xf32>
    %v8627 = stablehlo.multiply %v8617, %v8617 : tensor<96x576x1x1xf32>
    %v8628 = stablehlo.multiply %v8625, %v8627 : tensor<96x576x1x1xf32>
    %v8629 = stablehlo.add %v8626, %v8628 : tensor<96x576x1x1xf32>
    %v8630 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8631 = stablehlo.add %v8629, %v8630 : tensor<96x576x1x1xf32>
    %v8632 = stablehlo.sqrt %v8631 : tensor<96x576x1x1xf32>
    %v8633 = stablehlo.divide %v8617, %v8632 : tensor<96x576x1x1xf32>
    %v8634 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8635 = stablehlo.multiply %v8634, %b12pWm : tensor<96x576x1x1xf32>
    %v8636 = stablehlo.add %v8635, %v8633 : tensor<96x576x1x1xf32>
    %v8637 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8638 = stablehlo.multiply %v8637, %v8636 : tensor<96x576x1x1xf32>
    %v8639 = stablehlo.subtract %b12pW, %v8638 : tensor<96x576x1x1xf32>
    %v8640 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8641 = stablehlo.multiply %v8640, %b12pg : tensor<96xf32>
    %v8642 = stablehlo.add %v8641, %v2888 : tensor<96xf32>
    %v8643 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8644 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8645 = stablehlo.multiply %v8643, %b12pgv : tensor<96xf32>
    %v8646 = stablehlo.multiply %v8642, %v8642 : tensor<96xf32>
    %v8647 = stablehlo.multiply %v8644, %v8646 : tensor<96xf32>
    %v8648 = stablehlo.add %v8645, %v8647 : tensor<96xf32>
    %v8649 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8650 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8651 = stablehlo.multiply %v8649, %b12pgv : tensor<96xf32>
    %v8652 = stablehlo.multiply %v8642, %v8642 : tensor<96xf32>
    %v8653 = stablehlo.multiply %v8650, %v8652 : tensor<96xf32>
    %v8654 = stablehlo.add %v8651, %v8653 : tensor<96xf32>
    %v8655 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8656 = stablehlo.add %v8654, %v8655 : tensor<96xf32>
    %v8657 = stablehlo.sqrt %v8656 : tensor<96xf32>
    %v8658 = stablehlo.divide %v8642, %v8657 : tensor<96xf32>
    %v8659 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8660 = stablehlo.multiply %v8659, %b12pgm : tensor<96xf32>
    %v8661 = stablehlo.add %v8660, %v8658 : tensor<96xf32>
    %v8662 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8663 = stablehlo.multiply %v8662, %v8661 : tensor<96xf32>
    %v8664 = stablehlo.subtract %b12pg, %v8663 : tensor<96xf32>
    %v8665 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8666 = stablehlo.multiply %v8665, %b12pbt : tensor<96xf32>
    %v8667 = stablehlo.add %v8666, %v2891 : tensor<96xf32>
    %v8668 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8669 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8670 = stablehlo.multiply %v8668, %b12pbtv : tensor<96xf32>
    %v8671 = stablehlo.multiply %v8667, %v8667 : tensor<96xf32>
    %v8672 = stablehlo.multiply %v8669, %v8671 : tensor<96xf32>
    %v8673 = stablehlo.add %v8670, %v8672 : tensor<96xf32>
    %v8674 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8675 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8676 = stablehlo.multiply %v8674, %b12pbtv : tensor<96xf32>
    %v8677 = stablehlo.multiply %v8667, %v8667 : tensor<96xf32>
    %v8678 = stablehlo.multiply %v8675, %v8677 : tensor<96xf32>
    %v8679 = stablehlo.add %v8676, %v8678 : tensor<96xf32>
    %v8680 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8681 = stablehlo.add %v8679, %v8680 : tensor<96xf32>
    %v8682 = stablehlo.sqrt %v8681 : tensor<96xf32>
    %v8683 = stablehlo.divide %v8667, %v8682 : tensor<96xf32>
    %v8684 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8685 = stablehlo.multiply %v8684, %b12pbtm : tensor<96xf32>
    %v8686 = stablehlo.add %v8685, %v8683 : tensor<96xf32>
    %v8687 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8688 = stablehlo.multiply %v8687, %v8686 : tensor<96xf32>
    %v8689 = stablehlo.subtract %b12pbt, %v8688 : tensor<96xf32>
    %v8690 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8691 = stablehlo.multiply %v8690, %b13eW : tensor<576x96x1x1xf32>
    %v8692 = stablehlo.add %v8691, %v2609 : tensor<576x96x1x1xf32>
    %v8693 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8694 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8695 = stablehlo.multiply %v8693, %b13eWv : tensor<576x96x1x1xf32>
    %v8696 = stablehlo.multiply %v8692, %v8692 : tensor<576x96x1x1xf32>
    %v8697 = stablehlo.multiply %v8694, %v8696 : tensor<576x96x1x1xf32>
    %v8698 = stablehlo.add %v8695, %v8697 : tensor<576x96x1x1xf32>
    %v8699 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8700 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8701 = stablehlo.multiply %v8699, %b13eWv : tensor<576x96x1x1xf32>
    %v8702 = stablehlo.multiply %v8692, %v8692 : tensor<576x96x1x1xf32>
    %v8703 = stablehlo.multiply %v8700, %v8702 : tensor<576x96x1x1xf32>
    %v8704 = stablehlo.add %v8701, %v8703 : tensor<576x96x1x1xf32>
    %v8705 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8706 = stablehlo.add %v8704, %v8705 : tensor<576x96x1x1xf32>
    %v8707 = stablehlo.sqrt %v8706 : tensor<576x96x1x1xf32>
    %v8708 = stablehlo.divide %v8692, %v8707 : tensor<576x96x1x1xf32>
    %v8709 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8710 = stablehlo.multiply %v8709, %b13eWm : tensor<576x96x1x1xf32>
    %v8711 = stablehlo.add %v8710, %v8708 : tensor<576x96x1x1xf32>
    %v8712 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8713 = stablehlo.multiply %v8712, %v8711 : tensor<576x96x1x1xf32>
    %v8714 = stablehlo.subtract %b13eW, %v8713 : tensor<576x96x1x1xf32>
    %v8715 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8716 = stablehlo.multiply %v8715, %b13eg : tensor<576xf32>
    %v8717 = stablehlo.add %v8716, %v2627 : tensor<576xf32>
    %v8718 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8719 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8720 = stablehlo.multiply %v8718, %b13egv : tensor<576xf32>
    %v8721 = stablehlo.multiply %v8717, %v8717 : tensor<576xf32>
    %v8722 = stablehlo.multiply %v8719, %v8721 : tensor<576xf32>
    %v8723 = stablehlo.add %v8720, %v8722 : tensor<576xf32>
    %v8724 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8725 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8726 = stablehlo.multiply %v8724, %b13egv : tensor<576xf32>
    %v8727 = stablehlo.multiply %v8717, %v8717 : tensor<576xf32>
    %v8728 = stablehlo.multiply %v8725, %v8727 : tensor<576xf32>
    %v8729 = stablehlo.add %v8726, %v8728 : tensor<576xf32>
    %v8730 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8731 = stablehlo.add %v8729, %v8730 : tensor<576xf32>
    %v8732 = stablehlo.sqrt %v8731 : tensor<576xf32>
    %v8733 = stablehlo.divide %v8717, %v8732 : tensor<576xf32>
    %v8734 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8735 = stablehlo.multiply %v8734, %b13egm : tensor<576xf32>
    %v8736 = stablehlo.add %v8735, %v8733 : tensor<576xf32>
    %v8737 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8738 = stablehlo.multiply %v8737, %v8736 : tensor<576xf32>
    %v8739 = stablehlo.subtract %b13eg, %v8738 : tensor<576xf32>
    %v8740 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8741 = stablehlo.multiply %v8740, %b13ebt : tensor<576xf32>
    %v8742 = stablehlo.add %v8741, %v2630 : tensor<576xf32>
    %v8743 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8744 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8745 = stablehlo.multiply %v8743, %b13ebtv : tensor<576xf32>
    %v8746 = stablehlo.multiply %v8742, %v8742 : tensor<576xf32>
    %v8747 = stablehlo.multiply %v8744, %v8746 : tensor<576xf32>
    %v8748 = stablehlo.add %v8745, %v8747 : tensor<576xf32>
    %v8749 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8750 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8751 = stablehlo.multiply %v8749, %b13ebtv : tensor<576xf32>
    %v8752 = stablehlo.multiply %v8742, %v8742 : tensor<576xf32>
    %v8753 = stablehlo.multiply %v8750, %v8752 : tensor<576xf32>
    %v8754 = stablehlo.add %v8751, %v8753 : tensor<576xf32>
    %v8755 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8756 = stablehlo.add %v8754, %v8755 : tensor<576xf32>
    %v8757 = stablehlo.sqrt %v8756 : tensor<576xf32>
    %v8758 = stablehlo.divide %v8742, %v8757 : tensor<576xf32>
    %v8759 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8760 = stablehlo.multiply %v8759, %b13ebtm : tensor<576xf32>
    %v8761 = stablehlo.add %v8760, %v8758 : tensor<576xf32>
    %v8762 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8763 = stablehlo.multiply %v8762, %v8761 : tensor<576xf32>
    %v8764 = stablehlo.subtract %b13ebt, %v8763 : tensor<576xf32>
    %v8765 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8766 = stablehlo.multiply %v8765, %b13dW : tensor<576x1x3x3xf32>
    %v8767 = stablehlo.add %v8766, %v2636 : tensor<576x1x3x3xf32>
    %v8768 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8769 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8770 = stablehlo.multiply %v8768, %b13dWv : tensor<576x1x3x3xf32>
    %v8771 = stablehlo.multiply %v8767, %v8767 : tensor<576x1x3x3xf32>
    %v8772 = stablehlo.multiply %v8769, %v8771 : tensor<576x1x3x3xf32>
    %v8773 = stablehlo.add %v8770, %v8772 : tensor<576x1x3x3xf32>
    %v8774 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8775 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8776 = stablehlo.multiply %v8774, %b13dWv : tensor<576x1x3x3xf32>
    %v8777 = stablehlo.multiply %v8767, %v8767 : tensor<576x1x3x3xf32>
    %v8778 = stablehlo.multiply %v8775, %v8777 : tensor<576x1x3x3xf32>
    %v8779 = stablehlo.add %v8776, %v8778 : tensor<576x1x3x3xf32>
    %v8780 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8781 = stablehlo.add %v8779, %v8780 : tensor<576x1x3x3xf32>
    %v8782 = stablehlo.sqrt %v8781 : tensor<576x1x3x3xf32>
    %v8783 = stablehlo.divide %v8767, %v8782 : tensor<576x1x3x3xf32>
    %v8784 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8785 = stablehlo.multiply %v8784, %b13dWm : tensor<576x1x3x3xf32>
    %v8786 = stablehlo.add %v8785, %v8783 : tensor<576x1x3x3xf32>
    %v8787 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8788 = stablehlo.multiply %v8787, %v8786 : tensor<576x1x3x3xf32>
    %v8789 = stablehlo.subtract %b13dW, %v8788 : tensor<576x1x3x3xf32>
    %v8790 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8791 = stablehlo.multiply %v8790, %b13dg : tensor<576xf32>
    %v8792 = stablehlo.add %v8791, %v2654 : tensor<576xf32>
    %v8793 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8794 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8795 = stablehlo.multiply %v8793, %b13dgv : tensor<576xf32>
    %v8796 = stablehlo.multiply %v8792, %v8792 : tensor<576xf32>
    %v8797 = stablehlo.multiply %v8794, %v8796 : tensor<576xf32>
    %v8798 = stablehlo.add %v8795, %v8797 : tensor<576xf32>
    %v8799 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8800 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8801 = stablehlo.multiply %v8799, %b13dgv : tensor<576xf32>
    %v8802 = stablehlo.multiply %v8792, %v8792 : tensor<576xf32>
    %v8803 = stablehlo.multiply %v8800, %v8802 : tensor<576xf32>
    %v8804 = stablehlo.add %v8801, %v8803 : tensor<576xf32>
    %v8805 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8806 = stablehlo.add %v8804, %v8805 : tensor<576xf32>
    %v8807 = stablehlo.sqrt %v8806 : tensor<576xf32>
    %v8808 = stablehlo.divide %v8792, %v8807 : tensor<576xf32>
    %v8809 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8810 = stablehlo.multiply %v8809, %b13dgm : tensor<576xf32>
    %v8811 = stablehlo.add %v8810, %v8808 : tensor<576xf32>
    %v8812 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8813 = stablehlo.multiply %v8812, %v8811 : tensor<576xf32>
    %v8814 = stablehlo.subtract %b13dg, %v8813 : tensor<576xf32>
    %v8815 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8816 = stablehlo.multiply %v8815, %b13dbt : tensor<576xf32>
    %v8817 = stablehlo.add %v8816, %v2657 : tensor<576xf32>
    %v8818 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8819 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8820 = stablehlo.multiply %v8818, %b13dbtv : tensor<576xf32>
    %v8821 = stablehlo.multiply %v8817, %v8817 : tensor<576xf32>
    %v8822 = stablehlo.multiply %v8819, %v8821 : tensor<576xf32>
    %v8823 = stablehlo.add %v8820, %v8822 : tensor<576xf32>
    %v8824 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8825 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8826 = stablehlo.multiply %v8824, %b13dbtv : tensor<576xf32>
    %v8827 = stablehlo.multiply %v8817, %v8817 : tensor<576xf32>
    %v8828 = stablehlo.multiply %v8825, %v8827 : tensor<576xf32>
    %v8829 = stablehlo.add %v8826, %v8828 : tensor<576xf32>
    %v8830 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8831 = stablehlo.add %v8829, %v8830 : tensor<576xf32>
    %v8832 = stablehlo.sqrt %v8831 : tensor<576xf32>
    %v8833 = stablehlo.divide %v8817, %v8832 : tensor<576xf32>
    %v8834 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8835 = stablehlo.multiply %v8834, %b13dbtm : tensor<576xf32>
    %v8836 = stablehlo.add %v8835, %v8833 : tensor<576xf32>
    %v8837 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8838 = stablehlo.multiply %v8837, %v8836 : tensor<576xf32>
    %v8839 = stablehlo.subtract %b13dbt, %v8838 : tensor<576xf32>
    %v8840 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8841 = stablehlo.multiply %v8840, %b13pW : tensor<96x576x1x1xf32>
    %v8842 = stablehlo.add %v8841, %v2663 : tensor<96x576x1x1xf32>
    %v8843 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8844 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8845 = stablehlo.multiply %v8843, %b13pWv : tensor<96x576x1x1xf32>
    %v8846 = stablehlo.multiply %v8842, %v8842 : tensor<96x576x1x1xf32>
    %v8847 = stablehlo.multiply %v8844, %v8846 : tensor<96x576x1x1xf32>
    %v8848 = stablehlo.add %v8845, %v8847 : tensor<96x576x1x1xf32>
    %v8849 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8850 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8851 = stablehlo.multiply %v8849, %b13pWv : tensor<96x576x1x1xf32>
    %v8852 = stablehlo.multiply %v8842, %v8842 : tensor<96x576x1x1xf32>
    %v8853 = stablehlo.multiply %v8850, %v8852 : tensor<96x576x1x1xf32>
    %v8854 = stablehlo.add %v8851, %v8853 : tensor<96x576x1x1xf32>
    %v8855 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8856 = stablehlo.add %v8854, %v8855 : tensor<96x576x1x1xf32>
    %v8857 = stablehlo.sqrt %v8856 : tensor<96x576x1x1xf32>
    %v8858 = stablehlo.divide %v8842, %v8857 : tensor<96x576x1x1xf32>
    %v8859 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8860 = stablehlo.multiply %v8859, %b13pWm : tensor<96x576x1x1xf32>
    %v8861 = stablehlo.add %v8860, %v8858 : tensor<96x576x1x1xf32>
    %v8862 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x576x1x1xf32>
    %v8863 = stablehlo.multiply %v8862, %v8861 : tensor<96x576x1x1xf32>
    %v8864 = stablehlo.subtract %b13pW, %v8863 : tensor<96x576x1x1xf32>
    %v8865 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8866 = stablehlo.multiply %v8865, %b13pg : tensor<96xf32>
    %v8867 = stablehlo.add %v8866, %v2681 : tensor<96xf32>
    %v8868 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8869 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8870 = stablehlo.multiply %v8868, %b13pgv : tensor<96xf32>
    %v8871 = stablehlo.multiply %v8867, %v8867 : tensor<96xf32>
    %v8872 = stablehlo.multiply %v8869, %v8871 : tensor<96xf32>
    %v8873 = stablehlo.add %v8870, %v8872 : tensor<96xf32>
    %v8874 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8875 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8876 = stablehlo.multiply %v8874, %b13pgv : tensor<96xf32>
    %v8877 = stablehlo.multiply %v8867, %v8867 : tensor<96xf32>
    %v8878 = stablehlo.multiply %v8875, %v8877 : tensor<96xf32>
    %v8879 = stablehlo.add %v8876, %v8878 : tensor<96xf32>
    %v8880 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8881 = stablehlo.add %v8879, %v8880 : tensor<96xf32>
    %v8882 = stablehlo.sqrt %v8881 : tensor<96xf32>
    %v8883 = stablehlo.divide %v8867, %v8882 : tensor<96xf32>
    %v8884 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8885 = stablehlo.multiply %v8884, %b13pgm : tensor<96xf32>
    %v8886 = stablehlo.add %v8885, %v8883 : tensor<96xf32>
    %v8887 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8888 = stablehlo.multiply %v8887, %v8886 : tensor<96xf32>
    %v8889 = stablehlo.subtract %b13pg, %v8888 : tensor<96xf32>
    %v8890 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8891 = stablehlo.multiply %v8890, %b13pbt : tensor<96xf32>
    %v8892 = stablehlo.add %v8891, %v2684 : tensor<96xf32>
    %v8893 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8894 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8895 = stablehlo.multiply %v8893, %b13pbtv : tensor<96xf32>
    %v8896 = stablehlo.multiply %v8892, %v8892 : tensor<96xf32>
    %v8897 = stablehlo.multiply %v8894, %v8896 : tensor<96xf32>
    %v8898 = stablehlo.add %v8895, %v8897 : tensor<96xf32>
    %v8899 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8900 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8901 = stablehlo.multiply %v8899, %b13pbtv : tensor<96xf32>
    %v8902 = stablehlo.multiply %v8892, %v8892 : tensor<96xf32>
    %v8903 = stablehlo.multiply %v8900, %v8902 : tensor<96xf32>
    %v8904 = stablehlo.add %v8901, %v8903 : tensor<96xf32>
    %v8905 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8906 = stablehlo.add %v8904, %v8905 : tensor<96xf32>
    %v8907 = stablehlo.sqrt %v8906 : tensor<96xf32>
    %v8908 = stablehlo.divide %v8892, %v8907 : tensor<96xf32>
    %v8909 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8910 = stablehlo.multiply %v8909, %b13pbtm : tensor<96xf32>
    %v8911 = stablehlo.add %v8910, %v8908 : tensor<96xf32>
    %v8912 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v8913 = stablehlo.multiply %v8912, %v8911 : tensor<96xf32>
    %v8914 = stablehlo.subtract %b13pbt, %v8913 : tensor<96xf32>
    %v8915 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8916 = stablehlo.multiply %v8915, %b14eW : tensor<576x96x1x1xf32>
    %v8917 = stablehlo.add %v8916, %v2400 : tensor<576x96x1x1xf32>
    %v8918 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8919 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8920 = stablehlo.multiply %v8918, %b14eWv : tensor<576x96x1x1xf32>
    %v8921 = stablehlo.multiply %v8917, %v8917 : tensor<576x96x1x1xf32>
    %v8922 = stablehlo.multiply %v8919, %v8921 : tensor<576x96x1x1xf32>
    %v8923 = stablehlo.add %v8920, %v8922 : tensor<576x96x1x1xf32>
    %v8924 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8925 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8926 = stablehlo.multiply %v8924, %b14eWv : tensor<576x96x1x1xf32>
    %v8927 = stablehlo.multiply %v8917, %v8917 : tensor<576x96x1x1xf32>
    %v8928 = stablehlo.multiply %v8925, %v8927 : tensor<576x96x1x1xf32>
    %v8929 = stablehlo.add %v8926, %v8928 : tensor<576x96x1x1xf32>
    %v8930 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8931 = stablehlo.add %v8929, %v8930 : tensor<576x96x1x1xf32>
    %v8932 = stablehlo.sqrt %v8931 : tensor<576x96x1x1xf32>
    %v8933 = stablehlo.divide %v8917, %v8932 : tensor<576x96x1x1xf32>
    %v8934 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8935 = stablehlo.multiply %v8934, %b14eWm : tensor<576x96x1x1xf32>
    %v8936 = stablehlo.add %v8935, %v8933 : tensor<576x96x1x1xf32>
    %v8937 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x96x1x1xf32>
    %v8938 = stablehlo.multiply %v8937, %v8936 : tensor<576x96x1x1xf32>
    %v8939 = stablehlo.subtract %b14eW, %v8938 : tensor<576x96x1x1xf32>
    %v8940 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8941 = stablehlo.multiply %v8940, %b14eg : tensor<576xf32>
    %v8942 = stablehlo.add %v8941, %v2418 : tensor<576xf32>
    %v8943 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8944 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8945 = stablehlo.multiply %v8943, %b14egv : tensor<576xf32>
    %v8946 = stablehlo.multiply %v8942, %v8942 : tensor<576xf32>
    %v8947 = stablehlo.multiply %v8944, %v8946 : tensor<576xf32>
    %v8948 = stablehlo.add %v8945, %v8947 : tensor<576xf32>
    %v8949 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8950 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8951 = stablehlo.multiply %v8949, %b14egv : tensor<576xf32>
    %v8952 = stablehlo.multiply %v8942, %v8942 : tensor<576xf32>
    %v8953 = stablehlo.multiply %v8950, %v8952 : tensor<576xf32>
    %v8954 = stablehlo.add %v8951, %v8953 : tensor<576xf32>
    %v8955 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8956 = stablehlo.add %v8954, %v8955 : tensor<576xf32>
    %v8957 = stablehlo.sqrt %v8956 : tensor<576xf32>
    %v8958 = stablehlo.divide %v8942, %v8957 : tensor<576xf32>
    %v8959 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8960 = stablehlo.multiply %v8959, %b14egm : tensor<576xf32>
    %v8961 = stablehlo.add %v8960, %v8958 : tensor<576xf32>
    %v8962 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8963 = stablehlo.multiply %v8962, %v8961 : tensor<576xf32>
    %v8964 = stablehlo.subtract %b14eg, %v8963 : tensor<576xf32>
    %v8965 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8966 = stablehlo.multiply %v8965, %b14ebt : tensor<576xf32>
    %v8967 = stablehlo.add %v8966, %v2421 : tensor<576xf32>
    %v8968 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8969 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8970 = stablehlo.multiply %v8968, %b14ebtv : tensor<576xf32>
    %v8971 = stablehlo.multiply %v8967, %v8967 : tensor<576xf32>
    %v8972 = stablehlo.multiply %v8969, %v8971 : tensor<576xf32>
    %v8973 = stablehlo.add %v8970, %v8972 : tensor<576xf32>
    %v8974 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8975 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8976 = stablehlo.multiply %v8974, %b14ebtv : tensor<576xf32>
    %v8977 = stablehlo.multiply %v8967, %v8967 : tensor<576xf32>
    %v8978 = stablehlo.multiply %v8975, %v8977 : tensor<576xf32>
    %v8979 = stablehlo.add %v8976, %v8978 : tensor<576xf32>
    %v8980 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8981 = stablehlo.add %v8979, %v8980 : tensor<576xf32>
    %v8982 = stablehlo.sqrt %v8981 : tensor<576xf32>
    %v8983 = stablehlo.divide %v8967, %v8982 : tensor<576xf32>
    %v8984 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8985 = stablehlo.multiply %v8984, %b14ebtm : tensor<576xf32>
    %v8986 = stablehlo.add %v8985, %v8983 : tensor<576xf32>
    %v8987 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v8988 = stablehlo.multiply %v8987, %v8986 : tensor<576xf32>
    %v8989 = stablehlo.subtract %b14ebt, %v8988 : tensor<576xf32>
    %v8990 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8991 = stablehlo.multiply %v8990, %b14dW : tensor<576x1x3x3xf32>
    %v8992 = stablehlo.add %v8991, %v2429 : tensor<576x1x3x3xf32>
    %v8993 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8994 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v8995 = stablehlo.multiply %v8993, %b14dWv : tensor<576x1x3x3xf32>
    %v8996 = stablehlo.multiply %v8992, %v8992 : tensor<576x1x3x3xf32>
    %v8997 = stablehlo.multiply %v8994, %v8996 : tensor<576x1x3x3xf32>
    %v8998 = stablehlo.add %v8995, %v8997 : tensor<576x1x3x3xf32>
    %v8999 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v9000 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v9001 = stablehlo.multiply %v8999, %b14dWv : tensor<576x1x3x3xf32>
    %v9002 = stablehlo.multiply %v8992, %v8992 : tensor<576x1x3x3xf32>
    %v9003 = stablehlo.multiply %v9000, %v9002 : tensor<576x1x3x3xf32>
    %v9004 = stablehlo.add %v9001, %v9003 : tensor<576x1x3x3xf32>
    %v9005 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v9006 = stablehlo.add %v9004, %v9005 : tensor<576x1x3x3xf32>
    %v9007 = stablehlo.sqrt %v9006 : tensor<576x1x3x3xf32>
    %v9008 = stablehlo.divide %v8992, %v9007 : tensor<576x1x3x3xf32>
    %v9009 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v9010 = stablehlo.multiply %v9009, %b14dWm : tensor<576x1x3x3xf32>
    %v9011 = stablehlo.add %v9010, %v9008 : tensor<576x1x3x3xf32>
    %v9012 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576x1x3x3xf32>
    %v9013 = stablehlo.multiply %v9012, %v9011 : tensor<576x1x3x3xf32>
    %v9014 = stablehlo.subtract %b14dW, %v9013 : tensor<576x1x3x3xf32>
    %v9015 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9016 = stablehlo.multiply %v9015, %b14dg : tensor<576xf32>
    %v9017 = stablehlo.add %v9016, %v2447 : tensor<576xf32>
    %v9018 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9019 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9020 = stablehlo.multiply %v9018, %b14dgv : tensor<576xf32>
    %v9021 = stablehlo.multiply %v9017, %v9017 : tensor<576xf32>
    %v9022 = stablehlo.multiply %v9019, %v9021 : tensor<576xf32>
    %v9023 = stablehlo.add %v9020, %v9022 : tensor<576xf32>
    %v9024 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9025 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9026 = stablehlo.multiply %v9024, %b14dgv : tensor<576xf32>
    %v9027 = stablehlo.multiply %v9017, %v9017 : tensor<576xf32>
    %v9028 = stablehlo.multiply %v9025, %v9027 : tensor<576xf32>
    %v9029 = stablehlo.add %v9026, %v9028 : tensor<576xf32>
    %v9030 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9031 = stablehlo.add %v9029, %v9030 : tensor<576xf32>
    %v9032 = stablehlo.sqrt %v9031 : tensor<576xf32>
    %v9033 = stablehlo.divide %v9017, %v9032 : tensor<576xf32>
    %v9034 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9035 = stablehlo.multiply %v9034, %b14dgm : tensor<576xf32>
    %v9036 = stablehlo.add %v9035, %v9033 : tensor<576xf32>
    %v9037 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9038 = stablehlo.multiply %v9037, %v9036 : tensor<576xf32>
    %v9039 = stablehlo.subtract %b14dg, %v9038 : tensor<576xf32>
    %v9040 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9041 = stablehlo.multiply %v9040, %b14dbt : tensor<576xf32>
    %v9042 = stablehlo.add %v9041, %v2450 : tensor<576xf32>
    %v9043 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9044 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9045 = stablehlo.multiply %v9043, %b14dbtv : tensor<576xf32>
    %v9046 = stablehlo.multiply %v9042, %v9042 : tensor<576xf32>
    %v9047 = stablehlo.multiply %v9044, %v9046 : tensor<576xf32>
    %v9048 = stablehlo.add %v9045, %v9047 : tensor<576xf32>
    %v9049 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9050 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9051 = stablehlo.multiply %v9049, %b14dbtv : tensor<576xf32>
    %v9052 = stablehlo.multiply %v9042, %v9042 : tensor<576xf32>
    %v9053 = stablehlo.multiply %v9050, %v9052 : tensor<576xf32>
    %v9054 = stablehlo.add %v9051, %v9053 : tensor<576xf32>
    %v9055 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9056 = stablehlo.add %v9054, %v9055 : tensor<576xf32>
    %v9057 = stablehlo.sqrt %v9056 : tensor<576xf32>
    %v9058 = stablehlo.divide %v9042, %v9057 : tensor<576xf32>
    %v9059 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9060 = stablehlo.multiply %v9059, %b14dbtm : tensor<576xf32>
    %v9061 = stablehlo.add %v9060, %v9058 : tensor<576xf32>
    %v9062 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<576xf32>
    %v9063 = stablehlo.multiply %v9062, %v9061 : tensor<576xf32>
    %v9064 = stablehlo.subtract %b14dbt, %v9063 : tensor<576xf32>
    %v9065 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9066 = stablehlo.multiply %v9065, %b14pW : tensor<160x576x1x1xf32>
    %v9067 = stablehlo.add %v9066, %v2456 : tensor<160x576x1x1xf32>
    %v9068 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9069 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9070 = stablehlo.multiply %v9068, %b14pWv : tensor<160x576x1x1xf32>
    %v9071 = stablehlo.multiply %v9067, %v9067 : tensor<160x576x1x1xf32>
    %v9072 = stablehlo.multiply %v9069, %v9071 : tensor<160x576x1x1xf32>
    %v9073 = stablehlo.add %v9070, %v9072 : tensor<160x576x1x1xf32>
    %v9074 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9075 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9076 = stablehlo.multiply %v9074, %b14pWv : tensor<160x576x1x1xf32>
    %v9077 = stablehlo.multiply %v9067, %v9067 : tensor<160x576x1x1xf32>
    %v9078 = stablehlo.multiply %v9075, %v9077 : tensor<160x576x1x1xf32>
    %v9079 = stablehlo.add %v9076, %v9078 : tensor<160x576x1x1xf32>
    %v9080 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9081 = stablehlo.add %v9079, %v9080 : tensor<160x576x1x1xf32>
    %v9082 = stablehlo.sqrt %v9081 : tensor<160x576x1x1xf32>
    %v9083 = stablehlo.divide %v9067, %v9082 : tensor<160x576x1x1xf32>
    %v9084 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9085 = stablehlo.multiply %v9084, %b14pWm : tensor<160x576x1x1xf32>
    %v9086 = stablehlo.add %v9085, %v9083 : tensor<160x576x1x1xf32>
    %v9087 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160x576x1x1xf32>
    %v9088 = stablehlo.multiply %v9087, %v9086 : tensor<160x576x1x1xf32>
    %v9089 = stablehlo.subtract %b14pW, %v9088 : tensor<160x576x1x1xf32>
    %v9090 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9091 = stablehlo.multiply %v9090, %b14pg : tensor<160xf32>
    %v9092 = stablehlo.add %v9091, %v2474 : tensor<160xf32>
    %v9093 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9094 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9095 = stablehlo.multiply %v9093, %b14pgv : tensor<160xf32>
    %v9096 = stablehlo.multiply %v9092, %v9092 : tensor<160xf32>
    %v9097 = stablehlo.multiply %v9094, %v9096 : tensor<160xf32>
    %v9098 = stablehlo.add %v9095, %v9097 : tensor<160xf32>
    %v9099 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9100 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9101 = stablehlo.multiply %v9099, %b14pgv : tensor<160xf32>
    %v9102 = stablehlo.multiply %v9092, %v9092 : tensor<160xf32>
    %v9103 = stablehlo.multiply %v9100, %v9102 : tensor<160xf32>
    %v9104 = stablehlo.add %v9101, %v9103 : tensor<160xf32>
    %v9105 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9106 = stablehlo.add %v9104, %v9105 : tensor<160xf32>
    %v9107 = stablehlo.sqrt %v9106 : tensor<160xf32>
    %v9108 = stablehlo.divide %v9092, %v9107 : tensor<160xf32>
    %v9109 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9110 = stablehlo.multiply %v9109, %b14pgm : tensor<160xf32>
    %v9111 = stablehlo.add %v9110, %v9108 : tensor<160xf32>
    %v9112 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9113 = stablehlo.multiply %v9112, %v9111 : tensor<160xf32>
    %v9114 = stablehlo.subtract %b14pg, %v9113 : tensor<160xf32>
    %v9115 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9116 = stablehlo.multiply %v9115, %b14pbt : tensor<160xf32>
    %v9117 = stablehlo.add %v9116, %v2477 : tensor<160xf32>
    %v9118 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9119 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9120 = stablehlo.multiply %v9118, %b14pbtv : tensor<160xf32>
    %v9121 = stablehlo.multiply %v9117, %v9117 : tensor<160xf32>
    %v9122 = stablehlo.multiply %v9119, %v9121 : tensor<160xf32>
    %v9123 = stablehlo.add %v9120, %v9122 : tensor<160xf32>
    %v9124 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9125 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9126 = stablehlo.multiply %v9124, %b14pbtv : tensor<160xf32>
    %v9127 = stablehlo.multiply %v9117, %v9117 : tensor<160xf32>
    %v9128 = stablehlo.multiply %v9125, %v9127 : tensor<160xf32>
    %v9129 = stablehlo.add %v9126, %v9128 : tensor<160xf32>
    %v9130 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9131 = stablehlo.add %v9129, %v9130 : tensor<160xf32>
    %v9132 = stablehlo.sqrt %v9131 : tensor<160xf32>
    %v9133 = stablehlo.divide %v9117, %v9132 : tensor<160xf32>
    %v9134 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9135 = stablehlo.multiply %v9134, %b14pbtm : tensor<160xf32>
    %v9136 = stablehlo.add %v9135, %v9133 : tensor<160xf32>
    %v9137 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9138 = stablehlo.multiply %v9137, %v9136 : tensor<160xf32>
    %v9139 = stablehlo.subtract %b14pbt, %v9138 : tensor<160xf32>
    %v9140 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9141 = stablehlo.multiply %v9140, %b15eW : tensor<960x160x1x1xf32>
    %v9142 = stablehlo.add %v9141, %v2195 : tensor<960x160x1x1xf32>
    %v9143 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9144 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9145 = stablehlo.multiply %v9143, %b15eWv : tensor<960x160x1x1xf32>
    %v9146 = stablehlo.multiply %v9142, %v9142 : tensor<960x160x1x1xf32>
    %v9147 = stablehlo.multiply %v9144, %v9146 : tensor<960x160x1x1xf32>
    %v9148 = stablehlo.add %v9145, %v9147 : tensor<960x160x1x1xf32>
    %v9149 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9150 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9151 = stablehlo.multiply %v9149, %b15eWv : tensor<960x160x1x1xf32>
    %v9152 = stablehlo.multiply %v9142, %v9142 : tensor<960x160x1x1xf32>
    %v9153 = stablehlo.multiply %v9150, %v9152 : tensor<960x160x1x1xf32>
    %v9154 = stablehlo.add %v9151, %v9153 : tensor<960x160x1x1xf32>
    %v9155 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9156 = stablehlo.add %v9154, %v9155 : tensor<960x160x1x1xf32>
    %v9157 = stablehlo.sqrt %v9156 : tensor<960x160x1x1xf32>
    %v9158 = stablehlo.divide %v9142, %v9157 : tensor<960x160x1x1xf32>
    %v9159 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9160 = stablehlo.multiply %v9159, %b15eWm : tensor<960x160x1x1xf32>
    %v9161 = stablehlo.add %v9160, %v9158 : tensor<960x160x1x1xf32>
    %v9162 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9163 = stablehlo.multiply %v9162, %v9161 : tensor<960x160x1x1xf32>
    %v9164 = stablehlo.subtract %b15eW, %v9163 : tensor<960x160x1x1xf32>
    %v9165 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9166 = stablehlo.multiply %v9165, %b15eg : tensor<960xf32>
    %v9167 = stablehlo.add %v9166, %v2213 : tensor<960xf32>
    %v9168 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9169 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9170 = stablehlo.multiply %v9168, %b15egv : tensor<960xf32>
    %v9171 = stablehlo.multiply %v9167, %v9167 : tensor<960xf32>
    %v9172 = stablehlo.multiply %v9169, %v9171 : tensor<960xf32>
    %v9173 = stablehlo.add %v9170, %v9172 : tensor<960xf32>
    %v9174 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9175 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9176 = stablehlo.multiply %v9174, %b15egv : tensor<960xf32>
    %v9177 = stablehlo.multiply %v9167, %v9167 : tensor<960xf32>
    %v9178 = stablehlo.multiply %v9175, %v9177 : tensor<960xf32>
    %v9179 = stablehlo.add %v9176, %v9178 : tensor<960xf32>
    %v9180 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9181 = stablehlo.add %v9179, %v9180 : tensor<960xf32>
    %v9182 = stablehlo.sqrt %v9181 : tensor<960xf32>
    %v9183 = stablehlo.divide %v9167, %v9182 : tensor<960xf32>
    %v9184 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9185 = stablehlo.multiply %v9184, %b15egm : tensor<960xf32>
    %v9186 = stablehlo.add %v9185, %v9183 : tensor<960xf32>
    %v9187 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9188 = stablehlo.multiply %v9187, %v9186 : tensor<960xf32>
    %v9189 = stablehlo.subtract %b15eg, %v9188 : tensor<960xf32>
    %v9190 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9191 = stablehlo.multiply %v9190, %b15ebt : tensor<960xf32>
    %v9192 = stablehlo.add %v9191, %v2216 : tensor<960xf32>
    %v9193 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9194 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9195 = stablehlo.multiply %v9193, %b15ebtv : tensor<960xf32>
    %v9196 = stablehlo.multiply %v9192, %v9192 : tensor<960xf32>
    %v9197 = stablehlo.multiply %v9194, %v9196 : tensor<960xf32>
    %v9198 = stablehlo.add %v9195, %v9197 : tensor<960xf32>
    %v9199 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9200 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9201 = stablehlo.multiply %v9199, %b15ebtv : tensor<960xf32>
    %v9202 = stablehlo.multiply %v9192, %v9192 : tensor<960xf32>
    %v9203 = stablehlo.multiply %v9200, %v9202 : tensor<960xf32>
    %v9204 = stablehlo.add %v9201, %v9203 : tensor<960xf32>
    %v9205 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9206 = stablehlo.add %v9204, %v9205 : tensor<960xf32>
    %v9207 = stablehlo.sqrt %v9206 : tensor<960xf32>
    %v9208 = stablehlo.divide %v9192, %v9207 : tensor<960xf32>
    %v9209 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9210 = stablehlo.multiply %v9209, %b15ebtm : tensor<960xf32>
    %v9211 = stablehlo.add %v9210, %v9208 : tensor<960xf32>
    %v9212 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9213 = stablehlo.multiply %v9212, %v9211 : tensor<960xf32>
    %v9214 = stablehlo.subtract %b15ebt, %v9213 : tensor<960xf32>
    %v9215 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9216 = stablehlo.multiply %v9215, %b15dW : tensor<960x1x3x3xf32>
    %v9217 = stablehlo.add %v9216, %v2222 : tensor<960x1x3x3xf32>
    %v9218 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9219 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9220 = stablehlo.multiply %v9218, %b15dWv : tensor<960x1x3x3xf32>
    %v9221 = stablehlo.multiply %v9217, %v9217 : tensor<960x1x3x3xf32>
    %v9222 = stablehlo.multiply %v9219, %v9221 : tensor<960x1x3x3xf32>
    %v9223 = stablehlo.add %v9220, %v9222 : tensor<960x1x3x3xf32>
    %v9224 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9225 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9226 = stablehlo.multiply %v9224, %b15dWv : tensor<960x1x3x3xf32>
    %v9227 = stablehlo.multiply %v9217, %v9217 : tensor<960x1x3x3xf32>
    %v9228 = stablehlo.multiply %v9225, %v9227 : tensor<960x1x3x3xf32>
    %v9229 = stablehlo.add %v9226, %v9228 : tensor<960x1x3x3xf32>
    %v9230 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9231 = stablehlo.add %v9229, %v9230 : tensor<960x1x3x3xf32>
    %v9232 = stablehlo.sqrt %v9231 : tensor<960x1x3x3xf32>
    %v9233 = stablehlo.divide %v9217, %v9232 : tensor<960x1x3x3xf32>
    %v9234 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9235 = stablehlo.multiply %v9234, %b15dWm : tensor<960x1x3x3xf32>
    %v9236 = stablehlo.add %v9235, %v9233 : tensor<960x1x3x3xf32>
    %v9237 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9238 = stablehlo.multiply %v9237, %v9236 : tensor<960x1x3x3xf32>
    %v9239 = stablehlo.subtract %b15dW, %v9238 : tensor<960x1x3x3xf32>
    %v9240 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9241 = stablehlo.multiply %v9240, %b15dg : tensor<960xf32>
    %v9242 = stablehlo.add %v9241, %v2240 : tensor<960xf32>
    %v9243 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9244 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9245 = stablehlo.multiply %v9243, %b15dgv : tensor<960xf32>
    %v9246 = stablehlo.multiply %v9242, %v9242 : tensor<960xf32>
    %v9247 = stablehlo.multiply %v9244, %v9246 : tensor<960xf32>
    %v9248 = stablehlo.add %v9245, %v9247 : tensor<960xf32>
    %v9249 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9250 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9251 = stablehlo.multiply %v9249, %b15dgv : tensor<960xf32>
    %v9252 = stablehlo.multiply %v9242, %v9242 : tensor<960xf32>
    %v9253 = stablehlo.multiply %v9250, %v9252 : tensor<960xf32>
    %v9254 = stablehlo.add %v9251, %v9253 : tensor<960xf32>
    %v9255 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9256 = stablehlo.add %v9254, %v9255 : tensor<960xf32>
    %v9257 = stablehlo.sqrt %v9256 : tensor<960xf32>
    %v9258 = stablehlo.divide %v9242, %v9257 : tensor<960xf32>
    %v9259 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9260 = stablehlo.multiply %v9259, %b15dgm : tensor<960xf32>
    %v9261 = stablehlo.add %v9260, %v9258 : tensor<960xf32>
    %v9262 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9263 = stablehlo.multiply %v9262, %v9261 : tensor<960xf32>
    %v9264 = stablehlo.subtract %b15dg, %v9263 : tensor<960xf32>
    %v9265 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9266 = stablehlo.multiply %v9265, %b15dbt : tensor<960xf32>
    %v9267 = stablehlo.add %v9266, %v2243 : tensor<960xf32>
    %v9268 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9269 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9270 = stablehlo.multiply %v9268, %b15dbtv : tensor<960xf32>
    %v9271 = stablehlo.multiply %v9267, %v9267 : tensor<960xf32>
    %v9272 = stablehlo.multiply %v9269, %v9271 : tensor<960xf32>
    %v9273 = stablehlo.add %v9270, %v9272 : tensor<960xf32>
    %v9274 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9275 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9276 = stablehlo.multiply %v9274, %b15dbtv : tensor<960xf32>
    %v9277 = stablehlo.multiply %v9267, %v9267 : tensor<960xf32>
    %v9278 = stablehlo.multiply %v9275, %v9277 : tensor<960xf32>
    %v9279 = stablehlo.add %v9276, %v9278 : tensor<960xf32>
    %v9280 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9281 = stablehlo.add %v9279, %v9280 : tensor<960xf32>
    %v9282 = stablehlo.sqrt %v9281 : tensor<960xf32>
    %v9283 = stablehlo.divide %v9267, %v9282 : tensor<960xf32>
    %v9284 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9285 = stablehlo.multiply %v9284, %b15dbtm : tensor<960xf32>
    %v9286 = stablehlo.add %v9285, %v9283 : tensor<960xf32>
    %v9287 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9288 = stablehlo.multiply %v9287, %v9286 : tensor<960xf32>
    %v9289 = stablehlo.subtract %b15dbt, %v9288 : tensor<960xf32>
    %v9290 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9291 = stablehlo.multiply %v9290, %b15pW : tensor<160x960x1x1xf32>
    %v9292 = stablehlo.add %v9291, %v2249 : tensor<160x960x1x1xf32>
    %v9293 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9294 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9295 = stablehlo.multiply %v9293, %b15pWv : tensor<160x960x1x1xf32>
    %v9296 = stablehlo.multiply %v9292, %v9292 : tensor<160x960x1x1xf32>
    %v9297 = stablehlo.multiply %v9294, %v9296 : tensor<160x960x1x1xf32>
    %v9298 = stablehlo.add %v9295, %v9297 : tensor<160x960x1x1xf32>
    %v9299 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9300 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9301 = stablehlo.multiply %v9299, %b15pWv : tensor<160x960x1x1xf32>
    %v9302 = stablehlo.multiply %v9292, %v9292 : tensor<160x960x1x1xf32>
    %v9303 = stablehlo.multiply %v9300, %v9302 : tensor<160x960x1x1xf32>
    %v9304 = stablehlo.add %v9301, %v9303 : tensor<160x960x1x1xf32>
    %v9305 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9306 = stablehlo.add %v9304, %v9305 : tensor<160x960x1x1xf32>
    %v9307 = stablehlo.sqrt %v9306 : tensor<160x960x1x1xf32>
    %v9308 = stablehlo.divide %v9292, %v9307 : tensor<160x960x1x1xf32>
    %v9309 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9310 = stablehlo.multiply %v9309, %b15pWm : tensor<160x960x1x1xf32>
    %v9311 = stablehlo.add %v9310, %v9308 : tensor<160x960x1x1xf32>
    %v9312 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9313 = stablehlo.multiply %v9312, %v9311 : tensor<160x960x1x1xf32>
    %v9314 = stablehlo.subtract %b15pW, %v9313 : tensor<160x960x1x1xf32>
    %v9315 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9316 = stablehlo.multiply %v9315, %b15pg : tensor<160xf32>
    %v9317 = stablehlo.add %v9316, %v2267 : tensor<160xf32>
    %v9318 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9319 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9320 = stablehlo.multiply %v9318, %b15pgv : tensor<160xf32>
    %v9321 = stablehlo.multiply %v9317, %v9317 : tensor<160xf32>
    %v9322 = stablehlo.multiply %v9319, %v9321 : tensor<160xf32>
    %v9323 = stablehlo.add %v9320, %v9322 : tensor<160xf32>
    %v9324 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9325 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9326 = stablehlo.multiply %v9324, %b15pgv : tensor<160xf32>
    %v9327 = stablehlo.multiply %v9317, %v9317 : tensor<160xf32>
    %v9328 = stablehlo.multiply %v9325, %v9327 : tensor<160xf32>
    %v9329 = stablehlo.add %v9326, %v9328 : tensor<160xf32>
    %v9330 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9331 = stablehlo.add %v9329, %v9330 : tensor<160xf32>
    %v9332 = stablehlo.sqrt %v9331 : tensor<160xf32>
    %v9333 = stablehlo.divide %v9317, %v9332 : tensor<160xf32>
    %v9334 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9335 = stablehlo.multiply %v9334, %b15pgm : tensor<160xf32>
    %v9336 = stablehlo.add %v9335, %v9333 : tensor<160xf32>
    %v9337 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9338 = stablehlo.multiply %v9337, %v9336 : tensor<160xf32>
    %v9339 = stablehlo.subtract %b15pg, %v9338 : tensor<160xf32>
    %v9340 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9341 = stablehlo.multiply %v9340, %b15pbt : tensor<160xf32>
    %v9342 = stablehlo.add %v9341, %v2270 : tensor<160xf32>
    %v9343 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9344 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9345 = stablehlo.multiply %v9343, %b15pbtv : tensor<160xf32>
    %v9346 = stablehlo.multiply %v9342, %v9342 : tensor<160xf32>
    %v9347 = stablehlo.multiply %v9344, %v9346 : tensor<160xf32>
    %v9348 = stablehlo.add %v9345, %v9347 : tensor<160xf32>
    %v9349 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9350 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9351 = stablehlo.multiply %v9349, %b15pbtv : tensor<160xf32>
    %v9352 = stablehlo.multiply %v9342, %v9342 : tensor<160xf32>
    %v9353 = stablehlo.multiply %v9350, %v9352 : tensor<160xf32>
    %v9354 = stablehlo.add %v9351, %v9353 : tensor<160xf32>
    %v9355 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9356 = stablehlo.add %v9354, %v9355 : tensor<160xf32>
    %v9357 = stablehlo.sqrt %v9356 : tensor<160xf32>
    %v9358 = stablehlo.divide %v9342, %v9357 : tensor<160xf32>
    %v9359 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9360 = stablehlo.multiply %v9359, %b15pbtm : tensor<160xf32>
    %v9361 = stablehlo.add %v9360, %v9358 : tensor<160xf32>
    %v9362 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9363 = stablehlo.multiply %v9362, %v9361 : tensor<160xf32>
    %v9364 = stablehlo.subtract %b15pbt, %v9363 : tensor<160xf32>
    %v9365 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9366 = stablehlo.multiply %v9365, %b16eW : tensor<960x160x1x1xf32>
    %v9367 = stablehlo.add %v9366, %v1988 : tensor<960x160x1x1xf32>
    %v9368 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9369 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9370 = stablehlo.multiply %v9368, %b16eWv : tensor<960x160x1x1xf32>
    %v9371 = stablehlo.multiply %v9367, %v9367 : tensor<960x160x1x1xf32>
    %v9372 = stablehlo.multiply %v9369, %v9371 : tensor<960x160x1x1xf32>
    %v9373 = stablehlo.add %v9370, %v9372 : tensor<960x160x1x1xf32>
    %v9374 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9375 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9376 = stablehlo.multiply %v9374, %b16eWv : tensor<960x160x1x1xf32>
    %v9377 = stablehlo.multiply %v9367, %v9367 : tensor<960x160x1x1xf32>
    %v9378 = stablehlo.multiply %v9375, %v9377 : tensor<960x160x1x1xf32>
    %v9379 = stablehlo.add %v9376, %v9378 : tensor<960x160x1x1xf32>
    %v9380 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9381 = stablehlo.add %v9379, %v9380 : tensor<960x160x1x1xf32>
    %v9382 = stablehlo.sqrt %v9381 : tensor<960x160x1x1xf32>
    %v9383 = stablehlo.divide %v9367, %v9382 : tensor<960x160x1x1xf32>
    %v9384 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9385 = stablehlo.multiply %v9384, %b16eWm : tensor<960x160x1x1xf32>
    %v9386 = stablehlo.add %v9385, %v9383 : tensor<960x160x1x1xf32>
    %v9387 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9388 = stablehlo.multiply %v9387, %v9386 : tensor<960x160x1x1xf32>
    %v9389 = stablehlo.subtract %b16eW, %v9388 : tensor<960x160x1x1xf32>
    %v9390 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9391 = stablehlo.multiply %v9390, %b16eg : tensor<960xf32>
    %v9392 = stablehlo.add %v9391, %v2006 : tensor<960xf32>
    %v9393 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9394 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9395 = stablehlo.multiply %v9393, %b16egv : tensor<960xf32>
    %v9396 = stablehlo.multiply %v9392, %v9392 : tensor<960xf32>
    %v9397 = stablehlo.multiply %v9394, %v9396 : tensor<960xf32>
    %v9398 = stablehlo.add %v9395, %v9397 : tensor<960xf32>
    %v9399 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9400 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9401 = stablehlo.multiply %v9399, %b16egv : tensor<960xf32>
    %v9402 = stablehlo.multiply %v9392, %v9392 : tensor<960xf32>
    %v9403 = stablehlo.multiply %v9400, %v9402 : tensor<960xf32>
    %v9404 = stablehlo.add %v9401, %v9403 : tensor<960xf32>
    %v9405 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9406 = stablehlo.add %v9404, %v9405 : tensor<960xf32>
    %v9407 = stablehlo.sqrt %v9406 : tensor<960xf32>
    %v9408 = stablehlo.divide %v9392, %v9407 : tensor<960xf32>
    %v9409 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9410 = stablehlo.multiply %v9409, %b16egm : tensor<960xf32>
    %v9411 = stablehlo.add %v9410, %v9408 : tensor<960xf32>
    %v9412 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9413 = stablehlo.multiply %v9412, %v9411 : tensor<960xf32>
    %v9414 = stablehlo.subtract %b16eg, %v9413 : tensor<960xf32>
    %v9415 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9416 = stablehlo.multiply %v9415, %b16ebt : tensor<960xf32>
    %v9417 = stablehlo.add %v9416, %v2009 : tensor<960xf32>
    %v9418 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9419 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9420 = stablehlo.multiply %v9418, %b16ebtv : tensor<960xf32>
    %v9421 = stablehlo.multiply %v9417, %v9417 : tensor<960xf32>
    %v9422 = stablehlo.multiply %v9419, %v9421 : tensor<960xf32>
    %v9423 = stablehlo.add %v9420, %v9422 : tensor<960xf32>
    %v9424 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9425 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9426 = stablehlo.multiply %v9424, %b16ebtv : tensor<960xf32>
    %v9427 = stablehlo.multiply %v9417, %v9417 : tensor<960xf32>
    %v9428 = stablehlo.multiply %v9425, %v9427 : tensor<960xf32>
    %v9429 = stablehlo.add %v9426, %v9428 : tensor<960xf32>
    %v9430 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9431 = stablehlo.add %v9429, %v9430 : tensor<960xf32>
    %v9432 = stablehlo.sqrt %v9431 : tensor<960xf32>
    %v9433 = stablehlo.divide %v9417, %v9432 : tensor<960xf32>
    %v9434 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9435 = stablehlo.multiply %v9434, %b16ebtm : tensor<960xf32>
    %v9436 = stablehlo.add %v9435, %v9433 : tensor<960xf32>
    %v9437 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9438 = stablehlo.multiply %v9437, %v9436 : tensor<960xf32>
    %v9439 = stablehlo.subtract %b16ebt, %v9438 : tensor<960xf32>
    %v9440 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9441 = stablehlo.multiply %v9440, %b16dW : tensor<960x1x3x3xf32>
    %v9442 = stablehlo.add %v9441, %v2015 : tensor<960x1x3x3xf32>
    %v9443 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9444 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9445 = stablehlo.multiply %v9443, %b16dWv : tensor<960x1x3x3xf32>
    %v9446 = stablehlo.multiply %v9442, %v9442 : tensor<960x1x3x3xf32>
    %v9447 = stablehlo.multiply %v9444, %v9446 : tensor<960x1x3x3xf32>
    %v9448 = stablehlo.add %v9445, %v9447 : tensor<960x1x3x3xf32>
    %v9449 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9450 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9451 = stablehlo.multiply %v9449, %b16dWv : tensor<960x1x3x3xf32>
    %v9452 = stablehlo.multiply %v9442, %v9442 : tensor<960x1x3x3xf32>
    %v9453 = stablehlo.multiply %v9450, %v9452 : tensor<960x1x3x3xf32>
    %v9454 = stablehlo.add %v9451, %v9453 : tensor<960x1x3x3xf32>
    %v9455 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9456 = stablehlo.add %v9454, %v9455 : tensor<960x1x3x3xf32>
    %v9457 = stablehlo.sqrt %v9456 : tensor<960x1x3x3xf32>
    %v9458 = stablehlo.divide %v9442, %v9457 : tensor<960x1x3x3xf32>
    %v9459 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9460 = stablehlo.multiply %v9459, %b16dWm : tensor<960x1x3x3xf32>
    %v9461 = stablehlo.add %v9460, %v9458 : tensor<960x1x3x3xf32>
    %v9462 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9463 = stablehlo.multiply %v9462, %v9461 : tensor<960x1x3x3xf32>
    %v9464 = stablehlo.subtract %b16dW, %v9463 : tensor<960x1x3x3xf32>
    %v9465 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9466 = stablehlo.multiply %v9465, %b16dg : tensor<960xf32>
    %v9467 = stablehlo.add %v9466, %v2033 : tensor<960xf32>
    %v9468 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9469 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9470 = stablehlo.multiply %v9468, %b16dgv : tensor<960xf32>
    %v9471 = stablehlo.multiply %v9467, %v9467 : tensor<960xf32>
    %v9472 = stablehlo.multiply %v9469, %v9471 : tensor<960xf32>
    %v9473 = stablehlo.add %v9470, %v9472 : tensor<960xf32>
    %v9474 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9475 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9476 = stablehlo.multiply %v9474, %b16dgv : tensor<960xf32>
    %v9477 = stablehlo.multiply %v9467, %v9467 : tensor<960xf32>
    %v9478 = stablehlo.multiply %v9475, %v9477 : tensor<960xf32>
    %v9479 = stablehlo.add %v9476, %v9478 : tensor<960xf32>
    %v9480 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9481 = stablehlo.add %v9479, %v9480 : tensor<960xf32>
    %v9482 = stablehlo.sqrt %v9481 : tensor<960xf32>
    %v9483 = stablehlo.divide %v9467, %v9482 : tensor<960xf32>
    %v9484 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9485 = stablehlo.multiply %v9484, %b16dgm : tensor<960xf32>
    %v9486 = stablehlo.add %v9485, %v9483 : tensor<960xf32>
    %v9487 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9488 = stablehlo.multiply %v9487, %v9486 : tensor<960xf32>
    %v9489 = stablehlo.subtract %b16dg, %v9488 : tensor<960xf32>
    %v9490 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9491 = stablehlo.multiply %v9490, %b16dbt : tensor<960xf32>
    %v9492 = stablehlo.add %v9491, %v2036 : tensor<960xf32>
    %v9493 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9494 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9495 = stablehlo.multiply %v9493, %b16dbtv : tensor<960xf32>
    %v9496 = stablehlo.multiply %v9492, %v9492 : tensor<960xf32>
    %v9497 = stablehlo.multiply %v9494, %v9496 : tensor<960xf32>
    %v9498 = stablehlo.add %v9495, %v9497 : tensor<960xf32>
    %v9499 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9500 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9501 = stablehlo.multiply %v9499, %b16dbtv : tensor<960xf32>
    %v9502 = stablehlo.multiply %v9492, %v9492 : tensor<960xf32>
    %v9503 = stablehlo.multiply %v9500, %v9502 : tensor<960xf32>
    %v9504 = stablehlo.add %v9501, %v9503 : tensor<960xf32>
    %v9505 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9506 = stablehlo.add %v9504, %v9505 : tensor<960xf32>
    %v9507 = stablehlo.sqrt %v9506 : tensor<960xf32>
    %v9508 = stablehlo.divide %v9492, %v9507 : tensor<960xf32>
    %v9509 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9510 = stablehlo.multiply %v9509, %b16dbtm : tensor<960xf32>
    %v9511 = stablehlo.add %v9510, %v9508 : tensor<960xf32>
    %v9512 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9513 = stablehlo.multiply %v9512, %v9511 : tensor<960xf32>
    %v9514 = stablehlo.subtract %b16dbt, %v9513 : tensor<960xf32>
    %v9515 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9516 = stablehlo.multiply %v9515, %b16pW : tensor<160x960x1x1xf32>
    %v9517 = stablehlo.add %v9516, %v2042 : tensor<160x960x1x1xf32>
    %v9518 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9519 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9520 = stablehlo.multiply %v9518, %b16pWv : tensor<160x960x1x1xf32>
    %v9521 = stablehlo.multiply %v9517, %v9517 : tensor<160x960x1x1xf32>
    %v9522 = stablehlo.multiply %v9519, %v9521 : tensor<160x960x1x1xf32>
    %v9523 = stablehlo.add %v9520, %v9522 : tensor<160x960x1x1xf32>
    %v9524 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9525 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9526 = stablehlo.multiply %v9524, %b16pWv : tensor<160x960x1x1xf32>
    %v9527 = stablehlo.multiply %v9517, %v9517 : tensor<160x960x1x1xf32>
    %v9528 = stablehlo.multiply %v9525, %v9527 : tensor<160x960x1x1xf32>
    %v9529 = stablehlo.add %v9526, %v9528 : tensor<160x960x1x1xf32>
    %v9530 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9531 = stablehlo.add %v9529, %v9530 : tensor<160x960x1x1xf32>
    %v9532 = stablehlo.sqrt %v9531 : tensor<160x960x1x1xf32>
    %v9533 = stablehlo.divide %v9517, %v9532 : tensor<160x960x1x1xf32>
    %v9534 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9535 = stablehlo.multiply %v9534, %b16pWm : tensor<160x960x1x1xf32>
    %v9536 = stablehlo.add %v9535, %v9533 : tensor<160x960x1x1xf32>
    %v9537 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160x960x1x1xf32>
    %v9538 = stablehlo.multiply %v9537, %v9536 : tensor<160x960x1x1xf32>
    %v9539 = stablehlo.subtract %b16pW, %v9538 : tensor<160x960x1x1xf32>
    %v9540 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9541 = stablehlo.multiply %v9540, %b16pg : tensor<160xf32>
    %v9542 = stablehlo.add %v9541, %v2060 : tensor<160xf32>
    %v9543 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9544 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9545 = stablehlo.multiply %v9543, %b16pgv : tensor<160xf32>
    %v9546 = stablehlo.multiply %v9542, %v9542 : tensor<160xf32>
    %v9547 = stablehlo.multiply %v9544, %v9546 : tensor<160xf32>
    %v9548 = stablehlo.add %v9545, %v9547 : tensor<160xf32>
    %v9549 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9550 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9551 = stablehlo.multiply %v9549, %b16pgv : tensor<160xf32>
    %v9552 = stablehlo.multiply %v9542, %v9542 : tensor<160xf32>
    %v9553 = stablehlo.multiply %v9550, %v9552 : tensor<160xf32>
    %v9554 = stablehlo.add %v9551, %v9553 : tensor<160xf32>
    %v9555 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9556 = stablehlo.add %v9554, %v9555 : tensor<160xf32>
    %v9557 = stablehlo.sqrt %v9556 : tensor<160xf32>
    %v9558 = stablehlo.divide %v9542, %v9557 : tensor<160xf32>
    %v9559 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9560 = stablehlo.multiply %v9559, %b16pgm : tensor<160xf32>
    %v9561 = stablehlo.add %v9560, %v9558 : tensor<160xf32>
    %v9562 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9563 = stablehlo.multiply %v9562, %v9561 : tensor<160xf32>
    %v9564 = stablehlo.subtract %b16pg, %v9563 : tensor<160xf32>
    %v9565 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9566 = stablehlo.multiply %v9565, %b16pbt : tensor<160xf32>
    %v9567 = stablehlo.add %v9566, %v2063 : tensor<160xf32>
    %v9568 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9569 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9570 = stablehlo.multiply %v9568, %b16pbtv : tensor<160xf32>
    %v9571 = stablehlo.multiply %v9567, %v9567 : tensor<160xf32>
    %v9572 = stablehlo.multiply %v9569, %v9571 : tensor<160xf32>
    %v9573 = stablehlo.add %v9570, %v9572 : tensor<160xf32>
    %v9574 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9575 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9576 = stablehlo.multiply %v9574, %b16pbtv : tensor<160xf32>
    %v9577 = stablehlo.multiply %v9567, %v9567 : tensor<160xf32>
    %v9578 = stablehlo.multiply %v9575, %v9577 : tensor<160xf32>
    %v9579 = stablehlo.add %v9576, %v9578 : tensor<160xf32>
    %v9580 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9581 = stablehlo.add %v9579, %v9580 : tensor<160xf32>
    %v9582 = stablehlo.sqrt %v9581 : tensor<160xf32>
    %v9583 = stablehlo.divide %v9567, %v9582 : tensor<160xf32>
    %v9584 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9585 = stablehlo.multiply %v9584, %b16pbtm : tensor<160xf32>
    %v9586 = stablehlo.add %v9585, %v9583 : tensor<160xf32>
    %v9587 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<160xf32>
    %v9588 = stablehlo.multiply %v9587, %v9586 : tensor<160xf32>
    %v9589 = stablehlo.subtract %b16pbt, %v9588 : tensor<160xf32>
    %v9590 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9591 = stablehlo.multiply %v9590, %b17eW : tensor<960x160x1x1xf32>
    %v9592 = stablehlo.add %v9591, %v1781 : tensor<960x160x1x1xf32>
    %v9593 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9594 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9595 = stablehlo.multiply %v9593, %b17eWv : tensor<960x160x1x1xf32>
    %v9596 = stablehlo.multiply %v9592, %v9592 : tensor<960x160x1x1xf32>
    %v9597 = stablehlo.multiply %v9594, %v9596 : tensor<960x160x1x1xf32>
    %v9598 = stablehlo.add %v9595, %v9597 : tensor<960x160x1x1xf32>
    %v9599 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9600 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9601 = stablehlo.multiply %v9599, %b17eWv : tensor<960x160x1x1xf32>
    %v9602 = stablehlo.multiply %v9592, %v9592 : tensor<960x160x1x1xf32>
    %v9603 = stablehlo.multiply %v9600, %v9602 : tensor<960x160x1x1xf32>
    %v9604 = stablehlo.add %v9601, %v9603 : tensor<960x160x1x1xf32>
    %v9605 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9606 = stablehlo.add %v9604, %v9605 : tensor<960x160x1x1xf32>
    %v9607 = stablehlo.sqrt %v9606 : tensor<960x160x1x1xf32>
    %v9608 = stablehlo.divide %v9592, %v9607 : tensor<960x160x1x1xf32>
    %v9609 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9610 = stablehlo.multiply %v9609, %b17eWm : tensor<960x160x1x1xf32>
    %v9611 = stablehlo.add %v9610, %v9608 : tensor<960x160x1x1xf32>
    %v9612 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x160x1x1xf32>
    %v9613 = stablehlo.multiply %v9612, %v9611 : tensor<960x160x1x1xf32>
    %v9614 = stablehlo.subtract %b17eW, %v9613 : tensor<960x160x1x1xf32>
    %v9615 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9616 = stablehlo.multiply %v9615, %b17eg : tensor<960xf32>
    %v9617 = stablehlo.add %v9616, %v1799 : tensor<960xf32>
    %v9618 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9619 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9620 = stablehlo.multiply %v9618, %b17egv : tensor<960xf32>
    %v9621 = stablehlo.multiply %v9617, %v9617 : tensor<960xf32>
    %v9622 = stablehlo.multiply %v9619, %v9621 : tensor<960xf32>
    %v9623 = stablehlo.add %v9620, %v9622 : tensor<960xf32>
    %v9624 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9625 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9626 = stablehlo.multiply %v9624, %b17egv : tensor<960xf32>
    %v9627 = stablehlo.multiply %v9617, %v9617 : tensor<960xf32>
    %v9628 = stablehlo.multiply %v9625, %v9627 : tensor<960xf32>
    %v9629 = stablehlo.add %v9626, %v9628 : tensor<960xf32>
    %v9630 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9631 = stablehlo.add %v9629, %v9630 : tensor<960xf32>
    %v9632 = stablehlo.sqrt %v9631 : tensor<960xf32>
    %v9633 = stablehlo.divide %v9617, %v9632 : tensor<960xf32>
    %v9634 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9635 = stablehlo.multiply %v9634, %b17egm : tensor<960xf32>
    %v9636 = stablehlo.add %v9635, %v9633 : tensor<960xf32>
    %v9637 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9638 = stablehlo.multiply %v9637, %v9636 : tensor<960xf32>
    %v9639 = stablehlo.subtract %b17eg, %v9638 : tensor<960xf32>
    %v9640 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9641 = stablehlo.multiply %v9640, %b17ebt : tensor<960xf32>
    %v9642 = stablehlo.add %v9641, %v1802 : tensor<960xf32>
    %v9643 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9644 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9645 = stablehlo.multiply %v9643, %b17ebtv : tensor<960xf32>
    %v9646 = stablehlo.multiply %v9642, %v9642 : tensor<960xf32>
    %v9647 = stablehlo.multiply %v9644, %v9646 : tensor<960xf32>
    %v9648 = stablehlo.add %v9645, %v9647 : tensor<960xf32>
    %v9649 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9650 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9651 = stablehlo.multiply %v9649, %b17ebtv : tensor<960xf32>
    %v9652 = stablehlo.multiply %v9642, %v9642 : tensor<960xf32>
    %v9653 = stablehlo.multiply %v9650, %v9652 : tensor<960xf32>
    %v9654 = stablehlo.add %v9651, %v9653 : tensor<960xf32>
    %v9655 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9656 = stablehlo.add %v9654, %v9655 : tensor<960xf32>
    %v9657 = stablehlo.sqrt %v9656 : tensor<960xf32>
    %v9658 = stablehlo.divide %v9642, %v9657 : tensor<960xf32>
    %v9659 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9660 = stablehlo.multiply %v9659, %b17ebtm : tensor<960xf32>
    %v9661 = stablehlo.add %v9660, %v9658 : tensor<960xf32>
    %v9662 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9663 = stablehlo.multiply %v9662, %v9661 : tensor<960xf32>
    %v9664 = stablehlo.subtract %b17ebt, %v9663 : tensor<960xf32>
    %v9665 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9666 = stablehlo.multiply %v9665, %b17dW : tensor<960x1x3x3xf32>
    %v9667 = stablehlo.add %v9666, %v1808 : tensor<960x1x3x3xf32>
    %v9668 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9669 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9670 = stablehlo.multiply %v9668, %b17dWv : tensor<960x1x3x3xf32>
    %v9671 = stablehlo.multiply %v9667, %v9667 : tensor<960x1x3x3xf32>
    %v9672 = stablehlo.multiply %v9669, %v9671 : tensor<960x1x3x3xf32>
    %v9673 = stablehlo.add %v9670, %v9672 : tensor<960x1x3x3xf32>
    %v9674 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9675 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9676 = stablehlo.multiply %v9674, %b17dWv : tensor<960x1x3x3xf32>
    %v9677 = stablehlo.multiply %v9667, %v9667 : tensor<960x1x3x3xf32>
    %v9678 = stablehlo.multiply %v9675, %v9677 : tensor<960x1x3x3xf32>
    %v9679 = stablehlo.add %v9676, %v9678 : tensor<960x1x3x3xf32>
    %v9680 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9681 = stablehlo.add %v9679, %v9680 : tensor<960x1x3x3xf32>
    %v9682 = stablehlo.sqrt %v9681 : tensor<960x1x3x3xf32>
    %v9683 = stablehlo.divide %v9667, %v9682 : tensor<960x1x3x3xf32>
    %v9684 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9685 = stablehlo.multiply %v9684, %b17dWm : tensor<960x1x3x3xf32>
    %v9686 = stablehlo.add %v9685, %v9683 : tensor<960x1x3x3xf32>
    %v9687 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960x1x3x3xf32>
    %v9688 = stablehlo.multiply %v9687, %v9686 : tensor<960x1x3x3xf32>
    %v9689 = stablehlo.subtract %b17dW, %v9688 : tensor<960x1x3x3xf32>
    %v9690 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9691 = stablehlo.multiply %v9690, %b17dg : tensor<960xf32>
    %v9692 = stablehlo.add %v9691, %v1826 : tensor<960xf32>
    %v9693 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9694 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9695 = stablehlo.multiply %v9693, %b17dgv : tensor<960xf32>
    %v9696 = stablehlo.multiply %v9692, %v9692 : tensor<960xf32>
    %v9697 = stablehlo.multiply %v9694, %v9696 : tensor<960xf32>
    %v9698 = stablehlo.add %v9695, %v9697 : tensor<960xf32>
    %v9699 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9700 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9701 = stablehlo.multiply %v9699, %b17dgv : tensor<960xf32>
    %v9702 = stablehlo.multiply %v9692, %v9692 : tensor<960xf32>
    %v9703 = stablehlo.multiply %v9700, %v9702 : tensor<960xf32>
    %v9704 = stablehlo.add %v9701, %v9703 : tensor<960xf32>
    %v9705 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9706 = stablehlo.add %v9704, %v9705 : tensor<960xf32>
    %v9707 = stablehlo.sqrt %v9706 : tensor<960xf32>
    %v9708 = stablehlo.divide %v9692, %v9707 : tensor<960xf32>
    %v9709 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9710 = stablehlo.multiply %v9709, %b17dgm : tensor<960xf32>
    %v9711 = stablehlo.add %v9710, %v9708 : tensor<960xf32>
    %v9712 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9713 = stablehlo.multiply %v9712, %v9711 : tensor<960xf32>
    %v9714 = stablehlo.subtract %b17dg, %v9713 : tensor<960xf32>
    %v9715 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9716 = stablehlo.multiply %v9715, %b17dbt : tensor<960xf32>
    %v9717 = stablehlo.add %v9716, %v1829 : tensor<960xf32>
    %v9718 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9719 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9720 = stablehlo.multiply %v9718, %b17dbtv : tensor<960xf32>
    %v9721 = stablehlo.multiply %v9717, %v9717 : tensor<960xf32>
    %v9722 = stablehlo.multiply %v9719, %v9721 : tensor<960xf32>
    %v9723 = stablehlo.add %v9720, %v9722 : tensor<960xf32>
    %v9724 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9725 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9726 = stablehlo.multiply %v9724, %b17dbtv : tensor<960xf32>
    %v9727 = stablehlo.multiply %v9717, %v9717 : tensor<960xf32>
    %v9728 = stablehlo.multiply %v9725, %v9727 : tensor<960xf32>
    %v9729 = stablehlo.add %v9726, %v9728 : tensor<960xf32>
    %v9730 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9731 = stablehlo.add %v9729, %v9730 : tensor<960xf32>
    %v9732 = stablehlo.sqrt %v9731 : tensor<960xf32>
    %v9733 = stablehlo.divide %v9717, %v9732 : tensor<960xf32>
    %v9734 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9735 = stablehlo.multiply %v9734, %b17dbtm : tensor<960xf32>
    %v9736 = stablehlo.add %v9735, %v9733 : tensor<960xf32>
    %v9737 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<960xf32>
    %v9738 = stablehlo.multiply %v9737, %v9736 : tensor<960xf32>
    %v9739 = stablehlo.subtract %b17dbt, %v9738 : tensor<960xf32>
    %v9740 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9741 = stablehlo.multiply %v9740, %b17pW : tensor<320x960x1x1xf32>
    %v9742 = stablehlo.add %v9741, %v1835 : tensor<320x960x1x1xf32>
    %v9743 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9744 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9745 = stablehlo.multiply %v9743, %b17pWv : tensor<320x960x1x1xf32>
    %v9746 = stablehlo.multiply %v9742, %v9742 : tensor<320x960x1x1xf32>
    %v9747 = stablehlo.multiply %v9744, %v9746 : tensor<320x960x1x1xf32>
    %v9748 = stablehlo.add %v9745, %v9747 : tensor<320x960x1x1xf32>
    %v9749 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9750 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9751 = stablehlo.multiply %v9749, %b17pWv : tensor<320x960x1x1xf32>
    %v9752 = stablehlo.multiply %v9742, %v9742 : tensor<320x960x1x1xf32>
    %v9753 = stablehlo.multiply %v9750, %v9752 : tensor<320x960x1x1xf32>
    %v9754 = stablehlo.add %v9751, %v9753 : tensor<320x960x1x1xf32>
    %v9755 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9756 = stablehlo.add %v9754, %v9755 : tensor<320x960x1x1xf32>
    %v9757 = stablehlo.sqrt %v9756 : tensor<320x960x1x1xf32>
    %v9758 = stablehlo.divide %v9742, %v9757 : tensor<320x960x1x1xf32>
    %v9759 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9760 = stablehlo.multiply %v9759, %b17pWm : tensor<320x960x1x1xf32>
    %v9761 = stablehlo.add %v9760, %v9758 : tensor<320x960x1x1xf32>
    %v9762 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<320x960x1x1xf32>
    %v9763 = stablehlo.multiply %v9762, %v9761 : tensor<320x960x1x1xf32>
    %v9764 = stablehlo.subtract %b17pW, %v9763 : tensor<320x960x1x1xf32>
    %v9765 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9766 = stablehlo.multiply %v9765, %b17pg : tensor<320xf32>
    %v9767 = stablehlo.add %v9766, %v1853 : tensor<320xf32>
    %v9768 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9769 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9770 = stablehlo.multiply %v9768, %b17pgv : tensor<320xf32>
    %v9771 = stablehlo.multiply %v9767, %v9767 : tensor<320xf32>
    %v9772 = stablehlo.multiply %v9769, %v9771 : tensor<320xf32>
    %v9773 = stablehlo.add %v9770, %v9772 : tensor<320xf32>
    %v9774 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9775 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9776 = stablehlo.multiply %v9774, %b17pgv : tensor<320xf32>
    %v9777 = stablehlo.multiply %v9767, %v9767 : tensor<320xf32>
    %v9778 = stablehlo.multiply %v9775, %v9777 : tensor<320xf32>
    %v9779 = stablehlo.add %v9776, %v9778 : tensor<320xf32>
    %v9780 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9781 = stablehlo.add %v9779, %v9780 : tensor<320xf32>
    %v9782 = stablehlo.sqrt %v9781 : tensor<320xf32>
    %v9783 = stablehlo.divide %v9767, %v9782 : tensor<320xf32>
    %v9784 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9785 = stablehlo.multiply %v9784, %b17pgm : tensor<320xf32>
    %v9786 = stablehlo.add %v9785, %v9783 : tensor<320xf32>
    %v9787 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9788 = stablehlo.multiply %v9787, %v9786 : tensor<320xf32>
    %v9789 = stablehlo.subtract %b17pg, %v9788 : tensor<320xf32>
    %v9790 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9791 = stablehlo.multiply %v9790, %b17pbt : tensor<320xf32>
    %v9792 = stablehlo.add %v9791, %v1856 : tensor<320xf32>
    %v9793 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9794 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9795 = stablehlo.multiply %v9793, %b17pbtv : tensor<320xf32>
    %v9796 = stablehlo.multiply %v9792, %v9792 : tensor<320xf32>
    %v9797 = stablehlo.multiply %v9794, %v9796 : tensor<320xf32>
    %v9798 = stablehlo.add %v9795, %v9797 : tensor<320xf32>
    %v9799 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9800 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9801 = stablehlo.multiply %v9799, %b17pbtv : tensor<320xf32>
    %v9802 = stablehlo.multiply %v9792, %v9792 : tensor<320xf32>
    %v9803 = stablehlo.multiply %v9800, %v9802 : tensor<320xf32>
    %v9804 = stablehlo.add %v9801, %v9803 : tensor<320xf32>
    %v9805 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9806 = stablehlo.add %v9804, %v9805 : tensor<320xf32>
    %v9807 = stablehlo.sqrt %v9806 : tensor<320xf32>
    %v9808 = stablehlo.divide %v9792, %v9807 : tensor<320xf32>
    %v9809 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9810 = stablehlo.multiply %v9809, %b17pbtm : tensor<320xf32>
    %v9811 = stablehlo.add %v9810, %v9808 : tensor<320xf32>
    %v9812 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<320xf32>
    %v9813 = stablehlo.multiply %v9812, %v9811 : tensor<320xf32>
    %v9814 = stablehlo.subtract %b17pbt, %v9813 : tensor<320xf32>
    %v9815 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9816 = stablehlo.multiply %v9815, %hW : tensor<1280x320x1x1xf32>
    %v9817 = stablehlo.add %v9816, %v1632 : tensor<1280x320x1x1xf32>
    %v9818 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9819 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9820 = stablehlo.multiply %v9818, %hWv : tensor<1280x320x1x1xf32>
    %v9821 = stablehlo.multiply %v9817, %v9817 : tensor<1280x320x1x1xf32>
    %v9822 = stablehlo.multiply %v9819, %v9821 : tensor<1280x320x1x1xf32>
    %v9823 = stablehlo.add %v9820, %v9822 : tensor<1280x320x1x1xf32>
    %v9824 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9825 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9826 = stablehlo.multiply %v9824, %hWv : tensor<1280x320x1x1xf32>
    %v9827 = stablehlo.multiply %v9817, %v9817 : tensor<1280x320x1x1xf32>
    %v9828 = stablehlo.multiply %v9825, %v9827 : tensor<1280x320x1x1xf32>
    %v9829 = stablehlo.add %v9826, %v9828 : tensor<1280x320x1x1xf32>
    %v9830 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9831 = stablehlo.add %v9829, %v9830 : tensor<1280x320x1x1xf32>
    %v9832 = stablehlo.sqrt %v9831 : tensor<1280x320x1x1xf32>
    %v9833 = stablehlo.divide %v9817, %v9832 : tensor<1280x320x1x1xf32>
    %v9834 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9835 = stablehlo.multiply %v9834, %hWm : tensor<1280x320x1x1xf32>
    %v9836 = stablehlo.add %v9835, %v9833 : tensor<1280x320x1x1xf32>
    %v9837 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280x320x1x1xf32>
    %v9838 = stablehlo.multiply %v9837, %v9836 : tensor<1280x320x1x1xf32>
    %v9839 = stablehlo.subtract %hW, %v9838 : tensor<1280x320x1x1xf32>
    %v9840 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9841 = stablehlo.multiply %v9840, %hg : tensor<1280xf32>
    %v9842 = stablehlo.add %v9841, %v1650 : tensor<1280xf32>
    %v9843 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9844 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9845 = stablehlo.multiply %v9843, %hgv : tensor<1280xf32>
    %v9846 = stablehlo.multiply %v9842, %v9842 : tensor<1280xf32>
    %v9847 = stablehlo.multiply %v9844, %v9846 : tensor<1280xf32>
    %v9848 = stablehlo.add %v9845, %v9847 : tensor<1280xf32>
    %v9849 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9850 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9851 = stablehlo.multiply %v9849, %hgv : tensor<1280xf32>
    %v9852 = stablehlo.multiply %v9842, %v9842 : tensor<1280xf32>
    %v9853 = stablehlo.multiply %v9850, %v9852 : tensor<1280xf32>
    %v9854 = stablehlo.add %v9851, %v9853 : tensor<1280xf32>
    %v9855 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9856 = stablehlo.add %v9854, %v9855 : tensor<1280xf32>
    %v9857 = stablehlo.sqrt %v9856 : tensor<1280xf32>
    %v9858 = stablehlo.divide %v9842, %v9857 : tensor<1280xf32>
    %v9859 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9860 = stablehlo.multiply %v9859, %hgm : tensor<1280xf32>
    %v9861 = stablehlo.add %v9860, %v9858 : tensor<1280xf32>
    %v9862 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9863 = stablehlo.multiply %v9862, %v9861 : tensor<1280xf32>
    %v9864 = stablehlo.subtract %hg, %v9863 : tensor<1280xf32>
    %v9865 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9866 = stablehlo.multiply %v9865, %hbt : tensor<1280xf32>
    %v9867 = stablehlo.add %v9866, %v1653 : tensor<1280xf32>
    %v9868 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9869 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9870 = stablehlo.multiply %v9868, %hbtv : tensor<1280xf32>
    %v9871 = stablehlo.multiply %v9867, %v9867 : tensor<1280xf32>
    %v9872 = stablehlo.multiply %v9869, %v9871 : tensor<1280xf32>
    %v9873 = stablehlo.add %v9870, %v9872 : tensor<1280xf32>
    %v9874 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9875 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9876 = stablehlo.multiply %v9874, %hbtv : tensor<1280xf32>
    %v9877 = stablehlo.multiply %v9867, %v9867 : tensor<1280xf32>
    %v9878 = stablehlo.multiply %v9875, %v9877 : tensor<1280xf32>
    %v9879 = stablehlo.add %v9876, %v9878 : tensor<1280xf32>
    %v9880 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9881 = stablehlo.add %v9879, %v9880 : tensor<1280xf32>
    %v9882 = stablehlo.sqrt %v9881 : tensor<1280xf32>
    %v9883 = stablehlo.divide %v9867, %v9882 : tensor<1280xf32>
    %v9884 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9885 = stablehlo.multiply %v9884, %hbtm : tensor<1280xf32>
    %v9886 = stablehlo.add %v9885, %v9883 : tensor<1280xf32>
    %v9887 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %v9888 = stablehlo.multiply %v9887, %v9886 : tensor<1280xf32>
    %v9889 = stablehlo.subtract %hbt, %v9888 : tensor<1280xf32>
    %v9890 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9891 = stablehlo.multiply %v9890, %Wd : tensor<1280x10xf32>
    %v9892 = stablehlo.add %v9891, %v1576 : tensor<1280x10xf32>
    %v9893 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9894 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9895 = stablehlo.multiply %v9893, %Wdv : tensor<1280x10xf32>
    %v9896 = stablehlo.multiply %v9892, %v9892 : tensor<1280x10xf32>
    %v9897 = stablehlo.multiply %v9894, %v9896 : tensor<1280x10xf32>
    %v9898 = stablehlo.add %v9895, %v9897 : tensor<1280x10xf32>
    %v9899 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9900 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9901 = stablehlo.multiply %v9899, %Wdv : tensor<1280x10xf32>
    %v9902 = stablehlo.multiply %v9892, %v9892 : tensor<1280x10xf32>
    %v9903 = stablehlo.multiply %v9900, %v9902 : tensor<1280x10xf32>
    %v9904 = stablehlo.add %v9901, %v9903 : tensor<1280x10xf32>
    %v9905 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9906 = stablehlo.add %v9904, %v9905 : tensor<1280x10xf32>
    %v9907 = stablehlo.sqrt %v9906 : tensor<1280x10xf32>
    %v9908 = stablehlo.divide %v9892, %v9907 : tensor<1280x10xf32>
    %v9909 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9910 = stablehlo.multiply %v9909, %Wdm : tensor<1280x10xf32>
    %v9911 = stablehlo.add %v9910, %v9908 : tensor<1280x10xf32>
    %v9912 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1280x10xf32>
    %v9913 = stablehlo.multiply %v9912, %v9911 : tensor<1280x10xf32>
    %v9914 = stablehlo.subtract %Wd, %v9913 : tensor<1280x10xf32>
    %v9915 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9916 = stablehlo.multiply %v9915, %bd : tensor<10xf32>
    %v9917 = stablehlo.add %v9916, %v1578 : tensor<10xf32>
    %v9918 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9919 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9920 = stablehlo.multiply %v9918, %bdv : tensor<10xf32>
    %v9921 = stablehlo.multiply %v9917, %v9917 : tensor<10xf32>
    %v9922 = stablehlo.multiply %v9919, %v9921 : tensor<10xf32>
    %v9923 = stablehlo.add %v9920, %v9922 : tensor<10xf32>
    %v9924 = stablehlo.broadcast_in_dim %rho, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9925 = stablehlo.broadcast_in_dim %orho, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9926 = stablehlo.multiply %v9924, %bdv : tensor<10xf32>
    %v9927 = stablehlo.multiply %v9917, %v9917 : tensor<10xf32>
    %v9928 = stablehlo.multiply %v9925, %v9927 : tensor<10xf32>
    %v9929 = stablehlo.add %v9926, %v9928 : tensor<10xf32>
    %v9930 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9931 = stablehlo.add %v9929, %v9930 : tensor<10xf32>
    %v9932 = stablehlo.sqrt %v9931 : tensor<10xf32>
    %v9933 = stablehlo.divide %v9917, %v9932 : tensor<10xf32>
    %v9934 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9935 = stablehlo.multiply %v9934, %bdm : tensor<10xf32>
    %v9936 = stablehlo.add %v9935, %v9933 : tensor<10xf32>
    %v9937 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v9938 = stablehlo.multiply %v9937, %v9936 : tensor<10xf32>
    %v9939 = stablehlo.subtract %bd, %v9938 : tensor<10xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1564 : tensor<32x10xf32>
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
    return %v6014, %v6039, %v6064, %v6089, %v6114, %v6139, %v6164, %v6189, %v6214, %v6239, %v6264, %v6289, %v6314, %v6339, %v6364, %v6389, %v6414, %v6439, %v6464, %v6489, %v6514, %v6539, %v6564, %v6589, %v6614, %v6639, %v6664, %v6689, %v6714, %v6739, %v6764, %v6789, %v6814, %v6839, %v6864, %v6889, %v6914, %v6939, %v6964, %v6989, %v7014, %v7039, %v7064, %v7089, %v7114, %v7139, %v7164, %v7189, %v7214, %v7239, %v7264, %v7289, %v7314, %v7339, %v7364, %v7389, %v7414, %v7439, %v7464, %v7489, %v7514, %v7539, %v7564, %v7589, %v7614, %v7639, %v7664, %v7689, %v7714, %v7739, %v7764, %v7789, %v7814, %v7839, %v7864, %v7889, %v7914, %v7939, %v7964, %v7989, %v8014, %v8039, %v8064, %v8089, %v8114, %v8139, %v8164, %v8189, %v8214, %v8239, %v8264, %v8289, %v8314, %v8339, %v8364, %v8389, %v8414, %v8439, %v8464, %v8489, %v8514, %v8539, %v8564, %v8589, %v8614, %v8639, %v8664, %v8689, %v8714, %v8739, %v8764, %v8789, %v8814, %v8839, %v8864, %v8889, %v8914, %v8939, %v8964, %v8989, %v9014, %v9039, %v9064, %v9089, %v9114, %v9139, %v9164, %v9189, %v9214, %v9239, %v9264, %v9289, %v9314, %v9339, %v9364, %v9389, %v9414, %v9439, %v9464, %v9489, %v9514, %v9539, %v9564, %v9589, %v9614, %v9639, %v9664, %v9689, %v9714, %v9739, %v9764, %v9789, %v9814, %v9839, %v9864, %v9889, %v9914, %v9939, %v6011, %v6036, %v6061, %v6086, %v6111, %v6136, %v6161, %v6186, %v6211, %v6236, %v6261, %v6286, %v6311, %v6336, %v6361, %v6386, %v6411, %v6436, %v6461, %v6486, %v6511, %v6536, %v6561, %v6586, %v6611, %v6636, %v6661, %v6686, %v6711, %v6736, %v6761, %v6786, %v6811, %v6836, %v6861, %v6886, %v6911, %v6936, %v6961, %v6986, %v7011, %v7036, %v7061, %v7086, %v7111, %v7136, %v7161, %v7186, %v7211, %v7236, %v7261, %v7286, %v7311, %v7336, %v7361, %v7386, %v7411, %v7436, %v7461, %v7486, %v7511, %v7536, %v7561, %v7586, %v7611, %v7636, %v7661, %v7686, %v7711, %v7736, %v7761, %v7786, %v7811, %v7836, %v7861, %v7886, %v7911, %v7936, %v7961, %v7986, %v8011, %v8036, %v8061, %v8086, %v8111, %v8136, %v8161, %v8186, %v8211, %v8236, %v8261, %v8286, %v8311, %v8336, %v8361, %v8386, %v8411, %v8436, %v8461, %v8486, %v8511, %v8536, %v8561, %v8586, %v8611, %v8636, %v8661, %v8686, %v8711, %v8736, %v8761, %v8786, %v8811, %v8836, %v8861, %v8886, %v8911, %v8936, %v8961, %v8986, %v9011, %v9036, %v9061, %v9086, %v9111, %v9136, %v9161, %v9186, %v9211, %v9236, %v9261, %v9286, %v9311, %v9336, %v9361, %v9386, %v9411, %v9436, %v9461, %v9486, %v9511, %v9536, %v9561, %v9586, %v9611, %v9636, %v9661, %v9686, %v9711, %v9736, %v9761, %v9786, %v9811, %v9836, %v9861, %v9886, %v9911, %v9936, %v5998, %v6023, %v6048, %v6073, %v6098, %v6123, %v6148, %v6173, %v6198, %v6223, %v6248, %v6273, %v6298, %v6323, %v6348, %v6373, %v6398, %v6423, %v6448, %v6473, %v6498, %v6523, %v6548, %v6573, %v6598, %v6623, %v6648, %v6673, %v6698, %v6723, %v6748, %v6773, %v6798, %v6823, %v6848, %v6873, %v6898, %v6923, %v6948, %v6973, %v6998, %v7023, %v7048, %v7073, %v7098, %v7123, %v7148, %v7173, %v7198, %v7223, %v7248, %v7273, %v7298, %v7323, %v7348, %v7373, %v7398, %v7423, %v7448, %v7473, %v7498, %v7523, %v7548, %v7573, %v7598, %v7623, %v7648, %v7673, %v7698, %v7723, %v7748, %v7773, %v7798, %v7823, %v7848, %v7873, %v7898, %v7923, %v7948, %v7973, %v7998, %v8023, %v8048, %v8073, %v8098, %v8123, %v8148, %v8173, %v8198, %v8223, %v8248, %v8273, %v8298, %v8323, %v8348, %v8373, %v8398, %v8423, %v8448, %v8473, %v8498, %v8523, %v8548, %v8573, %v8598, %v8623, %v8648, %v8673, %v8698, %v8723, %v8748, %v8773, %v8798, %v8823, %v8848, %v8873, %v8898, %v8923, %v8948, %v8973, %v8998, %v9023, %v9048, %v9073, %v9098, %v9123, %v9148, %v9173, %v9198, %v9223, %v9248, %v9273, %v9298, %v9323, %v9348, %v9373, %v9398, %v9423, %v9448, %v9473, %v9498, %v9523, %v9548, %v9573, %v9598, %v9623, %v9648, %v9673, %v9698, %v9723, %v9748, %v9773, %v9798, %v9823, %v9848, %v9873, %v9898, %v9923, %loss, %bc1, %bc2, %v5162, %v5173, %v5178, %v5189, %v5194, %v5205, %v5210, %v5221, %v5226, %v5237, %v5242, %v5253, %v5258, %v5269, %v5274, %v5285, %v5290, %v5301, %v5306, %v5317, %v5322, %v5333, %v5338, %v5349, %v5354, %v5365, %v5370, %v5381, %v5386, %v5397, %v5402, %v5413, %v5418, %v5429, %v5434, %v5445, %v5450, %v5461, %v5466, %v5477, %v5482, %v5493, %v5498, %v5509, %v5514, %v5525, %v5530, %v5541, %v5546, %v5557, %v5562, %v5573, %v5578, %v5589, %v5594, %v5605, %v5610, %v5621, %v5626, %v5637, %v5642, %v5653, %v5658, %v5669, %v5674, %v5685, %v5690, %v5701, %v5706, %v5717, %v5722, %v5733, %v5738, %v5749, %v5754, %v5765, %v5770, %v5781, %v5786, %v5797, %v5802, %v5813, %v5818, %v5829, %v5834, %v5845, %v5850, %v5861, %v5866, %v5877, %v5882, %v5893, %v5898, %v5909, %v5914, %v5925, %v5930, %v5941, %v5946, %v5957, %v5962, %v5973, %v5978, %v5989 : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280xf32>, tensor<1280xf32>
  }
}
